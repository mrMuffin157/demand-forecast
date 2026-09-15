"""Raw sources -> typed Parquet.

The daily sales and the product master come from the database extract in `data/db/`
(produced by `scripts/make_sample_data.py` for the sample data, or by your own extract)
when there is one, and from hand-exported CSVs in `data/raw/` otherwise --
`config._find_raw` picks. A full daily CSV runs to gigabytes, which is painful to touch
repeatedly. This module streams whichever source it is given once through Polars and
writes a typed, compressed Parquet mirror that can be lazily scanned column-by-column
afterwards. Everything downstream reads that mirror, never the source.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from . import config as C
from .logging_utils import get_logger

log = get_logger(__name__)

# Polars sniffs dtypes from the first N rows. These columns are the ones where sniffing
# reliably gets it wrong (all-empty prefixes, "Y"/"N" strings, ids that look numeric).
_DAILY_OVERRIDES: dict[str, pl.DataType] = {
    "sku": pl.String,
    "product_id": pl.String,
    "id": pl.String,
    "is_oos": pl.String,
    "is_deal": pl.String,
    "date_purchased": pl.String,
    "date_created": pl.String,
    "last_updated": pl.String,
}

_WEEKLY_OVERRIDES: dict[str, pl.DataType] = {
    "id": pl.String,
    "product_id": pl.String,
    "sku": pl.String,
    "date": pl.String,
    "date_created": pl.String,
    "last_updated": pl.String,
    "date_type": pl.String,
    "event": pl.String,
    "master_ranks": pl.String,
    "minor_ranks": pl.String,
    "year_week": pl.Int32,
}

_PRODUCT_OVERRIDES: dict[str, pl.DataType] = {
    "id": pl.String,
    "sku": pl.String,
    "brand_id": pl.String,
    "main_image_id": pl.String,
    "revision_id": pl.String,
    "from_container_line_id": pl.String,
    "upc": pl.String,
    "mpn": pl.String,
    "master_carton_gtin": pl.String,
    "date_created": pl.String,
    "date_on_market": pl.String,
    "last_updated": pl.String,
    "tier": pl.String,
    "listing_status": pl.String,
    "material": pl.String,
    "channel_category": pl.String,
    "type": pl.String,
    "uom": pl.String,
    "carton_size": pl.String,
}


def _is_stale(dst: Path, src: Path) -> bool:
    """Does `dst` need rebuilding from `src`?

    `overwrite=False` exists so that iterating locally does not re-ingest gigabytes of
    CSV every time. In a scheduled pipeline that same default is a correctness bug: the data tier
    pulls a fresh `sales_daily.parquet` on every run, and a plain existence check would log
    "already exists -- skipping" and forecast off a panel frozen on the day it was first
    built. Nothing would raise, and the numbers would stay plausible -- which is the exact
    failure this whole path was changed to remove.

    So the skip is conditional on the artefact being NEWER than the source it derives
    from. That is mtime used the way a build system uses it -- is my output stale with
    respect to its input -- and not the way `config._find_raw` deliberately refuses to use
    it, which was to pick BETWEEN two candidate sources. The first question mtime answers
    well; the second it answers unpredictably, because a copy or an S3 sync rewrites it.
    """
    if not dst.exists():
        return True
    return dst.stat().st_mtime < src.stat().st_mtime


def csv_header(path: Path) -> list[str]:
    """Read just the header row of a (possibly enormous) CSV or Parquet source."""
    if path.suffix == ".parquet":
        return pl.scan_parquet(path).collect_schema().names()
    return pl.read_csv(path, n_rows=0).columns


def _scan_source(path: Path, overrides: dict[str, pl.DataType]) -> pl.LazyFrame:
    """Scan a raw source, CSV or Parquet, so both arrive in the same shape.

    The two inputs are the same table -- one exported to CSV by hand, one pulled from the
    database -- but they do not arrive with the same types. A CSV is all text
    until something says otherwise, which is what `overrides` is for; a Parquet from
    MySQL arrives already typed, with real Date, Datetime and Boolean columns.

    Rather than teach every downstream expression to accept both, this casts the
    overridden columns back to String on the Parquet path. `_parse_ts` and `_yn_to_bool`
    then see exactly what they have always seen -- a Date renders as ``2015-08-01``, a
    Datetime as ``2015-08-01 00:00:00.000000`` and a Boolean as ``true``/``false``, all
    three of which those two helpers already parse. One code path after this line, so the
    CSV and the pull cannot drift into producing different panels.
    """
    if path.suffix != ".parquet":
        return pl.scan_csv(
            path,
            schema_overrides=overrides,
            infer_schema_length=100_000,
            null_values=["", "NULL", "null", "\\N", "None"],
            ignore_errors=False,
            low_memory=True,
        )
    lf = pl.scan_parquet(path)
    present = set(lf.collect_schema().names())
    return lf.with_columns(
        [pl.col(c).cast(pl.String) for c in overrides if c in present]
    )


def _parse_ts(col: str) -> pl.Expr:
    """Parse the mixed timestamp formats in these dumps into a Datetime.

    Seen in the wild: ``2022-01-02 00:00:00``, ``2025-10-19 21:11:39.345000``,
    ``2017-01-01`` (date only), and empty strings.
    """
    s = pl.col(col).str.strip_chars().replace("", None)
    return (
        s.str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False)
        .fill_null(s.str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False))
        .fill_null(s.str.to_date("%Y-%m-%d", strict=False).cast(pl.Datetime))
        .alias(col)
    )


def _yn_to_bool(col: str) -> pl.Expr:
    """``"Y"``/``"N"`` (and a few ``1``/``0``/blank variants) -> Boolean."""
    return (
        pl.col(col)
        .str.strip_chars()
        .str.to_uppercase()
        .replace_strict(
            {"Y": True, "YES": True, "TRUE": True, "1": True,
             "N": False, "NO": False, "FALSE": False, "0": False, "": None},
            default=None,
            return_dtype=pl.Boolean,
        )
        .alias(col)
    )


def ingest_daily(src: Path | None = None, dst: Path | None = None, overwrite: bool = False) -> Path:
    """Stream the daily sales source (DB pull, else CSV) into Parquet."""
    src = src or C.RAW_DAILY
    dst = dst or C.PQ_DAILY
    if src is None:
        raise FileNotFoundError(
            f"No daily sales source found: neither {C.DB_EXTRACT / 'sales_daily.parquet'} "
            f"nor a daily_sales*.csv in {C.RAW}")
    if not overwrite and not _is_stale(dst, src):
        log.info("%s is newer than %s -- skipping (use overwrite=True to force)",
                 dst.name, src.name)
        return dst

    t0 = time.time()
    # Which of the two sources was taken is worth a line in the log, not an inference from
    # the filename: it is the difference between a pipeline that refreshes itself and one
    # forecasting off whatever CSV was last dropped on the volume.
    origin = "database pull" if src.suffix == ".parquet" else "manual CSV export"
    log.info("ingesting %s (%.2f GB, %s) -> %s",
             src.name, src.stat().st_size / 1e9, origin, dst.name)

    lf = _scan_source(src, _DAILY_OVERRIDES)

    exprs: list[pl.Expr] = [
        _parse_ts("date_purchased").dt.date().alias(C.DATE),
        _parse_ts("date_created").alias("row_created"),
        _parse_ts("last_updated").alias("row_updated"),
    ]
    exprs += [_yn_to_bool(c) for c in C.DAILY_YN_COLS]

    drop = {"date_purchased", "date_created", "last_updated", *C.DAILY_YN_COLS}
    passthrough = [c for c in csv_header(src) if c not in drop]

    lf = lf.select([pl.col(c) for c in passthrough] + exprs).with_columns(
        pl.col("sku").str.strip_chars(),
        pl.col("product_id").str.strip_chars(),
    )

    lf.sink_parquet(dst, compression="zstd", compression_level=3, row_group_size=250_000)
    _report(dst, t0)
    return dst


def ingest_weekly(src: Path | None = None, dst: Path | None = None, overwrite: bool = False) -> Path:
    """Stream the weekly sales CSV into Parquet, unpacking the comma-packed rank strings."""
    src = src or C.RAW_WEEKLY
    dst = dst or C.PQ_WEEKLY
    if src is None:
        raise FileNotFoundError(f"No weekly sales CSV found in {C.RAW}")
    if not overwrite and not _is_stale(dst, src):
        log.info("%s is newer than %s -- skipping", dst.name, src.name)
        return dst

    t0 = time.time()
    log.info("ingesting %s (%.2f GB) -> %s", src.name, src.stat().st_size / 1e9, dst.name)

    lf = pl.scan_csv(
        src,
        schema_overrides=_WEEKLY_OVERRIDES,
        infer_schema_length=100_000,
        null_values=["", "NULL", "null", "\\N", "None"],
        low_memory=True,
    )

    # "0,0,0,0,0,0,0" -> mean / min of the non-zero daily ranks for that week.
    rank_exprs: list[pl.Expr] = []
    for col in C.WEEKLY_PACKED_COLS:
        parts = (
            pl.col(col)
            .str.strip_chars()
            .str.split(",")
            .list.eval(pl.element().cast(pl.Float64, strict=False))
            .list.eval(pl.element().filter(pl.element() > 0))
        )
        base = col.rstrip("s")  # master_ranks -> master_rank
        rank_exprs += [
            parts.list.mean().alias(f"{base}_mean"),
            parts.list.min().alias(f"{base}_best"),
            parts.list.len().alias(f"{base}_days_ranked"),
        ]

    drop = {"date", "date_created", "last_updated", *C.WEEKLY_PACKED_COLS}
    passthrough = [c for c in csv_header(src) if c not in drop]

    lf = lf.select(
        [pl.col(c) for c in passthrough]
        + [
            _parse_ts("date").dt.date().alias(C.DATE),
            _parse_ts("date_created").alias("row_created"),
            _parse_ts("last_updated").alias("row_updated"),
        ]
        + rank_exprs
    ).with_columns(pl.col("sku").str.strip_chars(), pl.col("product_id").str.strip_chars())

    lf.sink_parquet(dst, compression="zstd", compression_level=3)
    _report(dst, t0)
    return dst


def ingest_product(
    src: Path | None = None, dst: Path | None = None, overwrite: bool = False
) -> Path:
    """Load the product master into Parquet, keeping only the columns worth modelling.

    Two sources, one output -- the same arrangement as `ingest_daily`. The database pull
    (`data/db/product.parquet`, `SELECT * FROM product`) is preferred, because a fresh data
    volume starts empty and a hand-exported CSV is the one input it cannot refresh.
    `tests/test_product_source.py` asserts both ingest to the same frame.
    """
    src = src or C.RAW_PRODUCT
    dst = dst or C.PQ_PRODUCT
    if src is None:
        raise FileNotFoundError(
            f"No product source found: neither {C.DB_EXTRACT / 'product.parquet'} "
            f"nor a product_*.csv in {C.RAW}")
    if not overwrite and not _is_stale(dst, src):
        log.info("%s is newer than %s -- skipping", dst.name, src.name)
        return dst

    t0 = time.time()
    pulled = src.suffix == ".parquet"
    log.info("ingesting %s (%s) -> %s", src.name,
             "database pull" if pulled else "manual CSV export", dst.name)

    if pulled:
        lf = _scan_source(src, _PRODUCT_OVERRIDES)
        # MySQL's flag columns arrive as Boolean; the CSV export writes them as 0/1, and
        # 0/1 Int64 is what the panel and every model have always seen.
        flags = [c for c, t in lf.collect_schema().items() if t == pl.Boolean]
        df = lf.with_columns([pl.col(c).cast(pl.Int64) for c in flags]).collect()
    else:
        df = pl.read_csv(
            src,
            schema_overrides=_PRODUCT_OVERRIDES,
            infer_schema_length=None,
            null_values=["", "NULL", "null", "\\N", "None"],
            truncate_ragged_lines=True,
        )
    have = [c for c in C.PRODUCT_KEEP_COLS if c in df.columns]
    missing = sorted(set(C.PRODUCT_KEEP_COLS) - set(have))
    if missing and pulled:
        # A narrow pull -- a query naming only a handful of columns -- would otherwise
        # ingest "fine" and silently remove most of the product features from every model.
        raise ValueError(f"{src} is missing product columns {missing}; re-pull the "
                         "product table with every column (SELECT *)")
    if missing:
        log.warning("product columns absent from export: %s", missing)

    df = df.select(have).with_columns(
        _parse_ts("date_created").alias("product_created"),
        _parse_ts("date_on_market").alias("date_on_market"),
        pl.col("sku").str.strip_chars(),
        pl.col("id").str.strip_chars().alias("product_id"),
    )
    df = df.drop("date_created", "id")
    df.write_parquet(dst, compression="zstd")
    _report(dst, t0)
    return dst


def _report(dst: Path, t0: float) -> None:
    n = pl.scan_parquet(dst).select(pl.len()).collect().item()
    log.info(
        "wrote %s: %s rows, %.0f MB, %.1fs",
        dst.name, f"{n:,}", dst.stat().st_size / 1e6, time.time() - t0,
    )


def ingest_all(overwrite: bool = False) -> dict[str, Path]:
    """Run the ingests. Product and daily are required; weekly is not.

    The weekly export exists only as a hand-exported CSV (the database table is a different
    population -- see config.py), and nothing the daily pipeline runs reads it. Requiring it
    would make a fresh data volume, which holds no CSVs at all, fail its first run here.
    `panel --freq weekly` still needs it and says so.
    """
    out = {"product": ingest_product(overwrite=overwrite)}
    if C.RAW_WEEKLY is not None:
        out["weekly"] = ingest_weekly(overwrite=overwrite)
    else:
        log.info("no weekly sales CSV in %s -- skipping (the daily pipeline does not use it)",
                 C.RAW)
    out["daily"] = ingest_daily(overwrite=overwrite)
    return out
