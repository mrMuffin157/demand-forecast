"""The demand history can come from a database extract or a CSV, and the two must agree.

The daily sales table can arrive as a hand-exported CSV in `data/raw/` or as a Parquet
extract in `data/db/` (produced by `scripts/make_sample_data.py` for the sample, or by your
own extract). A hand export is the ONE input a scheduled pipeline cannot refresh, so the
extract is preferred whenever it exists.

The risk that introduces is not that the extract fails -- a failed pull is loud. It is that
the two sources produce *slightly different panels*, because a CSV arrives as text and a
Parquet from a database arrives typed. A panel that differs by a type coercion produces a
plausible forecast, which is the failure mode nothing downstream can see. So the load-
bearing test here is `test_the_two_sources_produce_an_identical_frame`; the rest guard the
plumbing around it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from demandml import config as C          # noqa: E402
from demandml import ingest as I          # noqa: E402


# ======================================================================================
# The two sources must not drift
# ======================================================================================

# One row per awkward case rather than a wide sample: a date-only timestamp, a
# microsecond one, both truth values, and a null.
_ROWS = [
    ("ZD-103-1", "2015-08-01 00:00:00", "Y", "N", 3),
    ("ZD-103-1", "2026-08-10 00:25:05.345000", "N", "Y", 0),
    ("ZZ-010-2", "2026-08-11 12:00:00", "N", "N", 17),
]


def _as_csv(path: Path) -> Path:
    """The hand export: everything is text."""
    pl.DataFrame({
        "sku": [r[0] for r in _ROWS],
        "product_id": ["p1", "p1", "p2"],
        "id": ["1", "2", "3"],
        "date_purchased": ["2015-08-01", "2026-08-10", "2026-08-11"],
        "date_created": [r[1] for r in _ROWS],
        "last_updated": [r[1] for r in _ROWS],
        "is_oos": [r[2] for r in _ROWS],
        "is_deal": [r[3] for r in _ROWS],
        "salescount": [r[4] for r in _ROWS],
    }).write_csv(path)
    return path


def _as_pulled_parquet(path: Path) -> Path:
    """The database extract: the driver hands back real Date, Datetime and Int columns."""
    pl.DataFrame({
        "sku": [r[0] for r in _ROWS],
        "product_id": ["p1", "p1", "p2"],
        "id": [1, 2, 3],
        "date_purchased": [dt.date(2015, 8, 1), dt.date(2026, 8, 10), dt.date(2026, 8, 11)],
        "date_created": [dt.datetime.fromisoformat(r[1]) for r in _ROWS],
        "last_updated": [dt.datetime.fromisoformat(r[1]) for r in _ROWS],
        "is_oos": [r[2] for r in _ROWS],
        "is_deal": [r[3] for r in _ROWS],
        "salescount": [r[4] for r in _ROWS],
    }).write_parquet(path)
    return path


def _transform(src: Path) -> pl.DataFrame:
    """The daily ingest's own expressions, applied to whichever source is handed in.

    Kept in step with `ingest_daily` by using the same helpers rather than a copy of the
    literal, so a change there that breaks the equivalence shows up here.
    """
    lf = I._scan_source(src, I._DAILY_OVERRIDES)
    exprs = [
        I._parse_ts("date_purchased").dt.date().alias(C.DATE),
        I._parse_ts("date_created").alias("row_created"),
        I._parse_ts("last_updated").alias("row_updated"),
    ]
    exprs += [I._yn_to_bool(c) for c in C.DAILY_YN_COLS]
    drop = {"date_purchased", "date_created", "last_updated", *C.DAILY_YN_COLS}
    passthrough = [c for c in I.csv_header(src) if c not in drop]
    return lf.select([pl.col(c) for c in passthrough] + exprs).collect().sort("sku", C.DATE)


def test_the_two_sources_produce_an_identical_frame(tmp_path: Path):
    """A CSV export and a database pull of the same table must ingest to the same thing.

    This is the whole reason `_scan_source` casts the overridden columns back to String on
    the Parquet path instead of teaching every downstream expression to accept two types.
    If this ever fails, the panel depends on which source the volume happened to hold.
    """
    from_csv = _transform(_as_csv(tmp_path / "daily_sales_1.csv"))
    from_db = _transform(_as_pulled_parquet(tmp_path / "sales_daily.parquet"))

    assert from_csv.columns == from_db.columns
    assert from_csv.schema == from_db.schema, "the two sources disagree about types"
    assert from_csv.equals(from_db), "the two sources disagree about values"


def test_the_pulled_types_survive_the_round_trip(tmp_path: Path):
    """Spelled out, because the equality above would also pass if both were wrong.

    A date-only value, a microsecond timestamp and `Y`/`N` are the three shapes this
    table actually carries; all three have to land as Date, Datetime and Boolean.
    """
    got = _transform(_as_pulled_parquet(tmp_path / "sales_daily.parquet"))
    assert got.schema[C.DATE] == pl.Date
    assert got.schema["row_created"] == pl.Datetime
    assert got.schema["is_oos"] == pl.Boolean
    assert got.filter(pl.col("sku") == "ZZ-010-2")["is_deal"].to_list() == [False]
    assert got[C.DATE].to_list() == [
        dt.date(2015, 8, 1), dt.date(2026, 8, 10), dt.date(2026, 8, 11)]


# ======================================================================================
# Choosing between them
# ======================================================================================

def test_the_pull_wins_over_a_csv_and_the_csv_still_works(tmp_path: Path, monkeypatch):
    """Deterministic, not clever: the pull wins whenever it exists.

    Choosing by mtime would make the pipeline's input depend on metadata that a copy or an
    sync rewrites. `datacheck.py` is what catches a stale input, whichever source it
    came from -- so this function only has to be predictable.
    """
    raw, db = tmp_path / "raw", tmp_path / "db"
    raw.mkdir(); db.mkdir()
    monkeypatch.setattr(C, "RAW", raw)
    monkeypatch.setattr(C, "DB_EXTRACT", db)

    (raw / "daily_sales_202609111215.csv").write_text("sku\n")
    assert C._find_raw("daily_sales", pulled="sales_daily").suffix == ".csv"

    (db / "sales_daily.parquet").write_bytes(b"")
    chosen = C._find_raw("daily_sales", pulled="sales_daily")
    assert chosen.suffix == ".parquet" and chosen.parent == db

    # And with neither, the caller gets None rather than a path that does not exist --
    # `ingest_daily` turns that into a message naming both places it looked.
    (db / "sales_daily.parquet").unlink()
    (raw / "daily_sales_202609111215.csv").unlink()
    assert C._find_raw("daily_sales", pulled="sales_daily") is None


def test_the_daily_source_is_wired_to_the_pull_and_the_others_are_not():
    """Daily sales and the product master are wired to the pull (`tests/test_product_source.py`
    is the product master's equivalence guard).

    The weekly table is NOT wired, and that is deliberate: a weekly table can hold a
    different row population from the daily export, so the two need not be the same
    population. The daily pipeline does not read it, so nothing
    is blocked; swapping it blind would change the weekly panel for unexplained reasons.
    """
    src = __import__("inspect").getsource(C)
    assert 'RAW_DAILY = _find_raw("daily_sales", pulled="sales_daily")' in src
    assert 'RAW_WEEKLY = _find_raw("weekly_sales")' in src
    assert 'RAW_PRODUCT = _find_raw("product_", pulled="product")' in src


# ======================================================================================
# The skip that would have frozen the panel
# ======================================================================================

def test_a_fresh_pull_is_re_ingested_rather_than_skipped(tmp_path: Path):
    """`overwrite=False` must not mean "never rebuild".

    This is the trap that would make the whole change cosmetic: a scheduled job runs
    `demandml ingest` with the default `overwrite=False`, so a plain `dst.exists()` check
    logs "already exists -- skipping" on every run and the panel stays frozen on the day
    it was first built. No error, no symptom, a forecast off stale demand.
    """
    src, dst = tmp_path / "sales_daily.parquet", tmp_path / "daily_sales.parquet"
    src.write_bytes(b"x")
    dst.write_bytes(b"y")

    import os
    os.utime(dst, (1_000_000, 1_000_000))
    os.utime(src, (2_000_000, 2_000_000))          # a pull that landed after the build
    assert I._is_stale(dst, src), "a newer source must force a rebuild"

    os.utime(src, (1_000_000, 1_000_000))
    os.utime(dst, (2_000_000, 2_000_000))          # nothing new since the last build
    assert not I._is_stale(dst, src), "an up-to-date artefact must still be skipped"

    dst.unlink()
    assert I._is_stale(dst, src), "a missing artefact is always stale"
