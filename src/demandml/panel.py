"""Build the dense (sku x date) modelling panel.

Lag features are only meaningful on a *regular* time grid. The raw report is close to
dense but not guaranteed to be, so we reindex every SKU onto a complete daily (or
weekly) calendar between its first and last real sale, then filter out SKUs with too
little history to learn from.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from . import config as C
from .clean import (
    add_censoring_flags,
    add_imputed_target,
    add_winsorized_target,
    attach_product,
    clean_daily,
    clean_weekly,
    drop_leaky,
)
from .logging_utils import get_logger

log = get_logger(__name__)


def _life_bounds(lf: pl.LazyFrame, spec: C.PanelSpec) -> pl.LazyFrame:
    """First/last date worth keeping per SKU."""
    sold = pl.when(pl.col(C.TARGET) > 0).then(pl.col(C.DATE))
    agg = lf.group_by(C.SKU).agg(
        pl.col(C.DATE).min().alias("first_row"),
        pl.col(C.DATE).max().alias("last_row"),
        sold.min().alias("first_sale"),
        sold.max().alias("last_sale"),
        pl.col(C.TARGET).sum().alias("total_units"),
        (pl.col(C.TARGET) > 0).sum().alias("selling_days"),
    )
    start = pl.col("first_sale") if spec.trim_leading_zeros else pl.col("first_row")
    end = pl.col("last_sale") if spec.trim_trailing_zeros else pl.col("last_row")
    return agg.with_columns(
        pl.coalesce(start, pl.col("first_row")).alias("start"),
        pl.coalesce(end, pl.col("last_row")).alias("end"),
    ).with_columns((pl.col("end") - pl.col("start")).dt.total_days().alias("history_days"))



def find_complete_tail(lf: pl.LazyFrame, spec: C.PanelSpec) -> "dt.date | None":
    """Last date whose total volume looks like a complete period.

    A dump taken mid-day leaves a final partial period in the file: the last day can carry
    a tiny fraction of the units of the day before, and a model trained or scored on it
    sees a near-total demand collapse that never happened. We walk backwards from
    the end and drop any period whose total is under ``threshold`` x the median of the 28
    periods before it.
    """
    if not spec.incomplete_tail_threshold:
        return None
    daily = (
        lf.group_by(C.DATE)
        .agg(pl.col(C.TARGET).sum().alias("units"))
        .sort(C.DATE)
        .collect()
    )
    units = daily.get_column("units").to_list()
    dates = daily.get_column(C.DATE).to_list()
    cut = len(dates)
    for i in range(len(dates) - 1, max(len(dates) - 30, 0) - 1, -1):
        window = units[max(0, i - 28) : i]
        if not window:
            break
        ref = sorted(window)[len(window) // 2]
        if ref > 0 and units[i] < spec.incomplete_tail_threshold * ref:
            cut = i
        else:
            break
    if cut == len(dates):
        return None
    log.warning(
        "dropping %d incomplete trailing period(s) after %s (last kept volume %s)",
        len(dates) - cut, dates[cut - 1], f"{units[cut - 1]:,.0f}",
    )
    return dates[cut - 1]


def build_panel(
    spec: C.PanelSpec | None = None,
    freq: str = "daily",
    with_product: bool = True,
    write: bool = True,
) -> pl.DataFrame:
    """Clean -> trim -> densify -> enrich. Returns the panel and optionally writes it.

    ``freq`` is ``"daily"`` or ``"weekly"``.
    """
    spec = spec or C.DEFAULT_PANEL
    every = "1d" if freq == "daily" else "1w"

    lf = clean_daily() if freq == "daily" else clean_weekly()
    lf = drop_leaky(lf)

    last_complete = find_complete_tail(lf, spec)
    if last_complete is not None:
        lf = lf.filter(pl.col(C.DATE) <= last_complete)

    bounds = _life_bounds(lf, spec)
    keep = bounds.filter(
        (pl.col("history_days") >= spec.min_history_days)
        & (pl.col("total_units") >= spec.min_total_units)
    )

    n_all = bounds.select(pl.len()).collect().item()
    n_keep = keep.select(pl.len()).collect().item()
    log.info("%s panel: keeping %d/%d SKUs after history filters", freq, n_keep, n_all)

    lf = lf.join(keep.select(C.SKU, "start", "end", "total_units", "selling_days"), on=C.SKU)
    lf = lf.filter((pl.col(C.DATE) >= pl.col("start")) & (pl.col(C.DATE) <= pl.col("end")))

    df = lf.collect(engine="streaming")
    log.info("%s observed rows after trim: %s", freq, f"{len(df):,}")

    # --- densify: complete calendar per SKU -------------------------------------------
    df = (
        df.sort([C.SKU, C.DATE])
        .upsample(time_column=C.DATE, every=every, group_by=C.SKU, maintain_order=True)
        .with_columns(
            pl.col(C.SKU).forward_fill(),
            pl.col("start").forward_fill(),
            pl.col("end").forward_fill(),
            pl.col("total_units").forward_fill(),
            pl.col("selling_days").forward_fill(),
        )
    )

    # Rows created by upsampling are genuinely-missing report rows, not zero demand.
    # Flag them, then fill the volume columns with 0 (no sale was recorded) and leave the
    # rate/price columns to be forward-filled.
    zero_fill = [
        c
        for c in (C.TARGET, C.TARGET_FILTERED, "revenue", "order_count", "sessions",
                  "ad_cost", "ad_units", "impressions", "clicks", "discount_amount")
        if c in df.columns
    ]
    ffill = [
        c
        for c in ("unit_price", "minor_rank_ff", "master_rank_ff", "oos_threshold",
                  "discount_rate", "return_rate")
        if c in df.columns
    ]
    df = df.with_columns(
        pl.col(C.TARGET).is_null().alias("is_imputed_row"),
    ).with_columns(
        [pl.col(c).fill_null(0.0) for c in zero_fill]
        + [pl.col(c).forward_fill().over(C.SKU) for c in ffill]
        # The weekly export has no is_deal column, so fill only the flags present.
        + [pl.col(c).fill_null(False) for c in ("is_oos", "is_deal") if c in df.columns]
    )

    log.info("%s dense rows: %s", freq, f"{len(df):,}")

    # --- target variants --------------------------------------------------------------
    lf = df.lazy()
    lf = add_censoring_flags(lf)
    lf = add_imputed_target(lf)
    lf = add_winsorized_target(lf)
    if with_product:
        lf = attach_product(lf)
    df = lf.collect()

    if write:
        out = C.PANEL_DAILY if freq == "daily" else C.PANEL_WEEKLY
        df.write_parquet(out, compression="zstd")
        log.info("wrote %s (%.0f MB)", out.name, out.stat().st_size / 1e6)
    return df


def load_panel(freq: str = "daily") -> pl.LazyFrame:
    path = C.PANEL_DAILY if freq == "daily" else C.PANEL_WEEKLY
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `demandml panel` first")
    return pl.scan_parquet(path)
