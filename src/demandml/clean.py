"""Cleaning and canonicalisation of the daily / weekly sales rows.

What this module is actually defending against, based on inspecting the export:

* **Duplicate (sku, date) rows.** The dump is a report table that has been rebuilt over
  the years; the same SKU-day can appear more than once. We keep the most recently
  updated row.
* **Stock-out censoring.** ``is_oos = Y`` days have suppressed sales: the observed 0 is
  *not* zero demand, it is missing demand. Treating them as real zeros teaches the model
  to forecast the stock-out, not the demand. We keep them but mark them, expose a
  sample weight that can zero them out, and offer an imputed target.
* **Bulk orders.** A single wholesale order can put a spike of thousands of units on a
  SKU that normally sells a couple of dozen a day. The export already ships a bulk-filtered count; we also provide
  a winsorised variant.
* **Leaky pre-computed averages.** ``avg_daily_weekly``, ``no_oos_demand`` and friends are
  rolling stats computed by the reporting system at row-write time and almost certainly
  include the row's own day. They are dropped; features.py recomputes shifted versions.
* **Price only observable on days with sales.** ``revenue / salescount`` is undefined on
  zero-sale days, so the implied unit price is forward-filled within each SKU.
* **Pre-launch and post-delist zeros.** A SKU that launched years after the report begins has
  years of zero rows in it. Those are not demand observations. Trimmed in panel.py.
"""

from __future__ import annotations

import polars as pl

from . import config as C
from .logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Daily
# --------------------------------------------------------------------------------------


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num / den).otherwise(None)


def clean_daily(lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """Canonicalise the daily sales rows. Returns a LazyFrame, one row per (sku, date)."""
    lf = lf if lf is not None else pl.scan_parquet(C.PQ_DAILY)

    lf = lf.filter(pl.col(C.DATE).is_not_null() & pl.col(C.SKU).is_not_null())

    # --- de-duplicate -----------------------------------------------------------------
    # Prefer the most recently updated row, then the one with the most sales recorded
    # (a rebuild that produced 0 should not beat a row that has real numbers).
    lf = (
        lf.sort(["row_updated", C.TARGET_RAW], descending=[False, False], nulls_last=False)
        .unique(subset=[C.SKU, C.DATE], keep="last")
        .sort([C.SKU, C.DATE])
    )

    # --- derived economics ------------------------------------------------------------
    units = pl.col(C.TARGET_RAW).cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0)
    revenue = pl.col("revenue").cast(pl.Float64).fill_null(0.0)
    discount = (
        pl.col("deal_discount").cast(pl.Float64).fill_null(0.0)
        + pl.col("coupon_discount").cast(pl.Float64).fill_null(0.0)
        + pl.col("business_discount").cast(pl.Float64).fill_null(0.0)
    ).abs()

    lf = lf.with_columns(
        units.alias(C.TARGET),
        pl.col("big_order_filter_sales_count")
        .cast(pl.Float64)
        .fill_null(units)
        .clip(lower_bound=0.0)
        .alias(C.TARGET_FILTERED),
        revenue.alias("revenue"),
        discount.alias("discount_amount"),
        pl.col("is_oos").fill_null(False),
        pl.col("is_deal").fill_null(False),
    ).with_columns(
        # Realised average selling price; undefined when nothing sold.
        _safe_div(revenue, pl.col(C.TARGET)).alias("unit_price_raw"),
        _safe_div(discount, revenue + discount).alias("discount_rate"),
        _safe_div(pl.col("return_amount").cast(pl.Float64), revenue).alias("return_rate"),
        _safe_div(pl.col("clicks").cast(pl.Float64), pl.col("impressions").cast(pl.Float64))
        .alias("ctr"),
        _safe_div(pl.col(C.TARGET), pl.col("sessions").cast(pl.Float64))
        .alias("conversion_rate"),
        _safe_div(pl.col("ad_cost").cast(pl.Float64), pl.col("ad_units").cast(pl.Float64))
        .alias("cost_per_ad_unit"),
        _safe_div(pl.col("ad_units").cast(pl.Float64), pl.col(C.TARGET))
        .alias("ad_unit_share"),
        _safe_div(pl.col("final_profit").cast(pl.Float64), revenue).alias("margin"),
        _safe_div(revenue, pl.col("order_count").cast(pl.Float64)).alias("basket_value"),
        _safe_div(pl.col(C.TARGET), pl.col("order_count").cast(pl.Float64))
        .alias("units_per_order"),
        # Ranks use 0 as "unranked", which would otherwise read as "best possible rank".
        pl.col("minor_rank").cast(pl.Float64).replace(0.0, None).alias("minor_rank"),
        pl.col("master_rank").cast(pl.Float64).replace(0.0, None).alias("master_rank"),
        pl.col("sessions").cast(pl.Float64).alias("sessions"),
    )

    # Forward/backward-fill the price within a SKU so zero-sale days still carry a price.
    lf = lf.with_columns(
        pl.col("unit_price_raw")
        .forward_fill()
        .backward_fill()
        .over(C.SKU)
        .alias("unit_price"),
        pl.col("minor_rank").forward_fill().over(C.SKU).alias("minor_rank_ff"),
        pl.col("master_rank").forward_fill().over(C.SKU).alias("master_rank_ff"),
    )

    return lf


def add_censoring_flags(lf: pl.LazyFrame, target: str = C.TARGET) -> pl.LazyFrame:
    """Mark stock-out-censored observations and attach a default sample weight.

    ``is_censored`` is True where the day was flagged out of stock. ``sample_weight`` is
    0 for censored zero-sale days (the observation carries no information about demand)
    and 1 otherwise. Partial stock-outs -- flagged OOS but with sales -- keep weight 1
    but are still marked, since they understate demand rather than erase it.
    """
    return lf.with_columns(
        pl.col("is_oos").alias("is_censored"),
        pl.when(pl.col("is_oos") & (pl.col(target) <= 0))
        .then(0.0)
        .otherwise(1.0)
        .alias("sample_weight"),
    )


def add_imputed_target(
    lf: pl.LazyFrame,
    target: str = C.TARGET,
    out: str = "units_uncensored",
    window: int = 56,
) -> pl.LazyFrame:
    """Fill censored zero-sale days with an estimate of what demand would have been.

    Uses the SKU's trailing same-weekday median over ``window`` days, computed from
    *uncensored* days only, so a long stock-out cannot bootstrap itself. Where no
    estimate is available the original value is kept.

    This is a pragmatic point estimate, not a proper Tobit/censored likelihood -- it
    exists so you can compare "drop censored days" against "impute them" and see which
    backtests better.
    """
    uncensored = pl.when(pl.col("is_oos")).then(None).otherwise(pl.col(target))
    weeks = max(window // 7, 2)
    return lf.with_columns(
        uncensored.alias("_unc"),
        pl.col(C.DATE).dt.weekday().alias("_dow"),
    ).with_columns(
        pl.col("_unc")
        .shift(1)
        .rolling_median(window_size=weeks, min_samples=2)
        .over([C.SKU, "_dow"])
        .alias("_dow_median"),
        pl.col("_unc")
        .shift(1)
        .rolling_median(window_size=window, min_samples=7)
        .over(C.SKU)
        .alias("_roll_median"),
    ).with_columns(
        pl.when(pl.col("is_oos") & (pl.col(target) <= 0))
        .then(
            pl.coalesce(pl.col("_dow_median"), pl.col("_roll_median"), pl.lit(0.0))
        )
        .otherwise(pl.col(target))
        .alias(out)
    ).drop("_unc", "_dow", "_dow_median", "_roll_median")


def add_winsorized_target(
    lf: pl.LazyFrame,
    target: str = C.TARGET,
    out: str = "units_winsorized",
    window: int = 91,
    k: float = 6.0,
) -> pl.LazyFrame:
    """Cap bulk-order spikes at ``median + k * MAD`` of the trailing window.

    Uses a *trailing* (shifted) window so the cap itself never sees the future. MAD is
    scaled by 1.4826 to be comparable to a standard deviation.
    """
    med = pl.col(target).shift(1).rolling_median(window, min_samples=14).over(C.SKU)
    return lf.with_columns(med.alias("_med")).with_columns(
        ((pl.col(target).shift(1) - pl.col("_med")).abs())
        .rolling_median(window, min_samples=14)
        .over(C.SKU)
        .alias("_mad")
    ).with_columns(
        pl.min_horizontal(
            pl.col(target),
            pl.coalesce(
                pl.col("_med") + k * 1.4826 * pl.col("_mad"),
                pl.lit(float("inf")),
            ),
        ).alias(out)
    ).drop("_med", "_mad")


def drop_leaky(lf: pl.LazyFrame, extra: list[str] | None = None) -> pl.LazyFrame:
    """Remove the report's own rolling averages and bookkeeping timestamps."""
    cols = set(lf.collect_schema().names())
    doomed = (
        set(C.DAILY_LEAKY_COLS)
        | {"row_created", "row_updated", "id", "date_purchased"}
        | set(extra or [])
    ) & cols
    if doomed:
        log.debug("dropping %d leaky/bookkeeping columns", len(doomed))
    return lf.drop(doomed)


# --------------------------------------------------------------------------------------
# Product master
# --------------------------------------------------------------------------------------


def clean_product(lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """Canonicalise the product master, one row per SKU."""
    lf = lf if lf is not None else pl.scan_parquet(C.PQ_PRODUCT)
    schema = lf.collect_schema().names()

    lf = lf.filter(pl.col(C.SKU).is_not_null())

    # Volume/size features are more informative than three raw dimensions.
    dims = ("item_dimension_lengthx", "item_dimension_width", "item_dimension_depth")
    if all(d in schema for d in dims):
        lf = lf.with_columns(
            (pl.col(dims[0]) * pl.col(dims[1]) * pl.col(dims[2])).alias("item_volume")
        )
    cdims = ("carton_dimension_lengthx", "carton_dimension_width", "carton_dimension_depth")
    if all(d in schema for d in cdims):
        lf = lf.with_columns(
            (pl.col(cdims[0]) * pl.col(cdims[1]) * pl.col(cdims[2])).alias("carton_volume")
        )

    # SKUs follow a BRAND-FAMILY-VARIANT convention (AB-110-6, AC-111-W). The prefixes are
    # a free, high-signal hierarchy for pooling across related products.
    lf = lf.with_columns(
        pl.col(C.SKU).str.split("-").list.get(0, null_on_oob=True).alias("sku_brand"),
        pl.col(C.SKU).str.split("-").list.slice(0, 2).list.join("-").alias("sku_family"),
        pl.col(C.SKU).str.len_chars().alias("sku_len"),
    )

    if "sale_price" in schema and "po_price" in schema:
        lf = lf.with_columns(
            pl.when(pl.col("sale_price") > 0)
            .then((pl.col("sale_price") - pl.col("po_price")) / pl.col("sale_price"))
            .otherwise(None)
            .alias("list_margin")
        )

    cats = [c for c in C.PRODUCT_CATEGORICALS if c in schema]
    lf = lf.with_columns([pl.col(c).cast(pl.String).cast(pl.Categorical) for c in cats])

    return lf.unique(subset=[C.SKU], keep="first")


def attach_product(lf: pl.LazyFrame, product: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """Left-join the product master onto a sales panel and derive age features."""
    product = product if product is not None else clean_product()
    pcols = product.collect_schema().names()
    product = product.drop([c for c in ("product_id", "name") if c in pcols])

    out = lf.join(product, on=C.SKU, how="left")
    if "date_on_market" in product.collect_schema().names():
        out = out.with_columns(
            (pl.col(C.DATE) - pl.col("date_on_market").dt.date())
            .dt.total_days()
            .alias("days_on_market")
        )
    return out


# --------------------------------------------------------------------------------------
# Weekly
# --------------------------------------------------------------------------------------


def clean_weekly(lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """Canonicalise the weekly sales rows, one row per (sku, week-start date)."""
    lf = lf if lf is not None else pl.scan_parquet(C.PQ_WEEKLY)
    schema = lf.collect_schema().names()

    lf = lf.filter(pl.col(C.DATE).is_not_null() & pl.col(C.SKU).is_not_null())

    # ``date_type`` partitions the report by channel/scope; ALL is the total.
    if "date_type" in schema:
        lf = lf.filter(pl.col("date_type") == "ALL")

    lf = (
        lf.sort(["row_updated", "qty"], nulls_last=False)
        .unique(subset=[C.SKU, C.DATE], keep="last")
        .sort([C.SKU, C.DATE])
    )

    lf = lf.with_columns(
        pl.col("qty").cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0).alias(C.TARGET),
        pl.col("big_order_filter_qty")
        .cast(pl.Float64)
        .fill_null(pl.col("qty").cast(pl.Float64))
        .alias(C.TARGET_FILTERED),
        pl.col("oos_days").cast(pl.Float64).fill_null(0.0).alias("oos_days"),
        pl.col("sessions").cast(pl.Float64).alias("sessions"),
    ).with_columns(
        _safe_div(pl.col("sales_revenue").cast(pl.Float64), pl.col(C.TARGET))
        .alias("unit_price_raw"),
        (pl.col("oos_days") / 7.0).alias("oos_rate"),
        (pl.col("oos_days") >= 7).alias("is_oos"),
        # The weekly export has no is_deal flag; a non-zero deal discount is equivalent.
        (pl.col("deal_discount").cast(pl.Float64).fill_null(0.0).abs() > 0).alias("is_deal"),
        _safe_div(
            pl.col("deal_discount").cast(pl.Float64) + pl.col("coupon_discount").cast(pl.Float64),
            pl.col("sales_revenue").cast(pl.Float64),
        ).alias("discount_rate"),
        _safe_div(
            pl.col("return_amount").cast(pl.Float64),
            pl.col("sales_revenue").cast(pl.Float64),
        ).alias("return_rate"),
        _safe_div(pl.col(C.TARGET), pl.col("sessions")).alias("conversion_rate"),
    ).with_columns(
        pl.col("unit_price_raw").forward_fill().backward_fill().over(C.SKU).alias("unit_price")
    )

    return lf.drop([c for c in C.WEEKLY_LEAKY_COLS if c in lf.collect_schema().names()])
