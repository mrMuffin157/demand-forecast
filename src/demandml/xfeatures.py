"""Extended features (`xf_*`): in-stock demand rates and sibling stock-outs.

Built from the daily panel and joined onto the modelling tables in
`model_lab/prepare_data.py`, so a training row and a served row go through the same
expressions in the same pass -- the rule that keeps train/serve skew out.

Only models that ask for them see them (`LabModel.extended_features`, today only
`dev/blend.py`); `dev/common.load_dataset` hides the `xf_` prefix from everything else,
so xgboost_gbdt, random_forest and every other script keep exactly the columns they were
measured on. Why they exist (see docs/FINDINGS.md): neither set helps a single model, but
a three-model blend using them was slightly more precise at 90 d than the single model,
in most folds and at a lower inventory cost.

Timing convention as `features.py`: every window is TRAILING and includes the origin day,
like `units_mean_*`. Nothing here reads a day after the origin -- `tests/test_xfeatures.py`
perturbs the future and checks no feature moves.

IN-STOCK RATES: trailing demand features count an out-of-stock
day as a day that sold ~nothing, so a SKU out for half the quarter looks like half the
demand. `xf_instock_rate_{w}` is units per IN-STOCK day over w days.

SIBLINGS: a variant's siblings share its parent, the SKU minus its last `-segment`
(`AB-100-X` -> `AB-100`). That is NOT `sku_family` (first two segments). When one variant
is out of stock its siblings absorb part of its lost demand, so their
recent sales are inflated by a stock-out the model could not otherwise see. Weights are
each sibling's trailing-364-day units, so a dead variant going out counts for nothing.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from demandml import config as C

PREFIX = "xf_"
WINDOWS = (7, 28, 91, 364)

INSTOCK = ([f"{PREFIX}instock_rate_{w}" for w in WINDOWS]
           + [f"{PREFIX}instock_days_{w}" for w in WINDOWS]
           + [f"{PREFIX}instock_ratio_28_91", f"{PREFIX}instock_ratio_91_364", f"{PREFIX}instock_q_91"])
SIBLING = [f"{PREFIX}sib_oos_w7", f"{PREFIX}sib_oos_w28", f"{PREFIX}sib_oos_now",
           f"{PREFIX}fam_share_364", f"{PREFIX}n_sibs"]
NAMES = INSTOCK + SIBLING


def parent_expr(col: str = C.SKU) -> pl.Expr:
    return pl.col(col).str.replace(r"-[^-]+$", "")


def frame(panel: pl.LazyFrame | Path | None = None) -> pl.DataFrame:
    """(sku, date, xf_*) for every panel day. `panel` defaults to the daily panel."""
    if panel is None or isinstance(panel, Path):
        panel = pl.scan_parquet(panel or C.PROCESSED / "panel_daily.parquet")
    p = (panel.select(C.SKU, C.DATE, C.TARGET, "is_oos").sort(C.SKU, C.DATE)
         .with_columns(pl.col("is_oos").fill_null(False).cast(pl.Float64).alias("_o"),
                       pl.col(C.TARGET).fill_null(0).cast(pl.Float64).alias("_u"),
                       parent_expr().alias("_par"))
         .with_columns((1.0 - pl.col("_o")).alias("_in"),
                       pl.when(pl.col("_o") > 0).then(0.0).otherwise(pl.col("_u")).alias("_u_in")))
    roll = lambda e, w: e.rolling_sum(w, min_samples=1).over(C.SKU)  # noqa: E731
    p = p.with_columns(
        *[roll(pl.col("_in"), w).alias(f"_d{w}") for w in WINDOWS],
        *[roll(pl.col("_u_in"), w).alias(f"_s{w}") for w in WINDOWS],
        *[pl.col("_o").rolling_mean(w, min_samples=1).over(C.SKU).alias(f"_o{w}") for w in (7, 28)],
        roll(pl.col("_u"), 364).alias("_w"),
    ).with_columns(
        *[pl.when(pl.col(f"_d{w}") > 0).then(pl.col(f"_s{w}") / pl.col(f"_d{w}"))
          .alias(f"{PREFIX}instock_rate_{w}") for w in WINDOWS],
        *[pl.col(f"_d{w}").alias(f"{PREFIX}instock_days_{w}") for w in WINDOWS],
    ).with_columns(
        (pl.col(f"{PREFIX}instock_rate_28") / pl.col(f"{PREFIX}instock_rate_91")).alias(f"{PREFIX}instock_ratio_28_91"),
        (pl.col(f"{PREFIX}instock_rate_91") / pl.col(f"{PREFIX}instock_rate_364")).alias(f"{PREFIX}instock_ratio_91_364"),
        (pl.col(f"{PREFIX}instock_rate_91") * 91).alias(f"{PREFIX}instock_q_91"),
    ).collect(engine="streaming")

    fam = p.group_by("_par", C.DATE).agg(
        pl.col("_w").sum().alias("_W"), pl.len().alias("_N"),
        (pl.col("_w") * pl.col("_o7")).sum().alias("_WO7"),
        (pl.col("_w") * pl.col("_o28")).sum().alias("_WO28"),
        (pl.col("_w") * pl.col("_o")).sum().alias("_WO1"))
    p = p.join(fam, on=["_par", C.DATE], how="left")
    rest = pl.col("_W") - pl.col("_w")
    sib = lambda num, own: pl.when(rest > 0).then((pl.col(num) - pl.col("_w") * pl.col(own)) / rest)  # noqa: E731
    out = p.select(
        C.SKU, C.DATE, *INSTOCK,
        sib("_WO7", "_o7").alias(f"{PREFIX}sib_oos_w7"),
        sib("_WO28", "_o28").alias(f"{PREFIX}sib_oos_w28"),
        sib("_WO1", "_o").alias(f"{PREFIX}sib_oos_now"),
        pl.when(pl.col("_W") > 0).then(pl.col("_w") / pl.col("_W")).alias(f"{PREFIX}fam_share_364"),
        (pl.col("_N") - 1).cast(pl.Float64).alias(f"{PREFIX}n_sibs"),
    )
    # A ratio over zero is inf in polars; XGBoost refuses inf outright. Null instead.
    return out.with_columns([pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
                             for c in NAMES if out.schema[c].is_float()])
