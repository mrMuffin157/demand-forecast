"""Feature engineering for direct multi-horizon demand forecasting.

Timing convention (this is the whole ballgame -- read it before changing anything)
---------------------------------------------------------------------------------
Each row is a *forecast origin*: ``(sku, origin_date)``.

* Every dynamic feature on that row uses only data from ``origin_date`` or earlier.
  ``units_lag_0`` is the units sold on the origin date itself, which you genuinely know
  when you sit down to forecast; ``roll_mean_28`` is the mean over the 28 days ending on
  the origin date, inclusive.
* Targets are separate columns: ``y_h1 ... y_hH`` hold units on
  ``origin_date + 1 ... origin_date + H``.

So a model for horizon ``h`` trains on ``(features_at_t -> y_h)``. There is no way for a
horizon-h model to see day ``t+h``, by construction -- which is why the targets live in
their own columns rather than the features being re-shifted per horizon.

If your warehouse does not have yesterday's sales until T+2, set
``FeatureSpec.origin_lag = 2`` and every dynamic feature is pushed back accordingly,
while the targets stay put.
"""

from __future__ import annotations

import shutil

import polars as pl

from . import config as C
from .calendar_features import add_calendar
from .logging_utils import get_logger

log = get_logger(__name__)

# Columns that describe the row's identity or are targets -- never fed to a model.
ID_COLS = [C.SKU, C.DATE, "product_id", "name", "start", "end"]


# --------------------------------------------------------------------------------------
# Target-history features
# --------------------------------------------------------------------------------------


def target_history_exprs(
    col: str = C.TARGET, spec: C.FeatureSpec | None = None, prefix: str = "units"
) -> list[pl.Expr]:
    """Lags, rolling moments, same-weekday means, EWMs and intermittency stats.

    All windows end on the origin date (inclusive), then the whole block is shifted by
    ``spec.origin_lag``.
    """
    spec = spec or C.DEFAULT_FEATURES
    lag = spec.origin_lag
    y = pl.col(col).shift(lag).over(C.SKU)  # "what we know at the origin"
    out: list[pl.Expr] = [y.alias(f"{prefix}_lag_0")]

    for k in spec.lags:
        out.append(y.shift(k).over(C.SKU).alias(f"{prefix}_lag_{k}"))

    for w in spec.roll_windows:
        ms = max(2, w // 4)
        if "mean" in spec.roll_stats:
            out.append(y.rolling_mean(w, min_samples=ms).over(C.SKU).alias(f"{prefix}_mean_{w}"))
        if "std" in spec.roll_stats:
            out.append(y.rolling_std(w, min_samples=ms).over(C.SKU).alias(f"{prefix}_std_{w}"))
        if "median" in spec.roll_stats:
            out.append(
                y.rolling_median(w, min_samples=ms).over(C.SKU).alias(f"{prefix}_median_{w}")
            )
        if "max" in spec.roll_stats:
            out.append(y.rolling_max(w, min_samples=ms).over(C.SKU).alias(f"{prefix}_max_{w}"))
        if "zero_rate" in spec.roll_stats:
            out.append(
                (y == 0)
                .cast(pl.Float64)
                .rolling_mean(w, min_samples=ms)
                .over(C.SKU)
                .alias(f"{prefix}_zero_rate_{w}")
            )

    # Same weekday, N weeks back -- captures weekly shape without a dow x sku interaction.
    for n in spec.same_dow_windows:
        out.append(
            y.rolling_mean(7 * n, min_samples=n)
            .over(C.SKU)
            .alias(f"{prefix}_dow_mean_{n}w_all")
        )
        out.append(
            pl.col(col)
            .shift(lag)
            .rolling_mean(n, min_samples=2)
            .over([C.SKU, pl.col(C.DATE).dt.weekday()])
            .alias(f"{prefix}_same_dow_mean_{n}w")
        )

    for a in spec.ewm_alphas:
        tag = str(a).replace(".", "")
        out.append(y.ewm_mean(alpha=a, ignore_nulls=True).over(C.SKU).alias(f"{prefix}_ewm_{tag}"))

    return out


def trend_and_ratio_exprs(prefix: str = "units") -> list[pl.Expr]:
    """Momentum: short window vs long window. Ratios travel across SKUs of any scale."""
    def ratio(a: str, b: str, name: str) -> pl.Expr:
        return (
            pl.when(pl.col(b).abs() > 1e-9)
            .then(pl.col(a) / pl.col(b))
            .otherwise(None)
            .alias(name)
        )

    return [
        ratio(f"{prefix}_mean_7", f"{prefix}_mean_28", f"{prefix}_ratio_7_28"),
        ratio(f"{prefix}_mean_28", f"{prefix}_mean_91", f"{prefix}_ratio_28_91"),
        ratio(f"{prefix}_mean_28", f"{prefix}_mean_364", f"{prefix}_ratio_28_364"),
        ratio(f"{prefix}_mean_7", f"{prefix}_mean_364", f"{prefix}_ratio_7_364"),
        (pl.col(f"{prefix}_mean_7") - pl.col(f"{prefix}_mean_28")).alias(f"{prefix}_diff_7_28"),
        # Coefficient of variation: how predictable is this SKU at all?
        ratio(f"{prefix}_std_28", f"{prefix}_mean_28", f"{prefix}_cv_28"),
        ratio(f"{prefix}_std_91", f"{prefix}_mean_91", f"{prefix}_cv_91"),
        # Year-on-year: same period last year, which is the strongest signal for
        # seasonal products.
        ratio(f"{prefix}_mean_28", f"{prefix}_lag_364", f"{prefix}_yoy_ratio"),
    ]


def intermittency_exprs(col: str = C.TARGET, spec: C.FeatureSpec | None = None) -> list[pl.Expr]:
    """Syntetos-Boylan style intermittency descriptors.

    46% of SKU-days in this dataset have zero sales, so "how lumpy is this series" is a
    first-class feature, not a footnote. ADI = average interval between non-zero demands;
    CV2 = squared coefficient of variation of the non-zero demand sizes. Together they
    give the classic smooth / erratic / intermittent / lumpy quadrants.
    """
    spec = spec or C.DEFAULT_FEATURES
    lag = spec.origin_lag
    y = pl.col(col).shift(lag).over(C.SKU)
    nz = (y > 0).cast(pl.Float64)
    pos = pl.when(y > 0).then(y).otherwise(None)

    out: list[pl.Expr] = []
    for w in (91, 364):
        nz_rate = nz.rolling_mean(w, min_samples=14).over(C.SKU)
        out += [
            pl.when(nz_rate > 0).then(1.0 / nz_rate).otherwise(None).alias(f"adi_{w}"),
            (
                pos.rolling_std(w, min_samples=5).over(C.SKU)
                / pos.rolling_mean(w, min_samples=5).over(C.SKU)
            )
            .pow(2)
            .alias(f"cv2_{w}"),
            pos.rolling_mean(w, min_samples=5).over(C.SKU).alias(f"nonzero_mean_{w}"),
        ]

    # Days since the last non-zero sale, and since the last stock-out.
    out += [
        (
            pl.int_range(pl.len()).over(C.SKU)
            - pl.when(y > 0)
            .then(pl.int_range(pl.len()).over(C.SKU))
            .otherwise(None)
            .forward_fill()
            .over(C.SKU)
        ).alias("days_since_sale"),
        (
            pl.int_range(pl.len()).over(C.SKU)
            - pl.when(pl.col("is_oos"))
            .then(pl.int_range(pl.len()).over(C.SKU))
            .otherwise(None)
            .forward_fill()
            .over(C.SKU)
        ).alias("days_since_oos"),
    ]
    return out


# --------------------------------------------------------------------------------------
# Exogenous features
# --------------------------------------------------------------------------------------


def exog_exprs(spec: C.FeatureSpec | None = None, available: set[str] | None = None) -> list[pl.Expr]:
    """Lags and trailing means of the exogenous drivers (traffic, ads, price, rank, OOS).

    These are *endogenous in practice* -- ad spend responds to sales as much as sales
    respond to ad spend -- so only their past is used. Anything you can actually plan
    ahead (a booked deal, a price change) belongs in ``spec.known_future`` instead.
    """
    spec = spec or C.DEFAULT_FEATURES
    lag = spec.origin_lag
    out: list[pl.Expr] = []
    for c in spec.exog:
        if available is not None and c not in available:
            continue
        base = pl.col(c).cast(pl.Float64).shift(lag).over(C.SKU)
        for k in spec.exog_lags:
            out.append(base.shift(k).over(C.SKU).alias(f"{c}_lag_{k}") if k else base.alias(f"{c}_lag_0"))
        for w in spec.exog_roll_windows:
            ms = max(2, w // 4)
            out.append(base.rolling_mean(w, min_samples=ms).over(C.SKU).alias(f"{c}_mean_{w}"))
        out.append(
            pl.when(base.rolling_mean(spec.exog_roll_windows[-1], min_samples=7)
                    .over(C.SKU).abs() > 1e-9)
            .then(
                base.rolling_mean(spec.exog_roll_windows[0], min_samples=2).over(C.SKU)
                / base.rolling_mean(spec.exog_roll_windows[-1], min_samples=7).over(C.SKU)
            )
            .otherwise(None)
            .alias(f"{c}_momentum")
        )
    return out


def known_future_exprs(spec: C.FeatureSpec | None = None, available: set[str] | None = None) -> list[pl.Expr]:
    """Features whose *future* value is legitimately known at forecast time.

    A promo calendar is booked weeks ahead and a price change is a decision, not an
    observation, so for these we may look forward. Kept in a separate function so it is
    obvious which columns are allowed to break the past-only rule -- and so you can drop
    them wholesale if your deployment does not actually have a promo calendar.
    """
    spec = spec or C.DEFAULT_FEATURES
    out: list[pl.Expr] = []
    for c in spec.known_future:
        if available is not None and c not in available:
            continue
        out.append(pl.col(c).cast(pl.Float64).alias(f"kf_{c}_now"))
        # Relative price vs the SKU's own trailing level: the elasticity signal.
        if c == "unit_price":
            out.append(
                pl.when(
                    pl.col(c).cast(pl.Float64).rolling_mean(91, min_samples=14).over(C.SKU) > 1e-9
                )
                .then(
                    pl.col(c).cast(pl.Float64)
                    / pl.col(c).cast(pl.Float64).rolling_mean(91, min_samples=14).over(C.SKU)
                )
                .otherwise(None)
                .alias("kf_price_vs_91d")
            )
    return out


# --------------------------------------------------------------------------------------
# Cross-sectional features
# --------------------------------------------------------------------------------------


def add_group_features(df: pl.DataFrame, level: str = "sku_family") -> pl.DataFrame:
    """Family/brand-level demand on the origin date.

    A SKU with three weeks of history is unforecastable alone but perfectly forecastable
    as "the red variant of a family that sells 400/week". These columns carry that.
    Computed across the whole panel, so this runs once on the full frame rather than per
    SKU batch.
    """
    if level not in df.columns:
        return df
    grp = (
        df.group_by([level, C.DATE])
        .agg(
            pl.col(C.TARGET).sum().alias(f"{level}_units"),
            pl.col(C.TARGET).mean().alias(f"{level}_units_mean"),
            pl.len().alias(f"{level}_n_skus"),
        )
        .sort([level, C.DATE])
        .with_columns(
            pl.col(f"{level}_units")
            .rolling_mean(28, min_samples=7)
            .over(level)
            .alias(f"{level}_units_mean_28"),
        )
    )
    out = df.join(grp, on=[level, C.DATE], how="left")
    return out.with_columns(
        pl.when(pl.col(f"{level}_units") > 0)
        .then(pl.col(C.TARGET) / pl.col(f"{level}_units"))
        .otherwise(None)
        .alias(f"{level}_share")
    )


# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------


def target_exprs(horizons: tuple[int, ...] | range, col: str = C.TARGET) -> list[pl.Expr]:
    """``y_h{h}`` = the target ``h`` periods after the origin row."""
    return [pl.col(col).shift(-h).over(C.SKU).alias(f"y_h{h}") for h in horizons]


def cumulative_target_exprs(horizons: tuple[int, ...], col: str = C.TARGET) -> list[pl.Expr]:
    """``y_cum{h}`` = total demand over the next ``h`` periods.

    This, not the per-day value, is what a replenishment decision actually needs: how
    many units to cover the next lead-time window.
    """
    out = []
    for h in horizons:
        out.append(
            pl.col(col)
            .reverse()
            .rolling_sum(h, min_samples=h)
            .reverse()
            .shift(-1)
            .over(C.SKU)
            .alias(f"y_cum{h}")
        )
    return out


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def build_features(
    freq: str = "daily",
    spec: C.FeatureSpec | None = None,
    target: str = C.TARGET,
    horizons: tuple[int, ...] | None = None,
    # 270 is here because it is a horizon a PURCHASE can actually be exposed to, not a
    # round number: the coverage window is lead time + review period (assumptions.toml),
    # and when that runs past 180 days every order rests on a flat-rate extension past the
    # 180-day knot. 270 puts a real knot on the far side of such a window so
    # `policy.interpolate_cumulative` interpolates rather than extrapolates. It costs 90
    # more days of nulls at the recent end of the panel.
    cum_horizons: tuple[int, ...] = (7, 28, 30, 90, 180, 270),
    batch_size: int = 250,
    sku_limit: int | None = None,
    out_dir=None,
) -> "object":
    """Build the feature table and write it as Parquet parts, one per SKU batch.

    Batching by SKU keeps peak memory near ``batch_size / n_skus`` of the panel while
    remaining exactly equivalent to computing over the whole frame -- every window
    function here is partitioned ``over(SKU)``, so SKUs never interact. The one exception
    is ``add_group_features``, which is applied to the batch after the family aggregates
    have been joined on from the full panel.
    """
    from .panel import load_panel

    spec = spec or C.DEFAULT_FEATURES
    horizons = horizons or tuple(range(1, 29)) if freq == "daily" else tuple(range(1, 14))
    out_dir = out_dir or (C.FEATURES_DAILY if freq == "daily" else C.FEATURES_WEEKLY)

    panel = load_panel(freq)
    schema = set(panel.collect_schema().names())

    skus = (
        panel.select(pl.col(C.SKU).unique().sort())
        .collect()
        .get_column(C.SKU)
        .to_list()
    )
    if sku_limit:
        skus = skus[:sku_limit]
    log.info("building %s features for %d SKUs, horizons 1..%d", freq, len(skus), max(horizons))

    # Family-level daily aggregates, computed once over the full panel.
    fam = None
    if "sku_family" in schema:
        fam = (
            panel.group_by(["sku_family", C.DATE])
            .agg(
                pl.col(target).sum().alias("fam_units"),
                pl.len().alias("fam_n_skus"),
            )
            .sort(["sku_family", C.DATE])
            .with_columns(
                pl.col("fam_units").rolling_mean(28, min_samples=7).over("sku_family")
                .alias("fam_units_mean_28"),
                pl.col("fam_units").shift(1).over("sku_family").alias("fam_units_lag_1"),
                pl.col("fam_units").shift(7).over("sku_family").alias("fam_units_lag_7"),
            )
            .drop("fam_units")
            .collect()
            .lazy()
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_cols = 0
    for i in range(0, len(skus), batch_size):
        batch = skus[i : i + batch_size]
        lf = panel.filter(pl.col(C.SKU).is_in(batch)).sort([C.SKU, C.DATE])

        lf = lf.with_columns(
            target_history_exprs(target, spec)
            + exog_exprs(spec, available=schema)
            + known_future_exprs(spec, available=schema)
        )
        lf = lf.with_columns(intermittency_exprs(target, spec) + trend_and_ratio_exprs())
        lf = lf.with_columns(target_exprs(horizons, target) + cumulative_target_exprs(cum_horizons, target))

        if fam is not None:
            lf = lf.join(fam, on=["sku_family", C.DATE], how="left")

        df = add_calendar(lf, country=spec.country).collect(engine="streaming")

        if spec.float32:
            df = df.with_columns(
                [pl.col(c).cast(pl.Float32) for c, t in df.schema.items() if t == pl.Float64]
            )

        part = out_dir / f"part_{i // batch_size:04d}.parquet"
        df.write_parquet(part, compression="zstd")
        n_rows += len(df)
        n_cols = df.width
        log.info("  batch %d: %d SKUs, %s rows -> %s", i // batch_size, len(batch), f"{len(df):,}", part.name)

    size = sum(p.stat().st_size for p in out_dir.glob("*.parquet"))
    log.info("features: %s rows x %d cols, %.0f MB in %s", f"{n_rows:,}", n_cols, size / 1e6, out_dir)
    return out_dir


def load_features(freq: str = "daily") -> pl.LazyFrame:
    d = C.FEATURES_DAILY if freq == "daily" else C.FEATURES_WEEKLY
    if not d.exists() or not any(d.glob("*.parquet")):
        raise FileNotFoundError(f"{d} is empty -- run `demandml features` first")
    return pl.scan_parquet(d / "*.parquet")


def feature_columns(lf: pl.LazyFrame, drop_extra: list[str] | None = None) -> list[str]:
    """Every column that is safe to hand a model.

    This is an **allowlist**: a column qualifies only if it matches one of the engineered
    feature patterns, or is a known-static product attribute, or is a calendar column.
    Anything else -- including every raw same-day measurement in the export, and any
    column a future export adds -- is excluded by default.

    The polarity matters. An earlier denylist version of this function let
    ``salescount`` and ``big_order_filter_sales_count`` through: raw copies of the
    target, sitting in the feature matrix. They were same-day rather than future, so no
    backtest was invalidated, but they crowded out the properly-shifted features and the
    only reason it was noticed was a glance at the importance table. An allowlist cannot
    fail that way.
    """
    names = lf.collect_schema().names()
    banned = set(ID_COLS) | set(drop_extra or [])
    banned |= {n for n in names if n.startswith(("y_h", "y_cum"))}

    def allowed(n: str) -> bool:
        if n in banned:
            return False
        if n in C.STATIC_FEATURE_COLS or n in C.CALENDAR_FEATURE_COLS:
            return True
        if n.startswith(C.ENGINEERED_PREFIXES):
            return True
        return any(sub in n for sub in C.ENGINEERED_SUBSTRINGS)

    return [n for n in names if allowed(n)]


def rejected_columns(lf: pl.LazyFrame) -> list[str]:
    """The complement of ``feature_columns`` -- what is being withheld, and why it is
    worth eyeballing after a schema change. If something here should be a feature,
    classify it in ``config.STATIC_FEATURE_COLS`` rather than loosening the rule."""
    keep = set(feature_columns(lf))
    return [n for n in lf.collect_schema().names() if n not in keep]
