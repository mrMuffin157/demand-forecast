"""
=======================================================================================
 DEMAND PREDICTION -- ONE-FILE WALKTHROUGH
=======================================================================================

Read this top to bottom. It is the entire problem in one script:

    raw CSV  ->  SQL  ->  clean  ->  reshape  ->  features  ->  models  ->  graphs

Every step prints what it did, so you can watch the data change shape. The production
version of all of this lives in ``src/demandml/`` as proper modules -- this file is the
narrative version, deliberately linear and deliberately verbose.

Run it:

    uv run python examples/walkthrough.py

Takes a couple of minutes. It works on a 40-SKU sample read straight from the raw daily
sales CSV, so it does not need any of the build artefacts to exist -- or a database
connection. The pipeline proper reads the same table from the database extract
(`data/db/sales_daily.parquet`, produced by `scripts/make_sample_data.py` for the sample
data) and only falls back to a CSV when there is no extract. Reading the CSV here is a
choice for a self-contained demo, not the project's data source.

Libraries on show: polars (out-of-core dataframes), duckdb (SQL over Parquet), pandas
(interop), statsmodels (seasonal decomposition), scikit-learn (GLM + gradient boosting),
LightGBM, XGBoost, matplotlib + seaborn (graphs).
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
import numpy as np
import polars as pl
import seaborn as sns
import xgboost as xgb
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg backend call)

warnings.filterwarnings("ignore")
console = Console()

ROOT = Path(__file__).resolve().parents[1]
# Same data root as the pipeline (`config.DATA`); set DEMANDML_DATA to keep it elsewhere.
DATA = Path(os.environ.get("DEMANDML_DATA", ROOT / "data"))
RAW = DATA / "raw"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = DATA / "interim"
SCRATCH.mkdir(parents=True, exist_ok=True)

N_SKUS = 40          # sample size -- keeps the whole script to ~2 minutes
HORIZON = 7          # predict units 7 days after the forecast origin

# Validated palette: blue / orange / aqua, fixed order, CVD-safe.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, RED = "#9ec5f4", "#d03b3b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
T0 = time.time()


def step(n: int, title: str) -> None:
    console.rule(f"[bold cyan]STEP {n}[/]  {title}", style="cyan", align="left")


def show(df: pl.DataFrame, title: str = "", n: int = 6, max_cols: int = 9) -> None:
    """Print a Polars frame as a rich table.

    ``max_cols`` matters: a 15-column table in an 80-column terminal squashes every
    value to an ellipsis, which is worse than showing fewer columns.
    """
    cols = df.columns[:max_cols]
    extra = df.width - len(cols)
    d = df.select(cols).head(n)
    if extra > 0:
        title = f"{title}   (first {len(cols)} of {df.width} columns)"
    t = Table(title=title, header_style="bold cyan", title_justify="left")
    for c in d.columns:
        t.add_column(c, justify="right" if d.schema[c].is_numeric() else "left", max_width=22)
    for row in d.iter_rows():
        t.add_row(*[f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row])
    console.print(t)


# =======================================================================================
step(1, "IMPORT FROM CSV")
# =======================================================================================
# A full daily sales file runs to gigabytes -- too big to load naively. `scan_csv` builds
# a LAZY query: nothing is read until `.collect()`, and Polars pushes the column selection
# and row filter down into the CSV reader, so we only ever materialise what we ask for.

csv_path = sorted(RAW.glob("daily_sales_*.csv"))[-1]
console.print(f"source: [bold]{csv_path.name}[/]  ({csv_path.stat().st_size / 1e9:.2f} GB)")

lazy = pl.scan_csv(
    csv_path,
    # Sniffing gets these wrong: SKUs look numeric, the flags are "Y"/"N" strings.
    schema_overrides={"sku": pl.String, "is_oos": pl.String, "is_deal": pl.String,
                      "date_purchased": pl.String, "last_updated": pl.String},
    infer_schema_length=50_000,
    null_values=["", "NULL", "null"],
)
all_cols = lazy.collect_schema().names()
console.print(f"the export has [bold]{len(all_cols)}[/] columns; here are the ones we need:\n")
console.print(f"  [dim]{', '.join(all_cols[:14])} ...[/]\n")

# Sample STRATIFIED across the volume range. Taking the busiest N would pick the least
# intermittent SKUs in the catalogue and make the data look far tidier than it is.
ranked = (
    lazy.group_by("sku")
        .agg(pl.col("salescount").sum().alias("units"))
        .filter(pl.col("units") >= 200)       # below this there is nothing to learn
        .sort("units", descending=True)
        .collect()
)
idx = np.linspace(0, len(ranked) - 1, N_SKUS).round().astype(int)
sample = ranked[idx]
skus = sample.get_column("sku").to_list()
console.print(
    f"sampling [bold]{N_SKUS}[/] SKUs spread evenly across the volume range "
    f"({len(ranked)} SKUs qualify):\n"
    f"  busiest  {sample.get_column('units')[0]:>10,} lifetime units  ({skus[0]})\n"
    f"  median   {sample.get_column('units')[N_SKUS // 2]:>10,} lifetime units  "
    f"({skus[N_SKUS // 2]})\n"
    f"  quietest {sample.get_column('units')[-1]:>10,} lifetime units  ({skus[-1]})"
)

KEEP = ["sku", "date_purchased", "salescount", "revenue", "sessions", "ad_cost",
        "impressions", "clicks", "order_count", "is_oos", "is_deal", "deal_discount",
        "return_amount", "big_order_filter_sales_count", "last_updated"]

raw = (
    lazy.filter(pl.col("sku").is_in(skus))
        .select([c for c in KEEP if c in all_cols])
        .collect()
)
console.print(f"\nread [bold]{len(raw):,}[/] rows in {time.time() - T0:.1f}s "
              f"(the SKU filter was pushed into the reader)")
show(raw, "raw rows, exactly as they come out of the CSV")

# Park it as Parquet: 10x smaller, typed, and SQL-queryable in the next step.
pq = SCRATCH / "_walkthrough_sample.parquet"
raw.write_parquet(pq, compression="zstd")
console.print(f"cached as Parquet: {pq.stat().st_size / 1e6:.1f} MB "
              f"[dim](the CSV slice would be ~{len(raw) * 180 / 1e6:.0f} MB)[/]")


# =======================================================================================
step(2, "POKE AT IT WITH SQL (duckdb)")
# =======================================================================================
# DuckDB queries the Parquet file in place -- no load step, no server. This is the
# fastest way to answer "what does this data actually look like" before committing to a
# pipeline.

q = duckdb.sql(f"""
    SELECT
        substr(date_purchased, 1, 4)              AS year,
        count(*)                                  AS rows,
        sum(salescount)                           AS units,
        round(avg(salescount), 1)                 AS avg_units_per_day,
        round(100.0 * avg(CASE WHEN salescount = 0 THEN 1 ELSE 0 END), 1) AS pct_zero_days,
        round(100.0 * avg(CASE WHEN is_oos = 'Y'  THEN 1 ELSE 0 END), 1) AS pct_oos_days,
        round(sum(revenue) / 1000, 0)             AS revenue_k
    FROM '{pq}'
    GROUP BY 1 ORDER BY 1
""").pl()
show(q, "SQL straight over the Parquet file -- demand by year", n=12)

zero_share = float((raw.get_column("salescount") == 0).mean())
console.print(
    f"[yellow]Note the pct_zero_days column.[/] Across this sample "
    f"[bold]{zero_share:.0%}[/] of all SKU-days have zero sales. "
    f"That single fact rules out MAPE as a metric --\nit is undefined at zero -- and "
    f"makes squared-error loss a poor choice. It is the most\nimportant thing to know "
    f"about this dataset, and most of what follows is a response to it."
)


# =======================================================================================
step(3, "ORGANIZE AND MANIPULATE  (the actual cleaning)")
# =======================================================================================
# Six separate problems, each of which would quietly corrupt a model.

before = len(raw)
clean = raw

# (a) Parse the dates. They arrive as strings in two different formats.
clean = clean.with_columns(
    pl.col("date_purchased").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
    pl.col("last_updated").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
).drop("date_purchased", "last_updated" if "last_updated" in clean.columns else "date")

# (b) "Y"/"N" -> real booleans. Left as strings, a model treats them as categories and
#     you lose the ordering -- and `fill_null` cannot work on them.
clean = clean.with_columns(
    (pl.col("is_oos") == "Y").fill_null(False).alias("is_oos"),
    (pl.col("is_deal") == "Y").fill_null(False).alias("is_deal"),
)

# (c) De-duplicate. This is a report table that has been rebuilt over the years, so the
#     same SKU-day can appear more than once. Keep the most recently updated row.
dupes = len(clean) - clean.select(pl.struct("sku", "date").n_unique()).item()
clean = clean.sort("date").unique(subset=["sku", "date"], keep="last").sort(["sku", "date"])

# (d) Derive the economics the raw columns only imply.
clean = clean.with_columns(
    pl.col("salescount").cast(pl.Float64).fill_null(0).clip(lower_bound=0).alias("units"),
    pl.col("revenue").cast(pl.Float64).fill_null(0),
).with_columns(
    # Unit price is only observable on days when something sold.
    pl.when(pl.col("units") > 0).then(pl.col("revenue") / pl.col("units"))
      .otherwise(None).alias("price_raw"),
    pl.when(pl.col("impressions") > 0).then(pl.col("clicks") / pl.col("impressions"))
      .otherwise(None).alias("ctr"),
    pl.when(pl.col("sessions") > 0).then(pl.col("units") / pl.col("sessions"))
      .otherwise(None).alias("conversion"),
)

# (e) Forward-fill the price within each SKU, so zero-sale days still carry one.
#     `.over("sku")` is the important bit -- without it the fill would bleed across SKUs.
clean = clean.with_columns(
    pl.col("price_raw").forward_fill().backward_fill().over("sku").alias("price")
)

# (f) Stock-out censoring. THE subtle one. On an out-of-stock day sales are SUPPRESSED:
#     the observed 0 is missing demand, not zero demand. Train on it as a real zero and
#     you build a model that forecasts your own stock-outs. We flag it and give it a
#     sample weight of 0 so it can be excluded from the loss.
clean = clean.with_columns(
    pl.when(pl.col("is_oos") & (pl.col("units") <= 0)).then(0.0)
      .otherwise(1.0).alias("weight")
)

censored = int((clean.get_column("weight") == 0).sum())
t = Table(title="what the cleaning did", header_style="bold cyan", title_justify="left")
t.add_column("problem"); t.add_column("found", justify="right"); t.add_column("fix")
t.add_row("duplicate (sku, date) rows", f"{dupes:,}", "kept the most recently updated")
t.add_row("dates as strings", f"{before:,}", "parsed to Date")
t.add_row('"Y"/"N" flags', "2 cols", "cast to Boolean")
t.add_row("price undefined on zero-sale days",
          f"{clean.get_column('price_raw').null_count():,}", "forward-filled within SKU")
t.add_row("stock-out censored zeros", f"{censored:,}", "sample weight set to 0")
console.print(t)
show(clean.select("sku", "date", "units", "revenue", "price", "sessions", "is_oos", "weight"),
     "after cleaning")


# =======================================================================================
step(4, "RESHAPE INTO A PANEL  (one row per SKU per day, no gaps)")
# =======================================================================================
# Lag features are only meaningful on a regular grid. Two moves here:
#   1. Trim each SKU to its real selling life -- a SKU launched years after the report
#      begins has years of zero rows in it, and those are not demand observations.
#   2. `upsample` to fill any missing calendar days.

life = (
    clean.group_by("sku")
         .agg(pl.when(pl.col("units") > 0).then(pl.col("date")).min().alias("first_sale"),
              pl.when(pl.col("units") > 0).then(pl.col("date")).max().alias("last_sale"))
)
panel = (
    clean.join(life, on="sku")
         .filter((pl.col("date") >= pl.col("first_sale")) & (pl.col("date") <= pl.col("last_sale")))
         .drop("first_sale", "last_sale")
         .sort(["sku", "date"])
)
trimmed = len(clean) - len(panel)

panel = (
    panel.upsample(time_column="date", every="1d", group_by="sku", maintain_order=True)
         .with_columns(pl.col("sku").forward_fill())
         .with_columns(pl.col("units").is_null().alias("was_missing"))
         .with_columns([pl.col(c).fill_null(0.0) for c in
                        ("units", "revenue", "sessions", "ad_cost", "impressions",
                         "clicks", "order_count")]
                       + [pl.col("price").forward_fill().over("sku"),
                          pl.col("is_oos").fill_null(False),
                          pl.col("is_deal").fill_null(False),
                          pl.col("weight").fill_null(1.0)])
)

# The last day of the export is partial -- the dump was taken mid-day. Left in, it
# reads as a near-total demand collapse. Drop any trailing day under half the recent norm.
by_day = panel.group_by("date").agg(pl.col("units").sum().alias("u")).sort("date")
tail_u = by_day.get_column("u").to_list()
ref = float(np.median(tail_u[-29:-1]))
if tail_u[-1] < 0.5 * ref:
    cut = by_day.get_column("date")[-2]
    console.print(f"[yellow]dropping the final partial day[/] "
                  f"({tail_u[-1]:,.0f} units vs a {ref:,.0f} norm) -- keeping up to {cut}")
    panel = panel.filter(pl.col("date") <= cut)

console.print(
    f"\ntrimmed [bold]{trimmed:,}[/] pre-launch / post-delist zero rows\n"
    f"panel is now [bold]{len(panel):,}[/] rows x {panel.width} cols, "
    f"{panel.get_column('sku').n_unique()} SKUs x {panel.get_column('date').n_unique()} days\n"
    f"zero-sales share fell from "
    f"{(raw.get_column('salescount') == 0).mean():.1%} (raw) to "
    f"{(panel.get_column('units') == 0).mean():.1%} (panel) -- the difference was padding"
)


# =======================================================================================
step(5, "FEATURE ENGINEERING  (window functions, and the one rule that matters)")
# =======================================================================================
# THE RULE: a row is a forecast ORIGIN. Every feature on it may use data from that day
# or earlier -- never later. The target is a separate column holding units H days ahead.
#
# `.over("sku")` on every window is what keeps one SKU's history out of another's.

F = panel.sort(["sku", "date"]).with_columns([
    # --- lags: units k days before the origin -----------------------------------------
    *[pl.col("units").shift(k).over("sku").alias(f"lag_{k}") for k in
      (0, 1, 2, 3, 7, 14, 21, 28, 364)],

    # --- rolling moments: windows ENDING on the origin day, inclusive ------------------
    *[pl.col("units").rolling_mean(w, min_samples=2).over("sku").alias(f"mean_{w}")
      for w in (7, 14, 28, 91)],
    *[pl.col("units").rolling_std(w, min_samples=2).over("sku").alias(f"std_{w}")
      for w in (7, 28)],
    pl.col("units").rolling_median(28, min_samples=7).over("sku").alias("median_28"),
    pl.col("units").rolling_max(28, min_samples=7).over("sku").alias("max_28"),

    # --- exponentially weighted means: the strongest single feature in practice --------
    *[pl.col("units").ewm_mean(alpha=a, ignore_nulls=True).over("sku")
        .alias(f"ewm_{str(a).replace('.', '')}") for a in (0.1, 0.3, 0.6)],

    # --- intermittency: how lumpy is this series at all? ------------------------------
    (pl.col("units") == 0).cast(pl.Float64).rolling_mean(28, min_samples=7)
        .over("sku").alias("zero_rate_28"),

    # --- lagged exogenous drivers ------------------------------------------------------
    pl.col("sessions").shift(1).over("sku").alias("sessions_lag1"),
    pl.col("sessions").rolling_mean(7, min_samples=2).over("sku").alias("sessions_mean7"),
    pl.col("ad_cost").rolling_mean(7, min_samples=2).over("sku").alias("adcost_mean7"),
    pl.col("order_count").shift(1).over("sku").alias("orders_lag1"),

    # --- calendar: knowable arbitrarily far ahead -------------------------------------
    pl.col("date").dt.weekday().alias("dow"),
    pl.col("date").dt.month().alias("month"),
    pl.col("date").dt.ordinal_day().alias("doy"),
    pl.col("date").dt.year().alias("year"),
    (pl.col("date").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
]).with_columns([
    # Fourier terms give a tree model a smooth annual seasonality it can extrapolate.
    (2 * np.pi * pl.col("doy") / 365.25).sin().alias("fourier_sin1"),
    (2 * np.pi * pl.col("doy") / 365.25).cos().alias("fourier_cos1"),
    (4 * np.pi * pl.col("doy") / 365.25).sin().alias("fourier_sin2"),
    (4 * np.pi * pl.col("doy") / 365.25).cos().alias("fourier_cos2"),
    # Same weekday, averaged over the last 4 occurrences.
    pl.col("units").rolling_mean(4, min_samples=2)
      .over(["sku", pl.col("date").dt.weekday()]).alias("same_dow_mean4"),
    # Momentum: is this SKU running hot or cold against its own baseline?
    pl.when(pl.col("mean_28") > 0).then(pl.col("mean_7") / pl.col("mean_28"))
      .otherwise(None).alias("ratio_7_28"),
]).with_columns(
    # --- THE TARGET: units H days AFTER the origin. A negative shift, in its own column.
    pl.col("units").shift(-HORIZON).over("sku").alias("y")
)

FEATURES = [c for c in F.columns if c not in
            ("sku", "date", "y", "units", "revenue", "salescount", "price_raw",
             "big_order_filter_sales_count", "was_missing", "weight", "is_oos",
             "return_amount", "deal_discount", "sessions", "ad_cost", "impressions",
             "clicks", "order_count", "ctr", "conversion")]

console.print(f"built [bold]{len(FEATURES)}[/] features from {panel.width} raw columns\n")
console.print(f"  [dim]{', '.join(FEATURES)}[/]\n")

# --- PROVE the timing is right, rather than asserting it -------------------------------
one = F.filter(pl.col("sku") == skus[0]).sort("date")
u = one.get_column("units").to_numpy()
checks = [
    ("lag_0  == units[t]                 ", one["lag_0"].to_numpy()[1:-30], u[1:-30]),
    ("lag_1  == units[t-1]               ", one["lag_1"].to_numpy()[2:-30], u[1:-31]),
    ("lag_7  == units[t-7]               ", one["lag_7"].to_numpy()[8:-30], u[1:-37]),
    ("mean_7 == mean(units[t-6..t])      ", one["mean_7"].to_numpy()[6:-30],
     np.convolve(u, np.ones(7) / 7, "valid")[:-30]),
    (f"y      == units[t+{HORIZON}]  (the target)", one["y"].to_numpy()[1:-30], u[1 + HORIZON:-30 + HORIZON]),
]
# markup=False: rich would otherwise parse "[t]" / "[t-1]" as style tags and drop them.
t = Table(title="timing verification -- no feature may see the future",
          header_style="bold cyan", title_justify="left")
t.add_column("assertion"); t.add_column("result")
for label, got, want in checks:
    m = np.isfinite(got) & np.isfinite(want)
    ok = np.allclose(got[m], want[m], atol=1e-6)
    from rich.text import Text
    t.add_row(Text(label), Text("PASS", style="green") if ok else Text("FAIL", style="red"))
console.print(t)
show(F.select("sku", "date", "units", "lag_1", "lag_7", "mean_7", "mean_28", "ewm_03", "y"),
     "the modelling table: features from the past, target from the future")


# =======================================================================================
step(6, "DECOMPOSE THE SEASONALITY  (statsmodels)")
# =======================================================================================
# Before modelling, look at what structure is even there. STL splits the series into
# trend + seasonal + residual. If the seasonal component is large, the model needs
# calendar features; if the residual dominates, expect a low ceiling on accuracy.

from statsmodels.tsa.seasonal import STL  # noqa: E402

total = (F.group_by("date").agg(pl.col("units").sum().alias("units"))
          .sort("date").tail(728))
stl = STL(total.get_column("units").to_numpy(), period=7, robust=True).fit()
var = {"trend": np.var(stl.trend), "seasonal": np.var(stl.seasonal), "resid": np.var(stl.resid)}
tot = sum(var.values())
t = Table(title="STL variance decomposition (last 2 years, weekly period)",
          header_style="bold cyan", title_justify="left")
t.add_column("component"); t.add_column("share of variance", justify="right")
for k, v in var.items():
    t.add_row(k, f"{100 * v / tot:5.1f}%")
console.print(t)


# =======================================================================================
step(7, "SPLIT IN TIME  (never randomly)")
# =======================================================================================
# A random train/test split on a panel like this puts the same SKU's adjacent days on
# both sides. Lag features make them near-duplicates, and the score comes out 2-3x too
# good. So: one clean cut in time, with a HORIZON-sized gap so the training rows'
# targets cannot reach into the test window.

model_df = F.drop_nulls("y").filter(pl.col("weight") > 0)
dates = sorted(model_df.get_column("date").unique().to_list())
test_start = dates[-56]
train_end = dates[-56 - HORIZON - 1]

train = model_df.filter(pl.col("date") <= train_end)
test = model_df.filter(pl.col("date") >= test_start)

console.print(f"train: {train['date'].min()} -> {train['date'].max()}   [bold]{len(train):,}[/] rows")
console.print(f"[dim]gap:   {HORIZON} days, so no training target lands inside the test window[/]")
console.print(f"test:  {test['date'].min()} -> {test['date'].max()}   [bold]{len(test):,}[/] rows")

Xtr = train.select(FEATURES).to_pandas()
Xte = test.select(FEATURES).to_pandas()
ytr = train.get_column("y").to_numpy()
yte = test.get_column("y").to_numpy()


# =======================================================================================
step(8, "TRAIN SIX DIFFERENT MODELS")
# =======================================================================================


def wape(y, p):
    """Total absolute error / total actual units. Defined when y = 0, unlike MAPE."""
    return float(np.abs(y - np.clip(p, 0, None)).sum() / max(np.abs(y).sum(), 1e-9))


def bias(y, p):
    """Signed. Positive = over-forecasting. Costs differ, so always look at both."""
    return float((np.clip(p, 0, None) - y).sum() / max(np.abs(y).sum(), 1e-9))


preds: dict[str, np.ndarray] = {}
timings: dict[str, float] = {}


def fit(name: str, fn) -> None:
    t0 = time.time()
    preds[name] = np.clip(fn(), 0, None)
    timings[name] = time.time() - t0
    console.print(f"  {name:24s} wape={wape(yte, preds[name]):.4f}  "
                  f"bias={bias(yte, preds[name]):+.4f}  ({timings[name]:.1f}s)")


console.print("[bold]baselines[/] -- beat these or nothing here is earning its keep:")
# 1. Yesterday's number, carried forward.
fit("naive (yesterday)", lambda: test.get_column("lag_0").fill_null(0).to_numpy())
# 2. A 7-day trailing mean. Genuinely hard to beat.
fit("moving average (7d)", lambda: test.get_column("mean_7").fill_null(0).to_numpy())
# 3. Croston/SBA -- the textbook method for intermittent demand: mean non-zero demand
#    size divided by the mean interval between demands.
fit("croston (intermittent)", lambda: (
    (1 - 0.1 / 2) * test.get_column("mean_28").fill_null(0).to_numpy()
    / np.clip(1 - test.get_column("zero_rate_28").fill_null(0).to_numpy(), 0.05, 1)
    * np.clip(1 - test.get_column("zero_rate_28").fill_null(0).to_numpy(), 0.05, 1)
))

console.print("\n[bold]scikit-learn[/]:")
# 4. Poisson GLM -- a count model, interpretable, trains in seconds.
fit("sklearn PoissonGLM", lambda: make_pipeline(
    StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=400)
).fit(Xtr.fillna(0), ytr).predict(Xte.fillna(0)))
# 5. sklearn's own histogram gradient boosting, with Poisson loss.
fit("sklearn HistGradBoost", lambda: HistGradientBoostingRegressor(
    loss="poisson", max_iter=300, learning_rate=0.06, random_state=0
).fit(Xtr, ytr).predict(Xte))

console.print("\n[bold]gradient boosting[/] -- the workhorses:")
# 6. LightGBM with TWEEDIE loss. With this many zeros and a long right tail the target
#    is compound Poisson-Gamma, not Gaussian. Changing the loss is usually worth more
#    than any single feature.
lgbm = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2,
                         n_estimators=600, learning_rate=0.05, num_leaves=63,
                         min_child_samples=50, feature_fraction=0.8,
                         n_jobs=-1, verbose=-1, random_state=0)
fit("LightGBM (tweedie)", lambda: lgbm.fit(Xtr, ytr).predict(Xte))

# For contrast: the same model with the naive squared-error loss.
fit("LightGBM (squared err)", lambda: lgb.LGBMRegressor(
    objective="regression", n_estimators=600, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, n_jobs=-1, verbose=-1, random_state=0
).fit(Xtr, ytr).predict(Xte))

# 7. XGBoost -- a second opinion with a different inductive bias.
fit("XGBoost (tweedie)", lambda: xgb.XGBRegressor(
    objective="reg:tweedie", tweedie_variance_power=1.2, n_estimators=500,
    learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", n_jobs=-1, verbosity=0, random_state=0
).fit(Xtr, ytr).predict(Xte))

# 8. Quantile heads. The mean is not what a replenishment decision needs -- P90 is.
console.print("\n[bold]LightGBM quantiles[/] -- for safety stock, not point accuracy:")
qmodels = {}
for qq in (0.1, 0.5, 0.9):
    qmodels[qq] = lgb.LGBMRegressor(objective="quantile", alpha=qq, n_estimators=350,
                                    learning_rate=0.06, num_leaves=63, n_jobs=-1,
                                    verbose=-1, random_state=0).fit(Xtr, ytr)
qp = {qq: np.clip(m.predict(Xte), 0, None) for qq, m in qmodels.items()}
qp_sorted = np.sort(np.column_stack([qp[0.1], qp[0.5], qp[0.9]]), axis=1)
for i, qq in enumerate((0.1, 0.5, 0.9)):
    console.print(f"  P{int(qq * 100):<2d} realised share below = "
                  f"{np.mean(yte <= qp_sorted[:, i]):.3f}  (nominal {qq})")
console.print(
    "[dim]  P10 reads high because a low quantile of an intermittent series is ~0, and "
    "every\n  zero-demand day ties at or below it. That is the target's shape, not "
    "miscalibration.[/]"
)


# =======================================================================================
step(9, "SCORE THE LOT")
# =======================================================================================
base = wape(yte, preds["moving average (7d)"])
board = sorted(preds.items(), key=lambda kv: wape(yte, kv[1]))
t = Table(title=f"leaderboard -- horizon {HORIZON} days, {len(test):,} test rows",
          header_style="bold cyan", title_justify="left")
for c, j in (("model", "left"), ("WAPE", "right"), ("bias", "right"),
             ("MAE", "right"), ("vs 7d MA", "right"), ("fit", "right")):
    t.add_column(c, justify=j)
for name, p in board:
    w = wape(yte, p)
    imp = 100 * (1 - w / base)
    t.add_row(name, f"{w:.4f}", f"{bias(yte, p):+.4f}",
              f"{np.abs(yte - p).mean():.3f}",
              "[green]" + f"{imp:+.1f}%" + "[/]" if imp > 0 else f"{imp:+.1f}%",
              f"{timings[name]:.1f}s")
console.print(t)

imp_df = (pl.DataFrame({"feature": FEATURES,
                        "gain": lgbm.booster_.feature_importance("gain")})
            .sort("gain", descending=True))
show(imp_df, "what LightGBM actually leaned on", n=10)


# =======================================================================================
step(10, "GRAPHS")
# =======================================================================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.titlelocation": "left", "axes.titleweight": "semibold", "axes.titlesize": 10.5,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
    "legend.fontsize": 8.5, "text.color": INK, "figure.dpi": 130,
})

fig = plt.figure(figsize=(15.5, 13.2))
gs = fig.add_gridspec(5, 2, hspace=0.48, wspace=0.17,
                      height_ratios=[0.95, 1, 1, 1, 1],
                      top=0.955, bottom=0.045, left=0.055, right=0.985)

# --- (1) total demand over time, with its 28-day trend --------------------------------
ax = fig.add_subplot(gs[0, :])
td = F.group_by("date").agg(pl.col("units").sum().alias("u")).sort("date")
ax.plot(td["date"], td["u"], lw=0.7, color=BLUE_LIGHT, label="daily total")
ax.plot(td["date"], td["u"].rolling_mean(28), lw=2.0, color=BLUE, label="28-day mean")
ax.set_title(f"Total daily demand across {N_SKUS} SKUs")
ax.set_ylabel("units"); ax.legend(loc="upper left", ncols=2); ax.grid(axis="x", visible=False)

# --- (2) the zero problem -------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
uu = F.get_column("units").to_numpy()
ax.hist(uu[uu <= 60], bins=61, color=BLUE, zorder=3)
ax.set_title(f"Daily units per SKU — {(uu == 0).mean():.0%} are zero "
             f"(was {zero_share:.0%} before trimming)")
ax.set_xlabel("units sold that day"); ax.set_ylabel("SKU-days")
ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
ax.annotate(f"{(uu == 0).mean():.0%} zeros", xy=(0, (uu == 0).sum()),
            xytext=(14, -18), textcoords="offset points", fontsize=9,
            color=RED, fontweight="semibold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))

# --- (3) day-of-week seasonality ------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
dw = F.group_by("dow").agg(pl.col("units").mean().alias("u")).sort("dow")
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ax.bar(range(len(dw)), dw["u"], width=0.62, color=BLUE, zorder=3)
ax.set_xticks(range(len(dw)), [names[i - 1] for i in dw["dow"]])
ax.set_title("Weekly shape"); ax.set_ylabel("mean units/day")
ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
for i, v in enumerate(dw["u"]):
    ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)

# --- (4) STL decomposition ------------------------------------------------------------
ax = fig.add_subplot(gs[2, 0])
W = 112  # last 16 weeks -- the full 2 years renders the weekly component as a solid blob
xs = np.arange(W)
ax.plot(xs, stl.trend[-W:], color=BLUE, lw=2.0, label="trend", zorder=3)
ax.plot(xs, stl.seasonal[-W:] + stl.trend[-W:].mean(), color=ORANGE, lw=1.6,
        label="weekly seasonal (offset to trend)", zorder=4)
ax.plot(xs, (stl.trend + stl.seasonal + stl.resid)[-W:], color="#c3c2b7", lw=0.9,
        label="observed", zorder=2)
ax.set_title(f"STL decomposition — last {W // 7} weeks  "
             f"(trend {100 * var['trend'] / tot:.0f}% of variance, "
             f"seasonal {100 * var['seasonal'] / tot:.0f}%)")
ax.set_xlabel("days ago  ->  today"); ax.set_ylabel("units")
ax.legend(loc="upper left", ncols=3, fontsize=7.5)
ax.grid(axis="x", visible=False)

# --- (5) leaderboard ------------------------------------------------------------------
ax = fig.add_subplot(gs[2, 1])
nm = [n for n, _ in board][::-1]
wv = [wape(yte, p) for _, p in board][::-1]
cols = ["#184f95" if n == "moving average (7d)" else BLUE for n in nm]
ax.barh(range(len(nm)), wv, height=0.62, color=cols, zorder=3)
ax.axvline(base, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.set_yticks(range(len(nm)), nm, fontsize=8.5, color=INK)
ax.set_title("WAPE by model (dashed = 7-day MA baseline)")
ax.set_xlabel("WAPE (lower is better)"); ax.set_xlim(0, max(wv) * 1.24)
ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
for i, v in enumerate(wv):
    ax.text(v + max(wv) * 0.015, i, f"{v:.3f}", va="center", fontsize=8, color=INK)

# --- (6) feature importance -----------------------------------------------------------
ax = fig.add_subplot(gs[3, 0])
ti = (imp_df.with_columns((100 * pl.col("gain") / pl.col("gain").sum()).alias("pct"))
            .head(12).reverse())
ax.barh(range(len(ti)), ti["pct"], height=0.62, color=BLUE, zorder=3)
ax.set_yticks(range(len(ti)), ti["feature"], fontsize=8.5, color=INK)
ax.set_title("LightGBM feature importance — top 12")
ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
ax.set_xlabel("share of total gain (%)")
ax.set_xlim(0, float(ti["pct"].max()) * 1.16)
for i, v in enumerate(ti["pct"]):
    ax.text(v + float(ti["pct"].max()) * 0.012, i, f"{v:.1f}%", va="center",
            fontsize=7.5, color=INK)

# --- (7) predicted vs actual, best model ----------------------------------------------
ax = fig.add_subplot(gs[3, 1])
best_name, best_p = board[0]
sk = (test.group_by("sku").agg(pl.col("y").sum().alias("v"))
          .sort("v", descending=True).get_column("sku")[0])
mask = (test.get_column("sku") == sk).to_numpy()
dd = test.filter(pl.col("sku") == sk).get_column("date").to_list()
ax.plot(dd, yte[mask], color=BLUE, lw=2.0, label="actual")
ax.plot(dd, best_p[mask], color=ORANGE, lw=2.0, label=best_name)
ax.set_title(f"{sk} — actual vs {best_name}")
ax.set_ylabel("units"); ax.legend(loc="upper left", ncols=2)
ax.grid(axis="x", visible=False); ax.set_ylim(bottom=0)

# --- (8) prediction intervals ---------------------------------------------------------
ax = fig.add_subplot(gs[4, 0])
ax.fill_between(dd, qp_sorted[mask, 0], qp_sorted[mask, 2], color=BLUE_LIGHT,
                alpha=0.85, lw=0, label="P10–P90", zorder=2)
ax.plot(dd, qp_sorted[mask, 1], color=BLUE, lw=2.0, label="P50", zorder=3)
ax.plot(dd, yte[mask], color=ORANGE, lw=2.0, label="actual", zorder=4)
inside = np.mean((yte[mask] >= qp_sorted[mask, 0]) & (yte[mask] <= qp_sorted[mask, 2]))
ax.set_title(f"Quantile intervals — {inside:.0%} of actuals inside the band")
ax.set_ylabel("units"); ax.legend(loc="upper left", ncols=3)
ax.grid(axis="x", visible=False); ax.set_ylim(bottom=0)

# --- (9) residuals by demand level ----------------------------------------------------
ax = fig.add_subplot(gs[4, 1])
err = best_p - yte
buckets = np.digitize(yte, [0.5, 2, 5, 10, 25, 100])
labels = ["0", "1–2", "3–5", "6–10", "11–25", "26–100", "100+"]
means = [err[buckets == i].mean() if (buckets == i).sum() else 0.0 for i in range(7)]
ax.bar(range(7), means, width=0.62,
       color=[BLUE if m >= 0 else RED for m in means], zorder=3)
ax.axhline(0, color="#c3c2b7", lw=1.2, zorder=4)
ax.set_xticks(range(7), labels)
ax.set_title(f"Bias by demand level — {best_name}")
ax.set_xlabel("actual units that day"); ax.set_ylabel("mean signed error")
ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=BLUE, label="over-forecast"),
                   plt.Line2D([], [], marker="s", ls="", color=RED, label="under-forecast")],
          loc="upper left", ncols=2)

fig.suptitle("Demand prediction walkthrough — CSV to models to diagnostics",
             fontsize=13, fontweight="bold", x=0.006, ha="left", y=0.997)
out = OUT / "walkthrough.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)

console.rule(style="cyan")
console.print(f"[bold green]done in {time.time() - T0:.0f}s[/]  ->  [bold]{out}[/]")
console.print(f"[dim]9 panels: demand over time · the zero problem · weekly shape · STL ·"
              f" leaderboard · feature importance · actual vs predicted · intervals · bias[/]")
