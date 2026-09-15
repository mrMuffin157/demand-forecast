"""STEP 1 — turn the raw sales history into the one file every model in this folder reads.

    uv run python model_lab/prepare_data.py

It skips any stage that is already built, so re-running it is cheap. Add --force to
rebuild everything.

This is a pipeline stage, not only a lab convenience: it is the only writer of the two
tables `dev/forecast.py` trains on and forecasts from, so a scheduled data job should run
it after every pull. Because stages 1-3 are skipped when their outputs EXIST (not when
they are stale), such a refresh runs `demandml panel` / `demandml features` explicitly
first -- those always rebuild -- and then this file without --force.

-----------------------------------------------------------------------------------
WHAT IT DOES, IN ORDER
-----------------------------------------------------------------------------------
    data/db/sales_daily.parquet    one row per SKU per day: the database extract
    (or data/raw/*.csv)            (scripts/make_sample_data.py writes one for the
                                   sample, or bring your own); the CSV is only the
                                   fallback
         |  1. INGEST              typed Parquet, much smaller. Nothing is changed.
         v
    data/interim/*.parquet
         |  2. CLEAN + PANEL       this is where the real decisions are (see below)
         v
    data/processed/panel_daily.parquet
         |  3. FEATURES            hundreds of past-only predictors + the future totals we
         v                         are trying to predict
    data/processed/features_daily/
         |  4. THIS FILE'S OWN BIT thin it down to one compact modelling table
         v
    data/processed/lab_dataset.parquet      <- what every model file loads
    data/processed/open_origins.parquet     <- the newest origins, no targets yet:
                                               what a forecast starts from

Stages 1-3 are the project's existing `demandml` pipeline, imported rather than
reimplemented -- rewriting the cleaning would mean two versions of the truth. Stage 4 is
specific to this folder.

-----------------------------------------------------------------------------------
THE CLEANING DECISIONS THAT ACTUALLY MATTER (stages 2-3, worth knowing about)
-----------------------------------------------------------------------------------
* Stock-outs are censored, not zero. A day flagged out-of-stock sold 0 because nobody
  COULD buy it. Training on that teaches a model to predict your own stock-outs. Those
  days get sample_weight = 0, and (see below) origins whose future window is mostly
  stock-out get dropped from scoring entirely.
* Pre-launch and post-delist zeros are removed. A SKU launched years after the export
  begins still has rows of zeros going back to its first day; they are padding, not
  demand, and a large number of rows go.
* The export's last day is partial (a dump taken mid-day holds a tiny fraction of a
  normal day's units) and is dropped, or every model would see demand collapse to
  nearly nothing on the newest day.
* Bulk/wholesale orders are filtered out -- a single order worth months of a SKU's normal
  sales is not retail demand.
* The export ships pre-computed rolling averages (`avg_daily_weekly`, `no_oos_demand`,
  `avg8weeks`...). They were computed at row-write time and include the row's own day,
  i.e. they leak the answer. They are dropped and recomputed with correct timing.

-----------------------------------------------------------------------------------
STAGE 4: WHAT THIS FILE ITSELF DOES
-----------------------------------------------------------------------------------
1. Keeps one forecast origin per SKU per WEEK instead of per day.
   Daily origins are 86% redundant for a 6-month question -- Monday's and Tuesday's
   180-day windows share 179 days -- and weekly origins make every model in the folder
   run 7x faster for essentially the same information.

2. Builds the three targets: total units over the next 30 / 90 / 180 days.

   It also builds six SLICE targets, `y_cum_m1` .. `y_cum_m6` = units sold in month k
   of the future (days 1-30, 31-60, ... 151-180). Those exist for the additive models,
   which build a long forecast by predicting each month and summing, rather than
   predicting the long total directly. By construction
   y_cum_m1+..+y_cum_m6 == y_cum180, which tests/test_month_slices.py asserts.

3. Blanks out a target when its future window was mostly out of stock (>25% of days).
   You cannot measure a demand forecast on a period when the product was unbuyable --
   you would be scoring the model on the warehouse's behaviour, not on demand.
   A window that survives (up to 25% out of stock) also gets a TRAINING target,
   `y_cum{h}_lift` = the units divided by the window's in-stock share -- see LIFT_SUFFIX.

4. Keeps only the columns a model is allowed to see. That list is an ALLOWLIST from the
   main project (`features.feature_columns`): a column is included only if it is a
   recognised past-only engineered feature, a static product attribute, or a calendar
   column. Raw same-day measurements can never sneak in, which is what would otherwise
   quietly destroy the whole exercise.

The result: one row per SKU per week, a few hundred features, 3 targets (plus the month
slices and the y_cum270 validation target), in lab_dataset.parquet; and
open_origins.parquet beside it, split off in the same pass (see the comment above the
split).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demandml import config as C  # noqa: E402
from demandml.features import feature_columns, load_features  # noqa: E402
from demandml import xfeatures  # noqa: E402

from common import (  # noqa: E402
    DATASET, DATASET_META, HORIZONS, MONTH_DAYS, N_MONTHS, OPEN_ORIGINS,
)

#: Only build origins from here on. Earlier history is still used to *compute* the
#: features (a 364-day rolling mean needs a year of runway), it just does not become a
#: training row -- demand from many years back says little about today.
ORIGIN_START = date(2018, 1, 1)

#: Keep one origin per week. 1 = Monday.
ORIGIN_WEEKDAY = 1

#: Drop a target whose future window was out of stock more than this share of the time.
MAX_OOS_FRACTION = 0.25

#: TRAINING targets: `y_cum{h}_lift` = `y_cum{h}` / (share of the window in stock), for a
#: window that survived the rule above. A window 20% out of stock sold ~20% less than its
#: demand, and a model trained on the raw number learns that dip as if it were demand.
#: Lifting it made the 90-day forecast land within +/-40% of what sold slightly more
#: often, consistently across folds (see docs/FINDINGS.md). Scoring stays on `y_cum{h}` --
#: what sold is the truth; the lift only changes what a model is taught. Named `y_cum*`
#: so the feature ban catches it, and read by `dev/lifted_log.py`. It assumes an out-of-stock day would have sold at the
#: window's in-stock rate, which is crude for a SKU that ran out because it was surging.
LIFT_SUFFIX = "_lift"

#: How far back the "open origins" file reaches. Only the newest origin per SKU is
#: actually forecast from, but a SKU that sold nothing last week may have no row at the
#: very newest date, so the file keeps a short tail to fall back through.
OPEN_ORIGIN_DAYS = 56

#: Horizons carried as TARGETS but never modelled. `y_cum270` exists so the flat-rate
#: extension past the 180-day knot can be scored against truth instead of assumed: the
#: coverage window a purchase is exposed to is the lead time + review period (see
#: assumptions.toml), which can run past 180 days, and then every order quantity depends
#: on that extension (`dev/longhorizon.py` measures it). Deliberately NOT in `HORIZONS` --
#: adding it there would refit every model at a fourth horizon and rebuild every board,
#: which is a separate decision. It is named `y_cum*` so `feature_columns()`'s ban catches
#: it and it can never become a predictor.
VALIDATION_HORIZONS: tuple[int, ...] = (270,)


def _replace(path: Path, write) -> None:
    """Write next to `path`, then swap it in with one rename. Another store's job reads
    this file for pooled training (`dev/common.load_pool`), and a reader that
    lands mid-write would otherwise see a truncated Parquet and drop the store."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _stage(msg: str) -> float:
    print(f"\n=== {msg} " + "=" * max(4, 70 - len(msg)))
    return time.time()


def _done(t0: float) -> None:
    print(f"    ...done in {time.time() - t0:.0f}s")


def build_upstream(force: bool = False, sku_limit: int | None = None) -> None:
    """Stages 1-3: the project pipeline. Each is skipped if its output already exists."""
    from demandml.features import build_features
    from demandml.ingest import ingest_all
    from demandml.panel import build_panel

    if force or not C.PQ_DAILY.exists():
        t = _stage("STAGE 1/4  ingest: DB pull (else raw CSV) -> typed Parquet")
        ingest_all(overwrite=force)
        _done(t)
    else:
        print("=== STAGE 1/4  ingest: already built, skipping "
              "(--force to redo)".ljust(70, " "))

    if force or not C.PANEL_DAILY.exists():
        t = _stage("STAGE 2/4  panel: clean, trim dead history, densify the calendar")
        build_panel(freq="daily")
        _done(t)
    else:
        print("=== STAGE 2/4  panel: already built, skipping")

    need_feats = force or not C.FEATURES_DAILY.exists() or not any(
        C.FEATURES_DAILY.glob("*.parquet"))
    if not need_feats:
        # The lab needs 30/90/180-day targets. An older feature build only has 7/28.
        have = pl.scan_parquet(C.FEATURES_DAILY / "*.parquet").collect_schema().names()
        need_feats = any(f"y_cum{h}" not in have for h in HORIZONS)
        if need_feats:
            print("=== STAGE 3/4  features: rebuilding, the existing table has no "
                  "1/3/6-month targets")
    if need_feats:
        t = _stage("STAGE 3/4  features: past-only predictors + future totals")
        build_features(freq="daily", sku_limit=sku_limit)
        _done(t)
    else:
        print("=== STAGE 3/4  features: already built, skipping")


def build_lab_dataset() -> Path:
    """Stage 4: thin the (large) feature table down to one compact modelling table."""
    t = _stage("STAGE 4/4  lab dataset: weekly origins, 3 targets, allowlisted columns")
    lf = load_features("daily")
    have = set(lf.collect_schema().names())

    val_h = tuple(h for h in VALIDATION_HORIZONS if f"y_cum{h}" in have)
    if len(val_h) < len(VALIDATION_HORIZONS):
        print(f"    (no y_cum{VALIDATION_HORIZONS} in the feature table -- rebuild "
              f"features to score the long-window extrapolation)")
    missing = [f"y_cum{h}" for h in HORIZONS if f"y_cum{h}" not in have]
    if missing:
        raise SystemExit(
            f"the feature table has no {missing}. Rebuild it:\n"
            f"  uv run python model_lab/prepare_data.py --force"
        )

    # --- how much of each future window was out of stock? ------------------------------
    # Same reverse-rolling trick the targets use: reverse, take a trailing window, and
    # reverse back, which gives a *forward* window; .shift(-1) makes it exclusive of the
    # origin day itself. Partitioned by SKU so history never bleeds between products.
    oos = [
        (pl.col("is_censored").cast(pl.Float32)
         .reverse().rolling_mean(h, min_samples=h).reverse().shift(-1)
         .over(C.SKU).alias(f"_oos_frac_{h}")
        )
        for h in tuple(HORIZONS) + val_h
    ]

    # --- "the same window, one year ago" ----------------------------------------------
    # Shifting a target back 364 days turns it into a PAST observation: at origin t,
    # y_cum180 shifted by 364 covers days t-363 .. t-184, all of which are history. It is
    # the single most useful signal for a 6-month seasonal forecast and the allowlist
    # cannot classify it automatically (it starts with "y_cum"), so it is built and named
    # here. 364 rather than 365 keeps the day-of-week aligned.
    last_year = [
        pl.col(f"y_cum{h}").shift(364).over(C.SKU).alias(f"ly_cum{h}") for h in HORIZONS
    ] + [
        pl.col("units_mean_91").shift(364).over(C.SKU).alias("ly_mean_91"),
    ]
    ly_names = [f"ly_cum{h}" for h in HORIZONS] + ["ly_mean_91"]

    # --- the per-month slice targets (for the additive models) -------------------------
    # `y_cum_m{k}` = units sold in month k of the future, i.e. over days
    # 30*(k-1)+1 .. 30*k after the origin. It needs no new rolling window: y_cum30 at
    # row t+30*(k-1) is already "the 30 days starting there", so shifting y_cum30
    # forward by 30*(k-1) rows relabels it as month k of row t.
    #
    #   y_cum_m1 + y_cum_m2 + y_cum_m3  ==  y_cum90     (checked in tests)
    #
    # The name deliberately starts with "y_cum" so that `feature_columns()`'s ban on
    # that prefix catches these automatically and they can never become predictors.
    # MUST be a separate with_columns(): it reads _oos_frac_30 and y_cum30, and the
    # former is only created in the with_columns() immediately above.
    slices = []
    for k in range(1, N_MONTHS + 1):
        back = MONTH_DAYS * (k - 1)
        slices.append(
            pl.col("y_cum30").shift(-back).over(C.SKU).alias(f"y_cum_m{k}")
        )
        slices.append(
            pl.col("_oos_frac_30").shift(-back).over(C.SKU).alias(f"_oos_frac_m{k}")
        )
    slice_names = [f"y_cum_m{k}" for k in range(1, N_MONTHS + 1)]

    # --- extended features (`xf_*`): in-stock rates and sibling stock-outs -----------
    # Built from the daily panel (they need every SKU of a parent on the same day, which
    # the per-SKU-batched feature table cannot give) and joined onto the weekly origins
    # below, in this one pass, so a served row gets exactly what a trained row got. Only
    # `dev/blend.py` uses them; `dev/common.load_dataset` hides the prefix from the rest.
    xf = xfeatures.frame()
    xf_names = list(xfeatures.NAMES)

    ids = [C.SKU, C.DATE, C.TARGET, "sample_weight", "is_censored"]
    feats = feature_columns(lf) + ly_names + xf_names
    targets = [f"y_cum{h}" for h in HORIZONS]
    val_targets = [f"y_cum{h}" for h in val_h]
    lift_names = [f"y_cum{h}{LIFT_SUFFIX}" for h in HORIZONS]
    keep = [c for c in dict.fromkeys(ids + targets + lift_names + val_targets + slice_names
                                     + feats)
            if c in have or c in ly_names or c in slice_names or c in lift_names
            or c in xf_names]

    df = (
        lf.sort([C.SKU, C.DATE])
        .with_columns(oos + last_year)
        .with_columns(slices)
        .filter(
            (pl.col(C.DATE) >= ORIGIN_START)
            & (pl.col(C.DATE).dt.weekday() == ORIGIN_WEEKDAY)
        )
        .join(xf.lazy(), on=[C.SKU, C.DATE], how="left")
        # Blank a target out when its own future window was mostly unbuyable. Done per
        # horizon: a 30-day window can be usable while the 180-day one is not.
        .with_columns([
            pl.when(pl.col(f"_oos_frac_{h}") > MAX_OOS_FRACTION)
            .then(None)
            .otherwise(pl.col(f"y_cum{h}"))
            .alias(f"y_cum{h}")
            for h in tuple(HORIZONS) + val_h
        ])
        .with_columns([
            (pl.col(f"y_cum{h}") / (1.0 - pl.col(f"_oos_frac_{h}")))
            .alias(f"y_cum{h}{LIFT_SUFFIX}")
            for h in HORIZONS
        ])
        # Same rule for the month slices, each judged on its own month.
        .with_columns([
            pl.when(pl.col(f"_oos_frac_m{k}") > MAX_OOS_FRACTION)
            .then(None)
            .otherwise(pl.col(f"y_cum_m{k}"))
            .alias(f"y_cum_m{k}")
            for k in range(1, N_MONTHS + 1)
        ])
        .select(keep)
        .collect(engine="streaming")
    )

    # Two outputs from one pass, split on whether the future has happened yet.
    #
    #   DATASET       rows with at least one usable target -- what models train on.
    #   OPEN_ORIGINS  the newest origins, whose horizons run past the end of the data.
    #                 They have no targets and so are useless for training, but they are
    #                 the only rows you can actually FORECAST from: the newest row in
    #                 DATASET is a full horizon stale by construction (its 180-day target
    #                 had to have already happened). Splitting here rather than rebuilding
    #                 features elsewhere is the point -- a served row goes through exactly
    #                 the same expressions as a trained one, so there is no way for the
    #                 two to disagree.
    #
    # A row can also lack every target because the OOS rule blanked them all; those are
    # censored history, not open origins, so the split is on date as well.
    open_cut = df.get_column(C.DATE).max() - timedelta(days=OPEN_ORIGIN_DAYS)
    has_target = pl.any_horizontal([pl.col(c).is_not_null() for c in targets])
    open_rows = df.filter((pl.col(C.DATE) > open_cut) & ~has_target)
    df = df.filter(has_target)

    DATASET.parent.mkdir(parents=True, exist_ok=True)
    _replace(DATASET, lambda f: df.write_parquet(f, compression="zstd"))
    _replace(OPEN_ORIGINS, lambda f: open_rows.write_parquet(f, compression="zstd"))
    print(f"    open origins: {len(open_rows):,} rows after {open_cut} "
          f"-> {OPEN_ORIGINS.name}")
    meta = json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": len(df),
        "skus": int(df.get_column(C.SKU).n_unique()),
        "date_min": str(df.get_column(C.DATE).min()),
        "date_max": str(df.get_column(C.DATE).max()),
        "origin_start": str(ORIGIN_START),
        "origin_weekday": ORIGIN_WEEKDAY,
        "max_oos_fraction": MAX_OOS_FRACTION,
        "horizons": list(HORIZONS),
        "validation_horizons": list(val_h),
        "targets": targets,
        "train_targets": lift_names,
        "validation_targets": val_targets,
        "slice_targets": slice_names,
        "month_days": MONTH_DAYS,
        "features": [c for c in feats if c in df.columns],
        "derived_here": ly_names + xf_names,
    }, indent=2)
    _replace(DATASET_META, lambda f: f.write_text(meta))
    _done(t)

    print(f"\n{'=' * 78}")
    print(f"  {DATASET}")
    print(f"  {len(df):,} forecast origins x {len(feats)} features "
          f"({DATASET.stat().st_size / 1e6:.0f} MB on disk)")
    print(f"  {df.get_column(C.SKU).n_unique():,} SKUs, "
          f"{df.get_column(C.DATE).min()} -> {df.get_column(C.DATE).max()}")
    for h in HORIZONS:
        n = int(df.get_column(f"y_cum{h}").is_not_null().sum())
        print(f"  y_cum{h:<4} usable origins: {n:>9,}")
    for k in range(1, N_MONTHS + 1):
        n = int(df.get_column(f"y_cum_m{k}").is_not_null().sum())
        print(f"  y_cum_m{k}  (month {k} alone)  usable origins: {n:>9,}")
    print(f"{'=' * 78}")
    print("\nNow run a model, e.g.:")
    print("  uv run python model_lab/moving_average.py")
    print("  uv run python model_lab/lightgbm_gbdt.py")
    print("  uv run python model_lab/compare_models.py --run-all")
    return DATASET


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="rebuild every stage from the raw sources (DB pull, else CSV)")
    ap.add_argument("--sku-limit", type=int, default=None,
                    help="only build features for the first N SKUs (fast smoke test)")
    a = ap.parse_args()
    build_upstream(force=a.force, sku_limit=a.sku_limit)
    build_lab_dataset()
