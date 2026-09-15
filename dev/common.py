"""The small amount of plumbing that every model file in this folder shares.

This is the ONLY file here that is not a model. It exists because four things have to be
*identical* across models or the comparison is meaningless:

  1. the dataset they see            -> load_dataset()
  2. the backtest folds they use     -> make_folds()
  3. the metrics they are judged on  -> score()
  4. the figures and results they write -> run_model()

Everything that is specific to a model -- how it is shaped, what it assumes, how it is
trained -- lives in that model's own file, so you can read one file top to bottom and
understand one model completely.

-----------------------------------------------------------------------------------
THE FORECASTING QUESTION
-----------------------------------------------------------------------------------
Every row of the dataset is one *forecast origin*: "it is `date`, and I am looking at
SKU `sku`. How many units will it sell over the next N days?"

We ask that for three N:

    y_cum30   how many units over the next 1 month    (a reorder decision)
    y_cum90   how many units over the next 3 months   (a factory PO / container)
    y_cum180  how many units over the next 6 months   (a season / budget plan)

These are *cumulative* totals, not the demand on one single future day, because that is
the number a purchasing decision actually needs.

Features on a row only ever use data from `date` or earlier -- the timing convention the
whole project rests on (`tests/test_leakage.py` guards it).

-----------------------------------------------------------------------------------
WHY NOT A RANDOM SPLIT
-----------------------------------------------------------------------------------
The obvious request is "3 random splits, averaged". On this kind of dataset a random
split is not just imperfect, it is actively misleading, so what you get instead is
*rolling-origin* splits averaged -- same idea (train on some rows, test on held-out rows,
repeat, average), but the held-out rows are always in the FUTURE relative to the
training rows.

Why it matters: two rows for the same SKU one day apart are near-duplicates (their
28-day trailing averages overlap by 27 days out of 28). A random split puts one in
train and its twin in test, the model effectively memorises the answer, and the score
comes out far better than anything you would ever see on a real forward forecast. Every model here
would look brilliant and none of them would work.

By default the folds are three consecutive slices of recent history. `--folds` changes
the count, and for anything you will act on use more (around 12): three folds can sit
inside one demand regime and mis-state bias (`dev/regime.py`, docs/FINDINGS.md).

    fold 1:  train on everything up to A   ->  test on the 90 days after A
    fold 2:  train on everything up to B   ->  test on the 90 days after B
    fold 3:  train on everything up to C   ->  test on the 90 days after C

plus a gap of `horizon` days between the end of training and the start of testing, so a
6-month model's training targets cannot reach into its own test window. `wape_sd` (the
spread across those 3 folds) is reported next to every score: if a model beats another
by less than its own fold-to-fold spread, it has not actually beaten it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # write PNGs, never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# The lab reuses the project's metric definitions and its leakage assertion, so a number
# here means the same thing as a number from `uv run demandml train`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demandml import config as C  # noqa: E402
from demandml.metrics import critical_ratio, evaluate  # noqa: E402
from demandml.splits import Fold, assert_no_leakage  # noqa: E402

#: Every result, figure and report this folder writes is placed under LAB, i.e. INSIDE
#: the code tree. So the code tree must be writable; to run from a read-only install,
#: point the output directories somewhere writable (e.g. with symlinks).
LAB = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# Two parallel universes of output
# --------------------------------------------------------------------------------------
# The seven models exist in two versions: as they arrived from model_lab (`initial`,
# hand-set hyper-parameters) and after tune.py searched for better ones (`tuned`).
#
# Same code, same dataset, same folds, same features, same scoring -- the ONLY thing that
# differs between the two is the numbers handed to the estimator constructors. That is
# what makes the before/after comparison honest: any change in WAPE is attributable to
# the hyper-parameters and to nothing else.
#
# So each mode gets its own results dir and its own figures dir, and neither run can
# overwrite the other's numbers:
#
#     initial ->  results/        initial_figures/
#     tuned   ->  results_tuned/  tuned_figures/
#
MODES: dict[str, tuple[str, str]] = {
    "initial": ("results", "initial_figures"),
    "tuned": ("results_tuned", "tuned_figures"),
    # A third universe, for models rebuilt against the INVENTORY COST rather than WAPE
    # (see the cost-ratio section below). It is a separate pair of directories for the
    # same reason `tuned` is: a cost-trained model is a different model, and overwriting
    # the baseline's cached predictions with it would destroy the comparison that
    # justifies the change in the first place.
    "cost": ("results_cost", "cost_figures"),
}

#: Which universe the current process is writing into. Flipped by `--tuned` on any model
#: file, on compare_models.py or on dashboard.py -- never by editing this line.
#:
#: `DEMANDML_MODE` pins it for a whole process without a flag, which is what a scheduled
#: job needs: it must not depend on every entry point being invoked with the right
#: switch. It matters which half is used -- tuned parameters can improve WAPE and still
#: be WORSE on inventory cost (docs/FINDINGS.md), so a job that silently ran `tuned`
#: could buy against a half that was measured and rejected. `set_mode()` still wins over
#: the variable, so an explicit `--tuned` is never overridden by config.
MODE: str = os.environ.get("DEMANDML_MODE", "initial")
if MODE not in ("initial", "tuned", "cost"):
    raise SystemExit(f"DEMANDML_MODE={MODE!r} is not one of initial / tuned / cost")
#: The shortlisted models -- the whole board. By default every one of them runs; this is
#: the lab, and the losers stay here to be re-measured. `blend` (dev/blend.py) is the
#: eighth, added because it beat the single best model at 90 d on both precision and
#: cost.
SHORTLIST = ("moving_average", "croston", "random_forest", "xgboost_gbdt",
             "lightgbm_additive", "lgbm_quantile", "two_stage", "blend")


def active_models() -> tuple[str, ...] | None:
    """The models `DEMANDML_MODELS` (comma-separated) restricts this process to, or None
    when it is unset -- no restriction, the lab as it always was.

    A scheduled job can narrow the shortlist with it -- e.g. to the few models that
    actually get picked, plus `moving_average` kept as the floor -- for a fraction of the
    full board's fit time.

    It filters READS as well as runs, and that half is the point: a model left out of
    a refit still has its old `_preds/` and results JSON on disk, fitted on an older
    dataset and folds. Without the filter `forecast.choose` would let that stale backtest
    compete with the fresh ones and could pick it.
    """
    raw = os.environ.get("DEMANDML_MODELS", "").strip()
    if not raw:
        return None
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    bad = [n for n in names if n not in SHORTLIST]
    if bad or not names:
        raise SystemExit(f"DEMANDML_MODELS={raw!r}: {bad or 'empty'} not in the shortlist "
                         f"{', '.join(SHORTLIST)}")
    return names


def model_allowed(name: str) -> bool:
    """May this process run, rank or pick `name`? Always, unless DEMANDML_MODELS says."""
    keep = active_models()
    return keep is None or name in keep


RESULTS = LAB / "results"
#: Every PNG a run writes lands here -- the per-horizon figures run_model writes, the
#: comparison chart, and everything dashboard.py draws. Nothing else spells the directory
#: name out; every figure path is built from this constant, which is why the two modes
#: can share one code path.
FIGURES = LAB / "initial_figures"
#: What tune.py found: one JSON per model, best params per horizon + the search history.
TUNED_PARAMS = LAB / "tuned_params"
DATASET = C.PROCESSED / "lab_dataset.parquet"
DATASET_META = C.PROCESSED / "lab_dataset_meta.json"
#: Recent origins whose future has not happened yet, so they carry features but no
#: targets and are excluded from DATASET. They are what you forecast FROM: the newest
#: origin in DATASET is already a full horizon old by construction. Written by the same
#: pass of prepare_data.py, from the same expressions, so a served row and a trained row
#: cannot drift apart.
OPEN_ORIGINS = C.PROCESSED / "open_origins.parquet"

#: One "month" in days, and how many of them the longest horizon covers. The dataset
#: carries a `y_cum_m{k}` target for each (units sold in month k of the future), which
#: is what the additive models sum instead of predicting a long total in one shot.
MONTH_DAYS: int = 30
N_MONTHS: int = 6

#: Forecast horizons in days, and how to say them in English.
HORIZONS: tuple[int, ...] = (30, 90, 180)
HORIZON_LABEL = {30: "1 month ahead", 90: "3 months ahead", 180: "6 months ahead"}

#: Fixed colours so every figure in the folder reads the same way.
ACTUAL_C = "#4C6EF5"   # blue  = what really happened
PRED_C = "#F76707"     # orange = what the model said
GRID_C = "#D8DEE9"


# ======================================================================================
# The inventory cost ratio
# ======================================================================================
#
# Everything above this line measures how WRONG a forecast is. This measures how much
# being wrong COSTS, which is a different question and the one a purchasing decision
# actually asks.
#
# CU is what one unit of unmet demand costs; CO is what one unit of leftover stock costs
# over the horizon. Only the ratio matters, so CO is pinned at 1 and the preference is
# expressed entirely by CU.
#
# The ratio is a business input, not something the data can tell you. It is read from
# `assumptions.toml` (the shipped values are EXAMPLES for the sample data).
#
# CHANGING IT IS ONE NUMBER. Everything downstream -- the cost metric, the multiplier
# search, lgbm_quantile's target quantile -- reads it from here. `dev/cost_board.py`
# reports a few other ratios alongside the one in force so the choice stays visible and the board
# never depends on a number nobody can see.

from demandml import assumptions as _assumptions  # noqa: E402

CU: float = _assumptions.CU
CO: float = _assumptions.CO


def cost_ratio() -> tuple[float, float]:
    """The (cu, co) pair in force. Read this rather than the globals, so a caller that
    wants to sweep ratios has one place to override."""
    return CU, CO


def target_quantile() -> float:
    """The demand quantile that minimises cost at the ratio in force: CU/(CU+CO).

    This is the number to hand a quantile model. It is NOT a tuning knob -- it is
    determined by the cost ratio, so it must never appear in `tune.py`'s search space
    or the search will quietly re-optimise the stated cost preference away.
    """
    return critical_ratio(CU, CO)


# ======================================================================================
# Modes and tuned parameters
# ======================================================================================

def set_mode(mode: str) -> None:
    """Point RESULTS and FIGURES at one of the two universes described above.

    Rebinds the module globals rather than returning paths, because `run_model` and the
    figure helpers below read them at call time. Anything that imports them by name
    (compare_models.py, dashboard.py) must call `results_dir()` / `figures_dir()` after
    this instead of holding a stale `from common import RESULTS`.
    """
    global MODE, RESULTS, FIGURES
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
    res, figs = MODES[mode]
    MODE, RESULTS, FIGURES = mode, LAB / res, LAB / figs


def results_dir() -> Path:
    """The results directory for the mode currently in force."""
    return RESULTS


def figures_dir() -> Path:
    """The figures directory for the mode currently in force."""
    return FIGURES


# Apply whatever `DEMANDML_MODE` asked for. Setting the MODE global alone is not enough:
# RESULTS and FIGURES are separate globals that only `set_mode` rebinds, so without this
# a pinned mode would report itself correctly and still read and write the baseline
# directories -- the worst of both, since the logs would say `tuned` while the numbers
# came from `initial`.
set_mode(MODE)


def load_tuned(model_name: str) -> dict[str, Any] | None:
    """The tuned parameter file for one model, or None if it was never tuned.

    Shape: {"model": ..., "per_horizon": {"30": {"best_params": {...}, ...}, ...}}.
    """
    path = TUNED_PARAMS / f"{model_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def tuned_params_for(model_name: str, horizon: int) -> dict[str, Any]:
    """The winning hyper-parameters for one model at one horizon, or {} if there are none.

    Tuning is done PER HORIZON on purpose. A 1-month forecast and a 6-month forecast are
    different learning problems on this panel -- the short one is mostly "what is this
    SKU doing right now", the long one is mostly "what does this SKU do in a year" -- and
    the searches disagree about depth and learning rate accordingly. One shared parameter
    set would split the difference and be best at neither.
    """
    blob = load_tuned(model_name)
    if not blob:
        return {}
    entry = blob.get("per_horizon", {}).get(str(horizon))
    if not entry:
        return {}
    return dict(entry.get("best_params", {}))


# ======================================================================================
# Command line
# ======================================================================================

def parse_args(model_name: str, default_max_train_rows: int = 300_000) -> argparse.Namespace:
    """Every model file takes the same flags, so you can swap one for another blindly."""
    p = argparse.ArgumentParser(
        description=f"Backtest the {model_name} model on 1/3/6-month demand.",
    )
    p.add_argument("--folds", type=int, default=3,
                   help="number of rolling-origin splits to average over (default 3)")
    p.add_argument("--test-days", type=int, default=90,
                   help="length of each test window in days (default 90)")
    p.add_argument("--horizons", type=str, default=",".join(str(h) for h in HORIZONS),
                   help="comma-separated horizons in days (default 30,90,180)")
    p.add_argument("--sku-limit", type=int, default=None,
                   help="only the first N SKUs -- use this to get a quick answer")
    p.add_argument("--max-train-rows", type=int, default=default_max_train_rows,
                   help=f"cap on training rows per fold, 0 = no cap "
                        f"(default {default_max_train_rows:,} for this model)")
    p.add_argument("--no-save", action="store_true",
                   help="run and print, but do not write results/ or figures/")
    p.add_argument("--tuned", action="store_true",
                   help="use the hyper-parameters tune.py found (tuned_params/<model>.json) "
                        "and write to results_tuned/ + tuned_figures/ instead of "
                        "results/ + initial_figures/")
    p.add_argument("--cost", action="store_true",
                   help="build the model against the inventory cost ratio instead of "
                        "WAPE (e.g. aim a quantile model at the %.2f quantile rather "
                        "than the median) and write to results_cost/ + cost_figures/. "
                        "Only models that declare `cost_params` can honour this."
                        % target_quantile())
    args = p.parse_args()
    if args.tuned and args.cost:
        raise SystemExit(
            "--tuned and --cost each own a results directory, so they cannot be "
            "combined. Pick one.")
    set_mode("cost" if args.cost else "tuned" if args.tuned else "initial")
    return args


# ======================================================================================
# Data
# ======================================================================================

#: Extended features (`src/demandml/xfeatures.py`): in the dataset for every model, in
#: the feature list only for one that sets `LabModel.extended_features`.
XF_PREFIX = "xf_"


def model_features(model: "LabModel", feats: list[str]) -> list[str]:
    """The columns `model` trains and predicts on: `feats` minus the `xf_*` extended
    features unless the model asked for them. Every fit -- `run_model`'s folds and
    `forecast.py`'s serving fit -- goes through this, so the two cannot disagree."""
    if getattr(model, "extended_features", False):
        return feats
    return [c for c in feats if not c.startswith(XF_PREFIX)]


def load_dataset(sku_limit: int | None = None,
                 extended: bool = False) -> tuple[pl.DataFrame, list[str]]:
    """Load the prepared dataset and the list of columns that are safe to use.

    The feature list is read from the sidecar JSON written by prepare_data.py rather
    than re-derived here, so every model in the folder is guaranteed to see exactly the
    same columns. The `xf_*` extended features are left out unless `extended` -- so a
    probe or model written before they existed keeps the columns it was measured on.
    """
    if not DATASET.exists():
        raise SystemExit(
            f"{DATASET} does not exist.\n"
            f"Build it first:  uv run python model_lab/prepare_data.py"
        )
    meta = json.loads(DATASET_META.read_text())
    df = pl.read_parquet(DATASET)
    if sku_limit:
        keep = df.get_column(C.SKU).unique().sort().head(sku_limit).to_list()
        df = df.filter(pl.col(C.SKU).is_in(keep))
    feats = [c for c in meta["features"] if c in df.columns
             and (extended or not c.startswith(XF_PREFIX))]
    if pool_spec():
        df = df.with_columns(store_marker(current_store()))
        feats = feats + STORE_COLS
    print(f"  dataset: {len(df):,} origins x {len(feats)} features, "
          f"{df.get_column(C.SKU).n_unique():,} SKUs, "
          f"{df.get_column(C.DATE).min()} -> {df.get_column(C.DATE).max()}")
    return df, feats


# ======================================================================================
# Pooled training across stores
# ======================================================================================
#
# `DEMANDML_POOL` names every store's data directory (`us=/data,ca=/data/ca,...`, see
# `market.pool_env`). When it is set, every model this process fits -- each backtest fold
# in run_model and each horizon in forecast.py -- trains on the other stores' rows as well
# as its own, with one `store_<key>` column per store saying where a row came from. What is
# SCORED and FORECAST stays this store's own rows: the cached predictions, the model
# choice, the multiplier and every output downstream never see another store's SKU, which
# matters because stores can sell under the same SKU names.
#
# Why: pooled training beat one model per store in most folds (docs/FINDINGS.md). Unset, nothing
# here runs and every number is what it was.
#
# The marker is one-hot booleans rather than one string column so the sklearn models
# (`numeric_features` drops strings) can see it too. Each store's rows keep the holiday
# calendar its own build gave them (`config.MARKET_COUNTRY`).

sys.path.insert(0, str(Path(__file__).resolve().parent))
import market as _market  # noqa: E402 -- stdlib-only, lives next to this file
from lifted_log import lift_note, lift_stats  # noqa: E402

STORE_COLS = [f"store_{k}" for k in _market.MARKETS]


def pool_spec() -> dict[str, str]:
    return _market.parse_pool(os.environ.get("DEMANDML_POOL"))


def current_store() -> str:
    return _market.current().key


def store_marker(key: str) -> list[pl.Expr]:
    return [pl.lit(k == key).alias(f"store_{k}") for k in _market.MARKETS]


def load_pool(feats: list[str]) -> dict[str, pl.DataFrame]:
    """The OTHER stores' training rows -- `feats`, every `y_cum*` target (whichever one a
    model trains on), the weight, sku and date -- marked with their store. A store whose dataset is missing or unreadable
    is skipped with a warning rather than failing the run -- a store that has never been
    built must not stop this store's forecast -- and the result says which stores were used,
    so a run that pooled fewer stores than configured is visible in its output."""
    own = current_store()
    out: dict[str, pl.DataFrame] = {}
    for key, root in pool_spec().items():
        if key == own:
            continue
        path = Path(root) / "processed" / DATASET.name
        try:
            have = list(pl.read_parquet_schema(path))
            want = [c for c in feats if c not in STORE_COLS]
            cols = [c for c in have if c in set(want) or c.startswith("y_cum")
                    or c in ("sample_weight", C.SKU, C.DATE)]
            missing = [c for c in want if c not in set(have)]
            frame = pl.read_parquet(path, columns=cols).with_columns(store_marker(key))
        except Exception as e:  # noqa: BLE001 -- any unreadable store is skipped, loudly
            print(f"  ! pool: {key} skipped ({path}: {type(e).__name__}: {e})")
            continue
        if missing:
            print(f"  ! pool: {key} lacks {len(missing)} column(s), filled null: "
                  f"{missing[:6]}{' ...' if len(missing) > 6 else ''}")
        out[key] = frame
    if pool_spec():
        print(f"  pool: training with {', '.join(sorted(out)) or 'no other store'} "
              f"({sum(f.height for f in out.values()):,} rows) besides {own}")
    return out


def pool_training_rows(pool: dict[str, pl.DataFrame], h: int, train_end,
                       max_rows: int | None, seed: int = 0,
                       recent: bool = False) -> pl.DataFrame | None:
    """Every other store's rows with an `h`-day target and origin <= `train_end`, each
    store capped at `max_rows` on its own -- the cap the store's own rows get, so pooled
    and unpooled fits are comparable. The cap is a random sample, as `split` takes,
    or with `recent` the newest rows, as `forecast.py` takes. Callers pass a `train_end`
    whose windows have closed: a fold's train_end, or a forecast's as-of minus h."""
    ycol = f"y_cum{h}"
    parts = []
    for key, frame in pool.items():
        if ycol not in frame.columns:
            continue
        rows = frame.filter(pl.col(ycol).is_not_null() & (pl.col(C.DATE) <= train_end))
        if max_rows and rows.height > max_rows:
            rows = (rows.sort(C.DATE).tail(max_rows) if recent
                    else rows.sample(max_rows, seed=seed, shuffle=True))
        if rows.height:
            parts.append(rows)
    return pl.concat(parts, how="diagonal_relaxed") if parts else None


def with_pool(tr: pl.DataFrame, extra: pl.DataFrame | None) -> pl.DataFrame:
    """`tr` plus the pooled rows, in DATE order (the boosted models early-stop on the
    tail of the frame, which must be the latest dates, not the last store)."""
    if extra is None or not extra.height:
        return tr
    extra = extra.select([c for c in tr.columns if c in extra.columns])
    return pl.concat([tr, extra], how="diagonal_relaxed").sort(C.DATE)


#: Literal labels for the `meaningful` tag below -- shared so nothing has to
#: re-type (and risk drifting from) the string a filter compares against.
MEANINGFUL_LABEL = "1  meaningful (>=0.5/day, 90d+ history)"
THIN_LABEL = "2  thin / too new"


def add_top_meaningful_tags(df: pl.DataFrame) -> pl.DataFrame:
    """Two catalogue-scope cuts shared by every comparison in this folder.

    `vol_rank` / `top_n_band` rank SKUs by trailing 91-day volume *within each origin
    date* (so a SKU can move in or out of the top tier as it grows or fades --
    ranking once on a SKU's whole history would leak its own future busyness into
    every row).

    `meaningful` separates the catalogue that is actually worth carrying a forecast
    for (>=0.5 units/day trailing, 90+ days on market so the trailing window isn't
    mostly launch noise) from the long tail that is thin or too new to say much
    about. Same 0.5/day floor `segment_report.py`'s `volume_band` uses for its E cut.
    """
    df = df.with_columns(
        pl.col("units_mean_91").rank(method="ordinal", descending=True)
        .over(C.DATE).alias("vol_rank")
    )
    return df.with_columns(
        pl.when(pl.col("vol_rank") <= 10).then(pl.lit("1  top 10"))
        .when(pl.col("vol_rank") <= 50).then(pl.lit("2  top 11-50"))
        .when(pl.col("vol_rank") <= 200).then(pl.lit("3  top 51-200"))
        .otherwise(pl.lit("4  rest")).alias("top_n_band"),
        pl.when((pl.col("units_mean_91") >= 0.5) & (pl.col("days_on_market") >= 90))
        .then(pl.lit(MEANINGFUL_LABEL))
        .otherwise(pl.lit(THIN_LABEL)).alias("meaningful"),
    )


def numeric_features(df: pl.DataFrame, feats: list[str]) -> list[str]:
    """Only the numeric/boolean feature columns.

    scikit-learn models cannot take a raw string column (`sku_family`, `brand_name`).
    LightGBM and XGBoost can, via pandas `category`, so they use the full list and these
    models use this subset -- a real handicap for the linear models, and worth knowing
    about when you read their scores.
    """
    return [c for c in feats if df.schema[c].is_numeric() or df.schema[c] == pl.Boolean]


def to_pandas(X: pl.DataFrame):
    """Polars -> pandas with string columns as `category`, which is what LightGBM and
    XGBoost want. Both libraries reject pyarrow-backed strings outright."""
    import pandas as pd

    cats = [c for c, t in X.schema.items() if t in (pl.String, pl.Categorical, pl.Enum)]
    if cats:
        X = X.with_columns([pl.col(c).cast(pl.String).cast(pl.Categorical) for c in cats])
    out = X.to_pandas(use_pyarrow_extension_array=False)
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_bool_dtype(out[c].dtype):
            out[c] = out[c].astype("int8")
        elif not pd.api.types.is_numeric_dtype(out[c].dtype):
            out[c] = out[c].astype("category")
    return out


# ======================================================================================
# Splits
# ======================================================================================

def make_folds(df: pl.DataFrame, horizon: int, n_folds: int = 3,
               test_days: int = 90) -> list[Fold]:
    """Cut `n_folds` rolling-origin folds, newest last.

    Walking backwards from the most recent origin that has a complete target:

        [------------------ train ------------------][ gap ][-- test --]
                                                       ^
                                    `horizon` days wide, so that a training row's
                                    target window stops before the first test origin.
    """
    ycol = f"y_cum{horizon}"
    usable = df.filter(pl.col(ycol).is_not_null()).get_column(C.DATE)
    if usable.is_empty():
        raise SystemExit(f"no rows have a complete {ycol} target")
    first, last = usable.min(), usable.max()

    folds: list[Fold] = []
    for k in range(n_folds):
        test_end = last - timedelta(days=k * test_days)
        test_start = test_end - timedelta(days=test_days - 1)
        train_end = test_start - timedelta(days=horizon + 1)
        if train_end <= first + timedelta(days=365):
            print(f"  ! only {len(folds)} folds fit in the available history")
            break
        folds.append(Fold(index=0, train_start=first, train_end=train_end,
                          test_start=test_start, test_end=test_end))
    folds = sorted(folds, key=lambda f: f.test_start)
    return [Fold(i + 1, f.train_start, f.train_end, f.test_start, f.test_end)
            for i, f in enumerate(folds)]


def split(df: pl.DataFrame, fold: Fold, horizon: int,
          max_train_rows: int | None = None, seed: int = 0
          ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Slice one fold into (train, test) and assert the gap really holds."""
    ycol = f"y_cum{horizon}"
    d = pl.col(C.DATE)
    ok = pl.col(ycol).is_not_null()
    tr = df.filter(ok & (d >= fold.train_start) & (d <= fold.train_end))
    te = df.filter(ok & (d >= fold.test_start) & (d <= fold.test_end))
    if len(tr) and len(te):
        assert_no_leakage(tr, te, horizon)  # raises rather than quietly scoring a lie
    if max_train_rows and len(tr) > max_train_rows:
        # A RANDOM sample of rows *inside the training window*. Note this is not a random
        # split -- the window itself is still strictly before the test period. Sampling
        # beats keeping only the most recent N rows, which would silently throw away
        # every Christmas in the training data.
        tr = tr.sample(max_train_rows, seed=seed, shuffle=True)
    # Hand the training rows back in DATE order. Models that hold out a tail slice for
    # early stopping (the boosted ones) then really do get the latest part of the
    # training window. Without this the frame arrives sorted by SKU, and their "tail"
    # would be the last SKUs alphabetically -- inside the training window either way, so
    # never a leak, but a group holdout masquerading as a time holdout.
    return tr.sort(C.DATE), te


# ======================================================================================
# Scoring
# ======================================================================================

def score(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """The metric bundle. WAPE is the headline; read the other three with it.

    wape       sum|actual - predicted| / sum(actual). 0.30 reads "on average we are off
               by 30% of the volume we sell." Defined when actuals are zero, which MAPE
               is not -- a large share of SKU-days are zero, so MAPE is unusable.
    bias       signed version of the same thing. + = over-forecast (dead stock, cash
               tied up), - = under-forecast (stockouts, lost sales). A model can have
               great WAPE and be systematically wrong in one direction.
    shortfall  the units you would have failed to supply, as a share of demand, if you
               had ordered exactly the forecast. The inventory cost of under-forecasting.
    mae/rmse   average error in units. RMSE punishes big misses much harder, so
               RMSE >> MAE means the model occasionally blows up on a spike.
    cost       inventory cost per unit of demand at the CU:CO ratio above -- the only
               metric here that knows a stockout hurts more than dead stock. Rank on
               this for a purchasing decision; WAPE ranks the models differently and
               ranks them on the wrong thing. At CU == CO it IS WAPE.
    fill       1 - shortfall, i.e. the share of demand you would have met.
    """
    m = evaluate(y, yhat, cu=CU, co=CO)
    return {k: float(v) for k, v in m.items()}


# ======================================================================================
# Figures
# ======================================================================================

def _fig_for_horizon(model_name: str, horizon: int, per_fold: list[dict[str, Any]],
                     preds: pl.DataFrame, out: Path) -> Path:
    """One PNG per horizon: three simple panels answering three plain questions."""
    y = preds.get_column("actual").to_numpy()
    p = preds.get_column("predicted").to_numpy()
    wape = float(np.sum(np.abs(y - p)) / max(np.sum(np.abs(y)), 1e-9))
    bias = float(np.sum(p - y) / max(np.sum(np.abs(y)), 1e-9))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle(
        f"{model_name}  —  {HORIZON_LABEL.get(horizon, f'{horizon} days ahead')}:  "
        f"total units over the next {horizon} days",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # ---- Panel 1: did we land on the line? -------------------------------------------
    ax = axes[0]
    n = len(y)
    idx = np.random.default_rng(0).choice(n, size=min(n, 4000), replace=False)
    ax.scatter(y[idx], p[idx], s=8, alpha=0.25, color=ACTUAL_C, edgecolors="none")
    hi = max(float(np.percentile(y, 99.5)), float(np.percentile(p, 99.5)), 1.0)
    ax.plot([0, hi], [0, hi], color="#333", lw=1.2, ls="--", label="perfect forecast")
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_xlim(0, hi * 1.3); ax.set_ylim(0, hi * 1.3)
    ax.set_xlabel("actual units sold"); ax.set_ylabel("predicted units")
    ax.set_title("Every SKU-forecast, one dot\n(dots above the line = over-forecast)",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # ---- Panel 2: the average forecast against the average truth, over time ----------
    ax = axes[1]
    by_date = (preds.group_by(C.DATE)
                    .agg(pl.col("actual").mean(), pl.col("predicted").mean())
                    .sort(C.DATE))
    ax.plot(by_date.get_column(C.DATE), by_date.get_column("actual"),
            color=ACTUAL_C, lw=2.2, label="actual (average SKU)")
    ax.plot(by_date.get_column(C.DATE), by_date.get_column("predicted"),
            color=PRED_C, lw=2.2, ls="--", label="predicted (average SKU)")
    ax.set_title("Average forecast vs average actual\n(gap between the lines = bias)",
                 fontsize=10)
    ax.set_ylabel(f"units per SKU over {horizon} days")
    ax.set_xlabel("forecast origin date")
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    # ---- Panel 3: was the score stable across the 3 splits? --------------------------
    ax = axes[2]
    labels = [f"split {f['fold']}" for f in per_fold]
    vals = [f["wape"] for f in per_fold]
    ax.bar(labels, vals, color=ACTUAL_C, width=0.55)
    ax.axhline(float(np.mean(vals)), color=PRED_C, lw=2,
               label=f"average WAPE = {np.mean(vals):.3f}")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("Score on each of the 3 splits\n(big differences = unreliable model)",
                 fontsize=10)
    ax.legend(fontsize=9, frameon=False)

    for ax in axes:
        ax.grid(True, color=GRID_C, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.text(0.5, 0.005,
             f"averaged over {len(per_fold)} rolling-origin splits   |   "
             f"WAPE {wape:.3f}   bias {bias:+.3f}   {len(y):,} test forecasts",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ======================================================================================
# The runner
# ======================================================================================

@dataclass
class LabModel:
    """What a model file hands to `run_model`."""

    name: str                       #: filename stem, used for results/figures
    family: str                     #: "Baseline", "Linear / GLM", "Trees", ...
    blurb: str                      #: 1-3 sentences: what it does and how it is shaped
    pros: list[str]
    cons: list[str]
    #: build(horizon, **overrides) -> object with .fit / .predict. The keyword arguments
    #: are empty in `initial` mode and carry tune.py's winning parameters under --tuned,
    #: so every build lambda has to accept **kw even if it ignores it.
    build: Callable[..., Any]
    #: Set False for models that read one stored column and need no feature matrix.
    uses_features: bool = True
    #: Set True to receive the whole training frame as `fit(..., frame=tr)`. Only the
    #: additive models need it: they train one sub-model per future month and so have
    #: to reach the `y_cum_m{k}` columns, which are targets and therefore are not in
    #: the feature matrix. Everything else gets the plain three-argument fit.
    needs_frame: bool = False
    #: Per-model default for --max-train-rows. Slow learners (random forests) set this
    #: lower so that running the file plainly finishes in a sane amount of time.
    default_max_train_rows: int = 300_000
    #: How to rebuild this model against the inventory cost instead of WAPE, under
    #: `--cost`. A CALLABLE, not a dict, because it has to read `CU`/`CO` at run time --
    #: change the ratio and the model follows without editing anything here. Left None
    #: by models that have no honest way to express an asymmetric loss; `--cost` then
    #: refuses rather than silently running the WAPE model into a cost-labelled folder.
    cost_params: Callable[[], dict[str, Any]] | None = None
    #: Set True to also receive the `xf_*` extended features (`demandml/xfeatures.py`).
    #: Off for every model measured before they existed, so adding them changed nothing
    #: those models see; `dev/blend.py` is the one that asks.
    extended_features: bool = False
    #: Restrict the model to these horizons (None = every one). A model measured to win
    #: only at one horizon need not cost a retrain the others: with no `_preds` rows at a
    #: horizon, `forecast.choose` simply never sees it there.
    horizons: tuple[int, ...] | None = None
    notes: dict[str, Any] = field(default_factory=dict)


def run_model(model: LabModel) -> dict[str, Any]:
    """Backtest one model across every horizon, save figures + results, print a summary.

    This is deliberately the same 40 lines for every model in the folder. The only thing
    that differs between `random_forest.py` and `croston.py` is what `model.build(h)`
    returns.
    """
    args = parse_args(model.name, model.default_max_train_rows)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    if model.horizons is not None:
        horizons = [h for h in horizons if h in model.horizons]
    cap = args.max_train_rows or None

    # Hyper-parameter overrides, one set per horizon, empty in `initial` mode. Note the
    # model file itself is not edited by tuning: its PARAMS dict stays exactly as it was
    # when the shortlist was drawn, and the tuned numbers arrive here as keyword
    # arguments. That is what lets the same file produce both halves of the comparison.
    overrides: dict[int, dict[str, Any]] = {}
    if args.cost:
        if model.cost_params is None:
            raise SystemExit(
                f"--cost was passed but {model.name} declares no `cost_params`, so there "
                f"is no honest way to point it at the {target_quantile():.2f} quantile.\n"
                f"Running it unchanged would file a WAPE-trained model under "
                f"results_cost/ and make the comparison a lie.\n"
                f"Rank it on cost from its existing predictions instead:  "
                f"uv run python dev/cost_board.py")
        cp = model.cost_params()
        for h in horizons:
            overrides[h] = dict(cp)
        print(f"[cost] cu:co = {CU:g}:{CO:g}  ->  {cp}")
    if args.tuned:
        for h in horizons:
            overrides[h] = tuned_params_for(model.name, h)
        if not any(overrides.values()):
            raise SystemExit(
                f"--tuned was passed but {TUNED_PARAMS / (model.name + '.json')} has no "
                f"parameters for horizons {horizons}.\n"
                f"Run the search first:  uv run python dev/tune.py --model {model.name}"
            )

    print(f"\n{'=' * 86}\n{model.name}  ({model.family})   [{MODE}]\n{'=' * 86}")
    print(textwrap.fill(" ".join(model.blurb.split()), width=86))
    print(f"\n{'-' * 86}")
    df, feats = load_dataset(args.sku_limit, extended=model.extended_features)
    if not model.uses_features:
        feats = []
    # Other stores' rows, when DEMANDML_POOL is set. A model with no feature matrix
    # (moving_average, croston) is per-SKU by construction: pooling adds nothing to it.
    pool = load_pool(feats) if feats and pool_spec() else {}

    out: dict[str, Any] = {
        "model": model.name,
        "family": model.family,
        "blurb": " ".join(model.blurb.split()),
        "pros": model.pros,
        "cons": model.cons,
        "notes": model.notes,
        "run": {
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "folds": args.folds,
            "test_days": args.test_days,
            "sku_limit": args.sku_limit,
            "max_train_rows": cap,
            "n_features": len(feats),
            "n_skus": int(df.get_column(C.SKU).n_unique()),
            "mode": MODE,
            "pooled_with": sorted(pool),
        },
        "tuned_params": {str(h): v for h, v in overrides.items() if v},
        "horizons": {},
    }

    all_preds: list[pl.DataFrame] = []
    for h in horizons:
        ycol = f"y_cum{h}"
        if ycol not in df.columns:
            print(f"  ! {ycol} not in the dataset, skipping")
            continue
        folds = make_folds(df, h, n_folds=args.folds, test_days=args.test_days)
        print(f"\n[{HORIZON_LABEL.get(h, str(h) + 'd')}]  target = {ycol}")
        kw = overrides.get(h, {})
        if kw:
            print(f"  tuned: {', '.join(f'{k}={v}' for k, v in sorted(kw.items()))}")

        per_fold: list[dict[str, Any]] = []
        preds: list[pl.DataFrame] = []
        for fold in folds:
            tr, te = split(df, fold, h, max_train_rows=cap, seed=fold.index)
            if not len(tr) or not len(te):
                continue
            if pool:
                tr = with_pool(tr, pool_training_rows(pool, h, fold.train_end, cap,
                                                      seed=fold.index))
                assert_no_leakage(tr, te, h)
            y_tr = tr.get_column(ycol).to_numpy().astype(np.float64)
            y_te = te.get_column(ycol).to_numpy().astype(np.float64)
            # sample_weight is 0 on stock-out zero-days: the observed 0 is censored
            # demand, not real demand, so it must not pull the fit down.
            w_tr = tr.get_column("sample_weight").to_numpy().astype(np.float64)
            X_tr = tr.select(feats) if feats else tr
            X_te = te.select(feats) if feats else te

            t0 = time.time()
            est = model.build(h, **kw)
            if model.needs_frame:
                # The frame is the SAME rows as X_tr, in the same order -- it just still
                # has the target columns attached. Handing over training rows only; the
                # test frame is never passed, so this cannot become a leak.
                est.fit(X_tr, y_tr, w_tr, frame=tr)
            else:
                est.fit(X_tr, y_tr, w_tr)
            yhat = np.asarray(est.predict(X_te), dtype=np.float64).ravel()
            yhat = np.clip(np.nan_to_num(yhat), 0.0, None)  # demand is never negative
            secs = time.time() - t0
            lift = lift_stats(est)

            m = score(y_te, yhat)
            per_fold.append({"fold": fold.index, "seconds": round(secs, 1),
                             "train_rows": len(tr), "test_rows": len(te),
                             "train_end": str(fold.train_end),
                             "test_start": str(fold.test_start),
                             "test_end": str(fold.test_end), **m, **lift})
            preds.append(te.select(C.SKU, C.DATE).with_columns(
                pl.lit(h).alias("h"), pl.lit(fold.index).alias("fold"),
                pl.Series("actual", y_te), pl.Series("predicted", yhat)))
            print(f"  split {fold.index}: train -> {fold.train_end}  "
                  f"test {fold.test_start}..{fold.test_end}  "
                  f"wape={m['wape']:.4f}  bias={m['bias']:+.4f}  ({secs:.1f}s)"
                  f"{lift_note(lift)}")

        if not per_fold:
            continue
        agg = {k: float(np.mean([f[k] for f in per_fold]))
               for k in ("wape", "mae", "rmse", "bias", "shortfall", "smape",
                         "cost", "fill")}
        agg["wape_sd"] = float(np.std([f["wape"] for f in per_fold], ddof=1)) \
            if len(per_fold) > 1 else 0.0
        agg["seconds"] = float(np.mean([f["seconds"] for f in per_fold]))
        agg["folds"] = per_fold
        out["horizons"][str(h)] = agg
        print(f"  AVERAGE over {len(per_fold)} splits: "
              f"wape={agg['wape']:.4f} (+/-{agg['wape_sd']:.4f})  "
              f"bias={agg['bias']:+.4f}  shortfall={agg['shortfall']:.3f}  "
              f"cost={agg['cost']:.4f}  fill={agg['fill']:.3f}")

        if not args.no_save:
            allp = pl.concat(preds)
            fig = _fig_for_horizon(model.name, h, per_fold, allp,
                                   FIGURES / "all" / model.name / f"h{h:03d}.png")
            print(f"  figure -> {fig.relative_to(LAB.parent)}")
            all_preds.append(allp)

    if not args.no_save and out["horizons"]:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"{model.name}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"\nresults -> {path.relative_to(LAB.parent)}")
        if all_preds:
            # Raw per-row test predictions (SKU, date, horizon, fold, actual,
            # predicted), already fully computed above -- caching them costs one
            # parquet write and lets a later comparison slice by any SKU subset
            # (top sellers, the meaningful catalogue, ...) without refitting.
            preds_dir = RESULTS / "_preds"
            preds_dir.mkdir(parents=True, exist_ok=True)
            pl.concat(all_preds).write_parquet(
                preds_dir / f"{model.name}.parquet", compression="zstd")
            print(f"preds   -> {(preds_dir / f'{model.name}.parquet').relative_to(LAB.parent)}")
        print("compare every model you have run so far:  "
              "uv run python dev/compare_models.py")
    return out
