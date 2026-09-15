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

Features on a row only ever use data from `date` or earlier -- the timing convention
the whole project depends on.

-----------------------------------------------------------------------------------
WHY NOT A RANDOM SPLIT
-----------------------------------------------------------------------------------
The obvious request is "3 random splits, averaged". On this dataset a random split is
not just imperfect, it is actively misleading, so what you get instead is 3
*rolling-origin* splits averaged -- same idea (train on some rows, test on held-out rows,
repeat 3x, average), but the held-out rows are always in the FUTURE relative to the
training rows.

Why it matters: two rows for the same SKU one day apart are near-duplicates (their
28-day trailing averages overlap by 27 days out of 28). A random split puts one in
train and its twin in test, the model effectively memorises the answer, and the score
comes out far better than anything you would ever see in production. Every model here
would look brilliant and none of them would work.

The three folds are three consecutive slices of recent history:

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
from demandml.metrics import evaluate  # noqa: E402
from demandml.splits import Fold, assert_no_leakage  # noqa: E402

LAB = Path(__file__).resolve().parent
RESULTS = LAB / "results"
FIGURES = LAB / "figures"
DATASET = C.PROCESSED / "lab_dataset.parquet"
DATASET_META = C.PROCESSED / "lab_dataset_meta.json"
#: Recent origins whose future has not happened yet: features but no targets, so they are
#: excluded from DATASET and are the only rows a forward forecast can start from. Written
#: by the same pass of prepare_data.py, from the same expressions, so a served row and a
#: trained row cannot drift apart. Consumed by dev/forecast.py.
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
    return p.parse_args()


# ======================================================================================
# Data
# ======================================================================================

def load_dataset(sku_limit: int | None = None) -> tuple[pl.DataFrame, list[str]]:
    """Load the prepared dataset and the list of columns that are safe to use.

    The feature list is read from the sidecar JSON written by prepare_data.py rather
    than re-derived here, so every model in the folder is guaranteed to see exactly the
    same columns.
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
    feats = [c for c in meta["features"] if c in df.columns]
    print(f"  dataset: {len(df):,} origins x {len(feats)} features, "
          f"{df.get_column(C.SKU).n_unique():,} SKUs, "
          f"{df.get_column(C.DATE).min()} -> {df.get_column(C.DATE).max()}")
    return df, feats


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

    wape       sum|actual - predicted| / sum(actual). "On average we are off by this
               share of the volume we sell." Defined when actuals are zero, which MAPE
               is not -- a large share of SKU-days are zero, so MAPE is unusable.
    bias       signed version of the same thing. + = over-forecast (dead stock, cash
               tied up), - = under-forecast (stockouts, lost sales). A model can have
               great WAPE and be systematically wrong in one direction.
    shortfall  the units you would have failed to supply, as a share of demand, if you
               had ordered exactly the forecast. The inventory cost of under-forecasting.
    mae/rmse   average error in units. RMSE punishes big misses much harder, so
               RMSE >> MAE means the model occasionally blows up on a spike.
    """
    m = evaluate(y, yhat)
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
    build: Callable[[int], Any]     #: build(horizon) -> object with .fit / .predict
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
    notes: dict[str, Any] = field(default_factory=dict)


def run_model(model: LabModel) -> dict[str, Any]:
    """Backtest one model across every horizon, save figures + results, print a summary.

    This is deliberately the same 40 lines for every model in the folder. The only thing
    that differs between `random_forest.py` and `croston.py` is what `model.build(h)`
    returns.
    """
    args = parse_args(model.name, model.default_max_train_rows)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    cap = args.max_train_rows or None

    print(f"\n{'=' * 86}\n{model.name}  ({model.family})\n{'=' * 86}")
    print(textwrap.fill(" ".join(model.blurb.split()), width=86))
    print(f"\n{'-' * 86}")
    df, feats = load_dataset(args.sku_limit)
    if not model.uses_features:
        feats = []

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
        },
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

        per_fold: list[dict[str, Any]] = []
        preds: list[pl.DataFrame] = []
        for fold in folds:
            tr, te = split(df, fold, h, max_train_rows=cap, seed=fold.index)
            if not len(tr) or not len(te):
                continue
            y_tr = tr.get_column(ycol).to_numpy().astype(np.float64)
            y_te = te.get_column(ycol).to_numpy().astype(np.float64)
            # sample_weight is 0 on stock-out zero-days: the observed 0 is censored
            # demand, not real demand, so it must not pull the fit down.
            w_tr = tr.get_column("sample_weight").to_numpy().astype(np.float64)
            X_tr = tr.select(feats) if feats else tr
            X_te = te.select(feats) if feats else te

            t0 = time.time()
            est = model.build(h)
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

            m = score(y_te, yhat)
            per_fold.append({"fold": fold.index, "seconds": round(secs, 1),
                             "train_rows": len(tr), "test_rows": len(te),
                             "train_end": str(fold.train_end),
                             "test_start": str(fold.test_start),
                             "test_end": str(fold.test_end), **m})
            preds.append(te.select(C.SKU, C.DATE).with_columns(
                pl.lit(h).alias("h"), pl.lit(fold.index).alias("fold"),
                pl.Series("actual", y_te), pl.Series("predicted", yhat)))
            print(f"  split {fold.index}: train -> {fold.train_end}  "
                  f"test {fold.test_start}..{fold.test_end}  "
                  f"wape={m['wape']:.4f}  bias={m['bias']:+.4f}  ({secs:.1f}s)")

        if not per_fold:
            continue
        agg = {k: float(np.mean([f[k] for f in per_fold]))
               for k in ("wape", "mae", "rmse", "bias", "shortfall", "smape")}
        agg["wape_sd"] = float(np.std([f["wape"] for f in per_fold], ddof=1)) \
            if len(per_fold) > 1 else 0.0
        agg["seconds"] = float(np.mean([f["seconds"] for f in per_fold]))
        agg["folds"] = per_fold
        out["horizons"][str(h)] = agg
        print(f"  AVERAGE over {len(per_fold)} splits: "
              f"wape={agg['wape']:.4f} (+/-{agg['wape_sd']:.4f})  "
              f"bias={agg['bias']:+.4f}  shortfall={agg['shortfall']:.3f}")

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
              "uv run python model_lab/compare_models.py")
    return out
