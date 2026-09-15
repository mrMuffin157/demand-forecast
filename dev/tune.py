"""TUNE — search for better hyper-parameters for the seven models, honestly.

    uv run python dev/tune.py --model xgboost_gbdt
    uv run python dev/tune.py --all
    uv run python dev/tune.py --all --budget 0.5     # half the trials, for a quick pass

Writes `dev/tuned_params/<model>.json`. Nothing else in the folder changes: the model
files keep the PARAMS dicts they were written with, and the tuned numbers only ever
arrive as keyword overrides when a model is run with `--tuned`. That is deliberate —
`results/` + `initial_figures/` stay reproducible from the files as written, so the
before/after comparison always has a real "before" to point at.

=======================================================================================
THE ONLY THING THAT MATTERS IN THIS FILE
=======================================================================================
A hyper-parameter search is a way of fitting to whatever data you score it on. If the
search scores candidates on the same rows the leaderboard is computed from, the tuned
numbers are not estimates of anything — you have selected the parameters that happen to
suit those 90 days, and the improvement you report is partly just that selection.

So the search never sees a single test row. Concretely, for each horizon:

    1. Ask common.make_folds() where the OUTER (leaderboard) folds are.
    2. Take the earliest one's `train_end` and throw away every origin after it.
       That date is, by construction, before every outer test window AND before every
       outer gap, so the surviving rows are a subset of every outer fold's own
       training data.
    3. Cut fresh rolling-origin folds inside that surviving slice, and search there.

        |<---------------- tuning universe ---------------->|
        |                          [inner train][gap][inner test]
        |                                                   |
        +---------------------------------------------------+----- cut here
                                                            |
                                  outer fold 1: [ train ][gap][ TEST ]
                                  outer fold 2:      [  train  ][gap][ TEST ]
                                  outer fold 3:           [   train   ][gap][ TEST ]

    The tuning universe ends where outer fold 1's training data ends. Every inner test
    row is therefore a row that all three outer folds are allowed to train on, and no
    inner row is ever an outer test row.

The cost of this honesty is that the search is scored on slightly older data than the
leaderboard, which is the correct trade: the alternative is a number that flatters
itself. It also means a tuned model can come out WORSE on the board than the initial
one. When that happens it is information — it says the hand-set parameters were already
at or past the point where this dataset can tell two settings apart — and
compare_tuning.py reports it rather than hiding it.

=======================================================================================
HOW IT SEARCHES
=======================================================================================
Random search over the spaces at the bottom of this file, not grid search. With 6-9
knobs a grid is astronomically large and spends most of its budget re-measuring the
knobs that do not matter; random search over the same budget samples every knob at a
different value on every trial, which is strictly better when — as here — only two or
three of them actually move the score. The cheap models (moving_average, croston) have
small enough spaces to enumerate exhaustively, and do.

Per horizon, not once per model. A 1-month forecast and a 6-month forecast are different
problems on this panel: the short one is mostly "what is this SKU doing right now" and
the long one is mostly "what does this SKU do in a year". The searches pick different
depths and learning rates accordingly, and one shared setting would be best at neither.

Selection rule: mean WAPE across the inner folds, with the *initial* parameters entered
as trial 0. If nothing beats trial 0 the initial parameters win and are written out
unchanged, so `--tuned` is never worse than `--initial` by construction of the search
(it can still be worse on the board — see above).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as K  # noqa: E402
from common import (  # noqa: E402
    HORIZON_LABEL,
    HORIZONS,
    TUNED_PARAMS,
    load_dataset,
    make_folds,
    score,
    split,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demandml import config as C  # noqa: E402
from demandml.metrics import cost_from_wape_bias  # noqa: E402


# ======================================================================================
# The search spaces
# ======================================================================================
# Each entry is {parameter: list-of-values}. A trial picks one value per parameter,
# independently and uniformly. Values are chosen to bracket the file's default rather
# than to wander: the point is to find out whether the default is at an optimum, and a
# space that does not contain the default's neighbourhood cannot answer that.

Space = dict[str, list[Any]]

SPACES: dict[str, Space] = {
    # --- no learning, so the space is tiny and enumerated exhaustively ----------------
    "moving_average": {
        # Which trailing window to average. The file argues for matching the lookback to
        # the lookahead (28/91/182); this asks the data whether that argument is right.
        "column": ["units_mean_7", "units_mean_14", "units_mean_28", "units_mean_56",
                   "units_mean_91", "units_mean_182", "units_mean_364"],
        # A flat multiplier. A trailing mean over a panel whose stock-out days are
        # censored is a biased estimate of demand; this measures the bias.
        "scale": [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20],
    },
    "croston": {
        # The smoothing rate, which also sets the SBA de-bias factor (1 - alpha/2), so
        # this one knob moves both responsiveness and level.
        "alpha": [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
        "size_col": ["nonzero_mean_91", "nonzero_mean_364"],
        "interval_col": ["adi_91", "adi_364"],
    },

    # --- the learners -----------------------------------------------------------------
    "random_forest": {
        "n_estimators": [200, 300, 500],
        # None = grow until the leaves are pure. Included because min_samples_leaf is
        # the more direct way to control the same thing, and the two interact.
        "max_depth": [10, 14, 20, 28, None],
        "min_samples_leaf": [5, 10, 20, 40, 80],
        "max_features": ["sqrt", 0.2, 0.35, 0.5],
    },
    "xgboost_gbdt": {
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "max_depth": [6, 8, 10, 12],
        "min_child_weight": [5.0, 10.0, 20.0, 50.0],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_lambda": [0.5, 1.0, 5.0, 20.0],
        # 1.0 is Poisson, 2.0 is Gamma. Between them is the compound Poisson-Gamma that
        # intermittent demand actually looks like; where between them is an empirical
        # question and this is the knob that answers it.
        "tweedie_variance_power": [1.1, 1.2, 1.3, 1.5],
        "n_estimators": [800, 1500],
    },
    "lightgbm_additive": {
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [20, 50, 100, 200],
        "feature_fraction": [0.5, 0.7, 0.9],
        "bagging_fraction": [0.7, 0.8, 0.9],
        "lambda_l2": [0.5, 1.0, 5.0, 20.0],
        "tweedie_variance_power": [1.1, 1.2, 1.3, 1.5],
        "n_estimators": [800, 1500],
    },
    "lgbm_quantile": {
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [20, 50, 100, 200],
        "feature_fraction": [0.5, 0.7, 0.9],
        "lambda_l2": [0.5, 1.0, 5.0, 20.0],
        "n_estimators": [700, 1200],
        # NOT SEARCHED ANY MORE. `center` is the quantile the point forecast is read
        # off, and a WAPE-driven search does not treat it as a free parameter with an
        # empirical optimum -- it treats it as a free way to forecast low. Left in the
        # space, it gets pulled below the median at the long horizons, i.e. the search
        # decides on the operator's behalf that a stockout is cheap -- the opposite of
        # what any CU > CO says.
        #
        # The quantile is DERIVED from the cost ratio -- CU/(CU+CO), which is
        # `common.target_quantile()` -- and reached via `--cost`, not via tuning. Putting
        # it back in this grid would let the search overwrite the one number in this
        # project that encodes the configured cost asymmetry.
    },
    "two_stage": {
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [20, 50, 100, 200],
        "feature_fraction": [0.5, 0.7, 0.9],
        "lambda_l2": [0.5, 1.0, 5.0, 20.0],
        "clf_n_estimators": [300, 600, 1000],
        "reg_n_estimators": [400, 800, 1200],
        # Stage 2 fits "how many units, given at least one". L1 fits its median, L2 its
        # mean, Tweedie its actual shape.
        "reg_objective": ["regression_l1", "regression", "tweedie"],
    },
}

#: How many random trials each model gets, before --budget scales it. The cheap models
#: enumerate their whole space instead and ignore this. The expensive ones are sized so
#: that one model's full search (3 horizons x trials x inner folds) finishes in a
#: tolerable time on a laptop; random_forest is lowest because a forest fit is by far
#: the slowest thing in the folder.
TRIALS: dict[str, int] = {
    # A forest fit at this width is minutes, and a forest is the model least sensitive
    # to its settings, so spending the budget here buys the least.
    "random_forest": 8,
    "xgboost_gbdt": 16,
    # One trial here is n_months separate fits -- six of them at the 6-month horizon --
    # so its per-trial cost is roughly 3.5x a single boosted model's.
    "lightgbm_additive": 8,
    "lgbm_quantile": 12,
    "two_stage": 12,
}

#: Training rows per inner fold during the SEARCH only. Lower than the real run's cap on
#: purpose — the search has to rank candidates, not produce a final number, and ranking
#: survives subsampling far better than the absolute score does. The winner is then
#: re-run at full size by the model file itself.
SEARCH_TRAIN_ROWS: dict[str, int] = {
    # A forest fit is the slowest thing in the folder and the least sensitive to its
    # settings, so it gets the smallest sample.
    "random_forest": 35_000,
    "xgboost_gbdt": 70_000,
    # These three fit several sub-models per trial (6 months / 3 quantiles / 2 stages),
    # so their per-trial cost is a multiple of a single boosted fit.
    "lightgbm_additive": 40_000,
    "lgbm_quantile": 60_000,
    "two_stage": 60_000,
}
DEFAULT_SEARCH_TRAIN_ROWS = 70_000

#: The models whose whole space is small enough to enumerate.
EXHAUSTIVE = {"moving_average", "croston"}


# ======================================================================================
# Loading the model under test
# ======================================================================================

def load_model(name: str):
    """Import `dev/<name>.py` and hand back its MODEL, without running its __main__."""
    import importlib

    mod = importlib.import_module(name)
    if not hasattr(mod, "MODEL"):
        raise SystemExit(f"dev/{name}.py has no MODEL")
    return mod.MODEL


# ======================================================================================
# The tuning universe
# ======================================================================================

def tuning_universe(df: pl.DataFrame, horizon: int, outer_folds: int,
                    outer_test_days: int) -> tuple[pl.DataFrame, Any]:
    """Every origin the search is allowed to touch, and the date it was cut at.

    See the header. The cut is the earliest OUTER fold's `train_end`, so the universe is
    a subset of every outer fold's training data and contains no outer test row.
    """
    folds = make_folds(df, horizon, n_folds=outer_folds, test_days=outer_test_days)
    cut = min(f.train_end for f in folds)
    sub = df.filter(pl.col(C.DATE) <= cut)
    return sub, cut


def inner_folds(df_tune: pl.DataFrame, horizon: int, n_folds: int, test_days: int):
    """Rolling-origin folds cut inside the tuning universe, newest last."""
    return make_folds(df_tune, horizon, n_folds=n_folds, test_days=test_days)


# ======================================================================================
# One trial
# ======================================================================================

def evaluate_params(model, horizon: int, params: dict[str, Any],
                    df_tune: pl.DataFrame, feats: list[str], folds,
                    max_train_rows: int) -> dict[str, Any]:
    """Fit `model` with `params` on each inner fold and return the mean inner WAPE."""
    ycol = f"y_cum{horizon}"
    wapes, biases, secs = [], [], 0.0
    for fold in folds:
        tr, te = split(df_tune, fold, horizon, max_train_rows=max_train_rows,
                       seed=fold.index)
        if not len(tr) or not len(te):
            continue
        y_tr = tr.get_column(ycol).to_numpy().astype(np.float64)
        y_te = te.get_column(ycol).to_numpy().astype(np.float64)
        w_tr = tr.get_column("sample_weight").to_numpy().astype(np.float64)
        X_tr = tr.select(feats) if feats else tr
        X_te = te.select(feats) if feats else te

        t0 = time.time()
        est = model.build(horizon, **params)
        if model.needs_frame:
            est.fit(X_tr, y_tr, w_tr, frame=tr)
        else:
            est.fit(X_tr, y_tr, w_tr)
        yhat = np.clip(np.nan_to_num(
            np.asarray(est.predict(X_te), dtype=np.float64).ravel()), 0.0, None)
        secs += time.time() - t0
        m = score(y_te, yhat)
        wapes.append(m["wape"])
        biases.append(m["bias"])
    if not wapes:
        return {"wape": float("inf"), "bias": 0.0, "seconds": secs, "folds": 0}
    return {"wape": float(np.mean(wapes)), "bias": float(np.mean(biases)),
            "wape_sd": float(np.std(wapes, ddof=1)) if len(wapes) > 1 else 0.0,
            "seconds": round(secs, 1), "folds": len(wapes)}


def select(rows: list[dict[str, Any]], bias_aware: bool = True,
           on: str = "cost") -> dict[str, Any]:
    """Pick the winner from a horizon's trials.

    The obvious rule -- lowest inner WAPE -- is the one that got us into trouble. WAPE is
    a sum of absolute errors, so it is minimised by the conditional MEDIAN, which on this
    right-skewed panel sits below the conditional mean. In practice the multiplier that
    minimises WAPE sits below the multiplier that zeroes bias in nearly every
    model-horizon (compare_tuning.py draws it). So a pure-WAPE search does not merely
    tolerate a downward drift, it actively selects for one, and models come out of it
    under-forecasting more than they went in.

    What makes that worth fixing cheaply is the other half of the same measurement:
    forcing zero bias typically costs only a small fraction of the fold-to-fold noise in
    WAPE. The WAPE surface is nearly flat with respect to forecast level. So the honest
    setting and the accurate setting are almost the same setting -- you just have to ask
    for it.

    Hence the rule: take every candidate whose WAPE is within one fold-to-fold standard
    deviation of the best candidate's -- i.e. every setting that is not distinguishably
    worse -- and among those, take the one with the smallest |bias|. When the ranking is
    unambiguous this changes nothing; when several settings are tied on accuracy it
    breaks the tie toward the forecast that is not systematically short.

    `bias_aware=False` restores the old lowest-WAPE-wins rule, which is what
    `best_params_wape_only` in the output records for comparison.

    `on="cost"` (the default) replaces the whole
    tie-break heuristic with the thing it was approximating. The heuristic existed
    because WAPE could not express "being short is worse"; now it can, and
    `cost_from_wape_bias` recovers the cost EXACTLY from the two numbers every trial
    already stores -- so scoring a search on cost needs no predictions, no refitting and
    no change to how searches are run.

    Read the caveat in `reselect()` before trusting a cost re-selection of an OLD search:
    those pools were pruned to the best dozen trials BY WAPE, so a setting with mediocre
    WAPE and excellent calibration was discarded before it could be considered.
    """
    live = [r for r in rows if np.isfinite(r["wape"])]
    if not live:
        return rows[0]
    if on == "cost":
        cu, co = K.cost_ratio()
        return min(live, key=lambda r: cost_from_wape_bias(r["wape"], r["bias"], cu, co))
    best = min(live, key=lambda r: r["wape"])
    if not bias_aware:
        return best
    # The spread of the best candidate across its own inner folds. With only 2 folds this
    # is a crude estimate, so it is used as a tie-break width, never as a significance
    # test.
    sd = float(best.get("wape_sd") or 0.0)
    pool = [r for r in live if r["wape"] <= best["wape"] + sd]
    return min(pool, key=lambda r: abs(r["bias"]))


def candidates(name: str, space: Space, n_trials: int, rng: random.Random
               ) -> list[dict[str, Any]]:
    """The parameter sets to try, trial 0 always being "whatever the file already says".

    Trial 0 is the empty dict — no overrides — which makes the model build with its own
    PARAMS. Entering the incumbent in its own search is what lets the result be read as
    "the default was/was not already the best thing in this space".
    """
    out: list[dict[str, Any]] = [{}]
    if name in EXHAUSTIVE:
        import itertools
        keys = list(space)
        for combo in itertools.product(*(space[k] for k in keys)):
            out.append(dict(zip(keys, combo)))
        return out
    seen = {()}
    tries = 0
    while len(out) <= n_trials and tries < n_trials * 50:
        tries += 1
        pick = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(sorted((k, str(v)) for k, v in pick.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(pick)
    return out


# ======================================================================================
# One model
# ======================================================================================

def tune_model(name: str, df: pl.DataFrame, feats: list[str], args) -> dict[str, Any]:
    model = load_model(name)
    space = SPACES.get(name)
    if not space:
        raise SystemExit(f"no search space defined for {name}")
    if not model.uses_features:
        feats = []

    n_trials = max(1, int(round(TRIALS.get(name, 12) * args.budget)))
    cap = SEARCH_TRAIN_ROWS.get(name, DEFAULT_SEARCH_TRAIN_ROWS)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    print(f"\n{'=' * 86}\nTUNING  {name}   ({model.family})\n{'=' * 86}")
    print(f"  space: {len(space)} knobs, "
          f"{'exhaustive' if name in EXHAUSTIVE else f'{n_trials} random trials'} "
          f"per horizon, {args.inner_folds} inner folds, <={cap:,} train rows/fold")

    out: dict[str, Any] = {
        "model": name,
        "family": model.family,
        "search": {
            "trials": n_trials, "exhaustive": name in EXHAUSTIVE,
            "inner_folds": args.inner_folds, "inner_test_days": args.inner_test_days,
            "search_max_train_rows": cap, "seed": args.seed,
            "space": {k: [str(x) for x in v] for k, v in space.items()},
        },
        "per_horizon": {},
    }

    for h in horizons:
        ycol = f"y_cum{h}"
        if ycol not in df.columns:
            continue
        df_tune, cut = tuning_universe(df, h, args.outer_folds, args.outer_test_days)
        folds = inner_folds(df_tune, h, args.inner_folds, args.inner_test_days)
        if not folds:
            print(f"  ! {h}d: no inner folds fit before {cut}, skipping")
            continue

        rng = random.Random(args.seed + h)
        trials = candidates(name, space, n_trials, rng)
        print(f"\n  [{HORIZON_LABEL.get(h, str(h) + 'd')}]  "
              f"universe ends {cut} ({len(df_tune):,} origins), "
              f"{len(folds)} inner folds "
              f"{folds[0].test_start}..{folds[-1].test_end}, {len(trials)} trials")

        rows: list[dict[str, Any]] = []
        t_start = time.time()
        for i, params in enumerate(trials):
            res = evaluate_params(model, h, params, df_tune, feats, folds, cap)
            rows.append({"trial": i, "params": {k: v for k, v in params.items()},
                         **{k: v for k, v in res.items() if k != "folds"}})
            tag = "initial" if i == 0 else ""
            low = min((r["wape"] for r in rows if np.isfinite(r["wape"])), default=np.inf)
            mark = "  <-- best wape" if res["wape"] <= low else ""
            print(f"    trial {i:>3} {tag:<7} wape={res['wape']:.4f} "
                  f"bias={res['bias']:+.4f} ({res['seconds']:.0f}s){mark}")

        # Two winners are recorded: the one the selection rule actually chose, and the
        # one a pure-WAPE search would have chosen. Keeping both is what makes the cost
        # of the bias constraint auditable instead of asserted -- see select().
        best = select(rows, bias_aware=not args.wape_only)
        wape_best = select(rows, bias_aware=False)
        base = rows[0]["wape"]
        delta = (base - best["wape"]) / base * 100 if np.isfinite(base) and base else 0.0
        out["per_horizon"][str(h)] = {
            "best_params": best["params"],
            "inner_wape": best["wape"],
            "inner_bias": best["bias"],
            "inner_wape_initial": base,
            "inner_bias_initial": rows[0]["bias"],
            "inner_improvement_pct": delta,
            "best_trial": best["trial"],
            "selection": "wape_only" if args.wape_only else "bias_aware",
            "best_params_wape_only": wape_best["params"],
            "inner_wape_wape_only": wape_best["wape"],
            "inner_bias_wape_only": wape_best["bias"],
            "bias_constraint_cost_wape": best["wape"] - wape_best["wape"],
            "universe_end": str(cut),
            "inner_fold_windows": [f"{f.test_start}..{f.test_end}" for f in folds],
            "n_trials": len(rows),
            "search_seconds": round(time.time() - t_start, 1),
            "trials": sorted(rows, key=lambda r: r["wape"])[:12],
        }
        kept = "kept the initial parameters" if best["trial"] == 0 else \
               ", ".join(f"{k}={v}" for k, v in sorted(best["params"].items()))
        print(f"    => inner wape {base:.4f} -> {best['wape']:.4f} ({delta:+.1f}%)  "
              f"bias {rows[0]['bias']:+.3f} -> {best['bias']:+.3f}   {kept}")
        if best["trial"] != wape_best["trial"]:
            print(f"       (a wape-only rule would have taken trial "
                  f"{wape_best['trial']}: wape {wape_best['wape']:.4f} "
                  f"bias {wape_best['bias']:+.3f} — "
                  f"{best['wape'] - wape_best['wape']:+.4f} wape for "
                  f"{abs(wape_best['bias']) - abs(best['bias']):.3f} less |bias|)")

    if not args.no_save and out["per_horizon"]:
        TUNED_PARAMS.mkdir(parents=True, exist_ok=True)
        path = TUNED_PARAMS / f"{name}.json"
        path.write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  tuned params -> {path.relative_to(K.LAB.parent)}")
    return out


# ======================================================================================
# Entry point
# ======================================================================================

def reselect(names: list[str], args) -> None:
    """Re-pick every winner from the trials already stored, fitting nothing.

    Each search records its candidates with both `wape` and `bias`, so changing the
    SELECTION rule does not require re-running the SEARCH -- which matters, because the
    searches cost hours and this costs milliseconds.

    The honest limitation, stated here because the output does not show it: a search only
    keeps its best dozen trials by WAPE, so the pool this re-picks from is already
    filtered toward low WAPE. A setting with mediocre WAPE and excellent bias was thrown
    away before it got here. So this answers "which of the settings that are tied on
    accuracy is the least biased", which is the right question, but it is not the same as
    having searched under the constraint from the start. For that, re-run the search --
    the new rule is the default there too.
    """
    for name in names:
        path = TUNED_PARAMS / f"{name}.json"
        if not path.exists():
            print(f"  ! {name}: no {path.name} to re-select from, skipping")
            continue
        blob = json.loads(path.read_text())
        print(f"\n{name}")
        for h, entry in blob.get("per_horizon", {}).items():
            rows = entry.get("trials") or []
            if not rows:
                print(f"  {h:>4}d: no stored trials, left as it was")
                continue
            best = select(rows, bias_aware=not args.wape_only, on=args.select_on)
            wape_best = select(rows, bias_aware=False, on="wape")
            base = entry.get("inner_wape_initial", float("nan"))
            entry.update({
                "best_params": best["params"],
                "inner_wape": best["wape"],
                "inner_bias": best["bias"],
                "best_trial": best["trial"],
                "selection": "wape_only" if args.wape_only else "bias_aware",
                "reselected_from_stored_trials": True,
                "best_params_wape_only": wape_best["params"],
                "inner_wape_wape_only": wape_best["wape"],
                "inner_bias_wape_only": wape_best["bias"],
                "bias_constraint_cost_wape": best["wape"] - wape_best["wape"],
                "inner_improvement_pct": ((base - best["wape"]) / base * 100
                                          if np.isfinite(base) and base else 0.0),
            })
            same = "unchanged" if best["trial"] == wape_best["trial"] else (
                f"was trial {wape_best['trial']} "
                f"(wape {wape_best['wape']:.4f} bias {wape_best['bias']:+.3f})")
            print(f"  {h:>4}d: trial {best['trial']:>3}  "
                  f"wape {best['wape']:.4f}  bias {best['bias']:+.3f}   {same}")
        if not args.no_save:
            path.write_text(json.dumps(blob, indent=2, default=str))
    print("\nRe-selected. Nothing was re-fitted, so the numbers above are the same inner"
          "\nmeasurements as before -- only the choice among them changed. To see what it"
          "\ndoes on the test folds:")
    print("  uv run python dev/compare_models.py --run-all --tuned")
    print("  uv run python dev/compare_tuning.py")


#: Cheap first, so a full run produces something useful within a minute and the slow
#: boosted searches run while you are already reading numbers.
ORDER = ["moving_average", "croston", "lgbm_quantile", "two_stage",
         "lightgbm_additive", "xgboost_gbdt", "random_forest"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", action="append", default=None,
                   help="model to tune; repeatable. Default: all seven.")
    p.add_argument("--all", action="store_true", help="tune every model (the default)")
    p.add_argument("--horizons", type=str, default=",".join(str(h) for h in HORIZONS))
    p.add_argument("--budget", type=float, default=1.0,
                   help="scale every model's trial count (0.5 = half, for a quick pass)")
    p.add_argument("--inner-folds", type=int, default=2,
                   help="rolling-origin folds cut INSIDE the tuning universe (default 2)")
    p.add_argument("--inner-test-days", type=int, default=90)
    p.add_argument("--outer-folds", type=int, default=3,
                   help="must match what the leaderboard run uses, or the cut is wrong")
    p.add_argument("--outer-test-days", type=int, default=90)
    p.add_argument("--sku-limit", type=int, default=None,
                   help="only the first N SKUs -- for a smoke test of this file")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--wape-only", action="store_true",
                   help="select purely on inner WAPE, the way the first run did. Kept so "
                        "the cost of the bias constraint can be reproduced, not because "
                        "it is a good idea -- see select().")
    p.add_argument("--select-on", choices=("cost", "wape"), default="cost",
                   help="what the winner is chosen on. 'cost' uses the inventory cost at "
                        "common.CU:CO, recovered exactly from each trial's stored wape "
                        "and bias. 'wape' keeps the old accuracy-plus-bias-tiebreak rule.")
    p.add_argument("--reselect", action="store_true",
                   help="re-pick the winners from the trials already in tuned_params/ "
                        "under the current selection rule, without re-running any "
                        "search. Seconds instead of hours.")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    names = args.model or ORDER
    unknown = [n for n in names if n not in SPACES]
    if unknown:
        raise SystemExit(f"no search space for: {', '.join(unknown)}")

    if args.reselect:
        reselect(names, args)
        return

    df, feats = load_dataset(args.sku_limit)
    t0 = time.time()
    summary: list[tuple[str, str, float, float]] = []
    for name in names:
        res = tune_model(name, df, feats, args)
        for h, entry in res["per_horizon"].items():
            summary.append((name, h, entry["inner_wape_initial"], entry["inner_wape"]))

    print(f"\n{'=' * 86}\nSEARCH COMPLETE  ({(time.time() - t0) / 60:.1f} min)\n{'=' * 86}")
    print(f"{'model':<22}{'h':>5}{'inner wape (initial)':>22}{'(tuned)':>12}{'change':>10}")
    print("-" * 86)
    for name, h, a, b in summary:
        print(f"{name:<22}{h:>5}{a:>22.4f}{b:>12.4f}{(b - a) / a * 100:>9.1f}%")
    print("\nThese are INNER-fold numbers, measured before the leaderboard's test period.")
    print("They are what the search selected on, not a result. For the result:")
    print("  uv run python dev/compare_models.py --run-all --tuned")
    print("  uv run python dev/compare_tuning.py")


if __name__ == "__main__":
    main()
