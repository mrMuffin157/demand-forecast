"""LIGHTGBM QUANTILE — three separate models (P10/P50/P90) instead of one mean forecast.

    uv run python dev/lgbm_quantile.py

-----------------------------------------------------------------------------------
WHY THIS FILE EXISTS
-----------------------------------------------------------------------------------
It was on the model shortlist for one reason: none of the five point-forecast models in
this folder (random_forest, moving_average, xgboost_gbdt, lightgbm_additive, croston)
give a range, and a safety-stock decision needs one -- "order 40" and "order 40, but
could be anywhere from 25 to 70" are different decisions even when the headline number
is identical. `two_stage.py` is the shortlist's other gap for the same reason: both were
picked from `src/demandml`'s own experiment leaderboard (horizon 7, day-ahead target, a
SKU subset) and never run through THIS folder's rolling-origin splits on the
catalogue-wide, cumulative 30/90/180-day targets. This file and two_stage.py close that
gap.

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
A normal regression tree minimises squared error, which is minimised by the MEAN. Swap
the loss for the pinball (quantile) loss and the same tree structure instead minimises
error at one specific quantile -- ask for 0.9 and every leaf converges on "the value 90%
of the training examples in this leaf fell below", not the average.

There is no single model that produces three quantiles at once: this file trains three
independent LightGBM models, one per quantile (`alpha=0.1`, `alpha=0.5`, `alpha=0.9`),
each seeing the same rows, features and sample weights, differing only in what they are
each individually told to minimise. `predict()` returns the P50 model's output so this
can stand in for a point forecast on the leaderboard; the real product is
`predict_quantiles()`, and `q90 - q50` is the number a safety-stock policy actually
wants.

-----------------------------------------------------------------------------------
CONSEQUENCES OF FITTING THREE MODELS INDEPENDENTLY
-----------------------------------------------------------------------------------
* Triples the training cost of lightgbm_gbdt.py for the same data -- three boosted
  ensembles instead of one.
* QUANTILE CROSSING is possible: because each model is a separate fit, nothing enforces
  q10 <= q50 <= q90 pointwise. `predict_quantiles()` sorts the three columns row-wise to
  fix this, which is monotonicity-preserving and does not change any model's score --
  but it means the interval you read off is repaired, not naturally ordered.
* Pinball loss is NOT the same objective as Tweedie, so P50 here is not guaranteed to
  match (or beat) lightgbm_gbdt.py's Tweedie mean on WAPE. Comparing this file's WAPE
  column to lightgbm_gbdt.py's is exactly the point of running it.

-----------------------------------------------------------------------------------
HOW IT IS TRAINED
-----------------------------------------------------------------------------------
Same convention as every other boosted file here: stock-out days at sample_weight 0, the
most recent 10% of the training window (by date, not at random) held out for early
stopping on each of the three quantile models independently, same hyperparameters as
`src/demandml/models/gbdt.py`'s `LGBMQuantile` (this file is that model, re-run on this
folder's cumulative targets and full catalogue instead of the day-ahead, SKU-subset
comparison it was originally picked from).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from common import LabModel, run_model, target_quantile, to_pandas

#: Same defaults as src/demandml/models/gbdt.py's _LGB_DEFAULTS, minus the Tweedie knob
#: (pinball loss has no variance-power parameter).
PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 700,
    "n_jobs": -1,
    "verbose": -1,
}
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
EARLY_STOPPING_ROUNDS = 50
VAL_FRACTION = 0.10


def _fit_one(X: "Any", y: np.ndarray, w: np.ndarray | None, alpha: float, params: dict[str, Any]):
    import inspect

    import lightgbm as lgb

    n_val = int(len(X) * VAL_FRACTION)
    callbacks = [lgb.log_evaluation(period=0)]
    fit_kw: dict[str, Any] = {}
    if n_val > 1000:
        Xf, yf = X.iloc[:-n_val], y[:-n_val]
        Xv, yv = X.iloc[-n_val:], y[-n_val:]
        wf = None if w is None else w[:-n_val]
        sig = inspect.signature(lgb.LGBMRegressor.fit).parameters
        if "eval_X" in sig:  # lightgbm >= 4.6 renamed this argument
            fit_kw = {"eval_X": Xv, "eval_y": yv}
        else:
            fit_kw = {"eval_set": [(Xv, yv)]}
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))
    else:
        Xf, yf, wf = X, y, w

    model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
    model.fit(Xf, yf, sample_weight=wf, callbacks=callbacks, **fit_kw)
    return model


class LightGBMQuantile:
    def __init__(self, horizon: int, quantiles: tuple[float, ...] = QUANTILES,
                 center: float = 0.5, lo: float = 0.1, hi: float = 0.9, **params):
        self.horizon = horizon
        # WAPE is a sum of absolute errors, so in theory the conditional MEDIAN
        # minimises it and `center` should stay at 0.5. In practice the panel is
        # right-skewed and its zeros are censored, so the quantile that actually
        # minimises WAPE on held-out data need not be exactly 0.5. It is still NOT
        # searched: under --cost it is derived from CU/CO (see tune.py's SPACES note).
        self.center = float(center)
        self.quantiles = tuple(sorted({float(lo), self.center, float(hi)})) \
            if quantiles is QUANTILES else tuple(quantiles)
        self.params = {**PARAMS, **params}
        self.models_: dict[float, Any] = {}
        self.features_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        self.features_ = list(X.columns)
        Xp = to_pandas(X)
        y = np.clip(y, 0, None)
        for q in self.quantiles:
            self.models_[q] = _fit_one(Xp, y, w, q, self.params)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        mid = min(self.quantiles, key=lambda q: abs(q - self.center))
        return self.models_[mid].predict(to_pandas(X)[self.features_])

    def predict_quantiles(self, X: pl.DataFrame) -> pl.DataFrame:
        Xp = to_pandas(X)[self.features_]
        cols = {f"q{q}": self.models_[q].predict(Xp) for q in sorted(self.quantiles)}
        arr = np.sort(np.column_stack(list(cols.values())), axis=1)  # fix any crossing
        return pl.DataFrame({k: arr[:, i] for i, k in enumerate(cols)})

    def importances(self) -> list[tuple[str, float]]:
        mid = min(self.quantiles, key=lambda q: abs(q - self.center))
        g = self.models_[mid].booster_.feature_importance(importance_type="gain")
        return sorted(zip(self.features_, g.tolist()), key=lambda t: -t[1])[:25]


MODEL = LabModel(
    name="lgbm_quantile",
    family="Trees (gradient boosting)",
    blurb="""
    Three independent LightGBM models minimising pinball loss at the 10th, 50th and 90th
    percentiles instead of one model minimising squared or Tweedie error. The P50 model
    stands in as this file's point forecast; the actual product is the P10-P90 interval,
    which is what a safety-stock decision needs and none of this folder's other models
    provide.
    """,
    pros=[
        "The only model in this folder that gives a range instead of a single number -- "
        "`q90 - q50` is a safety-stock quantity directly, with no separate uncertainty "
        "model bolted on.",
        "Each quantile is optimised for exactly the question it answers, rather than "
        "deriving an interval from a point model's residuals after the fact.",
        "Same LightGBM machinery as lightgbm_gbdt.py -- native categoricals, native "
        "missing values, learns across SKUs.",
    ],
    cons=[
        "Three times the training cost of lightgbm_gbdt.py for the same data, because "
        "it is three separate boosted ensembles, not one model with three outputs.",
        "Quantile crossing is possible since the three models are fit independently; "
        "`predict_quantiles()` sorts to repair it, which hides rather than prevents the "
        "underlying inconsistency.",
        "Pinball loss is a different objective from Tweedie, so its P50 is not "
        "guaranteed to match or beat lightgbm_gbdt.py's mean on WAPE -- that comparison "
        "is the reason this file exists.",
        "Was picked for the shortlist from a smaller, non-catalogue-wide comparison "
        "(horizon 7, a SKU subset) in src/demandml's own leaderboard, not from this folder's "
        "splits -- this run is what settles whether it belongs here too.",
        "Every disadvantage of lightgbm_gbdt.py applies three times over: tuning "
        "burden, opacity, no extrapolation.",
    ],
    build=lambda h, **kw: LightGBMQuantile(h, **kw),
    # Under --cost, read the point forecast off the quantile the inventory cost actually
    # minimises rather than off the median. This is the one model here that can honour an
    # asymmetric loss natively: it already fits pinball loss, so the change is which
    # quantile it is asked for -- no new model, no new machinery. The value is DERIVED
    # from cu:co, never searched; see the note in tune.py's search space.
    cost_params=lambda: {"center": target_quantile(),
                         "quantiles": (0.1, target_quantile(), 0.9)},
    notes={**PARAMS, "quantiles": QUANTILES, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
           "val_fraction": VAL_FRACTION},
)

if __name__ == "__main__":
    run_model(MODEL)
