"""The LightGBM estimator the boosted models in this folder share.

This file is NOT a model — it never appears on the leaderboard and has no `__main__`.
It holds the one gradient-boosting estimator (`LightGBMTweedie`) and the one parameter
set (`PARAMS`) that `lightgbm_additive.py` builds on, so that "the LightGBM we use" is
defined in exactly one place.

It was lifted verbatim out of model_lab/lightgbm_gbdt.py, which is a *model* file that
did not make the shortlist. Only the shaping code came across; the model entry, its
pros/cons and its leaderboard row stayed behind in the lab.

Why Tweedie: demand here is compound Poisson-Gamma — a random number of orders, each of
a random size, with a real point mass at zero. Squared error assumes a bell curve and
pays for that assumption on every intermittent SKU.

Why the tail holdout: `common.split()` hands training rows back in date order, so the
last 10% really is the most recent slice of the training window. Early stopping on it is
a time holdout, and it still sits entirely before the horizon-sized gap, so it cannot
see the test period.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from common import to_pandas

PARAMS: dict[str, Any] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 800,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 50
VAL_FRACTION = 0.10


class LightGBMTweedie:
    def __init__(self, horizon: int, **params):
        self.horizon = horizon
        self.params = {**PARAMS, **params}
        self.model_ = None
        self.features_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        import inspect

        import lightgbm as lgb

        self.features_ = list(X.columns)
        Xp = to_pandas(X)
        y = np.clip(y, 0, None)  # Tweedie is undefined for negative targets

        # Hold out the TAIL of the training window for early stopping. common.split()
        # hands the rows back in date order, so this is the most recent 10% of the
        # training window -- a proper time holdout, and still entirely before the gap,
        # so no test information can reach it.
        n_val = int(len(Xp) * VAL_FRACTION)
        callbacks = [lgb.log_evaluation(period=0)]
        fit_kw: dict[str, Any] = {}
        if n_val > 1000:
            Xf, yf = Xp.iloc[:-n_val], y[:-n_val]
            Xv, yv = Xp.iloc[-n_val:], y[-n_val:]
            wf = None if w is None else w[:-n_val]
            sig = inspect.signature(lgb.LGBMRegressor.fit).parameters
            if "eval_X" in sig:  # lightgbm >= 4.6 renamed this argument
                fit_kw = {"eval_X": Xv, "eval_y": yv}
            else:
                fit_kw = {"eval_set": [(Xv, yv)]}
            callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))
        else:
            Xf, yf, wf = Xp, y, w

        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(Xf, yf, sample_weight=wf, callbacks=callbacks, **fit_kw)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self.model_.predict(to_pandas(X)[self.features_])

    def importances(self) -> list[tuple[str, float]]:
        g = self.model_.booster_.feature_importance(importance_type="gain")
        return sorted(zip(self.features_, g.tolist()), key=lambda t: -t[1])[:25]
