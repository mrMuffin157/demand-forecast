"""TWO-STAGE HURDLE — P(sale) x E[units | sale], instead of one model for both questions.

    uv run python dev/two_stage.py

-----------------------------------------------------------------------------------
WHY THIS FILE EXISTS
-----------------------------------------------------------------------------------
It was on the model shortlist on the strength of leading the gradient-boosted models in
`src/demandml`'s own experiment leaderboard -- a day-ahead target, horizon 7, a SKU
subset, a few rolling-origin folds. That is a real backtest, not a
random split, but it is a smaller and differently-shaped question than the one this
folder asks: cumulative 30/90/180-day totals, the full catalogue, this folder's own
feature build. `lgbm_quantile.py` was picked alongside it for the same reason -- neither
had been run through model_lab's splits. This file is what closes that gap for
`two_stage`.

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
Every other tree model in this folder asks one model to answer one question: "how many
units over the next N days?" On a target where many SKU-days are zero, that one model
has to simultaneously learn two different things -- whether a sale happens at all, and
if it does, how big it is -- and a single squared-error or Tweedie loss blends both
signals into one number.

A hurdle model asks two separate, purpose-built models instead:

    stage 1 (classifier)   P(units > 0)          -- "will this SKU move at all?"
    stage 2 (regressor)    E[units | units > 0]  -- "given that it moves, how much?"

    forecast = P(units > 0) * E[units | units > 0]

This is not cosmetic. The two questions genuinely have different drivers: whether a slow
SKU sells *anything* this month is closer to a launch-timing / stock-availability
question, while how much a SKU that is definitely selling moves is closer to a trend /
seasonality question. Croston's method (croston.py) is built on exactly the same
intuition for single-day intermittent series; this is the tree-model version of it,
applied to a cumulative multi-day target.

-----------------------------------------------------------------------------------
HOW IT IS SHAPED
-----------------------------------------------------------------------------------
* Stage 1 is a binary LightGBM classifier trained on `y > 0`, on every row. Its output is
  a probability, not a class.
* Stage 2 is a LightGBM regressor trained on `y` with ABSOLUTE error (`regression_l1`),
  and ONLY on the rows where `y > 0`. L1 rather than Tweedie/L2 because, once the
  zero-vs-nonzero split is handled by stage 1, the remaining distribution is still
  right-skewed and L1 (median-seeking) is less thrown by the occasional blockbuster than
  squared error is.
* Same `sample_weight = 0` on stock-out days as every other model here, on both stages --
  a censored zero must not teach stage 1 that a SKU stopped selling, and a censored
  nonzero row must not teach stage 2 the wrong size.
* Same early-stopping convention as lightgbm_gbdt.py: the most recent 10% of the training
  window (by date) held out for each stage independently. Stage 2's holdout is the tail
  of the POSITIVE rows only, since that is what it trains on.

-----------------------------------------------------------------------------------
HOW TO READ ITS SCORE
-----------------------------------------------------------------------------------
Against lightgbm_gbdt.py, the single-stage Tweedie model on identical data and folds --
that comparison is the whole point of running this file. Two ways it can land, both
informative: if it wins, the zero/nonzero split is worth the extra model; if it loses (or
wins by less than its own fold-to-fold spread), a single Tweedie loss was already doing
the job the hurdle was meant to do, at half the training cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from common import LabModel, run_model, to_pandas

#: Same base defaults as src/demandml/models/gbdt.py's _LGB_DEFAULTS, minus the Tweedie
#: knob -- neither stage uses a Tweedie objective.
BASE_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}
CLF_N_ESTIMATORS = 600
REG_N_ESTIMATORS = 800
EARLY_STOPPING_ROUNDS = 50
VAL_FRACTION = 0.10


def _tail_split(X: "Any", y: np.ndarray, w: np.ndarray | None):
    """The most recent VAL_FRACTION of rows (already date-sorted by common.split), held
    out for early stopping. Returns (fit rows, val rows) or (all rows, None) if too few
    rows to bother."""
    n_val = int(len(X) * VAL_FRACTION)
    if n_val <= 1000:
        return (X, y, w), None
    Xf, yf = X.iloc[:-n_val], y[:-n_val]
    Xv, yv = X.iloc[-n_val:], y[-n_val:]
    wf = None if w is None else w[:-n_val]
    return (Xf, yf, wf), (Xv, yv)


class TwoStageHurdle:
    def __init__(self, horizon: int, clf_n_estimators: int = CLF_N_ESTIMATORS,
                 reg_n_estimators: int = REG_N_ESTIMATORS,
                 reg_objective: str = "regression_l1", **params):
        self.horizon = horizon
        self.params = params
        # The two stages are separately sized because they are separately hard: stage 1
        # sees every row and learns a fairly smooth probability, stage 2 sees only the
        # positive rows and has to fit a heavy-tailed size. tune.py searches them apart.
        self.clf_n_estimators = int(clf_n_estimators)
        self.reg_n_estimators = int(reg_n_estimators)
        #: Stage 2's loss. L1 fits the MEDIAN positive size, which is the conservative
        #: choice on a heavy right tail; L2 fits the mean and chases spikes; Tweedie
        #: fits the compound Poisson-Gamma shape the raw demand actually has. Which one
        #: wins on WAPE is an empirical question, so it is in the search space.
        self.reg_objective = str(reg_objective)
        self.clf_ = None
        self.reg_ = None
        self.features_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        import inspect

        import lightgbm as lgb

        self.features_ = list(X.columns)
        Xp = to_pandas(X)
        y = np.clip(y, 0, None)
        base = {**BASE_PARAMS, **self.params}
        sig = inspect.signature(lgb.LGBMRegressor.fit).parameters
        use_eval_x = "eval_X" in sig

        # ---- stage 1: P(units > 0), every row -----------------------------------------
        (Xf, yf, wf), val = _tail_split(Xp, (y > 0).astype(int), w)
        callbacks = [lgb.log_evaluation(period=0)]
        fit_kw: dict[str, Any] = {}
        if val is not None:
            Xv, yv = val
            fit_kw = {"eval_X": Xv, "eval_y": yv} if use_eval_x else {"eval_set": [(Xv, yv)]}
            callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False,
                                                first_metric_only=True))
        self.clf_ = lgb.LGBMClassifier(objective="binary",
                                       n_estimators=self.clf_n_estimators, **base)
        # Every training window sold something (or none did): there is no zero/nonzero
        # split to learn, and a one-class classifier would report P(sale) = 0.
        # Stage 1 is then the constant share instead.
        self.p_const_ = None
        if np.unique(yf).size < 2:
            self.p_const_ = float(np.asarray(yf).mean())
        else:
            self.clf_.fit(Xf, yf, sample_weight=wf, callbacks=callbacks, **fit_kw)

        # ---- stage 2: E[units | units > 0], positive rows only ------------------------
        pos = y > 0
        Xpos, ypos = Xp[pos], y[pos]
        wpos = None if w is None else w[pos]
        (Xf2, yf2, wf2), val2 = _tail_split(Xpos, ypos, wpos)
        callbacks2 = [lgb.log_evaluation(period=0)]
        fit_kw2: dict[str, Any] = {}
        if val2 is not None:
            Xv2, yv2 = val2
            fit_kw2 = {"eval_X": Xv2, "eval_y": yv2} if use_eval_x else {"eval_set": [(Xv2, yv2)]}
            callbacks2.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False,
                                                 first_metric_only=True))
        reg_kw = dict(base)
        if self.reg_objective == "tweedie":
            reg_kw.setdefault("tweedie_variance_power", 1.2)
        self.reg_ = lgb.LGBMRegressor(objective=self.reg_objective,
                                      n_estimators=self.reg_n_estimators, **reg_kw)
        self.reg_.fit(Xf2, yf2, sample_weight=wf2, callbacks=callbacks2, **fit_kw2)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        Xp = to_pandas(X)[self.features_]
        p = (np.full(len(Xp), self.p_const_) if self.p_const_ is not None
             else self.clf_.predict_proba(Xp)[:, 1])
        size = self.reg_.predict(Xp)
        return p * size

    def importances(self) -> list[tuple[str, float]]:
        # The regressor's gain (stage 2, "how much") is usually the more informative
        # half to read -- stage 1's gain is dominated by whatever separates thin/new
        # SKUs from established ones.
        g = self.reg_.booster_.feature_importance(importance_type="gain")
        return sorted(zip(self.features_, g.tolist()), key=lambda t: -t[1])[:25]


MODEL = LabModel(
    name="two_stage",
    family="Trees (gradient boosting)",
    blurb="""
    A classifier predicting P(units > 0) multiplied by a regressor predicting E[units |
    units > 0], instead of one model asked to learn both the zero/nonzero split and the
    size at once. The tree-model version of the same idea behind Croston's method.
    """,
    pros=[
        "Splits a genuinely mixed question -- 'will it sell' and 'how much' -- into two "
        "models each purpose-built for one of them, rather than one loss blending both.",
        "Stage 2 trains only on rows that actually sold, so its L1 loss is not spent "
        "learning to predict zeros -- lightgbm_gbdt.py's single model has to do both.",
        "Scored the best WAPE of any gradient-boosted model in src/demandml's own "
        "leaderboard, beating every single-stage LightGBM variant there.",
        "Architecturally suited to this panel's many-zero-days problem, the same reason "
        "croston.py is on the shortlist -- worth having a tree-based answer to the same "
        "question.",
    ],
    cons=[
        "Two models to train, tune and keep in sync instead of one -- roughly double "
        "lightgbm_gbdt.py's training cost and twice the hyperparameter surface.",
        "Its shortlist ranking came from a smaller, non-catalogue-wide comparison "
        "(horizon 7, day-ahead target, a SKU subset) in src/demandml's leaderboard, not from "
        "this folder's cumulative 30/90/180-day, full-catalogue splits -- that gap is "
        "exactly what this run settles.",
        "The two stages are fit independently with no shared objective, so nothing "
        "guarantees the probability from stage 1 and the size from stage 2 are "
        "calibrated against each other the way a single joint model's outputs would be.",
        "Every disadvantage of lightgbm_gbdt.py applies to both stages: tuning burden, "
        "opacity, no extrapolation, overfitting risk.",
    ],
    build=lambda h, **kw: TwoStageHurdle(h, **kw),
    notes={**BASE_PARAMS, "clf_n_estimators": CLF_N_ESTIMATORS,
           "reg_n_estimators": REG_N_ESTIMATORS,
           "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "val_fraction": VAL_FRACTION},
)

if __name__ == "__main__":
    run_model(MODEL)
