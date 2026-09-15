"""LIGHTGBM with TWEEDIE loss — the model that usually wins on tabular demand data.

    uv run python model_lab/lightgbm_gbdt.py

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
Gradient boosting builds trees in SEQUENCE, and every tree after the first is fitted to
what the previous ones got wrong:

    prediction = tree1 + 0.05*tree2 + 0.05*tree3 + ... + 0.05*tree800

Tree 1 makes a rough guess. Tree 2 does not try to predict demand -- it tries to predict
tree 1's residual error. Tree 3 predicts what is still left. The learning rate (0.05)
means each tree only gets to nudge the answer 5% of the way, so no single tree can bully
the model, and hundreds of small corrections compose into a very precise fit.

This is why boosting beats a random forest (random_forest.py) on data like this: a forest
averages independent guesses to kill variance, while boosting attacks the remaining bias
directly, over and over.

-----------------------------------------------------------------------------------
THE SHAPING DECISION THAT MATTERS MOST: TWEEDIE LOSS
-----------------------------------------------------------------------------------
"Loss" is the definition of wrong that the model minimises, and the default -- squared
error -- assumes the target is roughly bell-shaped around its mean. Demand is nothing
like that. Even totalled over 30 days it is skewed: a floor of zero, a dense mass of
small values, and a long tail of blockbusters. Squared error on that shape over-forecasts
the quiet SKUs and under-forecasts the spikes, because being 200 units short on one big
SKU is punished 400x harder than being 10 short on twenty small ones.

Tweedie loss models the target as compound Poisson-Gamma: a random NUMBER of orders,
each of a random SIZE. That is not an analogy, it is literally how the units accumulate.
`tweedie_variance_power = 1.2` sits between Poisson (1.0, counts) and Gamma (2.0,
positive continuous), which matches an intermittent retail series.

On the data this was developed on, switching from squared error to Tweedie with
identical features and folds gave a clear WAPE improvement -- from changing one
parameter, more than most feature engineering buys. Changing the definition of "wrong"
is usually worth more than adding capacity.

-----------------------------------------------------------------------------------
THE OTHER SHAPING DECISIONS
-----------------------------------------------------------------------------------
* num_leaves = 127, min_child_samples = 100. LightGBM grows leaf-wise (it splits the
  single most promising leaf, wherever it is in the tree) rather than level-by-level.
  That is what makes it fast and accurate, and also what makes it overfit if unchecked --
  so the leaf count is capped and every leaf must rest on >= 100 rows.
* feature_fraction 0.7 / bagging_fraction 0.8: each tree sees 70% of columns and 80% of
  rows. Same de-correlation idea as the random forest, used here as regularisation.
* NATIVE CATEGORICALS. `sku_family`, `brand` and friends go in as categories, split
  directly with no one-hot blowup. The linear and sklearn-forest files cannot use these
  columns at all.
* NATIVE MISSING VALUES. NaN is sent down its own branch, so "this SKU has no 364-day
  history" stays a usable signal instead of being median-filled into a lie.
* n_estimators = 800 with EARLY STOPPING on a validation slice (see below).

-----------------------------------------------------------------------------------
HOW IT IS TRAINED
-----------------------------------------------------------------------------------
On each split's training window, with stock-out zero-days at sample_weight 0 so censored
observations cannot teach the model to predict its own stock-outs.

The last 10% of the training window (by date, not at random) is held out as a validation
set. Boosting gets monotonically better on training data forever, so the number of trees
has to be chosen by watching an unseen set: training stops when validation loss has not
improved for 50 rounds. That validation slice sits INSIDE the training window and before
the gap, so it never touches test data.

-----------------------------------------------------------------------------------
HOW TO READ ITS SCORE
-----------------------------------------------------------------------------------
Against moving_average.py. A gradient-boosted model with hundreds of features that cannot
beat one trailing average is a genuine and common result on intermittent demand, not a bug --
and it is the single most useful thing this folder can tell you. Check the spread across
the 3 splits too: if the margin is smaller than the spread, nothing has been proven.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from common import LabModel, run_model, to_pandas

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


MODEL = LabModel(
    name="lightgbm_gbdt",
    family="Trees (gradient boosting)",
    blurb="""
    Hundreds of small decision trees fitted in sequence, each correcting the errors the
    previous ones left, minimising Tweedie loss rather than squared error because demand
    is compound Poisson-Gamma rather than bell-shaped. Uses every feature in the table, including
    categoricals, and treats missing values as information.
    """,
    pros=[
        "Usually the most accurate family on tabular demand data, and the fastest way to "
        "get there.",
        "Tweedie loss fits the actual shape of intermittent demand -- worth more than "
        "most feature engineering.",
        "Uses everything: interactions, non-linearities, categoricals, missing values.",
        "Learns ACROSS SKUs, so a new product inherits patterns from similar products -- "
        "something no per-series baseline can do.",
        "Gain-based feature importances tell you which signals actually pay.",
    ],
    cons=[
        "Will overfit happily without early stopping and sensible leaf limits.",
        "Roughly 20 parameters that interact; the defaults here are informed guesses, "
        "not a tuned optimum.",
        "Minutes to train instead of milliseconds, and the artefact is megabytes.",
        "Much harder to explain to a planner than 'you sold 4 a day'.",
        "Cannot extrapolate beyond the range it trained on, so a SKU on a genuinely new "
        "trajectory is systematically under-forecast.",
        "Its edge over a moving average on intermittent SKUs is often smaller than the "
        "spread across splits -- check before believing it.",
    ],
    build=lambda h: LightGBMTweedie(h),
    notes={**PARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
           "val_fraction": VAL_FRACTION},
)

if __name__ == "__main__":
    run_model(MODEL)
