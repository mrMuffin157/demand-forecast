"""RIDGE REGRESSION on log(units) — the simplest thing that actually learns.

    uv run python model_lab/ridge_linear.py

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
One straight line through a few hundred dimensions (K features):

    log(1 + units over next N days) = b0 + b1*x1 + b2*x2 + ... + bK*xK

`fit` solves for the b's in closed form -- no iteration, no search. "Ridge" means a
penalty on the size of the coefficients (alpha), which is what keeps hundreds of
correlated features from producing wild, unstable weights.

-----------------------------------------------------------------------------------
HOW IT IS SHAPED, AND WHY
-----------------------------------------------------------------------------------
Three shaping decisions, and each one exists because of a specific property of demand
data:

1. LOG TARGET. Units per SKU span several orders of magnitude over a 6-month window. Fit a line
   to raw units and the handful of blockbuster SKUs dominate the loss completely -- the
   model spends all its capacity on them and predicts nonsense for the long tail. Fitting
   log(1+units) turns "off by 500 units" into "off by 20%", which is the error a planner
   actually cares about. Predictions are converted back with expm1.

2. STANDARDISED INPUTS. A ridge penalty is applied to the raw coefficient sizes, so
   without scaling it would punish a feature measured in dollars far more than the same
   feature measured in thousands of dollars. Every column is centred and scaled first so
   the penalty is applied fairly.

3. MEDIAN IMPUTATION. A linear model cannot handle a missing value at all, and this
   dataset has plenty (a SKU three months old has no 364-day average). Missing values are
   filled with the training-set median of that column. Note this is a real handicap
   versus the tree models, which treat "missing" as its own informative branch --
   "this SKU has no year of history" is itself a strong signal, and this model throws it
   away.

Numeric columns only: a linear model cannot consume a string category like `sku_family`
without one-hot encoding it into hundreds of columns.

-----------------------------------------------------------------------------------
HOW IT IS TRAINED
-----------------------------------------------------------------------------------
Closed-form least squares with an L2 penalty, on the training window of each split.
Stock-out zero-days carry sample_weight 0 so they cannot drag the line down. Seconds,
not minutes.

-----------------------------------------------------------------------------------
WHY THIS FILE IS WORTH KEEPING EVEN THOUGH IT WILL NOT WIN
-----------------------------------------------------------------------------------
It is the leak detector. Demand at SKU level is genuinely hard; nothing honest gets WAPE
near zero here. If this file ever reports a WAPE close to zero, do not celebrate --
a feature is carrying the answer, and every other model on the board is lying too. It
costs seconds to run and it is the cheapest insurance in the folder.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from common import LabModel, run_model

ALPHA = 1.0  # ridge penalty strength


class RidgeLog:
    def __init__(self, horizon: int, alpha: float = ALPHA):
        self.horizon = horizon
        self.alpha = alpha
        self.pipe_ = None
        self.features_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Numeric/boolean only -- a linear model cannot take a raw string category.
        self.features_ = [c for c, t in X.schema.items()
                          if t.is_numeric() or t == pl.Boolean]
        self.pipe_ = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("est", Ridge(alpha=self.alpha)),
        ])
        Xn = X.select(self.features_).cast(pl.Float64).to_numpy()
        self.pipe_.fit(Xn, np.log1p(np.clip(y, 0, None)), est__sample_weight=w)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        Xn = X.select(self.features_).cast(pl.Float64).to_numpy()
        return np.expm1(self.pipe_.predict(Xn))  # back out of log space


MODEL = LabModel(
    name="ridge_linear",
    family="Linear / GLM",
    blurb="""
    A single ridge-penalised linear equation fitted to log(1 + units), with every input
    standardised and missing values median-filled. Trains in seconds and is the folder's
    leak detector: if a linear model scores suspiciously well, a feature is leaking.
    """,
    pros=[
        "Fast to fit and trivial to deploy -- it is one number per feature and a dot "
        "product.",
        "Fully inspectable: each coefficient is a direct 'this much of that'.",
        "Very hard to overfit with the ridge penalty on, so it is a stable reference.",
        "Its score is a useful floor: anything a tree model gains over this is the value "
        "of non-linearity and interactions, measured rather than assumed.",
        "The cheapest leakage alarm you can run.",
    ],
    cons=[
        "Assumes effects are additive and linear in log space. Demand is not -- a promo "
        "on a top seller and the same promo on a dead SKU do completely different things.",
        "Cannot represent interactions unless you build them by hand.",
        "Median-imputing missing values destroys real information ('no year of history' "
        "is a signal); tree models keep it.",
        "Cannot use the categorical features at all without one-hot blowup.",
        "expm1 of a mean in log space under-predicts the mean in unit space, so it is "
        "systematically a bit low on the spikiest SKUs.",
    ],
    build=lambda h: RidgeLog(h),
    notes={"alpha": ALPHA, "target_transform": "log1p"},
)

if __name__ == "__main__":
    run_model(MODEL)
