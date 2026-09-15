"""POISSON GLM — a linear model that knows it is predicting a count.

    uv run python model_lab/poisson_glm.py

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
Same shape as ridge_linear.py -- one linear combination of the features -- but two
things change, and both come from the fact that units sold is a COUNT:

    predicted units = exp(b0 + b1*x1 + ... + bK*xK)

1. The exponential (the "log link") means the prediction can never be negative, and it
   makes effects MULTIPLICATIVE instead of additive: a coefficient says "Prime Day
   multiplies demand by 1.4", not "Prime Day adds 12 units". That is how demand drivers
   actually behave -- a promotion lifts a 500-unit SKU by more units than a 5-unit SKU,
   and by roughly the same percentage.

2. It is fitted with Poisson loss instead of squared error. Squared error assumes the
   spread of the errors is the same everywhere, which is flatly false for counts: a SKU
   averaging 1,000 units varies by hundreds, a SKU averaging 3 varies by 2. Poisson loss
   builds in "variance grows with the mean", so the model is not bullied into fitting
   the big SKUs at the expense of everything else.

-----------------------------------------------------------------------------------
HOW IT IS SHAPED, AND WHY
-----------------------------------------------------------------------------------
* Target left in raw units (NOT logged). The log link is already inside the model; taking
  logs as well would be applying the transform twice and would reintroduce the
  under-prediction bias that log-space fitting causes.
* A small L2 penalty (alpha=1e-4) -- just enough to keep the solver stable with hundreds
  of correlated features, not enough to bias the fit.
* Median imputation and no string columns, same handicap and same reasons as
  ridge_linear.py.

-----------------------------------------------------------------------------------
HOW IT IS TRAINED
-----------------------------------------------------------------------------------
Iteratively (there is no closed form for this loss) -- L-BFGS, up to 300 iterations, on
each split's training window, with stock-out days weighted 0. Slower than ridge, still
well under a minute.

-----------------------------------------------------------------------------------
THE ASSUMPTION IT MAKES THAT IS WRONG HERE
-----------------------------------------------------------------------------------
A Poisson distribution asserts variance = mean. Real demand is OVER-dispersed: the
variance is much larger than the mean, because of promotions, bulk orders and virality.
So the model's point forecast is reasonable, but any uncertainty it implies is far too
narrow -- do not use it to size safety stock. The gradient-boosted files handle this
better, either with Tweedie loss (which fits the dispersion rather than assuming it) or
with explicit quantiles.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from common import LabModel, run_model

ALPHA = 1e-4
MAX_ITER = 300


class PoissonGLM:
    def __init__(self, horizon: int, alpha: float = ALPHA, max_iter: int = MAX_ITER):
        self.horizon = horizon
        self.alpha = alpha
        self.max_iter = max_iter
        self.pipe_ = None
        self.features_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import PoissonRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.features_ = [c for c, t in X.schema.items()
                          if t.is_numeric() or t == pl.Boolean]
        self.pipe_ = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("est", PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)),
        ])
        Xn = X.select(self.features_).cast(pl.Float64).to_numpy()
        # Raw counts, not logged: the log link is already part of the model.
        self.pipe_.fit(Xn, np.clip(y, 0, None), est__sample_weight=w)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        Xn = X.select(self.features_).cast(pl.Float64).to_numpy()
        return self.pipe_.predict(Xn)  # already in units, already non-negative


MODEL = LabModel(
    name="poisson_glm",
    family="Linear / GLM",
    blurb="""
    A generalised linear model with a log link and Poisson loss, so effects are
    multiplicative and predictions cannot go negative. The statistically correct linear
    model for count data, and the honest reference point for the tree models.
    """,
    pros=[
        "Right error model for counts: variance is allowed to grow with the level.",
        "Multiplicative effects match how promotions and seasonality really work.",
        "Predictions are non-negative by construction, no clipping needed.",
        "Coefficients read as percentage lifts, which is directly interpretable.",
        "Trains in well under a minute on the full panel.",
    ],
    cons=[
        "Assumes variance EQUALS the mean; real demand is heavily over-dispersed, so any "
        "uncertainty it implies is far too tight for safety stock.",
        "Still linear: no interactions unless hand-built.",
        "Same imputation and no-categoricals handicap as ridge_linear.py.",
        "Iterative solver, so noticeably slower than ridge for little accuracy gain.",
        "Sensitive to extreme feature values because of the exponential link -- one bad "
        "input can produce an enormous forecast.",
    ],
    build=lambda h: PoissonGLM(h),
    notes={"alpha": ALPHA, "max_iter": MAX_ITER, "link": "log", "loss": "poisson"},
)

if __name__ == "__main__":
    run_model(MODEL)
