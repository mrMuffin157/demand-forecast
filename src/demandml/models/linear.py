"""Linear / GLM models.

Not expected to win, but they earn their place: a Poisson GLM on a handful of lags is
interpretable, trains in seconds, and is the sanity check that tells you whether the
tree models' gains are real signal or just capacity. Ridge on log1p is also the cheapest
way to detect a leaking feature -- if a linear model hits WAPE 0.05, something is wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .base import FittedBase, register


class SklearnDemand(FittedBase):
    """Wrap any sklearn regressor with median imputation + scaling, on numeric columns."""

    def __init__(self, estimator, name: str = "sklearn", log_target: bool = False, scale: bool = True):
        self.name = name
        self.estimator = estimator
        self.log_target = log_target
        self.scale = scale
        self.pipe_ = None
        self.features_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None, **kw: Any):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.features_ = [c for c, t in X.schema.items() if t.is_numeric() or t == pl.Boolean]
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if self.scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("est", self.estimator))
        self.pipe_ = Pipeline(steps)

        Xn = X.select(self.features_).cast(pl.Float64).to_numpy()
        yt = np.log1p(np.clip(y, 0, None)) if self.log_target else y
        try:
            self.pipe_.fit(Xn, yt, est__sample_weight=w)
        except TypeError:
            self.pipe_.fit(Xn, yt)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        Xn = X.select(self.features_).cast(pl.Float64).to_numpy()
        p = self.pipe_.predict(Xn)
        if self.log_target:
            p = np.expm1(p)
        return self._post(p)


@register("ridge_log", note="Ridge on log1p(units) -- fast leak detector")
def _ridge(**kw):
    from sklearn.linear_model import Ridge

    return SklearnDemand(Ridge(alpha=1.0), name="ridge_log", log_target=True, **kw)


@register("poisson_glm", note="Poisson GLM -- interpretable count baseline")
def _poisson(**kw):
    from sklearn.linear_model import PoissonRegressor

    return SklearnDemand(
        PoissonRegressor(alpha=1e-4, max_iter=300), name="poisson_glm", **kw
    )


@register("hist_gbr", note="sklearn HistGradientBoosting, Poisson loss (no deps)")
def _hgb(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return SklearnDemand(
        HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.06),
        name="hist_gbr",
        scale=False,
        **kw,
    )
