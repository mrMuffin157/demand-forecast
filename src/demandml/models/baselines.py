"""Baselines. Beat these or the pipeline is not earning its keep.

They are deliberately trivial and deliberately strong: on intermittent retail demand a
28-day trailing mean is a genuinely hard target, and plenty of published "ML beats
baseline" results do not clear it.

Each baseline reads a single pre-computed feature column rather than re-deriving history,
which keeps them exactly aligned with the ML models' information set -- same origin, same
data, no accidental advantage either way.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .base import FittedBase, register


class ColumnBaseline(FittedBase):
    """Predict a stored feature column, optionally scaled."""

    def __init__(self, column: str, scale: float = 1.0, name: str | None = None):
        self.column = column
        self.scale = scale
        self.name = name or f"baseline:{column}"

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None, **kw):
        if self.column not in X.columns:
            raise KeyError(f"{self.name} needs column {self.column!r}")
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self._post(X.get_column(self.column).to_numpy() * self.scale)


class GlobalMean(FittedBase):
    """The dumbest possible model: one number for everything. The floor."""

    name = "baseline:global_mean"

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None, **kw):
        self.mu_ = float(np.average(y, weights=w)) if len(y) else 0.0
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self._post(np.full(len(X), self.mu_))


class Croston(FittedBase):
    """Croston's method for intermittent demand, in its SBA-corrected form.

    Forecast = (smoothed non-zero demand size) / (smoothed interval between demands),
    with Syntetos-Boylan's ``(1 - alpha/2)`` bias correction. This is the textbook
    answer for series that are zero half the time, and it is the baseline any ML model
    on this dataset genuinely has to beat.

    Implemented from the stored ``nonzero_mean_*`` and ``adi_*`` features so it inherits
    the same past-only timing as everything else.
    """

    name = "baseline:croston_sba"

    def __init__(self, size_col: str = "nonzero_mean_91", adi_col: str = "adi_91", alpha: float = 0.1):
        self.size_col, self.adi_col, self.alpha = size_col, adi_col, alpha

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None, **kw):
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        size = X.get_column(self.size_col).to_numpy().astype(np.float64)
        adi = X.get_column(self.adi_col).to_numpy().astype(np.float64)
        adi = np.where(np.isfinite(adi) & (adi > 0), adi, 1.0)
        return self._post((1.0 - self.alpha / 2.0) * np.nan_to_num(size) / adi)


class SeasonalNaiveBlend(FittedBase):
    """Average of "same day last week" and "28-day trailing mean".

    Cheap, robust, and surprisingly competitive -- it has both the weekly shape and the
    level, which is most of what a daily retail forecast needs.
    """

    name = "baseline:seasonal_blend"

    def __init__(self, cols: tuple[str, ...] = ("units_lag_7", "units_mean_28")):
        self.cols = cols

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None, **kw):
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        arr = np.column_stack([X.get_column(c).to_numpy().astype(np.float64) for c in self.cols])
        return self._post(np.nanmean(arr, axis=1))


@register("naive_last", needs_features=True, note="yesterday's units, carried forward")
def _naive_last(**kw):
    return ColumnBaseline("units_lag_0", name="baseline:naive_last", **kw)


@register("naive_dow", needs_features=True, note="same weekday, mean of last 4 weeks")
def _naive_dow(**kw):
    return ColumnBaseline("units_same_dow_mean_4w", name="baseline:naive_dow", **kw)


@register("ma7", needs_features=True, note="7-day trailing mean")
def _ma7(**kw):
    return ColumnBaseline("units_mean_7", name="baseline:ma7", **kw)


@register("ma28", needs_features=True, note="28-day trailing mean (the one to beat)")
def _ma28(**kw):
    return ColumnBaseline("units_mean_28", name="baseline:ma28", **kw)


@register("ma91", needs_features=True, note="91-day trailing mean")
def _ma91(**kw):
    return ColumnBaseline("units_mean_91", name="baseline:ma91", **kw)


@register("ewm", needs_features=True, note="exponentially weighted mean, alpha=0.1")
def _ewm(**kw):
    return ColumnBaseline("units_ewm_01", name="baseline:ewm", **kw)


@register("seasonal_blend", needs_features=True, note="mean of lag-7 and 28d mean")
def _blend(**kw):
    return SeasonalNaiveBlend(**kw)


@register("croston", needs_features=True, note="Croston/SBA for intermittent demand")
def _croston(**kw):
    return Croston(**kw)


@register("global_mean", needs_features=True, note="single global constant (error floor)")
def _gm(**kw):
    return GlobalMean(**kw)
