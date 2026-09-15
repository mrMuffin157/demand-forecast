"""Gradient-boosted trees -- the workhorse for tabular demand forecasting.

Three things here are specific to *demand* rather than generic regression:

1. **Tweedie loss.** With a large share of zeros and a long right tail, the target is compound
   Poisson-Gamma, not Gaussian. Squared error on this data over-forecasts the zeros and
   under-forecasts the spikes; ``objective="tweedie"`` with ``variance_power ~ 1.1-1.3``
   fits the actual distribution. This is usually the single biggest win available.
2. **Quantile heads.** Replenishment needs P90, not the mean. ``LGBMQuantile`` fits one
   model per quantile so you can size safety stock directly.
3. **Native categoricals.** LightGBM splits on categories without one-hot blowup, which
   matters when ``sku_family`` has hundreds of levels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .base import FittedBase, register

# Reasonable starting point for ~1-3M rows x ~250 features on an 11-core machine.
_LGB_DEFAULTS: dict[str, Any] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "n_estimators": 700,
    "n_jobs": -1,
    "verbose": -1,
}


def _to_pandas(X: pl.DataFrame) -> "Any":
    """LightGBM/XGBoost want pandas with numeric/bool/category dtypes only.

    Polars strings arrive as pyarrow-backed ``str``, which both libraries reject, so
    every non-numeric column is converted to a pandas ``category``. Doing it here rather
    than in each model keeps the two backends consistent -- they then split on the same
    encoding, which matters when you are comparing them.
    """
    import pandas as pd

    cat_src = [
        c for c, t in X.schema.items()
        if t in (pl.String, pl.Categorical, pl.Enum) or t == pl.Object
    ]
    if cat_src:
        X = X.with_columns([pl.col(c).cast(pl.String).cast(pl.Categorical) for c in cat_src])

    df = X.to_pandas(use_pyarrow_extension_array=False)
    for c in df.columns:
        dt = df[c].dtype
        if isinstance(dt, pd.CategoricalDtype):
            continue
        if pd.api.types.is_bool_dtype(dt):
            df[c] = df[c].astype("int8")
        elif not pd.api.types.is_numeric_dtype(dt):
            df[c] = df[c].astype("category")
    return df


class LGBMDemand(FittedBase):
    """LightGBM regressor with demand-appropriate defaults and early stopping."""

    def __init__(
        self,
        name: str = "lgbm",
        early_stopping_rounds: int = 100,
        log_target: bool = False,
        **params: Any,
    ):
        self.name = name
        self.params = {**_LGB_DEFAULTS, **params}
        self.early_stopping_rounds = early_stopping_rounds
        #: log1p the target and expm1 the prediction. Helps with L2 loss; usually
        #: unnecessary (and slightly harmful) with Tweedie, which already handles scale.
        self.log_target = log_target
        self.model_ = None
        self.features_: list[str] = []

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        w: np.ndarray | None = None,
        eval_set: tuple[pl.DataFrame, np.ndarray] | None = None,
        **kw: Any,
    ):
        import lightgbm as lgb

        self.features_ = list(X.columns)
        Xp = _to_pandas(X)
        yt = np.log1p(np.clip(y, 0, None)) if self.log_target else y

        params = dict(self.params)
        callbacks = [lgb.log_evaluation(period=0)]
        fit_kw: dict[str, Any] = {}
        if eval_set is not None:
            Xv, yv = eval_set
            yv_t = np.log1p(np.clip(yv, 0, None)) if self.log_target else yv
            Xv_p = _to_pandas(Xv)[self.features_]
            # lightgbm >= 4.6 deprecated `eval_set` in favour of `eval_X`/`eval_y`.
            import inspect

            if "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters:
                fit_kw["eval_X"] = Xv_p
                fit_kw["eval_y"] = yv_t
            else:
                fit_kw["eval_set"] = [(Xv_p, yv_t)]
            callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds, verbose=False, first_metric_only=True)
            )

        self.model_ = lgb.LGBMRegressor(**params)
        self.model_.fit(Xp, yt, sample_weight=w, callbacks=callbacks, **fit_kw)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        p = self.model_.predict(_to_pandas(X)[self.features_])
        if self.log_target:
            p = np.expm1(p)
        return self._post(p)

    def importances(self, kind: str = "gain") -> pl.DataFrame:
        """Feature importances as a sorted frame -- the main diagnostic for "is a feature
        leaking?". A lag/rolling feature dominating by 10x is normal; a raw same-day
        column at the top means something slipped through ``feature_columns``."""
        imp = self.model_.booster_.feature_importance(importance_type=kind)
        return (
            pl.DataFrame({"feature": self.features_, kind: imp})
            .sort(kind, descending=True)
        )


class LGBMQuantile(FittedBase):
    """One LightGBM per quantile, exposing ``predict_quantiles``.

    ``predict`` returns the median so it can stand in for a point model, but the reason
    to use this is the interval: ``q90 - q50`` is the safety stock.
    """

    def __init__(self, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9), name: str = "lgbm_quantile", **params: Any):
        self.name = name
        self.quantiles = quantiles
        self.params = {
            **_LGB_DEFAULTS,
            **{"objective": "quantile", "tweedie_variance_power": None},
            **params,
        }
        self.params.pop("tweedie_variance_power", None)
        self.models_: dict[float, LGBMDemand] = {}

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None, **kw: Any):
        for q in self.quantiles:
            m = LGBMDemand(name=f"{self.name}_q{q}", **{**self.params, "alpha": q})
            self.models_[q] = m.fit(X, y, w, **kw)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        mid = min(self.quantiles, key=lambda q: abs(q - 0.5))
        return self.models_[mid].predict(X)

    def predict_quantiles(self, X: pl.DataFrame) -> pl.DataFrame:
        # Quantile crossing is possible when each level is fit independently; sorting the
        # rows fixes it and is monotonicity-preserving.
        cols = {f"q{q}": self.models_[q].predict(X) for q in sorted(self.quantiles)}
        arr = np.sort(np.column_stack(list(cols.values())), axis=1)
        return pl.DataFrame({k: arr[:, i] for i, k in enumerate(cols)})


class TwoStageDemand(FittedBase):
    """Classifier x regressor: P(sale today) * E[units | sale].

    An explicit hurdle model for intermittent series. Sometimes beats single-stage
    Tweedie on very sparse SKUs, because the "will it sell at all" question and the "how
    many" question have genuinely different drivers. Worth having in the comparison
    rather than assuming.
    """

    name = "two_stage"

    def __init__(self, **params: Any):
        self.params = params
        self.clf_ = None
        self.reg_ = None
        self.features_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None, **kw: Any):
        import lightgbm as lgb

        self.features_ = list(X.columns)
        Xp = _to_pandas(X)
        base = {**_LGB_DEFAULTS, **self.params}
        base.pop("tweedie_variance_power", None)

        clf_params = {**base, "objective": "binary", "n_estimators": 600}
        self.clf_ = lgb.LGBMClassifier(**clf_params)
        self.clf_.fit(Xp, (y > 0).astype(int), sample_weight=w,
                      callbacks=[lgb.log_evaluation(period=0)])

        pos = y > 0
        reg_params = {**base, "objective": "regression_l1", "n_estimators": 800}
        self.reg_ = lgb.LGBMRegressor(**reg_params)
        wp = None if w is None else w[pos]
        self.reg_.fit(Xp[pos], y[pos], sample_weight=wp,
                      callbacks=[lgb.log_evaluation(period=0)])
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        Xp = _to_pandas(X)[self.features_]
        p = self.clf_.predict_proba(Xp)[:, 1]
        size = self.reg_.predict(Xp)
        return self._post(p * size)


class XGBDemand(FittedBase):
    """XGBoost with Tweedie loss -- a second opinion with different inductive bias."""

    def __init__(self, name: str = "xgb", **params: Any):
        self.name = name
        self.params = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": 1.2,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
            "n_estimators": 1200,
            "tree_method": "hist",
            "enable_categorical": True,
            # ABOVE every categorical cardinality in the feature table, NOT below it.
            # XGBoost partitions a categorical's level ranking, so one split can carve
            # an arbitrary subset of levels out -- which on `sku_family` (hundreds of
            # levels, nearly a SKU identifier) stores a per-family level table from the
            # training window and over-forecasts badly the moment the level shifts. LightGBM
            # survives the same column via cat_smooth / min_data_per_group; XGBoost has
            # no equivalent, so one-vs-rest is the only guard. This was `1` -- the most
            # dangerous possible setting, forcing partitioning everywhere. See the
            # header of model_lab/xgboost_gbdt.py for the ablation that pinned it down.
            "max_cat_to_onehot": 4096,
            "n_jobs": -1,
            "verbosity": 0,
            **params,
        }
        self.model_ = None
        self.features_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None,
            eval_set: tuple[pl.DataFrame, np.ndarray] | None = None, **kw: Any):
        import xgboost as xgb

        self.features_ = list(X.columns)
        params = dict(self.params)
        fit_kw: dict[str, Any] = {}
        if eval_set is not None:
            params["early_stopping_rounds"] = kw.get("early_stopping_rounds", 100)
            fit_kw["eval_set"] = [(_to_pandas(eval_set[0])[self.features_], eval_set[1])]
            fit_kw["verbose"] = False
        self.model_ = xgb.XGBRegressor(**params)
        self.model_.fit(_to_pandas(X), y, sample_weight=w, **fit_kw)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self._post(self.model_.predict(_to_pandas(X)[self.features_]))


class ScaledTarget(FittedBase):
    """Fit on demand *relative to the SKU's own recent level*, then scale back.

    The problem this solves: one global model has to span SKUs selling 0.2 units/day and
    SKUs selling 400/day. Trees spend their capacity learning which SKU is which instead
    of learning demand dynamics, and the loss is dominated by the big sellers.

    Dividing the target by a past-only scale (the trailing 28-day mean, by default) makes
    every series roughly unit-scale, so the model learns one shared pattern -- "demand is
    running 1.3x its recent level going into Black Friday" -- and transfers it across the
    whole catalogue. Multiplying back at predict time restores the units.

    The scale column must itself be past-only, which ``units_mean_28`` is. Rows whose
    scale is zero or missing fall back to the unscaled target so early-life SKUs are not
    silently dropped.
    """

    def __init__(
        self,
        inner: str = "lgbm_tweedie",
        scale_col: str = "units_mean_28",
        floor: float = 0.05,
        name: str | None = None,
        **params: Any,
    ):
        self.inner_name = inner
        self.scale_col = scale_col
        self.floor = floor
        self.name = name or f"scaled:{inner}"
        self.params = params
        self.inner_ = None

    def _scale(self, X: pl.DataFrame) -> np.ndarray:
        if self.scale_col not in X.columns:
            raise KeyError(
                f"{self.name} needs {self.scale_col!r} in the feature matrix "
                "(it is produced by features.py -- do not exclude it)"
            )
        s = X.get_column(self.scale_col).to_numpy().astype(np.float64)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        # Floor keeps the division stable without distorting real low-volume SKUs much.
        return np.maximum(s, self.floor)

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None, **kw: Any):
        from . import get_model

        s = self._scale(X)
        self.inner_ = get_model(self.inner_name, **self.params)
        # Weight by scale so the relative-error fit still respects absolute volume --
        # without this the model optimises for the smallest SKUs in the catalogue.
        w2 = s if w is None else w * s
        ek = kw.pop("eval_set", None)
        if ek is not None:
            Xv, yv = ek
            kw["eval_set"] = (Xv, yv / self._scale(Xv))
        self.inner_.fit(X, y / s, w2, **kw)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self._post(self.inner_.predict(X) * self._scale(X))

    def importances(self, kind: str = "gain") -> pl.DataFrame:
        return self.inner_.importances(kind)


@register("lgbm_tweedie", note="LightGBM, Tweedie loss -- the default recommendation")
def _lgb_tweedie(**kw):
    return LGBMDemand(name="lgbm_tweedie", **kw)


@register("lgbm_poisson", note="LightGBM, Poisson loss")
def _lgb_poisson(**kw):
    return LGBMDemand(name="lgbm_poisson", objective="poisson",
                      **{k: v for k, v in kw.items() if k != "objective"})


@register("lgbm_l2", note="LightGBM, squared error (the naive choice, for contrast)")
def _lgb_l2(**kw):
    return LGBMDemand(name="lgbm_l2", objective="regression", **kw)


@register("lgbm_l1", note="LightGBM, absolute error -- optimises the median")
def _lgb_l1(**kw):
    return LGBMDemand(name="lgbm_l1", objective="regression_l1", **kw)


@register("lgbm_log_l2", note="LightGBM on log1p(units), squared error")
def _lgb_log(**kw):
    return LGBMDemand(name="lgbm_log_l2", objective="regression", log_target=True, **kw)


@register("lgbm_quantile", note="LightGBM quantile heads P10/P50/P90 for safety stock")
def _lgb_q(**kw):
    return LGBMQuantile(**kw)


@register("two_stage", note="P(sale) x E[units|sale] hurdle model")
def _two_stage(**kw):
    return TwoStageDemand(**kw)


@register("xgb_tweedie", note="XGBoost, Tweedie loss")
def _xgb(**kw):
    return XGBDemand(**kw)


@register("scaled_lgbm", note="LightGBM on units/trailing-mean -- normalises SKU scale")
def _scaled(**kw):
    return ScaledTarget(inner="lgbm_tweedie", **kw)


@register("scaled_lgbm_l2", note="Scaled target with squared-error loss")
def _scaled_l2(**kw):
    return ScaledTarget(inner="lgbm_l2", name="scaled:lgbm_l2", **kw)
