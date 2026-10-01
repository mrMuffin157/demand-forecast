"""The lifted log fit that `xgboost_gbdt` and `random_forest` use (`dev/lifted_log.py`).

Four ways it could go wrong silently, each of which would still produce a plausible
forecast:

  * the lift column reaching the FEATURE matrix -- it is a future quantity, so a model
    shown it would score brilliantly in the backtest and be useless live;
  * the calibration factor fitted against the lift (demand) instead of sales -- slightly
    less precise per SKU, and invisible otherwise;
  * a pooled row from a store whose dataset predates the column (lift null) being taught
    zero instead of its own sales;
  * `--cost` building the log fit, so a pinball loss gets applied on the log scale and
    then rescaled -- a quantile of nothing in particular.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dev"))

from lifted_log import LiftedLog, lift_column  # noqa: E402

from demandml.features import feature_columns  # noqa: E402

H = 90


class Recorder:
    """An inner estimator that remembers its target and predicts a constant."""

    def __init__(self, value: float):
        self.value = value
        self.y = None

    def fit(self, X, y, w=None):
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        return np.full(len(X), np.log1p(self.value))


def _frame(sales, lift):
    return pl.DataFrame({"f": np.arange(len(sales), dtype=float),
                         f"y_cum{H}": np.asarray(sales, dtype=float),
                         lift_column(H): pl.Series(lift, dtype=pl.Float64)})


def test_trains_on_the_lift_and_falls_back_per_row():
    fr = _frame([10, 20, 30], [12.0, None, 40.0])
    inner = Recorder(5.0)
    LiftedLog(inner, H).fit(fr.select("f"), fr[f"y_cum{H}"].to_numpy(), frame=fr)
    np.testing.assert_allclose(np.expm1(inner.y), [12.0, 20.0, 40.0])


def test_without_a_frame_it_is_the_plain_log_fit():
    inner = Recorder(5.0)
    LiftedLog(inner, H).fit(pl.DataFrame({"f": [0.0, 1.0]}), np.array([3.0, 7.0]))
    np.testing.assert_allclose(np.expm1(inner.y), [3.0, 7.0])


def test_calibrates_to_sales_not_to_the_lift():
    fr = _frame([10, 20, 30], [20.0, 40.0, 60.0])        # lift doubles every window
    m = LiftedLog(Recorder(5.0), H).fit(fr.select("f"), fr[f"y_cum{H}"].to_numpy(),
                                       frame=fr)
    # predictions sum to the SALES on the training rows (60), not the lift (120)
    assert m.predict(fr.select("f")).sum() == pytest.approx(60.0)


def test_calibration_ignores_censored_rows():
    fr = _frame([10, 20, 1000], [10.0, 20.0, 1000.0])
    w = np.array([1.0, 1.0, 0.0])
    m = LiftedLog(Recorder(5.0), H).fit(fr.select("f"), fr[f"y_cum{H}"].to_numpy(), w,
                                       frame=fr)
    assert m.predict(fr.select("f"))[:2].sum() == pytest.approx(30.0)


def test_the_lift_is_never_a_feature():
    lf = pl.LazyFrame({"units_mean_28": [1.0], f"y_cum{H}": [1.0], lift_column(H): [1.0]})
    feats = feature_columns(lf)
    assert lift_column(H) not in feats and "units_mean_28" in feats


def test_cost_mode_builds_the_plain_estimator():
    import xgboost_gbdt as X

    assert isinstance(X.MODEL.build(H), LiftedLog)
    cost = X.MODEL.build(H, **X.MODEL.cost_params())
    assert isinstance(cost, X.XGBoostTweedie)
    assert cost.params["objective"] == "reg:quantileerror"


def test_a_tuned_override_cannot_bring_tweedie_back():
    import xgboost_gbdt as X

    m = X.MODEL.build(H, objective="reg:tweedie", tweedie_variance_power=1.5)
    assert m.inner.params["objective"] == "reg:squarederror"
    assert m.inner.params["tweedie_variance_power"] is None


# ======================================================================================
# What the fit saw is recorded, so a results file can answer "was the lift used?"
# ======================================================================================

def test_lift_stats_record_coverage_and_lifted_share():
    from lifted_log import lift_note, lift_stats

    fr = _frame([10, 20, 30, 40], [12.0, None, 30.0, 50.0])
    est = LiftedLog(Recorder(5.0), H).fit(fr.select("f"), fr[f"y_cum{H}"].to_numpy(), frame=fr)
    assert lift_stats(est) == {"lift_coverage": 0.75, "lifted_share": 0.5}
    assert "75% of rows" in lift_note(lift_stats(est))


def test_no_lift_column_is_recorded_as_zero_coverage_and_warned():
    """The silent fallback -- a dataset without the column trains on plain sales."""
    from lifted_log import lift_note, lift_stats

    fr = _frame([10, 20], [12.0, 30.0]).drop(lift_column(H))
    est = LiftedLog(Recorder(5.0), H).fit(fr.select("f"), fr[f"y_cum{H}"].to_numpy(), frame=fr)
    assert lift_stats(est)["lift_coverage"] == 0
    assert "NO LIFT" in lift_note(lift_stats(est))


def test_a_plain_estimator_records_nothing():
    from lifted_log import lift_note, lift_stats

    assert lift_stats(Recorder(1.0)) == {} and lift_note({}) == ""


def test_an_unseen_category_at_predict_is_nan_without_a_pandas_warning():
    """pandas 3 deprecates `astype(dtype)` silently dropping values outside the dtype's
    categories; the estimator must null them itself and predict as if they were missing."""
    import warnings

    import xgboost_gbdt as X

    rng = np.random.default_rng(0)
    n = 400
    tr = pl.DataFrame({"x": rng.normal(size=n),
                       "line": pl.Series(rng.choice(["a", "b"], n)).cast(pl.Categorical)})
    y = np.where(tr["line"] == "a", 5.0, 50.0) + rng.normal(size=n)
    est = X.XGBoostTweedie(H, n_estimators=20).fit(tr, y)
    te = pl.DataFrame({"x": [0.0, 0.0, 0.0],
                       "line": pl.Series(["a", "zzz", None]).cast(pl.Categorical)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = est.predict(te)
    assert p[1] == pytest.approx(p[2])          # unseen level == missing
    assert p[0] != pytest.approx(p[1])
