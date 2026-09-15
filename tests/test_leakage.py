"""Leakage and timing tests.

These are the tests that matter in a forecasting repo. A model bug makes a score worse
and you notice; a timing bug makes the score *better* and you ship it. Everything here
asserts that features cannot see the future.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from demandml import config as C
from demandml.features import (
    cumulative_target_exprs,
    feature_columns,
    target_exprs,
    target_history_exprs,
)
from demandml.metrics import bias, evaluate, wape
from demandml.splits import Fold, assert_no_leakage, make_folds


def _toy_panel(n_sku: int = 3, n_days: int = 500, seed: int = 0) -> pl.DataFrame:
    """A synthetic panel with a known weekly pattern and known zeros."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_sku):
        dates = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(n_days)]
        dow = np.array([d.weekday() for d in dates])
        lam = 5.0 + 3.0 * (dow >= 5) + s
        units = rng.poisson(lam).astype(float)
        units[rng.random(n_days) < 0.2] = 0.0  # intermittency
        frames.append(
            pl.DataFrame(
                {
                    C.SKU: [f"SKU-{s}"] * n_days,
                    C.DATE: dates,
                    C.TARGET: units,
                    "is_oos": rng.random(n_days) < 0.05,
                    "sessions": rng.poisson(50, n_days).astype(float),
                    "unit_price": np.full(n_days, 10.0 + s),
                    "is_deal": rng.random(n_days) < 0.1,
                }
            )
        )
    return pl.concat(frames).sort([C.SKU, C.DATE])


@pytest.fixture(scope="module")
def featured() -> pl.DataFrame:
    spec = C.FeatureSpec(
        lags=(1, 7, 28), roll_windows=(7, 28), same_dow_windows=(4,), ewm_alphas=(0.3,),
        exog=("sessions",), exog_lags=(0, 1), exog_roll_windows=(7, 28),
    )
    return (
        _toy_panel()
        .lazy()
        .with_columns(target_history_exprs(C.TARGET, spec))
        .with_columns(target_exprs((1, 7, 14)) + cumulative_target_exprs((7,)))
        .collect()
    )


def test_lag_0_is_the_origin_day(featured):
    """units_lag_0 must equal the origin day's own units -- known at forecast time."""
    for sku in featured.get_column(C.SKU).unique():
        d = featured.filter(pl.col(C.SKU) == sku).sort(C.DATE)
        assert np.allclose(
            d.get_column("units_lag_0").to_numpy(), d.get_column(C.TARGET).to_numpy()
        )


def test_lags_look_strictly_backwards(featured):
    for k in (1, 7, 28):
        for sku in featured.get_column(C.SKU).unique():
            d = featured.filter(pl.col(C.SKU) == sku).sort(C.DATE)
            got = d.get_column(f"units_lag_{k}").to_numpy()[k:]
            want = d.get_column(C.TARGET).to_numpy()[:-k]
            assert np.allclose(got, want), f"units_lag_{k} is misaligned"


def test_rolling_mean_window_is_inclusive_and_past_only(featured):
    for sku in featured.get_column(C.SKU).unique():
        d = featured.filter(pl.col(C.SKU) == sku).sort(C.DATE)
        u = d.get_column(C.TARGET).to_numpy()
        got = d.get_column("units_mean_7").to_numpy()[6:]
        want = np.convolve(u, np.ones(7) / 7, "valid")
        assert np.allclose(got, want, atol=1e-6)


def test_targets_point_forwards(featured):
    for h in (1, 7, 14):
        for sku in featured.get_column(C.SKU).unique():
            d = featured.filter(pl.col(C.SKU) == sku).sort(C.DATE)
            got = d.get_column(f"y_h{h}").to_numpy()[:-h]
            want = d.get_column(C.TARGET).to_numpy()[h:]
            assert np.allclose(got, want), f"y_h{h} is misaligned"


def test_cumulative_target_sums_the_right_window(featured):
    for sku in featured.get_column(C.SKU).unique():
        d = featured.filter(pl.col(C.SKU) == sku).sort(C.DATE)
        u = d.get_column(C.TARGET).to_numpy()
        got = d.get_column("y_cum7").to_numpy()[:-7]
        want = np.convolve(u[1:], np.ones(7), "valid")
        assert np.allclose(got, want)


def test_features_never_touch_another_sku(featured):
    """Every window is partitioned over sku, so a SKU's first row must have null lags."""
    firsts = featured.group_by(C.SKU).first()
    for col in ("units_lag_1", "units_lag_7", "units_lag_28"):
        assert firsts.get_column(col).null_count() == len(firsts), (
            f"{col} is non-null on a SKU's first row -- history bled across SKUs"
        )


def test_no_feature_perfectly_predicts_the_target(featured):
    """The blunt instrument: if some feature correlates ~1.0 with a future target,
    something is leaking. Catches most mistakes a human would make editing features.py."""
    y = featured.get_column("y_h1")
    sub = featured.drop_nulls("y_h1")
    y = sub.get_column("y_h1").to_numpy()
    for c in sub.columns:
        if c.startswith(("y_h", "y_cum")) or c in (C.SKU, C.DATE, C.TARGET):
            continue
        x = sub.get_column(c)
        if not x.dtype.is_numeric():
            continue
        x = x.fill_null(0).to_numpy().astype(float)
        if np.std(x) < 1e-9:
            continue
        r = abs(np.corrcoef(x, y)[0, 1])
        assert r < 0.995, f"feature {c!r} correlates {r:.4f} with y_h1 -- leak?"


def test_feature_columns_excludes_targets_and_same_day_measurements():
    lf = _toy_panel().lazy().with_columns(target_exprs((1, 7)))
    feats = feature_columns(lf)
    assert not any(f.startswith(("y_h", "y_cum")) for f in feats)
    for banned in (C.TARGET, C.SKU, C.DATE, "sessions", "is_oos", "unit_price"):
        assert banned not in feats, f"{banned} must not be handed to a model raw"


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------


def test_folds_are_ordered_and_non_overlapping():
    dates = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(1200)]
    folds = make_folds(dates, C.SplitSpec(test_size=28, n_folds=4), horizon=7)
    assert len(folds) == 4
    for f in folds:
        assert f.train_end < f.test_start
        assert f.test_start <= f.test_end
    for a, b in zip(folds, folds[1:]):
        assert a.test_end < b.test_start, "test windows overlap"


def test_fold_gap_covers_the_horizon():
    """Train origins plus the horizon must not reach into the test window."""
    dates = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(1200)]
    for h in (1, 7, 28):
        for f in make_folds(dates, C.SplitSpec(test_size=28, n_folds=2), horizon=h):
            assert f.train_end + dt.timedelta(days=h) < f.test_start


def test_assert_no_leakage_actually_fires():
    train = pl.DataFrame({C.DATE: [dt.date(2024, 1, 31)]})
    test = pl.DataFrame({C.DATE: [dt.date(2024, 2, 1)]})
    assert_no_leakage(train, test, horizon=0) is None
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(train, test, horizon=7)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def test_wape_is_defined_when_actuals_are_zero():
    assert wape([0, 0, 10], [1, 1, 10]) == pytest.approx(0.2)
    assert np.isnan(wape([0, 0], [1, 1]))  # no volume -> undefined, not inf


def test_bias_sign_convention():
    assert bias([10, 10], [12, 12]) > 0  # over-forecast
    assert bias([10, 10], [8, 8]) < 0


def test_evaluate_returns_the_full_bundle():
    m = evaluate([1, 2, 3], [1, 2, 3], scale=1.0)
    assert m["wape"] == 0 and m["mae"] == 0 and m["n"] == 3
    assert {"rmsse", "mase", "bias", "shortfall"} <= set(m)


def test_sample_weight_zeroes_are_excluded():
    assert wape([10, 10], [10, 1000], w=[1, 0]) == pytest.approx(0.0)
