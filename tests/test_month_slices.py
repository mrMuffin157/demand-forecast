"""The month-slice targets that `model_lab/lightgbm_additive.py` sums.

The additive model is only meaningful if `y_cum_m1 + ... + y_cum_m6` really is the same
quantity as `y_cum180`. If the shift is off by one, the sum stops being the 180-day
total and the model is quietly answering a different question than the one it is scored
on -- which, like every timing bug in this repo, would show up as a *better* score
rather than a crash.

These tests rebuild the slices from a toy panel using exactly the expression
`model_lab/prepare_data.py` uses, and assert the identity holds.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from demandml import config as C
from demandml.features import cumulative_target_exprs

MONTH_DAYS = 30
N_MONTHS = 6


def _toy_panel(n_sku: int = 3, n_days: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_sku):
        dates = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(n_days)]
        units = rng.poisson(4.0 + s, n_days).astype(float)
        units[rng.random(n_days) < 0.2] = 0.0
        frames.append(pl.DataFrame({C.SKU: [f"SKU-{s}"] * n_days,
                                    C.DATE: dates, C.TARGET: units}))
    return pl.concat(frames).sort([C.SKU, C.DATE])


def _with_slices(df: pl.DataFrame) -> pl.DataFrame:
    """Cumulative targets + the per-month slices, exactly as prepare_data.py builds them."""
    out = df.with_columns(cumulative_target_exprs((30, 90, 180)))
    return out.with_columns([
        pl.col("y_cum30").shift(-MONTH_DAYS * (k - 1)).over(C.SKU).alias(f"y_cum_m{k}")
        for k in range(1, N_MONTHS + 1)
    ])


def test_month_1_is_the_30_day_total():
    """Month 1 is days 1..30, which is the definition of y_cum30."""
    d = _with_slices(_toy_panel()).drop_nulls(["y_cum30", "y_cum_m1"])
    assert len(d) > 0
    assert (d["y_cum_m1"] - d["y_cum30"]).abs().max() == 0.0


def test_first_three_months_sum_to_the_90_day_total():
    d = _with_slices(_toy_panel()).drop_nulls(
        ["y_cum90", "y_cum_m1", "y_cum_m2", "y_cum_m3"])
    assert len(d) > 0
    s = d["y_cum_m1"] + d["y_cum_m2"] + d["y_cum_m3"]
    assert (s - d["y_cum90"]).abs().max() == 0.0


def test_six_months_sum_to_the_180_day_total():
    cols = ["y_cum180"] + [f"y_cum_m{k}" for k in range(1, N_MONTHS + 1)]
    d = _with_slices(_toy_panel()).drop_nulls(cols)
    assert len(d) > 0
    s = sum(d[f"y_cum_m{k}"] for k in range(1, N_MONTHS + 1))
    assert (s - d["y_cum180"]).abs().max() == 0.0


def test_slices_never_reach_backwards():
    """A slice must be built only from days AFTER its origin.

    Rebuild month k by hand from the raw units of one SKU and compare. If any slice
    expression accidentally included the origin day or looked behind it, this fails.
    """
    df = _with_slices(_toy_panel(n_sku=1, n_days=400))
    u = df[C.TARGET].to_numpy()
    for k in (1, 3, 6):
        lo = MONTH_DAYS * (k - 1)
        col = df[f"y_cum_m{k}"].to_numpy()
        for t in (0, 50, 120):
            expected = u[t + 1 + lo: t + 1 + lo + MONTH_DAYS].sum()
            assert col[t] == expected, f"month {k} at row {t}"


def test_slice_targets_are_banned_from_the_feature_matrix():
    """`y_cum_m*` starts with 'y_cum', so the existing ban must already catch it."""
    from demandml.features import feature_columns

    df = _with_slices(_toy_panel())
    feats = feature_columns(df.lazy())
    for k in range(1, N_MONTHS + 1):
        assert f"y_cum_m{k}" not in feats
