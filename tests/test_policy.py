"""The order policy: the coverage window, the multiplier map, and the null that is not zero.

`dev/policy.py` is the only file in this project whose output someone spends money on,
so the failure modes that matter here are the quiet ones:

  1. The multiplier map could disagree with the cost board. The map exists to give a
     multiplier at quantiles the cost board never computed; the only evidence it is
     entitled to do that is that it REPRODUCES the cost board where they overlap. That
     is `test_ratio_curve_reproduces_the_grid_optimum`, and it is the load-bearing test
     in this file -- it catches a version weighting by actual instead of predicted,
     which over-orders the whole catalogue.
  2. The cumulative curve could go backwards. `forecast.py` picks the best model PER
     HORIZON, so the 180-day number can come from a different model than the 90-day one
     and nothing makes them agree. Interpolating a decreasing cumulative curve produces
     negative demand for a stretch of days and no error.
  3. A SKU with no stock record could be treated as empty. Polars evaluates `NaN > 0` as
     TRUE, so the obvious NaN sentinel silently turns every unknown-stock SKU into one
     to buy a full window of cover for. That is the most expensive single mistake
     available in this codebase and it has no symptom.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dev"))

import common  # noqa: E402
import policy as P  # noqa: E402
from demandml import config as C  # noqa: E402

RNG = np.random.default_rng(11)


# ------------------------------------------------------------------ the coverage window

def test_interpolation_hits_the_knots_exactly():
    cum = np.array([[10.0, 40.0, 70.0], [5.0, 5.0, 5.0]])
    for i, h in enumerate(common.HORIZONS):
        got, regime = P.interpolate_cumulative(cum, h)
        assert (regime == "interpolated").all()
        assert got == pytest.approx(cum[:, i])


def test_interpolation_is_monotone_in_the_window():
    cum = np.array([[10.0, 40.0, 70.0]])
    vals = [P.interpolate_cumulative(cum, w)[0][0] for w in range(1, 181)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_non_monotone_input_is_repaired_not_propagated():
    """A 180-day total below the 90-day one is two models disagreeing, not a forecast.
    Repaired with a running max -- never by interpolating through it."""
    cum = np.array([[100.0, 300.0, 250.0]])          # 180d < 90d
    got, _ = P.interpolate_cumulative(cum, 180)
    assert got[0] == pytest.approx(300.0)
    mid, _ = P.interpolate_cumulative(cum, 135)
    assert 300.0 - 1e-9 <= mid[0] <= 300.0 + 1e-9


def test_window_between_knots_is_a_straight_line():
    cum = np.array([[30.0, 90.0, 180.0]])            # 1 unit/day
    got, _ = P.interpolate_cumulative(cum, 105)
    assert got[0] == pytest.approx(105.0)


def test_beyond_the_last_horizon_extrapolates_flat_and_says_so():
    cum = np.array([[30.0, 90.0, 180.0]])            # final segment = 1 unit/day
    got, regime = P.interpolate_cumulative(cum, 240)
    assert (regime == "extrapolated").all()
    assert got[0] == pytest.approx(240.0)


def test_each_sku_gets_its_own_coverage_window():
    """The lead-time file has one row per SKU, so exposure is per SKU -- every row may be
    the same supplied constant, but a per-SKU figure need not be. Passing one window per SKU
    must read each SKU off ITS OWN window -- this is the bug that made --lead-time-file
    change the CSV column while leaving the demand every order was sized from unchanged."""
    cum = np.array([[30.0, 90.0, 180.0],             # 1 unit/day
                    [60.0, 180.0, 360.0]])           # 2 units/day
    got, regime = P.interpolate_cumulative(cum, np.array([105.0, 45.0]))
    assert got[0] == pytest.approx(105.0)            # SKU 0 at 105 d
    assert got[1] == pytest.approx(90.0)             # SKU 1 at 45 d
    assert (regime == "interpolated").all()


def test_mixed_windows_flag_only_the_extrapolated_skus():
    """A catalogue where some SKUs are inside the knots and some past them must report
    the regime per SKU, not collapse to one label -- the share past 180 d is exactly the
    share of the order book resting on an unvalidated flat-rate extension."""
    cum = np.array([[30.0, 90.0, 180.0], [30.0, 90.0, 180.0]])
    got, regime = P.interpolate_cumulative(cum, np.array([120.0, 240.0]))
    assert list(regime) == ["interpolated", "extrapolated"]
    assert got == pytest.approx([120.0, 240.0])


def test_negative_forecasts_are_clipped_before_interpolation():
    cum = np.array([[-5.0, 90.0, 180.0]])
    got, _ = P.interpolate_cumulative(cum, 15)
    assert got[0] >= 0.0


# ------------------------------------------------------------------ the multiplier map

def _grid_optimum(y, p, cu, co):
    ks = np.round(np.arange(0.80, 2.01, 0.02), 2)
    cost = [np.sum(cu * np.maximum(y - k * p, 0) + co * np.maximum(k * p - y, 0)) for k in ks]
    return float(ks[int(np.argmin(cost))])


@pytest.mark.parametrize("cu", [1.0, 2.0, 4.0, 5.0])
def test_ratio_curve_reproduces_the_grid_optimum(cu):
    """THE test in this file. The newsvendor first-order condition says the optimal
    multiplier is the cu/(cu+co) quantile of `actual/predicted` weighted by PREDICTED.
    If this drifts, the map is answering a different question than the cost board and
    every per-SKU multiplier derived from it is wrong by the same factor.

    Weighting by `actual` instead passes none of these.
    """
    co = 1.0
    p = RNG.gamma(3.0, 40.0, 20000)
    y = p * RNG.lognormal(-0.05, 0.45, 20000)
    probs = np.linspace(0.01, 0.99, 99)
    mults = P._wquantile(y / p, p, probs)
    got = float(np.interp(cu / (cu + co), probs, mults))
    assert got == pytest.approx(_grid_optimum(y, p, cu, co), abs=0.02)  # one grid step


def test_weighting_by_actual_is_measurably_wrong():
    """The mirror of the above, so the test suite records WHY the weight is what it is
    rather than leaving it as a style choice someone later 'simplifies'."""
    cu, co = 4.0, 1.0
    q = cu / (cu + co)
    p = RNG.gamma(3.0, 40.0, 20000)
    y = p * RNG.lognormal(-0.05, 0.45, 20000)
    probs = np.linspace(0.01, 0.99, 99)
    opt = _grid_optimum(y, p, cu, co)
    by_pred = float(np.interp(q, probs, P._wquantile(y / p, p, probs)))
    by_act = float(np.interp(q, probs, P._wquantile(y / p, y, probs)))
    assert abs(by_pred - opt) < abs(by_act - opt)


def test_multiplier_is_monotone_in_the_service_level():
    """A SKU that wants more cover must never get a smaller multiplier."""
    probs = np.linspace(0.01, 0.99, 99)
    mults = np.sort(RNG.uniform(0.5, 2.5, 99))
    q = np.linspace(0.05, 0.95, 40)
    k = P.multiplier_for(q, probs, mults)
    assert np.all(np.diff(k) >= -1e-12)


def test_multiplier_is_clipped_to_the_searched_band():
    probs = np.linspace(0.01, 0.99, 99)
    mults = np.linspace(0.01, 50.0, 99)          # a pathological error distribution
    k = P.multiplier_for(np.array([0.01, 0.5, 0.99]), probs, mults)
    assert np.all(k >= 0.80) and np.all(k <= 2.00)


# ------------------------------------------------------------------ null is not zero

def test_polars_treats_nan_as_greater_than_zero():
    """Not a test of our code -- a test of the assumption underneath it. If a future
    Polars makes `NaN > 0` false, the null handling in policy.py is merely redundant
    rather than load-bearing, and whoever reads this test should know which it is."""
    d = pl.DataFrame({"x": [float("nan"), 1.0, 0.0]})
    assert d.filter(pl.col("x") > 0).height == 2, "NaN no longer compares as > 0"


def test_unknown_stock_yields_null_not_zero_and_is_not_counted_as_an_order():
    """The expensive mistake: a SKU with no inventory record must not be ordered for.
    Reproduces the shape policy.py builds, including the dtype."""
    known = np.array([True, True, False, False])
    qty_all = np.array([100.0, 0.0, 250.0, 700.0])
    out = pl.DataFrame({"sku": list("ABCD")}).with_columns(
        order_qty=pl.when(pl.Series("k", known))
        .then(pl.Series("v", qty_all)).otherwise(None).cast(pl.Int64))
    assert out["order_qty"].to_list() == [100, 0, None, None]
    assert out.filter(pl.col("order_qty") > 0).height == 1
    assert out["order_qty"].null_count() == 2


def test_inventory_loader_marks_missing_skus_unknown():
    with_file = P.load_inventory(None, ["A", "B"], assume_zero=False)
    assert with_file["stock_known"].to_list() == [False, False]
    assumed = P.load_inventory(None, ["A", "B"], assume_zero=True)
    assert assumed["stock_known"].to_list() == [True, True]


def test_inventory_file_partial_coverage(tmp_path):
    f = tmp_path / "inv.csv"
    f.write_text("sku,on_hand,on_order\nA,10,5\n")
    inv = P.load_inventory(f, ["A", "B"], assume_zero=False)
    assert inv["stock_known"].to_list() == [True, False]
    assert inv["on_hand"].to_list() == [10.0, 0.0]
    assert inv["on_order"].to_list() == [5.0, 0.0]


def test_inventory_file_sums_duplicate_rows(tmp_path):
    """One SKU across two warehouses is two rows, not a conflict."""
    f = tmp_path / "inv.csv"
    f.write_text("sku,on_hand,on_order\nA,10,5\nA,7,0\n")
    inv = P.load_inventory(f, ["A"], assume_zero=False)
    assert inv["on_hand"].to_list() == [17.0]
    assert inv["on_order"].to_list() == [5.0]


def test_lead_time_file_overrides_the_default(tmp_path):
    f = tmp_path / "lt.csv"
    f.write_text("sku,lead_time_days\nA,140\n")
    lt = P.load_lead_times(f, 90, ["A", "B"])
    assert lt["lead_time_days"].to_list() == [140, 90]
    assert lt["lead_time_source"].to_list() == ["file", "default"]


# ------------------------------------------------------------------ catalogue coverage

def _mini_panel(tmp_path, monkeypatch, rows):
    """A panel with only the columns `coverage.classify` truly requires, and no stock
    extract -- which is also the degraded case the classifier has to survive."""
    import coverage as COV
    panel = tmp_path / "panel_daily.parquet"
    pl.DataFrame(rows, schema={C.SKU: pl.String, C.DATE: pl.Date, C.TARGET: pl.Float64},
                 orient="row").write_parquet(panel)
    monkeypatch.setattr(COV, "PANEL", panel)
    monkeypatch.setattr(COV, "INVENTORY", tmp_path / "no_inventory.parquet")
    return panel


def test_coverage_gap_finds_skus_with_no_forecast_and_says_why(tmp_path, monkeypatch):
    """A SKU that went quiet drops out of `open_origins.parquet` entirely, so it is
    absent from the order plan rather than present with a zero. 'Not in the file' reads
    to a planner as 'nothing to do', which is why the omission has to be reported -- and
    reported WITH A REASON, since 'ran out of stock' and 'was discontinued' call for
    opposite decisions."""
    origin = dt.date(2026, 8, 10)
    _mini_panel(tmp_path, monkeypatch, [
        ("PLANNED", origin - dt.timedelta(days=5), 10.0),
        ("QUIET", origin - dt.timedelta(days=5), 40.0),    # sold recently, not planned
        ("DEAD", origin - dt.timedelta(days=400), 999.0),  # long gone
        ("FUTURE", origin + dt.timedelta(days=3), 50.0),   # after the origin, ignored
    ])
    gap = P.coverage_gap(["PLANNED"], origin)
    by = dict(zip(gap[C.SKU].to_list(), gap["klass"].to_list()))
    assert "PLANNED" not in by
    # Sold inside the window, nothing on hand, nothing inbound -> unmet demand.
    assert by["QUIET"] == "starved"
    # No sales in 180 days and nothing to sell -> correctly absent from the plan.
    assert by["DEAD"] == "discontinued"
    assert gap.filter(pl.col(C.SKU) == "QUIET")["units_recent"].to_list() == [40.0]


def test_coverage_gap_is_empty_when_everything_is_planned(tmp_path, monkeypatch):
    origin = dt.date(2026, 8, 10)
    _mini_panel(tmp_path, monkeypatch, [("A", origin - dt.timedelta(days=1), 3.0)])
    assert P.coverage_gap(["A"], origin).height == 0


def test_coverage_gap_tolerates_a_missing_panel(tmp_path, monkeypatch):
    """A coverage check is a nice-to-have; it must never stop an order plan being written."""
    import coverage as COV
    monkeypatch.setattr(COV, "PANEL", tmp_path / "nope.parquet")
    assert P.coverage_gap(["A"], dt.date(2026, 8, 10)).height == 0
