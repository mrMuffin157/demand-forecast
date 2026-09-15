"""The inventory cost metric, and the guards that keep it meaning what it says.

Three separate things can go wrong here and none of them announce themselves:

  1. The cost could be mis-specified -- wired up so that it does not actually reward
     over-forecasting -- and the whole re-ranking would be noise wearing a suit.
  2. The target quantile could drift away from the cost ratio, so the model aims at one
     preference while the board scores it against another.
  3. `center` could quietly return to tune.py's search space, at which point a
     WAPE-driven search silently overrules the configured cost ratio -- which is
     exactly what a WAPE search does if allowed (it picks a quantile below 0.5).

Each has a test below. The first is the definition, the other two are tripwires.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dev"))

from demandml.metrics import (  # noqa: E402
    bias, cost_from_wape_bias, critical_ratio, evaluate, fill_rate, newsvendor_cost,
    service_level_shortfall, wape,
)

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ the definition

def test_symmetric_cost_is_exactly_wape():
    """At cu == co the cost must collapse to WAPE. If it does not, the two metrics are
    measuring different things for reasons other than asymmetry, and every comparison
    between the WAPE board and the cost board is confounded."""
    y = RNG.gamma(2.0, 20.0, 5000)
    p = y * RNG.uniform(0.5, 1.6, 5000)
    assert newsvendor_cost(y, p, 1.0, 1.0) == pytest.approx(wape(y, p), rel=1e-12)


def test_only_the_ratio_matters():
    """Cost scales linearly in (cu, co), so 8:2 must rank identically to 4:1. This is
    what licenses pinning co = 1 and expressing the whole preference as cu."""
    y = RNG.gamma(2.0, 20.0, 2000)
    p = y * RNG.uniform(0.5, 1.6, 2000)
    assert newsvendor_cost(y, p, 8.0, 2.0) == pytest.approx(
        2.0 * newsvendor_cost(y, p, 4.0, 1.0), rel=1e-12)


def test_asymmetric_cost_prefers_over_forecasting():
    """The point of the whole exercise: under 4:1, forecasting 20% high must beat
    forecasting 20% low. Under 1:1 they must tie. If this fails, nothing downstream
    means anything."""
    y = RNG.gamma(2.0, 20.0, 5000)
    high, low = y * 1.2, y * 0.8
    assert newsvendor_cost(y, high, 4.0, 1.0) < newsvendor_cost(y, low, 4.0, 1.0)
    assert newsvendor_cost(y, high, 1.0, 1.0) == pytest.approx(
        newsvendor_cost(y, low, 1.0, 1.0), rel=1e-12)


@pytest.mark.parametrize("cu", [1.0, 2.0, 4.0, 5.0, 9.0])
def test_cost_is_minimised_at_the_critical_ratio(cu):
    """The theoretical claim the target quantile rests on: a constant forecast against a
    known demand distribution minimises newsvendor cost at the cu/(cu+co) quantile.

    Verified empirically rather than assumed, because `target_quantile()` is the number
    handed to the quantile model -- if the identity does not hold, the model is being
    aimed at the wrong place with full confidence.
    """
    y = RNG.gamma(2.0, 20.0, 200_000)
    grid = np.quantile(y, np.arange(0.05, 0.99, 0.01))
    costs = [newsvendor_cost(y, np.full_like(y, c), cu, 1.0) for c in grid]
    best = grid[int(np.argmin(costs))]
    assert best == pytest.approx(np.quantile(y, critical_ratio(cu, 1.0)), rel=0.05)


def test_fill_rate_is_one_minus_shortfall():
    y = RNG.gamma(2.0, 20.0, 1000)
    p = y * RNG.uniform(0.5, 1.6, 1000)
    assert fill_rate(y, p) == pytest.approx(1.0 - service_level_shortfall(y, p))


def test_evaluate_exposes_cost_and_fill():
    y = RNG.gamma(2.0, 20.0, 500)
    m = evaluate(y, y * 1.1)
    assert {"cost", "fill"} <= set(m)
    # Same defaults on both sides: this checks the bundle exposes the cost, not a ratio.
    assert m["cost"] == pytest.approx(newsvendor_cost(y, y * 1.1))


@pytest.mark.parametrize("cu", [1.0, 4.0, 9.0])
def test_cost_recovers_exactly_from_wape_and_bias(cu):
    """The identity a search relies on to be re-scored without its predictions. If it
    ever stops holding exactly, every re-selection built on it is quietly wrong."""
    y = RNG.gamma(2.0, 20.0, 5000)
    p = y * RNG.uniform(0.4, 1.8, 5000)
    assert cost_from_wape_bias(wape(y, p), bias(y, p), cu, 1.0) == pytest.approx(
        newsvendor_cost(y, p, cu, 1.0), rel=1e-12)


# ------------------------------------------------------------------ the tripwires

def test_target_quantile_follows_the_cost_ratio():
    """`common.CU` (read from assumptions.toml) is the single place the ratio lives. Everything must
    read it, so that changing it changes the system rather than one corner of it."""
    import common

    assert common.target_quantile() == pytest.approx(
        common.CU / (common.CU + common.CO))

    original = common.CU
    try:
        common.CU = 9.0
        assert common.target_quantile() == pytest.approx(0.9)
    finally:
        common.CU = original


def test_center_is_not_in_the_tuning_search_space():
    """A WAPE-driven search treats the quantile as a free way to forecast low, and given
    the chance it will pick one below 0.5. The quantile is a statement of
    what a stockout costs, not a hyper-parameter, so it must be derived from the cost
    ratio and never searched."""
    import tune

    space = tune.SPACES["lgbm_quantile"]
    assert "center" not in space, (
        "`center` is back in lgbm_quantile's search space. A WAPE search will move it "
        "below 0.5 and silently overrule the cost ratio. Derive it from "
        "common.target_quantile() instead.")


def test_cost_mode_cannot_overwrite_the_other_halves():
    """Three universes, three directories. A cost-trained model is a different model; if
    it wrote into results/ it would destroy the baseline predictions that the cost board
    compares it against."""
    import common

    dirs = [common.MODES[m] for m in ("initial", "tuned", "cost")]
    assert len({d[0] for d in dirs}) == 3, "results directories collide"
    assert len({d[1] for d in dirs}) == 3, "figure directories collide"
