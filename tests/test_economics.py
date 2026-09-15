"""The per-SKU cost ratios, and the two properties that make them safe to buy against.

`dev/economics.py` replaces one global CU:CO with one ratio per SKU, derived from prices.
That is more information but also more ways to be wrong, and two of them would be
invisible:

  1. The three assumptions in that file -- holding rate, recapture, markdown risk -- are
     guesses that are hard to source. If the default policy depended on them, the whole
     per-SKU story would rest on made-up numbers. It must not, and `test_anchoring_*`
     below pins that: in anchored mode they cancel exactly.
  2. Order rounding must never round DOWN. Rounding a net requirement down to a carton
     boundary re-introduces exactly the stockout the multiplier was added to remove, and
     it would look like a rounding detail rather than a service-level decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import json

import numpy as np
import pytest
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dev"))

import common  # noqa: E402
import economics as E  # noqa: E402

RNG = np.random.default_rng(7)


def fake_products(n: int = 400) -> pl.DataFrame:
    """A synthetic product master spanning the awkward cases: missing prices, zero
    prices, a sale price below cost, no MOQ, no carton."""
    cost = RNG.uniform(1.0, 60.0, n)
    price = cost * RNG.uniform(1.2, 6.0, n)
    price[:20] = 0.0                       # no sale price at all
    cost[20:30] = 0.0                      # no cost at all
    price[30:40] = cost[30:40] * 0.5       # sale price below cost -> unusable margin
    return pl.DataFrame({
        "sku": [f"S{i:04d}" for i in range(n)],
        "average_cost": cost,
        "default_cost": cost,
        "po_price": cost * 1.05,
        "sale_price": price,
        "default_sale_price": np.zeros(n),
        "moq": RNG.choice([0, 0, 50, 100, 500], n),
        "carton_size": [str(int(v)) for v in RNG.choice([0, 1, 6, 12, 48], n)],
    })


# ------------------------------------------------------------------ the anchoring property

@pytest.mark.parametrize("h", [30, 90, 180])
def test_anchored_median_lands_on_the_configured_ratio(h):
    """The whole point of anchoring: the prices set the SPREAD, the configured ratio keeps
    the LEVEL. If the median SKU drifts off `common.CU:CO`, the file has quietly overruled a
    decision that was never the model's to make.

    Asserted in LOG space, which is where the rescale happens and therefore where the
    guarantee is exact. In raw space an even-length sample is off by a rounding error:
    `exp(median(log x))` is the geometric mean of the two middle values and `median(x)`
    the arithmetic one, so they differ by ~1e-5 relative on this fixture and agree
    exactly whenever the count is odd. The looser raw-space check below is there so a
    real drift -- which would be percent-scale, not 1e-5 -- still fails the test.
    """
    d = E.sku_ratios(h, anchor=True, product=fake_products())
    usable = d.filter(pl.col("ratio_raw").is_not_null())
    cu, co = common.cost_ratio()
    logs = np.log(usable["ratio"].to_numpy())
    assert float(np.median(logs)) == pytest.approx(np.log(cu / co), abs=1e-9)
    assert float(np.median(usable["ratio"].to_numpy())) == pytest.approx(cu / co, rel=1e-3)


@pytest.fixture(autouse=True)
def _unadopted(monkeypatch):
    """Every test in this file describes the DEFAULT configuration, which does not adopt
    measured per-SKU holding rates. Adopting a per-SKU storage file is opt-in and
    changes these properties materially -- see the adoption tests at the bottom, which
    turn it on deliberately."""
    monkeypatch.setattr(E, "HOLDING_FILE",
                        E.LAB / "tuned_params" / "__not_adopted__.json")


def test_anchoring_cancels_the_unsourceable_assumptions():
    """HOLDING_RATE, RECAPTURE and MARKDOWN_RISK are guesses. In anchored mode they are
    global constants that scale every SKU's cu and co identically, so the rescale divides
    them out. Triple all three and nothing may move.

    This is the test that lets the docstring claim the default policy does not depend on
    numbers that are hard to source. If it ever fails, that claim is false and the README must
    stop making it."""
    prod = fake_products()
    before = E.sku_ratios(90, anchor=True, product=prod)["ratio"].to_numpy()
    old = (E.HOLDING_RATE, E.RECAPTURE, E.MARKDOWN_RISK)
    try:
        E.HOLDING_RATE, E.RECAPTURE, E.MARKDOWN_RISK = 0.60, 0.05, 0.30
        after = E.sku_ratios(90, anchor=True, product=prod)["ratio"].to_numpy()
    finally:
        E.HOLDING_RATE, E.RECAPTURE, E.MARKDOWN_RISK = old
    assert np.max(np.abs(before - after)) < 1e-9


def test_anchored_ratio_is_horizon_invariant():
    """A corollary of the above, asserted separately because it LOOKS like a bug when
    you first see it: the horizon term is also a global constant, so it cancels too.
    Anyone 'fixing' this needs a per-SKU holding rate first."""
    prod = fake_products()
    a = E.sku_ratios(30, product=prod)["ratio"].to_numpy()
    b = E.sku_ratios(180, product=prod)["ratio"].to_numpy()
    assert np.max(np.abs(a - b)) < 1e-9


def test_raw_mode_does_depend_on_the_assumptions():
    """The mirror image: --raw is the mode where the guesses are real. If this passes
    trivially, the test above is passing for the wrong reason."""
    prod = fake_products()
    before = E.raw_ratios(90, prod)["ratio_raw"].to_numpy()
    old = E.HOLDING_RATE
    try:
        E.HOLDING_RATE = 0.60
        after = E.raw_ratios(90, prod)["ratio_raw"].to_numpy()
    finally:
        E.HOLDING_RATE = old
    ok = np.isfinite(before) & np.isfinite(after)
    assert np.max(np.abs(before[ok] - after[ok])) > 1e-6


def test_missing_prices_fall_back_to_the_global_ratio():
    """Absent information is not a reason to move a SKU off the configured ratio in
    either direction."""
    d = E.sku_ratios(90, product=fake_products())
    cu, co = common.cost_ratio()
    unusable = d.filter(pl.col("ratio_raw").is_null())
    assert unusable.height > 0, "the fixture should contain unusable SKUs"
    assert np.allclose(unusable["ratio"].to_numpy(), cu / co)


def test_ratios_stay_inside_the_supportable_band():
    d = E.sku_ratios(90, product=fake_products())
    r = d["ratio"].to_numpy()
    assert r.min() >= E.ANCHORED_FLOOR - 1e-9
    assert r.max() <= E.ANCHORED_CAP + 1e-9


def test_quantile_matches_the_ratio():
    """q must be the critical ratio of the ratio it sits next to, or the two columns
    disagree about the same SKU."""
    d = E.sku_ratios(90, product=fake_products())
    r, q = d["ratio"].to_numpy(), d["q"].to_numpy()
    assert np.allclose(q, r / (1.0 + r))


def test_higher_margin_means_more_cover():
    """The direction that justifies the whole file. Two SKUs at the same unit cost, one
    with twice the margin: the fatter one must be covered further into the tail."""
    prod = pl.DataFrame({
        "sku": ["THIN", "FAT"],
        "average_cost": [10.0, 10.0], "default_cost": [10.0, 10.0],
        "po_price": [10.0, 10.0],
        "sale_price": [12.0, 30.0], "default_sale_price": [0.0, 0.0],
        "moq": [0, 0], "carton_size": ["1", "1"],
    })
    d = E.sku_ratios(90, product=prod).sort("sku")
    q = dict(zip(d["sku"].to_list(), d["q"].to_list()))
    assert q["FAT"] > q["THIN"]


# ------------------------------------------------------------------ order rounding

def test_rounding_never_orders_less_than_needed():
    """The one-way property. Rounding down to a carton boundary would give back exactly
    the cover the multiplier was added to buy, and would look like a rounding detail."""
    qty = RNG.uniform(0, 5000, 2000)
    moq = RNG.choice([1, 50, 100, 500], 2000)
    carton = RNG.choice([1, 6, 12, 48], 2000)
    out = E.round_order(qty, moq, carton)
    assert np.all(out >= np.ceil(qty))


def test_rounding_respects_moq_and_carton():
    qty = RNG.uniform(1, 5000, 2000)
    moq = RNG.choice([1, 50, 100, 500], 2000)
    carton = RNG.choice([1, 6, 12, 48], 2000)
    out = E.round_order(qty, moq, carton)
    assert np.all(out >= moq)
    assert np.all(out % carton == 0)


def test_zero_stays_zero():
    """'Buy nothing' has to survive rounding, or every dead SKU in the catalogue gets an
    MOQ-sized order."""
    out = E.round_order(np.zeros(50), np.full(50, 500), np.full(50, 12))
    assert np.all(out == 0)


def test_order_multiple_cleans_blank_and_zero():
    """0 and '' mean 'no constraint' in this export, which is arithmetically 1 -- not 0,
    which would make every order a multiple of nothing."""
    prod = pl.DataFrame({
        "sku": ["A", "B", "C"],
        "average_cost": [1.0] * 3, "default_cost": [1.0] * 3, "po_price": [1.0] * 3,
        "sale_price": [2.0] * 3, "default_sale_price": [0.0] * 3,
        "moq": [0, 100, 0], "carton_size": ["", "12", "junk"],
    })
    m = E.order_multiple(prod).sort("sku")
    assert m["moq"].to_list() == [1, 100, 1]
    assert m["carton"].to_list() == [1, 12, 1]


# ======================================================================================
# Adopting the measured per-SKU holding rates
# ======================================================================================
#
# A per-SKU storage file (`economics.HOLDING_FILE`) replaces the `HOLDING_RATE x
# unit_cost` proxy in `co` with measured storage charges per SKU. The case for it is that
# it is a change to `co` that SURVIVES anchoring -- anchoring divides out global
# constants and a per-SKU rate is not one.
#
# That is true, and it is only half the story. The same mechanism that lets the measured
# rate through also stops the three unsourceable guesses cancelling, because
# `co = A*unit_cost + B` is no longer a pure per-SKU scaling. These two tests pin both
# halves so the trade-off is a decision rather than a surprise.

def _adopt(monkeypatch, tmp_path, rates: dict[str, float]):
    p = tmp_path / "holding_per_sku.json"
    p.write_text(json.dumps({"capital_rate": 0.12, "excess_months": 3.0,
                             "min_months": 3, "storage_per_unit_month": rates}))
    monkeypatch.setattr(E, "HOLDING_FILE", p)
    return p


def test_nothing_is_adopted_by_default():
    """The default configuration. If a stray `holding_per_sku.json` appears in the repo,
    every order quantity changes and this is the only thing that would say so."""
    assert E.measured_holding() is None


def test_adopting_lets_a_per_sku_rate_through_anchoring(monkeypatch, tmp_path):
    """The point of adopting. Two SKUs identical except for storage must end up with
    different service levels -- a global constant could never do that."""
    prod = fake_products()
    # The fixture deliberately makes the first 40 SKUs unusable (no price, no cost, or a
    # sale price below cost), so pick two that actually have a margin -- a SKU with no
    # `cu` falls back to the anchor and would hide the effect being tested.
    cheap, bulky = "S0100", "S0101"
    rates = {s: 0.20 for s in prod["sku"].to_list()}
    rates[cheap], rates[bulky] = 0.01, 2.50

    base = E.sku_ratios(90, product=prod).sort("sku")
    _adopt(monkeypatch, tmp_path, rates)
    after = E.sku_ratios(90, product=prod).sort("sku")

    assert not np.allclose(base["ratio"].to_numpy(), after["ratio"].to_numpy()), \
        "adoption changed nothing -- it never reached co"

    q = dict(zip(after["sku"].to_list(), after["q"].to_list()))
    # The bulky SKU costs more to hold, so it is covered LESS far into the tail.
    assert q[bulky] < q[cheap], f"bulky {q[bulky]} should be under cheap {q[cheap]}"


def test_adopting_costs_the_cancellation_property(monkeypatch, tmp_path):
    """The price of adopting, measured rather than assumed.

    Unadopted, tripling the three guesses moves the default ratios by ~1e-15. Adopted,
    it moves them by order 1 -- so the order quantities start depending on three numbers
    that are hard to source. That is why adoption is opt-in: a measured storage rate is
    better information, but it is not free."""
    prod = fake_products()
    _adopt(monkeypatch, tmp_path, {s: 0.20 for s in prod["sku"].to_list()})

    before = E.sku_ratios(90, anchor=True, product=prod)["ratio"].to_numpy()
    old = (E.HOLDING_RATE, E.RECAPTURE, E.MARKDOWN_RISK)
    try:
        E.HOLDING_RATE, E.RECAPTURE, E.MARKDOWN_RISK = 0.60, 0.05, 0.30
        after = E.sku_ratios(90, anchor=True, product=prod)["ratio"].to_numpy()
    finally:
        E.HOLDING_RATE, E.RECAPTURE, E.MARKDOWN_RISK = old

    moved = float(np.max(np.abs(before - after)))
    assert moved > 1e-3, (
        "adopted mode still cancels the guesses -- if this ever becomes true, the "
        "warning in economics.measured_holding() should be removed")
