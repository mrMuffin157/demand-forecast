"""Two files whose whole value is a discipline that is easy to break silently.

`dev/calibration.py` compares five ways of choosing a forecast multiplier, and four of
them have more parameters than the default global one. A per-SKU multiplier fitted and scored
on the same rows can drive cost to near zero and prove nothing. The only thing separating
that from a result is that fold 3 is never touched during fitting -- so that is what is
tested here, by handing the arms data whose fold 3 is deliberately poisoned. An arm that
peeks will produce a different multiplier; an arm that does not cannot.

`dev/track.py` scores issued forecasts against the panel. Its one arithmetic trap is the
window boundary: `units_lag_0` is the ORIGIN DAY's own sales and is a feature, so counting
the origin day in the actual would score the model on a day it was shown. That is the same
off-by-one the project's timing convention exists to prevent, and it makes the
tracked scores look better, which is why it needs a test rather than a comment.
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

import calibration as CAL  # noqa: E402
import track as T  # noqa: E402
from demandml import config as C  # noqa: E402

RNG = np.random.default_rng(23)


def fake_preds(n_sku: int = 120, poison: bool = False, seed: int = 5) -> pl.DataFrame:
    """Three folds of predictions. With `poison=True`, fold 3 is scaled so hard that any
    arm which looked at it would have to choose a wildly different multiplier.

    The generator is seeded PER CALL, not drawn from the module-level one. The isolation
    test compares a clean build against a poisoned build and needs them to differ in the
    poison and in nothing else; a shared generator would give the two calls different
    random draws throughout and the comparison would fail for every arm, cheating or not.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for f in (1, 2, 3):
        for i in range(n_sku):
            for wk in range(8):
                p = float(rng.gamma(3.0, 30.0))
                y = p * float(rng.lognormal(-0.08, 0.35))
                if poison and f == 3:
                    y *= 6.0
                rows.append({
                    "sku": f"S{i:03d}",
                    "date": dt.date(2025, 1, 1) + dt.timedelta(days=7 * (8 * (f - 1) + wk)),
                    "h": 30, "fold": f, "actual": y, "predicted": p,
                    "sku_brand": f"B{i % 5}",
                })
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ the honesty rule

@pytest.mark.parametrize("arm_name", ["global", "by line", "by volume", "per SKU"])
def test_arms_cannot_see_fold_three(arm_name):
    """Fit each arm on folds 1-2 of clean data and of data whose fold 3 is 6x. The fitted
    multipliers must be IDENTICAL -- the poison is only in rows the arm is forbidden to
    read. Any difference means the arm is selecting on the test fold and every number it
    reports is inflated."""
    fn = next(f for n, _, f in CAL.ARMS if n == arm_name)
    clean, dirty = fake_preds(), fake_preds(poison=True)
    tr_c = clean.filter(pl.col("fold") < 3)
    tr_d = dirty.filter(pl.col("fold") < 3)
    te = clean.filter(pl.col("fold") == 3)

    apply_c, _ = fn(tr_c, 4.0, 1.0)
    apply_d, _ = fn(tr_d, 4.0, 1.0)
    assert np.allclose(apply_c(te), apply_d(te)), f"{arm_name} is reading fold 3"


def test_poison_fixture_would_actually_move_a_multiplier():
    """Guards the guard. If fold 3's poison were too weak to change any fitted k, the
    test above would pass for every arm including a cheating one."""
    clean, dirty = fake_preds(), fake_preds(poison=True)
    k_clean = CAL.best_k(clean["actual"].to_numpy(), clean["predicted"].to_numpy(), 4.0, 1.0)
    k_dirty = CAL.best_k(dirty["actual"].to_numpy(), dirty["predicted"].to_numpy(), 4.0, 1.0)
    assert abs(k_clean - k_dirty) > 0.1


def test_best_k_finds_the_cost_minimum_on_the_grid():
    y = RNG.gamma(2.0, 50.0, 4000)
    p = y * RNG.uniform(0.6, 1.3, 4000)
    k = CAL.best_k(y, p, 4.0, 1.0)
    here = CAL.cost(y, p * k, 4.0, 1.0)
    for other in (k - 0.02, k + 0.02):
        assert CAL.cost(y, p * other, 4.0, 1.0) >= here - 1e-12


def test_asymmetric_cost_pushes_the_multiplier_up():
    """Sanity on the direction: a stockout-averse ratio must want MORE stock than a
    symmetric one, or the whole cost story is wired backwards."""
    y = RNG.gamma(2.0, 50.0, 4000)
    p = y * RNG.uniform(0.6, 1.3, 4000)
    assert CAL.best_k(y, p, 5.0, 1.0) >= CAL.best_k(y, p, 1.0, 1.0)


def test_thin_groups_fall_back_to_the_global_k():
    """A product line with a handful of rows must not get its own multiplier fitted from
    them -- that is how a 4-row segment ends up with k = 2.0."""
    d = fake_preds().filter(pl.col("fold") < 3)
    thin = d.head(20).with_columns(sku_brand=pl.lit("TINY"))
    tr = pl.concat([d, thin])
    apply, info = CAL.arm_by_line(tr, 4.0, 1.0)
    te = tr.head(1).with_columns(sku_brand=pl.lit("TINY"))
    assert apply(te)[0] == pytest.approx(info["k_global"])


def test_unseen_level_at_serving_time_falls_back():
    """A brand-new product line on the day you serve has no fitted k. It must get the
    global one, not an exception and not a zero."""
    tr = fake_preds().filter(pl.col("fold") < 3)
    apply, info = CAL.arm_by_line(tr, 4.0, 1.0)
    te = tr.head(1).with_columns(sku_brand=pl.lit("BRAND_NEW"))
    assert apply(te)[0] == pytest.approx(info["k_global"])


# ------------------------------------------------------------------ the window boundary

def _panel(rows) -> pl.LazyFrame:
    return pl.DataFrame(rows, schema={C.SKU: pl.String, C.DATE: pl.Date,
                                      C.TARGET: pl.Float64}, orient="row").lazy()


def test_actuals_exclude_the_origin_day():
    """The origin day's units are a FEATURE (`units_lag_0`). Counting them in the actual
    would score the model on a day it was shown, and would make every tracked number look
    better than it is."""
    origin = dt.date(2026, 1, 10)
    p = _panel([
        ("A", origin, 1000.0),                        # origin day -- must NOT count
        ("A", origin + dt.timedelta(days=1), 5.0),
        ("A", origin + dt.timedelta(days=30), 7.0),   # last day of a 30d window
        ("A", origin + dt.timedelta(days=31), 900.0),  # past the window -- must NOT count
    ])
    got = T.actuals(["A"], origin, 30, p)
    assert got["actual"].to_list() == [12.0]


def test_actuals_fill_missing_skus_with_zero():
    """A SKU that sold nothing is a zero, not a null -- a null would silently drop it
    from the scored population and flatter the result."""
    origin = dt.date(2026, 1, 10)
    p = _panel([("A", origin + dt.timedelta(days=1), 5.0)])
    got = T.actuals(["A", "B"], origin, 30, p).sort(C.SKU)
    assert got["actual"].to_list() == [5.0, 0.0]


def test_metrics_match_their_definitions():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 15.0, 30.0])
    m = T.metrics(y, p, 4.0, 1.0)
    assert m["units"] == 60.0
    assert m["wape"] == pytest.approx(7.0 / 60.0)
    assert m["bias"] == pytest.approx(-3.0 / 60.0)
    assert m["fill"] == pytest.approx(1.0 - 5.0 / 60.0)
    assert m["cost"] == pytest.approx((4.0 * 5.0 + 1.0 * 2.0) / 60.0)


def test_metrics_on_zero_demand_are_nan_not_zero():
    """A window with no demand has no defined WAPE. Returning 0 would report a perfect
    score for a SKU nobody bought."""
    m = T.metrics(np.zeros(5), np.ones(5), 4.0, 1.0)
    assert all(np.isnan(v) for v in m.values())
