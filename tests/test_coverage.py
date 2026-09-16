"""Catalogue coverage: which SKUs the order plan does not answer for, and why.

The rule these tests pin exists because the omission is silent. A SKU whose panel rows
stopped is absent from the order plan rather than present with a zero, and "not in the
file" reads to a planner as "nothing to do". Most of those SKUs are genuinely finished --
but a SKU that sold steadily while mostly out of stock and then vanished from the panel
is the opposite of finished.

The branch ORDER is the rule. `starved` has to fire before `dormant`, or the one class
that represents unmet demand gets swallowed by the one that means "nothing to do here".
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

import coverage as COV  # noqa: E402


def frame(rows: list[dict]) -> pl.DataFrame:
    """A classification input with every column `apply_rule` needs."""
    base = {"units_recent": 0.0, "on_hand": 0.0, "inbound": 0.0, "listing": None}
    return pl.DataFrame([{**base, **r} for r in rows],
                        schema_overrides={"listing": pl.String})


def klass_of(rows: list[dict], planned: set[str] | None = None) -> dict[str, str]:
    got = COV.apply_rule(frame(rows), planned or set())
    return dict(zip(got["sku"].to_list(), got["klass"].to_list()))


# --------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------

def test_sold_recently_no_stock_nothing_inbound_is_starved():
    """The class that matters. This is demand the panel measured and nobody could meet."""
    assert klass_of([{"sku": "A", "units_recent": 500.0}])["A"] == "starved"


def test_sold_recently_no_stock_but_inbound_is_incoming():
    """Real demand, but replenishment is already moving -- a different decision."""
    assert klass_of([{"sku": "A", "units_recent": 500.0, "inbound": 200.0}])["A"] \
        == "incoming"


def test_has_stock_and_is_not_selling_is_dormant():
    """Stock on the shelf and no sales is a listing or pricing problem. Ordering more
    units would make it worse, so this must not come out as `starved`."""
    assert klass_of([{"sku": "A", "on_hand": 400.0}])["A"] == "dormant"


def test_an_active_listing_with_nothing_at_all_is_dormant_not_discontinued():
    assert klass_of([{"sku": "A", "listing": "Active"}])["A"] == "dormant"


def test_no_sales_no_stock_nothing_inbound_inactive_is_discontinued():
    assert klass_of([{"sku": "A", "listing": "Inactive"}])["A"] == "discontinued"
    assert klass_of([{"sku": "A"}])["A"] == "discontinued"


def test_starved_beats_dormant_when_a_sku_is_both_selling_and_empty():
    """Branch order. A SKU with recent demand and no stock is starved even though an
    active listing would otherwise make it dormant -- if this inverts, the one actionable
    class disappears into the one that means 'leave it alone'."""
    got = klass_of([{"sku": "A", "units_recent": 50.0, "listing": "Active"}])
    assert got["A"] == "starved"


def test_planned_wins_over_everything():
    """A SKU already in the plan is answered for, whatever its stock looks like."""
    got = klass_of([{"sku": "A", "units_recent": 500.0}], planned={"A"})
    assert got["A"] == "planned"


def test_every_sku_lands_in_exactly_one_known_class():
    rows = [
        {"sku": "planned", "units_recent": 10.0},
        {"sku": "starved", "units_recent": 10.0},
        {"sku": "incoming", "units_recent": 10.0, "inbound": 5.0},
        {"sku": "dormant", "on_hand": 10.0},
        {"sku": "discontinued"},
    ]
    got = COV.apply_rule(frame(rows), {"planned"})
    assert got.height == len(rows)
    assert got["klass"].null_count() == 0
    assert set(got["klass"].to_list()) <= set(COV.CLASSES)
    assert dict(zip(got["sku"], got["klass"])) == {s["sku"]: s["sku"] for s in rows}


# --------------------------------------------------------------------------------------
# Against the real build, when it is present
# --------------------------------------------------------------------------------------

_HAVE_PANEL = COV.PANEL.exists()
needs_panel = pytest.mark.skipif(not _HAVE_PANEL, reason="no built panel")


@needs_panel
def test_the_classification_covers_the_whole_catalogue():
    """Nothing may fall between the classes -- that is how SKUs go missing silently in
    the first place."""
    origin = pl.scan_parquet(COV.PANEL).select(pl.col("date").max()).collect().item()
    df = COV.classify(origin)
    panel_skus = (pl.scan_parquet(COV.PANEL).select("sku").unique()
                  .collect(engine="streaming").height)
    assert df.height == panel_skus
    assert df["klass"].null_count() == 0


@needs_panel
def test_a_starved_sku_has_demand_and_no_way_to_meet_it():
    origin = pl.scan_parquet(COV.PANEL).select(pl.col("date").max()).collect().item()
    starved = COV.classify(origin).filter(pl.col("klass") == "starved")
    if not starved.height:
        pytest.skip("nothing starved on this build")
    assert (starved["units_recent"] > 0).all()
    assert (starved["on_hand"] == 0).all()
    assert (starved["inbound"] == 0).all()


@needs_panel
def test_no_planned_sku_is_also_reported_as_a_gap():
    """The plan and the not-covered file must partition the catalogue, not overlap it."""
    origin = pl.scan_parquet(COV.PANEL).select(pl.col("date").max()).collect().item()
    df = COV.classify(origin)
    planned = set(df.filter(pl.col("klass") == "planned")["sku"])
    gaps = set(df.filter(pl.col("klass") != "planned")["sku"])
    assert not (planned & gaps)


# --------------------------------------------------------------------------------------
# The plan / review split
# --------------------------------------------------------------------------------------

_ORDERS = Path(__file__).resolve().parents[1] / "dev" / "orders"


def _latest(pattern: str):
    got = sorted(_ORDERS.glob(pattern)) if _ORDERS.exists() else []
    return got[-1] if got else None


def test_a_recovered_sku_never_lands_in_the_plan():
    """`--recover` forecasts SKUs from an origin months old. Those are a weaker claim
    than a fresh one and belong in the review file, not merged into the quantities a
    planner signs off. Routing is on the explicit `recovered` flag, NOT on staleness --
    open_origins keeps a short tail, so an ordinary quiet SKU legitimately carries an
    origin a few weeks old and still belongs in the plan."""
    plan = _latest("order_plan_*[0-9].csv")
    if plan is None:
        pytest.skip("no order plan built")
    df = pl.read_csv(plan)
    if "recovered" not in df.columns:
        pytest.skip("plan predates the recover flag")
    assert not df["recovered"].any(), "a recovered SKU reached the plan"

    review = plan.with_name(plan.stem + "_review.csv")
    if review.exists():
        r = pl.read_csv(review)
        assert r["recovered"].all()
        assert not set(r["sku"]) & set(df["sku"]), "a SKU is in both files"
        assert (r["origin_stale_days"] > 0).all()
