"""The known-future override layer.

Two of these tests are the reason the file exists at all.

`test_an_event_entered_after_the_origin_is_never_applied` is the integrity rule. Without
it, re-running a historical origin would apply everything learned since, every tracked
score would improve, and the improvement would be hindsight -- the exact defect that makes
a planning table edited during the period it describes unusable as a benchmark.

`test_no_events_file_changes_nothing` is the safety rule. This layer is enabled by
default, so the no-override path has to be bit-for-bit what it was before it existed.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

import events as EV  # noqa: E402

ORIGIN = dt.date(2026, 8, 10)
SKUS = pl.Series("sku", ["ZD-104", "ZB-105", "ZB-106"])
LINES = pl.Series("sku_brand", ["ZD", "ZB", "ZB"])
BASE = np.array([100.0, 200.0, 300.0])


def ev(**kw) -> EV.Event:
    """One event, with every required field defaulted to something harmless."""
    d = dict(scope="ZD-104", start=ORIGIN + dt.timedelta(days=1),
             end=ORIGIN + dt.timedelta(days=30), kind="level", multiplier=2.0,
             units=None, probability=1.0, entered_at=ORIGIN, author="test", note="",
             row=2)
    d.update(kw)
    return EV.Event(**d)


def write(tmp_path: Path, *rows: str) -> Path:
    p = tmp_path / "events.csv"
    p.write_text(",".join(EV.COLUMNS) + "\n" + "\n".join(rows) + "\n")
    return p


# --------------------------------------------------------------------------------------
# The two rules that matter
# --------------------------------------------------------------------------------------

def test_no_events_file_changes_nothing(tmp_path):
    assert EV.load(tmp_path / "does_not_exist.csv") == []
    adj, factor, delta, n = EV.adjust(BASE, SKUS, ORIGIN, 90, [], LINES)
    assert np.array_equal(adj, BASE)
    assert np.all(factor == 1.0)
    assert np.all(delta == 0.0)
    assert np.all(n == 0)


def test_an_event_entered_after_the_origin_is_never_applied():
    """The whole point of `entered_at`. A row written tomorrow cannot help today."""
    future = ev(entered_at=ORIGIN + dt.timedelta(days=1))
    assert EV.eligible([future], ORIGIN, 90) == []
    adj, _, _, n = EV.adjust(BASE, SKUS, ORIGIN, 90, [future], LINES)
    assert np.array_equal(adj, BASE)
    assert np.all(n == 0)

    # ... and the same event IS applied once the origin catches up with it.
    later = ORIGIN + dt.timedelta(days=1)
    assert EV.eligible([future], later, 90) == [future]


def test_an_event_that_already_ended_is_not_applied():
    """Those days are in the panel the model was fitted on. Overriding them would
    double-count what actually happened."""
    over = ev(start=ORIGIN - dt.timedelta(days=30), end=ORIGIN - dt.timedelta(days=1))
    assert EV.eligible([over], ORIGIN, 90) == []


def test_the_window_opens_the_day_after_the_origin():
    """Same off-by-one as `track.actuals`: `units_lag_0` is a feature, so the origin day
    is something the model was shown, not something an override may move."""
    on_origin = ev(start=ORIGIN, end=ORIGIN)
    assert EV.overlap_days(on_origin, ORIGIN, 90) == 0
    day_after = ev(start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=1))
    assert EV.overlap_days(day_after, ORIGIN, 90) == 1


# --------------------------------------------------------------------------------------
# Arithmetic
# --------------------------------------------------------------------------------------

def test_a_full_window_multiplier_applies_in_full():
    e = ev(start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=90),
           multiplier=1.5)
    adj, factor, _, _ = EV.adjust(BASE, SKUS, ORIGIN, 90, [e], LINES)
    assert factor[0] == pytest.approx(1.5)
    assert adj[0] == pytest.approx(150.0)
    assert adj[1] == pytest.approx(200.0)      # different SKU, untouched


def test_a_partial_window_multiplier_is_prorated_by_days():
    """30 of 90 days at 2.0x is 1 + (30/90)*(2-1) = 1.333x, not 2x."""
    e = ev(start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=30),
           multiplier=2.0)
    _, factor, _, _ = EV.adjust(BASE, SKUS, ORIGIN, 90, [e], LINES)
    assert factor[0] == pytest.approx(1 + (30 / 90) * 1.0)


def test_the_same_event_dilutes_as_the_horizon_grows():
    """A 7-day promo is most of a 30-day window and almost none of a 180-day one. This is
    why promo input barely moves a long purchasing decision: a short deal is pro-rated
    across the whole window, so its effect shrinks as 1/h."""
    e = ev(start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=7),
           multiplier=1.8)
    factors = [EV.adjust(BASE, SKUS, ORIGIN, h, [e], LINES)[1][0] for h in (30, 90, 180)]
    assert factors[0] > factors[1] > factors[2] > 1.0
    assert factors[2] == pytest.approx(1 + (7 / 180) * 0.8)


def test_units_are_spread_over_the_event_and_only_the_overlap_lands():
    """+900 units over a 90-day event, of which 30 days fall inside a 30-day horizon."""
    e = ev(start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=90),
           multiplier=None, units=900.0, kind="launch")
    _, _, d30, _ = EV.adjust(BASE, SKUS, ORIGIN, 30, [e], LINES)
    _, _, d90, _ = EV.adjust(BASE, SKUS, ORIGIN, 90, [e], LINES)
    assert d30[0] == pytest.approx(300.0)
    assert d90[0] == pytest.approx(900.0)


def test_probability_scales_both_kinds_of_effect():
    m = ev(start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=90),
           multiplier=2.0, probability=0.5)
    _, factor, _, _ = EV.adjust(BASE, SKUS, ORIGIN, 90, [m], LINES)
    assert factor[0] == pytest.approx(1.5)          # 1 + 0.5*(2-1)

    u = ev(start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=90),
           multiplier=None, units=1000.0, probability=0.25, kind="launch")
    _, _, delta, _ = EV.adjust(BASE, SKUS, ORIGIN, 90, [u], LINES)
    assert delta[0] == pytest.approx(250.0)


def test_scaling_happens_before_units_are_added():
    """A catalogue-wide level dial must not also inflate an explicit launch quantity --
    8,000 units means 8,000 units."""
    dial = ev(scope="all", start=ORIGIN + dt.timedelta(days=1),
              end=ORIGIN + dt.timedelta(days=90), multiplier=2.0)
    launch = ev(scope="ZD-104", start=ORIGIN + dt.timedelta(days=1),
                end=ORIGIN + dt.timedelta(days=90), multiplier=None, units=1000.0,
                kind="launch")
    adj, _, _, _ = EV.adjust(BASE, SKUS, ORIGIN, 90, [dial, launch], LINES)
    assert adj[0] == pytest.approx(100.0 * 2.0 + 1000.0)


def test_multiplicative_events_compose():
    a = ev(scope="all", start=ORIGIN + dt.timedelta(days=1),
           end=ORIGIN + dt.timedelta(days=90), multiplier=1.5)
    b = ev(scope="ZD-104", start=ORIGIN + dt.timedelta(days=1),
           end=ORIGIN + dt.timedelta(days=90), multiplier=1.2)
    _, factor, _, _ = EV.adjust(BASE, SKUS, ORIGIN, 90, [a, b], LINES)
    assert factor[0] == pytest.approx(1.8)      # 1.5 * 1.2
    assert factor[1] == pytest.approx(1.5)      # only the catalogue dial


def test_a_discontinuation_zeroes_the_days_it_covers():
    e = ev(scope="ZB-105", kind="discontinue", multiplier=0.0,
           start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=3650))
    adj, _, _, _ = EV.adjust(BASE, SKUS, ORIGIN, 90, [e], LINES)
    assert adj[1] == pytest.approx(0.0)
    assert adj[0] == pytest.approx(100.0)


def test_demand_is_never_pushed_below_zero():
    e = ev(start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=90),
           multiplier=None, units=-99_999.0, kind="other")
    adj, _, _, _ = EV.adjust(BASE, SKUS, ORIGIN, 90, [e], LINES)
    assert adj[0] == 0.0


# --------------------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------------------

def test_scope_matches_a_sku_a_line_or_the_catalogue():
    win = dict(start=ORIGIN + dt.timedelta(days=1), end=ORIGIN + dt.timedelta(days=90),
               multiplier=2.0)
    one = EV.adjust(BASE, SKUS, ORIGIN, 90, [ev(scope="ZB-105", **win)], LINES)[3]
    line = EV.adjust(BASE, SKUS, ORIGIN, 90, [ev(scope="line:ZB", **win)], LINES)[3]
    allc = EV.adjust(BASE, SKUS, ORIGIN, 90, [ev(scope="all", **win)], LINES)[3]
    assert one.tolist() == [0, 1, 0]
    assert line.tolist() == [0, 1, 1]
    assert allc.tolist() == [1, 1, 1]


def test_a_line_scope_cannot_match_without_a_line_column():
    """Silently matching nothing is bad, but silently matching EVERYTHING would order the
    catalogue. `forecast.py` prints a warning when the column is absent."""
    e = ev(scope="line:ZB", start=ORIGIN + dt.timedelta(days=1),
           end=ORIGIN + dt.timedelta(days=90), multiplier=2.0)
    _, factor, _, n = EV.adjust(BASE, SKUS, ORIGIN, 90, [e], None)
    assert np.all(factor == 1.0)
    assert np.all(n == 0)


# --------------------------------------------------------------------------------------
# Parsing: every refusal is deliberate
# --------------------------------------------------------------------------------------

def test_effect_needs_a_sigil():
    """A bare `1.15` is ambiguous between a multiplier and 1.15 units, and the two
    readings differ by orders of magnitude."""
    assert EV.parse_effect("x1.15", 2) == (1.15, None)
    assert EV.parse_effect("+8000", 2) == (None, 8000.0)
    assert EV.parse_effect("-500", 2) == (None, -500.0)
    with pytest.raises(EV.EventError, match="not understood"):
        EV.parse_effect("1.15", 2)
    with pytest.raises(EV.EventError, match="not understood"):
        EV.parse_effect("15%", 2)


def test_an_absurd_multiplier_is_refused_not_clipped():
    with pytest.raises(EV.EventError, match="more than"):
        EV.parse_effect("x25", 2)


@pytest.mark.parametrize("row,match", [
    ("ZD-104,2027-01-01,2027-02-01,launch,+100,,2026-08-10,me,n", "probability` is required"),
    ("ZD-104,2027-01-01,2027-02-01,launch,+100,1.5,2026-08-10,me,n", "outside 0-1"),
    ("ZD-104,2027-01-01,2027-02-01,nonsense,+100,1,2026-08-10,me,n", "is not one of"),
    ("ZD-104,2027-02-01,2027-01-01,launch,+100,1,2026-08-10,me,n", "before `start_date`"),
    ("ZD-104,2027-01-01,,launch,+100,1,2026-08-10,me,n", "needs an `end_date`"),
    ("ZD-104,2027-01-01,2027-02-01,launch,+100,1,2026-08-10,,n", "author` is required"),
    (",2027-01-01,2027-02-01,launch,+100,1,2026-08-10,me,n", "scope` is required"),
    ("ZD-104,not-a-date,2027-02-01,launch,+100,1,2026-08-10,me,n", "not an ISO date"),
])
def test_bad_rows_are_refused_with_a_row_number(tmp_path, row, match):
    with pytest.raises(EV.EventError, match=match):
        EV.load(write(tmp_path, row))


def test_a_missing_column_is_refused(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text("scope,start_date,effect\nAA-01,2027-01-01,x1.5\n")
    with pytest.raises(EV.EventError, match="missing column"):
        EV.load(p)


def test_an_open_ended_multiplier_is_allowed(tmp_path):
    """A delisting or a permanent repricing has no end date."""
    got = EV.load(write(
        tmp_path, "ZD-104,2027-01-01,,discontinue,x0,1,2026-08-10,me,delisted"))
    assert len(got) == 1
    assert (got[0].end - got[0].start).days == EV.OPEN_ENDED_DAYS


def test_a_round_trip_through_the_file_preserves_the_numbers(tmp_path):
    got = EV.load(write(
        tmp_path,
        "ZB-105,2026-09-01,2026-11-30,launch,+8000,0.9,2026-08-01,planner,new colourway",
        "line:ZB,2026-09-01,,level,x1.15,0.6,2026-08-01,planner,ad push"))
    assert [e.scope for e in got] == ["ZB-105", "line:ZB"]
    assert got[0].units == 8000.0 and got[0].probability == 0.9
    assert got[1].multiplier == 1.15 and got[1].kind == "level"
    # Both were entered before the origin, so both are live at it.
    assert len(EV.eligible(got, ORIGIN, 180)) == 2


def test_blank_lines_are_skipped_not_refused(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text(",".join(EV.COLUMNS) + "\n"
                 + "ZD-104,2027-01-01,2027-02-01,launch,+100,1,2026-08-10,me,n\n"
                 + ",,,,,,,,\n")
    assert len(EV.load(p)) == 1


def test_the_report_renders_for_an_empty_file_and_a_full_one(tmp_path):
    empty = EV.report([], ORIGIN, (30, 90, 180), None)
    assert "empty or absent" in empty
    got = EV.load(write(
        tmp_path, "ZB-105,2026-09-01,2026-11-30,launch,+8000,0.9,2026-08-01,planner,x"))
    full = EV.report(got, ORIGIN, (30, 90, 180), None)
    assert "ZB-105" in full and "planner" in full


def test_comments_are_allowed_so_the_example_file_can_be_copied(tmp_path):
    """`dev/events.example.csv` is annotated with `#` lines. It has to be usable as-is
    when copied to `dev/events.csv`, or the example teaches a format that does not load."""
    p = tmp_path / "events.csv"
    p.write_text(",".join(EV.COLUMNS) + "\n"
                 + "# a note about why this row exists\n"
                 + "ZD-104,2027-01-01,2027-02-01,launch,+100,1,2026-08-10,me,n\n")
    assert len(EV.load(p)) == 1


def test_the_example_file_parses():
    example = Path(__file__).resolve().parents[1] / "dev" / "events.example.csv"
    if not example.exists():
        pytest.skip("no example file")
    got = EV.load(example)
    assert got, "the example should contain usable rows"
    assert {e.kind for e in got} <= set(EV.KINDS)
    assert all(0.0 <= e.probability <= 1.0 for e in got)


# --------------------------------------------------------------------------------------
# The append-only journal
# --------------------------------------------------------------------------------------

def test_locking_then_reloading_unchanged_rows_is_fine(tmp_path, monkeypatch):
    p = write(tmp_path, "ZD-104,2027-01-01,2027-02-01,launch,+100,1,2026-08-10,me,n")
    lock = tmp_path / "events.lock.json"
    monkeypatch.setattr(EV, "LOCKFILE", lock)
    EV.write_lock(EV.load(p, check_lock=False), lock)
    assert len(EV.load(p)) == 1          # no violation


def test_editing_a_locked_event_is_refused(tmp_path, monkeypatch):
    """An override rewritten after the fact is the failure that makes a forecast look
    prescient. It has to be visible, not silent."""
    p = write(tmp_path, "ZD-104,2027-01-01,2027-02-01,launch,+100,1,2026-08-10,me,n")
    lock = tmp_path / "events.lock.json"
    monkeypatch.setattr(EV, "LOCKFILE", lock)
    EV.write_lock(EV.load(p, check_lock=False), lock)

    write(tmp_path, "ZD-104,2027-01-01,2027-02-01,launch,+9999,1,2026-08-10,me,n")
    with pytest.raises(EV.LockError, match="changed or removed"):
        EV.load(p)


def test_backdating_a_new_event_is_refused(tmp_path, monkeypatch):
    """The one thing `entered_at` exists to prevent. A row added today claiming it was
    known last year would let a forecast be scored against knowledge it never had."""
    p = write(tmp_path, "ZD-104,2027-01-01,2027-02-01,launch,+100,1,2026-08-10,me,n")
    lock = tmp_path / "events.lock.json"
    monkeypatch.setattr(EV, "LOCKFILE", lock)
    EV.write_lock(EV.load(p, check_lock=False), lock)

    write(tmp_path,
          "ZD-104,2027-01-01,2027-02-01,launch,+100,1,2026-08-10,me,n",
          "ZB-105,2027-03-01,2027-04-01,launch,+500,1,2020-01-01,me,backdated")
    with pytest.raises(EV.LockError, match="cannot be entered into the past"):
        EV.load(p)


def test_an_honestly_new_event_is_accepted(tmp_path, monkeypatch):
    """The mirror image: a row entered today, after the lock, is ordinary use."""
    p = write(tmp_path, "ZD-104,2027-01-01,2027-02-01,launch,+100,1,2026-08-10,me,n")
    lock = tmp_path / "events.lock.json"
    monkeypatch.setattr(EV, "LOCKFILE", lock)
    EV.write_lock(EV.load(p, check_lock=False), lock)

    today = dt.date.today().isoformat()
    write(tmp_path,
          "ZD-104,2027-01-01,2027-02-01,launch,+100,1,2026-08-10,me,n",
          f"ZB-105,2027-03-01,2027-04-01,launch,+500,1,{today},me,new")
    assert len(EV.load(p)) == 2


def test_no_lock_file_means_no_constraint(tmp_path, monkeypatch):
    """The journal is opt-in. Without it the layer behaves exactly as before."""
    monkeypatch.setattr(EV, "LOCKFILE", tmp_path / "absent.json")
    p = write(tmp_path, "ZD-104,2027-01-01,2027-02-01,launch,+100,1,2020-01-01,me,n")
    assert len(EV.load(p)) == 1
