"""The retail-event calendar, and the one part of it that cannot be derived.

Black Friday, Cyber Monday and Thanksgiving fall out of the calendar, so they are
correct forever. Prime Day does not -- Amazon picks the dates -- so it is a hard-coded
table that RUNS OUT. When it does, `_amazon_prime_day` returns nothing and
`days_to_prime_day` quietly becomes the distance to the nearest year still in the
table: a plausible number, not a null, on the single largest demand spike of a
marketplace seller's year.

The table currently runs to 2028.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from demandml import calendar_features as cf


def test_the_table_covers_every_year_a_forecast_can_reach():
    """`add_calendar` builds events over `range(min.year, max.year + 2)`, so the table
    must lead the data by two years."""
    assert cf.prime_day_gaps(range(2015, 2029)) == []


def test_a_year_past_the_table_warns_rather_than_going_quiet():
    with pytest.warns(RuntimeWarning, match="no Prime Day dates"):
        cf.retail_event_frame(range(2029, 2031))


def test_derived_events_need_no_table():
    """Thanksgiving 2029 is the 4th Thursday of November whether or not anyone maintains
    a list, so BF/CM stay correct past the Prime Day horizon."""
    with pytest.warns(RuntimeWarning):
        ev = cf.retail_event_frame(range(2029, 2030))
    got = dict(zip(ev["event"].to_list(), ev["date"].to_list()))
    assert got["thanksgiving"] == dt.date(2029, 11, 22)
    assert got["black_friday"] == dt.date(2029, 11, 23)
    assert got["cyber_monday"] == dt.date(2029, 11, 26)


def test_known_prime_days_are_right():
    """A couple of anchors, including the year the pandemic moved it out of July."""
    assert dt.date(2020, 10, 13) in cf._amazon_prime_day(2020)
    assert dt.date(2025, 7, 8) in cf._amazon_prime_day(2025)
    # 2026 moved to June; the July projection it replaced was wrong by three weeks.
    assert dt.date(2026, 6, 23) in cf._amazon_prime_day(2026)
    assert dt.date(2026, 7, 14) not in cf._amazon_prime_day(2026)


def test_every_projected_year_lands_on_the_usual_weekdays():
    """Prime Day has been a Tuesday/Wednesday event since 2022. The projections should
    not silently drift onto a weekend."""
    for y in (2027, 2028):
        days = cf._amazon_prime_day(y)
        assert days, y
        assert {d.strftime("%a") for d in days} <= {"Tue", "Wed"}, (y, days)


def test_add_calendar_emits_the_distance_features():
    df = pl.DataFrame({"date": [dt.date(2026, 7, 1), dt.date(2026, 11, 27)]})
    out = cf.add_calendar(df)
    for c in ("days_to_prime_day", "days_to_black_friday", "days_to_cyber_monday",
              "days_to_thanksgiving", "days_to_christmas", "retail_event"):
        assert c in out.columns, c
    # 2026-07-01 -> Prime Day ended 2026-06-26, 5 days earlier (distance is unsigned).
    assert out["days_to_prime_day"].to_list()[0] == 5
