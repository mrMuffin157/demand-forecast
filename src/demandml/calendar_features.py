"""Calendar, holiday and retail-event features.

Separated from features.py because these are the only features that are *knowable
arbitrarily far into the future*, which makes them the backbone of long-horizon
forecasts.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import polars as pl

from . import config as C


def _holiday_frame(years: range, country: str = "US") -> pl.DataFrame:
    import holidays as hol

    cal = hol.country_holidays(country, years=list(years))
    rows = [{"date": d, "holiday_name": n} for d, n in sorted(cal.items())]
    if not rows:
        return pl.DataFrame(schema={"date": pl.Date, "holiday_name": pl.String})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _amazon_prime_day(year: int) -> list[dt.date]:
    """Prime Day dates. Mid-July in most years; moved by the 2020 pandemic, and to June
    in 2021 and 2026.

    Hard-coded because the event drives the single largest demand spike in the year for
    marketplace sellers and there is no rule to derive it from.
    """
    table = {
        2015: [(7, 15)],
        2016: [(7, 12)],
        2017: [(7, 10), (7, 11)],
        2018: [(7, 16), (7, 17)],
        2019: [(7, 15), (7, 16)],
        2020: [(10, 13), (10, 14)],
        2021: [(6, 21), (6, 22)],
        2022: [(7, 12), (7, 13), (10, 11), (10, 12)],
        2023: [(7, 11), (7, 12), (10, 10), (10, 11)],
        2024: [(7, 16), (7, 17), (10, 8), (10, 9)],
        2025: [(7, 8), (7, 9), (7, 10), (7, 11), (10, 7), (10, 8)],
        # Announced as June 23-26 (an earlier projection had guessed mid-July). Only the
        # US Prime Day is modelled; other countries' event dates differ.
        2026: [(6, 23), (6, 24), (6, 25), (6, 26), (10, 6), (10, 7)],
        2027: [(7, 13), (7, 14), (10, 5), (10, 6)],  # projected
        2028: [(7, 11), (7, 12), (10, 3), (10, 4)],  # projected
    }
    return [dt.date(year, m, d) for m, d in table.get(year, [])]


#: First year the table covers. Before this there was no Prime Day, so a missing year
#: below it is correct rather than an omission.
PRIME_DAY_FIRST_YEAR = 2015


def prime_day_gaps(years: range) -> list[int]:
    """Years at or after `PRIME_DAY_FIRST_YEAR` that the hard-coded table does not cover.

    The table cannot be derived -- Amazon picks the dates -- so it runs out, and when it
    does `_amazon_prime_day` returns an empty list and `days_to_prime_day` degrades
    SILENTLY into the distance to the nearest year that is still in the table. Prime Day
    is the largest single demand spike in the year for a marketplace seller, so that is
    a real loss of signal presented as a normal feature.

    Black Friday, Cyber Monday and Thanksgiving are derived from the calendar and never
    need this.
    """
    return [y for y in years
            if y >= PRIME_DAY_FIRST_YEAR and not _amazon_prime_day(y)]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """e.g. 4th Thursday of November = Thanksgiving."""
    d = dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=offset + 7 * (n - 1))


def retail_event_frame(years: range) -> pl.DataFrame:
    """Black Friday / Cyber Monday / Prime Day / Christmas-window flags."""
    gaps = prime_day_gaps(years)
    if gaps:
        warnings.warn(
            f"no Prime Day dates for {gaps}: `days_to_prime_day` will measure the "
            f"distance to the nearest year that IS in the table, which is wrong rather "
            f"than missing. Extend `_amazon_prime_day` in calendar_features.py.",
            RuntimeWarning, stacklevel=2)
    rows = []
    for y in years:
        thanksgiving = _nth_weekday(y, 11, 3, 4)  # Thursday == 3
        rows.append({"date": thanksgiving, "event": "thanksgiving"})
        rows.append({"date": thanksgiving + dt.timedelta(days=1), "event": "black_friday"})
        rows.append({"date": thanksgiving + dt.timedelta(days=4), "event": "cyber_monday"})
        for d in _amazon_prime_day(y):
            rows.append({"date": d, "event": "prime_day"})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def add_calendar(
    df: pl.DataFrame | pl.LazyFrame,
    date_col: str = C.DATE,
    country: str = "US",
    fourier_orders: tuple[tuple[float, int], ...] = ((7.0, 3), (365.25, 4)),
) -> pl.DataFrame | pl.LazyFrame:
    """Attach calendar parts, Fourier seasonality terms, holiday and retail-event flags.

    ``fourier_orders`` is a list of ``(period_in_days, n_harmonics)``. Fourier terms give
    tree models a smooth representation of seasonality that extrapolates, which raw
    week-of-year integers do not.
    """
    lazy = isinstance(df, pl.LazyFrame)
    d = pl.col(date_col)

    out = df.with_columns(
        d.dt.weekday().alias("dow"),
        d.dt.day().alias("dom"),
        d.dt.ordinal_day().alias("doy"),
        d.dt.week().alias("iso_week"),
        d.dt.month().alias("month"),
        d.dt.quarter().alias("quarter"),
        d.dt.year().alias("year"),
        (d.dt.weekday() >= 6).alias("is_weekend"),
        d.dt.month_start().eq(d).alias("is_month_start"),
        d.dt.month_end().eq(d).alias("is_month_end"),
        # A monotone time index lets a model pick up long-run trend/inflation directly.
        (d - pl.lit(dt.date(2015, 1, 1))).dt.total_days().alias("t_index"),
    )

    fourier: list[pl.Expr] = []
    for period, order in fourier_orders:
        tag = "w" if period < 30 else "y"
        base = pl.col("doy") if period > 30 else pl.col("dow")
        for k in range(1, order + 1):
            ang = 2.0 * np.pi * k * base.cast(pl.Float64) / period
            fourier += [
                ang.sin().alias(f"fourier_{tag}_sin_{k}"),
                ang.cos().alias(f"fourier_{tag}_cos_{k}"),
            ]
    out = out.with_columns(fourier)

    # --- holidays & retail events -----------------------------------------------------
    span = out.select(
        pl.col(date_col).min().alias("lo"), pl.col(date_col).max().alias("hi")
    )
    bounds = (span.collect() if lazy else span).row(0)
    years = range(bounds[0].year, bounds[1].year + 2)

    hol = _holiday_frame(years, country).rename({"date": date_col, "holiday_name": "_hol"})
    ev = retail_event_frame(years).rename({"date": date_col, "event": "_event"})
    # The weekly export ships its own `event` column; drop it rather than collide.
    existing = set(out.collect_schema().names())
    if "event" in existing:
        out = out.drop("event")
    if lazy:
        hol, ev = hol.lazy(), ev.lazy()

    out = out.join(hol, on=date_col, how="left").join(ev, on=date_col, how="left")
    out = out.with_columns(
        pl.col("_hol").is_not_null().alias("is_holiday"),
        pl.col("_event").fill_null("none").cast(pl.Categorical).alias("retail_event"),
    )

    # Distance to the nearest big event in either direction: demand ramps *before* the
    # day itself and slumps after, which a same-day flag cannot express.
    for name in ("black_friday", "cyber_monday", "prime_day", "thanksgiving"):
        dates = ev.filter(pl.col("_event") == name)
        if lazy:
            dates = dates.collect()
        vals = dates.get_column(date_col).to_list()
        if not vals:
            continue
        out = out.with_columns(
            pl.col(date_col)
            .map_batches(
                lambda s, v=np.array([x.toordinal() for x in vals]): pl.Series(
                    np.min(
                        np.abs(
                            np.asarray([d.toordinal() if d is not None else 0
                                        for d in s.to_list()])[:, None]
                            - v[None, :]
                        ),
                        axis=1,
                    ).astype(np.int32)
                ),
                return_dtype=pl.Int32,
            )
            .alias(f"days_to_{name}")
        )

    # Christmas shipping window: the ramp that matters is "days until Dec 25".
    out = out.with_columns(
        pl.when(pl.col("month") == 12)
        .then(25 - pl.col("dom"))
        .otherwise(None)
        .alias("days_to_christmas")
    )

    return out.drop("_hol", "_event")
