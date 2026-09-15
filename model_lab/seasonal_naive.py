"""SEASONAL MODEL — last year's same window, adjusted for how the SKU has grown since.

    uv run python model_lab/seasonal_naive.py

-----------------------------------------------------------------------------------
THE PROBLEM IT SOLVES
-----------------------------------------------------------------------------------
moving_average.py has one structural blind spot: it forecasts November from October.
Every trailing average does. On a marketplace where Black Friday, Prime Day and Q4 move
demand by multiples, that is not a rounding error -- and it gets worse the further ahead
you look, which is exactly where the 3- and 6-month forecasts live.

This model attacks that directly, using the oldest idea in forecasting: whatever happened
this time last year is the best single guess for what happens this time this year.

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
    forecast(next N days) = (units in the SAME N days last year) x (growth factor)

    growth factor = (average daily units over the last 91 days)
                    / (average daily units over the same 91 days a year ago)

The first term carries the seasonal SHAPE. The second carries the LEVEL: a SKU that has
doubled since last year gets last year's Christmas doubled, a SKU in decline gets it cut.
Without the growth factor this would be pure seasonal naive and would systematically
under-forecast every growing product.

Why 364 days and not 365: 364 = 52 x 7, so "one year ago" lands on the same day of the
week. On retail data the weekly cycle is strong enough that being one weekday out
measurably degrades the comparison.

-----------------------------------------------------------------------------------
HOW IT IS SHAPED, AND WHY
-----------------------------------------------------------------------------------
* The growth factor is CLAMPED to [0.25, 4.0]. Ratios are dangerous: a SKU that sold
  almost nothing last year but sells a little now produces a growth factor of 300, and
  multiplying last Christmas by 300 puts an absurd order into the system. Clamping is
  crude and it is the right kind of crude -- it bounds the worst case.
* It FALLS BACK to a moving average whenever last year is unusable: the SKU is younger
  than a year, or it sold nothing at all in that window. That is a sizeable share of
  rows, and a model with no fallback would score terribly for a reason that has nothing to do with
  seasonality.
* Both inputs (`ly_cum{N}`, `ly_mean_91`) are built in prepare_data.py by shifting a
  known quantity back 364 days, so they only ever describe the past.

-----------------------------------------------------------------------------------
HOW IT IS TRAINED
-----------------------------------------------------------------------------------
It is not. Like the other two baselines, there is nothing to fit -- which again means its
backtest score is an honest estimate rather than an optimistic one.

-----------------------------------------------------------------------------------
WHAT TO LOOK FOR
-----------------------------------------------------------------------------------
Compare it with moving_average.py PER HORIZON, not overall. The expectation is that it
loses at 1 month -- where recent level is what matters and last year is just noise -- and
closes the gap or wins at 6 months, where the forecast window contains a whole season
that a trailing average cannot see. If that pattern does not appear in the numbers, the
seasonality in this catalogue is weaker than the retail folklore assumes, which is itself
worth knowing before anyone builds a seasonal planning process on it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from common import LabModel, run_model

GROWTH_CLAMP = (0.25, 4.0)
FALLBACK_COL = "units_mean_28"  # used when last year is missing or empty


class SeasonalWithGrowth:
    def __init__(self, horizon: int, clamp: tuple[float, float] = GROWTH_CLAMP):
        self.horizon = horizon
        self.clamp = clamp
        self.ly_col = f"ly_cum{horizon}"

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        for c in (self.ly_col, "ly_mean_91", "units_mean_91", FALLBACK_COL):
            if c not in X.columns:
                raise KeyError(f"{c} is not in the dataset -- rerun prepare_data.py")
        return self  # nothing to learn

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        ly = X.get_column(self.ly_col).to_numpy().astype(np.float64)
        now91 = X.get_column("units_mean_91").to_numpy().astype(np.float64)
        then91 = X.get_column("ly_mean_91").to_numpy().astype(np.float64)
        fallback = np.nan_to_num(
            X.get_column(FALLBACK_COL).to_numpy().astype(np.float64)) * self.horizon

        # Growth: how the SKU's level compares with a year ago, bounded hard.
        with np.errstate(divide="ignore", invalid="ignore"):
            growth = np.where(then91 > 0, now91 / then91, 1.0)
        growth = np.clip(np.nan_to_num(growth, nan=1.0), *self.clamp)

        pred = ly * growth
        # Fall back wherever last year is missing or was itself a dead window.
        unusable = ~np.isfinite(ly) | (ly <= 0)
        return np.where(unusable, fallback, pred)


MODEL = LabModel(
    name="seasonal_naive",
    family="Baseline (seasonal)",
    blurb="""
    Takes the same calendar window one year earlier and scales it by how much the SKU's
    level has changed since, falling back to a trailing average when there is no usable
    year-ago window. The only cheap model here that can see a season coming.
    """,
    pros=[
        "The only baseline that can anticipate Q4, Prime Day and Black Friday instead of "
        "discovering them a month late.",
        "Gets relatively stronger as the horizon lengthens, which is where the expensive "
        "decisions are made.",
        "Still no training and no parameters beyond one clamp.",
        "Easy to explain and to audit: a planner can check the number by hand.",
        "The growth factor keeps it usable on products that have doubled or halved.",
    ],
    cons=[
        "Needs more than a year of history, so it is blind on new products -- a "
        "sizeable share of rows fall back to a plain average.",
        "Repeats last year's accidents: a stock-out, a viral week or a one-off wholesale "
        "order gets forecast as if it were seasonality.",
        "One year of evidence is one sample. Real seasonal estimation wants several.",
        "The clamp is arbitrary and will occasionally be the binding constraint.",
        "Assumes the calendar repeats; a moved Prime Day or a new competitor breaks it.",
    ],
    build=lambda h: SeasonalWithGrowth(h),
    notes={"growth_clamp": list(GROWTH_CLAMP), "lag_days": 364, "fallback": FALLBACK_COL},
)

if __name__ == "__main__":
    run_model(MODEL)
