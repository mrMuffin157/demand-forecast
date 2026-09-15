"""CROSTON / SBA — the textbook method for products that sell in bursts.

    uv run python dev/croston.py

-----------------------------------------------------------------------------------
THE PROBLEM IT SOLVES
-----------------------------------------------------------------------------------
A sizeable share of SKU-days in a retail panel are zero, and for the slow movers it is
most of them. A plain moving average handles that badly: a SKU that sells 10 units once a
fortnight has a trailing mean of ~0.7/day, and the model then predicts 0.7 units every
single day -- a number that never once occurs in reality.

Croston's insight (1972) is to stop modelling "units per day" as one quantity and split
it into two questions that behave much better:

    1. WHEN it sells, HOW MANY does it sell?   -> average non-zero order size, z
    2. HOW OFTEN does it sell?                 -> average interval between sales, p

    demand per day = z / p

Both z and p are estimated only from the days where something happened, so the long runs
of zeros stop dragging the estimate around.

-----------------------------------------------------------------------------------
HOW IT IS SHAPED, AND WHY
-----------------------------------------------------------------------------------
This is the Syntetos-Boylan Approximation (SBA), which is plain Croston times
(1 - alpha/2). That correction is not cosmetic: classical Croston is provably biased
upwards -- it estimates the demand rate on a ratio of two smoothed quantities, and the
expectation of a ratio is not the ratio of expectations. Left uncorrected it
over-forecasts slow movers, which is exactly where you can least afford dead stock.
SBA is the standard fix and it is what most commercial planning systems ship.

    forecast(next N days) = (1 - alpha/2) * z / p * N

z and p come from the pipeline's `nonzero_mean_91` and `adi_91` columns (average demand
interval), both computed strictly from the past, so the timing matches every other model
in the folder.

-----------------------------------------------------------------------------------
HOW IT IS TRAINED
-----------------------------------------------------------------------------------
It is not, in the usual sense. alpha (the smoothing rate) is fixed at 0.1, the standard
value. The two statistics it needs are already computed per SKU. Like the moving
average, it cannot overfit.

-----------------------------------------------------------------------------------
WHAT TO LOOK FOR IN ITS RESULTS
-----------------------------------------------------------------------------------
Watch `bias` and `shortfall`, not just WAPE. Croston is known to be conservative, and on
this dataset it tends to have a good WAPE with the WORST shortfall on the board -- that
means it is quietly under-forecasting, which in inventory terms is stockouts and lost
sales, not a saving. A model that is "accurate" by being timid is a specific trap, and
this file is here partly to show you what that looks like.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from common import LabModel, run_model

ALPHA = 0.1  # standard smoothing constant; the SBA correction is (1 - ALPHA/2)


class CrostonSBA:
    """demand rate = (1 - alpha/2) * (average non-zero size) / (average interval)."""

    def __init__(self, horizon: int, size_col: str = "nonzero_mean_91",
                 interval_col: str = "adi_91", alpha: float = ALPHA):
        self.horizon = horizon
        self.size_col = size_col
        self.interval_col = interval_col
        self.alpha = alpha

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        for c in (self.size_col, self.interval_col):
            if c not in X.columns:
                raise KeyError(f"{c} is not in the dataset")
        return self  # nothing to learn

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        size = X.get_column(self.size_col).to_numpy().astype(np.float64)
        interval = X.get_column(self.interval_col).to_numpy().astype(np.float64)
        # A SKU that sells every day has an interval of 1. Guard against 0 and nulls,
        # which would otherwise divide by zero or poison the whole prediction.
        interval = np.where(np.isfinite(interval) & (interval > 0), interval, 1.0)
        daily_rate = (1.0 - self.alpha / 2.0) * np.nan_to_num(size) / interval
        return daily_rate * self.horizon


MODEL = LabModel(
    name="croston",
    family="Baseline (intermittent demand)",
    blurb="""
    Splits demand into how much sells when it sells, and how often it sells, then
    divides one by the other. The Syntetos-Boylan correction removes the upward bias
    that plain Croston has. Purpose-built for slow and lumpy products.
    """,
    pros=[
        "Designed for exactly this data: intermittent, bursty, half-empty series.",
        "Usually the most STABLE model across splits -- useful if you care about the "
        "worst quarter rather than the average one.",
        "As cheap and as explainable as a moving average.",
        "Degrades gracefully on a SKU with almost no sales history.",
        "The industry default, so it is the number a planner will recognise.",
    ],
    cons=[
        "Tends to under-forecast, which shows up as the worst SHORTFALL on the board -- "
        "good-looking WAPE bought with stockouts.",
        "No seasonality, no trend, no price or promo effects.",
        "Pointless on fast movers that sell every day: it collapses to a slow average.",
        "alpha is a fixed guess, not fitted.",
        "Assumes the burst pattern is stationary; it is not, around launches and deals.",
    ],
    build=lambda h, **kw: CrostonSBA(h, **kw),
    notes={"alpha": ALPHA, "size_col": "nonzero_mean_91", "interval_col": "adi_91"},
)

if __name__ == "__main__":
    run_model(MODEL)
