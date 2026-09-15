"""MOVING AVERAGE — the baseline everything else has to beat.

    uv run python dev/moving_average.py

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
Take how many units this SKU has averaged per day recently, and assume the next N days
look the same:

    forecast(next N days) = (average units per day over the last W days) x N

That is the whole model. There are no parameters to learn and nothing to train.

-----------------------------------------------------------------------------------
HOW IT IS SHAPED, AND WHY
-----------------------------------------------------------------------------------
The one real choice is W, the lookback window. This file matches the lookback to the
lookahead:

    forecasting 1 month  ->  average the last 28 days
    forecasting 3 months ->  average the last 91 days
    forecasting 6 months ->  average the last 182 days

The reasoning: a short window tracks the current level fast but is noisy, a long window
is stable but slow to react. How much noise you can tolerate depends on how far ahead
you are committing. For a reorder you make again in 4 weeks, being responsive is worth
the noise. For a 6-month container booking, one freak week should not move the order --
you want the stable number. Averaging 6 months of history to predict 6 months forward
also quietly cancels seasonality: the lookback window contains roughly as much peak and
trough as the lookahead window does.

The windows come from columns the feature pipeline already computed with strictly
past-only timing (`units_mean_28` is the 28 days ENDING on the forecast date), so this
baseline sees exactly the information the fancy models see. No accidental advantage in
either direction, which is the only way a baseline comparison means anything.

-----------------------------------------------------------------------------------
HOW IT IS TRAINED
-----------------------------------------------------------------------------------
It is not. `fit()` does nothing. This matters when you read the leaderboard: this model
cannot overfit, so its backtest score is an unusually honest estimate of what it would
do in production. A model that beats it by 3% in backtest is not obviously better.

-----------------------------------------------------------------------------------
HOW IT IS SCORED
-----------------------------------------------------------------------------------
Identically to every other file here: the same rolling-origin splits, WAPE / bias / shortfall,
averaged. See common.py.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from common import LabModel, run_model

#: Which trailing-average column to use for each horizon. See the docstring above.
WINDOW_FOR_HORIZON = {30: "units_mean_28", 90: "units_mean_91", 180: "units_mean_182"}
FALLBACK = "units_mean_28"


class MovingAverage:
    """A daily rate, multiplied up to the length of the forecast window."""

    def __init__(self, horizon: int, column: str | None = None,
                 scale: float = 1.0):
        self.horizon = horizon
        #: Which trailing average to read. Defaults to the matched-lookback choice
        #: argued for above; tune.py is allowed to override it because "match the
        #: lookback to the lookahead" is a reasoned guess, not a measured result.
        self.column = column or WINDOW_FOR_HORIZON.get(horizon, FALLBACK)
        #: A flat multiplier on the daily rate. 1.0 is the textbook model. The only
        #: reason to move it is that a trailing mean over a censored panel is a
        #: biased estimate of demand, and the size of that bias is measurable.
        self.scale = float(scale)

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        if self.column not in X.columns:
            raise KeyError(f"{self.column} is not in the dataset")
        return self  # nothing to learn

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        daily_rate = X.get_column(self.column).to_numpy().astype(np.float64)
        # A brand new SKU has no trailing average yet. 0 is the honest answer; it is
        # also why WAPE on new products is bad for every model in this folder.
        daily_rate = np.nan_to_num(daily_rate, nan=0.0)
        return daily_rate * self.horizon * self.scale


MODEL = LabModel(
    name="moving_average",
    family="Baseline (no learning)",
    blurb="""
    Averages the SKU's recent daily sales and multiplies by the length of the forecast
    window, using a lookback as long as the lookahead (28 / 91 / 182 days). No training,
    no parameters, no features beyond one trailing average.
    """,
    pros=[
        "Free: no training, predicts in microseconds, cannot fail in production.",
        "Cannot overfit, so its backtest number is what you would really get.",
        "Completely explainable to a planner -- 'you sold 4/day, so 120 in a month'.",
        "Works on a brand-new SKU the day it has any history at all.",
        "Genuinely hard to beat on stable, high-volume products.",
    ],
    cons=[
        "Structurally blind to seasonality: it forecasts November from October.",
        "Ignores price, promotions, ads, rank and stock -- every lever you control.",
        "Lags turning points; a launch or a decline is only picked up after the fact.",
        "One trailing mean per SKU, fitted independently, means no cross-product learning at all.",
        "Cannot give you a range, only a point, so it says nothing about safety stock.",
    ],
    build=lambda h, **kw: MovingAverage(h, **kw),
    notes={"lookback_windows": WINDOW_FOR_HORIZON},
)

if __name__ == "__main__":
    run_model(MODEL)
