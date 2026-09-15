"""LIGHTGBM, ADDITIVE — predict each future month separately, then add them up.

    uv run python model_lab/lightgbm_additive.py

This file exists to answer one question: is it better to predict a 6-month total in one
shot, or to predict six months and sum them?

    lightgbm_gbdt.py      6 months = one model trained on y_cum180          "just 6 months"
    lightgbm_additive.py  6 months = m1 + m2 + m3 + m4 + m5 + m6            "1+1+1+1+1+1"

Same learner, same features, same folds, same metrics. The ONLY difference is how the
long horizon is assembled, so the score gap between these two files is a clean
measurement of that one decision. (`LightGBMTweedie` is imported from
`lightgbm_gbdt.py` rather than copied, precisely so it cannot drift.)

-----------------------------------------------------------------------------------
THE THREE WAYS TO BUILD A LONG FORECAST (and which one this is)
-----------------------------------------------------------------------------------
1. DIRECT — train one model on the 180-day total. What every other file here does.

2. DIRECT-THEN-AGGREGATE — train a separate model for each future month, each
   predicting only that month's units from what is known at the origin, then sum.
   **This file.** Sometimes called "multi-output direct".

3. RECURSIVE / ITERATED — predict month 1, pretend that prediction is now history,
   recompute every trailing feature from it, predict month 2, repeat six times.

This file does (2), not (3), and that is deliberate. Recursive forecasting feeds a
model its own output as input, so a small month-1 error is baked into month 2's
features, which corrupts month 3's, and so on: error compounds multiplicatively rather
than just adding. It also means rebuilding hundreds of trailing features six times
over from synthetic data. Option (2) gets the "sum of pieces" structure the honest way — every
sub-model reads only real, observed history from the origin date.

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
The dataset carries a target `y_cum_m{k}` = units sold in month k of the future:

    y_cum_m1   days   1.. 30 after the origin
    y_cum_m2   days  31.. 60
    ...
    y_cum_m6   days 151..180

For a horizon of H days the model trains H/30 sub-models, one per month, and predicts:

    forecast(H) = sum over k of  max(0, submodel_k.predict(x))

Each month's prediction is floored at zero BEFORE summing. Without that, a sub-model
that goes slightly negative on a quiet month would silently cancel out real demand
predicted for a busy one.

Note what this means at 1 month: H/30 = 1, so there is exactly one sub-model trained on
one month, which is the same thing lightgbm_gbdt.py does at that horizon. The two files
should score almost identically on `h030` — small differences are just the sub-model's
own randomness. That is a free sanity check: if the 1-month scores diverge a lot,
something in this file is wrong, not interesting.

Every sub-model is trained on the rows where ITS OWN month's target survived the
stock-out filter, which is not quite the same row set for every month (a product can be
buyable in month 1 and unbuyable in month 5). So the sub-models see slightly different
numbers of rows, and the count is printed.

-----------------------------------------------------------------------------------
WHY THIS MIGHT WIN
-----------------------------------------------------------------------------------
A 180-day total and a single month are genuinely different learning problems. The month
target is smaller, more homogeneous and closer in character to the trailing windows the
features describe, so each sub-model is being asked an easier question than "what is the
half-year total". You also get six times the target signal out of the same rows.

-----------------------------------------------------------------------------------
WHAT TO EXPECT FROM ITS SCORES
-----------------------------------------------------------------------------------
Read the comparison with lightgbm_gbdt.py horizon by horizon, and narrowly:

* At 1 month the two must agree almost exactly -- one sub-model IS the direct model. If
  they do, the correctness check is working; if they do not, something is broken.
* At 3 months, on the data this was developed on, additive came out slightly ahead, but
  by less than either model's own spread across splits. By this folder's own rule that
  is not an established win, and should not be quoted as one.
* At 6 months the gap was larger and just cleared the split spread. "Probably better at
  6 months", not "better".

So splitting the horizon into months tends to help where there is most to split -- the
longer the horizon, the more it helps. That is the expected direction, but it rests on
one dataset and three splits.

The feared bias blow-up (below) did not materialise there: the additive model ran
somewhat more under-forecast than the direct one at 6 months, but nothing like the
compounding the naive argument predicts. Six sub-models trained on real history (rather
than recursively on each other's output) apparently do not stack their errors that way.

Cost: roughly double the direct model's training time at the 6-month horizon, for six
times the fits.

-----------------------------------------------------------------------------------
WHY IT MIGHT LOSE
-----------------------------------------------------------------------------------
Two mechanisms that did not dominate here but are real and general, and are the first
things to check if you re-run this on different data:

* BIAS ADDS UP. If every sub-model leans low -- and boosting on skewed targets usually
  does -- the six errors all point the same way and none of them cancel in the sum; the
  direct model has only ONE number to be biased about. Watch the `bias` column, not just WAPE:
  this is where an additive model gives itself away. (Observed above: a real but small
  effect.)

* THE DIRECT MODEL GETS FREE NOISE CANCELLATION. A 180-day total is intrinsically
  smoother than any of its parts, because a quiet week and a busy week inside the window
  average out before the model ever sees the number. Predicting six noisy pieces and
  adding them puts that variance back in by hand.

-----------------------------------------------------------------------------------
THE THING IT GIVES YOU THAT NO DIRECT MODEL CAN
-----------------------------------------------------------------------------------
Even if it loses on WAPE, it produces a monthly PROFILE, not just a total: *when* in the
next six months the demand lands. A direct 180-day model cannot tell you that at any
price. For phasing a purchase order, scheduling container arrivals or planning warehouse
space, the shape is often worth more than a slightly better total -- so read this file's
score as "what does the shape cost me?", not purely as a ranking.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from common import MONTH_DAYS, N_MONTHS, LabModel, run_model
from lightgbm_gbdt import PARAMS, LightGBMTweedie


class AdditiveMonthly:
    """One LightGBMTweedie per future month; the forecast is their sum."""

    def __init__(self, horizon: int):
        if horizon % MONTH_DAYS:
            raise SystemExit(
                f"lightgbm_additive needs a horizon that is a whole number of "
                f"{MONTH_DAYS}-day months; got {horizon}."
            )
        self.horizon = horizon
        self.n_months = horizon // MONTH_DAYS
        if self.n_months > N_MONTHS:
            raise SystemExit(
                f"horizon {horizon} needs {self.n_months} monthly targets but the "
                f"dataset only carries {N_MONTHS} (y_cum_m1..y_cum_m{N_MONTHS}). "
                f"Raise N_MONTHS in common.py and re-run prepare_data.py."
            )
        self.models_: list[LightGBMTweedie] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None,
            frame: pl.DataFrame | None = None):
        """`y` (the H-day total) is ignored -- this model never sees it.

        It trains only on the per-month columns, which is the whole point. `frame` is
        the same training rows as `X` with the target columns still attached.
        """
        if frame is None:
            raise SystemExit("AdditiveMonthly needs the training frame; "
                             "set needs_frame=True on its LabModel.")

        self.models_ = []
        for k in range(1, self.n_months + 1):
            col = f"y_cum_m{k}"
            if col not in frame.columns:
                raise SystemExit(
                    f"the dataset has no {col}. It is built by stage 4 of "
                    f"prepare_data.py, which always re-runs -- so just:\n"
                    f"  uv run python model_lab/prepare_data.py"
                )
            yk = frame.get_column(col).to_numpy().astype(np.float64)
            # Train month k only on rows where month k's own target survived the
            # stock-out filter. A null here means "that month was mostly unbuyable",
            # which is missing information, not zero demand.
            keep = np.isfinite(yk)
            Xk = X.filter(pl.Series(keep)) if keep.sum() != len(keep) else X
            wk = None if w is None else w[keep]
            sub = LightGBMTweedie(MONTH_DAYS)
            sub.fit(Xk, yk[keep], wk)
            self.models_.append(sub)
            print(f"      month {k}: trained on {int(keep.sum()):,} rows")
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        # Floor each month at zero BEFORE summing, so a negative month cannot cancel
        # out real demand predicted for another one.
        parts = [np.clip(np.asarray(m.predict(X), dtype=np.float64).ravel(), 0.0, None)
                 for m in self.models_]
        return np.sum(parts, axis=0)

    def monthly_profile(self, X: pl.DataFrame) -> np.ndarray:
        """The per-month breakdown, shape (n_rows, n_months).

        This is the output a direct model cannot produce: not just how much will sell,
        but when it lands.
        """
        return np.column_stack([
            np.clip(np.asarray(m.predict(X), dtype=np.float64).ravel(), 0.0, None)
            for m in self.models_
        ])


MODEL = LabModel(
    name="lightgbm_additive",
    family="Trees (gradient boosting, additive)",
    blurb="""
    The same LightGBM/Tweedie learner as lightgbm_gbdt, but a long horizon is built by
    training one sub-model per future month and summing them (1+1+1+1+1+1) instead of
    predicting the whole total in one shot. Isolates one decision -- how to assemble a
    long forecast -- and is the only model here that also returns a monthly profile.
    """,
    pros=[
        "Tends to beat the same learner used directly at the long horizons. Only the "
        "6-month gap has cleared the split spread, so only that one counts as an "
        "established win.",
        "Each sub-model has an easier, more homogeneous job than 'predict the half-year "
        "total': a one-month target is closer in character to the trailing windows the "
        "features actually describe.",
        "Returns WHEN the demand lands, not just how much -- a monthly profile no "
        "direct model can produce at any price. Worth real money for phasing a PO or "
        "booking container slots.",
        "Six targets from the same rows, so more supervision out of the same data.",
        "The tail can be re-forecast without refitting the near months, and any single "
        "month can be inspected or overridden on its own.",
        "Degrades gracefully: at a 1-month horizon it reduces exactly to the direct "
        "model, which doubles as a built-in correctness check.",
    ],
    cons=[
        "BIAS ADDS. Six sub-models that all lean low give a 6-month total whose errors "
        "never cancel; the direct model only has one number to be wrong about. This is "
        "the failure mode to check first.",
        "Throws away free noise cancellation -- a 180-day total is intrinsically "
        "smoother than any of its six parts, and summing the parts puts that variance "
        "back in by hand.",
        "Roughly double the training time and six artefacts per horizon instead of "
        "one.",
        "Nothing forces the sum to agree with a directly-modelled total; the two "
        "answers can simply disagree and there is no reconciliation step here.",
        "The distant sub-models are still predicting from origin-date features, so "
        "month 6 is being asked a very hard question with stale inputs -- it does not "
        "get to lean on the aggregate's smoothness the way the direct model does.",
        "Needs the extra y_cum_m* targets in the dataset, so prepare_data.py has to "
        "have been run since they were added.",
    ],
    build=lambda h: AdditiveMonthly(h),
    needs_frame=True,
    notes={**PARAMS, "strategy": "direct-then-aggregate",
           "month_days": MONTH_DAYS, "max_months": N_MONTHS},
)

if __name__ == "__main__":
    run_model(MODEL)
