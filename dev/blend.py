"""BLEND -- three of the standard tree fits, averaged.

    uv run python dev/blend.py                     # backtest, like every model file
    uv run python dev/blend.py --folds 2           # smoke run

-----------------------------------------------------------------------------------
WHAT IT IS
-----------------------------------------------------------------------------------
The mean of three members, each the default tree estimator as the pipeline builds it
(lifted log fit, calibrated to sales on its own training rows -- `lifted_log.py`):

    base     the default model on the standard features: xgboost_gbdt at 90 d,
             random_forest at 180 d (`FOREST_BASE`)
    ext      xgboost_gbdt + the `xf_*` extended features
    rf_ext   random_forest + the `xf_*` extended features

`xf_*` (`src/demandml/xfeatures.py`) are in-stock demand rates and sibling stock-outs.
This is the only model that sees them (`extended_features=True`); `base` drops them again
so it is exactly the plain model.

-----------------------------------------------------------------------------------
WHY, AND WHERE IT HELPS
-----------------------------------------------------------------------------------
No member reliably beats the default model alone; the average can, because the forest's
misses differ from the boosted trees'. Averaging models whose errors are only partly
correlated is the oldest trick in forecasting, and it is the only reason this file
exists. In the probes behind it (variants fixed before scoring) the gain was clearest at
90 d, on both precision and inventory cost, and within noise at 30 d -- so it runs at 90
and 180 d (`horizons=(90, 180)`), never at 30 d.

At 180 d the base member is the FOREST, not xgboost. That variant was found after the
fact on one store, so treat it as a choice to re-check on new data rather than a rule.
`forecast.choose` picks per store and
horizon from whatever has `_preds` rows there, so a store where the blend does not pay
simply keeps a single model. See docs/FINDINGS.md. Re-measure before widening
`horizons` to 30.

-----------------------------------------------------------------------------------
COSTS
-----------------------------------------------------------------------------------
* Three fits per fold, and the forest runs at the board's 300k row cap (random_forest
  alone uses 150k): the slowest model on the board, and slower again when pooled.
* No `cost_params`: `--cost` refuses rather than filing a WAPE-trained blend there.
* Reopens the 7-model shortlist (`common.SHORTLIST`) by one, deliberately.
"""
from __future__ import annotations

import numpy as np

import random_forest
import xgboost_gbdt
from common import XF_PREFIX, LabModel, run_model


#: Horizons whose base member is the forest rather than xgboost.
FOREST_BASE = (180,)


class Blend:
    """Mean of member predictions. Each member fits on its own column subset and
    calibrates itself (LiftedLog), as the probe members did."""

    def __init__(self, horizon: int):
        self.horizon = horizon
        # The base member is the default model at that horizon: xgboost at 90 d, the
        # forest at 180 d (the forest-based blend is the one that held up on stores it
        # was not chosen on).
        base = random_forest.MODEL.build(horizon) if horizon in FOREST_BASE else xgboost_gbdt.build(horizon)
        self.members = [("base", base, False),
                        ("ext", xgboost_gbdt.build(horizon), True),
                        ("rf_ext", random_forest.MODEL.build(horizon), True)]
        self.cols_: dict[str, list[str]] = {}

    def fit(self, X, y, w=None, frame=None):
        base = [c for c in X.columns if not c.startswith(XF_PREFIX)]
        if len(base) == len(X.columns):
            print("    ! blend: no xf_* columns in the frame -- the dataset predates "
                  "them; every member trains on the standard features")
        for name, est, ext in self.members:
            self.cols_[name] = list(X.columns) if ext else base
            est.fit(X.select(self.cols_[name]), y, w, frame=frame)
        first = self.members[0][1]
        # `lifted_log.lift_stats` reads these off the top-level estimator.
        self.lift_coverage_, self.lifted_share_ = first.lift_coverage_, first.lifted_share_
        return self

    def predict(self, X) -> np.ndarray:
        preds = [np.clip(np.nan_to_num(np.asarray(est.predict(X.select(self.cols_[name])),
                                                  dtype=np.float64).ravel()), 0.0, None)
                 for name, est, _ in self.members]
        return np.mean(preds, axis=0)


MODEL = LabModel(
    name="blend",
    family="Ensemble (averaged trees)",
    blurb="""
    The mean of three tree fits: the default model at the horizon (xgboost at 90 d, the
    forest at 180 d), xgboost with in-stock and sibling stock-out features, and
    random_forest with both. No member reliably beats the default alone; the average can,
    because their errors are only partly correlated.
    """,
    pros=[
        "In the probes behind it, the one variant that beat the single model at 90 d on "
        "BOTH precision and inventory cost, with the variant fixed before scoring.",
        "Members are the standard estimators unchanged, so nothing new is being trusted "
        "except the averaging and the xf_* features.",
    ],
    cons=[
        "Noise at 30 d (not run there); at 180 d it does not win on every store, so a "
        "store's own board may keep the forest.",
        "Three fits per fold, the forest at 300k rows -- the slowest model on the board.",
        "Designed on one store; the other stores' picks are decided by their own boards.",
    ],
    build=lambda h, **kw: Blend(h),
    needs_frame=True,
    extended_features=True,
    horizons=(90, 180),
    notes={"members": ["xgboost_gbdt (standard)", "xgboost_gbdt + xf_*", "random_forest + xf_*"],
           "combine": "mean of calibrated member predictions"},
)

if __name__ == "__main__":
    run_model(MODEL)
