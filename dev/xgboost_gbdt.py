"""XGBOOST — the control on model_lab/lightgbm_gbdt.py, and the file where a real bug was found.

    uv run python dev/xgboost_gbdt.py

-----------------------------------------------------------------------------------
WHAT THIS FILE IS FOR
-----------------------------------------------------------------------------------
Same data, same folds, same features and matched hyper-parameters as lightgbm_gbdt.py,
so that any difference left over is attributable to the library rather than to the
problem. That is the whole job: it is the control that tells you whether the LightGBM
result belongs to the data or to one implementation.

For a while it did not do that job. It scored roughly twice LightGBM's WAPE and swung
wildly across folds at the same horizon, and the cause sat open in this header as a
known bug. The cause has been found. It is one parameter, it has nothing to do with
XGBoost being worse at anything, and it is worth reading because the same trap is
waiting in any table that has a high-cardinality categorical in it.

-----------------------------------------------------------------------------------
THE BUG: ONE HIGH-CARDINALITY CATEGORICAL
-----------------------------------------------------------------------------------
`sku_family` is very nearly a SKU identifier: hundreds of levels, a median of one SKU
per family, and most families containing exactly one SKU.

XGBoost's categorical handling defaults to PARTITION-BASED splits. For a categorical
feature it ranks the levels by their gradient statistic and cuts that ranking, so ONE
split can send an arbitrary subset of the families left and the rest right. A few of
those and the tree has stored a per-family level table read straight off the training
window. Under the Tweedie log link that table is a MULTIPLICATIVE offset, and it does
not update when the trailing-average features move. So when the catalogue-wide level
falls between train and test -- which is exactly what a 3-month forecast origin looks
like -- every offset is stale and high, and everything is over-forecast at once.

That is precisely the symptom that was recorded here. On the worst fold the demand
level dropped sharply between the training window and the test window: LightGBM's mean
prediction tracked the drop via the trailing features, while XGBoost's stayed anchored
near the training level.

LightGBM is handed the identical column and survives it, because its categorical
defaults shrink each level's offset back toward the parent node: `cat_smooth`,
`cat_l2`, `min_data_per_group`, `max_cat_threshold`. XGBoost has no `cat_smooth`
equivalent. The only shrinkage in its category ranking is `reg_lambda`, and at 1.0
that is nothing against Tweedie hessians in the hundreds.

THE FIX is `max_cat_to_onehot`. It is the cardinality BELOW which XGBoost uses
one-category-vs-rest splits instead of partitioning. Set it above every cardinality in
the table and every categorical becomes one-vs-rest, so a single split can no longer
isolate an arbitrary subset: memorising hundreds of family levels would cost hundreds
of separate splits, which the tree budget does not stretch to. Small categoricals such
as `brand_id` (a handful of levels) are unaffected in practice -- one-vs-rest and
partitioning are nearly the same thing when there are only a few levels to choose from.

What the diagnosis on the worst fold showed, in order of how much it settles:

* With one-vs-rest splits, XGBoost lands next to LightGBM on the same data and folds,
  and its bias flips from a large over-forecast to roughly where LightGBM's sits.
* Dropping `sku_family`, or dropping every categorical, gives essentially the same
  result as the fix. With all categoricals stripped the two libraries agree almost
  exactly -- the control finally does its job, and the LightGBM result is a property of
  the data, not of LightGBM.
* Leaving `sku_family` as the ONLY categorical makes XGBoost worse than it ever was,
  so the other categorical columns are innocent.
* The damage scales with how wide a partition a single split is allowed to cut:
  unrestricted (the default) is worst, capping `max_cat_threshold` at a few levels per
  side recovers part of it, one-vs-rest recovers all of it. A dose-response like that
  is what makes the mechanism more than a story.

Two other knobs were tried and are NOT the fix: a very large `reg_lambda` changes
nothing, because global L2 lands on the numeric splits too and is still nothing against
Tweedie hessians; a very large `min_child_weight` makes it slightly worse, because it
constrains a child node's hessian sum rather than the per-level shrinkage.

And the column itself is not the problem -- the way it is SPLIT is. Feed the identical
`sku_family` in as a plain integer code instead of a category, so that a split can only
cut a contiguous range of an arbitrary numbering, and the failure disappears too.
Nothing was removed; the model simply can no longer express the level table.

WHERE IT LANDED
-----------------------------------------------------------------------------------
Across all horizons and splits, mean WAPE roughly halved, the bias went from a clear
over-forecast to slightly under, and the fold-to-fold spread collapsed. The erratic
behaviour was the level memorisation firing hard on some folds and not others; it is
gone with it.

That can put it at the top of the WAPE board, and the board should not be read that
way: its lead over random_forest was far smaller than its own fold-to-fold spread, and
it lost to random_forest outright at the longest horizon. By this folder's own rule
that margin establishes nothing. What the result does establish is the thing this file
exists for -- two independent boosting implementations, given the same data, folds and
features, now agree.

-----------------------------------------------------------------------------------
WHY FOUR HYPOTHESES MISSED IT
-----------------------------------------------------------------------------------
Every one of the original four was about the objective, the link, or the leaf
constraints. None of them was about categorical handling, so none of them could have
found it. Recorded here because the ruling-out is still worth having:

    hypothesis                              result
    --------------------------------------  ------------------------------------------
    base_score not initialised to the mean  no material change
    min_child_weight too weak               no material change
    log-link steps too large (the
      max_delta_step guard XGBoost applies
      to count:poisson by default)          MUCH WORSE, and worse again combined with
                                            a larger min_child_weight
    the Tweedie implementation itself       not it -- count:poisson and even
                                            reg:squarederror also over-forecast

Three more were ruled out on the way to the real cause:

    premature early stopping                no -- training ran well past the point
                                            where the validation curve had flattened
    depth-wise vs leaf-wise tree growth     no -- XGBoost on lossguide stays broken,
                                            and LightGBM held to a fixed depth stays
                                            fine. The growth policy is worth almost
                                            nothing either way.
    min_child_weight pushed much harder     no -- it gets WORSE. It constrains the
                                            child node's hessian sum, not the
                                            per-category shrinkage, so it is the
                                            wrong knob for this failure.

One genuine mis-specification found along the way is still true and still worth
knowing, even though it was not the bug: `min_child_weight` in XGBoost is a sum of
HESSIANS, not a row count. Under a log link the hessians scale with the target, so
`min_child_weight=20` here is a far weaker constraint than LightGBM's
`min_child_samples=100`. Matching those two parameters by their numbers is a mistake.

A note on the column itself: `sku_family` is mildly bad for LightGBM too. Dropping it
improves LightGBM a little on the worst fold -- same direction, a small fraction of the
damage. A near-unique identifier is a weak feature for any model that can memorise it.
That is a separate question from this file's bug and lightgbm_gbdt.py has not been
retuned for it.

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
Read model_lab/lightgbm_gbdt.py first -- the boosting idea is explained there and is
identical here. The one structural difference between the libraries:

    LightGBM  leaf-wise: repeatedly split the single most promising leaf anywhere in
              the tree. Deeper, more irregular trees. Faster.
    XGBoost   depth-wise: fill each level before starting the next. More balanced,
              usually more conservative.

On this problem that difference is worth almost nothing, which is itself a result: see
the lossguide row above.

Settings: Tweedie loss at variance power 1.2, learning rate 0.05, max_depth 8 (up to 256
leaves, which is the same region as LightGBM's 127), colsample 0.7, subsample 0.8,
native categoricals as ONE-VS-REST (see MAX_CAT_TO_ONEHOT), and native NaN handling.
Trained on each split's training window with stock-out days at weight 0 and the most
recent 10% held out for early stopping, exactly as the LightGBM file does.

One implementation detail worth keeping: XGBoost 3.x refuses to predict on a category it
did not see in training, so `fit` records the training category dtypes and `predict`
re-imposes them, mapping unseen levels to NaN instead of raising.

-----------------------------------------------------------------------------------
WHAT `MODEL.build` RETURNS: THE LIFTED LOG FIT
-----------------------------------------------------------------------------------
`MODEL.build` does not return the Tweedie fit described above. It wraps the same
estimator in `lifted_log.LiftedLog`: squared error on log1p of the stock-out-lifted target
(`y_cum{h}_lift`, built by `prepare_data.py`), predicted back and scaled by one factor
fitted against SALES on the training rows. Chosen for precision -- the share of SKUs
within +/-40% of what sold -- where the log fit helped and the lift helped again, on
most folds (see `lifted_log.py` and docs/FINDINGS.md). `--cost` still builds the plain
estimator with a pinball loss on raw units, and `XGBoostTweedie` itself is unchanged,
so anything that builds it directly still measures what it says it measures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from common import LabModel, run_model, target_quantile, to_pandas
from lifted_log import LiftedLog

#: Cardinality below which XGBoost uses one-category-vs-rest splits instead of
#: partition-based ones. Set ABOVE every categorical in this table (the widest is
#: `sku_family`, at hundreds of levels) so that no single split can carve an arbitrary
#: subset of a near-unique identifier out of the catalogue. This one parameter is the
#: difference between a broken model and a working one on the worst fold -- see the header.
MAX_CAT_TO_ONEHOT = 4096

PARAMS: dict[str, Any] = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.2,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 20.0,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 800,
    "n_jobs": -1,
    "tree_method": "hist",
    "enable_categorical": True,
    "max_cat_to_onehot": MAX_CAT_TO_ONEHOT,
    "random_state": 0,
}
EARLY_STOPPING_ROUNDS = 50
VAL_FRACTION = 0.10

#: The log fit, over PARAMS: squared error on log1p(units), so Tweedie's power goes too.
#: Applied LAST, so a tuned override found for the Tweedie fit cannot put it back.
LOG_PARAMS: dict[str, Any] = {"objective": "reg:squarederror", "tweedie_variance_power": None}


def build(h: int, **kw):
    if "quantile_alpha" in kw:       # --cost: pinball loss on raw units, never the log fit
        return XGBoostTweedie(h, **kw)
    return LiftedLog(XGBoostTweedie(h, **{**kw, **LOG_PARAMS}), h)


class XGBoostTweedie:
    def __init__(self, horizon: int, **params):
        self.horizon = horizon
        self.params = {**PARAMS, **params}
        self.model_ = None
        self.features_: list[str] = []
        self.cat_dtypes_: dict[str, Any] = {}
        self.empty_cats_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        import xgboost as xgb

        import pandas as pd

        self.features_ = list(X.columns)
        Xp = to_pandas(X)
        y = np.clip(y, 0, None)

        # Remember the exact category sets seen in training. Converting the test frame
        # independently would give it its own category codes, and XGBoost 3.x refuses to
        # predict on a category it never trained on ("Found a category not in the
        # training set"). Re-imposing these dtypes at predict time maps unseen levels to
        # NaN instead, which XGBoost handles natively.
        self.cat_dtypes_ = {c: Xp[c].dtype for c in Xp.columns
                            if isinstance(Xp[c].dtype, pd.CategoricalDtype)}
        # A categorical with NO categories (all null in some store's data) is rejected
        # outright by XGBoost ("must have at least one category"); LightGBM takes it. It
        # carries no information either way, so it goes in as an all-NaN float. A store
        # without such a column is unaffected.
        self.empty_cats_ = [c for c, d in self.cat_dtypes_.items() if len(d.categories) == 0]
        for c in self.empty_cats_:
            del self.cat_dtypes_[c]
            Xp[c] = np.nan

        n_val = int(len(Xp) * VAL_FRACTION)
        params = dict(self.params)
        fit_kw: dict[str, Any] = {}
        if n_val > 1000:
            Xf, yf = Xp.iloc[:-n_val], y[:-n_val]
            wf = None if w is None else w[:-n_val]
            params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
            fit_kw["eval_set"] = [(Xp.iloc[-n_val:], y[-n_val:])]
            fit_kw["verbose"] = False
        else:
            Xf, yf, wf = Xp, y, w

        self.model_ = xgb.XGBRegressor(**params)
        self.model_.fit(Xf, yf, sample_weight=wf, **fit_kw)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        import pandas as pd

        Xp = to_pandas(X)[self.features_]
        for c, dtype in self.cat_dtypes_.items():
            if c in Xp.columns:
                # Unseen levels -> NaN, not an error. Nulled explicitly first: letting
                # `astype(dtype)` drop them is deprecated in pandas 3 and will raise.
                col = Xp[c].astype(object)
                Xp[c] = pd.Categorical(col.where(col.isin(dtype.categories)), dtype=dtype)
        for c in self.empty_cats_:
            Xp[c] = np.nan
        return self.model_.predict(Xp)

    def importances(self) -> list[tuple[str, float]]:
        score = self.model_.get_booster().get_score(importance_type="gain")
        return sorted(score.items(), key=lambda t: -t[1])[:25]


MODEL = LabModel(
    name="xgboost_gbdt",
    family="Trees (gradient boosting)",
    blurb="""
    The same sequential boosting idea as lightgbm_gbdt.py, growing trees level by level
    instead of leaf by leaf, and set up as the control that tells you whether the
    LightGBM result belongs to the data or to one library. It does: strip the
    categorical columns and the two land almost exactly on top of each other.
    """,
    pros=[
        "The control that tells you whether the LightGBM result belongs to the data or "
        "to one library -- and it says the data. With categoricals stripped the two "
        "implementations agree almost exactly on the same folds.",
        "Same modern conveniences: Tweedie loss, native NaN handling, native categoricals.",
        "The most battle-tested boosting implementation there is.",
        "Its history in this folder is the most useful thing in it: a 2x WAPE gap that "
        "looked like a library difference turned out to be one parameter governing how "
        "a high-cardinality near-identifier gets split. See the file header.",
        "Depth-wise balanced trees are usually more conservative than leaf-wise growth "
        "-- though on this problem the growth policy is worth almost nothing.",
    ],
    cons=[
        "Dangerous defaults on high-cardinality categoricals. Partition-based splits "
        "plus no `cat_smooth` equivalent let it memorise a per-level offset from the "
        "training window, which reads as a catastrophic over-forecast the moment the "
        "level shifts. `max_cat_to_onehot` has to be set deliberately.",
        "`min_child_weight` is a sum of hessians, not a row count, so it is not "
        "comparable to LightGBM's `min_child_samples` and cannot be matched by number.",
        "Every disadvantage of lightgbm_gbdt.py applies here too -- tuning burden, "
        "opacity, no extrapolation, overfitting risk.",
        "Usually slower than LightGBM for the same accuracy on wide tabular data.",
        "Two boosted models this similar are one model for most purposes; the second "
        "one earns its place as a control, not as a candidate.",
        """Its rank on this board is not established. Its WAPE lead over `random_forest` is much smaller than its own split-to-split spread, which is this folder's own definition of a margin that proves nothing -- and it can lose to `random_forest` outright at the longest horizon.""",
    ],
    build=build,
    # LiftedLog reads `y_cum{h}_lift` from the training frame.
    needs_frame=True,
    # Under --cost, swap Tweedie (which estimates a conditional MEAN) for pinball loss at
    # the quantile the inventory cost actually minimises. This is the same substitution
    # lgbm_quantile makes, applied to the accurate model rather than a mid-pack one --
    # the lesson being that the quantile objective fixes calibration but cannot rescue a
    # weaker base model.
    #
    # `tweedie_variance_power` has to go to None explicitly: cost_params is merged OVER
    # PARAMS, so leaving it set would hand xgboost a Tweedie parameter alongside a
    # quantile objective.
    cost_params=lambda: {"objective": "reg:quantileerror",
                         "quantile_alpha": target_quantile(),
                         "tweedie_variance_power": None},
    notes={**PARAMS, **LOG_PARAMS, "target": "log1p(y_cum{h}_lift), calibrated to sales",
           "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "val_fraction": VAL_FRACTION},
)

if __name__ == "__main__":
    run_model(MODEL)
