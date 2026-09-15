"""XGBOOST — the control on lightgbm_gbdt.py, and the file where a real bug was found.

    uv run python model_lab/xgboost_gbdt.py

-----------------------------------------------------------------------------------
WHAT THIS FILE IS FOR
-----------------------------------------------------------------------------------
Same data, same folds, same features and matched hyper-parameters as lightgbm_gbdt.py,
so that any difference left over is attributable to the library rather than to the
problem. That is the whole job: it is the control that tells you whether the LightGBM
result belongs to the data or to one implementation.

For a while it did not do that job. It scored roughly twice LightGBM's WAPE, and swung
wildly from fold to fold at the same horizon. The cause has been found. It is one
parameter, it has nothing to do with XGBoost being worse at anything, and it is worth
reading because the same trap is waiting in any table that has a high-cardinality
categorical in it.

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

That is precisely the symptom that was seen. On the worst fold, where demand in the
test window ran far below the training window's level, LightGBM's mean prediction
tracked the shift through the features, while XGBoost's stayed anchored near the
training level.

LightGBM is handed the identical column and survives it, because its categorical
defaults shrink each level's offset back toward the parent node: `cat_smooth=10`,
`cat_l2=10`, `min_data_per_group=100`, `max_cat_threshold=32`. XGBoost has no
`cat_smooth` equivalent. The only shrinkage in its category ranking is `reg_lambda`,
and at 1.0 that is nothing against Tweedie hessians, which grow with the target.

THE FIX is `max_cat_to_onehot`. It is the cardinality BELOW which XGBoost uses
one-category-vs-rest splits instead of partitioning. Set it above every cardinality in
the table and every categorical becomes one-vs-rest, so a single split can no longer
isolate an arbitrary subset: memorising hundreds of family levels would cost hundreds of
separate splits, which the tree budget does not stretch to. Small categoricals such as
`brand_id` (a handful of levels) are unaffected in practice -- one-vs-rest and
partitioning are nearly the same thing when there are only a few levels to choose from.

What the experiments on the worst fold showed, qualitatively:

  * With the fix, XGBoost drops from far worse than LightGBM to level with it, and its
    bias flips from a large over-forecast to slightly under, like LightGBM's.
  * Dropping `sku_family` alone does about as well as the fix; dropping every
    categorical does marginally better still, and LightGBM with every categorical
    dropped lands on essentially the same number.
  * Leaving `sku_family` in as the ONLY categorical makes XGBoost worse than it ever
    was -- so the other categorical columns are innocent.

The damage scales with how much of an arbitrary subset a single split is allowed to
encode, which is what makes the mechanism more than a story: unrestricted partitions
are worst, capping each side of a partition at a few levels recovers part of the gap,
and one-vs-rest recovers all of it.

Two other knobs were tried and are NOT the fix: a very large `reg_lambda` makes things
worse, because global L2 lands on the numeric splits too and is still nothing against
Tweedie hessians; a very large `min_child_weight` also makes things worse, because it
constrains a child node's hessian sum rather than the per-level shrinkage.

And the column itself is not the problem -- the way it is SPLIT is. Feed the identical
`sku_family` in as a plain integer code instead of a category, so that a split can only
cut a contiguous range of an arbitrary numbering, and the failure disappears too.
Nothing was removed; the model simply can no longer express the level table.

What settles it: strip the categoricals and the two libraries land on essentially the
same WAPE -- the control finally does its job, and the LightGBM result is a property of
the data, not of LightGBM.

WHERE IT LANDED
-----------------------------------------------------------------------------------
Across all three horizons and all three splits, mean WAPE roughly halved, the bias went
from a clear over-forecast to slightly under, and the fold-to-fold spread collapsed. The
erratic behaviour was the level memorisation firing hard on some folds and not others;
it is gone with it.

That can put it at or near the top of the board, and the board should not be read that
way: its lead over random_forest is far smaller than its own split spread, and it can
lose to random_forest outright at 6 months. By this folder's own rule that margin
establishes nothing. What the number does establish is the thing this file exists for
-- two independent boosting implementations, given the same data, folds and features,
now agree.

-----------------------------------------------------------------------------------
WHY THE FIRST HYPOTHESES MISSED IT
-----------------------------------------------------------------------------------
Every one of the original hypotheses was about the objective, the link, or the leaf
constraints. None of them was about categorical handling, so none of them could have
found it. Recorded here because the ruling-out is still worth having:

    hypothesis                              result
    --------------------------------------  ------------------------------------------
    base_score not initialised to the mean  no meaningful change
    min_child_weight too weak (-> 500/5000) no meaningful change
    log-link steps too large (max_delta_step
      = 0.7, the guard XGBoost applies to
      count:poisson by default)             MUCH WORSE, and worse again combined
                                            with a larger min_child_weight
    the Tweedie implementation itself       not it -- count:poisson and even
                                            reg:squarederror also over-forecast
    premature early stopping                no -- it ran well short of the round cap
                                            and the validation curve had flattened
    depth-wise vs leaf-wise tree growth     no -- XGBoost on lossguide stays broken,
                                            and LightGBM held to max_depth=8 stays
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
Read lightgbm_gbdt.py first -- the boosting idea is explained there and is identical
here. The one structural difference between the libraries:

    LightGBM  leaf-wise: repeatedly split the single most promising leaf anywhere in
              the tree. Deeper, more irregular trees. Faster.
    XGBoost   depth-wise: fill each level before starting the next. More balanced,
              usually more conservative.

On this problem that difference is worth almost nothing, which is itself a result: see
the growth-policy row above.

Settings: Tweedie loss at variance power 1.2, learning rate 0.05, max_depth 8 (up to 256
leaves, which is the same region as LightGBM's 127), colsample 0.7, subsample 0.8,
native categoricals as ONE-VS-REST (see MAX_CAT_TO_ONEHOT), and native NaN handling.
Trained on each split's training window with stock-out days at weight 0 and the most
recent 10% held out for early stopping, exactly as the LightGBM file does.

One implementation detail worth keeping: XGBoost 3.x refuses to predict on a category it
did not see in training, so `fit` records the training category dtypes and `predict`
re-imposes them, mapping unseen levels to NaN instead of raising.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from common import LabModel, run_model, to_pandas

#: Cardinality below which XGBoost uses one-category-vs-rest splits instead of
#: partition-based ones. Set ABOVE every categorical in this table (the widest is
#: `sku_family`, with hundreds of levels) so that no single split can carve an arbitrary
#: subset of a near-unique identifier out of the catalogue. On the worst fold this one
#: parameter was the difference between a broken model and one level with LightGBM --
#: see the header.
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


class XGBoostTweedie:
    def __init__(self, horizon: int, **params):
        self.horizon = horizon
        self.params = {**PARAMS, **params}
        self.model_ = None
        self.features_: list[str] = []
        self.cat_dtypes_: dict[str, Any] = {}

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
        Xp = to_pandas(X)[self.features_]
        for c, dtype in self.cat_dtypes_.items():
            if c in Xp.columns:
                Xp[c] = Xp[c].astype(dtype)  # unseen levels -> NaN, not an error
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
    categorical columns and the two land on essentially the same WAPE.
    """,
    pros=[
        "The control that tells you whether the LightGBM result belongs to the data or "
        "to one library -- and it says the data. With categoricals stripped the two "
        "implementations agree almost exactly on the same folds.",
        "Same modern conveniences: Tweedie loss, native NaN handling, native categoricals.",
        "The most battle-tested boosting implementation there is.",
        "Its history in this folder is the most useful thing in it: a large WAPE gap "
        "that looked like a library difference turned out to be one parameter governing "
        "how a high-cardinality near-identifier gets split. See the file header.",
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
        "Its rank on this board is not established. Its lead over `random_forest` is "
        "far smaller than its own split spread, which is this folder's own definition "
        "of a margin that proves nothing -- and it can lose to `random_forest` outright "
        "at the 6-month horizon.",
    ],
    build=lambda h: XGBoostTweedie(h),
    notes={**PARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
           "val_fraction": VAL_FRACTION},
)

if __name__ == "__main__":
    run_model(MODEL)
