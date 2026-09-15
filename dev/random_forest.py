"""RANDOM FOREST — hundreds of decision trees, each shown a different slice of the data.

    uv run python dev/random_forest.py
    uv run python dev/random_forest.py --max-train-rows 50000   # much faster

-----------------------------------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------------------------------
A single decision tree asks a chain of yes/no questions -- "is the 28-day average above
3.5?", "is it December?", "has this SKU been on sale for more than 400 days?" -- and each
leaf of the chain holds an average. One tree grown deep enough memorises the training
data; it is accurate on what it has seen and useless on anything new.

A random forest fixes that by growing 300 different trees, each deliberately handicapped
so that they make DIFFERENT mistakes:

    * each tree is trained on a random bootstrap sample of the rows
    * at each split, each tree may only consider a random subset of the features

Then it averages all 300 predictions. Individual errors are partly independent, so
averaging cancels much of the noise while keeping the signal. This is bagging, and the
key property is that it reduces variance without you having to tune anything carefully.

-----------------------------------------------------------------------------------
HOW IT IS SHAPED, AND WHY
-----------------------------------------------------------------------------------
* n_estimators = 300. More trees only ever helps accuracy (it is an average, it does not
  overfit with count) -- it just costs time. 300 is where the curve has flattened.
* max_features = "sqrt" (the square root of the feature count, per split). Forcing the trees apart is the whole
  mechanism. If every tree could see every feature they would all split on the 28-day
  average first and you would have 300 copies of one tree.
* min_samples_leaf = 20. With many SKU-days at zero and a long tail, leaves of 1-2
  rows would be fitting noise. 20 keeps every leaf average on real evidence.
* max_depth = 20, to bound both memory and the same over-fitting problem.
* Median imputation, because scikit-learn's forest cannot take NaN. This is a real
  handicap versus LightGBM/XGBoost, which route missing values down their own branch and
  learn from the fact that a value is missing.
* Raw units as the target of the bare estimator. A tree's prediction is an average of leaf members, so
  the scale problem that forces model_lab/ridge_linear.py into log space does not apply: a tree
  simply isolates the big SKUs into their own leaves. (The registered model nevertheless
  wraps it in `lifted_log.LiftedLog`, which fits log1p for precision reasons -- see below.)

-----------------------------------------------------------------------------------
HOW IT IS TRAINED
-----------------------------------------------------------------------------------
All 300 trees are grown independently and in parallel (n_jobs=-1) on each split's
training window, minimising squared error within each leaf. There is no sequential
correction step -- that is exactly what separates this from the boosting files.

Training rows are capped at 150,000 by default because this is by far the slowest model
in the folder; the boosted files handle the full window in less time.

-----------------------------------------------------------------------------------
RANDOM FOREST vs GRADIENT BOOSTING (the thing worth understanding)
-----------------------------------------------------------------------------------
Both are "lots of trees", and they are opposites:

    Random forest    grows trees in PARALLEL on random subsets and AVERAGES them.
                     Each tree is a full, independent guess. Reduces VARIANCE.
    Boosting         grows trees in SEQUENCE, each one fitted to the ERRORS the previous
                     ones left behind. Reduces BIAS.

The practical consequence: a forest is nearly impossible to break with bad settings but
plateaus early, while boosting reaches a lower error but will happily overfit if you let
it run too long. On tabular data like this, boosting almost always wins -- compare this
file's score with model_lab/lightgbm_gbdt.py and you are measuring exactly how much.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from common import LabModel, run_model
from lifted_log import LiftedLog

PARAMS = dict(
    n_estimators=300,
    max_depth=20,
    max_features="sqrt",
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=0,
)


class RandomForest:
    def __init__(self, horizon: int, **params):
        self.horizon = horizon
        self.params = {**PARAMS, **params}
        self.pipe_ = None
        self.features_: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        # Numeric only, and no NaN allowed -- both are sklearn-forest limitations that
        # the LightGBM/XGBoost files do not have.
        self.features_ = [c for c, t in X.schema.items()
                          if t.is_numeric() or t == pl.Boolean]
        self.pipe_ = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("est", RandomForestRegressor(**self.params)),
        ])
        Xn = X.select(self.features_).cast(pl.Float64).to_numpy()
        self.pipe_.fit(Xn, np.clip(y, 0, None), est__sample_weight=w)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        Xn = X.select(self.features_).cast(pl.Float64).to_numpy()
        return self.pipe_.predict(Xn)

    def importances(self) -> list[tuple[str, float]]:
        imp = self.pipe_.named_steps["est"].feature_importances_
        return sorted(zip(self.features_, imp.tolist()), key=lambda t: -t[1])[:25]


MODEL = LabModel(
    name="random_forest",
    family="Trees (bagging)",
    blurb="""
    300 decision trees grown in parallel on random row and feature subsets, then
    averaged. Captures non-linearity and interactions with almost no tuning, and is the
    reference point that shows how much sequential boosting is actually worth.
    """,
    pros=[
        "Captures non-linear effects and feature interactions automatically.",
        "Very forgiving: almost any reasonable settings give a reasonable model.",
        "Needs no scaling, no log transform, no distributional assumption.",
        "Hard to overfit by adding trees -- averaging only ever reduces variance.",
        "Gives usable feature importances for free.",
    ],
    cons=[
        "The slowest and heaviest model here: 300 deep trees over the full feature set.",
        "Averaging over-smooths: it systematically under-predicts genuine demand spikes, "
        "which is precisely where an inventory miss is expensive.",
        ("Squared-error leaves are the wrong loss for an intermittent, skewed count -- "
         "which is why it now fits log1p(units) (`lifted_log.py`) rather than raw units."),
        "Cannot extrapolate at all -- a leaf can only return an average of values it saw "
        "in training, so a SKU growing beyond anything in the training window is capped.",
        "scikit-learn's version cannot handle missing values or string categories, so it "
        "starts the race with less information than the boosted models.",
    ],
    # The forest fits log1p of the stock-out-lifted target and is scaled back to sales --
    # `lifted_log.py`. The log fit was a clear precision gain for the forest, largest at
    # the longest horizon (see docs/FINDINGS.md).
    build=lambda h, **kw: LiftedLog(RandomForest(h, **kw), h),
    needs_frame=True,
    default_max_train_rows=150_000,
    notes={**PARAMS, "target": "log1p(y_cum{h}_lift), calibrated to sales"},
)

if __name__ == "__main__":
    run_model(MODEL)
