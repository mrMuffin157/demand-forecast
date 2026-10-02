"""The extended features (`demandml/xfeatures.py`) and who may see them.

Two ways these go wrong without an error: a window that reads past the origin (a timing
bug makes scores BETTER, so nobody notices), and the new columns leaking into the models
that were measured without them (xgboost_gbdt, random_forest and every diagnostic script
would silently change)."""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev"))

from demandml import xfeatures as XF  # noqa: E402

import common  # noqa: E402


def _panel(n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    d0 = dt.date(2024, 1, 1)
    rows = []
    for sku in ("ZB-101-G", "ZB-101-W", "ZB-101-K", "ZC-102-1"):
        for i in range(n_days):
            oos = bool(rng.random() < 0.15)
            rows.append((sku, d0 + dt.timedelta(days=i),
                         0.0 if oos else float(rng.poisson(5)), oos))
    return pl.DataFrame(rows, schema=["sku", "date", "units", "is_oos"], orient="row")


def test_every_name_is_prefixed_and_none_is_a_target():
    assert all(n.startswith(XF.PREFIX) for n in XF.NAMES)
    assert not any("y_cum" in n or n.startswith("y_h") for n in XF.NAMES)
    got = XF.frame(_panel().lazy())
    assert set(XF.NAMES) <= set(got.columns)


def test_no_feature_reads_a_day_after_its_origin():
    """Rewrite everything after the cut -- units and stock-outs, every SKU -- and nothing
    on or before the cut may move."""
    p = _panel()
    cut = dt.date(2024, 9, 1)
    q = p.with_columns(
        pl.when(pl.col("date") > cut).then(pl.col("units") * 7 + 3).otherwise(pl.col("units")).alias("units"),
        pl.when(pl.col("date") > cut).then(~pl.col("is_oos")).otherwise(pl.col("is_oos")).alias("is_oos"))
    a = XF.frame(p.lazy()).filter(pl.col("date") <= cut).sort("sku", "date")
    b = XF.frame(q.lazy()).filter(pl.col("date") <= cut).sort("sku", "date")
    for c in XF.NAMES:
        assert a[c].fill_null(-1).equals(b[c].fill_null(-1)), f"{c} reads the future"


def test_siblings_share_a_parent_and_never_include_the_sku_itself():
    p = _panel()
    day = dt.date(2024, 6, 1)
    # Only the grey variant out on `day`: its siblings see a stock-out, it does not.
    p = p.with_columns(pl.when(pl.col("date") == day).then(pl.col("sku") == "ZB-101-G")
                       .otherwise(pl.col("is_oos")).alias("is_oos"))
    got = XF.frame(p.lazy()).filter(pl.col("date") == day)
    now = dict(zip(got["sku"], got[f"{XF.PREFIX}sib_oos_now"]))
    assert now["ZB-101-G"] == 0.0
    assert now["ZB-101-W"] > 0 and now["ZB-101-K"] > 0
    assert now["ZC-102-1"] is None          # no sibling: unknown, not zero
    sibs = dict(zip(got["sku"], got[f"{XF.PREFIX}n_sibs"]))
    assert sibs["ZB-101-W"] == 2 and sibs["ZC-102-1"] == 0


def test_only_a_model_that_asks_sees_the_extended_features():
    feats = ["units_mean_28", "ly_cum90", f"{XF.PREFIX}instock_rate_28"]
    xgb = importlib.import_module("xgboost_gbdt").MODEL
    rf = importlib.import_module("random_forest").MODEL
    blend = importlib.import_module("blend").MODEL
    assert common.model_features(xgb, feats) == feats[:2]
    assert common.model_features(rf, feats) == feats[:2]
    assert common.model_features(blend, feats) == feats


def test_the_blend_base_member_trains_without_them():
    """`base` must be exactly today's xgboost: the extended columns are dropped again."""
    B = importlib.import_module("blend")
    seen = {}

    class Rec:
        lift_coverage_, lifted_share_ = 1.0, 0.2

        def __init__(self, name):
            self.name = name

        def fit(self, X, y, w=None, frame=None):
            seen[self.name] = list(X.columns)

        def predict(self, X):
            return np.full(X.height, {"base": 1.0, "ext": 2.0, "rf_ext": 6.0}[self.name])

    b = B.Blend(90)
    b.members = [(n, Rec(n), ext) for n, _, ext in b.members]
    X = pl.DataFrame({"units_mean_28": [1.0, 2.0], f"{XF.PREFIX}instock_rate_28": [1.0, 2.0]})
    b.fit(X, np.ones(2), np.ones(2), frame=X)
    assert seen["base"] == ["units_mean_28"]
    assert seen["ext"] == seen["rf_ext"] == list(X.columns)
    assert np.allclose(b.predict(X), 3.0)       # the plain mean of the three
