"""The forward forecast reads a different file from the one models train on. These are
the properties that make that safe.

The failure mode being guarded against is not a crash. It is a forecast that looks
perfectly reasonable and is subtly wrong because the row it was computed from went
through a different code path than the rows the model learned on -- classic train/serve
skew, which produces plausible numbers and no error message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dev"))

from demandml import config as C  # noqa: E402

DATASET = C.PROCESSED / "lab_dataset.parquet"
OPEN = C.PROCESSED / "open_origins.parquet"
META = C.PROCESSED / "lab_dataset_meta.json"
HORIZONS = (30, 90, 180)

pytestmark = pytest.mark.skipif(
    not (DATASET.exists() and OPEN.exists()),
    reason="dataset not built; run model_lab/prepare_data.py")


@pytest.fixture(scope="module")
def frames():
    return pl.read_parquet(DATASET), pl.read_parquet(OPEN)


def test_open_origins_carry_no_targets(frames):
    """If an open origin had a target it would belong in the training set, and its
    presence here would mean the split leaked a scorable row out of the leaderboard."""
    _, op = frames
    for h in HORIZONS:
        col = f"y_cum{h}"
        assert op.get_column(col).null_count() == len(op), (
            f"{col} is populated for some open origins -- they are not open")


def test_training_and_open_rows_are_disjoint(frames):
    """No (sku, date) may appear in both, or a row would be both trained on and
    forecast from."""
    df, op = frames
    a = df.select(C.SKU, C.DATE).unique()
    b = op.select(C.SKU, C.DATE).unique()
    assert a.join(b, on=[C.SKU, C.DATE], how="inner").height == 0


def test_open_origins_are_newer_than_anything_trainable(frames):
    """The whole point: the newest row a model can TRAIN on is already a full horizon
    stale, because its target had to have happened. If the open set were not newer, the
    forecast would be starting from the same stale place and buying nothing."""
    df, op = frames
    assert op.get_column(C.DATE).max() > df.get_column(C.DATE).max()


def test_open_origins_have_every_feature_the_models_use(frames):
    """Same columns, or the served matrix is not the trained matrix."""
    df, op = frames
    feats = [c for c in json.loads(META.read_text())["features"] if c in df.columns]
    missing = [c for c in feats if c not in op.columns]
    assert not missing, f"open origins lack {len(missing)} feature(s): {missing[:5]}"
    for c in feats:
        assert op.schema[c] == df.schema[c], f"{c} has a different dtype when served"


def test_open_origins_are_not_mostly_empty(frames):
    """A feature that is populated in training but always null when served is
    train/serve skew, and no dtype check catches it.

    The comparison has to be like-for-like on the calendar. Some features are seasonal
    by construction -- `days_to_christmas` is null outside December -- so comparing
    summer origins against a training set spanning every December flags a column that
    is behaving exactly as designed. Restricting the training side to the same
    months-of-year as the open origins is what makes the check mean "this column is
    missing when served", rather than "it is not December".
    """
    df, op = frames
    feats = [c for c in json.loads(META.read_text())["features"] if c in df.columns]
    months = op.get_column(C.DATE).dt.month().unique().to_list()
    same_season = df.filter(pl.col(C.DATE).dt.month().is_in(months))
    assert same_season.height, "no training rows share a month with the open origins"
    bad = [c for c in feats
           if op.get_column(c).null_count() == len(op)
           and same_season.get_column(c).null_count() < same_season.height]
    assert not bad, (
        f"{len(bad)} feature(s) null for every open origin but populated in training "
        f"at the same time of year: {bad[:5]}")


def test_one_forecastable_row_per_sku_is_available(frames):
    """forecast.py takes the newest origin per SKU. That has to exist for a real share
    of the catalogue, or the forecast silently covers a fraction of the range."""
    df, op = frames
    covered = op.get_column(C.SKU).n_unique()
    total = df.get_column(C.SKU).n_unique()
    assert covered / total > 0.5, (
        f"only {covered}/{total} SKUs have a forecastable origin")
