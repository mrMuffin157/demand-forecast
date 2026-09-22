"""`datacheck.py`: the dataset must keep up with the panel.

This pins a failure that is easy to build into a scheduled pipeline: rebuild the panel
and the features on every run but forget `model_lab/prepare_data.py`, and
`dev/forecast.py` keeps training on whichever `lab_dataset.parquet` was last written.
Nothing raises:

  * the forecast origin freezes, so `forecast_<origin>.csv` keeps the same name and each
    run overwrites the last rather than adding history;
  * the ORDER PLAN still moves week to week, because stock is read straight from
    `data/db/` and draws down -- so the output looks alive while the demand underneath
    it is months old;
  * `invariants()`'s open-vs-lab ordering check stays green, because the two dataset
    files go stale TOGETHER and their relative order is untouched;
  * the `STALE_DAYS` check stays green too, because it measures the panel against the
    wall clock, not the dataset against the panel.

A wrong input that produces a plausible forecast is the one failure mode nothing else
in this project can see, which is what `datacheck.py` exists for. Hence these.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# `dev/` is a folder of scripts, not a package.
_spec = importlib.util.spec_from_file_location("datacheck", ROOT / "dev" / "datacheck.py")
datacheck = importlib.util.module_from_spec(_spec)
sys.modules["datacheck"] = datacheck
_spec.loader.exec_module(datacheck)


def _fp(panel_max: str, open_max: str, lab_max: str = "2026-07-13") -> dict:
    """A fingerprint carrying only what the freshness checks read."""
    return {
        "panel": {"rows": 1, "skus": 1, "date_min": "2015-09-10", "date_max": panel_max},
        "lab_dataset": {"rows": 1, "skus": 1, "date_max": lab_max},
        "open_origins": {"rows": 1, "skus": 1, "date_max": open_max},
        "columns": [],
    }


def _find(results, key: str):
    return [(lvl, name, msg) for lvl, name, msg in results if name == key]


# The normal gap on a correct build: the panel ends on a Thursday and origins are weekly
# Mondays, so open_origins ends three days earlier.
def test_a_freshly_built_dataset_passes():
    assert _find(datacheck.invariants(_fp("2026-08-13", "2026-08-10")),
                 "dataset is behind the panel") == []


@pytest.mark.parametrize("panel_max", ["2026-08-10", "2026-08-11", "2026-08-16"])
def test_every_legitimate_within_one_origin_week_gap_passes(panel_max):
    """0-6 days is ordinary: the panel advances daily, the origins only weekly."""
    assert _find(datacheck.invariants(_fp(panel_max, "2026-08-10")),
                 "dataset is behind the panel") == []


def test_one_skipped_rebuild_warns():
    """7-13 days means the origin that should exist does not. WARN, so `--strict` stops
    the job -- a forecast off a dataset a week behind is answering last week's question."""
    hits = _find(datacheck.invariants(_fp("2026-08-20", "2026-08-10")),
                 "dataset is behind the panel")
    assert [h[0] for h in hits] == ["WARN"]


def test_a_frozen_dataset_fails():
    """The bug itself: the panel keeps moving, the dataset never rebuilds. Two origin
    weeks is past any legitimate explanation, so this is a FAIL and `--accept` refuses
    to bless the build."""
    hits = _find(datacheck.invariants(_fp("2026-09-21", "2026-08-10")),
                 "dataset is behind the panel")
    assert [h[0] for h in hits] == ["FAIL"]


def test_the_ordering_check_alone_would_not_have_caught_it():
    """Why this file exists. Freeze BOTH dataset files and advance only the panel: the
    open-vs-lab ordering check sees a perfectly ordered pair and says nothing."""
    fp = _fp("2026-09-21", "2026-08-10", lab_max="2026-07-13")
    assert _find(datacheck.invariants(fp), "open origins are not newer") == []
    assert [h[0] for h in _find(datacheck.invariants(fp),
                                "dataset is behind the panel")] == ["FAIL"]
