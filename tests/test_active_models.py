"""`DEMANDML_MODELS`: a production job can narrow the board; the lab runs all of it.

The run half is a convenience. The READ half is the guard: a model the retrain stops
running keeps its old backtest on disk, and if `forecast.choose` or a board still read
it, a stale model fitted on old data could win and be used.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "dev" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


import common  # noqa: E402

FC = _load("forecast")
CM = _load("compare_models")
CB = _load("cost_board")

PROD = "moving_average,random_forest,xgboost_gbdt"


def test_unset_means_no_restriction():
    assert common.active_models() is None
    assert all(common.model_allowed(n) for n in (*common.SHORTLIST, "anything"))


def test_the_shortlist_is_the_board():
    assert tuple(Path(f).stem for f in CM.MODEL_FILES) == common.SHORTLIST


def test_names_outside_the_shortlist_are_refused(monkeypatch):
    for bad in ("xgboost", "random_forest,xgb_gbdt", " , "):
        monkeypatch.setenv("DEMANDML_MODELS", bad)
        with pytest.raises(SystemExit):
            common.active_models()


def test_run_all_runs_only_the_named_models(monkeypatch):
    monkeypatch.setenv("DEMANDML_MODELS", PROD)
    ran = []
    monkeypatch.setattr(CM.subprocess, "run",
                        lambda a, **k: ran.append(Path(a[1]).stem) or
                        type("R", (), {"returncode": 0})())
    CM.run_all([])
    assert ran == ["moving_average", "random_forest", "xgboost_gbdt"]


def _stale_board(tmp_path: Path) -> None:
    """`two_stage` is perfect and left over from an older retrain; the other two are
    fresh and middling. Every reader must behave as if `two_stage` were not there."""
    rng = np.random.default_rng(0)
    preds = tmp_path / "_preds"
    preds.mkdir()
    for name in ("two_stage", "random_forest", "xgboost_gbdt"):
        parts = []
        for f in range(1, 13):
            y = rng.uniform(50, 150, 200)
            p = y if name == "two_stage" else y * rng.uniform(0.6, 1.4, 200)
            parts.append(pl.DataFrame({"sku": range(200), "h": 90, "fold": f,
                                       "actual": y, "predicted": p}))
        pl.concat(parts).write_parquet(preds / f"{name}.parquet")
        (tmp_path / f"{name}.json").write_text(json.dumps(
            {"model": name, "horizons": {"90": {"wape": 0.0 if name == "two_stage" else 0.2,
                                                "wape_sd": 0.01}}}))


def test_a_stale_model_cannot_be_picked(tmp_path, monkeypatch):
    _stale_board(tmp_path)
    monkeypatch.setattr(FC, "results_dir", lambda: tmp_path)
    assert FC.choose(90, 4.0, 1.0)["model"] == "two_stage"         # the lab: all of it
    monkeypatch.setenv("DEMANDML_MODELS", PROD)
    assert FC.choose(90, 4.0, 1.0)["model"] in ("random_forest", "xgboost_gbdt")


def test_the_boards_leave_a_stale_model_off(tmp_path, monkeypatch):
    _stale_board(tmp_path)
    monkeypatch.setattr(CB, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(CM, "results_dir", lambda: tmp_path)
    monkeypatch.setenv("DEMANDML_MODELS", PROD)
    assert sorted(CB.load_preds()) == ["random_forest", "xgboost_gbdt"]
    assert sorted(r["model"] for r in CM.load_results()) == ["random_forest", "xgboost_gbdt"]


def test_choose_flags_the_moving_average_when_it_wins(tmp_path, monkeypatch):
    """A narrowed board keeps the floor in the race; when it wins, the pick says so."""
    rng = np.random.default_rng(2)
    preds = tmp_path / "_preds"
    preds.mkdir()
    for name, noise in (("moving_average", 0.05), ("xgboost_gbdt", 0.4)):
        parts = []
        for f in range(1, 13):
            y = rng.uniform(50, 150, 200)
            parts.append(pl.DataFrame({"sku": range(200), "h": 90, "fold": f, "actual": y,
                                       "predicted": y * rng.uniform(1 - noise, 1 + noise, 200)}))
        pl.concat(parts).write_parquet(preds / f"{name}.parquet")
    monkeypatch.setattr(FC, "results_dir", lambda: tmp_path)
    pick = FC.choose(90, 4.0, 1.0)
    assert pick["model"] == "moving_average" and pick["baseline_wins"]
    (preds / "moving_average.parquet").rename(preds / "random_forest.parquet")
    pick = FC.choose(90, 4.0, 1.0)
    assert not pick["baseline_wins"] and pick["cheaper_than_baseline"] is None


def test_run_all_reports_a_model_that_failed(monkeypatch):
    """A model that dies (e.g. an OOM kill) must not leave the retrain looking
    green: its results file from the previous retrain would be picked up as if fresh."""
    monkeypatch.setenv("DEMANDML_MODELS", PROD)
    monkeypatch.setattr(CM.subprocess, "run",
                        lambda a, **k: type("R", (), {
                            "returncode": -9 if Path(a[1]).stem == "xgboost_gbdt" else 0})())
    assert CM.run_all([]) == ["xgboost_gbdt"]
