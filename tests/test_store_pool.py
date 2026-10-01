"""Pooled training across stores (`DEMANDML_POOL`, dev/common.py).

Stores that share one seasonal shape can benefit from one model trained on all of them
(see docs/FINDINGS.md), so with pooling on every store's models train on every store's
rows -- and still score and forecast only their own. These tests pin the ways that can go
wrong silently:

  * unset, it must change nothing (no marker columns, no extra rows);
  * a pooled row must never reach past the fold's train_end, or a backtest scores a
    model that saw the test period through another store;
  * each store is capped on its own, and a missing store is skipped loudly.

Everything writes under `tmp_path`.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev"))

import common  # noqa: E402
import market as M  # noqa: E402

D0 = dt.date(2025, 1, 6)


def _dataset(n_weeks: int, sku: str, units: float) -> pl.DataFrame:
    dates = [D0 + dt.timedelta(weeks=i) for i in range(n_weeks)]
    return pl.DataFrame({
        "sku": sku, "date": dates, "units_mean_28": units, "doy": [d.timetuple().tm_yday
                                                                   for d in dates],
        "y_cum30": units * 30, "y_cum90": units * 90, "sample_weight": 1.0,
    }).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """A US dataset as this process's own, and CA / AU datasets in their own dirs."""
    roots = {k: tmp_path / k for k in ("us", "ca", "au")}
    for k, units in (("us", 10.0), ("ca", 2.0), ("au", 1.0)):
        (roots[k] / "processed").mkdir(parents=True)
        _dataset(40, "A", units).write_parquet(roots[k] / "processed" / "lab_dataset.parquet")
    meta = {"features": ["units_mean_28", "doy"]}
    (roots["us"] / "processed" / "lab_dataset_meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(common, "DATASET", roots["us"] / "processed" / "lab_dataset.parquet")
    monkeypatch.setattr(common, "DATASET_META",
                        roots["us"] / "processed" / "lab_dataset_meta.json")
    monkeypatch.setenv("DEMANDML_MARKET", "us")
    monkeypatch.delenv("DEMANDML_POOL", raising=False)
    return roots


def _pool_env(roots: dict[str, Path], **extra: str) -> str:
    return ",".join([f"{k}={v}" for k, v in roots.items()] +
                    [f"{k}={v}" for k, v in extra.items()])


def test_unset_changes_nothing(stores):
    df, feats = common.load_dataset()
    assert feats == ["units_mean_28", "doy"]
    assert not any(c.startswith("store_") for c in df.columns)
    assert common.load_pool(feats) == {}


def test_pooled_rows_are_marked_and_stop_at_train_end(stores, monkeypatch):
    monkeypatch.setenv("DEMANDML_POOL", _pool_env(stores))
    df, feats = common.load_dataset()
    assert feats[-len(common.STORE_COLS):] == common.STORE_COLS
    assert df.get_column("store_us").all() and not df.get_column("store_ca").any()

    pool = common.load_pool(feats)
    assert sorted(pool) == ["au", "ca"]              # never this store's own rows again
    assert "y_cum90" in pool["ca"].columns and "y_cum30" in pool["ca"].columns
    assert pool["ca"].get_column("store_ca").all()

    train_end = D0 + dt.timedelta(weeks=20)
    extra = common.pool_training_rows(pool, 90, train_end, max_rows=None)
    assert extra.get_column("date").max() <= train_end
    tr = common.with_pool(df.filter(pl.col("date") <= train_end), extra)
    assert tr.height == 3 * 21
    assert tr.get_column("date").is_sorted()          # early-stopping tail = latest dates
    assert set(tr.columns) == set(df.columns)


def test_each_store_is_capped_on_its_own(stores, monkeypatch):
    monkeypatch.setenv("DEMANDML_POOL", _pool_env(stores))
    _, feats = common.load_dataset()
    pool = common.load_pool(feats)
    end = D0 + dt.timedelta(weeks=39)
    sampled = common.pool_training_rows(pool, 90, end, max_rows=5, seed=1)
    assert sampled.group_by("store_ca").len().get_column("len").to_list() == [5, 5]
    recent = common.pool_training_rows(pool, 90, end, max_rows=5, recent=True)
    assert recent.get_column("date").min() == D0 + dt.timedelta(weeks=35)


def test_a_missing_store_is_skipped_loudly_not_fatal(stores, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DEMANDML_POOL", _pool_env(stores, us2=str(tmp_path / "nope")))
    _, feats = common.load_dataset()
    pool = common.load_pool(feats)
    assert sorted(pool) == ["au", "ca"]
    assert "us2 skipped" in capsys.readouterr().out


def test_parse_pool_refuses_a_store_it_does_not_know():
    assert M.parse_pool(None) == {} and M.parse_pool("") == {}
    assert M.parse_pool("us=/data, ca=/data/ca") == {"us": "/data", "ca": "/data/ca"}
    with pytest.raises(SystemExit):
        M.parse_pool("uk=/data/uk")
    with pytest.raises(SystemExit):
        M.parse_pool("/data/ca")
