"""`forecast.choose` picks each horizon's model on PRECISION (share within +/-40% of what
sold), and fits the multiplier on COST for whichever model it picked.

A board where the two rules disagree: `tight` is within +/-40% almost everywhere but
symmetric-noisy, so it costs more; `skewed` is exact on 90% of rows and far too low on the
other 10%, which is cheap on cost but leaves those rows outside +/-40%."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev"))
_spec = importlib.util.spec_from_file_location("forecast_pick", ROOT / "dev" / "forecast.py")
FC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FC)


def _board(tmp_path):
    rng = np.random.default_rng(3)
    preds = tmp_path / "_preds"
    preds.mkdir()
    parts = {"tight": [], "skewed": []}
    for f in range(1, 13):
        y = rng.uniform(50, 150, 400)
        parts["tight"].append(y * rng.uniform(0.62, 1.38, 400))
        low = rng.random(400) < 0.10
        parts["skewed"].append(np.where(low, y * 0.30, y * 1.0))
        for k in parts:
            pl.DataFrame({"sku": range(400), "h": 90, "fold": f, "actual": y,
                          "predicted": parts[k][-1]}).write_parquet(preds / f"{k}_{f}.tmp.parquet")
    for k in parts:
        pl.concat([pl.read_parquet(preds / f"{k}_{f}.tmp.parquet") for f in range(1, 13)]) \
            .write_parquet(preds / f"{k}.parquet")
        for f in range(1, 13):
            (preds / f"{k}_{f}.tmp.parquet").unlink()
    return preds


def _independent(preds, sel):
    w, c = {}, {}
    for k in ("tight", "skewed"):
        d = pl.read_parquet(preds / f"{k}.parquet").filter(pl.col("fold").is_in(sel))
        y, p = d["actual"].to_numpy(), d["predicted"].to_numpy()
        w[k] = FC.within40(y, p)
        c[k] = min(FC.cost(y, p * kk, 4.0, 1.0) for kk in FC.KS)
    return w, c


def test_the_pick_is_the_most_precise_model_and_k_stays_on_cost(tmp_path, monkeypatch):
    preds = _board(tmp_path)
    monkeypatch.setattr(FC, "results_dir", lambda: tmp_path)
    got = FC.choose(90, 4.0, 1.0)
    w, c = _independent(preds, list(range(1, 12)))
    assert got["pick_rule"] == "precision"
    assert got["model"] == max(w, key=w.get)
    assert got["cost_pick"] == min(c, key=c.get)
    assert got["model"] != got["cost_pick"], "fixture must make the two rules disagree"
    # k is the cost-optimal multiplier OF THE PICKED MODEL, fitted on every fold
    d = pl.read_parquet(preds / f"{got['model']}.parquet")
    y, p = d["actual"].to_numpy(), d["predicted"].to_numpy()
    assert got["k"] == float(FC.KS[int(np.argmin([FC.cost(y, p * kk, 4.0, 1.0) for kk in FC.KS]))])


def test_the_cost_rule_is_one_constant_away(tmp_path, monkeypatch):
    _board(tmp_path)
    monkeypatch.setattr(FC, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(FC, "PICK_RULE", "cost")
    got = FC.choose(90, 4.0, 1.0)
    assert got["model"] == got["cost_pick"]
