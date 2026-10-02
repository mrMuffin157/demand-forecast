"""`forecast.capped_rows`: the serving fit's row cap must not choose SKUs by name.

The regression: `tr.tail(cap)` on a frame sorted by SKU kept the alphabetically last SKUs
and left the first ones out of training entirely. The cap is now the backtest's own rule
(a seeded random sample), so serving trains the way the model was chosen."""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev"))
_spec = importlib.util.spec_from_file_location("forecast_mod", ROOT / "dev" / "forecast.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

WEEKS = 100


def _frame():
    # SKU-major order, the way load_dataset hands it back: A's whole history, then B's, ...
    days = [dt.date(2020, 1, 6) + dt.timedelta(weeks=i) for i in range(WEEKS)]
    skus = ["A", "B", "C", "D"]
    return pl.DataFrame({"sku": [s for s in skus for _ in days], "date": days * len(skus),
                         "y_cum90": [1.0] * (WEEKS * len(skus))})


def test_the_cap_keeps_every_sku_and_every_era():
    df = _frame()
    got = F.capped_rows(df, 100)
    assert got.height == 100
    assert set(got["sku"]) == set(df["sku"]), "a cap must not drop a SKU because of its name"
    # a sample across the whole window, not its newest slice (which cost precision)
    assert got["date"].min() < dt.date(2020, 1, 6) + dt.timedelta(weeks=WEEKS // 2)


def test_the_cap_is_reproducible_and_date_ordered():
    a, b = F.capped_rows(_frame(), 100), F.capped_rows(_frame(), 100)
    assert a.equals(b)
    for cap in (None, 100, 10_000):
        assert F.capped_rows(_frame(), cap)["date"].is_sorted()
