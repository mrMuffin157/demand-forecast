"""The product master has two sources, and they must ingest to the same frame.

`config.RAW_PRODUCT` prefers the database extract (`data/db/product.parquet`, produced by
`scripts/make_sample_data.py` for the sample or by your own extract) over a hand-exported
`data/raw/product_*.csv`, because a fresh data volume starts empty and the CSV is the one
input a scheduled job cannot refresh. That swap is only safe if the two produce an
identical product master: the panel and every model were built on the CSV's types (flags
as 0/1 Int64, codes like `tier` and `carton_size` as String).

The second guard is a narrow pull that selects only a few columns. Ingesting it would
"work" -- the CSV path only warns about missing columns -- and silently drop most of the
product features.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from demandml import config as C
from demandml import ingest

# One realistic row per shape the export produces: flags as 0/1, a null code, a datetime
# with and without fractional seconds.
ROWS = [
    {"id": "11", "sku": "AB-1", "is_whole_sale": 0, "is_pallet_ship": 1, "has_arrived": 1,
     "is_new_item": 0, "tier": None, "carton_size": 28, "channel_category": 10,
     "date_created": "2016-05-27 16:44:50", "date_on_market": "2016-05-27 16:44:50"},
    {"id": "12", "sku": "AB-2", "is_whole_sale": 1, "is_pallet_ship": 0, "has_arrived": 0,
     "is_new_item": 1, "tier": 2, "carton_size": 1, "channel_category": None,
     "date_created": "2022-10-03 00:00:00", "date_on_market": None},
]


def _frames(tmp_path: Path) -> tuple[Path, Path]:
    base = pl.DataFrame(ROWS)
    extra = {c: [1.5, 2.25] for c in C.PRODUCT_KEEP_COLS if c not in base.columns}
    base = base.with_columns(**{c: pl.Series(v) for c, v in extra.items()})
    csv = tmp_path / "product_2026.csv"
    base.write_csv(csv)
    # What a database driver typically hands back: flags as Boolean, codes as Int64,
    # timestamps as Datetime.
    flags = ["is_whole_sale", "is_pallet_ship", "has_arrived", "is_new_item"]
    db = base.with_columns(
        [pl.col(c).cast(pl.Boolean) for c in flags]
        + [pl.col(c).str.to_datetime("%Y-%m-%d %H:%M:%S") for c in ("date_created", "date_on_market")])
    pq = tmp_path / "product.parquet"
    db.write_parquet(pq)
    return csv, pq


def test_csv_and_pull_ingest_to_the_same_frame(tmp_path):
    csv, pq = _frames(tmp_path)
    a = pl.read_parquet(ingest.ingest_product(csv, tmp_path / "a.parquet", overwrite=True))
    b = pl.read_parquet(ingest.ingest_product(pq, tmp_path / "b.parquet", overwrite=True))
    assert_frame_equal(a, b)
    assert a.schema["is_whole_sale"] == pl.Int64, "flags must stay 0/1 as the models saw them"


def test_a_narrow_pull_refuses_rather_than_dropping_columns(tmp_path):
    _, pq = _frames(tmp_path)
    narrow = tmp_path / "narrow.parquet"
    pl.read_parquet(pq).select("id", "sku", "moq", "average_cost").write_parquet(narrow)
    with pytest.raises(ValueError, match="missing product columns"):
        ingest.ingest_product(narrow, tmp_path / "c.parquet", overwrite=True)


def test_a_volume_with_no_csvs_at_all_still_ingests(tmp_path, monkeypatch):
    """What a fresh data volume looks like: the database extract and nothing in raw/.

    If the product master and the weekly export were CSV-only, a first run on such a
    volume would stop at `demandml ingest`. The weekly one is simply skipped -- the daily
    pipeline never reads it.
    """
    _, pq = _frames(tmp_path)
    monkeypatch.setattr(C, "RAW_PRODUCT", pq)
    monkeypatch.setattr(C, "RAW_WEEKLY", None)
    monkeypatch.setattr(C, "PQ_PRODUCT", tmp_path / "product_out.parquet")
    monkeypatch.setattr(ingest, "ingest_daily", lambda overwrite=False: tmp_path / "daily.parquet")
    out = ingest.ingest_all()
    assert set(out) == {"product", "daily"}
    assert (tmp_path / "product_out.parquet").exists()
