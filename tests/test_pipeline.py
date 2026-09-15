"""Smoke tests for the data pipeline against the real Parquet artefacts.

Skipped automatically when the artefacts have not been built, so a fresh clone still has
a green test run.
"""

from __future__ import annotations

import polars as pl
import pytest

from demandml import config as C

pytestmark = pytest.mark.skipif(
    not C.PANEL_DAILY.exists(), reason="panel not built -- run `demandml panel`"
)


@pytest.fixture(scope="module")
def panel() -> pl.LazyFrame:
    return pl.scan_parquet(C.PANEL_DAILY)


def test_panel_has_one_row_per_sku_date(panel):
    n, n_unique = (
        panel.select(pl.len(), pl.struct(C.SKU, C.DATE).n_unique()).collect().row(0)
    )
    assert n == n_unique, "duplicate (sku, date) rows survived cleaning"


def test_panel_calendar_is_dense_per_sku(panel):
    """Every SKU must cover a gap-free daily range, or lag features are misaligned."""
    gaps = (
        panel.sort([C.SKU, C.DATE])
        .with_columns(
            (pl.col(C.DATE) - pl.col(C.DATE).shift(1).over(C.SKU)).dt.total_days().alias("d")
        )
        .filter(pl.col("d") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    assert gaps == 0, f"{gaps} calendar gaps in the panel"


def test_target_is_non_negative_and_finite(panel):
    bad = panel.filter(
        (pl.col(C.TARGET) < 0) | pl.col(C.TARGET).is_null() | pl.col(C.TARGET).is_infinite()
    ).select(pl.len()).collect().item()
    assert bad == 0


def test_censored_days_get_zero_weight(panel):
    bad = (
        panel.filter(
            pl.col("is_oos") & (pl.col(C.TARGET) <= 0) & (pl.col("sample_weight") > 0)
        )
        .select(pl.len())
        .collect()
        .item()
    )
    assert bad == 0, "stock-out zero days must carry zero sample weight"


def test_winsorized_target_never_exceeds_raw(panel):
    bad = (
        panel.filter(pl.col("units_winsorized") > pl.col(C.TARGET) + 1e-6)
        .select(pl.len())
        .collect()
        .item()
    )
    assert bad == 0


def test_trimming_removed_prelaunch_zeros(panel):
    """A SKU's first panel row should be a real sale, not a pre-launch zero."""
    firsts = panel.sort([C.SKU, C.DATE]).group_by(C.SKU).first().collect()
    zero_starts = (firsts.get_column(C.TARGET) == 0).sum()
    assert zero_starts == 0, f"{zero_starts} SKUs still start on a zero-demand day"


@pytest.mark.skipif(
    not (C.FEATURES_DAILY.exists() and any(C.FEATURES_DAILY.glob("*.parquet"))),
    reason="features not built",
)
def test_feature_table_matches_panel_rows(panel):
    n_panel = panel.select(pl.len()).collect().item()
    n_feat = pl.scan_parquet(C.FEATURES_DAILY / "*.parquet").select(pl.len()).collect().item()
    assert n_feat == n_panel


@pytest.mark.skipif(
    not (C.FEATURES_DAILY.exists() and any(C.FEATURES_DAILY.glob("*.parquet"))),
    reason="features not built",
)
def test_models_all_instantiate():
    from demandml.models import REGISTRY, get_model

    for name in REGISTRY:
        m = get_model(name)
        assert hasattr(m, "fit") and hasattr(m, "predict"), name


@pytest.mark.skipif(
    not (C.FEATURES_DAILY.exists() and any(C.FEATURES_DAILY.glob("*.parquet"))),
    reason="features not built",
)
def test_no_raw_same_day_column_reaches_the_model():
    """The allowlist regression test.

    An earlier denylist version of ``feature_columns`` let ``salescount`` and
    ``big_order_filter_sales_count`` -- raw copies of the target -- into the feature
    matrix. Any raw panel column that is not an explicitly-classified static attribute
    must be rejected.
    """
    from demandml.features import feature_columns

    lf = pl.scan_parquet(C.FEATURES_DAILY / "*.parquet")
    panel_cols = set(pl.scan_parquet(C.PANEL_DAILY).collect_schema().names())
    feats = set(feature_columns(lf))
    unclassified = sorted((feats & panel_cols) - C.STATIC_FEATURE_COLS - C.CALENDAR_FEATURE_COLS)
    assert not unclassified, (
        f"raw panel columns reached the feature matrix: {unclassified}. "
        "Classify them in config.STATIC_FEATURE_COLS or leave them rejected."
    )


@pytest.mark.skipif(
    not (C.FEATURES_DAILY.exists() and any(C.FEATURES_DAILY.glob("*.parquet"))),
    reason="features not built",
)
def test_target_columns_are_never_features():
    from demandml.features import feature_columns

    lf = pl.scan_parquet(C.FEATURES_DAILY / "*.parquet")
    for f in feature_columns(lf):
        assert not f.startswith(("y_h", "y_cum")), f
        assert f not in (C.TARGET, C.TARGET_FILTERED, "units_winsorized", "units_uncensored")
