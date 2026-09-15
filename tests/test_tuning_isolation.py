"""The hyper-parameter search must never score a row the leaderboard is computed from.

This is the same class of bug as tests/test_leakage.py guards against, and it fails the
same way: silently, and in the direction that makes the numbers look better. If the
search can see the test period it will select the settings that suit those 90 days, the
tuned board will improve, and nothing anywhere will complain — you would simply be
reporting a selection as if it were an estimate.

So these tests assert the geometry directly, on the real dataset and the real fold
boundaries, rather than trusting that dev/tune.py's cut is where its docstring says.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dev"))

from demandml import config as C  # noqa: E402

DATASET = C.PROCESSED / "lab_dataset.parquet"
pytestmark = pytest.mark.skipif(
    not DATASET.exists(),
    reason="lab_dataset.parquet not built; run model_lab/prepare_data.py",
)

HORIZONS = (30, 90, 180)
OUTER_FOLDS = 3
OUTER_TEST_DAYS = 90


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    # Only the columns the fold geometry depends on -- the full frame is large and
    # nothing here looks at a feature.
    cols = [C.DATE, C.SKU] + [f"y_cum{h}" for h in HORIZONS]
    return pl.read_parquet(DATASET, columns=cols)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_tuning_universe_ends_before_every_test_window(df, horizon):
    """No origin the search can see is inside, or after, any leaderboard test window."""
    from common import make_folds
    from tune import tuning_universe

    outer = make_folds(df, horizon, n_folds=OUTER_FOLDS, test_days=OUTER_TEST_DAYS)
    sub, cut = tuning_universe(df, horizon, OUTER_FOLDS, OUTER_TEST_DAYS)

    assert len(sub), "the tuning universe is empty"
    last_origin = sub.get_column(C.DATE).max()
    for f in outer:
        assert last_origin < f.test_start, (
            f"h={horizon}: the search can see origins up to {last_origin}, but outer "
            f"fold {f.index} starts testing on {f.test_start}"
        )
        # Stronger: the universe must sit inside every outer fold's TRAINING window, so
        # an inner test row is always a row the outer model was allowed to learn from.
        assert last_origin <= f.train_end, (
            f"h={horizon}: universe ends {last_origin}, past outer fold {f.index}'s "
            f"train_end {f.train_end} -- it is inside that fold's gap"
        )
    assert cut == min(f.train_end for f in outer)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_inner_folds_stay_inside_the_universe(df, horizon):
    """The search's own train/test windows are both entirely within the cut."""
    from common import make_folds
    from tune import inner_folds, tuning_universe

    sub, cut = tuning_universe(df, horizon, OUTER_FOLDS, OUTER_TEST_DAYS)
    folds = inner_folds(sub, horizon, n_folds=2, test_days=OUTER_TEST_DAYS)
    assert folds, f"h={horizon}: no inner folds fit before {cut}"

    for f in folds:
        assert f.test_end <= cut, (
            f"h={horizon}: inner fold {f.index} tests up to {f.test_end}, past the "
            f"cut at {cut}"
        )
        # The same horizon-sized gap the leaderboard uses, so an inner training row's
        # own target window cannot reach its inner test period either.
        gap = (f.test_start - f.train_end).days
        assert gap > horizon, (
            f"h={horizon}: inner fold {f.index} leaves only a {gap}-day gap"
        )


@pytest.mark.parametrize("horizon", HORIZONS)
def test_inner_and_outer_test_rows_never_overlap(df, horizon):
    """Set-level check on the actual (sku, date) rows, not just the dates.

    The window arithmetic above could be right and the slicing still wrong, so this
    compares the rows that each side would really score.
    """
    from common import make_folds, split
    from tune import inner_folds, tuning_universe

    outer = make_folds(df, horizon, n_folds=OUTER_FOLDS, test_days=OUTER_TEST_DAYS)
    sub, _ = tuning_universe(df, horizon, OUTER_FOLDS, OUTER_TEST_DAYS)
    inner = inner_folds(sub, horizon, n_folds=2, test_days=OUTER_TEST_DAYS)

    def rows(frame, folds):
        out = set()
        for f in folds:
            _, te = split(frame, f, horizon)
            out |= set(zip(te.get_column(C.SKU).to_list(),
                           te.get_column(C.DATE).to_list()))
        return out

    outer_test = rows(df, outer)
    inner_test = rows(sub, inner)
    assert outer_test and inner_test
    shared = outer_test & inner_test
    assert not shared, (
        f"h={horizon}: {len(shared)} rows are scored by BOTH the search and the "
        f"leaderboard"
    )


def test_the_two_modes_never_share_a_directory():
    """`initial` and `tuned` must write to disjoint places, or one overwrites the other.

    The whole before/after comparison rests on this: if a --tuned run could land in
    results/, the "before" half would be silently replaced by the "after" half and the
    diff would show nothing changed.
    """
    import common

    paths = set()
    for mode in common.MODES:
        common.set_mode(mode)
        for p in (common.results_dir(), common.figures_dir()):
            assert p not in paths, f"{mode} reuses {p}"
            paths.add(p)
    assert len(paths) == 2 * len(common.MODES)
    common.set_mode("initial")  # leave the module as we found it


def test_tuned_params_are_only_applied_when_asked():
    """A model built with no overrides is the model its own file describes.

    Tuning must not edit the model files, so `build(h)` with no keywords has to keep
    returning the estimator with the PARAMS dict as written -- that is what makes
    results/ reproducible from the source at any later date.
    """
    import importlib

    for name in ("xgboost_gbdt", "random_forest", "lgbm_quantile", "two_stage"):
        mod = importlib.import_module(name)
        shipped = dict(getattr(mod, "PARAMS", None) or getattr(mod, "BASE_PARAMS"))
        est = mod.MODEL.build(90)
        got = dict(getattr(est, "params", {}) or {})
        for k, v in shipped.items():
            assert got.get(k, v) == v, (
                f"{name}: build(90) with no overrides changed {k} "
                f"from {v!r} to {got.get(k)!r}"
            )
