"""Rolling-origin (expanding-window) backtest splits.

Random k-fold on a panel like this is self-deception: the same SKU's neighbouring days
land in train and test, lag features make them near-duplicates, and the score comes out
two to three times better than anything you would see in production. Every split in this
module is a clean cut in time, applied simultaneously across all SKUs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from . import config as C


@dataclass(frozen=True)
class Fold:
    """One backtest fold. ``train_end`` is the last origin date allowed in training."""

    index: int
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date

    def describe(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start}..{self.train_end} "
            f"| test {self.test_start}..{self.test_end}"
        )


def make_folds(
    dates: list[dt.date] | pl.Series,
    spec: C.SplitSpec | None = None,
    horizon: int = 1,
) -> list[Fold]:
    """Build expanding-window folds over the observed date range.

    The gap between train and test is ``horizon + spec.gap`` periods. That is the piece
    people forget: a horizon-14 model trained on origins up to ``T`` has already seen the
    actual value at ``T + 14``, so if the test set starts at ``T + 1`` the first 14 test
    origins overlap targets the model was fit on. Holding out ``horizon`` periods removes
    that overlap.
    """
    spec = spec or C.DEFAULT_SPLITS
    ds = sorted(set(dates.to_list() if isinstance(dates, pl.Series) else dates))
    n = len(ds)
    total_gap = horizon + spec.gap

    folds: list[Fold] = []
    for k in range(spec.n_folds):
        # Fold 0 is the most recent; higher k steps further back in time.
        test_end_i = n - 1 - k * spec.test_size
        test_start_i = test_end_i - spec.test_size + 1
        train_end_i = test_start_i - total_gap - 1
        min_train = spec.max_train_size or 365
        if test_start_i < 0 or train_end_i < min_train:
            break  # ran out of history going backwards
        train_start_i = (
            max(0, train_end_i - spec.max_train_size + 1) if spec.max_train_size else 0
        )
        folds.append(
            Fold(
                index=spec.n_folds - 1 - k,
                train_start=ds[train_start_i],
                train_end=ds[train_end_i],
                test_start=ds[test_start_i],
                test_end=ds[test_end_i],
            )
        )
    return sorted(folds, key=lambda f: f.test_start)


def holdout(
    dates: list[dt.date] | pl.Series, test_size: int = 28, horizon: int = 1, gap: int = 0
) -> Fold:
    """A single most-recent holdout -- the quick check while iterating."""
    ds = sorted(set(dates.to_list() if isinstance(dates, pl.Series) else dates))
    test_end_i = len(ds) - 1
    test_start_i = test_end_i - test_size + 1
    train_end_i = test_start_i - horizon - gap - 1
    return Fold(0, ds[0], ds[train_end_i], ds[test_start_i], ds[test_end_i])


def apply_fold(lf: pl.LazyFrame, fold: Fold, part: str) -> pl.LazyFrame:
    """Slice a feature table to one side of a fold. ``part`` is "train" or "test"."""
    d = pl.col(C.DATE)
    if part == "train":
        return lf.filter((d >= fold.train_start) & (d <= fold.train_end))
    if part == "test":
        return lf.filter((d >= fold.test_start) & (d <= fold.test_end))
    raise ValueError(f"part must be 'train' or 'test', got {part!r}")


def assert_no_leakage(train: pl.DataFrame, test: pl.DataFrame, horizon: int) -> None:
    """Fail loudly if a fold's train origins could see the test period's targets."""
    tr_end = train.get_column(C.DATE).max()
    te_start = test.get_column(C.DATE).min()
    reach = tr_end + dt.timedelta(days=horizon)
    if reach >= te_start:
        raise AssertionError(
            f"leakage: training origins reach {reach} (train_end {tr_end} + h{horizon}) "
            f"but test starts {te_start}. Increase SplitSpec.gap."
        )
