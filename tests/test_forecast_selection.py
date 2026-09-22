"""`dev/forecast.choose()` picks the model that decides what gets bought.

A selector written as `fold < 3` literally, on a board that carries twelve folds, selects
on the two oldest windows and discards the nine after them -- and a handful of
consecutive folds describe a season, not a stable property of the model (see
`dev/regime.py` and docs/FINDINGS.md).

The failure mode is the reason these tests exist rather than a comment: a selector that
quietly narrows its evidence still returns a plausible model name, still prints a
plausible fill rate, and nothing downstream can tell. So what is pinned here is not the
chosen model -- that is allowed to change when the data does -- but the DISCIPLINE:

  * selection sees every fold except the held-back newest one,
  * the held-back fold changes nothing about the pick,
  * `folds_won` really counts per-fold wins, so a pooled winner that leads in a minority
    of folds is visible as one instead of being reported as a ranking.

Predictions are synthesised rather than read from `dev/results/_preds/`, so this runs in
CI with no build artefacts and cannot drift when the board is re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dev"))

import forecast as F  # noqa: E402

H = 30
CU, CO = 4.0, 1.0   # a local fixture ratio; only cu > co matters here
N_FOLDS = 12


def write_preds(tmp: Path, spec: dict[str, list[float]], n: int = 400,
                seed: int = 11) -> Path:
    """One parquet per model under `<tmp>/_preds/`, shaped like the real cache.

    `spec` maps a model name to one dispersion per fold. The level of a forecast is not
    what costs money here -- `choose()` fits a multiplier per model, which absorbs any
    constant factor exactly -- so a model is made better or worse purely by the SPREAD of
    its ratio to the truth. Smaller is better, and that is the only lever these tests
    pull.
    """
    d = tmp / "_preds"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    actual = rng.integers(5, 200, n).astype(float)
    for model, sigmas in spec.items():
        frames = []
        for fold, sigma in enumerate(sigmas, start=1):
            ratio = np.exp(rng.normal(0.0, sigma, n))
            frames.append(pl.DataFrame({
                "h": np.full(n, H, dtype=np.int64),
                "fold": np.full(n, fold, dtype=np.int64),
                "actual": actual,
                "predicted": actual * ratio,
            }))
        pl.concat(frames).write_parquet(d / f"{model}.parquet")
    return d


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    """Point `choose()` at a scratch prediction cache instead of the real one."""
    def _build(spec: dict[str, list[float]], **kw) -> None:
        write_preds(tmp_path, spec, **kw)
        monkeypatch.setattr(F, "results_dir", lambda: tmp_path)
    return _build


def test_selection_uses_every_fold_but_the_newest(cache):
    """A model that is excellent on folds 1-2 and terrible afterwards must not win.

    This is the regression itself. `early_star` is built to beat everything on exactly
    the two folds the old `fold < 3` selector looked at, and to be the worst model on
    every fold after them. A selector reading two folds picks it; one reading eleven
    cannot.
    """
    cache({
        "early_star": [0.05, 0.05] + [0.90] * 10,
        "steady": [0.35] * N_FOLDS,
    })
    pick = F.choose(H, CU, CO)

    assert pick["model"] == "steady", (
        "selection is being made on the first folds only -- `early_star` wins folds 1-2 "
        "and loses the other nine")
    assert pick["folds_selected"] == N_FOLDS - F.HOLDOUT_FOLDS
    assert pick["folds_holdout"] == [N_FOLDS]


def test_the_holdout_fold_cannot_change_the_pick(cache):
    """Poison the newest fold for the model that should win; it must still win.

    The quoted fill rate is only honest if the fold it is read from took no part in
    choosing. Here the winner's newest fold is made catastrophic -- any selector that
    consulted it would switch models -- so the pick is the whole assertion.
    """
    clean = {"steady": [0.35] * N_FOLDS, "noisy": [0.60] * N_FOLDS}
    cache(clean)
    baseline = F.choose(H, CU, CO)

    poisoned = {"steady": [0.35] * (N_FOLDS - 1) + [4.0], "noisy": [0.60] * N_FOLDS}
    cache(poisoned)
    after = F.choose(H, CU, CO)

    assert baseline["model"] == "steady"
    assert after["model"] == baseline["model"], (
        "the held-back fold leaked into selection -- poisoning it changed the pick")
    assert after["k_holdout"] == pytest.approx(baseline["k_holdout"]), (
        "the selection multiplier moved, so the held-back fold reached the k search too")


def test_folds_won_counts_real_per_fold_wins(cache):
    """`folds_won` is a stability claim, so it has to be a count and not a flourish.

    A strictly dominant model wins every selection fold. Anything less means the number
    is being derived from pooled cost rather than measured fold by fold, which would make
    a coin-toss ranking look settled.
    """
    cache({"dominant": [0.10] * N_FOLDS, "worse": [0.70] * N_FOLDS})
    pick = F.choose(H, CU, CO)

    assert pick["model"] == "dominant"
    assert pick["folds_won"] == pick["folds_selected"] == N_FOLDS - F.HOLDOUT_FOLDS


def test_a_split_field_is_reported_as_split(cache):
    """Two models of equal quality must not produce a confident-looking win.

    Same dispersion, different noise draws: whichever edges ahead on pooled cost should
    take only about half the folds. The guard is loose on purpose -- the point is that
    `folds_won` tracks the evidence, not that it hits a particular number.
    """
    cache({"twin_a": [0.45] * N_FOLDS, "twin_b": [0.45] * N_FOLDS})
    pick = F.choose(H, CU, CO)

    sel = pick["folds_selected"]
    assert 0 < pick["folds_won"] < sel, (
        f"one of two equally good models won {pick['folds_won']}/{sel} folds -- "
        f"`folds_won` is not measuring per-fold outcomes")


def test_a_thin_board_still_runs_but_is_not_silent(cache, capsys):
    """Three folds is a season, not a trend -- it must warn rather than refuse.

    `forecast.MIN_SELECTION_FOLDS` asks for more. A freshly built board with three is still usable, and
    failing on it would make the common case unrunnable; saying nothing would reproduce
    the original bug with extra steps.
    """
    cache({"only": [0.3, 0.3, 0.3]})
    pick = F.choose(H, CU, CO)

    assert pick["folds_selected"] == 3 - F.HOLDOUT_FOLDS
    assert "provisional" in capsys.readouterr().out


def test_one_fold_is_refused(cache):
    """With nothing left after the holdout there is no selection to make."""
    cache({"only": [0.3]})
    with pytest.raises(SystemExit, match="fold"):
        F.choose(H, CU, CO)
