"""Train on log1p of the stock-out-lifted target, predict back on the scale of SALES.

Shared by `xgboost_gbdt.py` and `random_forest.py`, which wrap their estimator in
`LiftedLog` by default. Three changes from fitting raw `y_cum{h}`, each judged on rolling
folds scored against what actually sold (see docs/FINDINGS.md):

  LOG TARGET   squared error on log1p(units), exp'd back. Each SKU's miss is judged in
               proportion to its size instead of in units, so the big sellers stop
               dominating the fit. More SKUs land within +/-40%, most clearly at the
               longest horizon and for small sellers.
  LIFT         a training window that was partly out of stock (up to the share that
               `prepare_data.MAX_OOS_FRACTION` keeps) sold less than its demand, and the
               model learned that dip as if it were demand. `y_cum{h}_lift` divides by the
               window's in-stock share. A small, consistent gain across folds and size
               bands, much of it on SKUs that stayed IN stock.
  CALIBRATION  one factor c = sum(sales) / sum(prediction) over a sample of the training
               rows, against SALES (`y`, never the lift). A log fit aims below the mean
               (plain log under-forecasts totals by a few percent); c puts the level back.
               Calibrating to the lifted (demand) level instead fixes more of the bias but
               costs some precision -- a log model's misses run longer on the high side,
               so a level aimed at sales keeps more SKUs inside the band. Totals still run
               slightly low; that is the price of precision and it is stated, not hidden.

The lift is read from `frame`, so the LabModel must set `needs_frame=True`. Called without
a frame (or on rows whose store's dataset predates the column -- pooled training appends
other stores' rows), a row falls back to its plain `y_cum{h}`: the old behaviour, not an
error.
"""

from __future__ import annotations

import numpy as np
import polars as pl

#: Training rows the calibration factor is fitted on -- a sample is plenty for one ratio.
CAL_ROWS = 100_000


def lift_column(h: int) -> str:
    return f"y_cum{h}_lift"


def lift_stats(est) -> dict:
    """`{"lift_coverage", "lifted_share"}` for a fitted `LiftedLog`, `{}` for anything
    else -- so a results file says whether the lift was trained on, not just the code."""
    if not hasattr(est, "lift_coverage_"):
        return {}
    return {"lift_coverage": round(est.lift_coverage_, 4),
            "lifted_share": round(est.lifted_share_, 4)}


def lift_note(stats: dict) -> str:
    """One log fragment: the shares, or a warning when a LiftedLog saw no lift at all."""
    if not stats:
        return ""
    if stats["lift_coverage"] == 0:
        return "  ! NO LIFT in the training frame -- trained on unlifted sales"
    return f"  lift on {stats['lift_coverage']:.0%} of rows, {stats['lifted_share']:.1%} lifted"


class LiftedLog:
    def __init__(self, inner, horizon: int, seed: int = 0):
        self.inner = inner
        self.horizon = horizon
        self.seed = seed
        self.c_ = 1.0
        # What the fit actually saw, recorded in every results JSON / forecast JSON
        # (`lift_stats`): coverage = share of training rows with a lift, lifted = share
        # raised above sales. Coverage 0 is the silent fallback to unlifted sales.
        self.lift_coverage_ = 0.0
        self.lifted_share_ = 0.0

    def fit(self, X: pl.DataFrame, y: np.ndarray, w: np.ndarray | None = None,
            frame: pl.DataFrame | None = None):
        sales = np.clip(np.nan_to_num(np.asarray(y, dtype=np.float64)), 0, None)
        target = sales
        col = lift_column(self.horizon)
        if frame is not None and col in frame.columns:
            lift = frame.get_column(col).cast(pl.Float64).to_numpy()
            ok = np.isfinite(lift)
            target = np.where(ok, np.clip(lift, 0, None), sales)
            self.lift_coverage_ = float(ok.mean()) if len(ok) else 0.0
            self.lifted_share_ = float((ok & (lift > sales + 1e-9)).mean())
        self.inner.fit(X, np.log1p(target), w)

        keep = np.flatnonzero(w > 0) if w is not None else np.arange(len(sales))
        rng = np.random.default_rng(self.seed)
        if len(keep) > CAL_ROWS:
            keep = np.sort(rng.choice(keep, CAL_ROWS, replace=False))
        if len(keep):
            p = self._raw(X[keep])
            self.c_ = float(sales[keep].sum() / max(p.sum(), 1e-9))
        return self

    def _raw(self, X: pl.DataFrame) -> np.ndarray:
        p = np.asarray(self.inner.predict(X), dtype=np.float64).ravel()
        return np.clip(np.expm1(np.nan_to_num(p)), 0, None)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self._raw(X) * self.c_

    def importances(self) -> list[tuple[str, float]]:
        return self.inner.importances()
