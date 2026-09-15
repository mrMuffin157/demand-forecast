"""Forecast accuracy metrics chosen for intermittent, multi-scale demand.

Why not MAPE: a large share of SKU-days are zero, so MAPE is undefined or infinite on
much of the data. The workhorse here is **WAPE** (a.k.a. weighted MAPE) -- total absolute error
divided by total actual units. It is defined at zero, it aggregates honestly across SKUs
of wildly different volume, and it is the metric an inventory planner actually feels.
"""

from __future__ import annotations

import numpy as np

from . import assumptions as _A


def _prep(y: np.ndarray, yhat: np.ndarray, w: np.ndarray | None = None):
    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(yhat) & (w > 0)
    return y[ok], yhat[ok], w[ok]


def mae(y, yhat, w=None) -> float:
    y, yhat, w = _prep(y, yhat, w)
    return float(np.average(np.abs(y - yhat), weights=w)) if y.size else float("nan")


def rmse(y, yhat, w=None) -> float:
    y, yhat, w = _prep(y, yhat, w)
    return float(np.sqrt(np.average((y - yhat) ** 2, weights=w))) if y.size else float("nan")


def wape(y, yhat, w=None) -> float:
    """Sum|e| / Sum|y|. The primary headline metric."""
    y, yhat, w = _prep(y, yhat, w)
    den = np.sum(w * np.abs(y))
    return float(np.sum(w * np.abs(y - yhat)) / den) if den > 0 else float("nan")


def bias(y, yhat, w=None) -> float:
    """Signed relative error. Positive = over-forecasting. Over- and under-forecasting
    cost very different amounts in inventory, so never look at WAPE without this."""
    y, yhat, w = _prep(y, yhat, w)
    den = np.sum(w * np.abs(y))
    return float(np.sum(w * (yhat - y)) / den) if den > 0 else float("nan")


def smape(y, yhat, w=None) -> float:
    y, yhat, w = _prep(y, yhat, w)
    den = np.abs(y) + np.abs(yhat)
    m = den > 0
    return float(np.average(2.0 * np.abs(y - yhat)[m] / den[m], weights=w[m])) if m.any() else 0.0


def rmsse(y, yhat, scale: float) -> float:
    """Root mean squared scaled error (the M5 metric), given a per-series scale."""
    y, yhat, _ = _prep(y, yhat)
    if scale <= 0 or not y.size:
        return float("nan")
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / scale)


def mase(y, yhat, scale: float) -> float:
    """Mean absolute scaled error, scaled by an in-sample naive MAE."""
    y, yhat, _ = _prep(y, yhat)
    if scale <= 0 or not y.size:
        return float("nan")
    return float(np.mean(np.abs(y - yhat)) / scale)


def pinball(y, yhat, q: float, w=None) -> float:
    """Quantile loss -- the metric for the safety-stock question, not the mean question."""
    y, yhat, w = _prep(y, yhat, w)
    d = y - yhat
    return float(np.average(np.maximum(q * d, (q - 1.0) * d), weights=w)) if y.size else float("nan")


def coverage(y, lo, hi, w=None) -> float:
    """Empirical coverage of a prediction interval. Should match its nominal level."""
    y, lo, w = _prep(y, lo, w)
    hi = np.asarray(hi, dtype=np.float64).ravel()[: y.size]
    return float(np.average(((y >= lo) & (y <= hi)).astype(float), weights=w)) if y.size else float("nan")


def service_level_shortfall(y, yhat, w=None) -> float:
    """Units of unmet demand as a share of total demand -- the stockout cost of a
    forecast used as-is for replenishment."""
    y, yhat, w = _prep(y, yhat, w)
    den = np.sum(w * y)
    return float(np.sum(w * np.maximum(y - yhat, 0.0)) / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------------------
# Inventory cost. The reporting metrics above are symmetric; a purchasing decision is not.
# --------------------------------------------------------------------------------------

def newsvendor_cost(y, yhat, cu: float = _A.CU, co: float = _A.CO, w=None) -> float:
    """Expected inventory cost per unit of demand, if you order exactly the forecast.

        cost = cu * units short  +  co * units of dead stock,   divided by total demand.

    `cu` is what one unit of unmet demand costs (lost margin, a stocked-out listing
    losing rank, a customer who buys elsewhere); `co` is what one unit of leftover
    stock costs over the horizon (capital, storage, markdown risk). Only their RATIO
    matters -- the result scales linearly in (cu, co) -- so fix co = 1 and express the
    preference as cu (the ratio lives in assumptions.toml).

    Two properties worth knowing:

    * At cu == co this is exactly WAPE. The whole difference between "accuracy" and
      "inventory cost" is the gap between cu and co, which is why a symmetric metric
      cannot express a stockout-averse business.
    * It is minimised by the cu/(cu+co) quantile of demand, not by the mean. Whenever
      cu > co that quantile sits above the median -- so a model trained to predict the
      average is solving the wrong problem before any hyper-parameter is touched.
    """
    y, yhat, w = _prep(y, yhat, w)
    den = np.sum(w * y)
    if den <= 0:
        return float("nan")
    short = np.maximum(y - yhat, 0.0)
    over = np.maximum(yhat - y, 0.0)
    return float(np.sum(w * (cu * short + co * over)) / den)


def fill_rate(y, yhat, w=None) -> float:
    """Share of demand you would actually have met. 1 - shortfall, stated the way a
    purchasing team states it ("we were 95% in stock")."""
    return 1.0 - service_level_shortfall(y, yhat, w)


def cost_from_wape_bias(wape: float, bias: float,
                        cu: float = _A.CU, co: float = _A.CO) -> float:
    """`newsvendor_cost` recovered EXACTLY from a WAPE and a bias, with no predictions.

    Because |y - p| = short + over and sum(p - y) = over - short (both divided by the
    same sum(y)), the two reported numbers determine the split:

        short = (wape - bias) / 2          over = (wape + bias) / 2
        cost  = cu * short + co * over  =  ((cu + co) * wape + (co - cu) * bias) / 2

    Exact, not an approximation, and linear in both arguments -- so it also holds for
    fold-averaged wape and bias, not just for a single scoring run.

    Why it earns its place: a hyper-parameter search stores wape and bias per trial and
    throws the predictions away. This lets an old search be re-scored under a new cost
    ratio for free, instead of re-running it for hours.
    """
    return ((cu + co) * float(wape) + (co - cu) * float(bias)) / 2.0


def critical_ratio(cu: float = _A.CU, co: float = _A.CO) -> float:
    """The demand quantile that minimises newsvendor_cost -- the number to hand a
    quantile model as its target: cu / (cu + co)."""
    return float(cu) / (float(cu) + float(co))


def evaluate(y, yhat, w=None, scale: float | None = None,
             cu: float = _A.CU, co: float = _A.CO) -> dict[str, float]:
    """The standard metric bundle, returned as a flat dict for logging."""
    out = {
        "wape": wape(y, yhat, w),
        "mae": mae(y, yhat, w),
        "rmse": rmse(y, yhat, w),
        "bias": bias(y, yhat, w),
        "smape": smape(y, yhat, w),
        "shortfall": service_level_shortfall(y, yhat, w),
        "cost": newsvendor_cost(y, yhat, cu, co, w),
        "fill": fill_rate(y, yhat, w),
        "n": int(_prep(y, yhat, w)[0].size),
    }
    if scale:
        out["rmsse"] = rmsse(y, yhat, scale)
        out["mase"] = mase(y, yhat, scale)
    return out


def evaluate_by(
    y, yhat, groups, w=None, min_n: int = 30
) -> dict[str, dict[str, float]]:
    """Per-group metrics (e.g. per SKU, per demand class) so you can see *where* a model
    wins rather than just whether it wins on average."""
    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    groups = np.asarray(groups)
    w = None if w is None else np.asarray(w, dtype=np.float64).ravel()
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_n:
            continue
        out[str(g)] = evaluate(y[m], yhat[m], None if w is None else w[m])
    return out
