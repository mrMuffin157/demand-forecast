"""Should each horizon's model be picked on PRECISION instead of inventory cost?

    DEMANDML_MODELS=moving_average,random_forest,xgboost_gbdt,blend uv run python dev/pick_rule.py
    uv run python dev/pick_rule.py --preds <other store's results/_preds> --label CA

`forecast.choose` can pick the model with the lowest inventory cost (the CU:CO ratio in
assumptions.toml, each model on its own best multiplier) or the most precise one -- the
share of SKUs whose TYPICAL forecast lands within ±40% of what sold. The two rules can
disagree: cost rewards aiming high, precision rewards being close.

Both rules are run the way `forecast.choose` runs: picked on every fold but the newest
(`HOLDOUT_FOLDS`), then judged on the held-out fold, plus folds won across all folds. The
multiplier is unaffected either way -- it is still fitted on cost for whichever model is
picked, so ordering keeps its asymmetric target.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

import common
import forecast as F


def w40(y, p):
    m = y > 0
    return float((np.abs(p[m] - y[m]) / y[m] <= 0.40).mean()) if m.any() else float("nan")


def best_k_cost(y, p, cu, co):
    cs = [F.cost(y, p * k, cu, co) for k in F.KS]
    i = int(np.argmin(cs))
    return float(F.KS[i]), float(cs[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, default=None)
    ap.add_argument("--label", type=str, default="US")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    common.set_mode("initial")
    cu, co = common.CU, common.CO
    d = Path(a.preds) if a.preds else common.results_dir() / "_preds"
    preds = {p.stem: pl.read_parquet(p) for p in sorted(d.glob("*.parquet")) if common.model_allowed(p.stem)}
    L = [f"### {a.label}", "",
         "| horizon | cost pick | precision pick | ±40% on held-out fold: cost pick → precision pick | "
         "folds where precision pick is more precise | cost on held-out fold (own k) |",
         "|---|---|---|---|---|---|"]
    for h in (30, 90, 180):
        P = {m: v.filter(pl.col("h") == h) for m, v in preds.items()}
        P = {m: v for m, v in P.items() if v.height}
        if len(P) < 2:
            continue
        folds = sorted(set().union(*[set(v["fold"].unique().to_list()) for v in P.values()]))
        sel, hold = folds[:-F.HOLDOUT_FOLDS], folds[-F.HOLDOUT_FOLDS:]
        stats = {}
        for m, v in P.items():
            tr, te = v.filter(pl.col("fold").is_in(sel)), v.filter(pl.col("fold").is_in(hold))
            y, p = tr["actual"].to_numpy(), tr["predicted"].to_numpy()
            k, c = best_k_cost(y, p, cu, co)
            bias = abs(float((p.sum() - y.sum()) / y.sum()))
            yt, pt = te["actual"].to_numpy(), te["predicted"].to_numpy()
            per = {f: w40(v.filter(pl.col("fold") == f)["actual"].to_numpy(),
                          v.filter(pl.col("fold") == f)["predicted"].to_numpy()) for f in folds}
            stats[m] = dict(k=k, cost_sel=c, w40_sel=w40(y, p), bias=bias, w40_hold=w40(yt, pt),
                            cost_hold=F.cost(yt, pt * k, cu, co), per=per)
        cp = min(stats, key=lambda m: stats[m]["cost_sel"])
        pp = max(stats, key=lambda m: (round(stats[m]["w40_sel"], 4), -stats[m]["bias"]))
        won = sum(stats[pp]["per"][f] > stats[cp]["per"][f] for f in folds) if pp != cp else None
        L.append(f"| {h} d | `{cp}` | `{pp}`{' (same)' if pp == cp else ''} | "
                 f"{stats[cp]['w40_hold']:.2%} → {stats[pp]['w40_hold']:.2%} | "
                 f"{'—' if won is None else f'{won}/{len(folds)}'} | "
                 f"{stats[cp]['cost_hold']:.4f} → {stats[pp]['cost_hold']:.4f} |")
    L.append("")
    text = "\n".join(L)
    print(text)
    if a.out:
        with open(a.out, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
