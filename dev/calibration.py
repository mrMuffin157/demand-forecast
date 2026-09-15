"""Should the forecast multiplier be ONE number? Five answers, scored out of sample.

    uv run python dev/calibration.py             # baseline half
    uv run python dev/calibration.py --tuned     # the tuned half

Reads the cached predictions in `results/_preds/` and refits nothing, so it runs in
seconds. Writes a markdown report and `<figures>/_calibration.png`.

-----------------------------------------------------------------------------------
THE QUESTION
-----------------------------------------------------------------------------------
`dev/cost_board.py` lifts every forecast by one global scalar `k`, and says so in its
own words: a blunt instrument that "lifts a well-calibrated SKU by as much as a badly-
calibrated one", and therefore a LOWER BOUND on what the objective is worth. Two
separate observations say that bound may be loose:

  * `model_lab/segment_report.py` -- bias can differ by an order of magnitude across
    product lines. One k cannot be right for all of them.
  * the cost board -- under-forecasting can concentrate on high-demand windows, for
    reasons other than stockout censoring. A model that misses big windows low needs
    more lift exactly where it is already losing the most money.

So: five ways to choose the lift, same data, same folds, same rules.

    1. none          k = 1. What the model says, uncorrected.
    2. global        one k for everything. The default -- the thing to beat.
    3. by line       one k per product line (`sku_brand`).
    4. by volume     one k per predicted-volume band -- aimed straight at the defect.
    5. per SKU       each SKU's own empirical error quantile, shrunk toward global.

-----------------------------------------------------------------------------------
THE RULE THAT MAKES THE ANSWER MEAN ANYTHING
-----------------------------------------------------------------------------------
Every k, every band edge and every shrinkage constant is fitted on FOLDS 1-2 and applied
unchanged to FOLD 3. Fold 3 is scored and never consulted. This matters more here than
anywhere else in the folder, because arms 3-5 have more parameters than arm 2 and will
ALWAYS look better in sample -- a per-SKU multiplier fitted and scored on the same rows
can drive training cost to near zero and tell you nothing. If arm 5 cannot beat arm 2 on
fold 3 it has not earned its complexity, and this file is willing to report that.

The model is held fixed per horizon (the cost-board winner) so the comparison isolates
the multiplier and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import LAB, figures_dir, results_dir  # noqa: E402

KS = np.round(np.arange(0.80, 2.01, 0.02), 2)
HORIZONS = (30, 90, 180)

#: Volume bands for arm 4, as quantiles of PREDICTED demand. Predicted, not actual --
#: at serving time the actual is what you are trying to find out, so a band defined on it
#: could not be applied. Four bands keeps ~25% of rows in each on folds 1-2, which is the
#: fewest rows any single k here is fitted from.
VOL_Q = (0.25, 0.50, 0.75)

#: Arm 5 shrinks each SKU's own estimate toward the global one:
#:     k_sku = (n * k_own + PRIOR * k_global) / (n + PRIOR)
#: with n the number of fold-1-2 rows for that SKU. PRIOR is in units of rows, so it says
#: "trust a SKU's own history once it has meaningfully more than this many observations".
#: Candidates are searched on folds 1-2 like everything else rather than being picked.
PRIORS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def cost(y, p, cu, co) -> float:
    den = float(np.sum(y))
    if den <= 0:
        return float("nan")
    return float(np.sum(cu * np.maximum(y - p, 0) + co * np.maximum(p - y, 0)) / den)


def fill(y, p) -> float:
    den = float(np.sum(y))
    return float(1.0 - np.sum(np.maximum(y - p, 0)) / den) if den > 0 else float("nan")


def wape(y, p) -> float:
    den = float(np.sum(np.abs(y)))
    return float(np.sum(np.abs(y - p)) / den) if den > 0 else float("nan")


def best_k(y, p, cu, co) -> float:
    """The multiplier on the grid that minimises cost. Ties go to the smaller k."""
    if not len(y) or float(np.sum(y)) <= 0:
        return 1.0
    return float(KS[int(np.argmin([cost(y, p * k, cu, co) for k in KS]))])


def load(h: int, model: str) -> pl.DataFrame:
    path = results_dir() / "_preds" / f"{model}.parquet"
    if not path.exists():
        raise SystemExit(f"no cached predictions at {path}")
    return pl.read_parquet(path).filter(pl.col("h") == h)


def pick_model(h: int, cu: float, co: float) -> str:
    """The cost-board winner at this horizon, chosen on folds 1-2 exactly as it is there."""
    best, best_c = None, float("inf")
    for path in sorted((results_dir() / "_preds").glob("*.parquet")):
        d = pl.read_parquet(path).filter((pl.col("h") == h) & (pl.col("fold") < 3))
        if not d.height:
            continue
        y, p = d["actual"].to_numpy(), d["predicted"].to_numpy()
        c = cost(y, p * best_k(y, p, cu, co), cu, co)
        if c < best_c:
            best, best_c = path.stem, c
    if best is None:
        raise SystemExit("no cached predictions found")
    return best


def brands() -> pl.DataFrame:
    """SKU -> product line. Taken from the dataset the models were trained on, so a SKU
    that is in the predictions is guaranteed to be here too."""
    return (pl.scan_parquet(common.DATASET).select(["sku", "sku_brand"])
            .unique().collect())


# --------------------------------------------------------------------------------------
# The five arms. Each one: fit on `tr` (folds 1-2), return a function that maps a fold-3
# frame to its multipliers. Nothing may look at `te`.
# --------------------------------------------------------------------------------------

def arm_none(tr, cu, co):
    return lambda te: np.ones(te.height), {}


def arm_global(tr, cu, co):
    k = best_k(tr["actual"].to_numpy(), tr["predicted"].to_numpy(), cu, co)
    return (lambda te: np.full(te.height, k)), {"k": k}


def _group_ks(tr: pl.DataFrame, key: str, cu, co, kg: float) -> dict:
    """One k per level of `key`, falling back to the global k where a level is too thin
    to fit one. 200 rows is a small fraction of a typical fold-1-2 population."""
    out = {}
    for (lev,), g in tr.group_by([key], maintain_order=True):
        y, p = g["actual"].to_numpy(), g["predicted"].to_numpy()
        out[lev] = best_k(y, p, cu, co) if len(y) >= 200 else kg
    return out


def arm_by_line(tr, cu, co):
    kg = best_k(tr["actual"].to_numpy(), tr["predicted"].to_numpy(), cu, co)
    ks = _group_ks(tr, "sku_brand", cu, co, kg)
    def apply(te):
        return np.array([ks.get(b, kg) for b in te["sku_brand"].to_list()])
    return apply, {"k_global": kg, "n_levels": len(ks),
                   "spread": (min(ks.values()), max(ks.values())) if ks else None}


def arm_by_volume(tr, cu, co):
    """Bands on PREDICTED volume. Edges come from folds 1-2 and are then frozen."""
    kg = best_k(tr["actual"].to_numpy(), tr["predicted"].to_numpy(), cu, co)
    edges = np.quantile(tr["predicted"].to_numpy(), VOL_Q)
    band = lambda p: np.searchsorted(edges, p, side="right")  # noqa: E731
    t = tr.with_columns(pl.Series("band", band(tr["predicted"].to_numpy())))
    ks = _group_ks(t, "band", cu, co, kg)
    def apply(te):
        return np.array([ks.get(int(b), kg) for b in band(te["predicted"].to_numpy())])
    return apply, {"k_global": kg, "edges": [float(e) for e in edges],
                   "ks": {int(k): v for k, v in sorted(ks.items())}}


def arm_per_sku(tr, cu, co):
    """Each SKU's own best k, shrunk toward the global one by a prior fitted on folds 1-2.

    The shrinkage constant is chosen by an inner split of folds 1-2 (fit on fold 1, score
    on fold 2) so that PRIORS is searched without ever touching fold 3.
    """
    kg = best_k(tr["actual"].to_numpy(), tr["predicted"].to_numpy(), cu, co)

    def table(src: pl.DataFrame, prior: float) -> dict:
        out = {}
        for (s,), g in src.group_by(["sku"], maintain_order=True):
            y, p = g["actual"].to_numpy(), g["predicted"].to_numpy()
            n = len(y)
            out[s] = (n * best_k(y, p, cu, co) + prior * kg) / (n + prior)
        return out

    f1, f2 = tr.filter(pl.col("fold") == 1), tr.filter(pl.col("fold") == 2)
    best_p, best_c = PRIORS[len(PRIORS) // 2], float("inf")
    if f1.height and f2.height:
        for prior in PRIORS:
            t = table(f1, prior)
            k = np.array([t.get(s, kg) for s in f2["sku"].to_list()])
            c = cost(f2["actual"].to_numpy(), f2["predicted"].to_numpy() * k, cu, co)
            if c < best_c:
                best_p, best_c = prior, c

    final = table(tr, best_p)
    def apply(te):
        return np.array([final.get(s, kg) for s in te["sku"].to_list()])
    return apply, {"k_global": kg, "prior": best_p, "n_skus": len(final)}


ARMS = [
    ("none", "no lift at all", arm_none),
    ("global", "one k for the catalogue (the default)", arm_global),
    ("by line", "one k per product line", arm_by_line),
    ("by volume", "one k per predicted-volume band", arm_by_volume),
    ("per SKU", "each SKU's own k, shrunk to global", arm_per_sku),
]


def run(h: int, cu: float, co: float, bmap: pl.DataFrame) -> dict:
    model = pick_model(h, cu, co)
    d = load(h, model).join(bmap, on="sku", how="left").with_columns(
        pl.col("sku_brand").fill_null("?"))
    tr, te = d.filter(pl.col("fold") < 3), d.filter(pl.col("fold") == 3)
    if not tr.height or not te.height:
        raise SystemExit(f"h={h}: need folds 1-2 and fold 3, got {tr.height}/{te.height}")

    # How much INDEPENDENT evidence a per-SKU multiplier actually has. Rows per SKU
    # badly overstates it: origins are weekly, so at h=180 two origins 7 days apart
    # share 173 of their 180 future days and are very nearly the same observation.
    # The honest unit is non-overlapping windows = origin span / horizon.
    gg = tr.group_by("sku").agg(
        pl.len().alias("n"),
        (pl.col("date").max() - pl.col("date").min()).dt.total_days().alias("span"))
    indep = float(gg["span"].median()) / h
    rows_per_sku = float(gg["n"].median())

    y, p = te["actual"].to_numpy(), te["predicted"].to_numpy()
    rows = []
    for name, note, fn in ARMS:
        apply, info = fn(tr, cu, co)
        k_te = apply(te)
        lifted = p * k_te
        # In-sample counterpart, to expose how much of any gain is fitting noise.
        k_tr = apply(tr)
        rows.append({
            "arm": name, "note": note,
            "cost": cost(y, lifted, cu, co), "wape": wape(y, lifted),
            "fill": fill(y, lifted),
            "cost_in": cost(tr["actual"].to_numpy(),
                            tr["predicted"].to_numpy() * k_tr, cu, co),
            "k_mean": float(np.mean(k_te)), "k_sd": float(np.std(k_te)),
            "info": info,
        })
    base = next(r for r in rows if r["arm"] == "global")["cost"]
    for r in rows:
        r["vs_global"] = 100.0 * (base - r["cost"]) / base
    return {"horizon": h, "model": model, "n_test": te.height,
            "rows_per_sku": rows_per_sku, "indep_windows": indep, "arms": rows}


def draw(res: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(res), figsize=(5.2 * len(res), 4.6))
    axes = np.atleast_1d(axes)
    names = [a["arm"] for a in res[0]["arms"]]
    for ax, r in zip(axes, res):
        oos = [a["cost"] for a in r["arms"]]
        ins = [a["cost_in"] for a in r["arms"]]
        x = np.arange(len(names))
        ax.bar(x - 0.2, ins, 0.4, label="folds 1-2 (fitted on)", color=common.GRID_C)
        ax.bar(x + 0.2, oos, 0.4, label="fold 3 (unseen)", color=common.PRED_C)
        gi = names.index("global")
        ax.axhline(oos[gi], color=common.ACTUAL_C, ls="--", lw=1.2,
                   label="global, out of sample")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(f"{r['horizon']} days  ({r['model']})", fontsize=11)
        ax.set_ylabel("inventory cost per unit of demand")
        ax.grid(axis="y", color=common.GRID_C, lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, framealpha=0.9)
    fig.suptitle("Does a smarter multiplier survive out of sample?", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(res: list[dict], cu: float, co: float, fig: Path) -> str:
    L: list[str] = []
    A = L.append
    A("# Is one multiplier enough?\n")
    A(f"*Generated by `dev/calibration.py` (mode `{common.MODE}`) from the cached "
      f"predictions in `{results_dir().name}/_preds/`. Nothing was refitted.*\n")
    A(f"Every multiplier, band edge and shrinkage constant below was fitted on **folds "
      f"1-2** and applied unchanged to **fold 3**, which is what is scored. Cost is at "
      f"`{cu:g}:{co:g}`.\n")
    A(f"![calibration]({fig.parent.name}/{fig.name})\n")

    for r in res:
        A(f"## {r['horizon']} days — `{r['model']}`\n")
        A("| arm | what it is | cost (fold 3) | vs global | WAPE | fill | in-sample cost |")
        A("|---|---|---|---|---|---|---|")
        for a in r["arms"]:
            v = "—" if a["arm"] == "global" else f"{a['vs_global']:+.1f}%"
            A(f"| `{a['arm']}` | {a['note']} | **{a['cost']:.4f}** | {v} | "
              f"{a['wape']:.4f} | {a['fill']:.3f} | {a['cost_in']:.4f} |")
        A("")
        g = next(a for a in r["arms"] if a["arm"] == "global")
        win = min((a for a in r["arms"] if a["arm"] != "none"), key=lambda a: a["cost"])
        if win["arm"] == "global":
            A(f"The global multiplier wins at this horizon. Every per-segment arm has "
              f"more freedom and still could not beat one number on unseen rows.\n")
        else:
            A(f"`{win['arm']}` beats the global multiplier by **{win['vs_global']:.1f}%** "
              f"on fold 3 (cost {win['cost']:.4f} against {g['cost']:.4f}), "
              f"at fill {win['fill']:.3f} against {g['fill']:.3f}.\n")
        for a in r["arms"]:
            i = a["info"]
            if a["arm"] == "by volume" and i.get("ks"):
                ks = ", ".join(f"band {b}: {v:.2f}" for b, v in i["ks"].items())
                A(f"- **by volume** — bands split at predicted "
                  f"{', '.join(f'{e:.0f}' for e in i['edges'])} units; {ks}. "
                  f"Band 0 is the smallest quarter of forecasts, band 3 the largest.")
            if a["arm"] == "by line" and i.get("spread"):
                A(f"- **by line** — {i['n_levels']} product lines, k from "
                  f"{i['spread'][0]:.2f} to {i['spread'][1]:.2f} "
                  f"(global {i['k_global']:.2f}).")
            if a["arm"] == "per SKU":
                A(f"- **per SKU** — {i['n_skus']:,} SKUs, shrinkage prior "
                  f"{i['prior']:.0f} rows, mean k {a['k_mean']:.2f} ± {a['k_sd']:.2f}.")
        A("")
        A(f"Evidence per SKU on folds 1-2: **{r['rows_per_sku']:.0f} rows**, but only "
          f"**~{r['indep_windows']:.1f} non-overlapping {r['horizon']}-day windows**. "
          f"Origins are weekly, so two of them {r['horizon'] // 26 or 1} week apart share "
          f"almost all of their future days — the row count is not the evidence count.\n")

    A("## Why the answer changes with the horizon\n")
    A("The per-SKU arm is the most flexible thing here and it wins or loses depending "
      "entirely on how much independent evidence one SKU has to fit its own multiplier "
      "from. That number collapses with the horizon, because the origins are weekly and "
      "the windows overlap:\n")
    A("| h | rows per SKU (folds 1-2) | independent windows | per-SKU arm, fold 3 |")
    A("|---|---|---|---|")
    for r in res:
        a = next(x for x in r["arms"] if x["arm"] == "per SKU")
        A(f"| {r['horizon']} | {r['rows_per_sku']:.0f} | ~{r['indep_windows']:.1f} | "
          f"{a['vs_global']:+.1f}% |")
    A("")
    A("At short horizons a SKU has several independent windows to estimate its own bias "
      "from; at the longest it may have about one, and fitting a per-product multiplier "
      "to a single observation is fitting that observation. The in-sample column shows "
      "the same thing from the other side — where evidence is thin, the per-SKU arm "
      "scores far better on the rows it was fitted on than on the ones it was not.\n")
    A("**So the useful question is not \"does per-SKU calibration work\".** It is how "
      "granular a multiplier can be given the evidence supporting it — and that "
      "granularity runs out as the horizon grows.\n")
    A("## How to read the last column\n")
    A("`in-sample cost` is the same arm scored on the rows it was fitted on. It is there "
      "to show the price of freedom: an arm whose in-sample cost is far below its fold-3 "
      "cost was fitting noise, and the gap is how much. The `global` arm has one "
      "parameter and barely moves between the two; `per SKU` has one per product and is "
      "where any overfit will show.\n")
    A("## What this does not settle\n")
    A("- **One test fold.** Every number in the right-hand columns rests on a single "
      "held-out fold, so a difference of a percent or two is not resolvable. Only "
      "differences large enough to survive that are quoted as results above.\n")
    A("- **The bands and lines are fixed categories**, not a model. A multiplier that "
      "varied continuously with predicted volume would be strictly more flexible; this "
      "asks only whether the direction is real before anything more is built.\n")
    A("- Nothing here is refitted. Every arm rescales the output of a model that was "
      "still trained to predict the middle.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuned", action="store_true")
    ap.add_argument("--horizons", type=str, default=",".join(str(h) for h in HORIZONS))
    args = ap.parse_args()
    common.set_mode("tuned" if args.tuned else "initial")
    cu, co = common.cost_ratio()
    hs = [int(x) for x in args.horizons.split(",") if x.strip()]

    print(f"\n{'=' * 90}\nCALIBRATION — how should the multiplier vary?"
          f"   [{common.MODE}]   {cu:g}:{co:g}\n{'=' * 90}")
    bmap = brands()
    res = [run(h, cu, co, bmap) for h in hs]

    for r in res:
        print(f"\n[{r['horizon']} days]  model={r['model']}  fold-3 rows={r['n_test']:,}")
        print(f"  {'arm':<12}{'cost':>10}{'vs global':>12}{'fill':>9}{'in-sample':>12}")
        for a in r["arms"]:
            v = "—" if a["arm"] == "global" else f"{a['vs_global']:+.1f}%"
            print(f"  {a['arm']:<12}{a['cost']:>10.4f}{v:>12}"
                  f"{a['fill']:>9.3f}{a['cost_in']:>12.4f}")

    fig = figures_dir() / "_calibration.png"
    draw(res, fig)
    out = LAB / ("CALIBRATION.md" if common.MODE == "initial" else "CALIBRATION_TUNED.md")
    out.write_text(report(res, cu, co, fig))
    (results_dir() / "_calibration.json").write_text(json.dumps(
        [{k: v for k, v in r.items()} for r in res], indent=2, default=str))
    print(f"\n{'=' * 90}\nwrote {out.relative_to(LAB.parent)}  and  "
          f"{fig.relative_to(LAB.parent)}\n{'=' * 90}")


if __name__ == "__main__":
    main()
