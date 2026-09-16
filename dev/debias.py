"""Is the models' bias a level or a shape, and which output correction fixes it?

    uv run python dev/debias.py                # all arms, all horizons
    uv run python dev/debias.py --model xgboost_gbdt

Writes a markdown report and `<figures>/_debias.png`. Nothing is refitted: this reads the
cached predictions and corrects their OUTPUT, so it runs in seconds.

THE QUESTION
============
A model can be biased overall, and it can be biased differently in different parts of its
range. This file decomposes the bias against the model's own PREDICTED volume (quintiles),
which is the only banding that could be applied at serving time. If the smallest quintile
is over-forecast while the middle is under-forecast, and the slope of actual on predicted
is still ~1, then the bias is NOT ordinary shrinkage toward the mean -- the model is not
compressing its range, it is mis-levelling different parts of it in different directions.

Stock-out censoring is the first thing to rule out before reading any bias here: if
windows with no stock-out days show the same shortfall, censoring is not the cause.

WHY THAT MATTERS
================
A single global multiplier -- what `forecast.py` and `cost_board.py` use by default --
multiplies every SKU by the same `k`. Against a bias that changes sign across the range,
one number is the wrong shape of instrument: it cannot lift the middle without lifting an
already-over-forecast bottom by the same factor. Whatever it achieves, it achieves by
averaging two opposite errors.

A mechanism that COULD produce such a shape is the log link. LightGBM and XGBoost fit
Tweedie with a log link, so a prediction is `exp(sum of leaf values)`; L2 regularisation
shrinks leaf values toward zero, which after exponentiation is a multiplicative pull
toward 1, heaviest where the summed leaf value is largest. It is worth testing directly
(refit across a wide range of `reg_lambda` and watch the bias) rather than assuming: if
the bias barely moves, or moves the wrong way, the mechanism is something else, and every
correction below is an empirical patch on an effect that is not understood. That is a
weaker position than "we know why and we are compensating for it", and it is the honest
one. See docs/FINDINGS.md.

THE ARMS
========
Each arm is a function applied to the model's prediction. All are fitted on folds 1-2 and
scored on fold 3, which is the only honest way to compare instruments of different
flexibility -- a per-SKU correction has one parameter per SKU and will always win in
sample.

    none        use the model as trained
    global      one k for the catalogue                       (1 parameter, the default)
    loglinear   p -> b * p^a, fitted in log space             (2 parameters)
    isotonic    p -> p * g(log p), g monotone                 (non-parametric, ordered)
    per SKU     each SKU its own k, shrunk to global          (one per SKU)

`loglinear` is included because the log-link story predicts it should work: if the
defect were a multiplicative pull toward 1 in log space, a power law in log space would
be its exact inverse. Unless that story is confirmed, treat it as one more empirical
shape rather than a mechanism-derived fix. It is only one parameter more than the default.

WHAT THIS FILE DOES NOT CLAIM
=============================
Correcting the output is not the same as fixing the model. Every arm here rescales a
prediction that was still TRAINED to aim at the middle; none of them adds information.
The deeper fix -- retraining at the cost-minimising quantile -- is a separate experiment
(`--cost` on a model that supports it). So this is a patch, not a repair, and the ceiling
on it is set by how much of the error is calibration rather than accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import LAB, figures_dir, results_dir  # noqa: E402

#: Shrinkage prior for the per-SKU arm, in rows. A SKU with fewer effective observations
#: than this is pulled most of the way back to the global multiplier.
PER_SKU_PRIOR = 40

#: Number of bins for the isotonic arm. More bins is more freedom and more overfit; this
#: is deliberately coarse because the evidence per bin is what limits it.
ISO_BINS = 12


# ======================================================================================
# Metrics
# ======================================================================================

def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    cu, co = common.cost_ratio()
    tot = max(float(np.sum(np.abs(y))), 1e-9)
    short = np.clip(y - p, 0.0, None).sum()
    over = np.clip(p - y, 0.0, None).sum()
    return {"wape": float(np.sum(np.abs(y - p)) / tot),
            "bias": float(np.sum(p - y) / tot),
            "cost": float((cu * short + co * over) / tot),
            "fill": float(1.0 - short / tot)}


def optimal_k(y: np.ndarray, p: np.ndarray, min_n: int = 50) -> float:
    """The cost-minimising scalar: the target quantile of `actual/predicted` weighted BY
    PREDICTED. That weighting falls out of the first-order condition -- it is not a style
    choice, and weighting by actual instead over-orders the whole catalogue (see
    `tests/test_policy.py::test_ratio_curve_reproduces_the_grid_optimum`).

    `min_n` has to be an argument rather than a constant. The catalogue-wide fit has tens
    of thousands of rows and can afford a high floor; a PER-SKU fit has a few dozen at
    most, because origins are weekly and there are only two training folds. A floor of 50 applied to both
    silently returns 1.0 for every SKU -- i.e. "no correction" dressed up as a fitted
    one, which reads as the per-SKU arm not working rather than never having run.
    """
    ok = p > 0
    if ok.sum() < min_n:
        return 1.0
    r = y[ok] / p[ok]
    o = np.argsort(r)
    w = p[ok][o]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(common.target_quantile(), cw, r[o]))


# ======================================================================================
# The arms
# ======================================================================================

def fit_global(y, p):
    k = optimal_k(y, p)
    return lambda q: q * k, {"k": round(k, 4)}


def fit_loglinear(y, p):
    """p -> exp(b) * p^a, fitted by weighted least squares in log space.

    Motivated by the log-link story (a power law inverts a multiplicative pull in log
    space, and `a > 1` would be its signature). Unless that story is confirmed, treat
    this simply as a two-parameter shape that happens to fit a curved calibration error. Fitted on log(actual) vs log(predicted), then shifted so
    the result sits at the cost-minimising quantile rather than the mean.
    """
    ok = (p > 0) & (y > 0)
    if ok.sum() < 200:
        return fit_global(y, p)
    lx, ly = np.log(p[ok]), np.log(y[ok])
    w = p[ok]                                     # weight by volume, as the metric does
    a, b = np.polyfit(lx, ly, 1, w=np.sqrt(w))

    def raw(q):
        return np.where(q > 0, np.exp(b) * np.power(np.maximum(q, 1e-9), a), q)

    # The power law targets the conditional geometric mean; lift it to the target
    # quantile with one scalar, the same instrument the global arm uses.
    k = optimal_k(y, raw(p))
    return (lambda q: raw(q) * k), {"a": round(float(a), 4), "b": round(float(b), 4),
                                    "k": round(k, 4)}


def fit_isotonic(y, p):
    """p -> p * g(log p) with g monotone, fitted on binned log-predicted.

    Non-parametric version of the same idea: let the correction vary with predicted
    volume, but force it to vary monotonically so it cannot chase noise bin to bin.
    """
    ok = p > 0
    if ok.sum() < 500:
        return fit_global(y, p)
    lp = np.log(p[ok])
    edges = np.quantile(lp, np.linspace(0, 1, ISO_BINS + 1))
    edges = np.unique(edges)
    if len(edges) < 4:
        return fit_global(y, p)
    idx = np.clip(np.digitize(lp, edges[1:-1]), 0, len(edges) - 2)

    centres, ks = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 30:
            continue
        centres.append(float(np.mean(lp[m])))
        ks.append(optimal_k(y[ok][m], p[ok][m], min_n=30))
    if len(ks) < 3:
        return fit_global(y, p)

    # Pool-adjacent-violators: force the multiplier to be monotone in log-volume.
    ks = np.asarray(ks, dtype=float)
    for _ in range(len(ks)):
        bad = np.flatnonzero(np.diff(ks) < 0)
        if not len(bad):
            break
        i = bad[0]
        ks[i:i + 2] = ks[i:i + 2].mean()
    centres = np.asarray(centres, dtype=float)

    def f(q):
        lq = np.log(np.maximum(q, 1e-9))
        k = np.interp(lq, centres, ks)
        return np.where(q > 0, q * k, q)

    return f, {"bins": len(ks), "k_lo": round(float(ks[0]), 3),
               "k_hi": round(float(ks[-1]), 3)}


def fit_per_sku(y, p, sku):
    kg = optimal_k(y, p)
    df = pl.DataFrame({"sku": sku, "y": y, "p": p})
    per = {}
    for s, g in df.group_by("sku"):
        name = s[0] if isinstance(s, tuple) else s
        yy = g["y"].to_numpy().astype(float)
        pp = g["p"].to_numpy().astype(float)
        if len(yy) < 3 or pp.sum() <= 0:
            continue
        k = optimal_k(yy, pp, min_n=3)
        n = float(pp.sum())
        # Shrink toward the global k by effective volume, not row count: an origin on a
        # 2-unit SKU is not evidence worth the same as one on a 2,000-unit SKU.
        wgt = n / (n + PER_SKU_PRIOR)
        per[name] = wgt * k + (1 - wgt) * kg

    def f(q, s=None):
        if s is None:
            return q * kg
        k = np.array([per.get(x, kg) for x in s], dtype=float)
        return q * k

    return f, {"k_global": round(kg, 4), "skus": len(per)}


# ======================================================================================
# Experiment
# ======================================================================================

ARMS = ("none", "global", "loglinear", "isotonic", "per SKU")


def run_model(model: str, horizons: list[int]) -> list[dict]:
    d = pl.read_parquet(results_dir() / "_preds" / f"{model}.parquet")
    folds = sorted(d["fold"].unique().to_list())
    if len(folds) < 3:
        return []
    train_folds, test_fold = folds[:-1], folds[-1]

    out = []
    for h in horizons:
        s = d.filter(pl.col("h") == h)
        tr = s.filter(pl.col("fold").is_in(train_folds))
        te = s.filter(pl.col("fold") == test_fold)
        if tr.height < 500 or te.height < 200:
            continue
        ytr = tr["actual"].to_numpy().astype(float)
        ptr = tr["predicted"].to_numpy().astype(float)
        yte = te["actual"].to_numpy().astype(float)
        pte = te["predicted"].to_numpy().astype(float)
        ste = te["sku"].to_list()

        fitted = {
            "none": (lambda q: q, {}),
            "global": fit_global(ytr, ptr),
            "loglinear": fit_loglinear(ytr, ptr),
            "isotonic": fit_isotonic(ytr, ptr),
        }
        f_sku, p_sku = fit_per_sku(ytr, ptr, tr["sku"].to_list())

        for arm in ARMS:
            if arm == "per SKU":
                corrected = f_sku(pte, ste)
                params = p_sku
                in_samp = f_sku(ptr, tr["sku"].to_list())
            else:
                fn, params = fitted[arm]
                corrected = fn(pte)
                in_samp = fn(ptr)
            m = score(yte, corrected)
            m_in = score(ytr, in_samp)
            out.append({"model": model, "h": h, "arm": arm, "n": te.height,
                        **m, "cost_in_sample": m_in["cost"],
                        "params": json.dumps(params)})
    return out


# ======================================================================================
# Report
# ======================================================================================

def write_report(df: pl.DataFrame) -> Path:
    cu, co = common.cost_ratio()
    L: list[str] = []
    L.append("# The bias: what shape it has, and what actually corrects it\n")
    L.append(f"*Generated by `dev/debias.py` (mode `{common.MODE}`) from the cached "
             f"predictions in `{results_dir().name}/_preds/`. Nothing was refitted.*\n")
    L.append("Every arm is fitted on **folds 1-2** and scored on **fold 3**, which is the "
             "only honest way to compare corrections of different flexibility: a per-SKU "
             "map has one parameter per SKU and wins in sample by construction. "
             f"Cost at `{cu:g}:{co:g}`.\n")
    L.append(f"![debias]({figures_dir().name}/_debias.png)\n")

    L.append("## Is the bias a shape or a level?\n")
    L.append("Bias decomposed against the model's own predicted volume. If the sign "
             "changes across the quintiles, the bias is a shape, not a level:\n")
    L.append("| model | h | Q1 (smallest) | Q2 | Q3 | Q4 | Q5 (largest) |")
    L.append("|---|---|---|---|---|---|---|")
    for model in sorted(df["model"].unique()):
        d = pl.read_parquet(results_dir() / "_preds" / f"{model}.parquet")
        for h in sorted(df.filter(pl.col("model") == model)["h"].unique()):
            s = d.filter((pl.col("h") == h) & (pl.col("actual") > 0))
            y = s["actual"].to_numpy().astype(float)
            p = s["predicted"].to_numpy().astype(float)
            edges = np.quantile(p, [.2, .4, .6, .8])
            band = np.digitize(p, edges)
            cells = []
            for b in range(5):
                m = band == b
                cells.append(f"{(p[m] - y[m]).sum() / max(y[m].sum(), 1e-9):+.3f}"
                             if m.sum() > 50 else "—")
            L.append(f"| `{model}` | {h} | " + " | ".join(cells) + " |")
    L.append("\nA single global multiplier lifts all five columns by the same factor. "
             "Against a bias that changes sign across the range, that is the wrong shape "
             "of instrument.\n")

    L.append("## Which correction wins, out of sample\n")
    for h in sorted(df["h"].unique()):
        L.append(f"### {h} days\n")
        L.append("| model | arm | cost | vs global | WAPE | bias | fill | in-sample cost |")
        L.append("|---|---|---|---|---|---|---|---|")
        sub = df.filter(pl.col("h") == h)
        for model in sorted(sub["model"].unique()):
            ms = sub.filter(pl.col("model") == model)
            g = ms.filter(pl.col("arm") == "global")
            gc = float(g["cost"][0]) if g.height else float("nan")
            for arm in ARMS:
                r = ms.filter(pl.col("arm") == arm)
                if not r.height:
                    continue
                r = r.to_dicts()[0]
                rel = "—" if arm == "global" else f"{100 * (gc - r['cost']) / gc:+.1f}%"
                L.append(f"| `{model}` | {arm} | **{r['cost']:.4f}** | {rel} | "
                         f"{r['wape']:.3f} | {r['bias']:+.3f} | {r['fill']:.3f} | "
                         f"{r['cost_in_sample']:.4f} |")
        L.append("")

    # the verdict
    wins = {a: 0 for a in ARMS if a != "global"}
    tot = 0
    for (model, h), g in df.group_by(["model", "h"]):
        gr = g.filter(pl.col("arm") == "global")
        if not gr.height:
            continue
        gc = float(gr["cost"][0])
        tot += 1
        for a in wins:
            r = g.filter(pl.col("arm") == a)
            if r.height and float(r["cost"][0]) < gc:
                wins[a] += 1
    L.append("## Verdict\n")
    L.append("### 1. What the global multiplier is worth\n")
    gains = []
    for (model, h), g in df.group_by(["model", "h"]):
        n_ = g.filter(pl.col("arm") == "none")
        g_ = g.filter(pl.col("arm") == "global")
        if n_.height and g_.height:
            gains.append(100 * (float(n_["cost"][0]) - float(g_["cost"][0]))
                         / float(n_["cost"][0]))
    if gains:
        L.append(f"Going from `none` to `global` is worth **{np.mean(gains):.0f}% of "
                 f"inventory cost on average** (range {min(gains):.0f}% to "
                 f"{max(gains):.0f}%) across {len(gains)} model-horizons. That is the "
                 f"default correction; compare every arm below against it.\n")

    L.append("### 2. What each arm adds past the global scalar\n")
    L.append(f"Out of **{tot} model-horizons**, each arm beat the global multiplier in:\n")
    for a, w in sorted(wins.items(), key=lambda x: -x[1]):
        L.append(f"- **{a}** — {w}/{tot}")
    L.append("")
    L.append("Look for structure the counts hide: whether the answer is consistent "
             "*within* a horizon and reverses *between* them:\n")
    L.append("| arm | " + " | ".join(f"{h} d" for h in sorted(df['h'].unique())) + " |")
    L.append("|---" * (len(df["h"].unique()) + 1) + "|")
    for a in ARMS:
        if a in ("none", "global"):
            continue
        cells = []
        for h in sorted(df["h"].unique()):
            s = df.filter(pl.col("h") == h)
            beat = tot_h = 0
            for model in s["model"].unique():
                ms = s.filter(pl.col("model") == model)
                gr = ms.filter(pl.col("arm") == "global")
                ar = ms.filter(pl.col("arm") == a)
                if gr.height and ar.height:
                    tot_h += 1
                    beat += float(ar["cost"][0]) < float(gr["cost"][0])
            cells.append(f"{beat}/{tot_h}")
        L.append(f"| **{a}** | " + " | ".join(cells) + " |")
    L.append("")
    L.append("An arm that sweeps every model at one horizon and loses to every model at "
             "the next is not a property of the correction, it is a property of the "
             "horizon. That pattern does not support adopting one arm catalogue-wide.\n")

    L.append("### 3. Compare with `calibration.py` before trusting a per-SKU verdict\n")
    L.append("`calibration.py` also scores a per-SKU multiplier, but it is a different "
             "*instrument wearing the same name*: it shrinks each SKU's multiplier toward "
             "the global one by ROW COUNT, and this file shrinks by PREDICTED VOLUME "
             f"(prior {PER_SKU_PRIOR} units). If the two disagree, that disagreement is "
             "the finding.\n")
    L.append("Volume-shrinkage barely restrains exactly the high-volume SKUs the "
             "volume-weighted metric is made of, so it can overfit where it costs most. "
             "**A correction whose verdict flips with its own shrinkage constant has not "
             "been established.** In that case prefer the global scalar until one of the "
             "two is tested on more than a single held-out fold.\n")
    L.append("Read the `in-sample cost` column with the rest: an arm whose in-sample cost "
             "sits far below its fold-3 cost was fitting noise, and the gap is how much. "
             "The `global` arm has one parameter and barely moves between the two.\n")
    L.append("**The correction is not a repair.** Every arm rescales a prediction that "
             "was still trained to aim at the middle. The deeper fix — retraining at the "
             "cost-minimising quantile — is a separate experiment. The ceiling on "
             "everything here is set by how much of the error is calibration rather than "
             "accuracy.\n")

    path = LAB / ("DEBIAS.md" if common.MODE == "initial"
                  else f"DEBIAS_{common.MODE.upper()}.md")
    path.write_text("\n".join(L))
    return path


def draw(df: pl.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = sorted(df["h"].unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(5.4 * len(hs), 5), squeeze=False)
    for ax, h in zip(axes[0], hs):
        sub = df.filter(pl.col("h") == h)
        models = sorted(sub["model"].unique())
        x = np.arange(len(models))
        w = 0.8 / len(ARMS)
        for i, arm in enumerate(ARMS):
            vals = []
            for m in models:
                r = sub.filter((pl.col("model") == m) & (pl.col("arm") == arm))
                vals.append(float(r["cost"][0]) if r.height else np.nan)
            ax.bar(x + i * w, vals, w, label=arm, alpha=.9)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{h} days", fontsize=11)
        ax.grid(axis="y", alpha=.25)
    axes[0][0].set_ylabel(f"inventory cost ({common.CU:g}:{common.CO:g}) — lower better")
    axes[0][-1].legend(fontsize=8, frameon=False)
    fig.suptitle("Correcting the bias — every arm fitted on folds 1-2, "
                 "scored on fold 3", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose and correct the forecast bias.")
    ap.add_argument("--model", type=str, default=None,
                    help="one model (default: every cached model)")
    ap.add_argument("--horizons", type=str,
                    default=",".join(str(h) for h in common.HORIZONS))
    ap.add_argument("--tuned", action="store_true")
    args = ap.parse_args()
    common.set_mode("tuned" if args.tuned else "initial")

    horizons = [int(h) for h in args.horizons.split(",")]
    preds = results_dir() / "_preds"
    models = ([args.model] if args.model
              else sorted(p.stem for p in preds.glob("*.parquet")))

    rows = []
    for m in models:
        if not (preds / f"{m}.parquet").exists():
            print(f"  ! no cached predictions for {m}")
            continue
        print(f"  {m} ...", flush=True)
        rows += run_model(m, horizons)
    if not rows:
        raise SystemExit("nothing to score")
    df = pl.DataFrame(rows)

    figures_dir().mkdir(parents=True, exist_ok=True)
    fig = figures_dir() / "_debias.png"
    draw(df, fig)
    rep = write_report(df)
    (results_dir() / "_debias.json").write_text(json.dumps(df.to_dicts(), indent=2))

    print(f"\n{'=' * 84}\nDEBIAS [{common.MODE}]  (fit folds 1-2, scored fold 3)\n{'=' * 84}")
    for h in horizons:
        sub = df.filter(pl.col("h") == h)
        if sub.is_empty():
            continue
        print(f"\n  --- {h} days ---")
        for m in sorted(sub["model"].unique()):
            ms = sub.filter(pl.col("model") == m)
            g = ms.filter(pl.col("arm") == "global")
            gc = float(g["cost"][0]) if g.height else float("nan")
            parts = []
            for arm in ARMS:
                r = ms.filter(pl.col("arm") == arm)
                if not r.height:
                    continue
                c = float(r["cost"][0])
                rel = "" if arm == "global" else f" ({100 * (gc - c) / gc:+.1f}%)"
                parts.append(f"{arm}={c:.4f}{rel}")
            print(f"    {m:20s} " + "  ".join(parts))
    print(f"\nreport -> {rep}\nfigure -> {fig}")


if __name__ == "__main__":
    main()
