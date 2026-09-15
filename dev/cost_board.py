"""Rank the seven models on what a forecast error COSTS, not on how big it is.

    uv run python dev/cost_board.py            # baseline half
    uv run python dev/cost_board.py --tuned    # the tuned half

Reads the cached predictions in `results/_preds/` and refits nothing, so it runs in
seconds. Writes `dev/COST.md` and `<figures>/_cost_board.png`.

-----------------------------------------------------------------------------------
WHY THIS EXISTS ALONGSIDE compare_models.py
-----------------------------------------------------------------------------------
`compare_models.py` ranks on WAPE. WAPE is a sum of absolute errors, so it treats a
unit short and a unit spare as the same mistake. For most inventory they are not: a
lost sale usually costs more than a unit of dead stock, and the cost ratio in
`assumptions.toml` (`CU:CO`) says by how much.

That single change can reorder the board. It also changes what the best forecast IS:
WAPE is minimised by the conditional median, the inventory cost by the CU/(CU+CO)
quantile. Whenever CU > CO that quantile sits above the median, so every model here --
all of which aim at the middle -- is aiming low on purpose and needs correcting.

-----------------------------------------------------------------------------------
THE HONEST PART
-----------------------------------------------------------------------------------
A multiplier fitted and scored on the same rows proves nothing. So:

    the multiplier k AND the model are chosen on folds 1-2
    both are then applied UNCHANGED to fold 3, which is what gets reported

Fold 3 is never consulted during selection. The `oos` columns below are therefore
numbers you could actually have earned. The in-sample column is printed next to them
only so the gap between the two is visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import common
from common import LAB, figures_dir, results_dir

# Ratios reported side by side, as a sensitivity band around the configured `CU:CO`
# (assumptions.toml), shown so that nobody has to take any single ratio on faith. The
# configured one is marked "default" wherever it appears in this list.
RATIOS = (1.5, 2.0, 4.0)

# Multiplier grid. Wide enough to contain the optimum at much steeper ratios than the
# ones above, fine enough that the grid is not the binding constraint on the answer.
KS = np.round(np.arange(0.80, 2.01, 0.02), 2)

HORIZONS = (30, 90, 180)


def newsvendor(y: np.ndarray, p: np.ndarray, cu: float, co: float) -> float:
    """Cost per unit of demand. Identical to demandml.metrics.newsvendor_cost; inlined
    on raw arrays because this file evaluates it ~25k times in the multiplier search."""
    den = np.sum(y)
    if den <= 0:
        return float("nan")
    return float(np.sum(cu * np.maximum(y - p, 0.0) + co * np.maximum(p - y, 0.0)) / den)


def fill(y: np.ndarray, p: np.ndarray) -> float:
    den = np.sum(y)
    return float(1.0 - np.sum(np.maximum(y - p, 0.0)) / den) if den > 0 else float("nan")


def wape(y: np.ndarray, p: np.ndarray) -> float:
    den = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / den) if den > 0 else float("nan")


def load_preds() -> dict[str, pl.DataFrame]:
    d = results_dir() / "_preds"
    if not d.exists():
        raise SystemExit(f"no cached predictions at {d} -- run the models first")
    out = {p.stem: pl.read_parquet(p) for p in sorted(d.glob("*.parquet"))
           if common.model_allowed(p.stem)}   # DEMANDML_MODELS: no stale leftovers
    if not out:
        raise SystemExit(f"no parquet files in {d}")
    return out


def split(df: pl.DataFrame, h: int) -> tuple[tuple, tuple]:
    """(train arrays, test arrays) for one horizon: folds 1-2 vs fold 3."""
    s = df.filter(pl.col("h") == h)
    tr = s.filter(pl.col("fold") < 3)
    te = s.filter(pl.col("fold") == 3)
    return ((tr["actual"].to_numpy(), tr["predicted"].to_numpy()),
            (te["actual"].to_numpy(), te["predicted"].to_numpy()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuned", action="store_true",
                    help="score the tuned half (results_tuned/ + tuned_figures/)")
    args = ap.parse_args()
    common.set_mode("tuned" if args.tuned else "initial")

    preds = load_preds()
    models = sorted(preds)
    cu_default, co = common.cost_ratio()
    print(f"mode={common.MODE}  models={len(models)}  "
          f"default ratio {cu_default:g}:{co:g}  "
          f"target quantile {common.target_quantile():.2f}\n")

    # ---------------------------------------------------------------- the board
    # Every model, unscaled, pooled over all folds, at the default ratio. This is the
    # direct answer to "does ranking on cost instead of WAPE change the order".
    board = []
    for m in models:
        row = {"model": m}
        for h in HORIZONS:
            s = preds[m].filter(pl.col("h") == h)
            y, p = s["actual"].to_numpy(), s["predicted"].to_numpy()
            row[f"wape{h}"] = wape(y, p)
            row[f"cost{h}"] = newsvendor(y, p, cu_default, co)
            row[f"bias{h}"] = (float(np.sum(p - y) / np.sum(np.abs(y))) if s.height
                               else float("nan"))
        # A model run at only some horizons (`LabModel.horizons`, e.g. blend at 90 d) has no
        # all-horizon average: nan, and ranked after every complete model rather than
        # first (a bare sort puts nan wherever it falls). Its own horizons still compete
        # in the per-horizon pick below.
        row["horizons"] = [h for h in HORIZONS if preds[m].filter(pl.col("h") == h).height]
        row["wape"] = float(np.mean([row[f"wape{h}"] for h in HORIZONS]))
        row["cost"] = float(np.mean([row[f"cost{h}"] for h in HORIZONS]))
        row["bias"] = float(np.mean([row[f"bias{h}"] for h in HORIZONS]))
        board.append(row)
    last = lambda key: (lambda r: (bool(np.isnan(r[key])), r[key]))  # noqa: E731
    board.sort(key=last("cost"))
    by_wape = sorted(board, key=last("wape"))

    print("=" * 78)
    print(f"BOARD, unscaled, ranked on inventory cost at {cu_default:g}:{co:g}")
    print("=" * 78)
    print(f"{'#':<3}{'model':<20}{'cost':>8}{'wape':>8}{'bias':>9}   {'WAPE rank':>10}")
    for i, r in enumerate(board, 1):
        wr = by_wape.index(r) + 1
        move = "" if wr == i else f"  ({wr - i:+d})"
        print(f"{i:<3}{r['model']:<20}{r['cost']:>8.4f}{r['wape']:>8.4f}"
              f"{r['bias']:>+9.4f}   {wr:>10}{move}")

    # ------------------------------------------------- out-of-sample uplift per ratio
    rows = []
    for cu in RATIOS:
        for h in HORIZONS:
            hm = [m for m in models if preds[m].filter(pl.col("h") == h).height]
            tr = {m: split(preds[m], h)[0] for m in hm}
            te = {m: split(preds[m], h)[1] for m in hm}

            # What ranking on WAPE would have given you: best-WAPE model, no multiplier.
            base = min(hm, key=lambda m: wape(*tr[m]))
            y_b, p_b = te[base]

            # What ranking on cost gives you: best (model, k) on folds 1-2.
            best_m, best_k, best_c = None, 1.0, float("inf")
            for m in hm:
                y_t, p_t = tr[m]
                for k in KS:
                    c = newsvendor(y_t, p_t * k, cu, co)
                    if c < best_c:
                        best_m, best_k, best_c = m, float(k), c
            y_s, p_s = te[best_m]
            p_s = p_s * best_k

            rows.append(dict(
                cu=cu, h=h, base=base, model=best_m, k=best_k,
                cost_base=newsvendor(y_b, p_b, cu, co),
                cost_oos=newsvendor(y_s, p_s, cu, co),
                cost_is=best_c,
                fill_base=fill(y_b, p_b), fill_oos=fill(y_s, p_s),
                stock=100.0 * (np.sum(p_s) - np.sum(p_b)) / np.sum(p_b),
            ))
    res = pl.DataFrame(rows)
    res = res.with_columns(
        (100.0 * (pl.col("cost_base") - pl.col("cost_oos")) / pl.col("cost_base")).alias("gain"))

    print("\n" + "=" * 78)
    print("CHOSEN ON FOLDS 1-2, SCORED ON FOLD 3 (nothing below is fitted on fold 3)")
    print("=" * 78)
    for cu in RATIOS:
        mark = "  <- default" if cu == cu_default else ""
        print(f"\nCu:Co = {cu:g}:{co:g}   target quantile "
              f"{cu / (cu + co):.2f}{mark}")
        print(f"  {'h':>4}  {'model':<18}{'k':>6}{'cost':>9}{'vs WAPE pick':>14}"
              f"{'fill':>16}{'stock':>9}")
        for r in res.filter(pl.col("cu") == cu).iter_rows(named=True):
            print(f"  {r['h']:>4}  {r['model']:<18}{r['k']:>6.2f}{r['cost_oos']:>9.4f}"
                  f"{r['gain']:>13.1f}%"
                  f"{r['fill_base']:>10.3f} ->{r['fill_oos']:>5.3f}"
                  f"{r['stock']:>8.1f}%")

    tuning = price_tuning(cu_default, co)
    retrained = price_cost_mode(cu_default, co)
    write_report(board, by_wape, res, cu_default, co, tuning, retrained)
    draw(board, res, preds, cu_default, co)


def price_tuning(cu: float, co: float) -> list[dict] | None:
    """What the hyper-parameter search did, valued in inventory cost instead of WAPE.

    Worth computing because the two answers can disagree. The search was driven by WAPE,
    and a WAPE search on a right-skewed panel tends to push bias negative (see
    compare_tuning.py). Under a symmetric metric that is a curiosity. Under an asymmetric
    cost it is a bill, and this is where it gets added up.

    Returns None when only one of the two halves has been run.
    """
    mode = common.MODE
    try:
        common.set_mode("initial")
        base = load_preds()
        common.set_mode("tuned")
        tuned = load_preds()
    except SystemExit:
        return None
    finally:
        common.set_mode(mode)

    rows = []
    for m in sorted(set(base) & set(tuned)):
        r = {"model": m}
        for label, src in (("base", base), ("tuned", tuned)):
            cs, ws = [], []
            for h in HORIZONS:
                s_ = src[m].filter(pl.col("h") == h)
                y, p = s_["actual"].to_numpy(), s_["predicted"].to_numpy()
                cs.append(newsvendor(y, p, cu, co))
                ws.append(wape(y, p))
            r[f"cost_{label}"] = float(np.mean(cs))
            r[f"wape_{label}"] = float(np.mean(ws))
        r["d_cost"] = 100.0 * (r["cost_tuned"] - r["cost_base"]) / r["cost_base"]
        r["d_wape"] = 100.0 * (r["wape_tuned"] - r["wape_base"]) / r["wape_base"]
        rows.append(r)
    rows.sort(key=lambda r: r["d_cost"])
    return rows


def best_k(y: np.ndarray, p: np.ndarray, cu: float, co: float) -> float:
    """The multiplier that minimises cost on the rows given. Only ever called on folds
    1-2; calling it on fold 3 would make every number below a selection, not a result."""
    return float(KS[int(np.argmin([newsvendor(y, p * k, cu, co) for k in KS]))])


def price_cost_mode(cu: float, co: float) -> list[dict] | None:
    """Compare models REBUILT against the cost (results_cost/) with the same models
    merely rescaled (results/ + a multiplier).

    The question this settles: is it better to retrain at the target quantile, or to
    leave the WAPE-trained model alone and lift its output? The multiplier is one scalar
    for every SKU, so the retrained model ought to win -- it can lift a volatile SKU more
    than a predictable one. Whether it actually does is a measurement, not a prediction,
    and it need not: retraining fixes calibration, not accuracy.

    Everything is scored on fold 3; every k is chosen on folds 1-2.
    """
    mode = common.MODE
    try:
        common.set_mode("initial")
        base = load_preds()
        common.set_mode("cost")
        cm = load_preds()
    except SystemExit:
        return None
    finally:
        common.set_mode(mode)

    rows = []
    for m in sorted(set(base) & set(cm)):
        for h in HORIZONS:
            (y_tr, p_tr), (y_te, p_te) = split(base[m], h)
            (yc_tr, pc_tr), (yc_te, pc_te) = split(cm[m], h)
            k_base = best_k(y_tr, p_tr, cu, co)
            k_cost = best_k(yc_tr, pc_tr, cu, co)
            rows.append(dict(
                model=m, h=h,
                base=newsvendor(y_te, p_te, cu, co),
                base_k=newsvendor(y_te, p_te * k_base, cu, co), k_base=k_base,
                cost=newsvendor(yc_te, pc_te, cu, co),
                cost_k=newsvendor(yc_te, pc_te * k_cost, cu, co), k_cost=k_cost,
                fill_cost=fill(yc_te, pc_te),
                wape_cost=wape(yc_te, pc_te), wape_base=wape(y_te, p_te),
            ))
    return rows


def write_report(board, by_wape, res, cu, co, tuning=None, retrained=None) -> None:
    L = []
    A = L.append
    A("# The board, ranked on inventory cost\n")
    A(f"*Generated by `dev/cost_board.py` (mode `{common.MODE}`) from the cached "
      f"predictions in `{results_dir().name}/_preds/`. Nothing was refitted.*\n")
    A("## The ratio\n")
    A(f"`CU:CO = {cu:g}:{co:g}` — one unit short costs {cu:g}x one unit of dead "
      f"stock. The cost-minimising forecast is therefore the "
      f"**{cu / (cu + co):.0%} quantile** of demand, not the middle.\n")
    A("Change it in one place (`assumptions.toml`, `CU` / `CO`) and re-run this file.\n")

    A(f"## Ranked on cost ({cu:g}:{co:g}), unscaled, all folds\n")
    A("| # | model | cost | WAPE | bias | WAPE rank |")
    A("|---|---|---|---|---|---|")
    for i, r in enumerate(board, 1):
        wr = by_wape.index(r) + 1
        move = "" if wr == i else f" ({wr - i:+d})"
        A(f"| {i} | `{r['model']}` | **{r['cost']:.4f}** | {r['wape']:.4f} | "
          f"{r['bias']:+.4f} | {wr}{move} |")
    A("")
    top_c, top_w = board[0]["model"], by_wape[0]["model"]
    if top_c != top_w:
        A(f"**The board reorders.** `{top_w}` leads on WAPE; `{top_c}` leads on cost, "
          f"because its bias is {board[0]['bias']:+.4f} against "
          f"{by_wape[0]['bias']:+.4f}. Between two models of *comparable accuracy*, the "
          f"asymmetric cost is decided by which one aims low — being slightly worse on "
          f"average is cheaper than being systematically short.\n")
        A("Accuracy has not stopped mattering, though: a model with milder bias but "
          "much worse accuracy still ranks low on cost. Bias breaks ties within an "
          "accuracy band; it does not override the band.\n")
    else:
        A(f"`{top_c}` leads on both metrics at this ratio.\n")

    A("## What it is worth, out of sample\n")
    A("Model **and** multiplier chosen on folds 1–2, applied unchanged to fold 3. "
      "The baseline is what ranking on WAPE would have handed you: the best-WAPE "
      "model, unscaled.\n")
    for r_cu in RATIOS:
        mark = " — **default**" if r_cu == cu else ""
        A(f"### {r_cu:g}:{co:g}{mark}\n")
        A("| h | model | k | cost | vs WAPE pick | fill | stock |")
        A("|---|---|---|---|---|---|---|")
        for r in res.filter(pl.col("cu") == r_cu).iter_rows(named=True):
            A(f"| {r['h']} | `{r['model']}` | {r['k']:.2f} | {r['cost_oos']:.4f} | "
              f"**{r['gain']:+.1f}%** | {r['fill_base']:.3f} → {r['fill_oos']:.3f} | "
              f"{r['stock']:+.1f}% |")
        A("")

    if tuning:
        A(f"## What the tuning run cost, at {cu:g}:{co:g}\n")
        A("`dev/tune.py` searched on WAPE. Priced against the inventory cost instead, "
          "the same parameter changes look different — negative is better in both "
          "columns.\n")
        A("| model | WAPE change | cost change |")
        A("|---|---|---|")
        for r in tuning:
            A(f"| `{r['model']}` | {r['d_wape']:+.1f}% | **{r['d_cost']:+.1f}%** |")
        worse = [r for r in tuning if r["d_cost"] > 0]
        flips = [r for r in tuning if r["d_wape"] < 0 < r["d_cost"]]
        A("")
        A(f"**{len(worse)} of {len(tuning)} models are worse under the cost metric** "
          f"after tuning.")
        if flips:
            names = ", ".join(f"`{r['model']}`" for r in flips)
            A(f" {len(flips)} of them ({names}) improved on WAPE while getting worse on "
              f"cost — the search bought its accuracy by forecasting lower, which is "
              f"the one thing an asymmetric cost penalises.")
        A("\nSo the tuned parameters should **not** be adopted wholesale. Adopt only "
          "where both columns agree.\n")

    A("## How to read the multiplier\n")
    A("`k` is a single global scalar applied to every SKU, which is a crude instrument: "
      "it lifts a well-calibrated SKU by as much as a badly-calibrated one. These "
      "numbers are therefore a **lower bound** on what the objective is worth. A "
      "quantile model varies the uplift per SKU and should beat them — "
      "`lgbm_quantile` already fits quantile losses, so pointing it at "
      f"`common.target_quantile()` ({cu / (cu + co):.2f}) costs one keyword, not a new "
      "model.\n")
    if retrained:
        A(f"## Retraining at the {cu / (cu + co):.2f} quantile vs just lifting the output\n")
        A("Every number scored on fold 3; every `k` chosen on folds 1–2. `retrained` is "
          "the model rebuilt against pinball loss at the target quantile "
          "(`results_cost/`); `lifted` is the ordinary WAPE-trained model with a single "
          "global multiplier on its output.\n")
        A("| model | h | as-is | lifted (k) | retrained | retrained + k (k) |")
        A("|---|---|---|---|---|---|")
        for r in retrained:
            A(f"| `{r['model']}` | {r['h']} | {r['base']:.4f} | "
              f"{r['base_k']:.4f} ({r['k_base']:.2f}) | **{r['cost']:.4f}** | "
              f"{r['cost_k']:.4f} ({r['k_cost']:.2f}) |")
        A("")
        wins = sum(1 for r in retrained if r["cost"] < r["base_k"])
        A(f"Retraining beats lifting in **{wins} of {len(retrained)}** model-horizons. ")
        if wins * 2 < len(retrained):
            A("So the per-SKU instrument is *not* automatically better than the blunt "
              "one. Retraining changes what the model aims at; it does not make the "
              "model more accurate, and on a weaker base model the calibration gain "
              "does not cover the accuracy gap.\n")
        still = [r for r in retrained if r["k_cost"] > 1.02]
        if still:
            A(f"\n{len(still)} of {len(retrained)} retrained model-horizons still want "
              f"a multiplier above 1.0, which means aiming at the right quantile of the "
              f"*fitted* distribution does not by itself deliver that quantile of the "
              f"*realised* one.\n")
            A("\n**Stock-out censoring is the obvious suspect, and an unlikely one.** "
              "`prepare_data.py` already nulls any target whose future window was more "
              "than 25% out of stock, so the surviving windows carry few stock-out days. "
              "And censoring pushes the other way: a suppressed actual makes a model "
              "look like it OVER-forecasts. To check, score only the windows with *no* "
              "stock-out days at all; if the models still under-forecast there, the "
              "defect is in the model, not the data. Windows that stocked out often did "
              "so because demand was high, and models tend to miss high demand low.\n")

    A("## What this does not say\n")
    A("- One test fold, chosen on two. Compare any cost difference with the "
      "fold-to-fold spread on the WAPE board: a difference of a few percent between "
      "two models is usually not resolvable. Large uplifts can survive that; the "
      "*ordering* of the middle of the board usually cannot.\n")
    A("- The cost is per unit of demand, not money. It is `cu`·short + `co`·over "
      "divided by total demand, so it is comparable across horizons and equals WAPE "
      "when `cu == co`.\n")
    A("- Nothing here is refitted at the new objective. Every model on this board was "
      "still *trained* to predict the middle; only its output has been rescaled.\n")

    out = LAB / ("COST_TUNED.md" if common.MODE == "tuned" else "COST.md")
    out.write_text("\n".join(L))
    print(f"\nreport -> {out.relative_to(LAB.parent)}")


def draw(board, res, preds, cu, co) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    names = [r["model"] for r in board]

    # 1. cost rank vs WAPE rank -------------------------------------------------------
    ax = axes[0][0]
    by_wape = sorted(board, key=lambda r: r["wape"])
    y_cost = {r["model"]: i for i, r in enumerate(board)}
    y_wape = {r["model"]: i for i, r in enumerate(by_wape)}
    for m in names:
        moved = y_cost[m] != y_wape[m]
        ax.plot([0, 1], [y_wape[m], y_cost[m]], "-o",
                color="crimson" if moved else "0.7", lw=2.2 if moved else 1.2,
                ms=6, zorder=3 if moved else 2)
        ax.text(-0.04, y_wape[m], m, ha="right", va="center", fontsize=9)
        ax.text(1.04, y_cost[m], m, ha="left", va="center", fontsize=9,
                fontweight="bold" if moved else "normal")
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ranked on WAPE", f"ranked on cost {cu:g}:{co:g}"])
    ax.set_yticks([])
    ax.set_title("Changing the objective reorders the board", fontsize=12, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # 2. cost vs multiplier, h=90 -----------------------------------------------------
    ax = axes[0][1]
    for m in names:
        s = preds[m].filter(pl.col("h") == 90)
        y, p = s["actual"].to_numpy(), s["predicted"].to_numpy()
        c = [newsvendor(y, p * k, cu, co) for k in KS]
        ax.plot(KS, c, lw=1.8, label=m)
        ax.plot(KS[int(np.argmin(c))], min(c), "o", ms=5,
                color=ax.lines[-1].get_color())
    ax.axvline(1.0, color="0.4", ls="--", lw=1, label="as forecast")
    ax.set_xlabel("multiplier k applied to the forecast")
    ax.set_ylabel(f"inventory cost per unit ({cu:g}:{co:g})")
    ax.set_title("Where each model's optimum sits (above 1.0 = it aims low)",
                 fontsize=12, loc="left")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # 3. the fill / stock trade-off ---------------------------------------------------
    ax = axes[1][0]
    for h, mk in zip(HORIZONS, ("o", "s", "^")):
        d = res.filter(pl.col("h") == h).sort("cu")
        ax.plot(d["stock"], 100 * d["fill_oos"], mk + "-", lw=1.8, ms=7, label=f"h={h}")
        for r in d.iter_rows(named=True):
            ax.annotate(f"{r['cu']:g}:1", (r["stock"], 100 * r["fill_oos"]),
                        textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_xlabel("extra stock carried vs the WAPE pick (%)")
    ax.set_ylabel("demand actually met (%)")
    ax.set_title("What each ratio buys, and what it costs to buy it",
                 fontsize=12, loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 4. bias explains the reorder ----------------------------------------------------
    ax = axes[1][1]
    xs = [r["bias"] for r in board]
    ys = [r["cost"] for r in board]
    ax.scatter(xs, ys, s=70, color="crimson", zorder=3)
    for r in board:
        ax.annotate(r["model"], (r["bias"], r["cost"]), textcoords="offset points",
                    xytext=(7, 3), fontsize=9)
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set_xlabel("bias  (negative = forecasts low)")
    ax.set_ylabel(f"inventory cost per unit ({cu:g}:{co:g})")
    ax.set_title("Among comparably accurate models, bias decides the order",
                 fontsize=12, loc="left")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Inventory-cost board — {common.MODE} — cu:co = {cu:g}:{co:g}, "
                 f"target quantile {cu / (cu + co):.2f}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = figures_dir() / "_cost_board.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"figure -> {out.relative_to(LAB.parent)}")


if __name__ == "__main__":
    main()
