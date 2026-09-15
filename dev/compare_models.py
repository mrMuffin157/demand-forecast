"""THE SCOREBOARD — compares every model that has been run and writes the report.

    uv run python dev/compare_models.py              # report from stored results
    uv run python dev/compare_models.py --run-all    # re-run every model first

Each model file writes dev/results/<name>.json when it runs. This file reads all of
them, ranks the models, works out WHY each one is where it is, and writes:

    dev/REPORT.md                  the written comparison
    dev/initial_figures/_comparison_all.png    one chart per horizon, every model on it

Nothing here re-fits anything (unless you pass --run-all), so it is instant, and a model
you ran last week still appears next to one you ran today. Delete a JSON from results/ to
drop that model off the board.

HOW THE RANKING IS DECIDED
--------------------------
Primary key is mean WAPE across the three horizons -- total absolute error divided by
total units, averaged over the 1-, 3- and 6-month questions. Two things then qualify it,
and both appear in every model's verdict:

* wape_sd, the spread across the splits. A model that beats another by less than its
  own spread has not actually beaten it, and the report says so in those words.
* shortfall and bias. Two models with the same WAPE are not equivalent if one gets there
  by under-forecasting: that WAPE is bought with stockouts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common as K  # noqa: E402
from common import HORIZON_LABEL, HORIZONS, LAB, figures_dir, results_dir, set_mode  # noqa: E402

#: Which report the current mode writes. The tuned board is a separate document so the
#: two can be read side by side; compare_tuning.py is the thing that diffs them.
REPORTS = {"initial": LAB / "REPORT.md", "tuned": LAB / "REPORT_TUNED.md"}
BASELINE = "moving_average"  # the model everything is measured against

#: The seven shortlisted models, in the order they should be run and listed.
#: Nothing else belongs on this board -- see docs/FINDINGS.md for what was dropped and why.
MODEL_FILES = [
    "moving_average.py",   # the floor everything is measured against
    "croston.py",          # intermittent-demand specialist
    "random_forest.py",    # bagged trees
    "xgboost_gbdt.py",     # boosted trees
    "lightgbm_additive.py",# boosted trees, per-month then summed
    "lgbm_quantile.py",    # P10/P50/P90 instead of a point
    "two_stage.py",        # P(sale) x E[units | sale]
    "blend.py",            # mean of xgb, xgb + xf_*, rf + xf_* (the eighth, added later)
]


# ======================================================================================
# Running / loading
# ======================================================================================

def run_all(passthrough: list[str]) -> list[str]:
    """Re-run every model file, so the scoreboard is rebuilt from scratch -- every model
    `DEMANDML_MODELS` allows (all seven when it is unset). Returns the ones that failed.

    A failure does not stop the others, but the caller must not treat the board as
    rebuilt: the failed model's results file from the PREVIOUS retrain is still on disk,
    and `forecast.choose` would pick a model and multiplier off that stale backtest."""
    files = [f for f in MODEL_FILES if K.model_allowed(Path(f).stem)]
    if len(files) < len(MODEL_FILES):
        print(f"  DEMANDML_MODELS: running {len(files)} of {len(MODEL_FILES)} -- "
              f"{', '.join(Path(f).stem for f in files)}")
    failed = []
    for i, f in enumerate(files, 1):
        path = LAB / f
        if not path.exists():
            print(f"  ! {f} not found, skipping")
            failed.append(Path(f).stem)
            continue
        print(f"\n########## [{i}/{len(files)}] {f} {'#' * 40}")
        r = subprocess.run([sys.executable, str(path), *passthrough], cwd=str(LAB))
        if r.returncode != 0:
            print(f"  ! {f} exited with code {r.returncode} -- continuing without it")
            failed.append(Path(f).stem)
    return failed


def load_results() -> list[dict]:
    RESULTS = results_dir()
    if not RESULTS.exists():
        raise SystemExit(f"no results in {RESULTS.name}/ yet. Run a model, or: "
                         f"uv run python dev/compare_models.py --run-all"
                         f"{' --tuned' if K.MODE == 'tuned' else ''}")
    # A leading underscore marks an auxiliary file (segment_report.py's _segments.json,
    # ...), not a model board entry -- same convention segment_report.py's
    # best_on_board() already uses.
    # ... and a model DEMANDML_MODELS leaves out is not on the board, even when an old
    # results file of it is still on disk (see `common.active_models`).
    out = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))
           if not p.name.startswith("_") and K.model_allowed(p.stem)]
    if not out:
        raise SystemExit(f"{RESULTS.name}/ is empty. Run a model first.")
    return out


def board_key(results: list[dict]):
    """Sort key for the overall board: models with every horizon first, by mean WAPE; a
    model run at only some (`LabModel.horizons`, e.g. blend at 90 d) after them. Its mean
    covers different horizons, so ranking it against the rest would compare 90-day error
    with an average that includes the easier 30 and harder 180."""
    full = max(len(r["horizons"]) for r in results)
    return lambda r: (len(r["horizons"]) < full, mean_wape(r))


def partial(r: dict, ranked: list[dict]) -> bool:
    return len(r["horizons"]) < max(len(x["horizons"]) for x in ranked)


def mean_wape(r: dict) -> float:
    """Average WAPE over the horizons this model actually has results for."""
    vals = [h["wape"] for h in r["horizons"].values()]
    return float(np.mean(vals)) if vals else float("inf")


# ======================================================================================
# Verdicts -- the "why is it ranked here" text, derived from the numbers
# ======================================================================================

def verdict(r: dict, ranked: list[dict], base: dict | None) -> str:
    """Build the explanation for one model's position, from its own measurements."""
    bits: list[str] = []
    pos = ranked.index(r) + 1
    best = ranked[0]

    if partial(r, ranked):
        hs = ", ".join(f"{h} d" for h in sorted(r["horizons"], key=int))
        bits.append(
            f"Run at {hs} only, so not ranked on the all-horizon mean ({mean_wape(r):.3f} "
            f"covers {hs}); compare it in that horizon's table. ")
    elif pos == 1:
        bits.append(
            f"**Best overall** at {mean_wape(r):.3f} mean WAPE. ")
    else:
        gap = mean_wape(r) - mean_wape(best)
        spread = float(np.mean([h["wape_sd"] for h in r["horizons"].values()] or [0]))
        real = "a real gap" if gap > spread else (
            "within its own split-to-split spread, so this ordering is not settled")
        bits.append(
            f"Ranked #{pos} at {mean_wape(r):.3f} mean WAPE, {gap:+.3f} against "
            f"`{best['model']}` — {real}. ")

    if base is not None and r["model"] != BASELINE:
        wins, losses = [], []
        for h in HORIZONS:
            k = str(h)
            if k in r["horizons"] and k in base["horizons"]:
                d = (base["horizons"][k]["wape"] - r["horizons"][k]["wape"]) \
                    / base["horizons"][k]["wape"] * 100
                (wins if d > 0 else losses).append(f"{HORIZON_LABEL[h]} {d:+.1f}%")
        if wins:
            bits.append(f"Beats the moving average at {', '.join(wins)}. ")
        if losses:
            bits.append(f"Loses to it at {', '.join(losses)}. ")
        if not wins:
            bits.append(
                "It does not beat one trailing average anywhere, which is the result "
                "worth taking seriously rather than explaining away. ")

    sf = float(np.mean([h["shortfall"] for h in r["horizons"].values()]))
    bi = float(np.mean([h["bias"] for h in r["horizons"].values()]))
    if bi < -0.05:
        bits.append(
            f"Systematically **under**-forecasts (bias {bi:+.3f}, shortfall {sf:.3f}): "
            f"whatever it saves on accuracy it spends on stockouts. ")
    elif bi > 0.05:
        bits.append(
            f"Systematically **over**-forecasts (bias {bi:+.3f}): safe on service, "
            f"expensive in tied-up cash and dead stock. ")
    else:
        bits.append(f"Close to unbiased (bias {bi:+.3f}, shortfall {sf:.3f}). ")

    secs = float(np.mean([h["seconds"] for h in r["horizons"].values()]))
    if secs < 1:
        bits.append("Costs nothing to run.")
    elif secs < 60:
        bits.append(f"About {secs:.0f}s to fit per split.")
    else:
        bits.append(f"About {secs / 60:.1f} minutes to fit per split — the expensive "
                    f"option, which its margin has to justify.")
    return "".join(bits)


# ======================================================================================
# Figure
# ======================================================================================

def comparison_figure(ranked: list[dict], out: Path) -> Path:
    hs = [h for h in HORIZONS if any(str(h) in r["horizons"] for r in ranked)]
    fig, axes = plt.subplots(1, len(hs), figsize=(6.0 * len(hs), 0.55 * len(ranked) + 3.4))
    axes = np.atleast_1d(axes)
    fig.suptitle("Which model forecasts best, at each planning horizon",
                 fontsize=15, fontweight="bold")

    for ax, h in zip(axes, hs):
        rows = [r for r in ranked if str(h) in r["horizons"]]
        rows.sort(key=lambda r: r["horizons"][str(h)]["wape"], reverse=True)
        names = [r["model"] for r in rows]
        vals = [r["horizons"][str(h)]["wape"] for r in rows]
        errs = [r["horizons"][str(h)]["wape_sd"] for r in rows]
        # Green for the winner, grey-blue for the rest, orange for the baseline.
        colors = ["#2F9E44" if v == min(vals) else
                  "#F76707" if n == BASELINE else "#4C6EF5"
                  for n, v in zip(names, vals)]
        ax.barh(names, vals, xerr=errs, color=colors, height=0.62,
                error_kw={"ecolor": "#555", "capsize": 3, "lw": 1})
        for i, v in enumerate(vals):
            ax.text(v + max(errs) * 0.35 + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, max(v + e for v, e in zip(vals, errs)) * 1.22)
        ax.set_title(f"{HORIZON_LABEL[h]}  (next {h} days)", fontsize=12)
        ax.set_xlabel("WAPE — lower is better   (bars = spread across the splits)")
        ax.grid(True, axis="x", color="#D8DEE9", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.text(0.5, 0.012, "green = best at that horizon   |   orange = the "
             "moving-average baseline   |   whiskers overlapping means the ordering "
             "between those two is not established", ha="center", fontsize=9,
             color="#555")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ======================================================================================
# Report
# ======================================================================================

def _table(ranked: list[dict], h: int) -> str:
    rows = [r for r in ranked if str(h) in r["horizons"]]
    rows.sort(key=lambda r: r["horizons"][str(h)]["wape"])
    base = next((r for r in rows if r["model"] == BASELINE), None)
    out = [f"| # | model | WAPE | ± split spread | bias | shortfall | "
           f"better than `{BASELINE}` by |",
           "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        m = r["horizons"][str(h)]
        if base is not None:
            b = base["horizons"][str(h)]["wape"]
            delta = f"{(1 - m['wape'] / b) * 100:+.1f}%" if r["model"] != BASELINE else "—"
        else:
            delta = "—"
        out.append(f"| {i} | `{r['model']}` | **{m['wape']:.3f}** | ±{m['wape_sd']:.3f} "
                   f"| {m['bias']:+.3f} | {m['shortfall']:.3f} | {delta} |")
    return "\n".join(out)


def write_report(ranked: list[dict], fig: Path) -> Path:
    base = next((r for r in ranked if r["model"] == BASELINE), None)
    any_run = ranked[0]["run"]
    tuned = K.MODE == "tuned"
    L = [
        f"# Model comparison{' — tuned' if tuned else ''}",
        "",
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} by "
        f"`dev/compare_models.py{' --tuned' if tuned else ''}` from {len(ranked)} "
        f"model runs in `dev/{results_dir().name}/`.*",
        "",
        (f"These are the seven with the hyper-parameters `dev/tune.py` chose, searched "
         f"on folds that end before this board's test period begins. The same board "
         f"with the parameters the model files define is [REPORT.md](REPORT.md), and "
         f"[TUNING.md](TUNING.md) is the difference between them — read that one before "
         f"quoting an improvement, because it is the only one that checks whether a "
         f"change is bigger than the fold-to-fold noise."
         if tuned else
         f"These are the seven with the hyper-parameters their own files define — no "
         f"search, and therefore nothing selected against any window of this data. The "
         f"tuned board is [REPORT_TUNED.md](REPORT_TUNED.md) and the difference between "
         f"them is [TUNING.md](TUNING.md)."),
        "",
        "Every model below answers the same three questions — how many units will this "
        "SKU sell over the next **30**, **90** and **180** days — on the same data, the "
        "same rolling-origin splits and the same metrics. Scores are the average "
        "over the splits.",
        "",
        f"Setup: {any_run['n_skus']:,} SKUs, {any_run['folds']} splits of "
        f"{any_run['test_days']} days each, {any_run['n_features']} features available.",
        "",
        f"![comparison]({fig.relative_to(LAB).as_posix()})",
        "",
        "## Overall ranking",
        "",
        "Mean WAPE across the three horizons. Lower is better.",
        "",
        "| # | model | family | mean WAPE | 1 month | 3 months | 6 months |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, 1):
        cells = [f"{r['horizons'][str(h)]['wape']:.3f}" if str(h) in r["horizons"]
                 else "—" for h in HORIZONS]
        L.append(f"| {i} | `{r['model']}` | {r['family']} | **{mean_wape(r):.3f}** | "
                 + " | ".join(cells) + " |")

    L += ["", "## How to read this", "",
          "**WAPE** is total absolute error divided by total units. 0.20 means that "
          "across all SKUs the forecast was off by 20% of the volume actually sold. "
          "MAPE is not used because a sizeable share of SKU-days are zero, where MAPE is "
          "undefined.",
          "",
          "**± split spread** is the standard deviation of WAPE across the splits. "
          "It is the column that decides whether a ranking means anything: *if one model "
          "beats another by less than its own spread, it has not beaten it.*",
          "",
          "**bias** is signed. Negative means under-forecasting — which looks fine in "
          "WAPE and costs you stockouts and lost sales. Positive means over-forecasting "
          "— dead stock and tied-up cash. **shortfall** is the share of real demand you "
          "would have failed to supply had you ordered exactly the forecast.",
          "",
          "Accuracy also gets harder with horizon, so compare models *within* a column, "
          "not across them.",
          ""]

    for h in HORIZONS:
        if not any(str(h) in r["horizons"] for r in ranked):
            continue
        L += [f"## {HORIZON_LABEL[h].capitalize()} — total units over the next {h} days",
              "", _table(ranked, h), ""]

    L += ["## Why each model is ranked where it is", ""]
    for r in ranked:
        L += [f"### {ranked.index(r) + 1}. `{r['model']}` — {r['family']}", "",
              r["blurb"], "",
              verdict(r, ranked, base), "",
              "**Advantages**", ""]
        L += [f"- {p}" for p in r["pros"]]
        L += ["", "**Disadvantages**", ""]
        L += [f"- {c}" for c in r["cons"]]
        figs = sorted((figures_dir() / "all" / r["model"]).glob("*.png"))
        if figs:
            L += ["", "Its own forecast-vs-actual figures: "
                  + ", ".join(f"[{p.stem}]({p.relative_to(LAB).as_posix()})"
                              for p in figs)]
        L += [""]

    L += ["## Caveats worth keeping in mind", "",
          "- A few splits are enough to notice instability, not enough to settle a close "
          "race. Raise `--folds` if a margin matters.",
          "- Every score here is an average over the whole catalogue. A model that wins overall "
          "can lose badly on the slow movers; per-class breakdowns live in the main "
          "project (`uv run demandml report`).",
          "- The tree models were given a cap on training rows so the folder runs in "
          "minutes. Removing it (`--max-train-rows 0`) usually helps them a little.",
          "- These are point forecasts. A replenishment decision needs a *range* — the "
          "main project's `lgbm_quantile` model gives P10/P50/P90, and the P90 minus P50 "
          "gap is your safety stock.",
          ""]
    report = REPORTS[K.MODE]
    report.write_text("\n".join(L))
    return report


# ======================================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rank every model and write the report.")
    ap.add_argument("--run-all", action="store_true",
                    help="re-run every model file before building the report")
    ap.add_argument("--tuned", action="store_true",
                    help="build the board from results_tuned/ (the run made with the "
                         "hyper-parameters tune.py found) instead of results/")
    args, passthrough = ap.parse_known_args()
    set_mode("tuned" if args.tuned else "initial")
    if args.tuned:
        # Every model file needs the same flag, or --run-all would rebuild the INITIAL
        # results and silently overwrite the "before" half of the comparison.
        passthrough = [*passthrough, "--tuned"]

    failed = run_all(passthrough) if args.run_all else []  # extra flags go to each model

    results = load_results()
    results.sort(key=board_key(results))
    fig = comparison_figure(results, figures_dir() / "_comparison_all.png")
    path = write_report(results, fig)

    print(f"\n{'=' * 78}")
    print(f"BOARD [{K.MODE}]")
    print(f"{'#':>2}  {'model':<22} {'mean WAPE':>10}   " +
          "  ".join(f"{HORIZON_LABEL[h]:>13}" for h in HORIZONS))
    for i, r in enumerate(results, 1):
        cells = "  ".join(
            f"{r['horizons'][str(h)]['wape']:>13.3f}" if str(h) in r["horizons"]
            else f"{'—':>13}" for h in HORIZONS)
        print(f"{i:>2}  {r['model']:<22} {mean_wape(r):>10.3f}   {cells}")
    print(f"{'=' * 78}")
    print(f"report -> {path.relative_to(LAB.parent)}")
    print(f"chart  -> {fig.relative_to(LAB.parent)}")
    if failed:
        # Non-zero so a scheduled retrain stops here and publishes nothing:
        # the board above still carries the failed models' results from the last retrain.
        raise SystemExit(f"FAILED: {', '.join(failed)} did not refit; their results on "
                         f"disk are from an earlier run")
