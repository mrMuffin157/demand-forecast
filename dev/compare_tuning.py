"""BEFORE AND AFTER — what tuning actually changed, and whether it was real.

    uv run python dev/compare_tuning.py

Reads both universes at once:

    results/        + initial_figures/     the seven models as their files define them
    results_tuned/  + tuned_figures/       the same seven after tune.py picked their knobs

and writes:

    dev/TUNING.md                              the written before/after
    dev/tuned_figures/_vs_initial_board.png    every model, both versions, per horizon
    dev/tuned_figures/_vs_initial_delta.png    the change, against the noise floor
    dev/tuned_figures/_vs_initial_transfer.png did the search's gain survive the test?
    dev/tuned_figures/_vs_initial_cost.png     what the WAPE change was bought with
    dev/tuned_figures/_vs_initial_by_volume.png where in the catalogue it landed
    dev/tuned_figures/_vs_initial_per_sku.png  SKU by SKU, not just in aggregate

=======================================================================================
THE QUESTION THIS FILE EXISTS TO ANSWER
=======================================================================================
Not "did WAPE go down" — it almost always goes down somewhere, and a search with enough
trials will always find something. The three questions that matter are:

1.  **Is the change bigger than the noise?** Every score here is the mean of several
    rolling-origin splits, and `wape_sd` is how much it moved between them. A tuned model that
    improves by less than the split-to-split spread of either version has not been shown
    to be better. `_vs_initial_delta.png` draws that noise floor explicitly, and the
    verdict column in TUNING.md refuses to call anything inside it an improvement.

2.  **Did the gain transfer?** tune.py measured every candidate on inner folds that sit
    strictly BEFORE the leaderboard's test period (see its header). So there are two
    numbers per model per horizon: how much better the winner looked during the search,
    and how much better it actually is on the test period. When the second is much
    smaller than the first, the search fitted the search window — which is the normal
    failure mode of tuning and the reason it is worth measuring rather than assuming.
    `_vs_initial_transfer.png` plots one against the other.

3.  **What was it bought with?** WAPE is symmetric and inventory is not. A model can cut
    WAPE by forecasting lower — fewer units of error, more stockouts — and the board
    will applaud. `_vs_initial_cost.png` puts the WAPE change next to the bias and
    shortfall change so that trade is visible instead of hidden.

A model whose tuned score is WORSE is left on the board as it is. That is a result: it
says this dataset cannot distinguish the two settings, or that the search window was not
representative of the test window, and both are things worth knowing before anyone
spends a week on a bigger search.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GRID_C, HORIZON_LABEL, HORIZONS, LAB, MODES, TUNED_PARAMS  # noqa: E402
from compare_models import MODEL_FILES, mean_wape  # noqa: E402
from dashboard import BANDS, COLOR, TITLE_KW, _clean, band_of, per_sku_wape  # noqa: E402

MODELS = [f[:-3] for f in MODEL_FILES]
INITIAL_RESULTS = LAB / MODES["initial"][0]
TUNED_RESULTS = LAB / MODES["tuned"][0]
OUT = LAB / MODES["tuned"][1]
REPORT = LAB / "TUNING.md"

#: One colour per side, used the same way in every chart on the page.
BEFORE_C = "#ADB5BD"   # grey   = initial, the incumbent
AFTER_C = "#1971C2"    # blue   = tuned
WORSE_C = "#E03131"    # red    = tuning made it worse
BETTER_C = "#2F9E44"   # green  = tuning made it better
NOISE_C = "#FFD8A8"    # amber  = inside the noise floor, no verdict


# ======================================================================================
# Loading both sides
# ======================================================================================

def load_side(results: Path) -> dict[str, dict]:
    out = {}
    for m in MODELS:
        p = results / f"{m}.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
    return out


def load_preds(results: Path, name: str) -> pl.DataFrame | None:
    p = results / "_preds" / f"{name}.parquet"
    return pl.read_parquet(p) if p.exists() else None


def tuned_blob(name: str) -> dict | None:
    p = TUNED_PARAMS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell(res: dict | None, h: int, key: str = "wape") -> float:
    """One metric for one model at one horizon, or NaN if that run does not exist."""
    if not res:
        return float("nan")
    e = res.get("horizons", {}).get(str(h))
    return float(e[key]) if e and key in e else float("nan")


def noise_floor(a: dict | None, b: dict | None, h: int) -> float:
    """How much this comparison could move for free.

    The larger of the two versions' split-to-split spreads. A difference smaller than
    this is not evidence: re-running with different fold boundaries could produce it.
    """
    sds = [cell(a, h, "wape_sd"), cell(b, h, "wape_sd")]
    sds = [s for s in sds if np.isfinite(s)]
    return float(max(sds)) if sds else 0.0


def consistency(before: dict, after: dict, name: str) -> tuple[int, int, str]:
    """How many horizons moved which way, and what that agreement is worth.

    The per-horizon noise test is deliberately strict and treats each horizon as its own
    experiment. But three horizons of the same model agreeing is itself evidence the
    strict test throws away: if tuning did nothing, each horizon is a coin flip, so all
    three landing on the same side happens 2/8 of the time. That is not proof, and it is
    not independent either -- the three share a dataset and a catalogue -- but a clean
    sweep is worth saying out loud next to three individually-inconclusive rows.
    """
    deltas = [cell(after[name], h) - cell(before[name], h) for h in HORIZONS]
    deltas = [d for d in deltas if np.isfinite(d)]
    better = sum(d < 0 for d in deltas)
    n = len(deltas)
    if n < 2:
        note = ""
    elif better == n:
        note = (f"all {n} horizons improved — a clean sweep, which is 1-in-"
                f"{2 ** (n - 1)} if tuning were doing nothing")
    elif better == 0:
        note = (f"all {n} horizons got worse — also 1-in-{2 ** (n - 1)}, and pointing "
                f"the other way")
    else:
        note = f"{better} of {n} horizons improved — no consistent direction"
    return better, n, note


def verdict_for(before: float, after: float, floor: float) -> tuple[str, str]:
    """(label, colour) for one before/after pair."""
    if not (np.isfinite(before) and np.isfinite(after)):
        return "not run", BEFORE_C
    d = after - before
    if abs(d) <= floor:
        return "inside the noise", NOISE_C
    return ("better", BETTER_C) if d < 0 else ("worse", WORSE_C)


# ======================================================================================
# 1 · The board, both versions
# ======================================================================================

def fig_board(before: dict, after: dict, out: Path) -> Path:
    """Every model at every horizon, initial bar next to tuned bar."""
    names = sorted(set(before) | set(after),
                   key=lambda n: mean_wape(after.get(n) or before[n]))
    fig, axes = plt.subplots(1, len(HORIZONS) + 1, figsize=(20, 6.4))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.30, wspace=0.22)
    fig.suptitle("Before and after tuning — every model, every horizon", x=0.055,
                 y=0.955, ha="left", fontsize=19, fontweight="bold", color="#212529")
    fig.text(0.055, 0.895,
             "grey = the shortlist as its files define it   ·   blue = after tune.py chose "
             "the hyper-parameters   ·   whiskers = spread across the splits",
             ha="left", fontsize=11, color="#495057")
    fig.text(0.055, 0.855,
             "Where the whiskers overlap, the two versions are not distinguishable on "
             "this data no matter which bar is shorter.",
             ha="left", fontsize=10, color="#868E96", style="italic")

    panels = [(h, HORIZON_LABEL[h].capitalize()) for h in HORIZONS] + \
             [(None, "Mean over the three horizons")]
    for ax, (h, title) in zip(axes, panels):
        x = np.arange(len(names))
        if h is None:
            b = [mean_wape(before[n]) if n in before else np.nan for n in names]
            a = [mean_wape(after[n]) if n in after else np.nan for n in names]
            be = ae = [0.0] * len(names)
        else:
            b = [cell(before.get(n), h) for n in names]
            a = [cell(after.get(n), h) for n in names]
            be = [cell(before.get(n), h, "wape_sd") for n in names]
            ae = [cell(after.get(n), h, "wape_sd") for n in names]
        ax.bar(x - 0.20, b, width=0.38, color=BEFORE_C, yerr=be, label="initial",
               error_kw={"ecolor": "#495057", "capsize": 3, "lw": 1.0})
        ax.bar(x + 0.20, a, width=0.38, color=AFTER_C, yerr=ae, label="tuned",
               error_kw={"ecolor": "#1864AB", "capsize": 3, "lw": 1.0})
        for xi, bv, av in zip(x, b, a):
            if np.isfinite(bv) and np.isfinite(av):
                d = (av - bv) / bv * 100
                ax.text(xi, max(bv, av) * 1.02 + 0.006, f"{d:+.1f}%", ha="center",
                        fontsize=8.5, fontweight="bold",
                        color=BETTER_C if d < 0 else WORSE_C)
        ax.set_xticks(x, names, rotation=38, ha="right", fontsize=9)
        ax.set_title(title, **TITLE_KW)
        ax.set_ylabel("WAPE — lower is better")
        _clean(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=10.5,
               loc="upper right", bbox_to_anchor=(0.99, 0.945), ncols=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


# ======================================================================================
# 2 · The change, against the noise floor
# ======================================================================================

def fig_delta(before: dict, after: dict, out: Path) -> Path:
    """Signed WAPE change per model per horizon, with the noise floor drawn behind it.

    The amber band is +/- the larger of the two versions' split-to-split spreads. A bar
    that does not escape the band is not a result, however long it is.
    """
    names = [n for n in MODELS if n in before and n in after]
    # Height follows the number of models so the bars keep a constant thickness whether
    # two models have been tuned or all seven have.
    height = 2.4 + 0.62 * len(names)
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(17.5, height), sharey=False)
    head = 1.55 / height          # keep the title block a fixed number of inches tall
    fig.subplots_adjust(left=0.105, right=0.99, top=1 - head, bottom=0.62 / height,
                        wspace=0.26)
    fig.suptitle("How much moved, and was it more than noise?", x=0.06,
                 y=1 - 0.36 / height, ha="left", fontsize=19, fontweight="bold",
                 color="#212529")
    fig.text(0.06, 1 - 0.72 / height,
             "Bars are tuned WAPE minus initial WAPE. Left of zero = tuning helped.",
             ha="left", fontsize=11, color="#495057")
    fig.text(0.06, 1 - 0.99 / height,
             "The amber band is the split-to-split spread of the noisier of the two "
             "versions — anything inside it is a coin flip, not an improvement.",
             ha="left", fontsize=10, color="#868E96", style="italic")

    for ax, h in zip(axes, HORIZONS):
        y = np.arange(len(names))
        deltas = [cell(after[n], h) - cell(before[n], h) for n in names]
        floors = [noise_floor(before[n], after[n], h) for n in names]
        for yi, f in zip(y, floors):
            ax.barh(yi, 2 * f, left=-f, height=0.80, color=NOISE_C, alpha=0.55, zorder=1)
        cols = [verdict_for(cell(before[n], h), cell(after[n], h), f)[1]
                for n, f in zip(names, floors)]
        ax.barh(y, deltas, height=0.44, color=cols, zorder=3)
        for yi, d in zip(y, deltas):
            ax.text(d + (0.0015 if d >= 0 else -0.0015), yi, f"{d:+.3f}",
                    va="center", ha="left" if d >= 0 else "right", fontsize=8.5,
                    fontweight="bold", zorder=4)
        ax.axvline(0, color="#212529", lw=1.1, zorder=2)
        ax.set_yticks(y, names, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel("change in WAPE (tuned − initial)")
        ax.set_title(HORIZON_LABEL[h].capitalize(), **TITLE_KW)
        lim = max(max(abs(np.array(deltas)), default=0.01),
                  max(floors, default=0.01)) * 1.55
        ax.set_xlim(-lim, lim)
        _clean(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (BETTER_C, WORSE_C, NOISE_C)]
    # In the header band rather than inside an axis: when every bar is inside the noise
    # floor the panels are full of amber and a legend placed in one of them lands on a
    # bar, which is precisely the bar a reader is trying to measure.
    fig.legend(handles, ["better than noise", "worse than noise",
                         "inside the noise floor"],
               frameon=False, fontsize=10, ncols=3, loc="upper right",
               bbox_to_anchor=(0.99, 1 - 0.30 / height))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


# ======================================================================================
# 3 · Did the search's gain survive the test period?
# ======================================================================================

def fig_transfer(before: dict, after: dict, out: Path) -> Path:
    """Inner-fold improvement (what the search saw) vs test improvement (what is real).

    Points on the diagonal transferred perfectly. Points below it lost some of the gain
    crossing into the test period, which is the expected and honest outcome. Points in
    the lower-right quadrant improved during the search and got WORSE on the test —
    the search fitted its own window.
    """
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.10)
    fig.suptitle("Did the gain transfer out of the search window?", x=0.09, y=0.965,
                 ha="left", fontsize=19, fontweight="bold", color="#212529")
    fig.text(0.09, 0.905,
             "x = improvement tune.py measured on its inner folds (all before the test "
             "period).   y = improvement on the leaderboard's own folds.",
             ha="left", fontsize=11, color="#495057")
    fig.text(0.09, 0.862,
             "On the dashed line the gain transferred exactly. Below it, some of what "
             "the search found was specific to the window it searched in.",
             ha="left", fontsize=10, color="#868E96", style="italic")

    pts: list[tuple[float, float, str, int]] = []
    for n in MODELS:
        blob = tuned_blob(n)
        if not blob or n not in before or n not in after:
            continue
        for h in HORIZONS:
            e = blob.get("per_horizon", {}).get(str(h))
            if not e:
                continue
            b, a = cell(before[n], h), cell(after[n], h)
            if not (np.isfinite(b) and np.isfinite(a)):
                continue
            pts.append((e["inner_improvement_pct"], (b - a) / b * 100, n, h))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # Fit the axes to the data rather than forcing a symmetric range around zero. Most
    # models cluster within a few percent of the origin while one or two baselines sit
    # far out; a symmetric range sized by the outliers squashes the cluster into an
    # unreadable pile. Zero stays on both axes because the sign of each number is the
    # point, and the diagonal is still y = x whatever the limits are.
    def bounds(vals: list[float]) -> tuple[float, float]:
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = max((hi - lo) * 0.16, 1.5)
        return lo - pad, hi + pad

    xlo, xhi = bounds(xs)
    ylo, yhi = bounds(ys)
    lim = max(xhi - xlo, yhi - ylo)  # one scale for the label-spacing arithmetic
    # Labels are nudged off the marker and flipped to the inside of the plot near the
    # right edge, and points that would collide are stacked downwards, so a cluster at
    # the origin stays readable instead of overprinting itself.
    used: list[tuple[float, float]] = []
    for x, y, n, h in sorted(pts, key=lambda p: (-p[1], p[0])):
        ax.scatter(x, y, s={30: 70, 90: 150, 180: 270}[h],
                   color=COLOR.get(n, "#495057"), alpha=0.85, edgecolor="white",
                   lw=1.3, zorder=3)
        right = x < (xlo + xhi) / 2 + (xhi - xlo) * 0.22
        # Labels are long, so two points only a few percent apart horizontally still
        # collide. Step downwards until this one clears every label already placed.
        dy = 5.0
        for _ in range(12):
            if not any(abs(x - ux) < (xhi - xlo) * 0.34
                       and abs(y + dy * lim / 55 - uy) < (yhi - ylo) * 0.062
                       for ux, uy in used):
                break
            dy -= 13.0
        used.append((x, y + dy * lim / 55))
        ax.annotate(f"{n} {h}d", (x, y), textcoords="offset points",
                    xytext=(9 if right else -9, dy), fontsize=8, color="#495057",
                    ha="left" if right else "right", va="center", zorder=4)

    d0, d1 = min(xlo, ylo), max(xhi, yhi)
    ax.plot([d0, d1], [d0, d1], ls="--", color="#212529", lw=1.1, zorder=1,
            label="perfect transfer")
    ax.axhline(0, color="#868E96", lw=0.9, zorder=1)
    ax.axvline(0, color="#868E96", lw=0.9, zorder=1)
    ax.fill_between([xlo, xhi], [ylo, ylo], [0, 0], color=WORSE_C, alpha=0.055,
                    zorder=0)
    ax.text(xhi - (xhi - xlo) * 0.02, ylo + (yhi - ylo) * 0.02,
            "search found a gain that did not survive",
            ha="right", va="bottom", fontsize=9.5, color=WORSE_C, style="italic")
    ax.set_xlim(xlo, xhi), ax.set_ylim(ylo, yhi)
    ax.set_xlabel("improvement on the search's own inner folds (%)")
    ax.set_ylabel("improvement on the leaderboard's test folds (%)")
    ax.set_title("marker size = horizon (30d / 90d / 180d)", **TITLE_KW)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _clean(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


# ======================================================================================
# 3b · Was this model worth tuning at all?
# ======================================================================================

def fig_search(out: Path) -> Path:
    """How much the score moved across the search, per model per horizon.

    The single most useful thing to know before spending hours on a bigger search is
    whether the knobs do anything. Each row here is one model at one horizon, and each
    dot is one of the best trials the search recorded. If the dots sit on top of each
    other, every setting in the space scored the same and the "winner" is the noise
    floor of a 2-fold measurement, not a discovery — tuning that model harder will buy
    nothing. If they spread out, the space matters and a bigger budget might pay.

    The initial parameters are marked separately, because where the incumbent sits in
    that spread is the whole question: near the left edge means it was already good, and
    far right means it was a poor guess.
    """
    rows: list[tuple[str, int, list[float], float, float]] = []
    for n in MODELS:
        blob = tuned_blob(n)
        if not blob:
            continue
        for h in HORIZONS:
            e = blob.get("per_horizon", {}).get(str(h))
            if not e or not e.get("trials"):
                continue
            vals = [t["wape"] for t in e["trials"] if np.isfinite(t["wape"])]
            if vals:
                rows.append((n, h, vals, e["inner_wape_initial"], e["inner_wape"]))
    if not rows:
        return out

    height = 1.9 + 0.34 * len(rows)
    fig, ax = plt.subplots(figsize=(13.5, height))
    head = 1.5 / height
    fig.subplots_adjust(left=0.20, right=0.965, top=1 - head, bottom=0.52 / height)
    fig.suptitle("Was this model worth tuning?", x=0.035, y=1 - 0.34 / height,
                 ha="left", fontsize=19, fontweight="bold", color="#212529")
    fig.text(0.035, 1 - 0.70 / height,
             "Each dot is one of the best trials the search kept, scored on its inner "
             "folds. Tight cluster = the knobs do nothing on this data.",
             ha="left", fontsize=11, color="#495057")
    fig.text(0.035, 1 - 0.97 / height,
             "Where the grey diamond (the parameters the file defines) sits in that "
             "spread says whether the original guess was already a good one.",
             ha="left", fontsize=10, color="#868E96", style="italic")

    for i, (n, h, vals, init, best) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = min(min(vals), init), max(max(vals), init)
        ax.plot([lo, hi], [y, y], color=GRID_C, lw=2.0, zorder=1, solid_capstyle="round")
        ax.scatter(vals, [y] * len(vals), s=32, color=COLOR.get(n, "#495057"),
                   alpha=0.55, edgecolor="white", lw=0.7, zorder=3)
        ax.scatter([init], [y], marker="D", s=58, color=BEFORE_C, edgecolor="#495057",
                   lw=1.1, zorder=4)
        ax.scatter([best], [y], marker="|", s=260, color=BETTER_C, lw=2.4, zorder=5)
        span = (init - best) / init * 100 if init else 0.0
        ax.text(hi + (hi - lo) * 0.04 + 1e-4, y, f"{span:+.1f}%", va="center",
                fontsize=8.5, color=BETTER_C if span > 0 else "#868E96",
                fontweight="bold")

    ax.set_yticks(range(len(rows)),
                  [f"{n}  ·  {h}d" for n, h, *_ in reversed(rows)], fontsize=9)
    ax.set_xlabel("inner-fold WAPE (the number the search selected on)")
    ax.set_title("grey diamond = the file's parameters   ·   green bar = what the "
                 "search picked   ·   dots = the other trials it kept", **TITLE_KW)
    ax.margins(x=0.10)
    _clean(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


# ======================================================================================
# 3c · Why WAPE cannot be the objective on its own
# ======================================================================================

def k_min_wape(y: np.ndarray, p: np.ndarray) -> float:
    """The scalar multiplier on a forecast that minimises its WAPE."""
    grid = np.linspace(0.5, 1.6, 111)
    k0 = grid[int(np.argmin([np.sum(np.abs(y - k * p)) for k in grid]))]
    fine = np.linspace(k0 - 0.01, k0 + 0.01, 41)
    return float(fine[int(np.argmin([np.sum(np.abs(y - k * p)) for k in fine]))])


def fig_objective(out: Path) -> Path:
    """Take each model's forecast and ask two questions of the same numbers.

    1. What multiplier would minimise its WAPE?
    2. What multiplier would make it unbiased?

    If WAPE were a neutral yardstick those would coincide. They generally do not, and the
    gap has a predictable direction: WAPE is a sum of absolute errors, minimised per row by the
    conditional MEDIAN, and on a right-skewed target the median sits below the mean. So
    the WAPE-optimal forecast is systematically short of the honest one -- which is why a
    search that optimises WAPE alone walks every model downward.

    The second panel is the one that makes this actionable rather than merely true: the
    WAPE cost of moving from the first multiplier to the second. It is typically small:
    the surface is nearly flat with respect to level, so being unbiased is close to free
    -- you only have to ask for it, which is what tune.select() does.
    """
    rows: list[tuple[str, int, float, float, float]] = []
    for n in MODELS:
        d = load_preds(INITIAL_RESULTS, n)
        if d is None:
            continue
        for h in HORIZONS:
            sub = d.filter(pl.col("h") == h)
            if not len(sub):
                continue
            y = sub.get_column("actual").to_numpy().astype(float)
            p = sub.get_column("predicted").to_numpy().astype(float)
            if p.sum() <= 0 or np.abs(y).sum() <= 0:
                continue
            kw, kb = k_min_wape(y, p), float(y.sum() / p.sum())
            w = lambda k: float(np.sum(np.abs(y - k * p)) / np.sum(np.abs(y)))
            rows.append((n, h, kw, kb, w(kb) - w(kw)))
    if not rows:
        return out

    height = 2.0 + 0.30 * len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, height),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    head = 1.6 / height
    fig.subplots_adjust(left=0.175, right=0.975, top=1 - head, bottom=0.60 / height,
                        wspace=0.30)
    fig.suptitle("Why WAPE cannot be the objective on its own", x=0.03,
                 y=1 - 0.36 / height, ha="left", fontsize=19, fontweight="bold",
                 color="#212529")
    fig.text(0.03, 1 - 0.74 / height,
             "Scale each model's forecast by a constant. One multiplier minimises WAPE; "
             "a different one makes the forecast unbiased.",
             ha="left", fontsize=11, color="#495057")
    fig.text(0.03, 1 - 1.02 / height,
             "On a right-skewed target WAPE's is typically the lower of the two — so "
             "optimising WAPE alone selects for under-forecasting.",
             ha="left", fontsize=10, color="#868E96", style="italic")

    ax = axes[0]
    y_pos = np.arange(len(rows))[::-1]
    for (n, h, kw, kb, _), yi in zip(rows, y_pos):
        ax.plot([kw, kb], [yi, yi], color=GRID_C, lw=2.2, zorder=1,
                solid_capstyle="round")
        ax.scatter([kw], [yi], s=54, color=WORSE_C, zorder=3)
        ax.scatter([kb], [yi], s=54, color=BETTER_C, marker="D", zorder=3)
    ax.axvline(1.0, color="#868E96", lw=0.9, ls=":", zorder=0)
    ax.set_yticks(y_pos, [f"{n} · {h}d" for n, h, *_ in rows], fontsize=8.5)
    ax.set_xlabel("multiplier applied to the forecast")
    ax.set_title("red = what WAPE wants   ·   green = what being unbiased wants",
                 **TITLE_KW)
    gaps = [r[2] - r[3] for r in rows]
    ax.text(0.98, 0.015,
            f"WAPE's multiplier is the lower one in {sum(g < 0 for g in gaps)} "
            f"of {len(gaps)} cases",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            color=WORSE_C, style="italic")
    _clean(ax)

    ax = axes[1]
    costs = [r[4] for r in rows]
    ax.barh(y_pos, costs, height=0.62, color=AFTER_C, zorder=3)
    ax.axvline(0, color="#212529", lw=1.0)
    ax.set_yticks(y_pos, [""] * len(rows))
    ax.set_xlabel("WAPE given up to be unbiased")
    ax.set_title("and what that costs", **TITLE_KW)
    ax.text(0.97, 0.015,
            f"mean {np.mean(costs):+.4f}\ncompare with the split-to-split spread",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            color="#495057", style="italic")
    _clean(ax)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


# ======================================================================================
# 4 · What the WAPE change was bought with
# ======================================================================================

def fig_cost(before: dict, after: dict, out: Path) -> Path:
    """WAPE, bias and shortfall side by side, before and after.

    WAPE treats a unit over and a unit under as the same mistake. Inventory does not: an
    over-forecast is cash on a shelf, an under-forecast is a lost sale and a customer who
    buys elsewhere. If tuning cut WAPE by shifting the whole forecast down, this is where
    it shows up.
    """
    names = [n for n in MODELS if n in before and n in after]
    metrics = [("wape", "WAPE", "lower is better"),
               ("bias", "bias", "0 is honest; − is under-forecasting"),
               ("shortfall", "shortfall", "demand you could not have supplied")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.79, bottom=0.28, wspace=0.22)
    fig.suptitle("What the accuracy change was bought with", x=0.06, y=0.955,
                 ha="left", fontsize=19, fontweight="bold", color="#212529")
    fig.text(0.06, 0.893,
             "Averaged over the three horizons. A WAPE gain that comes with a more "
             "negative bias was paid for in stockouts, not earned.",
             ha="left", fontsize=11, color="#495057")

    for ax, (key, title, sub) in zip(axes, metrics):
        x = np.arange(len(names))
        b = [float(np.nanmean([cell(before[n], h, key) for h in HORIZONS]))
             for n in names]
        a = [float(np.nanmean([cell(after[n], h, key) for h in HORIZONS]))
             for n in names]
        ax.bar(x - 0.20, b, width=0.38, color=BEFORE_C, label="initial")
        ax.bar(x + 0.20, a, width=0.38, color=AFTER_C, label="tuned")
        if key in ("bias",):
            ax.axhline(0, color="#212529", lw=1.0)
        ax.set_xticks(x, names, rotation=38, ha="right", fontsize=9)
        ax.set_title(f"{title} — {sub}", **TITLE_KW)
        _clean(ax)
    # One legend for the whole figure, in the header band, so it can never land on a bar.
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=10.5,
               loc="upper right", bbox_to_anchor=(0.99, 0.945), ncols=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


# ======================================================================================
# 5 · Where in the catalogue the change landed
# ======================================================================================

HORIZON_FOR_BANDS = 90


def fig_by_volume(out: Path, horizon: int = HORIZON_FOR_BANDS) -> Path:
    """The same before/after, cut by how much the SKU actually sells.

    A catalogue-wide WAPE is a volume-weighted average, so a change of 0.01 can be one
    big seller moving or the entire long tail moving, and those are different decisions.
    """
    rows: list[dict[str, Any]] = []
    for n in MODELS:
        for label, res in (("initial", INITIAL_RESULTS), ("tuned", TUNED_RESULTS)):
            d = load_preds(res, n)
            if d is None:
                continue
            s = d.filter(pl.col("h") == horizon)
            # Units per day = the MEAN of this SKU's cumulative h-day totals, divided by
            # h. Not the sum: every SKU contributes one row per weekly origin in the test
            # window, so summing would scale the rate by the number of origins and push
            # the whole catalogue into the top band. This is the identical expression
            # dashboard.by_volume() uses, so "2-10/day" means the same size of SKU in
            # both charts.
            vol = s.group_by("sku").agg(v=pl.col("actual").mean() / horizon)
            s = s.join(vol, on="sku")
            s = s.with_columns(pl.Series("band", band_of(s.get_column("v").to_numpy())))
            agg = (s.group_by("band")
                   .agg(err=(pl.col("predicted") - pl.col("actual")).abs().sum(),
                        vol=pl.col("actual").abs().sum(), n=pl.col("sku").n_unique()))
            for r in agg.iter_rows(named=True):
                rows.append({"model": n, "side": label, "band": r["band"],
                             "wape": r["err"] / max(r["vol"], 1e-9), "n": r["n"]})
    if not rows:
        return out
    t = pl.DataFrame(rows)
    bands = [b[2] for b in BANDS if b[2] in set(t.get_column("band").to_list())]
    names = [n for n in MODELS if n in set(t.get_column("model").to_list())]

    fig, axes = plt.subplots(1, len(bands), figsize=(3.55 * len(bands), 6.4))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.78, bottom=0.30, wspace=0.24)
    fig.suptitle(f"Where in the catalogue the tuning landed "
                 f"({HORIZON_LABEL[horizon]})", x=0.055, y=0.955, ha="left",
                 fontsize=19, fontweight="bold", color="#212529")
    fig.text(0.055, 0.893,
             "The headline number is volume-weighted, so it is dominated by the big "
             "sellers. These are the same models scored inside one size band at a time.",
             ha="left", fontsize=11, color="#495057")

    for ax, band in zip(axes, bands):
        sub = t.filter(pl.col("band") == band)
        x = np.arange(len(names))
        b = [float(sub.filter((pl.col("model") == n) & (pl.col("side") == "initial"))
                   .get_column("wape").to_list()[0]) if len(sub.filter(
                       (pl.col("model") == n) & (pl.col("side") == "initial"))) else np.nan
             for n in names]
        a = [float(sub.filter((pl.col("model") == n) & (pl.col("side") == "tuned"))
                   .get_column("wape").to_list()[0]) if len(sub.filter(
                       (pl.col("model") == n) & (pl.col("side") == "tuned"))) else np.nan
             for n in names]
        ax.bar(x - 0.20, b, width=0.38, color=BEFORE_C, label="initial")
        ax.bar(x + 0.20, a, width=0.38, color=AFTER_C, label="tuned")
        n_sku = int(sub.get_column("n").max() or 0)  # same for both sides by construction
        ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8.5)
        ax.set_title(f"{band}  ({n_sku} SKUs)", **TITLE_KW)
        ax.set_ylabel("WAPE")
        _clean(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=10.5,
               loc="upper right", bbox_to_anchor=(0.99, 0.945), ncols=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


# ======================================================================================
# 6 · SKU by SKU
# ======================================================================================

def fig_per_sku(out: Path) -> Path:
    """For each model, the share of SKUs the tuned version forecasts better.

    An aggregate WAPE can improve because one enormous SKU improved. This is the same
    comparison run once per SKU and counted, which says whether the change is broad or
    narrow — the same distinction dashboard.py's _head_to_head.png draws between models.
    """
    rows = []
    for n in MODELS:
        bi, ti = load_preds(INITIAL_RESULTS, n), load_preds(TUNED_RESULTS, n)
        if bi is None or ti is None:
            continue
        for h in HORIZONS:
            b = per_sku_wape(bi.filter(pl.col("h") == h)).rename({"wape": "w_before"})
            a = per_sku_wape(ti.filter(pl.col("h") == h)).rename({"wape": "w_after"})
            j = b.select("sku", "w_before", "vol").join(a.select("sku", "w_after"),
                                                        on="sku", how="inner")
            if not len(j):
                continue
            wb = j.get_column("w_before").to_numpy()
            wa = j.get_column("w_after").to_numpy()
            rows.append({"model": n, "h": h, "n": len(j),
                         "win": float(np.mean(wa < wb)),
                         "median_delta": float(np.median(wa - wb))})
    if not rows:
        return out
    t = pl.DataFrame(rows)
    names = [n for n in MODELS if n in set(t.get_column("model").to_list())]

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.28, wspace=0.22)
    fig.suptitle("SKU by SKU, not just in aggregate", x=0.07, y=0.955, ha="left",
                 fontsize=19, fontweight="bold", color="#212529")
    fig.text(0.07, 0.893,
             "A headline WAPE is volume-weighted and can be moved by one big seller. "
             "These count every SKU equally.",
             ha="left", fontsize=11, color="#495057")

    ax = axes[0]
    x = np.arange(len(names))
    w = 0.26
    for i, h in enumerate(HORIZONS):
        vals = [float(t.filter((pl.col("model") == n) & (pl.col("h") == h))
                      .get_column("win").to_list()[0])
                if len(t.filter((pl.col("model") == n) & (pl.col("h") == h))) else np.nan
                for n in names]
        ax.bar(x + (i - 1) * w, vals, width=w * 0.92,
               color=plt.cm.Blues(0.42 + 0.22 * i), label=f"{h}d")
    ax.axhline(0.5, color="#212529", lw=1.2, ls="--")
    # Above the bars, not on the line, so it cannot land on top of a 50%-ish bar --
    # which is exactly the case this label is there to explain.
    ax.text(-0.45, 0.06,
            "above the dashed line = the change helped most SKUs;  below it = the "
            "aggregate gain came from a few big ones",
            ha="left", va="center", fontsize=9, color="#495057", style="italic")
    ax.set_xticks(x, names, rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("share of SKUs the tuned version forecasts better")
    ax.set_ylim(0, 1)
    ax.set_title("How broad the change is", **TITLE_KW)
    ax.text(0.0, 1.012, "the dashed line is half the catalogue",
            transform=ax.transAxes, fontsize=9, color="#868E96", style="italic")
    ax.legend(frameon=False, fontsize=9, ncols=3, loc="upper right")
    _clean(ax)

    ax = axes[1]
    for i, h in enumerate(HORIZONS):
        vals = [float(t.filter((pl.col("model") == n) & (pl.col("h") == h))
                      .get_column("median_delta").to_list()[0])
                if len(t.filter((pl.col("model") == n) & (pl.col("h") == h))) else np.nan
                for n in names]
        ax.bar(x + (i - 1) * w, vals, width=w * 0.92,
               color=plt.cm.Blues(0.42 + 0.22 * i), label=f"{h}d")
    ax.axhline(0, color="#212529", lw=1.1)
    ax.set_xticks(x, names, rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("median per-SKU change in WAPE")
    ax.set_title("How big the change is on a typical SKU", **TITLE_KW)
    ax.text(0.0, 1.012, "below zero = the median SKU improved",
            transform=ax.transAxes, fontsize=9, color="#868E96", style="italic")
    ax.legend(frameon=False, fontsize=9, ncols=3)
    _clean(ax)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


# ======================================================================================
# The written report
# ======================================================================================

def _param_diff(name: str, h: int) -> str:
    blob = tuned_blob(name)
    if not blob:
        return "—"
    e = blob.get("per_horizon", {}).get(str(h))
    if not e:
        return "—"
    best = e.get("best_params") or {}
    if not best:
        return "*kept the file's defaults*"
    return ", ".join(f"`{k}` = {v}" for k, v in sorted(best.items()))


def breadth(name: str) -> float:
    """Share of SKUs the tuned version forecasts better, averaged over the horizons.

    The headline WAPE is volume-weighted, so it can improve because one large SKU
    improved while most of the catalogue got worse. This counts every SKU once.
    """
    bi, ti = load_preds(INITIAL_RESULTS, name), load_preds(TUNED_RESULTS, name)
    if bi is None or ti is None:
        return float("nan")
    shares = []
    for h in HORIZONS:
        b = per_sku_wape(bi.filter(pl.col("h") == h)).rename({"wape": "wb"})
        a = per_sku_wape(ti.filter(pl.col("h") == h)).rename({"wape": "wa"})
        j = b.select("sku", "wb").join(a.select("sku", "wa"), on="sku", how="inner")
        if len(j):
            shares.append(float(np.mean(j.get_column("wa").to_numpy()
                                        < j.get_column("wb").to_numpy())))
    return float(np.mean(shares)) if shares else float("nan")


def mean_metric(res: dict, key: str) -> float:
    return float(np.nanmean([cell(res, h, key) for h in HORIZONS]))


def write_report(before: dict, after: dict, figs: list[tuple[str, str, Path]]) -> Path:
    names = sorted(set(before) & set(after),
                   key=lambda n: mean_wape(after[n]))
    L = [
        "# Tuning — before and after",
        "",
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} by `dev/compare_tuning.py`, from "
        f"`results/` (initial) and `results_tuned/` (tuned).*",
        "",
        "Both halves are the same code, the same dataset, the same rolling-origin "
        "splits, the same features and the same metrics. The only difference between "
        "them is the numbers handed to the estimator constructors, which is what makes "
        "any change below attributable to the tuning and to nothing else.",
        "",
        "The search itself never saw a test row — `dev/tune.py` cuts the dataset at the "
        "earliest leaderboard fold's `train_end` and searches inside what is left. So "
        "everything here is an out-of-sample measurement of a decision made out of "
        "sample, and a tuned model is genuinely allowed to come out worse.",
        "",
        "## The headline",
        "",
        "Mean over the three horizons. Read the last three columns with the first: a "
        "WAPE that fell while bias got more negative and shortfall rose was not earned, "
        "it was borrowed from your service level — and a WAPE that fell while fewer "
        "than half the SKUs improved was carried by a handful of big sellers.",
        "",
        "| model | WAPE before → after | change | verdict | horizons better | "
        "bias before → after | shortfall before → after | SKUs improved |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        b, a = mean_wape(before[n]), mean_wape(after[n])
        floor = float(np.mean([noise_floor(before[n], after[n], h) for h in HORIZONS]))
        label, _ = verdict_for(b, a, floor)
        bb, ba = mean_metric(before[n], "bias"), mean_metric(after[n], "bias")
        sb, sa = mean_metric(before[n], "shortfall"), mean_metric(after[n], "shortfall")
        w = breadth(n)
        flag = "" if not np.isfinite(w) else ("" if w >= 0.5 else " ⚠")
        better, nh, _ = consistency(before, after, n)
        L.append(f"| `{n}` | {b:.4f} → {a:.4f} | {(a - b) / b * 100:+.1f}% | {label} | "
                 f"{better}/{nh} | {bb:+.3f} → {ba:+.3f} | {sb:.3f} → {sa:.3f} | "
                 f"{w * 100:.0f}%{flag} |")
    sweeps = [f"`{n}` — {consistency(before, after, n)[2]}" for n in names
              if consistency(before, after, n)[0] in (0, len(HORIZONS))]
    L += [""]
    if sweeps:
        L += ["The verdict column judges each horizon on its own and is strict about "
              "it. **horizons better** is the evidence that test discards: three "
              "horizons of one model agreeing is 1-in-4 if tuning does nothing, so a "
              "clean sweep of three individually-inconclusive rows still means "
              "something. Sweeps in this run:",
              ""] + [f"- {x}" for x in sweeps] + [""]
    L += ["\"Inside the noise\" means the change is smaller than the model's own "
          "split-to-split spread — the amount its score moves when you shift the fold "
          "boundaries. It is not a result in either direction.",
          "",
          "⚠ marks a model whose headline WAPE moved one way while most of its SKUs "
          "moved the other. That is not a contradiction — WAPE is volume-weighted and "
          "the per-SKU count is not — but it means the improvement is narrow, and "
          "whether you want it depends on whether your cost is concentrated in the "
          "same SKUs the gain is.",
          ""]

    for title, caption, path in figs:
        L += [f"## {title}", "", caption, "",
              f"![{title}]({path.relative_to(LAB).as_posix()})", ""]

    L += ["## Horizon by horizon", ""]
    for h in HORIZONS:
        L += [f"### {HORIZON_LABEL[h].capitalize()}", "",
              "| model | before | after | change | noise floor | verdict | what the "
              "search chose |",
              "|---|---|---|---|---|---|---|"]
        for n in sorted(names, key=lambda m: cell(after[m], h)):
            b, a = cell(before[n], h), cell(after[n], h)
            floor = noise_floor(before[n], after[n], h)
            label, _ = verdict_for(b, a, floor)
            L.append(f"| `{n}` | {b:.4f} | {a:.4f} | {(a - b) / b * 100:+.1f}% | "
                     f"±{floor:.4f} | {label} | {_param_diff(n, h)} |")
        L += [""]

    # Flag the specific failure mode where a symmetric metric rewards a level shift,
    # using this run's own numbers rather than asserting it in the abstract.
    bought: list[str] = []
    drifted = 0
    for n in names:
        db = mean_metric(after[n], "bias") - mean_metric(before[n], "bias")
        ds = mean_metric(after[n], "shortfall") - mean_metric(before[n], "shortfall")
        dw = mean_wape(after[n]) - mean_wape(before[n])
        if db < 0 and ds > 0:
            drifted += 1
        if dw < 0 and db < -0.01 and ds > 0.01:
            bought.append(f"- `{n}`: WAPE {dw:+.4f}, but bias {db:+.3f} and shortfall "
                          f"{ds:+.3f} — the forecast moved down, not closer.")
    # One model drifting down is a coincidence. Most of them drifting down is the
    # objective doing it, and that is a different and much more important statement,
    # so say it first and say it in those terms.
    systematic = names and drifted >= max(2, (len(names) + 1) // 2)
    if bought or systematic:
        L += ["## When a WAPE win is not a win", ""]
    if systematic:
        L += [f"**{drifted} of the {len(names)} tuned models came out with a more "
              f"negative bias and a higher shortfall than they went in with.** That is "
              f"not {drifted} coincidences. A search that minimises WAPE on a "
              f"right-skewed, partly-censored panel will find that forecasting a little "
              f"low reduces symmetric error, and it will take that trade in whatever "
              f"knob happens to be available — so the tuned catalogue as a whole "
              f"under-forecasts more than the untuned one did.",
              "",
              "Before adopting any of these parameters, re-check the safety stock that "
              "was sized against the old bias: the models are now wrong in a more "
              "consistent direction, which is exactly the kind of error a buffer sized "
              "on the old numbers will not absorb.",
              ""]
    if bought:
        L += ["WAPE counts a unit over and a unit under as the same mistake. Inventory "
              "does not: over-forecasting is cash on a shelf, under-forecasting is a "
              "lost sale. So any knob that can shift the whole forecast down — "
              "`moving_average`'s `scale`, Croston's `alpha` through its "
              "`(1 - alpha/2)` correction, a Tweedie variance power, a quantile "
              "`center` below 0.5 — has a way to buy WAPE that costs service level, and "
              "a search that optimises WAPE will take it every time.",
              "",
              "In this run that happened to:", ""] + bought + [
              "",
              "These are not bugs in the search; the search did what it was asked. They "
              "are a reason to decide, per model, whether you want the tuned parameters "
              "at all — and if you do, to re-check the safety stock that was sized "
              "against the old bias. The alternative is to tune against an asymmetric "
              "objective instead, which is a different and larger piece of work: it "
              "means putting a price on a stockout.",
              ""]

    L += ["## How to read a negative result", "",
          "A model that did not improve has told you one of three things, and they have "
          "different consequences:",
          "",
          "1. **Its default parameters were already at an optimum this data can resolve.** "
          "The most common outcome for the boosted models, which are famously flat near "
          "their optimum on tabular data — you can move `num_leaves` by 4x and move WAPE "
          "by less than the fold spread. Nothing to do.",
          "2. **The search window was not like the test window.** The inner folds sit "
          "three to nine months earlier than the test folds. If demand changed shape in "
          "between, the search optimised for a catalogue that no longer exists. Visible "
          "as a point in the lower-right of the transfer chart.",
          "3. **The search was too small.** 10-16 random trials over 8 knobs samples a "
          "space thinly. Raise it with `--budget 3` and re-run if a model matters "
          "enough to spend the hours on.",
          "",
          "## Reproducing either half", "",
          "```bash",
          "uv run python dev/tune.py --all                       # the search",
          "uv run python dev/compare_models.py --run-all         # the initial board",
          "uv run python dev/compare_models.py --run-all --tuned # the tuned board",
          "uv run python dev/dashboard.py                        # initial_figures/",
          "uv run python dev/dashboard.py --tuned                # tuned_figures/",
          "uv run python dev/compare_tuning.py                   # this file",
          "```",
          ""]
    REPORT.write_text("\n".join(L))
    return REPORT


# ======================================================================================

def main() -> None:
    before, after = load_side(INITIAL_RESULTS), load_side(TUNED_RESULTS)
    if not before:
        raise SystemExit("no initial results. Run: uv run python dev/compare_models.py "
                         "--run-all")
    if not after:
        raise SystemExit("no tuned results yet. Run:\n"
                         "  uv run python dev/tune.py --all\n"
                         "  uv run python dev/compare_models.py --run-all --tuned")
    missing = sorted(set(before) - set(after))
    if missing:
        print(f"  ! not tuned yet, left out of the comparison: {', '.join(missing)}")

    OUT.mkdir(parents=True, exist_ok=True)
    figs = [
        ("Every model, both versions",
         "The board twice over. The percentage above each pair is the change; the "
         "whiskers are the spread across the splits, and where they overlap the "
         "two versions are not distinguishable.",
         fig_board(before, after, OUT / "_vs_initial_board.png")),
        ("The change, against the noise floor",
         "The same numbers as a signed difference, with the amount the score moves for "
         "free drawn behind each bar. Only bars that escape the amber band are results.",
         fig_delta(before, after, OUT / "_vs_initial_delta.png")),
        ("Did the gain transfer?",
         "The search measured its winners on folds that end before the test period "
         "begins. This is what it thought it had won against what it actually won.",
         fig_transfer(before, after, OUT / "_vs_initial_transfer.png")),
        ("Was this model worth tuning?",
         "Each dot is one of the best trials the search recorded, scored on its inner "
         "folds. A tight cluster means every setting in the space scored the same and "
         "the winner is measurement noise — tuning that model harder will buy nothing.",
         fig_search(OUT / "_vs_initial_search.png")),
        ("Why WAPE cannot be the objective on its own",
         "Two multipliers per model: the one that minimises WAPE, and the one that makes "
         "the forecast unbiased. On a right-skewed target WAPE's is typically the lower — "
         "and if the gap costs little in WAPE, the constraint is worth imposing.",
         fig_objective(OUT / "_vs_initial_objective.png")),
        ("What the change was bought with",
         "WAPE next to bias and shortfall. A WAPE improvement that arrives with a more "
         "negative bias was paid for in stockouts.",
         fig_cost(before, after, OUT / "_vs_initial_cost.png")),
        ("Where in the catalogue it landed",
         "The headline is volume-weighted. This is the same comparison inside one size "
         "band at a time, where the big sellers cannot carry it.",
         fig_by_volume(OUT / "_vs_initial_by_volume.png")),
        ("SKU by SKU",
         "How broad the change is, and how big it is on a typical SKU rather than a "
         "volume-weighted one.",
         fig_per_sku(OUT / "_vs_initial_per_sku.png")),
    ]
    for title, _, path in figs:
        print(f"  figure  {path.relative_to(LAB.parent)}")
    path = write_report(before, after, figs)
    print(f"\n  report  {path.relative_to(LAB.parent)}")

    print(f"\n{'=' * 78}\nBEFORE -> AFTER  (mean WAPE over 1/3/6 months)\n{'=' * 78}")
    for n in sorted(set(before) & set(after), key=lambda m: mean_wape(after[m])):
        b, a = mean_wape(before[n]), mean_wape(after[n])
        floor = float(np.mean([noise_floor(before[n], after[n], h) for h in HORIZONS]))
        label, _ = verdict_for(b, a, floor)
        print(f"  {n:<22} {b:.4f} -> {a:.4f}  {(a - b) / b * 100:>+6.1f}%   {label}")
    print("=" * 78)


if __name__ == "__main__":
    main()
