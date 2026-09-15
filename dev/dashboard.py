"""THE VISUAL LAYER — one card per model, four cross-model charts, one HTML page.

    uv run python dev/dashboard.py              # everything, from cached preds
    uv run python dev/dashboard.py --only xgboost_gbdt
    uv run python dev/dashboard.py --no-html    # just the PNGs

This file never fits a model. It reads what the model files already wrote --

    results/<name>.json          the scores, fold by fold
    results/_preds/<name>.parquet  every test row: sku, date, horizon, fold, actual, predicted

-- so it runs in seconds and can be re-styled as often as you like without paying for a
refit. Run a model file to change the numbers; run this file to change the pictures.

-----------------------------------------------------------------------------------
WHAT EACH CARD ANSWERS
-----------------------------------------------------------------------------------
Six panels, in the order you would actually interrogate a forecast:

  1  How accurate is it, and at which horizon?   bars vs the baseline, whiskers = the
                                                 spread across the splits
  2  Where does it sit on individual rows?       predicted vs actual, log-log. The
                                                 diagonal is perfect; the cloud's shape
                                                 tells you where it breaks
  3  Is it calibrated?                           bin the rows by forecast size, plot
                                                 what actually happened. Off the diagonal
                                                 = systematic over/under at that size
  4  Who does it fail?                           bias by SKU volume band. A model can
                                                 look unbiased catalogue-wide purely
                                                 because two bands cancel
  5  Is it stable?                               the same score, fold by fold. A model
                                                 whose folds disagree has not been
                                                 measured, it has been sampled
  6  Does it beat the alternative per SKU?       not on the total, on each SKU. A model
                                                 can win the aggregate and lose most SKUs

Panels 4 and 6 are the ones that change decisions. The aggregate WAPE on the leaderboard
is a single number over a catalogue that is not homogeneous, and the whole point of
keeping seven models is that different ones own different parts of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
from common import (  # noqa: E402
    GRID_C, HORIZON_LABEL, HORIZONS, LAB, figures_dir, results_dir, set_mode,
)
from common import HORIZONS as _H, _fig_for_horizon  # noqa: E402,F401
from compare_models import BASELINE, MODEL_FILES, board_key, mean_wape  # noqa: E402

#: One page per mode, so the tuned dashboard never overwrites the initial one and the
#: two can be opened side by side in a browser.
HTMLS = {"initial": LAB / "dashboard.html", "tuned": LAB / "dashboard_tuned.html"}

#: Board order, and the one colour each model keeps in every chart in this folder.
MODELS = [f[:-3] for f in MODEL_FILES]
COLOR = {
    "moving_average":    "#F76707",  # orange -- the baseline, everywhere
    "croston":           "#FAB005",
    "random_forest":     "#2F9E44",
    "xgboost_gbdt":      "#1971C2",
    "lightgbm_additive": "#6741D9",
    "lgbm_quantile":     "#0C8599",
    "two_stage":         "#C2255C",
}
#: One line on why this model is on the shortlist at all -- the capability it owns,
#: which is not the same question as where it ranks.
KEEPS = {
    "moving_average":    "The floor. Free, explainable, and the number every other "
                         "model has to beat to justify its existence.",
    "croston":           "Built for intermittent demand -- the thin and newly-launched "
                         "SKUs the tree models handle worst.",
    "random_forest":     "Strong accuracy with almost nothing to tune, and it aims a "
                         "little higher than the boosted trees.",
    "xgboost_gbdt":      "The accurate boosted model, with a different failure mode "
                         "from the forest.",
    "lightgbm_additive": "The only model that returns a monthly profile instead of one "
                         "total -- it says WHEN the units land.",
    "lgbm_quantile":     "The only model that returns a range (P10/P50/P90). Safety "
                         "stock is a question about the tail, not the mean.",
    "two_stage":         "P(sale) x E[units | sale]: the tree-based answer to "
                         "intermittency, kept for the same reason as Croston.",
}

TITLE_KW = dict(fontsize=11, fontweight="bold", loc="left", color="#212529")


# ======================================================================================
# Loading
# ======================================================================================

def load_board() -> list[dict]:
    """Every result JSON that belongs to a shortlisted model, best mean WAPE first."""
    out = []
    for m in MODELS:
        p = results_dir() / f"{m}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
        else:
            print(f"  ! {m}: no {p.parent.name}/{m}.json -- run it first, skipping")
    return sorted(out, key=board_key(out)) if out else out


def load_preds(name: str) -> pl.DataFrame | None:
    p = results_dir() / "_preds" / f"{name}.parquet"
    if not p.exists():
        print(f"  ! {name}: no cached predictions, panels 2-6 will be empty")
        return None
    return pl.read_parquet(p)


def wape_of(d: pl.DataFrame) -> float:
    a = d.get_column("actual").to_numpy()
    p = d.get_column("predicted").to_numpy()
    return float(np.sum(np.abs(a - p)) / max(np.sum(np.abs(a)), 1e-9))


def per_sku_wape(d: pl.DataFrame) -> pl.DataFrame:
    """WAPE for each SKU separately. SKUs with no demand at all in the window are
    dropped rather than scored -- their denominator is zero and any forecast is
    infinitely wrong, which would dominate every distribution they appear in."""
    return (d.group_by("sku")
            .agg(err=(pl.col("actual") - pl.col("predicted")).abs().sum(),
                 vol=pl.col("actual").sum())
            .filter(pl.col("vol") > 0)
            .with_columns(wape=pl.col("err") / pl.col("vol")))


#: Volume bands, in units per day. The cuts are the ones model_lab/segment_report.py
#: uses for `volume_band`, so "0.5-2/day" means the same size of SKU in both places --
#: but measured differently: there it is the trailing 91-day average feature, here it is
#: the demand that actually turned up in the test window. A diagnostic, not a feature,
#: so reading the test period is legitimate; it would not be if a model saw it.
BANDS = [(0.0, 0.5, "<0.5/day"), (0.5, 2.0, "0.5-2/day"), (2.0, 10.0, "2-10/day"),
         (10.0, 50.0, "10-50/day"), (50.0, np.inf, ">=50/day")]


def band_of(rate: np.ndarray) -> np.ndarray:
    """Label each SKU by its realised units/day. `rate` must already be per day."""
    out = np.empty(len(rate), dtype=object)
    for lo, hi, label in BANDS:
        out[(rate >= lo) & (rate < hi)] = label
    return out


# ======================================================================================
# The card
# ======================================================================================

def card(res: dict, board: list[dict], out: Path) -> Path:
    name = res["model"]
    d = load_preds(name)
    # Every card is read against a reference. Normally that is the moving-average
    # baseline; the baseline's own card is read against the best model on the board
    # instead. `board[0]` is not safe for that -- when the baseline IS the best model on
    # the board (a two-model board, or a run where nothing beat it) it would compare the
    # model against itself and panels 1 and 6 would silently show a flat zero.
    ref_name = BASELINE if name != BASELINE else next(
        (r["model"] for r in board if r["model"] != name), name)
    ref = next((r for r in board if r["model"] == ref_name), None)
    if ref_name == name:
        ref = None  # a one-model board: no reference exists, draw the card without one
    rank = [r["model"] for r in board].index(name) + 1

    fig = plt.figure(figsize=(17.5, 10.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26,
                          left=0.055, right=0.985, top=0.855, bottom=0.075)

    fig.suptitle(f"{name}", x=0.055, y=0.965, ha="left",
                 fontsize=21, fontweight="bold", color=COLOR.get(name, "#212529"))
    fig.text(0.055, 0.925,
             f"#{rank} of {len(board)} on the board   ·   mean WAPE "
             f"{mean_wape(res):.3f}   ·   {res['family']}",
             ha="left", fontsize=12, color="#495057")
    fig.text(0.055, 0.888, KEEPS.get(name, ""), ha="left", fontsize=10.5,
             color="#868E96", style="italic")

    # ---------------------------------------------------------------- 1  by horizon
    ax = fig.add_subplot(gs[0, 0])
    hs = [h for h in HORIZONS if str(h) in res["horizons"]]
    x = np.arange(len(hs))
    vals = [res["horizons"][str(h)]["wape"] for h in hs]
    errs = [res["horizons"][str(h)]["wape_sd"] for h in hs]
    rv = [ref["horizons"][str(h)]["wape"] for h in hs] if ref else [0.0] * len(hs)
    if ref:
        ax.bar(x, rv, width=0.72, color="#DEE2E6")
    ax.bar(x, vals, width=0.42, color=COLOR.get(name, "#4C6EF5"), yerr=errs,
           error_kw={"ecolor": "#343A40", "capsize": 4, "lw": 1.2})
    # Above BOTH bars, so the number never lands on the reference bar behind it.
    for xi, v, e, r in zip(x, vals, errs, rv):
        ax.text(xi, max(v + e, r) + 0.012, f"{v:.3f}", ha="center", fontsize=9.5,
                fontweight="bold")
    ax.set_ylim(0, max(max(v + e for v, e in zip(vals, errs)), max(rv)) * 1.28)
    ax.set_xticks(x, [f"{h}d\n{HORIZON_LABEL[h].split(' ahead')[0]}" for h in hs])
    ax.set_ylabel("WAPE — lower is better")
    ax.set_title("1 · How accurate, at which horizon", **TITLE_KW)
    if ref:
        ax.text(0.02, 0.96, f"grey = {ref_name}, for reference", transform=ax.transAxes,
                va="top", fontsize=8.5, color="#868E96")
    _clean(ax)

    # ------------------------------------------------------- 2  predicted vs actual
    ax = fig.add_subplot(gs[0, 1])
    if d is not None:
        s = d.filter(pl.col("h") == 90)
        a = np.clip(s.get_column("actual").to_numpy(), 0.5, None)
        p = np.clip(s.get_column("predicted").to_numpy(), 0.5, None)
        ax.hexbin(a, p, xscale="log", yscale="log", gridsize=42, mincnt=1,
                  cmap="Blues", bins="log", linewidths=0)
        lim = [0.5, max(a.max(), p.max()) * 1.1]
        ax.plot(lim, lim, color="#212529", lw=1.1, ls="--")
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_xlabel("actual units, next 90 days (log)")
        ax.set_ylabel("forecast (log)")
        ax.text(0.03, 0.95, "above the line = over-forecast\nbelow = under-forecast",
                transform=ax.transAxes, va="top", fontsize=8.5, color="#495057")
    ax.set_title("2 · Every forecast, against what happened", **TITLE_KW)
    _clean(ax, grid=False)

    # ------------------------------------------------------------- 3  calibration
    ax = fig.add_subplot(gs[0, 2])
    if d is not None:
        s = d.filter(pl.col("h") == 90)
        q = (s.with_columns(bin=pl.col("predicted").qcut(10, labels=[str(i) for i in range(10)],
                                                         allow_duplicates=True))
             .group_by("bin")
             .agg(pred=pl.col("predicted").mean(), act=pl.col("actual").mean(),
                  n=pl.len())
             .sort("pred"))
        pm = q.get_column("pred").to_numpy()
        am = q.get_column("act").to_numpy()
        # Ratio, not a 45-degree line. On a log-log diagonal every model looks perfect,
        # because the spread between the smallest and largest SKU is four orders of
        # magnitude and swamps a 30% miss. actual/forecast puts the miss on the axis.
        ratio = am / np.clip(pm, 1e-9, None)
        cols = ["#C92A2A" if r > 1 else "#1864AB" for r in ratio]
        ax.bar(np.arange(len(ratio)), ratio - 1, bottom=1, color=cols, width=0.72)
        ax.axhline(1, color="#212529", lw=1.2)
        ax.set_xticks(np.arange(len(ratio)),
                      [f"{v:,.0f}" for v in pm], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("forecast decile — mean units forecast in that decile")
        ax.set_ylabel("actual ÷ forecast")
        top = max(0.45, float(np.abs(ratio - 1).max()) * 1.35)
        ax.set_ylim(max(0.0, 1 - top), 1 + top)
        ax.text(0.03, 0.95, "above 1 = forecast too low for that size\nbelow 1 = too high",
                transform=ax.transAxes, va="top", fontsize=8.5, color="#495057")
    ax.set_title("3 · Calibration — is a forecast of size X honest?", **TITLE_KW)
    _clean(ax)

    # --------------------------------------------------------- 4  bias by volume band
    ax = fig.add_subplot(gs[1, 0])
    if d is not None:
        s = d.filter(pl.col("h") == 90)
        vol = s.group_by("sku").agg(v=pl.col("actual").mean() / 90)
        s = s.join(vol, on="sku")
        s = s.with_columns(pl.Series("band", band_of(s.get_column("v").to_numpy())))
        g = (s.group_by("band")
             .agg(bias=(pl.col("predicted") - pl.col("actual")).sum()
                  / pl.col("actual").abs().sum(),
                  wape=(pl.col("predicted") - pl.col("actual")).abs().sum()
                  / pl.col("actual").abs().sum(),
                  skus=pl.col("sku").n_unique()))
        order = [b[2] for b in BANDS]
        g = g.with_columns(pl.col("band").cast(pl.Enum(order))).sort("band")
        labels = g.get_column("band").to_list()
        bias = g.get_column("bias").to_numpy()
        cols = ["#C92A2A" if b < 0 else "#1864AB" for b in bias]
        ax.barh(np.arange(len(labels)), bias, color=cols, height=0.6)
        ax.axvline(0, color="#212529", lw=1)
        pad = max(np.abs(bias).max(), 0.05) * 2.6
        # Each row holds exactly one bar, so the other side of zero is always free:
        # put the caption there and it can never sit on top of the bar or the ticks.
        for i, (b, w, n) in enumerate(zip(bias, g.get_column("wape").to_numpy(),
                                          g.get_column("skus").to_list())):
            ax.text(pad * (0.04 if b < 0 else -0.04), i,
                    f"{b:+.2f}   (WAPE {w:.2f}, {n} SKUs)",
                    va="center", ha="left" if b < 0 else "right", fontsize=8.5,
                    color="#495057")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xlabel("bias   —   red = under-forecast (stockouts),  "
                      "blue = over (dead stock)")
        ax.set_xlim(-pad, pad)
    ax.set_title("4 · Who it gets wrong, and in which direction", **TITLE_KW)
    _clean(ax)

    # ------------------------------------------------------------- 5  fold stability
    ax = fig.add_subplot(gs[1, 1])
    for h in hs:
        folds = res["horizons"][str(h)]["folds"]
        xs = [f["fold"] for f in folds]
        ys = [f["wape"] for f in folds]
        ax.plot(xs, ys, "-o", lw=2, ms=6, label=f"{h}d")
        ax.annotate(f"{h}d", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(7, 0), fontsize=9, va="center")
    f0 = res["horizons"][str(hs[0])]["folds"]
    ax.set_xticks([f["fold"] for f in f0],
                  [f"split {f['fold']}\n{f['test_start'][:7]}" for f in f0])
    ax.set_ylabel("WAPE on that split")
    ax.set_title("5 · Does the score hold up across the backtest", **TITLE_KW)
    ax.text(0.02, 0.04, "lines that cross or fan out mean the ranking\n"
                        "on the board is inside the noise",
            transform=ax.transAxes, fontsize=8.5, color="#495057")
    _clean(ax)

    # ----------------------------------------------------- 6  per-SKU head to head
    ax = fig.add_subplot(gs[1, 2])
    dref = load_preds(ref_name) if ref_name != name else None
    if d is not None and dref is not None:
        a = per_sku_wape(d.filter(pl.col("h") == 90)).select("sku", "wape", "vol")
        b = per_sku_wape(dref.filter(pl.col("h") == 90)).select("sku", wape_ref="wape")
        j = a.join(b, on="sku")
        diff = (j.get_column("wape") - j.get_column("wape_ref")).to_numpy()
        diff = np.clip(diff, -0.6, 0.6)
        better = float((diff < 0).mean())
        ax.hist(diff, bins=45, color=COLOR.get(name, "#4C6EF5"), alpha=0.85)
        ax.axvline(0, color="#212529", lw=1.2)
        ax.set_xlabel(f"this model's SKU WAPE  −  {ref_name}'s   (clipped at ±0.6)")
        ax.set_ylabel("SKUs")
        ax.text(0.03, 0.95,
                f"better on {better:.0%} of the {len(diff):,} SKUs\n"
                f"worse on {1 - better:.0%}",
                transform=ax.transAxes, va="top", fontsize=10, fontweight="bold",
                color="#212529")
        ax.text(0.03, 0.78, "← this model wins", transform=ax.transAxes, fontsize=8.5,
                color="#2F9E44")
        ax.text(0.97, 0.78, f"{ref_name} wins →", transform=ax.transAxes, fontsize=8.5,
                color="#C92A2A", ha="right")
    ax.set_title("6 · SKU by SKU against "
                 + (ref_name if ref_name != name else "— nothing else on the board"),
                 **TITLE_KW)
    _clean(ax)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _titles(ax, title: str, sub: str = "") -> None:
    """Heading plus one line of explanation, with enough pad that they never collide."""
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left",
                 pad=30 if sub else 12)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=10, color="#495057")


def _clean(ax, grid: bool = True) -> None:
    if grid:
        ax.grid(True, color=GRID_C, lw=0.6)
        ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)


# ======================================================================================
# Cross-model charts
# ======================================================================================

def horizon_profile(board: list[dict], out: Path) -> Path:
    """One line per model: does it hold up as the question gets longer?"""
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ends: list[tuple[float, float, str]] = []
    for r in board:
        hs = [h for h in HORIZONS if str(h) in r["horizons"]]
        ys = [r["horizons"][str(h)]["wape"] for h in hs]
        ax.plot(hs, ys, marker="o", lw=2.4, ms=7,
                color=COLOR.get(r["model"], "#868E96"), label=r["model"],
                ls="--" if r["model"] == BASELINE else "-")
        ends.append((hs[-1], ys[-1], r["model"]))

    # At the 6-month end of the chart several models can land very close together, so
    # the labels are pushed apart vertically rather than overprinted. The
    # dot stays on the true value; only the text moves.
    ends.sort(key=lambda t: -t[1])
    span = max(e[1] for e in ends) - min(e[1] for e in ends)
    gap, prev = span * 0.055, None
    for x, y, n in ends:
        ylab = y if prev is None else min(y, prev - gap)
        ax.annotate(n, (x, ylab), textcoords="offset points", xytext=(11, 0),
                    fontsize=9.5, va="center", color=COLOR.get(n, "#868E96"))
        prev = ylab
    ax.set_xticks(list(HORIZONS), [f"{h} days\n({HORIZON_LABEL[h]})" for h in HORIZONS])
    ax.set_xlim(20, 260)
    ax.set_ylabel("WAPE — lower is better")
    _titles(ax, "Accuracy as the question gets longer",
            "a flat line is a model that does not degrade; the ordering is not the "
            "same at every horizon")
    _clean(ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def head_to_head(board: list[dict], out: Path, horizon: int = 90) -> Path:
    """% of SKUs where the row model beats the column model. The aggregate WAPE is one
    number over a catalogue that is not homogeneous; this is the same comparison run
    once per SKU, which is how you find out whether a win is broad or carried by a
    handful of big sellers."""
    names = [r["model"] for r in board]
    per: dict[str, pl.DataFrame] = {}
    for n in names:
        d = load_preds(n)
        if d is not None:
            per[n] = per_sku_wape(d.filter(pl.col("h") == horizon)).select("sku", **{n: "wape"})
    names = [n for n in names if n in per]
    if len(names) < 2:
        return out
    j = per[names[0]]
    for n in names[1:]:
        j = j.join(per[n], on="sku")
    M = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for k, b in enumerate(names):
            if i != k:
                M[i, k] = float((j.get_column(a) < j.get_column(b)).mean())

    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.25, vmax=0.75)
    for i in range(len(names)):
        for k in range(len(names)):
            if i == k:
                ax.text(k, i, "—", ha="center", va="center", color="#868E96")
            else:
                ax.text(k, i, f"{M[i, k]:.0%}", ha="center", va="center", fontsize=10,
                        fontweight="bold" if abs(M[i, k] - 0.5) > 0.1 else "normal",
                        color="#212529")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(names)), names, fontsize=9.5)
    _titles(ax, "How often the row model beats the column model, SKU by SKU",
            f"{HORIZON_LABEL[horizon]}, {len(j):,} SKUs with demand in the window · "
            f"green row = wins broadly, not just on the big sellers")
    fig.colorbar(im, ax=ax, shrink=0.72, label="share of SKUs the row model wins")
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def by_volume(board: list[dict], out: Path, horizon: int = 90) -> Path:
    """WAPE per volume band, every model. This is the chart that justifies keeping
    seven: the winner catalogue-wide need not be the winner in every band."""
    names = [r["model"] for r in board]
    order = [b[2] for b in BANDS]
    rows: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for n in names:
        d = load_preds(n)
        if d is None:
            continue
        s = d.filter(pl.col("h") == horizon)
        vol = s.group_by("sku").agg(v=pl.col("actual").mean() / horizon)
        s = s.join(vol, on="sku")
        s = s.with_columns(pl.Series("band", band_of(s.get_column("v").to_numpy())))
        g = (s.group_by("band")
             .agg(wape=(pl.col("predicted") - pl.col("actual")).abs().sum()
                  / pl.col("actual").abs().sum(), skus=pl.col("sku").n_unique())
             .with_columns(pl.col("band").cast(pl.Enum(order))).sort("band"))
        # A band with no SKUs at this horizon is NaN, so every model keeps one value
        # per band and the bars stay aligned with their labels.
        by_band = dict(zip(g.get_column("band").to_list(), g.get_column("wape").to_list()))
        rows[n] = [by_band.get(b, float("nan")) for b in order]
        counts = dict(zip(g.get_column("band").to_list(), g.get_column("skus").to_list()))

    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    w = 0.8 / max(len(rows), 1)
    x = np.arange(len(order))
    for i, (n, vals) in enumerate(rows.items()):
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w * 0.92,
               color=COLOR.get(n, "#868E96"), label=n)
    # Mark the winner in each band ON the winning bar -- which band belongs to which
    # model is the entire point of this chart, and it is not always the same model.
    names_in = list(rows)
    for j in range(len(order)):
        col = [rows[n][j] for n in names_in]
        if all(np.isnan(col)):
            continue
        i = int(np.nanargmin(col))
        ax.text(j + i * w - 0.4 + w / 2, col[i] + 0.012, "\u2605", ha="center",
                fontsize=11, color="#212529")
    ax.set_xticks(x, [f"{lab}\n{counts.get(lab, 0)} SKUs" for lab in order])
    ax.set_xlabel("SKU size — units per day actually sold in the test window")
    ax.set_ylabel("WAPE — lower is better")
    _titles(ax, "The same seven models, cut by how big the SKU is",
            "\u2605 = best in that band · thin SKUs are where every model is worst, and "
            "where the specialists earn their place on the list")
    ax.legend(fontsize=9.5, frameon=False, ncols=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.17))
    _clean(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def tradeoff(board: list[dict], out: Path) -> Path:
    """Accuracy against the direction of the error. Two models with the same WAPE are
    not the same purchase order: one of them is buying that score with stockouts."""
    fig, ax = plt.subplots(figsize=(10.5, 7))
    placed: list[tuple[float, float]] = []
    for r in board:
        w = mean_wape(r)
        sf = float(np.mean([h["shortfall"] for h in r["horizons"].values()]))
        bias = float(np.mean([h["bias"] for h in r["horizons"].values()]))
        ax.scatter(w, sf, s=260 + abs(bias) * 2600, color=COLOR.get(r["model"], "#868E96"),
                   alpha=0.82, edgecolor="white", lw=1.6, zorder=3)
        # Two models can land on top of each other (two similar boosted models may);
        # flip the second one's label above the bubble instead of overprinting.
        clash = any(abs(w - pw) < 0.006 and abs(sf - ps) < 0.004 for pw, ps in placed)
        ax.annotate(f"{r['model']}\nbias {bias:+.3f}", (w, sf), textcoords="offset points",
                    xytext=(0, 20 if clash else -34), ha="center", fontsize=9,
                    color="#343A40")
        placed.append((w, sf))
    ax.set_xlabel("mean WAPE — how wrong  (lower is better)")
    ax.set_ylabel("mean shortfall — units you could not supply, as a share of demand")
    _titles(ax, "Being wrong vs being short",
            "bottom-left is the good corner · bubble size = how one-sided the error is · "
            "shortfall is the half of the error that costs a sale")
    ax.margins(0.18)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ======================================================================================
# The page
# ======================================================================================

def write_html(board: list[dict], cards: dict[str, Path], cross: list[tuple[str, str, Path]],
               out: Path) -> Path:
    """A single self-contained page: leaderboard, the four cross-model charts, then one
    section per model. Images are referenced by relative path, so the page only works
    from inside this folder -- which is the point, it is a local dashboard, not a
    deliverable."""
    def rel(p: Path) -> str:
        # Percent-encode: these paths carry a directory name that is free to change,
        # and a space or other unsafe character in a src attribute is only rescued by
        # browser leniency.
        return quote(str(p.relative_to(LAB)))

    rows = []
    for i, r in enumerate(board, 1):
        cells = "".join(
            f"<td>{r['horizons'][str(h)]['wape']:.3f}"
            f"<span class=sd> ±{r['horizons'][str(h)]['wape_sd']:.3f}</span></td>"
            if str(h) in r["horizons"] else "<td>—</td>" for h in HORIZONS)
        # A colour swatch rather than coloured text: the chart colours are chosen to be
        # told apart as bars, and some of them (croston's amber) are unreadable as type.
        rows.append(
            f"<tr><td class=rank>{i}</td>"
            f"<td><a href='#{r['model']}'>"
            f"<span class=dot style='background:{COLOR.get(r['model'], '#868E96')}'></span>"
            f"<b>{r['model']}</b></a>"
            f"<div class=keep>{KEEPS.get(r['model'], '')}</div></td>"
            f"<td class=mean>{mean_wape(r):.3f}</td>{cells}</tr>")

    sections = []
    for r in board:
        n = r["model"]
        img = f"<img src='{rel(cards[n])}' alt='{n} card'>" if n in cards else ""
        pros = "".join(f"<li>{p}</li>" for p in r.get("pros", [])[:4])
        cons = "".join(f"<li>{c}</li>" for c in r.get("cons", [])[:4])
        sections.append(f"""
<section id="{n}">
  <h2 style="border-left:6px solid {COLOR.get(n, '#868E96')}">{n}</h2>
  <p class=blurb>{r.get('blurb', '')}</p>
  {img}
  <div class=cols>
    <div><h4>Strengths</h4><ul>{pros}</ul></div>
    <div><h4>Costs</h4><ul>{cons}</ul></div>
  </div>
</section>""")

    cross_html = "".join(
        f"<figure><img src='{rel(p)}' alt='{t}'><figcaption><b>{t}</b> — {c}</figcaption></figure>"
        for t, c, p in cross)

    tuned = K.MODE == "tuned"
    # A banner on every page pointing at the other half, so whichever one you opened you
    # can always find its counterpart and the diff between them. Relative links, because
    # both pages and TUNING.md live side by side in dev/.
    other = "dashboard.html" if tuned else "dashboard_tuned.html"
    other_label = "the models as their files are written" if tuned else \
        "the same models after tune.py chose the hyper-parameters"
    banner = (
        f"<div class=banner><b>{'Tuned' if tuned else 'Initial'}</b> — "
        f"{'hyper-parameters chosen by ' if tuned else 'the parameters the files define, no search. '}"
        f"{'<code>dev/tune.py</code>.' if tuned else ''} "
        f"Compare against <a href='{other}'>{other_label}</a>, or read the "
        f"<a href='TUNING.md'>before/after</a>.</div>")

    out.write_text(f"""<!doctype html>
<meta charset=utf-8><title>{'Tuned' if tuned else 'Initial'} — 7 models</title>
<style>
 :root {{ color-scheme: light; }}
 body {{ font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        max-width: 1180px; margin: 0 auto; padding: 32px 20px 80px; color: #212529; }}
 h1 {{ font-size: 30px; margin: 0 0 4px; }}
 .sub {{ color: #868E96; margin: 0 0 28px; }}
 table {{ border-collapse: collapse; width: 100%; margin-bottom: 40px; }}
 th, td {{ padding: 9px 10px; text-align: right; border-bottom: 1px solid #E9ECEF; }}
 th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
 th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #868E96; }}
 .rank {{ color: #ADB5BD; }} .mean {{ font-weight: 700; }}
 .sd {{ color: #ADB5BD; font-size: 11px; }}
 .keep {{ color: #868E96; font-size: 12px; max-width: 460px; margin-left: 18px; }}
 .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
         margin-right: 8px; vertical-align: middle; }}
 td a {{ text-decoration: none; }} td a:hover b {{ text-decoration: underline; }}
 img {{ width: 100%; border: 1px solid #E9ECEF; border-radius: 8px; }}
 figure {{ margin: 0 0 34px; }}
 figcaption {{ color: #495057; font-size: 13px; margin-top: 8px; }}
 section {{ margin: 54px 0; }}
 h2 {{ padding-left: 12px; margin-bottom: 6px; }}
 .blurb {{ color: #495057; margin: 0 0 16px; }}
 .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 26px; margin-top: 16px; }}
 h4 {{ margin: 0 0 6px; font-size: 13px; text-transform: uppercase; color: #868E96; }}
 ul {{ margin: 0; padding-left: 18px; color: #495057; font-size: 13.5px; }}
 a {{ color: inherit; }}
 .banner {{ background: #F1F3F5; border-left: 5px solid #1971C2; border-radius: 6px;
            padding: 12px 16px; margin: 0 0 26px; font-size: 13.5px; color: #495057; }}
 .banner a {{ color: #1971C2; }}
</style>
<h1>The seven{' — tuned' if tuned else ''}</h1>
<p class=sub>Rolling-origin backtest, 3 splits, full catalogue.
Built by <code>dev/dashboard.py</code> from <code>dev/{results_dir().name}/</code>.</p>
{banner}
<table>
 <tr><th>#</th><th>model</th><th>mean WAPE</th>
 {''.join(f'<th>{h}d</th>' for h in HORIZONS)}</tr>
 {''.join(rows)}
</table>
<h2>All seven at once</h2>
{cross_html}
{''.join(sections)}
""")
    return out


# ======================================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build every figure for the seven models.")
    ap.add_argument("--only", type=str, default=None,
                    help="rebuild one model's card instead of all of them")
    ap.add_argument("--no-html", action="store_true", help="PNGs only")
    ap.add_argument("--tuned", action="store_true",
                    help="draw results_tuned/ into tuned_figures/ instead of "
                         "results/ into initial_figures/")
    a = ap.parse_args()
    set_mode("tuned" if a.tuned else "initial")
    FIGURES = figures_dir()
    CARDS = FIGURES / "cards"
    HTML = HTMLS[K.MODE]

    board = load_board()
    if not board:
        raise SystemExit(f"no results in dev/{results_dir().name}/ -- run the models "
                         f"first:\n  uv run python dev/compare_models.py --run-all"
                         f"{' --tuned' if a.tuned else ''}")
    print(f"drawing [{K.MODE}] -> {FIGURES.name}/")

    # The per-horizon figures a model file writes as it runs. They are pure functions of
    # the cached predictions, so redrawing them here costs seconds and does not need the
    # hour a refit would -- which is what lets `initial_figures/all/` exist at all, given
    # that half's results were copied from model_lab rather than produced in place.
    if not a.only:
        for r in board:
            n = r["model"]
            d = load_preds(n)
            if d is None:
                continue
            for h in HORIZONS:
                e = r["horizons"].get(str(h))
                sub = d.filter(pl.col("h") == h)
                if not e or not len(sub):
                    continue
                _fig_for_horizon(n, h, e["folds"], sub,
                                 FIGURES / "all" / n / f"h{h:03d}.png")
        print(f"per-horizon figures -> {FIGURES.name}/all/")

    cards: dict[str, Path] = {}
    for r in board:
        n = r["model"]
        p = CARDS / f"{n}.png"
        if a.only and a.only != n:
            if p.exists():
                cards[n] = p
            continue
        print(f"card  {n}")
        cards[n] = card(r, board, p)

    cross: list[tuple[str, str, Path]] = []
    if not a.only:
        print("cross-model charts")
        cross = [
            ("Accuracy by horizon",
             "every model on one axis; the ordering need not be the same for the 1-month "
             "and 6-month questions.",
             horizon_profile(board, FIGURES / "_horizon_profile.png")),
            ("Head to head, per SKU",
             "aggregate WAPE can be carried by a few big sellers. This is the same "
             "comparison run once per SKU.",
             head_to_head(board, FIGURES / "_head_to_head.png")),
            ("Accuracy by SKU size",
             "the catalogue is not homogeneous, and neither is the answer to which "
             "model is best.",
             by_volume(board, FIGURES / "_by_volume.png")),
            ("Error vs shortfall",
             "the same WAPE bought two different ways: one model over-orders, another "
             "runs you out of stock.",
             tradeoff(board, FIGURES / "_tradeoff.png")),
        ]

    if not a.no_html and not a.only:
        print(f"page  -> {write_html(board, cards, cross, HTML).relative_to(LAB.parent)}")
    for n, p in cards.items():
        print(f"       {p.relative_to(LAB.parent)}")
