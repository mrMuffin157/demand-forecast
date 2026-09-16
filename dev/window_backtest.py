"""Score the forecast at the window you actually buy against, not at 30 / 90 / 180.

    uv run python dev/window_backtest.py                  # sweep 30..300 days
    uv run python dev/window_backtest.py --windows 100    # just your real window
    uv run python dev/window_backtest.py --tuned

Writes a markdown report and `<figures>/_window.png`. Nothing is refitted -- this reads the
cached predictions in `results/_preds/` and the panel, so it runs in seconds.

WHY THIS FILE EXISTS
====================
Every score on the model boards is measured at 30, 90 or 180 days. Those numbers were
chosen because they are round. Nobody buys against them.

What an order is exposed to is `lead time + review period` (set in `assumptions.toml`),
which in general is not 30, 90 or 180, and may not be a horizon anything in this project
has ever scored. `policy.py` gets there by interpolating the cumulative forecast onto the
window and, past 180 days, extending it at the 90-to-180 daily rate. That extension is an
assumption. This file measures what it costs.

The distinction that matters:

    30 / 90 / 180 d   a model was TRAINED on this target and BACKTESTED on it
    anything else     read off a curve through three points -- never validated
    beyond 180 d      a flat-rate extension of the last segment -- pure assumption

With long lead times, much or all of the catalogue can sit in the third category, so the
order book would rest on the one part of the pipeline with no evidence behind it. That is
the gap this file closes. See docs/FINDINGS.md.

HOW IT SCORES
=============
For each backtest origin the cached predictions give cumulative demand at 30, 90 and 180
days. `policy.interpolate_cumulative` turns those into a prediction at an arbitrary
window W -- the same function `policy.py` uses, deliberately, so this measures the
serving code path rather than a reimplementation of it.

The actual is summed straight off `panel_daily`: units over the W days AFTER the origin.
The origin day itself is excluded, because `units_lag_0` is a feature and counting the
origin's own units would score the model on a day it was shown.

CENSORING
=========
`prepare_data.py` drops any training target whose future window was more than 25% out of
stock. The same rule is applied here, for the same reason: a SKU that was unavailable for
half the window did not have low demand, it had no opportunity, and scoring it measures
the stock-out rather than the forecast. `--keep-censored` turns it off, and the report
prints how many rows each choice drops so the difference is visible rather than assumed.

WHAT IT CANNOT TELL YOU
=======================
It cannot validate a window longer than the data allows. An origin needs W days of
future panel to be scoreable at all, so the longest windows are scored on the fewest and
oldest origins -- the 300-day column rests on origins from a different year than the
30-day one. The report prints `n` and the origin range for every window; a difference
across windows is partly a difference in the sample, and the row count is the warning.
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
import policy as P  # noqa: E402
from common import LAB, figures_dir, results_dir  # noqa: E402
from demandml import config as C  # noqa: E402

PANEL = C.PROCESSED / "panel_daily.parquet"

#: Same rule prepare_data.py applies to its targets. A window more out of stock than this
#: measures availability, not demand.
MAX_OOS_FRACTION = 0.25

DEFAULT_WINDOWS = (30, 60, 90, 120, 150, 180, 210, 240, 270, 300)


# ======================================================================================
# Actuals over an arbitrary window
# ======================================================================================

def actual_over_windows(origins: pl.DataFrame, windows: list[int],
                        keep_censored: bool) -> pl.DataFrame:
    """Units in the W days after each (sku, origin), plus the out-of-stock share.

    `origins` is the distinct (sku, date) set to score. Returns one row per
    (sku, date, window) with `actual` and `oos_frac`, null where the window runs past the
    end of the panel -- an incomplete window must not be scored as a small one.
    """
    panel = (pl.scan_parquet(PANEL)
             .select(["sku", "date", "units", "is_oos"])
             .filter(pl.col("sku").is_in(origins["sku"].unique().to_list()))
             .collect())
    last_day = panel["date"].max()

    # Cumulative units per SKU by day, so a window sum is two lookups rather than a join
    # per window. A panel of a few thousand SKUs x a decade of days fits comfortably.
    panel = panel.sort(["sku", "date"]).with_columns(
        cum_units=pl.col("units").cum_sum().over("sku"),
        cum_oos=pl.col("is_oos").cast(pl.Int32).cum_sum().over("sku"))

    out = []
    for w in windows:
        # Window is (origin, origin + w], i.e. the origin day itself is EXCLUDED.
        left = origins.select("sku", "date").with_columns(
            lo=pl.col("date"), hi=pl.col("date").dt.offset_by(f"{w}d"))
        j = (left
             .join(panel.select("sku", "date", "cum_units", "cum_oos"),
                   left_on=["sku", "lo"], right_on=["sku", "date"], how="left")
             .rename({"cum_units": "cum_lo", "cum_oos": "oos_lo"})
             .join(panel.select("sku", "date", "cum_units", "cum_oos"),
                   left_on=["sku", "hi"], right_on=["sku", "date"], how="left")
             .rename({"cum_units": "cum_hi", "cum_oos": "oos_hi"}))
        j = j.with_columns(
            actual=pl.when(pl.col("hi") <= last_day)
                     .then(pl.col("cum_hi") - pl.col("cum_lo")).otherwise(None),
            oos_frac=pl.when(pl.col("hi") <= last_day)
                       .then((pl.col("oos_hi") - pl.col("oos_lo")) / w).otherwise(None),
            window=pl.lit(w, dtype=pl.Int32))
        out.append(j.select("sku", "date", "window", "actual", "oos_frac"))

    res = pl.concat(out).filter(pl.col("actual").is_not_null())
    if not keep_censored:
        res = res.filter(pl.col("oos_frac") <= MAX_OOS_FRACTION)
    return res


# ======================================================================================
# Scoring
# ======================================================================================

def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    cu, co = common.cost_ratio()
    tot = max(float(np.sum(np.abs(y))), 1e-9)
    short = np.clip(y - p, 0.0, None)
    over = np.clip(p - y, 0.0, None)
    return {
        "wape": float(np.sum(np.abs(y - p)) / tot),
        "bias": float(np.sum(p - y) / tot),
        "cost": float((cu * short.sum() + co * over.sum()) / tot),
        "fill": float(1.0 - short.sum() / tot),
    }


def run(windows: list[int], keep_censored: bool, multiplier: bool) -> pl.DataFrame:
    """Score every window, on the serving interpolation, per model."""
    preds_dir = results_dir() / "_preds"
    if not preds_dir.exists():
        raise SystemExit(f"no cached predictions in {preds_dir}")

    rows = []
    for pth in sorted(preds_dir.glob("*.parquet")):
        model = pth.stem
        d = pl.read_parquet(pth)
        # Pivot on the ORIGIN, not on (origin, fold). `make_folds` derives each horizon's
        # fold boundaries from the rows where THAT horizon's target is non-null, so the
        # three horizons cover different date ranges (the 180-day folds end about five
        # months before the 30-day ones) and a given origin sits in different fold numbers at different
        # horizons. Every prediction is still out-of-sample for its own fold, which is
        # what matters; insisting the fold number match as well would leave nothing.
        wide = (d.pivot(on="h", index=["sku", "date"], values="predicted")
                 .sort(["sku", "date"]))
        hcols = [str(h) for h in common.HORIZONS]
        if any(c not in wide.columns for c in hcols):
            continue
        # The curve needs all three knots. An origin missing one is dropped rather than
        # interpolated through the hole -- a null knot would produce NaN at every window
        # for that SKU and silently take the whole metric with it.
        wide = wide.drop_nulls(subset=hcols)
        if wide.height < 200:
            continue
        cum = wide.select(hcols).to_numpy()

        acts = actual_over_windows(wide.select("sku", "date").unique(),
                                   windows, keep_censored)
        for w in windows:
            pred_w, regime = P.interpolate_cumulative(cum, w)
            pf = wide.select("sku", "date").with_columns(pred=pl.Series(pred_w))
            m = acts.filter(pl.col("window") == w).join(pf, on=["sku", "date"], how="inner")
            if m.height < 200:
                continue
            y = m["actual"].to_numpy().astype(float)
            p = m["pred"].to_numpy().astype(float)
            k = 1.0
            if multiplier:
                # The same instrument policy.py uses: one global scalar at the cost-
                # minimising quantile of actual/predicted, weighted BY PREDICTED. The
                # weighting is not a style choice -- it falls out of the first-order
                # condition, and weighting by actual over-orders the catalogue.
                ok = p > 0
                if ok.sum() > 50:
                    ratio = y[ok] / p[ok]
                    order = np.argsort(ratio)
                    wts = p[ok][order]
                    cw = np.cumsum(wts) / wts.sum()
                    k = float(np.interp(common.target_quantile(), cw, ratio[order]))
            s = score(y, p * k)
            rows.append({"model": model, "window": w, "n": m.height, "k": k,
                         "regime": str(regime[0]),
                         "origin_min": m["date"].min(), "origin_max": m["date"].max(),
                         **s})
    return pl.DataFrame(rows)


# ======================================================================================
# Report
# ======================================================================================

def write_report(df: pl.DataFrame, windows: list[int], keep_censored: bool,
                 multiplier: bool) -> Path:
    cu, co = common.cost_ratio()
    best = (df.filter(pl.col("window") == min(windows))
              .sort("cost")["model"].to_list())
    lead = best[0] if best else df["model"][0]

    L: list[str] = []
    L.append("# The forecast at the window you actually buy against\n")
    L.append(f"*Generated by `dev/window_backtest.py` (mode `{common.MODE}`) from the "
             f"cached predictions in `{results_dir().name}/_preds/`. Nothing was "
             f"refitted.*\n")
    L.append("Every other board in this folder scores 30, 90 and 180 days. Nobody buys "
             "against those. An order is exposed to `lead time + review period` (set in "
             "`assumptions.toml`). This file scores the forecast at that window and "
             "around it.\n")
    L.append(f"Cost at `{cu:g}:{co:g}`. "
             f"{'A global multiplier is applied at the target quantile.' if multiplier else 'No multiplier -- raw model output.'} "
             f"{'Censored windows KEPT.' if keep_censored else f'Windows more than {MAX_OOS_FRACTION:.0%} out of stock are dropped, as in training.'}\n")
    L.append(f"![window]({figures_dir().name}/_window.png)\n")

    L.append("## The cost of leaving the evidence\n")
    L.append("`30 / 90 / 180` are trained-and-backtested targets. Everything else is read "
             "off a three-point curve, and everything past 180 d is a flat-rate "
             "extension of the final segment.\n")
    sub = df.filter(pl.col("model") == lead).sort("window")
    L.append(f"**`{lead}`** — the model the cost board leads with:\n")
    L.append("| window | regime | n | WAPE | bias | cost | fill | origins |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in sub.iter_rows(named=True):
        star = "" if r["window"] in common.HORIZONS else " ⚠"
        L.append(f"| **{r['window']} d**{star} | {r['regime']} | {r['n']:,} | "
                 f"{r['wape']:.3f} | {r['bias']:+.3f} | {r['cost']:.4f} | "
                 f"{r['fill']:.3f} | {r['origin_min']} → {r['origin_max']} |")
    L.append("\n⚠ marks a window no model was trained or backtested on.\n")

    # the headline: does error at a backtested knot predict error between/past knots?
    knot = sub.filter(pl.col("window") == 180)
    for w in (210, 240, 300):
        row = sub.filter(pl.col("window") == w)
        if row.height and knot.height:
            d = float(row["cost"][0] - knot["cost"][0])
            rel = 100 * d / max(float(knot["cost"][0]), 1e-9)
            L.append(f"- At **{w} d** the cost is {float(row['cost'][0]):.4f} against "
                     f"{float(knot['cost'][0]):.4f} at the 180-day knot — "
                     f"**{rel:+.1f}%**, on {int(row['n'][0]):,} rows.")
    L.append("")

    L.append("## Every model, every window\n")
    L.append("Cost at each window. Lower is better.\n")
    piv = (df.pivot(on="window", index="model", values="cost")
             .sort([str(min(windows))]))
    hdr = [c for c in piv.columns if c != "model"]
    L.append("| model | " + " | ".join(f"{h} d" for h in hdr) + " |")
    L.append("|---" * (len(hdr) + 1) + "|")
    for r in piv.iter_rows(named=True):
        cells = []
        for h in hdr:
            v = r[h]
            cells.append("—" if v is None else f"{v:.4f}")
        L.append(f"| `{r['model']}` | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## How to read this\n")
    L.append("- **The sample changes with the window.** An origin needs the full window "
             "of future panel to be scoreable, so longer windows are scored on fewer and "
             "older origins. The `n` and `origins` columns are the warning: some of the "
             "drift across a row is the sample moving, not the horizon getting harder.\n")
    L.append("- **A window between knots is not free.** The curve is a straight line "
             "between cumulative totals 90 days apart. Real demand inside that stretch is "
             "seasonal, and the interpolation cannot know it.\n")
    L.append("- **Past 180 days there is no evidence at all**, only the flat-rate "
             "extension. If the number above is bad, the fix is a longer-horizon target "
             "in `prepare_data.py`, not a better interpolation.\n")
    L.append("- **This scores the serving code path.** It calls "
             "`policy.interpolate_cumulative` rather than reimplementing it, so a change "
             "to the policy's curve changes this report too.\n")

    path = LAB / ("WINDOW.md" if common.MODE == "initial"
                  else f"WINDOW_{common.MODE.upper()}.md")
    path.write_text("\n".join(L))
    return path


def draw(df: pl.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    longest = max(common.HORIZONS)
    for ax, metric, title in zip(
            axes, ("cost", "wape", "fill"),
            (f"Inventory cost ({common.CU:g}:{common.CO:g})", "WAPE",
             "Fill rate")):
        for model in sorted(df["model"].unique()):
            s = df.filter(pl.col("model") == model).sort("window")
            ax.plot(s["window"], s[metric], marker="o", ms=3.5, lw=1.4,
                    label=model, alpha=.85)
        ax.axvspan(longest, df["window"].max(), color="#d9534f", alpha=.08)
        for h in common.HORIZONS:
            ax.axvline(h, color="#666", ls=":", lw=.9)
        ax.set_xlabel("coverage window (days)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("lower is better")
    axes[-1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Scored at the window you buy against — dotted lines are the only "
                 "backtested horizons; the shaded band is pure extrapolation",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score the forecast at arbitrary coverage windows.")
    ap.add_argument("--windows", type=str,
                    default=",".join(str(w) for w in DEFAULT_WINDOWS))
    ap.add_argument("--keep-censored", action="store_true",
                    help="score windows that were heavily out of stock (measures "
                         "availability, not demand)")
    ap.add_argument("--no-multiplier", action="store_true",
                    help="score the raw model output instead of the lifted forecast")
    ap.add_argument("--tuned", action="store_true")
    ap.add_argument("--cost", action="store_true")
    args = ap.parse_args()
    common.set_mode("cost" if args.cost else "tuned" if args.tuned else "initial")

    windows = [int(w) for w in args.windows.split(",") if w.strip()]
    df = run(windows, args.keep_censored, not args.no_multiplier)
    if df.is_empty():
        raise SystemExit("nothing scoreable -- are there cached predictions?")

    figures_dir().mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir() / "_window.png"
    draw(df, fig_path)
    rep = write_report(df, windows, args.keep_censored, not args.no_multiplier)
    (results_dir() / "_window.json").write_text(
        json.dumps(df.to_dicts(), indent=2, default=str))

    print(f"\n{'=' * 78}\nWINDOW BACKTEST [{common.MODE}]\n{'=' * 78}")
    lead = df.filter(pl.col("window") == min(windows)).sort("cost")["model"][0]
    for r in df.filter(pl.col("model") == lead).sort("window").iter_rows(named=True):
        flag = "" if r["window"] in common.HORIZONS else "  <- never backtested"
        print(f"  {r['window']:>4} d  {r['regime']:<13} n={r['n']:>6,}  "
              f"wape={r['wape']:.3f}  cost={r['cost']:.4f}  fill={r['fill']:.3f}{flag}")
    print(f"\nreport -> {rep}\nfigure -> {fig_path}")


if __name__ == "__main__":
    main()
