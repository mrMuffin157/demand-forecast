"""Score the forecasts this project actually ISSUED against what actually happened.

    uv run python dev/track.py                 # every closed horizon, every forecast
    uv run python dev/track.py --min-coverage 1.0   # only fully-elapsed windows

Everything else in this folder scores a backtest: a forecast made from an origin that was
already in the past, by a model fitted for the purpose, on folds chosen by the folder.
That is the right way to compare models and the wrong way to find out whether the number
somebody ordered against was any good.

This file closes the loop. It reads `dev/forecasts/forecast_<origin>.csv` -- the real
output, with the real model choice and the real multiplier baked in -- waits for the
window to elapse, and scores it against the panel. Writes `dev/TRACK.md`.

-----------------------------------------------------------------------------------
WHY THIS IS NOT THE SAME NUMBER AS THE BACKTEST
-----------------------------------------------------------------------------------
An issued forecast differs from its own backtest in ways that only show up here:

  * it is made from an OPEN origin, whose features come from `open_origins.parquet`
    rather than from the training file. If those two ever drift apart, the backtest
    stays clean and this number degrades -- which is the entire failure mode
    `tests/test_forecast_origins.py` exists to prevent, measured rather than asserted.
  * the model and the multiplier were chosen on backtest folds of data ending before
    the origin. Here they meet a period nobody selected them on.
  * it is a single origin, not an average over many folds and thousands of rows. One bad
    quarter shows up at full strength instead of being averaged away.

So a gap between this table and the backtest boards is not necessarily a bug -- one origin is a
small sample and the confidence interval is wide. A gap that PERSISTS across origins is
a bug, and this is the only file that can see it.

-----------------------------------------------------------------------------------
BOTH COLUMNS GET SCORED, AND THEY ARE JUDGED DIFFERENTLY
-----------------------------------------------------------------------------------
`forecast_typical` should be roughly UNBIASED -- it is the revenue-planning number, and
being systematically wrong in either direction is a defect.

`forecast_planning` should be BIASED HIGH, on purpose, by about the multiplier. Judging
it on WAPE would mark down the deliberate stockout-aversion encoded in the cost ratio. Its
metric is fill rate against the target, and its cost at `common.CU:CO`.

-----------------------------------------------------------------------------------
AND SO DOES THE HUMAN INPUT
-----------------------------------------------------------------------------------
Where a forecast carries known-future overrides (`dev/events.py`), this scores the
issued number against `forecast_planning_base` -- the same model output with no override
applied. That comparison is the only thing standing between an events file and a
hand-maintained forecast table that nobody ever scores and that quietly fills up with
hindsight.

It is only meaningful because `events.eligible()` refuses to apply a row entered after
the origin. Without that rule this table would improve every time somebody added an
event, and the improvement would be fictional.

-----------------------------------------------------------------------------------
PARTIAL WINDOWS
-----------------------------------------------------------------------------------
A 180-day forecast made six weeks ago is not scoreable yet, but it is not uninformative
either. Where a window is only partly elapsed this scores the elapsed part against the
same fraction of the forecast, and labels the row with its coverage so a 35%-elapsed
window is never quoted as a result. `--min-coverage` filters them out entirely.
The pro-rating assumes demand is flat within the window, which is wrong in a seasonal
month -- one more reason the partial rows are labelled rather than trusted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import LAB  # noqa: E402
from demandml import config as C  # noqa: E402

FORECASTS = LAB / "forecasts"
PANEL = C.PROCESSED / "panel_daily.parquet"


def actuals(skus: list[str], origin: dt.date, horizon: int,
            panel: pl.LazyFrame) -> pl.DataFrame:
    """Units sold per SKU over (origin, origin + horizon], plus how much of the window
    the data actually covers.

    The window OPENS the day after the origin. `units_lag_0` is the origin day's own
    sales and is a feature, so counting the origin day in the target as well would score
    the model on a day it was shown -- the same off-by-one the project's whole timing
    convention (origin-day features, targets strictly after) exists to prevent.
    """
    lo = origin + dt.timedelta(days=1)
    hi = origin + dt.timedelta(days=horizon)
    w = (panel.filter((pl.col(C.SKU).is_in(skus))
                      & (pl.col(C.DATE) >= lo) & (pl.col(C.DATE) <= hi))
         .group_by(C.SKU).agg(pl.col(C.TARGET).sum().alias("actual"))
         .collect(engine="streaming"))
    return pl.DataFrame({C.SKU: skus}).join(w, on=C.SKU, how="left").with_columns(
        actual=pl.col("actual").fill_null(0.0))


def metrics(y: np.ndarray, p: np.ndarray, cu: float, co: float) -> dict:
    den = float(np.sum(y))
    if den <= 0:
        return {k: float("nan") for k in ("wape", "bias", "fill", "cost", "units", "w40", "w50")}
    short, over = np.maximum(y - p, 0.0), np.maximum(p - y, 0.0)
    # Share of SKUs within +/-40% / 50% of what sold (actual > 0) -- the precision the
    # forecast is judged on, next to the cost it is bought on.
    pos = y > 0
    ape = np.abs(p[pos] - y[pos]) / y[pos]
    return {
        "wape": float(np.sum(np.abs(y - p)) / den),
        "bias": float(np.sum(p - y) / den),
        "fill": float(1.0 - np.sum(short) / den),
        "cost": float(np.sum(cu * short + co * over) / den),
        "units": den,
        "w40": float((ape <= 0.40).mean()) if pos.any() else float("nan"),
        "w50": float((ape <= 0.50).mean()) if pos.any() else float("nan"),
    }


def score_file(csv: Path, panel: pl.LazyFrame, last_day: dt.date,
               cu: float, co: float, min_cov: float) -> list[dict]:
    fc = pl.read_csv(csv, try_parse_dates=True)
    need = {"sku", "origin_date", "horizon_days", "forecast_typical", "forecast_planning"}
    if not need.issubset(fc.columns):
        return []
    origin = fc["origin_date"][0]
    if isinstance(origin, str):
        origin = dt.date.fromisoformat(origin)

    out = []
    for h in sorted(fc["horizon_days"].unique().to_list()):
        s = fc.filter(pl.col("horizon_days") == h).sort("sku")
        elapsed = (last_day - origin).days
        cov = min(1.0, max(0.0, elapsed / h))
        if cov <= 0 or cov < min_cov:
            continue
        a = actuals(s["sku"].to_list(), origin, min(h, elapsed), panel)
        y = a["actual"].to_numpy()
        # Pro-rate the forecast to the part of the window that has actually elapsed.
        t = s["forecast_typical"].to_numpy() * cov
        pl_ = s["forecast_planning"].to_numpy() * cov
        row = {
            "file": csv.name, "origin": str(origin), "horizon": int(h),
            "coverage": cov, "closed": cov >= 1.0, "skus": s.height,
            "model": s["model"][0] if "model" in s.columns else "?",
            "multiplier": float(s["multiplier"][0]) if "multiplier" in s.columns else float("nan"),
            "expected_fill": (float(s["expected_fill_rate"][0])
                              if "expected_fill_rate" in s.columns else float("nan")),
        }
        row |= {f"typ_{k}": v for k, v in metrics(y, t, cu, co).items()}
        row |= {f"pln_{k}": v for k, v in metrics(y, pl_, cu, co).items()}

        # Did the human input help? Only scoreable where the forecast carried overrides
        # and the base column was written alongside them.
        if "forecast_planning_base" in s.columns and "n_events" in s.columns:
            n_ev = int((s["n_events"] > 0).sum())
            row["n_event_skus"] = n_ev
            if n_ev:
                b = s["forecast_planning_base"].to_numpy() * cov
                mb = metrics(y, b, cu, co)
                row |= {f"base_{k}": v for k, v in mb.items()}
                row["event_cost_change"] = row["pln_cost"] - mb["cost"]
        out.append(row)
    return out


def report(rows: list[dict], cu: float, co: float, last_day: dt.date) -> str:
    L: list[str] = []
    A = L.append
    A("# Did the forecasts we issued actually work?\n")
    A(f"*Generated by `dev/track.py` from `dev/forecasts/`, scored against "
      f"`panel_daily.parquet` (data through **{last_day}**). Cost at `{cu:g}:{co:g}`.*\n")
    if not rows:
        A("No issued forecast has a window with any elapsed days yet. "
          "Run `dev/forecast.py`, wait, and run this again.\n")
        return "\n".join(L)

    closed = [r for r in rows if r["closed"]]
    A(f"{len(rows)} forecast-horizons scoreable, {len(closed)} with a fully elapsed "
      f"window.\n")

    A("## The planning number — the one you buy against\n")
    A("`forecast_planning` is *supposed* to run high; the multiplier put it there. So it "
      "is judged on fill rate against what the backtest promised, not on WAPE.\n")
    A("| origin | h | model | k | coverage | fill achieved | fill promised | cost | units |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["origin"], r["horizon"])):
        cov = "**closed**" if r["closed"] else f"{r['coverage']:.0%} elapsed"
        A(f"| {r['origin']} | {r['horizon']} | `{r['model']}` | {r['multiplier']:.2f} | "
          f"{cov} | **{r['pln_fill']:.3f}** | {r['expected_fill']:.3f} | "
          f"{r['pln_cost']:.4f} | {r['pln_units']:,.0f} |")
    A("")
    A("## The typical number — the one you plan revenue on\n")
    A("`forecast_typical` should be close to unbiased. A persistent sign here is a "
      "defect, not a policy.\n")
    A("| origin | h | coverage | WAPE | bias | fill |")
    A("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["origin"], r["horizon"])):
        cov = "**closed**" if r["closed"] else f"{r['coverage']:.0%}"
        A(f"| {r['origin']} | {r['horizon']} | {cov} | {r['typ_wape']:.4f} | "
          f"{r['typ_bias']:+.4f} | {r['typ_fill']:.3f} |")
    A("")

    A("## Against what the backtest promised\n")
    gaps = [r for r in rows if np.isfinite(r["expected_fill"])]
    if gaps:
        A("| origin | h | coverage | fill promised | fill achieved | gap |")
        A("|---|---|---|---|---|---|")
        for r in sorted(gaps, key=lambda r: (r["origin"], r["horizon"])):
            g = r["pln_fill"] - r["expected_fill"]
            cov = "**closed**" if r["closed"] else f"{r['coverage']:.0%}"
            A(f"| {r['origin']} | {r['horizon']} | {cov} | {r['expected_fill']:.3f} | "
              f"{r['pln_fill']:.3f} | {g:+.3f} |")
        A("")
        worst = min(gaps, key=lambda r: r["pln_fill"] - r["expected_fill"])
        A(f"Widest shortfall against promise: **{worst['origin']} / "
          f"{worst['horizon']}d**, {worst['pln_fill']:.3f} achieved against "
          f"{worst['expected_fill']:.3f} promised "
          f"({worst['pln_fill'] - worst['expected_fill']:+.3f}).\n")

    ev = [r for r in rows if r.get("n_event_skus")]
    if ev:
        A("## Did the known-future overrides help?\n")
        A("`forecast_planning` as issued, against the same model output with no "
          "override applied. Only forecasts that carried at least one event appear. "
          "Negative `change` means the human input reduced inventory cost.\n")
        A("| origin | h | SKUs with an event | cost base | cost issued | change |")
        A("|---|---|---|---|---|---|")
        for r in sorted(ev, key=lambda r: (r["origin"], r["horizon"])):
            d = r["event_cost_change"]
            pct = 100 * d / r["base_cost"] if r.get("base_cost") else float("nan")
            A(f"| {r['origin']} | {r['horizon']} | {r['n_event_skus']:,} | "
              f"{r['base_cost']:.4f} | {r['pln_cost']:.4f} | **{pct:+.2f}%** |")
        A("")
        A("Read this against `dev/event_backtest.py`'s report, which replays the promo "
          "calendar over every backtest origin. A single origin here is a "
          "far weaker observation than that, so treat a favourable row as encouragement "
          "to keep collecting rather than as evidence the overrides work.\n")
    elif any("n_event_skus" in r for r in rows):
        A("## Did the known-future overrides help?\n")
        A("No issued forecast carried an override, so there is nothing to compare. "
          "Every number above is the model's own.\n")

    A("## How to read this\n")
    A("- **One origin is a small sample.** The backtest averages many folds over "
      "many SKU-windows; a row here is one date. A single row differing from "
      "the backtest by a few points is expected. The same sign appearing on every origin, "
      "as more accumulate, is not.\n")
    A("- **Partial rows are pro-rated flat.** A window that is 40% elapsed has its "
      "forecast multiplied by 0.40, which assumes demand spreads evenly. Across a "
      "seasonal boundary it does not. Treat anything not marked **closed** as an early "
      "indicator.\n")
    A("- **Censoring is not corrected here.** The backtest drops windows that were "
      "heavily out of stock; this scores what the panel recorded. A SKU that stocked out "
      "will show as over-forecast when it was really under-supplied, so `fill achieved` "
      "is a floor, not a point estimate.\n")
    return "\n".join(L)


def _is_backfill(csv_path: Path) -> bool:
    meta = csv_path.with_suffix(".json")
    return meta.exists() and "backfill" in json.loads(meta.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="skip windows less than this fraction elapsed (0-1)")
    ap.add_argument("--forecast", type=str, default=None, help="score one file only")
    ap.add_argument("--metrics", type=str, default=None,
                    help="also write flat key/value metrics here for a monitor to scrape")
    ap.add_argument("--alert-gap", type=float, default=0.05,
                    help="exit non-zero if realised fill is this far under what the "
                         "backtest promised, on a CLOSED window (default 0.05)")
    args = ap.parse_args()
    cu, co = common.cost_ratio()

    if not PANEL.exists():
        raise SystemExit(f"{PANEL} does not exist. Build it:  uv run demandml panel")
    files = ([Path(args.forecast)] if args.forecast
             else sorted(FORECASTS.glob("forecast_*.csv")))
    # A `forecast.py --as-of` rebuild is not a forecast anyone acted on, and this file is
    # the record of what was ISSUED -- scoring rebuilds here would mix the two.
    files = [f for f in files if args.forecast or not _is_backfill(f)]
    if not files:
        raise SystemExit("no forecasts in dev/forecasts/ -- run dev/forecast.py first")

    panel = pl.scan_parquet(PANEL)
    last_day = panel.select(pl.col(C.DATE).max()).collect().item()

    print(f"\n{'=' * 92}\nFORECAST TRACK RECORD   panel data through {last_day}"
          f"   cu:co = {cu:g}:{co:g}\n{'=' * 92}")
    rows: list[dict] = []
    for f in files:
        got = score_file(f, panel, last_day, cu, co, args.min_coverage)
        rows += got
        print(f"  {f.name:<34} {len(got)} scoreable horizon(s)")

    if rows:
        print(f"\n  {'origin':<12}{'h':>5}{'cov':>7}{'fill':>8}{'promised':>10}"
              f"{'cost':>9}{'typ bias':>10}")
        for r in sorted(rows, key=lambda r: (r["origin"], r["horizon"])):
            print(f"  {r['origin']:<12}{r['horizon']:>5}{r['coverage']:>6.0%}"
                  f"{r['pln_fill']:>8.3f}{r['expected_fill']:>10.3f}"
                  f"{r['pln_cost']:>9.4f}{r['typ_bias']:>+10.4f}")

    out = LAB / "TRACK.md"
    out.write_text(report(rows, cu, co, last_day))
    (FORECASTS / "_track.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n{'=' * 92}\nwrote {out.relative_to(LAB.parent)}")

    # --- metrics, for something other than a human to read ---------------------------
    #
    # A markdown report is the right output for a person reading one quarter's numbers
    # and the wrong one for a schedule. Catalogue bias swings fold to fold by far more
    # than its average (docs/FINDINGS.md), so a multiplier fitted once is wrong in one
    # direction for part of any year. Checked only occasionally, nobody would see that
    # until after the next long-window order was already committed.
    closed = [r for r in rows if r["closed"] and np.isfinite(r["expected_fill"])]
    worst_gap = min((r["pln_fill"] - r["expected_fill"] for r in closed), default=None)
    metrics_out = {
        "track_scoreable_horizons": len(rows),
        "track_closed_horizons": len(closed),
        "track_panel_last_day": str(last_day),
        "track_panel_age_days": (dt.date.today() - last_day).days,
    }
    for r in rows:
        tag = f"{r['origin']}_h{r['horizon']}"
        metrics_out |= {
            f"fill_achieved__{tag}": round(r["pln_fill"], 5),
            f"fill_promised__{tag}": round(r["expected_fill"], 5),
            f"fill_gap__{tag}": round(r["pln_fill"] - r["expected_fill"], 5),
            f"typical_bias__{tag}": round(r["typ_bias"], 5),
            f"cost__{tag}": round(r["pln_cost"], 5),
            f"coverage__{tag}": round(r["coverage"], 4),
        }
    if worst_gap is not None:
        metrics_out["fill_gap_worst_closed"] = round(worst_gap, 5)
    if args.metrics:
        Path(args.metrics).write_text(json.dumps(metrics_out, indent=2, sort_keys=True))
        print(f"wrote {args.metrics}  ({len(metrics_out)} metrics)")
    print(f"{'=' * 92}")

    # Only a CLOSED window can fail the build. A 2%-elapsed 180-day forecast is pro-rated
    # flat across a seasonal boundary and is an early indicator, never a verdict -- see
    # "Partial windows" above.
    if worst_gap is not None and worst_gap < -abs(args.alert_gap):
        raise SystemExit(
            f"\nALERT: realised fill is {abs(worst_gap):.3f} under what the backtest "
            f"promised on a closed window (threshold {args.alert_gap:.3f}).\n"
            f"Check dev/TRACK.md. Either the demand regime moved (dev/regime.py) or the "
            f"open-origin feature path has drifted from the training one.")
    if not closed:
        print("\n  No window has fully elapsed yet, so nothing here is a verdict. "
              "Nothing this project produces has been validated outside a backtest.")


if __name__ == "__main__":
    main()
