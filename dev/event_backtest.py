"""Does telling the model about a known promo actually make the forecast better?

    uv run python dev/event_backtest.py                  # the cost leader, all horizons
    uv run python dev/event_backtest.py --model random_forest
    uv run python dev/event_backtest.py --uplift 2.0     # override the fitted uplift

`dev/events.py` gives a human a way to override the forecast. This file tests whether the
mechanism does anything, on the one event class that usually has both point-in-time
knowledge and recorded outcomes: a promo calendar (`data/db/deals.parquet`, produced by
`scripts/make_sample_data.py` for the sample or by your own extract), where each deal
carries the `date_created` that says when it was first booked.

It replays history honestly. At every backtest origin it asks *which deals were already
in the system on that date*, converts them to `events.Event` objects, and pushes them
through `events.adjust` -- the same function `forecast.py` calls -- then rescores the
cached predictions. No model is refitted; this measures the override layer, not an
estimator.

-----------------------------------------------------------------------------------
THIS IS AN UPPER BOUND, NOT AN ESTIMATE
-----------------------------------------------------------------------------------
Three of the assumptions here are generous, all in the same direction:

  * **The uplift is fitted on this data.** A person typing a number into `events.csv`
    does not know the catalogue's true promotional response; this does, because it
    measured it on the panel.
  * **Every booked deal is entered.** In practice somebody has to remember to write the
    row, and the ones they forget are invisible.
  * **The deal is assumed to run as booked.** Only the aggregate cancellation rate is
    applied, as a flat probability -- no attempt is made to guess which specific deals
    will be pulled.

So if the gain here is small, the gain from the real thing is smaller.

There is a reason to expect it to be small, and to get WORSE as the assumed uplift grows:
the forecast multiplier was fitted on backtest error measured over a history that already
contained the promo days, so the average promotional response is inside `k` already and an
explicit override counts it twice. The sensitivity sweep in the report checks exactly that.
See docs/FINDINGS.md.

-----------------------------------------------------------------------------------
WHAT THE NOTICE PERIOD ALREADY IMPLIES
-----------------------------------------------------------------------------------
If deals are booked only weeks ahead and run for a day or a few, a long forecast window
contains a handful of deals that were knowable at the origin and many that were not, and
the knowable ones occupy a tiny share of the window. `dev/events.py`'s pro-rating does the
rest of the arithmetic: a 1-day event at uplift `u` inside an `h`-day window is a
`1 + (u - 1) / h` adjustment. The report prints the notice periods it actually finds.
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
import events as EV  # noqa: E402
from common import LAB, results_dir  # noqa: E402
from demandml import config as C  # noqa: E402

DEALS = C.DATA / "db" / "deals.parquet"
PANEL = C.PROCESSED / "panel_daily.parquet"

#: Statuses meaning the deal did not run. `Disabled` is counted as not-run: whatever the
#: operational distinction from `Canceled`, the units did not get the promotional lift.
DID_NOT_RUN = {"canceled", "cancel", "disabled"}
RAN = {"ended", "running"}


def load_deals(panel_skus: set[str]) -> pl.DataFrame:
    if not DEALS.exists():
        raise SystemExit(
            f"{DEALS} does not exist. Generate the sample data:\n"
            "  uv run python scripts/make_sample_data.py\n"
            f"or supply your own promo calendar there. Without it this file has nothing "
            f"to replay.")
    d = (pl.read_parquet(DEALS)
         .filter(pl.col("sku").is_not_null()
                 & pl.col("start_date").is_not_null()
                 & pl.col("end_date").is_not_null()
                 & pl.col("date_created").is_not_null())
         .with_columns(pl.col("sku").cast(pl.String),
                       pl.col("start_date").dt.date().alias("start"),
                       pl.col("end_date").dt.date().alias("end"),
                       pl.col("date_created").dt.date().alias("created"),
                       pl.col("deal_status").str.to_lowercase().alias("st"))
         .filter(pl.col("sku").is_in(list(panel_skus)))
         .filter(pl.col("end") >= pl.col("start")))
    # A deal booked after it started is a record-keeping artefact, not foreknowledge.
    return d.filter(pl.col("created") <= pl.col("start"))


def run_rate(deals: pl.DataFrame) -> float:
    """Share of *resolved* deals that actually ran. Used as the flat probability."""
    res = deals.filter(pl.col("st").is_in(list(DID_NOT_RUN | RAN)))
    if not res.height:
        return 1.0
    return float(res.filter(pl.col("st").is_in(list(RAN))).height / res.height)


def fit_uplift(before: dt.date) -> tuple[float, int]:
    """Volume-weighted demand uplift on deal days, measured only on panel days strictly
    BEFORE the first scored origin, so the number is not fitted on what it is scored on.
    """
    p = (pl.scan_parquet(PANEL).filter(pl.col(C.DATE) < before)
         .select(C.SKU, "is_deal", C.TARGET).collect(engine="streaming"))
    g = (p.group_by(C.SKU).agg(
            on=pl.col(C.TARGET).filter(pl.col("is_deal")).mean(),
            off=pl.col(C.TARGET).filter(~pl.col("is_deal")).mean(),
            n_on=pl.col("is_deal").sum(),
            n_off=(~pl.col("is_deal")).sum(),
            vol=pl.col(C.TARGET).sum())
         .filter((pl.col("n_on") >= 20) & (pl.col("n_off") >= 20) & (pl.col("off") > 0)))
    if not g.height:
        return 1.0, 0
    r = (g["on"] / g["off"]).to_numpy()
    w = g["vol"].to_numpy()
    ok = np.isfinite(r) & (w > 0)
    return float(np.sum(r[ok] * w[ok]) / np.sum(w[ok])), int(ok.sum())


def metrics(y: np.ndarray, p: np.ndarray, cu: float, co: float) -> dict:
    den = float(np.sum(y))
    if den <= 0:
        return {k: float("nan") for k in ("wape", "bias", "fill", "cost")}
    short, over = np.maximum(y - p, 0.0), np.maximum(p - y, 0.0)
    return {"wape": float(np.sum(np.abs(y - p)) / den),
            "bias": float(np.sum(p - y) / den),
            "fill": float(1.0 - np.sum(short) / den),
            "cost": float(np.sum(cu * short + co * over) / den)}


def replay(preds: pl.DataFrame, deals: pl.DataFrame, horizon: int, uplift: float,
           prob: float, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Adjusted predictions for one horizon, via `events.adjust` (the serving path).

    Returns (actual, predicted, adjusted, n_origins_touched, n_events_used).
    `k` is the forecast multiplier: the override is applied to the model's own output and
    the whole thing is then lifted, which is the order `forecast.py` uses.
    """
    ys, ps, adjs = [], [], []
    touched = used = 0
    for (origin,), grp in preds.group_by([C.DATE], maintain_order=True):
        grp = grp.sort(C.SKU)
        skus = grp.get_column(C.SKU)
        base = grp["predicted"].to_numpy()

        lo, hi = origin + dt.timedelta(days=1), origin + dt.timedelta(days=horizon)
        live = deals.filter((pl.col("created") <= origin)
                            & (pl.col("start") <= hi) & (pl.col("end") >= lo))
        evs = [EV.Event(scope=r["sku"], start=r["start"], end=r["end"], kind="promo",
                        multiplier=uplift, units=None, probability=prob,
                        entered_at=r["created"], author="deals", note="", row=0)
               for r in live.iter_rows(named=True)]
        adj, _, _, n = EV.adjust(base, skus, origin, horizon, evs, None)
        if evs:
            used += len(evs)
            touched += int((n > 0).sum())
        ys.append(grp["actual"].to_numpy())
        ps.append(base * k)
        adjs.append(adj * k)
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(adjs), touched, used)


def shipped_k(model: str, horizon: int, cu: float, co: float) -> float:
    """The multiplier the cost board picks for this model/horizon, on all folds."""
    d = pl.read_parquet(results_dir() / "_preds" / f"{model}.parquet") \
        .filter(pl.col("h") == horizon)
    y, p = d["actual"].to_numpy(), d["predicted"].to_numpy()
    ks = np.round(np.arange(0.80, 2.01, 0.02), 2)
    costs = [np.sum(cu * np.maximum(y - p * k, 0) + co * np.maximum(p * k - y, 0)) for k in ks]
    return float(ks[int(np.argmin(costs))])


def report(rows: list[dict], meta: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# Would a promo calendar have helped?\n")
    A(f"*Generated by `dev/event_backtest.py` (mode `{common.MODE}`) from the cached "
      f"predictions in `{results_dir().name}/_preds/` and `data/db/deals.parquet`. "
      f"Nothing was refitted.*\n")
    A(f"Replayed **{meta['deals']:,} deals** across **{meta['skus']:,} SKUs**, using the "
      f"same `events.adjust` the forecast calls. At every origin only the deals "
      f"already created on that date were used.\n")
    A(f"- Demand uplift on deal days: **{meta['uplift']:.3f}x**, volume-weighted over "
      f"{meta['uplift_skus']:,} SKUs, fitted on panel days before {meta['fit_before']}.")
    A(f"- Probability a booked deal runs: **{meta['prob']:.0%}** "
      f"({meta['ran']:,} ran / {meta['resolved']:,} resolved).")
    A(f"- Longest notice on any deal in the history: **{meta['max_notice']:.0f} days** "
      f"(median {meta['med_notice']:.0f}).\n")

    A("## The result\n")
    A("Cost at `{}:{}`. Every column is the same predictions with and without the "
      "override; negative `change` is better.\n".format(*[f"{v:g}" for v in meta["ratio"]]))
    A("| h | origins with a deal | SKU-windows adjusted | WAPE base → events | "
      "cost base → events | change |")
    A("|---|---|---|---|---|---|")
    for r in rows:
        dc = r["cost_ev"] - r["cost_base"]
        pct = 100 * dc / r["cost_base"] if r["cost_base"] else float("nan")
        A(f"| {r['h']} | {r['origins']:,} | {r['touched']:,} ({r['touched_pct']:.1f}%) | "
          f"{r['wape_base']:.4f} → {r['wape_ev']:.4f} | "
          f"{r['cost_base']:.4f} → {r['cost_ev']:.4f} | **{pct:+.2f}%** |")
    A("")

    best = min(rows, key=lambda r: (r["cost_ev"] - r["cost_base"]) / r["cost_base"])
    worst = max(rows, key=lambda r: (r["cost_ev"] - r["cost_base"]) / r["cost_base"])
    A(f"Best horizon: **{best['h']} d** "
      f"({100 * (best['cost_ev'] - best['cost_base']) / best['cost_base']:+.2f}% cost). "
      f"Worst: **{worst['h']} d** "
      f"({100 * (worst['cost_ev'] - worst['cost_base']) / worst['cost_base']:+.2f}%).\n")
    A("Compare every change against the fold-to-fold spread of the board it is computed "
      "from: a change inside that spread, or one whose sign flips across horizons, is "
      "not a gain.\n")

    if meta.get("sweep"):
        A("## Is the fitted uplift just too timid?\n")
        A("The obvious objection to a null result above is that the fitted uplift is too "
          "timid. The same replay under more aggressive assumptions:\n")
        A("| uplift assumed | p | " + " | ".join(f"{h} d" for h in meta["horizons"]) + " |")
        A("|---|---|" + "---|" * len(meta["horizons"]))
        for s in meta["sweep"]:
            cells = " | ".join(f"{100 * v:+.2f}%" for v in s["by_h"])
            A(f"| {s['uplift']:.2f}x | {s['prob']:.0%} | {cells} |")
        A("")
        A("If a bigger uplift makes it worse, the reason is the multiplier: "
          "`forecast.py` already lifts every SKU by a multiplier fitted on backtest "
          "error — and that error was measured over a history that CONTAINS the promo "
          "days. The average promotional response is therefore already inside `k`. "
          "Adding an explicit promo override on top counts it twice, and the damage is "
          "largest at the shortest horizon, where a deal occupies the largest share of "
          "the window.\n")
        A("This is a real constraint on the override layer, not a promo-specific quirk: "
          "**an event that resembles something the model has seen many times before is "
          "already priced into the multiplier.** The events worth entering are the ones "
          "with no precedent in the SKU's history — a launch, a delisting, a wholesale "
          "commitment, a step change in the demand level — not a recurrence of the "
          "ordinary.\n")

    A("## The mechanism, in one line\n")
    A("A deal that lifts that SKU's demand by "
      f"**{meta['uplift']:.2f}x**, pro-rated across the forecast window, is:\n")
    A("| window | a 1-day deal | a 7-day deal |")
    A("|---|---|---|")
    for h in meta["horizons"]:
        A(f"| {h} d | x{1 + (1 / h) * (meta['uplift'] - 1):.4f} | "
          f"x{1 + (7 / h) * (meta['uplift'] - 1):.4f} |")
    A("")
    A("The adjustment is diluted by exactly the ratio of event days to window days, so "
      "the same promo that matters against a short replenishment decision is worth a "
      "rounding error against the much longer window (lead time + review period) an "
      "order is actually placed over.\n")

    A("## What this settles, and what it does not\n")
    A("- **It is an upper bound.** The uplift was fitted on this panel, every booked "
      "deal was entered, and no attempt was made to guess which specific deals get "
      "pulled. A real planner has none of those three advantages, so the real gain is "
      "smaller than the number above.\n")
    A("- **It does not test the override layer's other uses.** Promos are the only "
      "event class with recorded point-in-time history, which is why they are what gets "
      "replayed here. A launch or a delisting cannot be backtested this way at all — "
      "nothing in the extract carries a planned launch date. Those are the events the "
      "layer is actually for; "
      "this file measures the one case where the answer can be checked.\n")
    A("- **The uplift is one number for the catalogue.** Per-SKU promotional response "
      "can vary widely across the SKUs with enough of both kinds of day. Someone "
      "entering a per-SKU figure could do better than this replay does, and "
      "`events.csv` allows it.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default="xgboost_gbdt")
    ap.add_argument("--horizons", type=str, default=",".join(str(h) for h in common.HORIZONS))
    ap.add_argument("--uplift", type=float, default=None,
                    help="demand multiplier on deal days (default: fitted from the panel)")
    ap.add_argument("--probability", type=float, default=None,
                    help="chance a booked deal runs (default: measured)")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the uplift sensitivity sweep")
    ap.add_argument("--tuned", action="store_true")
    args = ap.parse_args()
    common.set_mode("tuned" if args.tuned else "initial")

    cu, co = common.cost_ratio()
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    path = results_dir() / "_preds" / f"{args.model}.parquet"
    if not path.exists():
        raise SystemExit(f"no cached predictions at {path}")
    preds_all = pl.read_parquet(path)
    panel_skus = set(preds_all[C.SKU].unique().to_list())

    print(f"\n{'=' * 92}\nEVENT BACKTEST   [{common.MODE}]   model={args.model}   "
          f"cu:co = {cu:g}:{co:g}\n{'=' * 92}")

    deals = load_deals(panel_skus)
    first_origin = preds_all[C.DATE].min()
    prob = args.probability if args.probability is not None else run_rate(deals)
    if args.uplift is not None:
        uplift, uplift_skus = args.uplift, 0
    else:
        uplift, uplift_skus = fit_uplift(first_origin)

    notice = (deals["start"] - deals["created"]).dt.total_days()
    res = deals.filter(pl.col("st").is_in(list(DID_NOT_RUN | RAN)))
    print(f"  deals         {deals.height:,} joining {deals['sku'].n_unique():,} SKUs")
    print(f"  uplift        {uplift:.3f}x  (fitted on panel before {first_origin}"
          f"{f', {uplift_skus:,} SKUs' if uplift_skus else ', supplied'})")
    print(f"  run rate      {prob:.1%}")
    print(f"  notice        median {notice.median():.0f} d, max {notice.max():.0f} d\n")

    rows = []
    for h in horizons:
        p = preds_all.filter(pl.col("h") == h).sort([C.DATE, C.SKU])
        if not p.height:
            continue
        k = shipped_k(args.model, h, cu, co)
        y, base, adj, touched, used = replay(p, deals, h, uplift, prob, k)
        mb, me = metrics(y, base, cu, co), metrics(y, adj, cu, co)
        row = {"h": h, "k": k, "origins": int(p[C.DATE].n_unique()),
               "rows": len(y), "touched": touched,
               "touched_pct": 100 * touched / max(len(y), 1), "events": used,
               "wape_base": mb["wape"], "wape_ev": me["wape"],
               "cost_base": mb["cost"], "cost_ev": me["cost"],
               "bias_base": mb["bias"], "bias_ev": me["bias"],
               "fill_base": mb["fill"], "fill_ev": me["fill"]}
        rows.append(row)
        print(f"  [{h:>3} d]  k={k:.2f}  {touched:,}/{len(y):,} rows adjusted "
              f"({row['touched_pct']:.1f}%)   cost {mb['cost']:.4f} -> {me['cost']:.4f}"
              f"  ({100 * (me['cost'] - mb['cost']) / mb['cost']:+.2f}%)")

    if not rows:
        raise SystemExit("nothing scored")

    # Sensitivity: is the null result just a timid uplift? Same replay, harder settings.
    sweep = []
    if not args.no_sweep:
        print("\n  sensitivity (same replay, more aggressive assumptions):")
        for u, pr in ((uplift, prob), (2.0, prob), (2.0, 1.0), (3.0, 1.0)):
            by_h = []
            for r in rows:
                p = preds_all.filter(pl.col("h") == r["h"]).sort([C.DATE, C.SKU])
                y, base, adj, _, _ = replay(p, deals, r["h"], u, pr, r["k"])
                mb, me = metrics(y, base, cu, co), metrics(y, adj, cu, co)
                by_h.append((me["cost"] - mb["cost"]) / mb["cost"])
            sweep.append({"uplift": u, "prob": pr, "by_h": by_h})
            cells = "  ".join(f"{h}d {100 * v:+6.2f}%"
                              for h, v in zip([r["h"] for r in rows], by_h))
            print(f"    {u:.2f}x @ {pr:.0%}   {cells}")

    meta = {"deals": deals.height, "skus": int(deals["sku"].n_unique()),
            "uplift": uplift, "uplift_skus": uplift_skus, "prob": prob,
            "ran": int(res.filter(pl.col("st").is_in(list(RAN))).height),
            "resolved": int(res.height), "fit_before": str(first_origin),
            "max_notice": float(notice.max()), "med_notice": float(notice.median()),
            "ratio": (cu, co), "horizons": [r["h"] for r in rows],
            "model": args.model, "sweep": sweep}

    out = LAB / "EVENT_BACKTEST.md"
    out.write_text(report(rows, meta))
    (results_dir() / "_event_backtest.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2, default=str))
    print(f"\n{'=' * 92}\nwrote {out.relative_to(LAB.parent)}\n{'=' * 92}")


if __name__ == "__main__":
    main()
