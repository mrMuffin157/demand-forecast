"""Produce a forward demand forecast -- the one thing in this folder that is not a backtest.

    uv run python dev/forecast.py                  # all horizons, writes a CSV
    uv run python dev/forecast.py --horizons 90    # just the quarter
    uv run python dev/forecast.py --sku-limit 50   # a quick look
    uv run python dev/forecast.py --as-of 2026-06-29   # what it would have said then

Everything else here scores the past. This answers "how many will we sell over the next
30 / 90 / 180 days, starting from today", for every SKU, and writes it to
`dev/forecasts/forecast_<origin>.csv`.

-----------------------------------------------------------------------------------
WHAT IT DECIDES FOR YOU, AND ON WHAT EVIDENCE
-----------------------------------------------------------------------------------
Three choices are made automatically, all from the cached backtest predictions in
`results/_preds/`, none of them from this run's own output:

  which model      the best one AT THAT HORIZON under `PICK_RULE` -- most SKUs within
                   +/-40% of what sold, or lowest inventory cost. It need not be the
                   same model at every horizon, nor the WAPE leader.
  the multiplier   the scalar that minimises inventory cost at `common.CU:CO`. It is
                   normally above 1.0, which is not a fudge: the models aim at the
                   TYPICAL outcome, purchasing under an asymmetric cost needs a higher
                   quantile, and on a right-skewed catalogue the average is higher than
                   the typical outcome too (see docs/FINDINGS.md).
  expected fill    measured on the last backtest fold, where neither the model nor the
                   multiplier had been fitted. It is a real out-of-sample number, not a
                   property of the fit below.

-----------------------------------------------------------------------------------
TWO NUMBERS PER SKU, AND THE DIFFERENCE MATTERS
-----------------------------------------------------------------------------------
  forecast_typical    the model's own output. The best guess at what a normal outcome
                      looks like. Use it for revenue planning, not for buying.
  forecast_planning   that number lifted by the multiplier. What to plan SUPPLY against,
                      given that being short costs `CU` times what being long costs.

`forecast_planning` is deliberately higher than `forecast_typical`. If they were the
same you would be planning to be short about half the time.

-----------------------------------------------------------------------------------
KNOWN FUTURE EVENTS
-----------------------------------------------------------------------------------
A model fitted on history cannot know about a launch, a delisting or a booked wholesale
order. `dev/events.csv` is where a human says so, and this file applies it -- see
`dev/events.py` for the file format and for why an override is applied AFTER the model
rather than fed in as a feature.

Both numbers above are written twice: `forecast_typical` / `forecast_planning` are what
ships and what everything downstream reads, and `forecast_typical_base` /
`forecast_planning_base` are the model's own output with no override. `event_factor`,
`event_delta` and `n_events` decompose the difference. With no events file the two are
identical and the extra columns are 1.0 / 0.0 / 0, which is asserted in
`tests/test_events.py`.

-----------------------------------------------------------------------------------
THE SKUs THAT ARE NOT HERE
-----------------------------------------------------------------------------------
This starts from `open_origins.parquet`, so a SKU whose panel rows stopped is absent
rather than zero -- and "not in the file" reads to whoever orders as "nothing to do".
On a real catalogue that can be a sizeable minority of SKUs. Most are genuinely finished.
Some stopped selling because they ran out of stock, which is the opposite of finished.

`--recover` forecasts the recoverable ones (`dev/coverage.py` classes them `starved` or
`incoming`) from their OWN last real origin. Those rows carry `origin_stale_days > 0` and
their own `origin_date`; `policy.py` routes them to a review file rather than merging
them into the plan, because a months-old origin is a weaker claim than a fresh one and
whoever decides should see which is which.

Events are applied at the PLAN origin for every row, including recovered ones. A
recovered forecast is being read as "what this would sell over the next h days", so an
event is judged against the decision window rather than against the stale origin's own.

-----------------------------------------------------------------------------------
--as-of: THE FORECAST THIS WOULD HAVE ISSUED ON AN EARLIER DATE
-----------------------------------------------------------------------------------
`--as-of D` rebuilds the forecast as of origin D using nothing that was not known on D:
  * it forecasts from each SKU's newest origin on or before D (within the 56-day tail
    `open_origins` keeps, so a quiet SKU is treated exactly as it would have been);
  * it trains only on rows whose target window had CLOSED by D -- origin + h <= D --
    the same horizon-sized gap the backtest folds use;
  * it picks the model and multiplier from backtest predictions whose windows had closed
    by D, not from the whole cached board;
  * events apply only if entered by D (`events.eligible` already enforces that).
The result is written as `forecast_<D>.csv` with `"backfill"` in its JSON, and it
refuses to overwrite a forecast that was actually issued. Use it to fill in a period that
has no forecast from before it began; `track.py` skips these files, because it is the
record of what was ISSUED.

-----------------------------------------------------------------------------------
WHAT THIS IS NOT
-----------------------------------------------------------------------------------
It is a demand forecast, NOT an order quantity. An order is

    forecast_planning  -  stock on hand  -  already on order  +  safety stock

and this file has no visibility of the first two: the panel records what SOLD and
whether an item was out of stock, never how many units were sitting in a warehouse. So
the subtraction happens downstream -- `dev/policy.py` takes on hand / on order per SKU
from the CSV you pass as `--inventory` and does the arithmetic. Handing you a column called
`order_qty` computed here, without inventory, would be worse than not handing you one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import events as EV  # noqa: E402
from common import (  # noqa: E402
    HORIZONS, LAB, OPEN_ORIGINS, load_dataset, results_dir,
)
from demandml import config as C  # noqa: E402

OUT_DIR = LAB / "forecasts"
#: The floor. Never meant to ship; if it is ever the pick, that is flagged, not hidden.
BASELINE = "moving_average"

#: `open_origins` keeps this many days of tail, so a SKU quiet for a few weeks still has
#: an origin to forecast from. `--as-of` looks back the same distance from its date.
AS_OF_TAIL_DAYS = 56


def as_of_origins(pool: pl.DataFrame, as_of: dt.date) -> pl.DataFrame:
    """One row per SKU: its newest origin on or before `as_of`, within the tail."""
    lo = as_of - dt.timedelta(days=AS_OF_TAIL_DAYS)
    return (pool.filter(pl.col(C.DATE).is_between(lo, as_of))
                .sort([C.SKU, C.DATE]).group_by(C.SKU).last().sort(C.SKU))


def training_rows(df: pl.DataFrame, h: int, as_of: dt.date | None) -> pl.DataFrame:
    """Rows with a target -- and, as of a date, only those whose h-day window had closed
    by it. Without that gap a rebuilt forecast would train on outcomes it could not
    have known, and look better than the real one would have."""
    tr = df.filter(pl.col(f"y_cum{h}").is_not_null())
    if as_of is not None:
        tr = tr.filter(pl.col(C.DATE) <= as_of - dt.timedelta(days=h))
    return tr
#: How `choose` picks each horizon's model. "precision": most SKUs whose typical
#: forecast lands within +/-40% of what sold -- the measure a per-SKU forecast is
#: judged by. "cost": lowest inventory cost at `common.CU:CO`. Either way the MULTIPLIER
#: k is still fitted on cost for the picked model, so the planning number keeps its
#: asymmetric target. `dev/pick_rule.py` compares the two; they usually agree.
PICK_RULE = "precision"


def within40(y: np.ndarray, p: np.ndarray) -> float:
    """Share of rows with sales whose forecast is within +/-40% of what sold."""
    m = y > 0
    return float((np.abs(p[m] - y[m]) / y[m] <= 0.40).mean()) if m.any() else 0.0


def capped_rows(tr: pl.DataFrame, cap: int | None, seed: int = 0) -> pl.DataFrame:
    """The training rows capped to `cap` by a seeded RANDOM sample, handed back in DATE order.

    This is the rule `common.split` uses for every backtest fold, so the serving fit
    trains the way the model and its multiplier were chosen. It used to be `tr.tail(cap)`,
    commented "most recent rows" -- but `load_dataset` sorts by SKU, so it kept the
    alphabetically LAST SKUs and left the first ones out of training altogether. That
    turned out to cost little (the trees forecast an unseen SKU from features, not
    identity), while the NEWEST-rows cut the comment intended was measurably worse: it
    throws away the early years' holiday seasons. Date order gives the boosted models the
    latest dates as their early-stopping tail."""
    if cap and tr.height > cap:
        tr = tr.sample(cap, seed=seed, shuffle=True)
    return tr.sort(C.DATE, C.SKU)


KS = np.round(np.arange(0.80, 2.01, 0.02), 2)


def cost(y: np.ndarray, p: np.ndarray, cu: float, co: float) -> float:
    den = np.sum(y)
    return float(np.sum(cu * np.maximum(y - p, 0) + co * np.maximum(p - y, 0)) / den) \
        if den > 0 else float("nan")


def fill(y: np.ndarray, p: np.ndarray) -> float:
    den = np.sum(y)
    return float(1.0 - np.sum(np.maximum(y - p, 0)) / den) if den > 0 else float("nan")


#: Folds held back from selection so the quoted fill rate is measured somewhere neither
#: the model nor the multiplier was chosen. The NEWEST folds, not the oldest: they are
#: the closest analogue to the period a live forecast actually lands in.
HOLDOUT_FOLDS = 1

#: A multiplier should be fitted on no fewer than ~8 folds, because the fold-to-fold bias
#: swing dwarfs its own average (`dev/regime.py`, docs/FINDINGS.md). Choosing a MODEL on
#: fewer is the same mistake one level up. A thin board still runs -- it is just a
#: statement about one season, and says so.
MIN_SELECTION_FOLDS = 8


def choose(horizon: int, cu: float, co: float, as_of: dt.date | None = None) -> dict:
    """Pick the model and the multiplier for one horizon, from cached backtests only.

    Selection uses every fold except the newest `HOLDOUT_FOLDS`, which are kept back for
    the quoted fill rate -- so that number is one the procedure actually earned rather
    than one it was selected on. The multiplier that ships is then re-fitted on ALL
    folds: selection is done, and discarding evidence for the final value would be
    superstition rather than rigour.

    This used to read `fold < 3` and `fold == 3` literally. With many folds on the board
    that selected the model on the two OLDEST windows and discarded everything after them
    -- the regime problem `dev/regime.py` describes, one level up.

    The MODEL is picked by `PICK_RULE` (by default precision: share within +/-40% of what
    sold); the MULTIPLIER is always fitted on cost, for whichever model is picked.

    `folds_won` is the honesty column. It counts how often the chosen model actually wins
    fold by fold in the pick rule's own terms (most precise, or cheapest on its own
    multiplier). A winner that only leads in half the folds has not beaten anything; at
    180 days that is the usual case, so the number is reported rather than hidden.
    """
    d = results_dir() / "_preds"
    # Only the models this run allows: one DEMANDML_MODELS leaves out may still have an
    # old cached backtest on disk, and it must not be able to win from there.
    preds = {p.stem: pl.read_parquet(p).filter(pl.col("h") == horizon)
             for p in sorted(d.glob("*.parquet")) if common.model_allowed(p.stem)}
    if as_of is not None:   # only what had been scored by then -- see `--as-of`
        cut = as_of - dt.timedelta(days=horizon)
        preds = {k: v.filter(pl.col("date") <= cut) for k, v in preds.items()}
    preds = {k: v for k, v in preds.items() if v.height}
    if not preds:
        raise SystemExit(f"no cached predictions for h={horizon} in {d}")

    folds = sorted({int(f) for df in preds.values()
                    for f in df.get_column("fold").unique().to_list()})
    if len(folds) <= HOLDOUT_FOLDS:
        raise SystemExit(
            f"h={horizon}: {len(folds)} fold(s) cached, need more than {HOLDOUT_FOLDS} "
            f"so that something is left to select on -- re-run the models with --folds")
    sel, holdout = folds[:-HOLDOUT_FOLDS], folds[-HOLDOUT_FOLDS:]
    if len(sel) < MIN_SELECTION_FOLDS:
        print(f"  ! h={horizon}: selecting on {len(sel)} folds, fewer than the "
              f"{MIN_SELECTION_FOLDS} recommended -- treat the pick as provisional")

    # Each candidate on its own best multiplier, so the comparison is between models
    # rather than between accidents of level. Ranking them unscaled instead just rewards
    # whichever happens to aim highest, which the multiplier supplies for free.
    picks: dict[str, tuple[float, float]] = {}
    prec: dict[str, tuple[float, float]] = {}
    for name, df in preds.items():
        tr = df.filter(pl.col("fold").is_in(sel))
        y, p = tr["actual"].to_numpy(), tr["predicted"].to_numpy()
        costs = [cost(y, p * kk, cu, co) for kk in KS]
        i = int(np.argmin(costs))
        picks[name] = (float(KS[i]), float(costs[i]))
        prec[name] = (within40(y, p), abs(float((p.sum() - y.sum()) / y.sum())) if y.sum() else 0.0)
    cost_pick = min(picks, key=lambda n: picks[n][1])
    if PICK_RULE == "precision":
        # Most SKUs within +/-40% of what sold on the TYPICAL forecast (raw predictions),
        # ties to the less biased. 4 decimals so a
        # difference in the noise does not decide it on its own.
        name = max(prec, key=lambda n: (round(prec[n][0], 4), -prec[n][1]))
    else:
        name = cost_pick
    k_sel = picks[name][0]
    # The sanity check: how much cheaper the pick is than the moving-average floor, each
    # on its own multiplier. When the floor itself wins, the trees have stopped learning
    # anything a trailing average does not know -- usually an input gone wrong -- and
    # the forecast JSON records it (`baseline_wins`) so a report can say so at the top.
    base = picks.get(BASELINE)
    vs_base = (1.0 - picks[name][1] / base[1]) if base and base[1] > 0 else None

    won = 0
    for f in sel:
        per = {}
        for n, df in preds.items():
            s = df.filter(pl.col("fold") == f)
            if s.height:
                y_f, p_f = s["actual"].to_numpy(), s["predicted"].to_numpy()
                # "won" in the pick rule's own terms: most precise, or cheapest
                per[n] = (-within40(y_f, p_f) if PICK_RULE == "precision"
                          else cost(y_f, p_f * picks[n][0], cu, co))
        if per and min(per, key=lambda n: per[n]) == name:
            won += 1

    te = preds[name].filter(pl.col("fold").is_in(holdout))
    y, p = te["actual"].to_numpy(), te["predicted"].to_numpy()

    allf = preds[name]
    ya, pa = allf["actual"].to_numpy(), allf["predicted"].to_numpy()
    k = float(KS[int(np.argmin([cost(ya, pa * kk, cu, co) for kk in KS]))])

    return {
        "horizon": horizon, "model": name, "k": k, "k_holdout": k_sel,
        "fill_oos": fill(y, p * k_sel), "fill_unlifted": fill(y, p),
        "cost_oos": cost(y, p * k_sel, cu, co),
        "wape_oos": float(np.sum(np.abs(y - p)) / np.sum(np.abs(y))),
        "folds_selected": len(sel), "folds_won": won,
        "folds_holdout": [int(f) for f in holdout],
        "baseline_wins": name == BASELINE,
        "cheaper_than_baseline": vs_base,
        "pick_rule": PICK_RULE,
        "within40_sel": prec[name][0],
        "cost_pick": cost_pick,
        "backtest_finished": _backtest_finished(name),
    }


def _backtest_finished(model: str) -> str | None:
    """When the cached backtest of `model` was fitted (`run.finished` in its results JSON).
    Recorded so a report can flag a stale board: a scheduled retrain that silently never
    runs raises no error, and a weekly forecast would keep using the old board without a
    word."""
    try:
        run = json.loads((results_dir() / f"{model}.json").read_text()).get("run", {})
    except (OSError, ValueError):
        return None
    return run.get("finished")


def _backtest_pool(model: str) -> list[str]:
    """Which stores the cached backtest of `model` trained with (`run.pooled_with` in its
    results JSON; empty for a run that predates pooling or did not pool)."""
    try:
        run = json.loads((results_dir() / f"{model}.json").read_text()).get("run", {})
    except (OSError, ValueError):
        return []
    return sorted(run.get("pooled_with") or [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizons", type=str, default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--sku-limit", type=int, default=None)
    ap.add_argument("--max-train-rows", type=int, default=300_000,
                    help="cap on training rows, 0 = no cap (default 300,000)")
    ap.add_argument("--out", type=str, default=None, help="CSV path (default dev/forecasts/)")
    ap.add_argument("--tuned", action="store_true", help="use the tuned half's models")
    ap.add_argument("--events", type=str, default=None,
                    help="known-future events CSV (default dev/events.csv)")
    ap.add_argument("--no-events", action="store_true",
                    help="ignore the events file entirely")
    ap.add_argument("--max-stale", type=int, default=365,
                    help="refuse to recover a SKU whose last usable origin is older than "
                         "this many days (default 365)")
    ap.add_argument("--as-of", type=str, default=None,
                    help="rebuild the forecast as of this origin date (YYYY-MM-DD) from "
                         "only what was known then -- see the docstring")
    ap.add_argument("--recover", action="store_true",
                    help="also forecast SKUs missing from open_origins that dev/coverage.py "
                         "classes as starved or incoming, from their own last real origin")
    args = ap.parse_args()
    common.set_mode("tuned" if args.tuned else "initial")

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    cu, co = common.cost_ratio()

    if not OPEN_ORIGINS.exists():
        raise SystemExit(
            f"{OPEN_ORIGINS} does not exist -- it holds the recent origins whose future\n"
            f"has not happened yet, which are the only rows a forward forecast can start\n"
            f"from. Rebuild it:  uv run python model_lab/prepare_data.py")

    print(f"\n{'=' * 86}\nFORWARD FORECAST   [{common.MODE}]   cu:co = {cu:g}:{co:g}\n{'=' * 86}")
    # Every column any model may ask for, `xf_*` included; each fit below takes only its
    # own model's (`common.model_features`), as `run_model` did in the backtest.
    df, feats = load_dataset(args.sku_limit, extended=True)

    open_df = pl.read_parquet(OPEN_ORIGINS)
    # Pooled training (DEMANDML_POOL): other stores' rows train the model; only this
    # store's origins are forecast. See common.py, "Pooled training across stores".
    pool = common.load_pool(feats) if common.pool_spec() else {}
    if common.pool_spec():
        open_df = open_df.with_columns(common.store_marker(common.current_store()))
    if args.sku_limit:
        open_df = open_df.filter(pl.col(C.SKU).is_in(df.get_column(C.SKU).unique()))
    # One row per SKU: its most recent origin. A SKU that has gone quiet may have no row
    # at the newest date, so this falls back through the short tail the file keeps.
    as_of = dt.date.fromisoformat(args.as_of) if args.as_of else None
    if as_of is not None:
        if args.recover:
            raise SystemExit("--as-of and --recover cannot be combined: a recovered SKU is "
                             "forecast from its own stale origin, which is not 'as of' D")
        cols = [c for c in open_df.columns if c in df.columns]
        # Not `pool`: that name is the store dict the fits below train with.
        origins = (pl.concat([df.select(cols), open_df.select(cols)], how="vertical_relaxed")
                     .unique([C.SKU, C.DATE], keep="first"))
        open_df = as_of_origins(origins, as_of)
        if open_df.is_empty():
            raise SystemExit(f"no origins in the {AS_OF_TAIL_DAYS} days up to {as_of} -- "
                             f"the dataset does not reach back that far")
        origin = open_df.get_column(C.DATE).max()
        print(f"  AS OF {as_of}: {len(open_df):,} SKUs, newest origin {origin}")
    else:
        open_df = (open_df.sort([C.SKU, C.DATE]).group_by(C.SKU).last()
                   .sort(C.SKU))
        origin = open_df.get_column(C.DATE).max()
        print(f"  open origins: {len(open_df):,} SKUs, newest origin {origin}")

    # A SKU whose panel rows stopped has no open origin and so gets no forecast and no
    # order line -- it is absent rather than zero, which reads as "nothing to do". Most
    # of those are genuinely finished; some ran out of stock and stopped selling because
    # nobody could buy them, which is the opposite of finished. `coverage.py` separates
    # the two using daily on-hand and inbound, and `--recover` forecasts the recoverable
    # ones from their OWN last real origin, however stale that is.
    n_recovered = 0
    recovered_skus: set[str] = set()
    if args.recover:
        import coverage as COV   # noqa: PLC0415 -- optional path, keep the import local
        cls = COV.classify(origin, planned=set(open_df.get_column(C.SKU).to_list()))
        want = cls.filter(pl.col("klass").is_in(["starved", "incoming"]))
        want_skus = want.get_column(C.SKU).to_list()
        if want_skus:
            rec = (df.filter(pl.col(C.SKU).is_in(want_skus))
                   .sort([C.SKU, C.DATE]).group_by(C.SKU).last().sort(C.SKU))
            # A starved SKU is heavily out of stock, so `prepare_data.py` blanked its
            # recent targets and its last row in the TRAINING file can be years old --
            # the last time it had a clean window. Predicting from features that stale is
            # not a forecast, it is an echo, so those are refused and reported instead.
            rec = rec.with_columns(
                _stale=(pl.lit(origin) - pl.col(C.DATE)).dt.total_days())
            too_old = rec.filter(pl.col("_stale") > args.max_stale)
            rec = rec.filter(pl.col("_stale") <= args.max_stale).drop("_stale")
            if too_old.height:
                print(f"  recovered:    {too_old.height} SKU(s) REFUSED -- last usable "
                      f"origin older than {args.max_stale} d "
                      f"({', '.join(too_old.get_column(C.SKU).to_list()[:5])}"
                      f"{'...' if too_old.height > 5 else ''}). They are in "
                      f"dev/COVERAGE.md; decide them by hand.")
            if rec.height:
                open_df = pl.concat([open_df, rec.select(open_df.columns)],
                                    how="vertical").sort(C.SKU)
                n_recovered = rec.height
                recovered_skus = set(rec.get_column(C.SKU).to_list())
                stale = [(origin - d).days for d in rec.get_column(C.DATE).to_list()]
                kept = cls.filter(pl.col(C.SKU).is_in(list(recovered_skus)))
                print(f"  recovered:    {n_recovered} SKU(s) missing from open_origins "
                      f"({kept.filter(pl.col('klass') == 'starved').height} starved, "
                      f"{kept.filter(pl.col('klass') == 'incoming').height} incoming); "
                      f"origins {min(stale)}-{max(stale)} days stale")
        if not n_recovered:
            print("  recovered:    nothing to recover -- every SKU with demand is planned")
    if as_of is None:
        print(f"  training on every origin up to {df.get_column(C.DATE).max()}\n")
    else:
        print(f"  training only on windows closed by {as_of} (origin + h <= {as_of})\n")
        shipped = OUT_DIR / f"forecast_{origin}.json"
        if shipped.exists() and "backfill" not in json.loads(shipped.read_text()):
            raise SystemExit(f"{shipped.name} is a forecast that was ISSUED -- a rebuild never "
                             f"overwrites one")

    # Known-future overrides. Loaded before anything is fitted so a malformed file fails
    # in a second rather than after an hour of training.
    ev_path = Path(args.events) if args.events else EV.EVENTS
    try:
        all_events = [] if args.no_events else EV.load(ev_path)
    except EV.EventError as exc:
        raise SystemExit(f"\n{exc}\n")
    ev_lines = EV.lines_for(open_df)
    ev_skus = open_df.get_column(C.SKU)
    if all_events:
        note = "" if ev_lines is not None else "   (no product-line column: `line:` scopes cannot match)"
        print(f"  events: {len(all_events)} in {ev_path.name}{note}")
    elif not args.no_events:
        print(f"  events: none ({ev_path.name} absent or empty) -- "
              f"forecasts are the model's own")

    out_rows: list[pl.DataFrame] = []
    chosen: list[dict] = []
    for h in horizons:
        pick = choose(h, cu, co, as_of)
        chosen.append(pick)
        holdout = ", ".join(str(f) for f in pick["folds_holdout"])
        won, of = pick["folds_won"], pick["folds_selected"]
        best = "most precise" if pick["pick_rule"] == "precision" else "cheapest"
        print(f"[{h} days]  model={pick['model']}  multiplier={pick['k']:.2f}"
              f"   (picked on {pick['pick_rule']}: {pick['within40_sel']:.1%} within ±40% on "
              f"{of} folds, {best} in {won})")
        if pick["cost_pick"] != pick["model"]:
            print(f"  (on inventory cost alone the pick would be {pick['cost_pick']})")
        print(f"  backtest (fold {holdout}, unseen): wape {pick['wape_oos']:.3f}   "
              f"fill {pick['fill_unlifted']:.3f} -> {pick['fill_oos']:.3f} once lifted")
        if pick["baseline_wins"]:
            print(f"  ! BASELINE WINS: {BASELINE} is the pick at {h} days -- "
                  f"the learned models are no better than a trailing average; check the "
                  f"inputs (datacheck) before trusting this forecast")
        elif pick["cheaper_than_baseline"] is not None:
            print(f"  {pick['cheaper_than_baseline']:.1%} cheaper than {BASELINE}")
        if won * 2 <= of:
            print(f"  ! this model is {best} in only {won} of {of} folds -- the pick "
                  f"is inside the noise at this horizon, not a settled ranking")

        model = importlib.import_module(pick["model"]).MODEL
        ycol = f"y_cum{h}"
        cap = args.max_train_rows or None
        tr = capped_rows(training_rows(df, h, as_of), cap)
        pick["pooled_with"] = sorted(pool)
        if pool and feats and model.uses_features:
            # The same as-of cut as `training_rows`: a window must have closed by then.
            # The other stores' rows are capped the backtest's way too (a seeded random
            # sample, as `run_model` pools them), not newest-first: `capped_rows`.
            end = (as_of - dt.timedelta(days=h)) if as_of is not None else dt.date.max
            tr = common.with_pool(tr, common.pool_training_rows(pool, h, end, cap, seed=0))
            pooled_runs = _backtest_pool(pick["model"])
            if pooled_runs != sorted(pool):
                print(f"  ! training pooled with {sorted(pool)} but {pick['model']}'s "
                      f"cached backtest pooled with {pooled_runs} -- model and multiplier "
                      f"were chosen on the other setup until the next retrain")

        est = model.build(h)
        mfeats = common.model_features(model, feats)
        X = tr.select(mfeats) if mfeats and model.uses_features else tr
        y = tr.get_column(ycol).to_numpy()
        w = tr.get_column("sample_weight").to_numpy() if "sample_weight" in tr.columns else None
        t0 = time.time()
        if model.needs_frame:
            est.fit(X, y, w, frame=tr)
        else:
            est.fit(X, y, w)
        Xf = open_df.select(mfeats) if mfeats and model.uses_features else open_df
        raw = np.clip(np.nan_to_num(np.asarray(est.predict(Xf), dtype=np.float64).ravel()),
                      0.0, None)
        pick.update(common.lift_stats(est))
        print(f"  fitted on {len(tr):,} rows in {time.time() - t0:.0f}s, "
              f"predicted {len(raw):,} SKUs{common.lift_note(common.lift_stats(est))}\n")

        # Known-future overrides, applied to the model's output. `adj` is what ships;
        # `raw` is kept alongside it so the difference stays auditable and `track.py`
        # can score whether the human input helped.
        adj, factor, delta, n_ev = EV.adjust(raw, ev_skus, origin, h, all_events, ev_lines)
        pick["events_applied"] = int((n_ev > 0).sum())
        pick["events_units_delta"] = float(adj.sum() - raw.sum())
        if pick["events_applied"]:
            print(f"  events: {pick['events_applied']:,} SKUs adjusted, "
                  f"{adj.sum() - raw.sum():+,.0f} units "
                  f"({100 * (adj.sum() / max(raw.sum(), 1) - 1):+.1f}%)\n")

        out_rows.append(open_df.select(
            pl.col(C.SKU),
            pl.col(C.DATE).alias("origin_date"),
            pl.lit(h).alias("horizon_days"),
            pl.lit(pick["model"]).alias("model"),
            pl.Series("forecast_typical", np.round(adj, 1)),
            pl.Series("forecast_planning", np.round(adj * pick["k"], 1)),
            pl.lit(round(pick["k"], 3)).alias("multiplier"),
            pl.lit(round(pick["fill_oos"], 3)).alias("expected_fill_rate"),
            pl.Series("forecast_typical_base", np.round(raw, 1)),
            pl.Series("forecast_planning_base", np.round(raw * pick["k"], 1)),
            pl.Series("event_factor", np.round(factor, 4)),
            pl.Series("event_delta", np.round(delta, 1)),
            pl.Series("n_events", n_ev),
            # 0 for an ordinary row. Positive means the forecast was made from an origin
            # that old, because the SKU stopped appearing in the panel -- the answer is
            # "what would it have sold", which is the right question for a reorder and
            # the wrong one for revenue planning.
            (pl.lit(origin) - pl.col(C.DATE)).dt.total_days().alias("origin_stale_days"),
            # True only for a SKU brought back by --recover. NOT the same as a stale
            # origin: open_origins keeps a short tail, so an ordinary quiet SKU can
            # legitimately carry an origin a few weeks old and still belong in the plan.
            pl.col(C.SKU).is_in(list(recovered_skus)).alias("recovered"),
        ))

    res = pl.concat(out_rows).sort([C.SKU, "horizon_days"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else OUT_DIR / f"forecast_{origin}.csv"
    res.write_csv(path)

    print("=" * 86)
    print(f"{'h':>5}  {'model':<20}{'x':>6}{'typical':>14}{'planning':>14}"
          f"{'extra':>8}{'fill':>8}")
    for pick in chosen:
        s = res.filter(pl.col("horizon_days") == pick["horizon"])
        t, p = s["forecast_typical"].sum(), s["forecast_planning"].sum()
        print(f"{pick['horizon']:>5}  {pick['model']:<20}{pick['k']:>6.2f}"
              f"{t:>14,.0f}{p:>14,.0f}{100 * (p / t - 1):>7.0f}%"
              f"{pick['fill_oos']:>8.3f}")
    print("=" * 86)
    shown = path.resolve().relative_to(LAB.parent) if path.resolve().is_relative_to(LAB.parent) \
        else path
    print(f"\nforecast -> {shown}  ({len(res):,} rows)")
    print("\n`forecast_planning` is a DEMAND figure, not an order. Subtract stock on hand\n"
          "and units already on order before you buy -- dev/policy.py does that, from\n"
          "the stock CSV you pass as --inventory.")

    # Next to the CSV: the default path is dev/forecasts/forecast_<origin>.json as before,
    # and an `--out` run no longer drops a sidecar among the issued forecasts.
    path.with_suffix(".json").write_text(json.dumps({
        "origin": str(origin), "mode": common.MODE, "cu": cu, "co": co,
        "target_quantile": common.target_quantile(),
        "dataset_last_origin": str(df.get_column(C.DATE).max()),
        **({"backfill": {
            "as_of": str(as_of), "made_on": str(dt.date.today()),
            "note": "rebuilt after the fact from data known on as_of only: training "
                    "windows closed by then, model choice from backtests closed by then. "
                    "Not a forecast that was issued."}} if as_of is not None else {}),
        "skus": int(res.get_column(C.SKU).n_unique()),
        "events_file": str(ev_path) if not args.no_events else None,
        "events_loaded": len(all_events),
        "events": [e.describe() for e in all_events],
        "horizons": chosen,
    }, indent=2))

    # The events report is regenerated here rather than only by `events.py`, so the
    # markdown always describes the origin the current forecast was actually made from.
    # Not for an `--as-of` rebuild: in a scheduled chain that may run AFTER the real
    # forecast and would leave the report describing a months-old origin.
    if as_of is None:
        (LAB / "EVENTS.md").write_text(
            EV.report(all_events, origin, tuple(horizons), ev_path))


if __name__ == "__main__":
    main()
