"""Turn a demand forecast into an order quantity. The last mile, and the only file here
that produces a number you can actually send to a factory.

    uv run python dev/policy.py --inventory data/inventory/position_sample.csv
    uv run python dev/policy.py --inventory <your stock CSV> \
        --lead-time-file <sku,lead_time_days CSV> --review 30

`dev/forecast.py` answers "how many will we sell in the next 30 / 90 / 180 days" and
stops there, deliberately, because it has no visibility of stock. This file closes that
gap using three inputs the sales history does not contain:

    lead time     how long between placing the order and having it sellable
    review period how long until you can place the NEXT order
    inventory     units on hand and units already on order, per SKU

All three are supplied by the caller. Inventory is `--inventory`, a CSV of
`sku,on_hand,on_order` (`scripts/make_sample_data.py` writes one for the sample data,
`data/inventory/position_sample.csv`; in real use it is a snapshot of your own stock
system). Lead time is `--lead-time` (one number for every SKU), `--lead-time-file` (a
`sku,lead_time_days` CSV, per SKU), or, with neither, the default in assumptions.toml --
which is an example value, and the script says so loudly. The review period is
`--review`, defaulting to assumptions.toml as well; it is a policy choice, not an
observable.

-----------------------------------------------------------------------------------
WHY 30 / 90 / 180 ARE THE WRONG NUMBERS TO BUY AGAINST
-----------------------------------------------------------------------------------
Those three horizons were chosen because they are round, and every backtest in this
folder is built on them. None of them is your exposure. What you are exposed to is

    coverage window  =  lead time  +  review period

-- everything that can be demanded between committing to this order and being able to
fix a mistake with the next one. The model boards cannot see that error, because they
score the horizons, not the decision.

So this file interpolates the cumulative forecast onto YOUR window. Cumulative demand at
0, 30, 90 and 180 days gives four points on a monotone curve; the value at any window in
between is a straight-line read off that curve. Beyond 180 days it extrapolates at the
90-to-180 daily rate and says so, loudly -- no model in this project is trained past
six months, and the extension's cost is measured only after the fact
(`dev/window_backtest.py`, `dev/longhorizon.py`).

THE WINDOW IS PER SKU, AND IT MAY BE PAST THE EVIDENCE
------------------------------------------------------
Lead time can differ per SKU (`--lead-time-file`), so exposure is per SKU: a product
that takes months longer to arrive is exposed for longer, and one catalogue-wide window
under-buys the slow SKU and over-buys the fast one. An earlier version computed a single
window from the scalar `--lead-time` and silently ignored `--lead-time-file` when doing
it, so per-SKU lead times reached the output CSV but never reached the demand every
order was sized from. `tests/test_policy.py::test_each_sku_gets_its_own_coverage_window`
pins the fix.

The uncomfortable consequence for long supply chains: once lead time plus review exceeds
the longest horizon (180 d), every order quantity rests partly on the flat-rate
extension, which no model was trained on. The script reports how many SKUs that affects.
`dev/window_backtest.py` and `dev/longhorizon.py` measure that error; read them before
treating any of these quantities as precise.

-----------------------------------------------------------------------------------
THE SERVICE LEVEL IS PER SKU, AND IT COMES FROM PRICES
-----------------------------------------------------------------------------------
`dev/forecast.py` lifts every SKU by one global multiplier. Here each SKU gets its own,
derived from its own margin and unit cost via `dev/economics.py`: a high-margin, cheap-
to-hold product should be covered further into the tail than a thin-margin, expensive
one, and the global multiplier cannot express that. `--flat-ratio` turns this off and
reverts to one service level for the catalogue, so the difference stays visible.

The quantile-to-multiplier map is empirical, not assumed. It is the demand-weighted
distribution of `actual / predicted` over the backtest folds: to cover the 85th
percentile you multiply by whatever factor covered 85% of realised demand historically.
At the global ratio it reproduces the cost-board multiplier to within the grid step,
which is the check that the two approaches agree where they overlap.

Safety stock is NOT a separate term here. Ordering the q-th quantile of demand over the
coverage window IS the safety stock decision -- the buffer is the distance between that
quantile and the middle, and it is already inside `demand_planning`. Adding a further
safety-stock formula on top would double-count it.

-----------------------------------------------------------------------------------
WHAT IT REFUSES TO DO
-----------------------------------------------------------------------------------
Without an inventory file it does not invent one. A SKU with no stock record gets
`order_qty` = null, not zero and not a full window of demand -- silently ordering a
whole coverage window for a product with a warehouse full of it is the single most
expensive mistake this file could make. `--assume-zero-stock` overrides that for a new
catalogue where the assumption is genuinely true.

-----------------------------------------------------------------------------------
THE PLAN, AND THE REVIEW PILE NEXT TO IT
-----------------------------------------------------------------------------------
`forecast.py --recover` brings back SKUs whose panel rows stopped but which ran out of
stock rather than dying -- they carry `origin_stale_days > 0`, meaning the forecast was
made from an origin that many days old. Those do NOT go into the main plan. A forecast
anchored three months back is a weaker claim than one anchored last week, and merging the
two would hide which is which behind a single column of quantities.

They go to `<plan>_review.csv` with quantities computed exactly the same way, so a
human is deciding rather than discovering. `<plan>_not_covered.csv` is the output of
`dev/coverage.py`: every catalogue SKU with no line, classified by why -- starved,
incoming, dormant or discontinued -- instead of a simple recent-sales filter, which
cannot see a SKU that has been starved for longer than its window.
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
import economics as E  # noqa: E402
from common import LAB, results_dir  # noqa: E402
from demandml import assumptions as _assumptions  # noqa: E402
from demandml import config as C  # noqa: E402

FORECASTS = LAB / "forecasts"
OUT_DIR = LAB / "orders"
PANEL = C.PROCESSED / "panel_daily.parquet"

#: A SKU that has sold within this many days of the origin but has no forecast is
#: reported as a coverage gap. 60 days is deliberately generous: the point is to catch
#: anything a planner might still consider live, not to guess which ones are discontinued.
RECENT_SALE_DAYS = 60

#: Placeholder lead time, in days, used when `--lead-time` is not given -- as the scalar
#: and for any SKU missing from `--lead-time-file`. It comes from assumptions.toml, is an
#: EXAMPLE value and is ALMOST CERTAINLY NOT YOURS. Nothing in this project measures
#: lead time; the script shouts when it falls back to this.
DEFAULT_LEAD_TIME = _assumptions.LEAD_TIME_DAYS

#: Days until the next order can be placed (assumptions.toml). A policy choice.
DEFAULT_REVIEW = _assumptions.REVIEW_DAYS

#: The knots of the cumulative-demand curve, in days. Must match `common.HORIZONS`.
KNOTS = (0,) + tuple(common.HORIZONS)


# --------------------------------------------------------------------------------------
# The cumulative curve
# --------------------------------------------------------------------------------------

def interpolate_cumulative(cum: np.ndarray,
                           window: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative demand at `window` days, from its values at KNOTS[1:].

    `cum` is (n_skus, len(HORIZONS)). `window` is either one number for the catalogue or
    ONE NUMBER PER SKU -- the latter is the real case, because lead time can differ per
    SKU and a SKU that takes months longer to arrive is exposed for longer. Returns the demand column and a per-SKU regime array, because extrapolating past
    the longest horizon is a materially weaker claim than interpolating inside it and the
    caller has to be able to say which SKUs are which.

    The curve is forced monotone first. It has to be: `forecast.py` picks the best model
    PER HORIZON, so the 180-day number can come from a different model than the 90-day
    one and nothing makes them agree. A cumulative total that falls as the window grows
    is not a forecast, it is two forecasts disagreeing, and interpolating between them
    would produce negative demand for a stretch of days.
    """
    cum = np.maximum.accumulate(np.clip(np.asarray(cum, dtype=np.float64), 0.0, None),
                                axis=1)
    knots = np.asarray(KNOTS, dtype=np.float64)
    full = np.column_stack([np.zeros(len(cum)), cum])       # cumulative at day 0 is 0
    w = np.broadcast_to(np.asarray(window, dtype=np.float64), (len(cum),)).copy()

    out = np.empty(len(cum), dtype=np.float64)
    inside = w <= knots[-1]

    # Inside the knots: ordinary linear interpolation on that SKU's own curve.
    for i in np.flatnonzero(inside):
        out[i] = np.interp(w[i], knots, full[i])

    # Past the last knot: continue at the daily rate of the final segment. Flat, not
    # growing -- a trend fitted to two points 90 days apart and run forwards is not
    # something this data supports.
    beyond = ~inside
    if beyond.any():
        rate = (full[beyond, -1] - full[beyond, -2]) / (knots[-1] - knots[-2])
        out[beyond] = full[beyond, -1] + rate * (w[beyond] - knots[-1])

    regime = np.where(inside, "interpolated", "extrapolated")
    return out, regime


# --------------------------------------------------------------------------------------
# The quantile -> multiplier map, learned from realised errors
# --------------------------------------------------------------------------------------

def _wquantile(x: np.ndarray, w: np.ndarray, qs: np.ndarray) -> np.ndarray:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return np.ones_like(qs)
    return np.interp(qs, (c - 0.5 * w) / c[-1], x)


def ratio_curve(model: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """The distribution of `actual / predicted` on the backtest, weighted by PREDICTED.

    The weighting is not a taste question, it falls out of the cost function. Minimising

        sum_i  cu * max(y_i - k*p_i, 0)  +  co * max(k*p_i - y_i, 0)

    over a single multiplier k gives the first-order condition

        cu * SUM{p_i : r_i > k}  =  co * SUM{p_i : r_i < k},     r_i = y_i / p_i

    i.e. k is the cu/(cu+co) quantile of the ratio weighted by p, the PREDICTION. Not by
    the actual, and not unweighted -- both of those answer a different question and are
    visibly wrong in practice: the prediction-weighted quantile agrees with the cost
    board's grid optimum to within its grid step, while the actual-weighted one lands
    noticeably higher and would over-order the whole catalogue by several percent.

    That agreement is the whole justification for this curve. It reproduces the cost
    board's answer at the cost board's quantile, which is what licenses using it to
    interpolate multipliers at every OTHER quantile, where there is no cost-board number
    to compare against.

    Returns (probs, multipliers) suitable for np.interp.
    """
    path = results_dir() / "_preds" / f"{model}.parquet"
    if not path.exists():
        raise SystemExit(f"no cached predictions at {path} -- run the model first")
    d = pl.read_parquet(path).filter((pl.col("h") == horizon)
                                     & (pl.col("predicted") > 0)
                                     & (pl.col("actual") >= 0))
    if not d.height:
        raise SystemExit(f"no usable cached rows for {model} at h={horizon}")
    r = (d["actual"] / d["predicted"]).to_numpy()
    w = d["predicted"].to_numpy()
    ok = np.isfinite(r) & np.isfinite(w) & (w >= 0)
    probs = np.linspace(0.01, 0.99, 99)
    return probs, _wquantile(r[ok], w[ok], probs)


def multiplier_for(q: np.ndarray, probs: np.ndarray, mults: np.ndarray) -> np.ndarray:
    """Read each SKU's multiplier off the empirical curve. Never below 1/never absurd:
    clipped to the same band the cost board searched, so one pathological ratio cannot
    produce a 6x order."""
    return np.clip(np.interp(np.asarray(q, dtype=np.float64), probs, mults), 0.80, 2.00)


# --------------------------------------------------------------------------------------
# Inputs the sales data does not contain
# --------------------------------------------------------------------------------------

def load_lead_times(path: Path | None, default: int, skus: list[str]) -> pl.DataFrame:
    """Per-SKU lead times from a CSV (`sku,lead_time_days`), falling back to `default`.

    A per-SKU file matters more than it looks: an air-freighted accessory and a sea-
    freighted pallet in the same catalogue differ by two months of exposure, and one
    global number prices both wrong in opposite directions.
    """
    base = pl.DataFrame({"sku": skus}).with_columns(
        lead_time_days=pl.lit(default, dtype=pl.Int64), lead_time_source=pl.lit("default"))
    if path is None:
        return base
    if not path.exists():
        raise SystemExit(f"--lead-time-file {path} does not exist")
    lt = pl.read_csv(path)
    need = {"sku", "lead_time_days"}
    if not need.issubset(lt.columns):
        raise SystemExit(f"{path} must have columns {sorted(need)}, has {lt.columns}")
    lt = (lt.select(pl.col("sku").cast(pl.String),
                    pl.col("lead_time_days").cast(pl.Int64, strict=False))
          .filter(pl.col("lead_time_days") > 0).unique(subset=["sku"], keep="last"))
    return (base.join(lt, on="sku", how="left", suffix="_f")
            .with_columns(
                lead_time_source=pl.when(pl.col("lead_time_days_f").is_not_null())
                .then(pl.lit("file")).otherwise(pl.lit("default")),
                lead_time_days=pl.coalesce("lead_time_days_f", "lead_time_days"))
            .drop("lead_time_days_f"))


def load_inventory(path: Path | None, skus: list[str], assume_zero: bool) -> pl.DataFrame:
    """Units on hand and on order per SKU. `stock_known` is the column that matters."""
    base = pl.DataFrame({"sku": skus})
    if path is None:
        return base.with_columns(on_hand=pl.lit(0.0), on_order=pl.lit(0.0),
                                 stock_known=pl.lit(bool(assume_zero)))
    if not path.exists():
        raise SystemExit(
            f"--inventory {path} does not exist.\n"
            f"Shape it like data/inventory/position_sample.csv: sku,on_hand,on_order")
    inv = pl.read_csv(path)
    if "sku" not in inv.columns:
        raise SystemExit(f"{path} must have a `sku` column, has {inv.columns}")
    for c in ("on_hand", "on_order"):
        if c not in inv.columns:
            inv = inv.with_columns(pl.lit(0.0).alias(c))
    inv = (inv.select(pl.col("sku").cast(pl.String),
                      pl.col("on_hand").cast(pl.Float64, strict=False).fill_null(0.0),
                      pl.col("on_order").cast(pl.Float64, strict=False).fill_null(0.0))
           .group_by("sku").agg(pl.col("on_hand").sum(), pl.col("on_order").sum())
           .with_columns(stock_known=pl.lit(True)))
    return (base.join(inv, on="sku", how="left")
            .with_columns(
                stock_known=pl.col("stock_known").fill_null(bool(assume_zero)),
                on_hand=pl.col("on_hand").fill_null(0.0),
                on_order=pl.col("on_order").fill_null(0.0)))


def coverage_gap(planned: list[str], origin, days: int = RECENT_SALE_DAYS) -> pl.DataFrame:
    """Every catalogue SKU with no order line, classified by WHY it has none.

    This exists because the omission is SILENT and asymmetric. `open_origins.parquet`
    holds only SKUs with a recent enough origin to forecast from, so a product that went
    quiet drops out of the plan entirely -- it does not appear with a zero, it simply is
    not there. Someone reading the order plan has no way to tell that the catalogue has
    more SKUs in it than the plan does, and "not in the file" reads as "nothing to do"
    rather than "not answered".

    A plain 60-day sales filter cannot draw the distinction that matters: a SKU starved
    of stock for longer than 60 days looks identical to one that was discontinued, and it
    is the one a reorder is most urgent for -- e.g. a SKU that sold steadily while mostly
    out of stock and then vanished from the panel.

    `dev/coverage.py` draws it, from daily on-hand and inbound in `data/db/`. This
    delegates rather than reimplementing, so the order plan and `dev/COVERAGE.md` can
    never disagree about which SKUs are missing.
    """
    try:
        import coverage as COV   # noqa: PLC0415 -- optional, keep the import local
        df = COV.classify(origin, planned=set(planned))
    # SystemExit is NOT an Exception -- `classify` raises it when the panel is absent, so
    # catching Exception alone would let a missing coverage check kill an order plan that
    # is otherwise complete. The check is a nice-to-have; the plan is not.
    except (Exception, SystemExit) as exc:         # noqa: BLE001 -- never block a plan
        print(f"  (coverage classification unavailable: {exc})")
        return pl.DataFrame({"sku": [], "klass": [], "last_sale": [], "units_recent": []})
    return (df.filter(pl.col("klass") != "planned")
            .select(C.SKU, "klass", "last_sale", "units_recent", "oos_recent",
                    "on_hand", "inbound", "stale_days")
            .sort(["klass", "units_recent"], descending=[False, True]))


def latest_forecast() -> tuple[Path, Path]:
    if not FORECASTS.exists() or not any(FORECASTS.glob("forecast_*.csv")):
        raise SystemExit(
            "no forecast to work from. Produce one first:\n"
            "    uv run python dev/forecast.py")
    csv = sorted(FORECASTS.glob("forecast_*.csv"))[-1]
    return csv, csv.with_suffix(".json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", type=str, default=None,
                    help="forecast CSV (default: the newest in dev/forecasts/)")
    ap.add_argument("--lead-time", type=int, default=None,
                    help=f"days from order to sellable (default {DEFAULT_LEAD_TIME}, "
                         f"which is a placeholder, not your number)")
    ap.add_argument("--lead-time-file", type=str, default=None,
                    help="CSV of sku,lead_time_days overriding --lead-time per SKU")
    ap.add_argument("--review", type=int, default=DEFAULT_REVIEW,
                    help=f"days until the next order can be placed (default {DEFAULT_REVIEW})")
    ap.add_argument("--inventory", type=str, default=None,
                    help="CSV of sku,on_hand,on_order")
    ap.add_argument("--assume-zero-stock", action="store_true",
                    help="treat SKUs with no inventory record as empty (dangerous)")
    ap.add_argument("--flat-ratio", action="store_true",
                    help="one service level for every SKU instead of per-SKU economics")
    ap.add_argument("--no-round", action="store_true",
                    help="skip MOQ / carton rounding")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--tuned", action="store_true")
    args = ap.parse_args()
    common.set_mode("tuned" if args.tuned else "initial")

    lead_default = args.lead_time if args.lead_time is not None else DEFAULT_LEAD_TIME
    cu, co = common.cost_ratio()

    fc_path = Path(args.forecast) if args.forecast else latest_forecast()[0]
    meta_path = fc_path.with_suffix(".json")
    fc = pl.read_csv(fc_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    print(f"\n{'=' * 92}\nORDER PLAN   [{common.MODE}]   cu:co = {cu:g}:{co:g}\n{'=' * 92}")
    plan_origin = fc["origin_date"].max()
    print(f"  forecast      {fc_path.name}   origin {plan_origin}")

    # ---- wide: one row per SKU, one column per horizon -------------------------------
    # One row per SKU of how stale its origin is. `--recover` forecasts SKUs whose panel
    # rows stopped, from their own last real origin; those are routed to a review file
    # rather than into the plan, so the flag has to survive the pivot.
    # Route on the explicit `recovered` flag, NOT on staleness. open_origins keeps a
    # short tail, so an ordinary quiet SKU can carry an origin a few weeks old and still
    # belong in the plan; only a SKU that `--recover` brought back is a weaker claim.
    have_rec = "recovered" in fc.columns
    stale_by_sku = (fc.group_by("sku").agg(
        pl.col("origin_stale_days").max().alias("stale")
        if "origin_stale_days" in fc.columns else pl.lit(0, dtype=pl.Int64).alias("stale"),
        (pl.col("recovered").any() if have_rec
         else pl.lit(False)).alias("recovered")).sort("sku"))

    wide = fc.pivot(on="horizon_days", index="sku", values="forecast_typical")
    hcols = [str(h) for h in common.HORIZONS]
    missing = [c for c in hcols if c not in wide.columns]
    if missing:
        raise SystemExit(f"{fc_path.name} has no forecast for horizons {missing}; "
                         f"re-run dev/forecast.py without --horizons")
    wide = wide.sort("sku")
    skus = wide["sku"].to_list()
    cum = wide.select(hcols).to_numpy()

    # ---- the coverage window, PER SKU ------------------------------------------------
    # Lead time can differ per SKU, so exposure is per SKU. Deriving one window from a
    # scalar default and applying it to the whole catalogue -- which is what this file
    # used to do, silently ignoring --lead-time-file for this purpose -- under-buys every
    # slow-arriving SKU and over-buys every fast one.
    lts = load_lead_times(Path(args.lead_time_file) if args.lead_time_file else None,
                          lead_default, skus)
    lt_days = lts["lead_time_days"].to_numpy().astype(np.float64)
    windows = lt_days + args.review
    window = int(round(float(np.median(windows))))     # representative, for reporting
    print(f"  lead time     median {np.median(lt_days):.0f} d "
          f"(p10 {np.quantile(lt_days, .1):.0f} / p90 {np.quantile(lt_days, .9):.0f})"
          f"{' per SKU from file' if args.lead_time_file else ' (scalar)'}"
          f"   review {args.review} d")
    print(f"  coverage      median {window} d   "
          f"(p10 {np.quantile(windows, .1):.0f} / p90 {np.quantile(windows, .9):.0f})")
    if args.lead_time is None and not args.lead_time_file:
        print("  !! LEAD TIME IS THE PLACEHOLDER. Pass --lead-time or --lead-time-file.")

    n_nonmono = int((np.diff(cum, axis=1) < -1e-9).any(axis=1).sum())
    demand_typical, regime = interpolate_cumulative(cum, windows)
    n_extrap = int((regime == "extrapolated").sum())
    print(f"  curve         from {'/'.join(hcols)} d; "
          f"{n_nonmono:,} SKUs needed monotone repair")
    if n_extrap:
        print(f"  !! {n_extrap:,}/{len(skus):,} SKUs ({100 * n_extrap / len(skus):.0f}%) "
              f"have a coverage window beyond the longest backtested horizon "
              f"({common.HORIZONS[-1]} d).\n"
              f"     For those, everything past {common.HORIZONS[-1]} d is a flat-rate "
              f"extension, not a measured forecast.")

    # ---- per-SKU service level -------------------------------------------------------
    ratios = E.sku_ratios(min(window, common.HORIZONS[-1]))
    econ = wide.select("sku").join(ratios, on="sku", how="left")
    q = econ["q"].fill_null(common.target_quantile()).to_numpy()
    if args.flat_ratio:
        q = np.full(len(skus), common.target_quantile())

    # The model whose error distribution defines the multiplier: the one that produced
    # the forecast at the horizon nearest the coverage window.
    near = min(common.HORIZONS, key=lambda h: abs(h - window))
    model = (fc.filter(pl.col("horizon_days") == near)["model"][0]
             if "model" in fc.columns else None)
    probs, mults = ratio_curve(model, near)
    k = multiplier_for(q, probs, mults)
    k_global = float(np.interp(common.target_quantile(), probs, mults))
    demand_planning = demand_typical * k

    print(f"  service level {'FLAT ' if args.flat_ratio else 'per SKU'} "
          f"q median {np.median(q):.3f} (p10 {np.quantile(q, .1):.3f} / "
          f"p90 {np.quantile(q, .9):.3f})")
    print(f"  multiplier    from {model} errors at {near} d: "
          f"median {np.median(k):.2f} (p10 {np.quantile(k, .1):.2f} / "
          f"p90 {np.quantile(k, .9):.2f});  global q={common.target_quantile():.2f} "
          f"-> {k_global:.2f}")
    shipped = fc.filter(pl.col("horizon_days") == near)["multiplier"][0]
    print(f"  cross-check   the forecast used {shipped:.2f} at {near} d for the same "
          f"quantile; this map says {k_global:.2f}")

    # ---- inventory and rounding ------------------------------------------------------
    inv = load_inventory(Path(args.inventory) if args.inventory else None,
                         skus, args.assume_zero_stock)
    mult = wide.select("sku").join(E.order_multiple(), on="sku", how="left").with_columns(
        moq=pl.col("moq").fill_null(1), carton=pl.col("carton").fill_null(1))

    on_hand = inv["on_hand"].to_numpy()
    on_order = inv["on_order"].to_numpy()
    known = inv["stock_known"].to_numpy()
    net = np.clip(demand_planning - on_hand - on_order, 0.0, None)

    if args.no_round:
        qty_all = np.ceil(net)
    else:
        qty_all = E.round_order(net, mult["moq"].to_numpy(),
                                mult["carton"].to_numpy()).astype(np.float64)

    # A SKU with no stock record gets NULL, not NaN and not zero. Polars evaluates
    # `NaN > 0` as TRUE, so a NaN sentinel here would silently count every unknown-stock
    # SKU as one to buy -- and would land in the CSV as the string "NaN" for whoever
    # reads it next. Nulls are the only representation that survives both.
    null_if_unknown = lambda col: (  # noqa: E731
        pl.when(pl.Series("k", known)).then(pl.Series("v", col)).otherwise(None))
    order_units = float(qty_all[known].sum())
    net_units = float(net[known].sum())

    out = pl.DataFrame({
        "sku": skus,
        "origin_date": [plan_origin] * len(skus),
        "lead_time_days": lts["lead_time_days"],
        "review_days": [args.review] * len(skus),
        "coverage_days": windows.astype(int),
        "curve": regime,
        "service_level": np.round(q, 4),
        "multiplier": np.round(k, 3),
        "demand_typical": np.round(demand_typical, 1),
        "demand_planning": np.round(demand_planning, 1),
        "on_hand": on_hand, "on_order": on_order, "stock_known": known,
        "moq": mult["moq"], "carton": mult["carton"],
    }).with_columns(
        net_requirement=null_if_unknown(np.round(net, 1)),
        order_qty=null_if_unknown(qty_all).cast(pl.Int64),
    ).sort("sku")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else OUT_DIR / f"order_plan_{plan_origin}.csv"

    # Split the plan from the review pile. A recovered SKU's forecast comes from an
    # origin months old; its quantity is computed identically but it is a weaker claim,
    # and merging the two would hide which is which behind one column of numbers.
    meta_j = out.select("sku").join(stale_by_sku, on="sku", how="left")
    out = out.with_columns(
        origin_stale_days=pl.Series("s", meta_j["stale"].fill_null(0).to_numpy()),
        recovered=pl.Series("r", meta_j["recovered"].fill_null(False).to_numpy()))
    review = out.filter(pl.col("recovered"))
    out = out.filter(~pl.col("recovered"))
    out.write_csv(path)
    if review.height:
        rpath = path.with_name(path.stem + "_review.csv")
        review.write_csv(rpath)

    n_known = int(out["stock_known"].sum())
    buy = out.filter(pl.col("order_qty") > 0)
    print(f"\n{'-' * 92}")
    print(f"  SKUs              {out.height:,} in the plan"
          + (f"   + {review.height:,} recovered, routed to review" if review.height else ""))
    print(f"  stock known       {n_known:,}"
          f"{'' if n_known else '   <-- no inventory file: no order quantities computed'}")
    print(f"  demand over {window}d  typical {demand_typical.sum():,.0f} units"
          f"   planning {demand_planning.sum():,.0f} units"
          f"   (+{100 * (demand_planning.sum() / max(demand_typical.sum(), 1) - 1):.0f}%)")
    if n_known:
        print(f"  net requirement   {net_units:,.0f} units")
        print(f"  ORDER             {order_units:,.0f} units across {buy.height:,} SKUs"
              f"{'' if args.no_round else '   (after MOQ + carton rounding)'}")
        if not args.no_round:
            up = order_units - net_units
            print(f"  rounding added    {up:,.0f} units "
                  f"({100 * up / max(net_units, 1):.1f}%)")
    if n_known == 0:
        print("  order_qty is null for every SKU, on purpose. Supply --inventory, or")
        print("  pass --assume-zero-stock if the warehouse really is empty.")
    elif n_known < out.height:
        print(f"  no stock record   {out.height - n_known:,} SKUs -> order_qty is null, "
              f"not zero. They are in the CSV; decide them by hand.")
    if review.height:
        ru = review.filter(pl.col("order_qty") > 0)["order_qty"].sum() or 0
        print(f"  REVIEW            {review.height:,} SKU(s) forecast from a stale origin "
              f"({review['origin_stale_days'].min()}-{review['origin_stale_days'].max()} d "
              f"old): {ru:,.0f} units, NOT in the plan")
        print(f"                    -> {rpath.name}   (ran out of stock rather than died; "
              f"see dev/COVERAGE.md)")

    origin_date = out["origin_date"][0]
    if isinstance(origin_date, str):
        origin_date = dt.date.fromisoformat(origin_date)
    gap = coverage_gap(out["sku"].to_list() + review["sku"].to_list(), origin_date)
    if gap.height:
        gpath = path.with_name(path.stem + "_not_covered.csv")
        gap.write_csv(gpath)
        counts = {r["klass"]: r["count"] for r in
                  gap.group_by("klass").len(name="count").to_dicts()}
        print(f"  NOT COVERED       {gap.height:,} catalogue SKUs have no order line: "
              + ", ".join(f"{v:,} {k}" for k, v in sorted(counts.items())))
        starved = gap.filter(pl.col("klass") == "starved")
        if starved.height:
            print(f"     !! {starved.height} STARVED ({starved['units_recent'].sum():,.0f} "
                  f"units in 180 d, out of stock, nothing inbound) -- re-run "
                  f"dev/forecast.py --recover to price them")
        print(f"                    -> {gpath.name}   (full classification in "
              f"dev/COVERAGE.md)")
    print(f"{'-' * 92}")
    try:
        shown = path.relative_to(LAB.parent)
    except ValueError:
        shown = path                      # --out pointed outside the project
    print(f"\norder plan -> {shown}  ({out.height:,} rows)")

    (path.with_suffix(".json")).write_text(json.dumps({
        "forecast": fc_path.name, "origin": str(fc["origin_date"][0]),
        "mode": common.MODE, "cu": cu, "co": co,
        "lead_time_days": lead_default, "lead_time_is_placeholder":
            args.lead_time is None and not args.lead_time_file,
        "review_days": args.review, "coverage_days_median": window,
        "coverage_days_p10": float(np.quantile(windows, .1)),
        "coverage_days_p90": float(np.quantile(windows, .9)),
        "skus_extrapolated_beyond_longest_horizon": n_extrap,
        "flat_ratio": bool(args.flat_ratio), "rounded": not args.no_round,
        "multiplier_model": model, "multiplier_horizon": near,
        "k_at_global_q": k_global, "k_shipped_by_cost_board": float(shipped),
        "skus": len(skus), "stock_known": n_known,
        "non_monotone_repaired": n_nonmono,
        "not_covered_skus": int(gap.height),
        "not_covered_by_class": {r["klass"]: r["count"] for r in
                                 gap.group_by("klass").len(name="count").to_dicts()}
        if gap.height else {},
        "review_skus": int(review.height),
        "review_units": float(review.filter(pl.col("order_qty") > 0)["order_qty"].sum() or 0),
        "demand_typical": float(demand_typical.sum()),
        "demand_planning": float(demand_planning.sum()),
        "order_units": order_units if n_known else None,
        "forecast_meta": meta,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
