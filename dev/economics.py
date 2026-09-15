"""Per-SKU inventory economics -- what being short and being long actually cost, in money.

    uv run python dev/economics.py            # inspect the derived ratios
    uv run python dev/economics.py --raw      # without the anchoring step

`common.CU:CO` (from assumptions.toml) is ONE number for the whole catalogue. This file
replaces it with one ratio per SKU per horizon, derived from the prices that are already
in the product master (`average_cost`, `po_price`, `sale_price`) and otherwise used only
as features.

-----------------------------------------------------------------------------------
WHY A SINGLE GLOBAL RATIO IS WRONG
-----------------------------------------------------------------------------------
A $2 accessory and a $200 unit do not have the same cost of being wrong. Nor do two
products at the same price with different margins: the one you buy at 90% of retail
loses almost nothing when it sells out and ties up almost everything when it does not.
The global ratio prices both identically, so it necessarily over-covers one and
under-covers the other. Every unit of that error is real money and none of it shows up
in WAPE, in the cost board, or anywhere else in this folder.

-----------------------------------------------------------------------------------
THE TWO COSTS
-----------------------------------------------------------------------------------
    cu  one unit of demand you could not meet
        = gross margin foregone, net of the share of that demand you recover later

            (sale_price - po_price) * (1 - RECAPTURE)

        RECAPTURE is the fraction of a stockout that is deferred rather than lost --
        the customer comes back next month. On a marketplace with substitutes next to
        you it is low. It is NOT zero, and pretending it is inflates every ratio here.
        Pushing the other way and deliberately NOT modelled: a stocked-out listing
        loses search rank, so the true cu has a tail this does not capture. The two
        omissions point in opposite directions, which is the only reason one flat
        number is tolerable.

    co  one unit still sitting there when the horizon ends
        = capital + storage + insurance over the horizon, plus a write-down risk

            average_cost * HOLDING_RATE * (h / 365)  +  average_cost * MARKDOWN_RISK

        Note the horizon term. Overstock is not a write-off -- it is stock held LONGER,
        and how much longer depends on the horizon you are buying for. So co is
        horizon-dependent and the ratio is too, which the global constant cannot express:
        a 30-day cover and a 180-day cover genuinely have different economics.

-----------------------------------------------------------------------------------
WHY THE RAW RATIO CANNOT BE USED AS-IS -- READ THIS BEFORE CHANGING ANYTHING
-----------------------------------------------------------------------------------
On a high-margin catalogue the formula above produces very large ratios -- critical
ratios deep in the upper tail of demand, close to "order for the worst case". That is what
the arithmetic says and it is not a bug in the arithmetic; it is what pure newsvendor
economics always says about a high-margin product. It is still not a number to buy
against, for two reasons:

  1. NOTHING HERE CAN ESTIMATE THAT FAR INTO THE TAIL. A handful of backtest folds
     support maybe a 90th percentile before the estimate is being carried by a few
     rows. A quantile you cannot measure is not a policy, it is a guess with a decimal
     point.
  2. IT OVERRIDES THE CHOSEN RISK APPETITE. The cost ratio in assumptions.toml is the
     owner's statement of how much a stockout should outweigh dead stock. The raw
     newsvendor level silently replaces it with something far more extreme.

So the default mode is ANCHORED: the raw ratios set the *relative* structure -- which
SKUs deserve more cover than the average one and which deserve less -- and that
structure is then rescaled so the median SKU lands exactly on `common.CU:CO`. The
assumptions file keeps control of the LEVEL; the prices decide the SPREAD. That is the
one split that uses the price information without overriding a decision that was never
the model's to make.

`--raw` skips the rescale and is there so the gap stays visible.

-----------------------------------------------------------------------------------
THE THREE ASSUMPTIONS BELOW DO NOT AFFECT THE ANCHORED POLICY. THIS IS ON PURPOSE.
-----------------------------------------------------------------------------------
HOLDING_RATE, RECAPTURE and MARKDOWN_RISK (assumptions.toml) are guesses -- nothing in a
sales history measures any of them. In ANCHORED mode they cancel exactly, and
`tests/test_economics.py` asserts it: each one is a single global constant, so at a fixed
horizon every SKU's `cu` and `co` are scaled by the same factor, the rescale divides that
factor out, and what survives is

    ratio_i  proportional to  unit_margin_i / unit_cost_i

Triple all three and the anchored ratios move by ~1e-15. So the part of this file you
would have to defend to a finance team -- the three numbers nobody can source -- has no
influence on what gets ordered. The per-SKU structure rests entirely on margin and cost,
which are measured, and the LEVEL rests on the stated cost ratio.

They are still real in `--raw` mode, which is exactly why `--raw` is not the default.
The horizon term cancels for the same reason, which is why the anchored ratios are
identical at 30, 90 and 180 days. That is not a bug and not a bug to "fix": a genuine
horizon effect would need a per-SKU holding rate or an obsolescence date.

-----------------------------------------------------------------------------------
WHAT THIS DOES NOT KNOW
-----------------------------------------------------------------------------------
Prices are a current snapshot, not a history -- the product master has one row per SKU.
A SKU whose price moved during the backtest is priced here at today's number. By default
there is no per-SKU holding rate (a pallet of garden furniture and a box of cables
genuinely differ) and no obsolescence date. Those would both sharpen `co`; the optional
per-SKU storage file (`HOLDING_FILE`, below) supplies the first.
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
from common import LAB  # noqa: E402
from demandml import assumptions as _assumptions  # noqa: E402
from demandml import config as C  # noqa: E402

# --------------------------------------------------------------------------------------
# The knobs. Three numbers, all of them assumptions, all of them in one place.
# --------------------------------------------------------------------------------------

#: Annual cost of carrying one unit, as a fraction of its unit cost: cost of capital,
#: storage, insurance, shrink. Set in assumptions.toml because nothing in the data
#: measures it. If you know your real warehouse rate, that is the number to change.
HOLDING_RATE: float = _assumptions.HOLDING_RATE

#: Share of stocked-out demand that comes back rather than going to a competitor.
#: Keep it conservative: on a marketplace with substitutes a stockout is mostly a lost
#: sale. Raising it lowers every cu and therefore every service level.
RECAPTURE: float = _assumptions.RECAPTURE

#: Probability-weighted write-down on a unit still held when the horizon closes,
#: as a fraction of unit cost. Covers obsolescence, damage and markdown-to-clear.
MARKDOWN_RISK: float = _assumptions.MARKDOWN_RISK

#: The raw ratio is clipped into this band BEFORE anchoring, so one SKU with a broken
#: price cannot drag the rescale. The band is wide on purpose -- it is a guard against
#: bad data, not a second opinion about economics.
RATIO_FLOOR, RATIO_CAP = 0.25, 200.0

#: After anchoring, the spread is clipped again to the band the evidence can actually
#: support. 9:1 is a 0.90 critical ratio, which is about as far into the tail as a
#: handful of backtest folds can validate; 1:1 is symmetric, i.e. "this SKU gets no preference
#: for over-stock at all". Anything outside this is not measurable here.
ANCHORED_FLOOR, ANCHORED_CAP = 1.0, 9.0

PRODUCT = C.INTERIM / "product.parquet"


def load_product() -> pl.DataFrame:
    """The product master, reduced to the columns this file needs."""
    if not PRODUCT.exists():
        raise SystemExit(
            f"{PRODUCT} does not exist. Build it:  uv run demandml ingest")
    keep = ["sku", "average_cost", "po_price", "sale_price",
            "default_cost", "default_sale_price", "moq", "carton_size"]
    d = pl.read_parquet(PRODUCT)
    missing = [c for c in keep if c not in d.columns]
    if missing:
        raise SystemExit(f"product.parquet is missing {missing}")
    return d.select(keep).unique(subset=["sku"], keep="first")


def unit_economics(product: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-SKU `unit_margin` and `unit_cost`, with the fallbacks spelled out.

    `po_price` is what you pay the factory and `average_cost` is the landed figure the
    books carry; they are usually close and either is a defensible basis for holding
    cost. `average_cost` is preferred because it is the one that includes freight, with
    `default_cost` and then `po_price` as fallbacks -- a zero in the product master means
    "not filled in", not "free".
    """
    d = product if product is not None else load_product()
    pos = lambda c: pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(None)  # noqa: E731
    return d.with_columns(
        unit_cost=pl.coalesce(pos("average_cost"), pos("default_cost"), pos("po_price")),
        price=pl.coalesce(pos("sale_price"), pos("default_sale_price")),
        buy=pl.coalesce(pos("po_price"), pos("average_cost"), pos("default_cost")),
    ).with_columns(
        # Margin can only be trusted when both legs are present AND the sign is sane.
        # A sale_price below po_price is a data error, not a loss-leader strategy.
        unit_margin=pl.when((pl.col("price") > 0) & (pl.col("buy") > 0)
                            & (pl.col("price") > pl.col("buy")))
        .then(pl.col("price") - pl.col("buy")).otherwise(None),
    )


#: Optional MEASURED per-SKU storage rates (JSON: `storage_per_unit_month` = {sku: cost
#: per unit per month}, optional `capital_rate` and `excess_months`). Supply it from your
#: own warehouse or marketplace storage charges. When this file exists it replaces the
#: `HOLDING_RATE x unit_cost` proxy in `co` -- see `measured_holding()`.
HOLDING_FILE = LAB / "tuned_params" / "holding_per_sku.json"


def measured_holding() -> dict | None:
    """The adopted per-SKU storage rates, or None if nothing has been adopted.

    This is the one input to `co` that SURVIVES anchoring. The rescale in `sku_ratios`
    divides out any global constant exactly -- `tests/test_economics.py` asserts that
    tripling `HOLDING_RATE`, `RECAPTURE` and `MARKDOWN_RISK` together moves the anchored
    ratios by ~1e-15 -- so none of the three guesses in this file can change an order
    quantity. A per-SKU rate is not a common factor, so it can and does -- and, for the
    same reason, once it is present the three guesses stop cancelling exactly.

    Absent by default. Writing the file is a deliberate act; nothing here creates it.
    """
    if not HOLDING_FILE.exists():
        return None
    try:
        got = json.loads(HOLDING_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return got if got.get("storage_per_unit_month") else None


def raw_ratios(horizon: int, product: pl.DataFrame | None = None) -> pl.DataFrame:
    """`cu`, `co` and their raw ratio per SKU at one horizon, straight from the formulas.

    No anchoring, no opinion -- this is what the prices say on their own.

    `co` has two forms. The default is the textbook one, holding cost strictly
    proportional to what the unit cost. Where a per-SKU storage file (`HOLDING_FILE`)
    is present, the storage term becomes per-SKU and measured:

        co = unit_cost * (CAPITAL_RATE * years + MARKDOWN_RISK)   [proportional]
             + storage_per_unit_month * EXCESS_MONTHS            [measured, per SKU]

    Storage is typically billed by volume, not by value. Storage cost and unit cost are
    correlated, so the proportional form captures something real, but it cannot see the
    cheap-and-bulky versus expensive-and-compact axis -- which is what a measured
    per-SKU rate adds.
    """
    e = unit_economics(product)
    years = horizon / 365.0
    adopted = measured_holding()
    if adopted:
        rates = adopted["storage_per_unit_month"]
        cap = float(adopted.get("capital_rate", HOLDING_RATE))
        months = float(adopted.get("excess_months", 3.0))
        med = float(np.median(list(rates.values()))) if rates else 0.0
        e = e.with_columns(
            _storage=pl.col("sku").replace_strict(rates, default=med)
            .cast(pl.Float64).fill_null(med))
        co_expr = (pl.col("unit_cost") * (cap * years + MARKDOWN_RISK)
                   + pl.col("_storage") * months)
    else:
        co_expr = pl.col("unit_cost") * (HOLDING_RATE * years + MARKDOWN_RISK)

    return e.with_columns(
        cu=pl.col("unit_margin") * (1.0 - RECAPTURE),
        co=co_expr,
    ).with_columns(
        ratio_raw=pl.when((pl.col("cu") > 0) & (pl.col("co") > 0))
        .then((pl.col("cu") / pl.col("co")).clip(RATIO_FLOOR, RATIO_CAP))
        .otherwise(None),
    ).select(["sku", "unit_cost", "unit_margin", "moq", "carton_size",
              "cu", "co", "ratio_raw"])


def sku_ratios(horizon: int, anchor: bool = True,
               weights: pl.DataFrame | None = None,
               product: pl.DataFrame | None = None) -> pl.DataFrame:
    """The per-SKU cost ratio and critical quantile actually used downstream.

    With `anchor=True` (the default) the raw ratios are rescaled in log space so that the
    median SKU sits on `common.CU/common.CO`, then clipped to the band a handful of
    backtest folds can support. Log space because a ratio is multiplicative -- "twice as
    stockout-averse" is the meaningful step, not "one more unit of cu".

    `weights` is an optional frame of `sku` + `weight` (units of demand, typically) so
    the anchor lands on the median *unit sold* rather than the median SKU. Which one you
    want is a real choice: SKU-median treats every product as one vote, demand-median
    lets the big sellers set the level. Default is SKU-median -- the small products are
    exactly the ones the global ratio was serving worst, so letting the big ones anchor
    would re-create the problem this file exists to fix.

    SKUs with no usable prices fall back to the global ratio. That is the honest
    default: absent information is not a reason to move a SKU off the stated cost ratio
    in either direction.
    """
    cu_g, co_g = common.cost_ratio()
    target = cu_g / co_g
    d = raw_ratios(horizon, product)

    usable = d.filter(pl.col("ratio_raw").is_not_null())
    if not usable.height:
        raise SystemExit("no SKU has usable prices -- check data/interim/product.parquet")

    if anchor:
        logs = np.log(usable.get_column("ratio_raw").to_numpy())
        if weights is not None:
            w = (usable.select("sku").join(weights, on="sku", how="left")
                 .get_column("weight").fill_null(0.0).to_numpy())
            centre = _weighted_median(logs, w) if w.sum() > 0 else float(np.median(logs))
        else:
            centre = float(np.median(logs))
        shift = np.log(target) - centre           # move the median onto the stated ratio
        d = d.with_columns(
            ratio=(pl.col("ratio_raw").log() + shift).exp()
            .clip(ANCHORED_FLOOR, ANCHORED_CAP))
    else:
        d = d.with_columns(ratio=pl.col("ratio_raw"))

    return d.with_columns(
        ratio=pl.col("ratio").fill_null(target),
    ).with_columns(
        # The critical ratio: the demand quantile that minimises this SKU's own cost.
        q=pl.col("ratio") / (1.0 + pl.col("ratio")),
        horizon=pl.lit(horizon),
    )


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def order_multiple(product: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-SKU `moq` and `carton` as clean positive integers, for rounding an order.

    Both are 0 or blank in the product master when they do not apply, which is not the same as
    1 -- but 1 is what "no constraint" means arithmetically, so that is what they become.
    `carton_size` arrives as a string; anything non-numeric becomes 1 rather than an
    exception, because a malformed carton size should not stop a purchase order.
    """
    d = product if product is not None else load_product()
    return d.select(
        pl.col("sku"),
        moq=pl.col("moq").cast(pl.Float64, strict=False).fill_null(0).clip(0, None)
        .cast(pl.Int64),
        carton=pl.col("carton_size").cast(pl.Float64, strict=False).fill_null(1)
        .fill_nan(1).clip(1, None).round().cast(pl.Int64),
    ).with_columns(moq=pl.when(pl.col("moq") > 0).then(pl.col("moq")).otherwise(1))


def round_order(qty: np.ndarray, moq: np.ndarray, carton: np.ndarray) -> np.ndarray:
    """Round a raw order quantity up to something a factory will actually accept.

    Order of operations matters and is deliberate: lift to the MOQ first, then round UP
    to a whole carton. Rounding down would silently re-introduce the stockout risk the
    whole multiplier exists to remove, so everything here rounds up -- the cost of the
    extra part-carton is `co` per unit, and `co` is the cheap side of the ratio.

    A zero order stays zero. "Buy nothing" must survive rounding or every dead SKU in the
    catalogue gets an MOQ-sized order.
    """
    qty = np.asarray(qty, dtype=np.float64)
    moq = np.maximum(np.asarray(moq, dtype=np.float64), 1.0)
    carton = np.maximum(np.asarray(carton, dtype=np.float64), 1.0)
    out = np.where(qty > 0, np.maximum(np.ceil(qty), moq), 0.0)
    out = np.where(out > 0, np.ceil(out / carton) * carton, 0.0)
    return out.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="store_true", help="skip the anchoring step")
    ap.add_argument("--horizons", type=str, default="30,90,180")
    args = ap.parse_args()

    cu_g, co_g = common.cost_ratio()
    print(f"\n{'=' * 84}\nPER-SKU INVENTORY ECONOMICS"
          f"   [{'raw' if args.raw else 'anchored'}]\n{'=' * 84}")
    print(f"  global ratio in force   {cu_g:g}:{co_g:g}  -> q = {common.target_quantile():.3f}")
    print(f"  holding {HOLDING_RATE:.0%}/yr   recapture {RECAPTURE:.0%}   "
          f"markdown {MARKDOWN_RISK:.0%}\n")

    prod = load_product()
    e = unit_economics(prod)
    n_m = e.get_column("unit_margin").is_not_null().sum()
    print(f"  {e.height:,} SKUs in the product master; {n_m:,} with a usable margin "
          f"({n_m / e.height:.0%})\n")

    for h in [int(x) for x in args.horizons.split(",") if x.strip()]:
        d = sku_ratios(h, anchor=not args.raw, product=prod)
        u = d.filter(pl.col("ratio_raw").is_not_null())
        r, q = u.get_column("ratio").to_numpy(), u.get_column("q").to_numpy()
        rr = u.get_column("ratio_raw").to_numpy()
        print(f"[{h:>3} days]  raw ratio    median {np.median(rr):7.1f}   "
              f"p10 {np.quantile(rr, .1):7.1f}   p90 {np.quantile(rr, .9):7.1f}")
        print(f"            in force     median {np.median(r):7.2f}   "
              f"p10 {np.quantile(r, .1):7.2f}   p90 {np.quantile(r, .9):7.2f}")
        print(f"            quantile q   median {np.median(q):7.3f}   "
              f"p10 {np.quantile(q, .1):7.3f}   p90 {np.quantile(q, .9):7.3f}")
        print(f"            at the cap: {(r >= ANCHORED_CAP - 1e-9).sum():,} SKUs   "
              f"at the floor: {(r <= ANCHORED_FLOOR + 1e-9).sum():,}\n")

    m = order_multiple(prod)
    print(f"  order multiples: {(m['moq'] > 1).sum():,} SKUs with a real MOQ "
          f"(median {m.filter(pl.col('moq') > 1)['moq'].median()}), "
          f"{(m['carton'] > 1).sum():,} with a carton > 1 "
          f"(median {m.filter(pl.col('carton') > 1)['carton'].median()})")
    print("=" * 84)


if __name__ == "__main__":
    main()
