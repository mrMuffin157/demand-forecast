"""Which catalogue SKUs does the order plan not answer for, and why.

    uv run python dev/coverage.py                   # classify + write dev/COVERAGE.md
    uv run python dev/coverage.py --origin 2026-08-10

`dev/forecast.py` starts from `open_origins.parquet`, which holds the newest origins in
the panel. A SKU that stopped appearing in the sales history has no row there, gets no
forecast, and therefore gets no line in the order plan. It does not appear with a zero --
it is simply absent, and "not in the file" reads to whoever orders as "nothing to do".

On a real catalogue that can be a sizeable minority of SKUs, and nearly all of them are
absent because the PANEL stops for them, not because anything downstream filtered them
out. The sales table stops emitting rows and the pipeline has no opinion about it.

-----------------------------------------------------------------------------------
WHY THIS IS NOT MERELY UNTIDY
-----------------------------------------------------------------------------------
Most of those SKUs are genuinely finished and should be absent. But the reason a listing
stops producing rows is not always that it died. A SKU that sold steadily while mostly
out of stock and then vanished from the panel is not a dead product, it is a product
nobody could buy -- and it is exactly the SKU a reorder is most urgent for.

A simpler check -- flag SKUs that sold within 60 days of the origin -- cannot see this:
anything starved for longer than that drops out silently, and it undercounts the SKUs with
real demand in the last 180 days. `policy.coverage_gap` delegates here for that reason.

With daily stock history the distinction becomes available. `data/db/inventory_daily`
carries daily on-hand and inbound per SKU (the sample generator,
`scripts/make_sample_data.py`, writes one; so can your own extract), so "stopped selling
because it was delisted" and "stopped selling because the warehouse was empty" are
separable facts rather than a guess.

-----------------------------------------------------------------------------------
THE CLASSES
-----------------------------------------------------------------------------------
  planned       has a forecast and an order line. Nothing to do.
  starved       sold recently, ran out, nothing inbound. THE ONE THAT MATTERS -- this is
                unmet demand, and the pipeline was silently omitting it.
  incoming      out of stock but stock is already on the water. Demand is real; the
                reorder may not be urgent. Shown so a human can judge, not hidden.
  dormant       has stock, is not selling. Usually a listing problem, not a supply one.
                Ordering more would be the wrong answer.
  discontinued  no sales, no stock, nothing inbound, listing not active. Correctly absent.

Nothing here is a decision. This file sorts the catalogue into "the plan answers for it",
"the plan should have answered for it", and "the plan is right to be silent", and puts the
middle group in front of a human.

-----------------------------------------------------------------------------------
WHAT IT DOES NOT DO
-----------------------------------------------------------------------------------
It does not fabricate panel rows. Extending every quiet SKU to the end of the panel with
zero-unit days would put those zeros in front of every model in this folder, and they are
censored days rather than measured ones -- the distinction `sample_weight = 0` exists to
preserve. Recovering a forecast for a starved SKU is `forecast.py --recover`, which
predicts from the SKU's own last real origin and labels how stale it is.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
from common import LAB, OPEN_ORIGINS  # noqa: E402
from demandml import config as C  # noqa: E402

PANEL = C.PROCESSED / "panel_daily.parquet"
INVENTORY = C.DATA / "db" / "inventory_daily.parquet"

#: How far back to look for "did this thing sell at all recently". Six months, matching
#: the longest horizon the project forecasts, so a SKU with demand anywhere inside the
#: window the models are trained on cannot be classed as finished.
RECENT_DAYS = 180

#: Days of on-hand history to judge whether a SKU is currently buyable. Short, because
#: the question is about now, not about the year.
STOCK_DAYS = 28

CLASSES = ("planned", "starved", "incoming", "dormant", "discontinued")


def stock_position(skus: list[str], origin: dt.date) -> pl.DataFrame:
    """On-hand and inbound per SKU as of the origin, from the DB extract.

    Returns an empty-but-correctly-shaped frame when `data/db/` is absent, so the
    classification degrades to a sales-only one rather than failing. The report says
    which of the two it did.
    """
    empty = pl.DataFrame(schema={"sku": pl.String, "on_hand": pl.Float64,
                                 "inbound": pl.Float64, "days_with_stock": pl.Int64})
    if not INVENTORY.exists():
        return empty
    lo = origin - dt.timedelta(days=STOCK_DAYS)
    d = (pl.scan_parquet(INVENTORY)
         .filter((pl.col("date") > lo) & (pl.col("date") <= origin)
                 & pl.col("sku").is_in(skus))
         .with_columns(
             pl.sum_horizontal(
                 pl.col("inbound_working").fill_null(0.0),
                 pl.col("inbound_shipped").fill_null(0.0),
                 pl.col("inbound_receiving").fill_null(0.0)).alias("_inb"))
         .group_by("sku")
         .agg(on_hand=pl.col("on_hand").fill_null(0.0).last(),
              inbound=pl.col("_inb").last(),
              days_with_stock=(pl.col("on_hand").fill_null(0.0) > 0).sum())
         .collect(engine="streaming"))
    return d if d.height else empty


def classify(origin: dt.date, planned: set[str] | None = None) -> pl.DataFrame:
    """One row per catalogue SKU: what it did lately, what stock it has, and the class."""
    if not PANEL.exists():
        raise SystemExit(f"{PANEL} does not exist. Build it:  uv run demandml panel")

    lo = origin - dt.timedelta(days=RECENT_DAYS)
    lf = pl.scan_parquet(PANEL)
    # Both of these are optional. A panel built by an older version of the pipeline, or a
    # minimal one, still gets classified -- on less evidence, but without failing. An
    # order plan must never be blocked by its own coverage check.
    cols = set(lf.collect_schema().names())
    oos_expr = (pl.col("is_oos").cast(pl.Float64).filter(pl.col(C.DATE) > lo)
                .mean().fill_null(0.0) if "is_oos" in cols
                else pl.lit(0.0)).alias("oos_recent")
    listing_expr = (pl.col("listing_status").cast(pl.String).last()
                    if "listing_status" in cols
                    else pl.lit(None, dtype=pl.String)).alias("listing")
    hist = (lf.filter(pl.col(C.DATE) <= origin)
            .group_by(C.SKU)
            .agg(last_panel_row=pl.col(C.DATE).max(),
                 last_sale=pl.col(C.DATE).filter(pl.col(C.TARGET) > 0).max(),
                 units_recent=pl.col(C.TARGET)
                     .filter(pl.col(C.DATE) > lo).sum().fill_null(0.0),
                 oos_recent=oos_expr, listing=listing_expr)
            .collect(engine="streaming"))

    skus = hist[C.SKU].to_list()
    stock = stock_position(skus, origin)
    df = (hist.join(stock, left_on=C.SKU, right_on="sku", how="left")
          .with_columns(on_hand=pl.col("on_hand").fill_null(0.0),
                        inbound=pl.col("inbound").fill_null(0.0),
                        days_with_stock=pl.col("days_with_stock").fill_null(0),
                        stale_days=(pl.lit(origin) - pl.col("last_panel_row")).dt.total_days()))

    if planned is None:
        planned = set(pl.read_parquet(OPEN_ORIGINS, columns=[C.SKU])[C.SKU].unique()) \
            if OPEN_ORIGINS.exists() else set()
    return apply_rule(df, planned)


def apply_rule(df: pl.DataFrame, planned: set[str]) -> pl.DataFrame:
    """The classification itself, separated from where the numbers came from.

    Needs `sku`, `units_recent`, `on_hand`, `inbound` and `listing`. Kept pure so
    `tests/test_coverage.py` can pin the rule without building a panel -- the ordering of
    the branches IS the rule, and getting `starved` to fire before `dormant` is the whole
    point of the file.
    """
    sold = pl.col("units_recent") > 0
    has_stock = pl.col("on_hand") > 0
    has_inbound = pl.col("inbound") > 0
    is_active = pl.col("listing").cast(pl.String).str.to_lowercase() == "active"

    return df.with_columns(
        planned=pl.col(C.SKU).is_in(list(planned)),
    ).with_columns(
        klass=pl.when(pl.col("planned")).then(pl.lit("planned"))
        # Sold lately and cannot be bought now, with nothing on the way. Unmet demand.
        .when(sold & ~has_stock & ~has_inbound).then(pl.lit("starved"))
        # Sold lately, empty, but replenishment is already moving.
        .when(sold & ~has_stock & has_inbound).then(pl.lit("incoming"))
        # Sitting on stock and not selling: a listing problem, not a supply one. An
        # active listing with no stock and no sales is also not a supply problem yet.
        .when(has_stock | is_active).then(pl.lit("dormant"))
        .otherwise(pl.lit("discontinued")).alias("klass")
    ).sort(["klass", "units_recent"], descending=[False, True])


def report(df: pl.DataFrame, origin: dt.date, have_stock: bool) -> str:
    L: list[str] = []
    A = L.append
    A("# What the order plan does not answer for\n")
    A(f"*Generated by `dev/coverage.py` at origin **{origin}**, from "
      f"`panel_daily.parquet`"
      f"{' and `data/db/inventory_daily.parquet`' if have_stock else ''}.*\n")

    if not have_stock:
        A("> **`data/db/inventory_daily.parquet` is absent, so this ran on sales history "
          "alone.** Without stock it cannot tell a SKU that was delisted from one that "
          "ran out, which is the entire distinction this file exists to draw. Every "
          "`starved` row below is a guess until the extract is present.\n")

    n = df.height
    counts = {r["klass"]: r["count"] for r in
              df.group_by("klass").len(name="count").to_dicts()}
    A(f"{n:,} SKUs in the panel. "
      f"{counts.get('planned', 0):,} have an order line; "
      f"{n - counts.get('planned', 0):,} do not.\n")

    A("| class | SKUs | units sold in last 180 d | what it means |")
    A("|---|---|---|---|")
    meaning = {
        "planned": "has a forecast and an order line",
        "starved": "**sold recently, out of stock, nothing inbound — unmet demand**",
        "incoming": "out of stock but already being replenished",
        "dormant": "has stock or an active listing, is not selling",
        "discontinued": "no sales, no stock, nothing inbound — correctly absent",
    }
    for k in CLASSES:
        sub = df.filter(pl.col("klass") == k)
        if not sub.height:
            continue
        A(f"| `{k}` | {sub.height:,} | {sub['units_recent'].sum():,.0f} | {meaning[k]} |")
    A("")

    for k in ("starved", "incoming"):
        sub = df.filter(pl.col("klass") == k)
        if not sub.height:
            continue
        A(f"## `{k}` — {sub.height} SKU(s)\n")
        if k == "starved":
            A("These sold inside the last 180 days, cannot be bought today, and have "
              "nothing on the water. They are missing from the order plan and they are "
              "the reason this file exists.\n")
        else:
            A("Out of stock, but a shipment is already moving. Judge whether it is "
              "enough; do not assume the plan's silence means it is.\n")
        A("| sku | last sale | units 180 d | out of stock | on hand | inbound | "
          "panel stale |")
        A("|---|---|---|---|---|---|---|")
        for r in sub.head(40).to_dicts():
            A(f"| `{r['sku']}` | {r['last_sale'] or '—'} | {r['units_recent']:,.0f} | "
              f"{r['oos_recent']:.0%} | {r['on_hand']:,.0f} | {r['inbound']:,.0f} | "
              f"{r['stale_days']} d |")
        if sub.height > 40:
            A(f"\n*({sub.height - 40} more in the CSV.)*")
        A("")

    A("## How to read this\n")
    A("- **`starved` is the actionable class.** Everything else is either already "
      "answered or correctly silent. A `starved` SKU has demand the panel can measure "
      "and no way to meet it, and the forecast pipeline was omitting it entirely.\n")
    A("- **A recovered forecast is stale by construction.** `forecast.py --recover` "
      "predicts these from the SKU's own last real origin, which may be months old, and "
      "labels the staleness. It answers 'what would this have sold', which is the right "
      "question for a reorder and the wrong one for revenue planning.\n")
    A("- **`dormant` usually is not a supply problem.** Stock on the shelf and no sales "
      "is a listing, price or ranking problem, and ordering more units would make it "
      "worse. It is listed here so the silence is deliberate rather than accidental.\n")
    A("- **The classes are evidence, not status flags.** `listing_status` is often null "
      "for exactly the SKUs that went missing, so this reasons from what the SKU "
      "actually did rather than from what a column claims.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origin", type=str, default=None,
                    help="as-of date (default: the newest panel day)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.origin:
        origin = dt.date.fromisoformat(args.origin)
    else:
        origin = pl.scan_parquet(PANEL).select(pl.col(C.DATE).max()).collect().item()

    print(f"\n{'=' * 92}\nCATALOGUE COVERAGE   origin {origin}\n{'=' * 92}")
    df = classify(origin)
    have_stock = INVENTORY.exists()
    if not have_stock:
        print(f"  !! {INVENTORY} absent -- classifying on sales history alone")

    for k in CLASSES:
        sub = df.filter(pl.col("klass") == k)
        if sub.height:
            print(f"  {k:<14} {sub.height:>5,} SKUs"
                  f"   {sub['units_recent'].sum():>9,.0f} units in last {RECENT_DAYS} d")

    csv = LAB / "forecasts" / f"coverage_{origin}.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(csv)
    out = Path(args.out) if args.out else LAB / "COVERAGE.md"
    out.write_text(report(df, origin, have_stock))

    starved = df.filter(pl.col("klass") == "starved")
    if starved.height:
        print(f"\n  {starved.height} STARVED SKU(s): sold "
              f"{starved['units_recent'].sum():,.0f} units in the last {RECENT_DAYS} d, "
              f"cannot be bought, nothing inbound.")
        print(f"  Recover forecasts for them with:  "
              f"uv run python dev/forecast.py --recover")
    print(f"\nwrote {out.relative_to(LAB.parent)} and {csv.relative_to(LAB.parent)}")


if __name__ == "__main__":
    main()
