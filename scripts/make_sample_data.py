"""Generate a small SYNTHETIC dataset so the whole pipeline and test suite run end to end.

    uv run python scripts/make_sample_data.py            # -> data/ (or $DEMANDML_DATA)
    uv run python scripts/make_sample_data.py --skus 120 --start 2021-01-01

Nothing here comes from a real business. Every number is drawn from the random process
below, seeded so the output is reproducible. The files have the same columns and types the
pipeline expects from its sales database extract:

    data/db/sales_daily.parquet       one row per SKU per day (units, revenue, stock-out flag, ...)
    data/db/product.parquet           product master (price, cost, dimensions, category, ...)
    data/db/inventory_daily.parquet   daily on-hand / inbound stock per SKU
    data/db/deals.parquet             a promotion calendar
    data/inventory/position_sample.csv  current on-hand / on-order, for dev/policy.py
    data/raw/daily_sales_sample.csv   the same daily sales as CSV (used by examples/walkthrough.py,
                                      and the pipeline's fallback when data/db/ is absent)

The demand process is deliberately realistic enough to exercise every code path: weekly and
yearly seasonality, a summer sale event and a Black Friday / Cyber Monday peak, trends,
SKU families with several variants, intermittent low sellers, launches part-way through
the history, discontinued SKUs, stock-out spells (sales censored to ~0 and flagged),
promotions, and the occasional bulk order.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def _summer_event(year: int) -> tuple[dt.date, dt.date]:
    """A two-day mid-July sale event (second Tuesday-Wednesday of July)."""
    d = dt.date(year, 7, 1)
    while d.weekday() != 1:
        d += dt.timedelta(days=1)
    d += dt.timedelta(days=7)
    return d, d + dt.timedelta(days=1)


def _black_friday(year: int) -> dt.date:
    d = dt.date(year, 11, 1)
    while d.weekday() != 3:
        d += dt.timedelta(days=1)
    return d + dt.timedelta(days=22)          # the day after the 4th Thursday


def make_catalogue(rng: np.random.Generator, n_skus: int, start: dt.date, end: dt.date):
    """SKU codes as BRAND-FAMILY-VARIANT, e.g. ZA-104-2, grouped into families."""
    brands = ["ZA", "ZB", "ZC", "ZD"]
    skus, fam_of = [], []
    fam_no = 100
    while len(skus) < n_skus:
        brand = brands[rng.integers(len(brands))]
        n_var = int(rng.choice([1, 1, 2, 3, 4]))
        variants = ["1", "2", "3", "4"] if rng.random() < 0.5 else ["B", "G", "K", "W"]
        for v in variants[:n_var]:
            skus.append(f"{brand}-{fam_no}-{v}")
            fam_of.append(f"{brand}-{fam_no}")
        fam_no += 1
    skus, fam_of = skus[:n_skus], fam_of[:n_skus]

    span = (end - start).days
    rows = []
    for i, (sku, fam) in enumerate(zip(skus, fam_of)):
        # Volume: log-normal, so a few SKUs dominate and many sell a handful a day, plus
        # a tail of slow movers (a few units a month) whose forecast windows are often
        # empty -- the case intermittent-demand models (Croston, the hurdle model) exist for.
        if rng.random() < 0.15:
            level = float(rng.uniform(0.02, 0.08))
        else:
            level = float(np.exp(rng.normal(1.3, 1.1)))
        launch = start if rng.random() < 0.7 else start + dt.timedelta(days=int(rng.integers(60, span - 400)))
        retire = None
        if rng.random() < 0.08:
            retire = start + dt.timedelta(days=int(rng.integers(span // 2, span - 120)))
        cost = round(float(np.exp(rng.normal(1.6, 0.6))), 2)
        price = round(cost * float(rng.uniform(2.0, 4.5)), 2)
        rows.append(dict(
            sku=sku, family=fam, level=level, launch=launch, retire=retire,
            trend=float(rng.normal(0.0, 0.25)),          # log-change per year
            q4=float(rng.uniform(0.5, 2.5)),             # strength of the Q4 peak
            summer=float(rng.uniform(0.2, 1.0)),
            event_lift=float(rng.uniform(0.8, 2.5)),
            oos_prone=float(rng.uniform(0.0, 0.004)),    # daily chance a stock-out starts
            cost=cost, price=price, idx=i,
        ))
    return rows


def simulate_sales(rng: np.random.Generator, cat, start: dt.date, end: dt.date):
    days = pl.date_range(start, end, "1d", eager=True).to_list()
    n = len(days)
    doy = np.array([d.timetuple().tm_yday for d in days])
    dow = np.array([d.weekday() for d in days])
    years = np.array([(d - start).days / 365.25 for d in days])
    events = set()
    for y in range(start.year, end.year + 1):
        a, b = _summer_event(y)
        events |= {a, b}
        bf = _black_friday(y)
        events |= {bf, bf + dt.timedelta(days=3)}
    is_event = np.array([d in events for d in days])
    week_shape = np.array([0.95, 0.97, 0.98, 1.0, 1.05, 1.06, 0.99])[dow]

    out, inv = [], []
    for s in cat:
        yearly = (1 + s["q4"] * np.exp(-((doy - 340) / 25.0) ** 2)
                  + s["summer"] * np.exp(-((doy - 190) / 30.0) ** 2))
        mu = s["level"] * week_shape * yearly * np.exp(s["trend"] * years)
        mu = np.where(is_event, mu * (1 + s["event_lift"]), mu)
        # Promotions: a few 3-7 day deal windows a year, ~+40% demand.
        deal = np.zeros(n, bool)
        for _ in range(int(rng.integers(2, 6) * (n / 365))):
            a = int(rng.integers(0, n - 7))
            deal[a:a + int(rng.integers(3, 8))] = True
        mu = np.where(deal, mu * 1.4, mu)
        # Negative-binomial demand (over-dispersed Poisson).
        r = 2.0
        demand = rng.negative_binomial(r, r / (r + mu))
        # Stock-out spells: once started, last 5-40 days; sales mostly lost.
        oos = np.zeros(n, bool)
        i = 0
        while i < n:
            if rng.random() < s["oos_prone"]:
                L = int(rng.integers(5, 41))
                oos[i:i + L] = True
                i += L
            i += 1
        sold = np.where(oos, rng.binomial(demand, 0.05), demand)
        # Occasional bulk/wholesale order: a spike the bulk-filtered count excludes.
        bulk = np.zeros(n, int)
        for j in np.flatnonzero(rng.random(n) < 0.002):
            bulk[j] = int(rng.integers(50, 400))
        alive = np.array([(d >= s["launch"]) and (s["retire"] is None or d < s["retire"])
                          for d in days])
        sold_total = np.where(alive, sold + bulk, 0)
        on_hand = np.where(oos | ~alive, 0.0, np.round(mu * rng.uniform(20, 120)))
        for k, d in enumerate(days):
            if d < s["launch"]:
                continue
            u = int(sold_total[k])
            rev = round(u * s["price"] * (0.85 if deal[k] else 1.0), 2)
            out.append((s["sku"], d, u, int(sold[k]) if alive[k] else 0, rev, bool(oos[k]),
                        bool(deal[k]), s["cost"], s["price"]))
            inv.append((s["sku"], d, float(on_hand[k])))
    return out, inv


def build_frames(rng, cat, sales, inv, end):
    sku, date, units, filt, rev, oos, deal, cost, price = map(list, zip(*sales))
    n = len(sku)
    units_a = np.array(units)
    rev_a = np.array(rev)
    impressions = rng.poisson(np.maximum(units_a, 1) * 60)
    clicks = rng.binomial(impressions, 0.03)
    sessions = np.maximum(clicks + rng.poisson(units_a * 3), units_a)
    item_cost = units_a * np.array(cost)
    created = [dt.datetime.combine(d, dt.time(6)) for d in date]
    daily = pl.DataFrame({
        "sku": sku,
        "salescount": units_a.astype(np.int64),
        "date_purchased": date,
        "is_oos": ["Y" if x else "N" for x in oos],
        "date_created": created,
        "oos_threshold": np.full(n, 0, np.int64),
        "revenue": rev_a,
        "profit": np.round(rev_a - item_cost, 2),
        "final_profit": np.round(rev_a * 0.7 - item_cost, 2),
        "ad_cost": np.round(rng.gamma(1.0, 0.4, n) * units_a, 2),
        "return_amount": np.round(rev_a * rng.uniform(0, 0.05, n), 2),
        "id": np.arange(1, n + 1, dtype=np.int64),
        "order_count": np.maximum(np.round(units_a / rng.uniform(1.0, 1.3, n)), 0).astype(np.int64),
        "ad_units": rng.binomial(units_a, 0.25).astype(np.int64),
        "impressions": impressions.astype(np.int64),
        "clicks": clicks.astype(np.int64),
        "last_updated": created,
        "big_order_filter_sales_count": np.array(filt, np.int64),
        "duty": np.round(item_cost * 0.05, 2),
        "fba_destination_fee": np.round(units_a * 0.3, 2),
        "is_deal": ["Y" if x else "N" for x in deal],
        "total_item_cost": np.round(item_cost, 2),
        "total_po_price": np.round(item_cost * 0.9, 2),
        "ocean_freight_cost": np.round(units_a * 0.4, 2),
        "payment_cost": np.round(rev_a * 0.03, 2),
        "sales_insurance_cost": np.round(rev_a * 0.005, 2),
        "shipping_cost": np.round(units_a * 3.5, 2),
        "warehouse_handling_cost": np.round(units_a * 0.2, 2),
        "warehouse_storage_fee": np.round(units_a * 0.1, 2),
        "to_door_charge": np.zeros(n),
        "local_charge": np.zeros(n),
        "deal_discount": np.where(deal, -np.round(rev_a * 0.15, 2), 0.0),
        # The real extract carries pre-computed rolling averages that leak the target;
        # the pipeline drops them. They are filled with noise here so that holds.
        "avg_same_weekday": rng.uniform(0, 5, n),
        "avg_daily_weekly": rng.uniform(0, 5, n),
        "avg_daily_monthly": rng.uniform(0, 5, n),
        "avg_daily_monthly_last_year": rng.uniform(0, 5, n),
        "avg_daily_monthly_last_year2": rng.uniform(0, 5, n),
        "product_id": [str(1000 + sorted(set(sku)).index(s)) for s in sku],
        "unit_tax": np.round(rev_a * 0.07, 2),
        "no_oos_demand": rng.uniform(0, 5, n),
        "base_on_last_month_sales_demand": rng.uniform(0, 5, n),
        "daily_fba_storage_fee": np.round(rng.uniform(0, 0.5, n), 3),
        "sessions": sessions.astype(np.int64),
        "minor_rank": rng.integers(1, 5_000, n).astype(np.int64),
        "master_rank": rng.integers(1, 200_000, n).astype(np.int64),
        "coupon_discount": np.zeros(n),
        "daily_fba_long_term_storage_fee": np.zeros(n),
        "refurbished_product_sales_count": rng.binomial(units_a, 0.01).astype(np.int64),
        "business_discount": np.zeros(n),
        "program_fee": np.zeros(n),
        "fba_customer_returns_fee": np.zeros(n),
        "next_season_fba_fee": np.zeros(n),
        "next_season_fba_storage_fee": np.zeros(n),
    })

    m = len(cat)
    ts = lambda d: dt.datetime.combine(d, dt.time(0))  # noqa: E731
    product = pl.DataFrame({
        "id": [str(1000 + s["idx"]) for s in cat],
        "average_cost": [s["cost"] for s in cat],
        "brand_id": [str(1 + "ZAZBZCZD".index(s["sku"][:2]) // 2) for s in cat],
        "carton_dimension_depth": rng.uniform(20, 60, m).round(1),
        "carton_dimension_lengthx": rng.uniform(30, 80, m).round(1),
        "carton_dimension_width": rng.uniform(20, 60, m).round(1),
        "carton_size": rng.choice([6, 12, 24, 48], m).astype(np.int64),
        "creator": pl.Series([None] * m, dtype=pl.Null),
        "date_created": [ts(s["launch"]) for s in cat],
        "date_on_market": [ts(s["launch"]) for s in cat],
        "default_cost": [s["cost"] for s in cat],
        "dimension_depth": rng.uniform(5, 30, m).round(1),
        "dimension_lengthx": rng.uniform(5, 40, m).round(1),
        "dimension_width": rng.uniform(5, 30, m).round(1),
        "is_whole_sale": rng.random(m) < 0.1,
        "item_dimension_depth": rng.uniform(5, 30, m).round(1),
        "item_dimension_lengthx": rng.uniform(5, 40, m).round(1),
        "item_dimension_width": rng.uniform(5, 30, m).round(1),
        "last_updated": [ts(end)] * m,
        "long_description": ["Synthetic sample product"] * m,
        "main_image_id": [None] * m,
        "moq": rng.choice([0, 50, 100, 200], m).astype(np.int64),
        "mpn": [None] * m,
        "name": [f"Sample product {s['sku']}" for s in cat],
        "po_price": [round(s["cost"] * 0.9, 2) for s in cat],
        "sale_price": [s["price"] for s in cat],
        "short_description": [None] * m,
        "sku": [s["sku"] for s in cat],
        "status": np.where([s["retire"] is None for s in cat], 1, 0).astype(np.int64),
        "upc": [None] * m,
        "weight": rng.uniform(0.1, 10, m).round(2),
        "import_product_name": [None] * m,
        "material": rng.choice(["steel", "plastic", "fabric", "rubber"], m),
        "disable_product_review": np.zeros(m, np.int64),
        "master_carton_gtin": [None] * m,
        "channel_category": rng.integers(1, 6, m).astype(np.int64),
        "enable_front_page": np.zeros(m, bool),
        "search_text": [None] * m,
        "pallethi": rng.uniform(3, 8, m).round(0),
        "pallet_rest": np.zeros(m),
        "palletti": rng.uniform(4, 10, m).round(0),
        "is_pallet_ship": rng.random(m) < 0.2,
        "currency": ["USD"] * m,
        "exchange_rate": np.ones(m),
        "revision_id": [None] * m,
        "default_sale_price": [s["price"] for s in cat],
        "type": np.zeros(m, np.int64),
        "uom": np.ones(m, np.int64),
        "from_container_line_id": pl.Series([None] * m, dtype=pl.Int64),
        "is_new_item": [s["launch"] > end - dt.timedelta(days=365) for s in cat],
        "has_arrived": np.ones(m, bool),
        "percent_of_master_sku": np.ones(m),
        "tier": rng.integers(1, 4, m).astype(np.int64),
        "listing_status": ["Active" if s["retire"] is None else "Inactive" for s in cat],
        "buy_box_share": rng.uniform(0.8, 1.0, m).round(3),
    })

    isku, idate, ioh = map(list, zip(*inv))
    k = len(isku)
    inventory = pl.DataFrame({
        "sku": isku, "date": idate, "on_hand": ioh,
        "inbound_working": np.zeros(k), "inbound_shipped": np.zeros(k),
        "inbound_receiving": np.zeros(k), "reserved": np.zeros(k),
        "unfulfillable": np.zeros(k), "cust_reserved": np.zeros(k),
        "n_asins": np.ones(k, np.int64),
    })

    deal_rows = (daily.filter(pl.col("is_deal") == "Y")
                 .with_columns((pl.col("date_purchased").diff().over("sku") != pl.duration(days=1))
                               .fill_null(True).cum_sum().over("sku").alias("_run"))
                 .group_by("sku", "_run")
                 .agg(pl.col("date_purchased").min().alias("s"), pl.col("date_purchased").max().alias("e"),
                      pl.col("salescount").sum().alias("u"), pl.col("revenue").sum().alias("r")))
    nd = deal_rows.height
    deals = pl.DataFrame({
        "amazon_sku": deal_rows["sku"], "asin": [f"SAMPLE{i:05d}" for i in range(nd)],
        "sku": deal_rows["sku"], "deal_type": ["LIGHTNING"] * nd,
        "amazon_deal_type": pl.Series([None] * nd, dtype=pl.Null),
        "start_date": deal_rows["s"].cast(pl.Datetime("us")),
        "end_date": deal_rows["e"].cast(pl.Datetime("us")),
        "your_price": np.full(nd, 20.0), "deal_price": np.full(nd, 17.0),
        "deal_fee": np.full(nd, 70.0),
        "deal_status": rng.choice(["Ended", "Ended", "Ended", "Canceled"], nd),
        "units_sold": deal_rows["u"].cast(pl.Int64), "deal_sales": deal_rows["r"],
        "committed_qty": np.full(nd, 100, np.int64),
        "date_created": deal_rows["s"].cast(pl.Datetime("us")) - dt.timedelta(days=21),
    })

    last = inventory.filter(pl.col("date") == pl.col("date").max())
    position = last.select("sku", pl.col("on_hand").cast(pl.Int64),
                           pl.lit(0, pl.Int64).alias("on_order"))
    return daily, product, inventory, deals, position


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("DEMANDML_DATA", ROOT / "data")))
    ap.add_argument("--skus", type=int, default=120)
    ap.add_argument("--start", default="2022-01-03")
    ap.add_argument("--end", default="2025-12-28")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    cat = make_catalogue(rng, args.skus, start, end)
    sales, inv = simulate_sales(rng, cat, start, end)
    daily, product, inventory, deals, position = build_frames(rng, cat, sales, inv, end)

    db = args.out / "db"
    db.mkdir(parents=True, exist_ok=True)
    (args.out / "inventory").mkdir(parents=True, exist_ok=True)
    daily.write_parquet(db / "sales_daily.parquet")
    product.write_parquet(db / "product.parquet")
    inventory.write_parquet(db / "inventory_daily.parquet")
    deals.write_parquet(db / "deals.parquet")
    position.write_csv(args.out / "inventory" / "position_sample.csv")
    (args.out / "raw").mkdir(parents=True, exist_ok=True)
    daily.write_csv(args.out / "raw" / "daily_sales_sample.csv")
    print(f"wrote {daily.height:,} sales rows for {product.height} SKUs "
          f"({start} .. {end}) and {deals.height} deals under {args.out}")


if __name__ == "__main__":
    main()
