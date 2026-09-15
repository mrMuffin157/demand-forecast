"""Central paths, raw-file locations and column groupings.

Everything downstream imports from here so that a schema surprise only has to be
fixed in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path(os.environ.get("DEMANDML_ROOT", Path(__file__).resolve().parents[2]))

# `DEMANDML_DATA` is separate from `DEMANDML_ROOT` on purpose. When the pipeline runs as a
# container the code and the data do not live together: the code is baked into the image
# while `data/` is a mounted volume, and a multi-gigabyte panel has no business sitting
# inside an image. Locally both default to the repo, so nothing changes.
DATA = Path(os.environ.get("DEMANDML_DATA", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXPERIMENTS = Path(os.environ.get("DEMANDML_EXPERIMENTS", ROOT / "experiments"))

# `DEMANDML_REPORTS` for the same reason as the two above, and it is not optional in a
# container: these paths are CREATED at import time, so with the code tree read-only --
# a root-owned /app, or a read-only root filesystem, both of which are ordinary container
# hardening -- `import demandml` raises PermissionError before anything runs.
# Reports are output artefacts and belong on the data volume, not in the code tree.
REPORTS = Path(os.environ.get("DEMANDML_REPORTS", ROOT / "reports"))
FIGURES = REPORTS / "figures"

for _p in (INTERIM, PROCESSED, EXPERIMENTS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)


# Where the database extract lives (for the sample data, `scripts/make_sample_data.py`
# writes it; otherwise your own extract). Kept here so every module resolves it the same
# way -- modules that resolve it from their own location while others go through `config`
# read different directories with no error once the volume is mounted anywhere but the repo.
DB_EXTRACT = DATA / "db"


def _find_raw(prefix: str, pulled: str | None = None) -> Path | None:
    """Locate a raw source: the database pull if there is one, else the newest CSV.

    `pulled` names a table in `DB_EXTRACT` (the database extract). It WINS over the CSV
    when present, and the order is deliberate rather than arbitrary: the pull is the table
    the CSV was exported from -- identical column set -- and it is the only one of the two
    that a scheduled job can refresh. A CSV in `data/raw/` is a snapshot somebody took by
    hand on a date nobody recorded.

    Nothing here compares freshness. Choosing by mtime would make the pipeline's input
    depend on file metadata that a copy or an S3 sync rewrites, and a rule you cannot
    predict from the filesystem listing is worse than one you can. `datacheck.py`
    fingerprints date coverage on every build and is what catches a stale input, whichever
    of the two it came from -- so the job of this function is to be predictable, not
    clever.
    """
    if pulled:
        p = DB_EXTRACT / f"{pulled}.parquet"
        if p.exists():
            return p
    matches = sorted(RAW.glob(f"{prefix}*.csv"))
    return matches[-1] if matches else None


# Only the daily table is wired to the pull. The weekly one is NOT, and that is a finding
# rather than an omission: the database's weekly table held noticeably fewer rows than the
# weekly export, so the two are not the same population and swapping them blind would
# change the weekly panel for unexplained reasons. The daily pipeline does not read it
# (`panel --freq daily` calls `clean_daily` only), so nothing is blocked on resolving it.
# The product master comes from the pull too (`SELECT * FROM product`), with the CSV as
# the fallback. CSV-only would fail on a fresh data volume, which holds no CSVs -- the
# first build would stop at `demandml ingest`.
RAW_DAILY = _find_raw("daily_sales", pulled="sales_daily")
RAW_WEEKLY = _find_raw("weekly_sales")
RAW_PRODUCT = _find_raw("product_", pulled="product")

# Parquet mirrors of the raw sources -- the DB pull or the CSV fallback (typed,
# compressed, ~10x smaller than the CSV, lazily scannable).
PQ_DAILY = INTERIM / "daily_sales.parquet"
PQ_WEEKLY = INTERIM / "weekly_sales.parquet"
PQ_PRODUCT = INTERIM / "product.parquet"

# Cleaned + densified panels.
PANEL_DAILY = PROCESSED / "panel_daily.parquet"
PANEL_WEEKLY = PROCESSED / "panel_weekly.parquet"
# Feature tables are written as a directory of Parquet parts (one per SKU batch) so that
# building them never needs the whole panel in memory at once.
FEATURES_DAILY = PROCESSED / "features_daily"
FEATURES_WEEKLY = PROCESSED / "features_weekly"

# --------------------------------------------------------------------------------------
# Keys / targets
# --------------------------------------------------------------------------------------

SKU = "sku"
DATE = "date"

#: Raw units sold. Censored by stock-outs and inflated by occasional bulk orders.
TARGET_RAW = "salescount"
#: Units sold with bulk/wholesale orders filtered out -- the better retail-demand proxy.
TARGET_FILTERED = "units_filtered"
#: Default modelling target (see clean.py for how it is derived).
TARGET = "units"

# --------------------------------------------------------------------------------------
# Column groups in the DAILY export
# --------------------------------------------------------------------------------------

#: Booleans stored as the strings "Y" / "N".
DAILY_YN_COLS = ["is_oos", "is_deal"]

DAILY_ID_COLS = ["id", "sku", "product_id", "date_purchased", "date_created", "last_updated"]

#: Demand / volume signals.
DAILY_VOLUME_COLS = [
    "salescount",
    "big_order_filter_sales_count",
    "order_count",
    "refurbished_product_sales_count",
    "no_oos_demand",
    "base_on_last_month_sales_demand",
]

#: Money in / out on the day.
DAILY_MONEY_COLS = [
    "revenue",
    "profit",
    "final_profit",
    "return_amount",
    "total_item_cost",
    "total_po_price",
    "duty",
    "fba_destination_fee",
    "ocean_freight_cost",
    "payment_cost",
    "sales_insurance_cost",
    "shipping_cost",
    "warehouse_handling_cost",
    "warehouse_storage_fee",
    "to_door_charge",
    "local_charge",
    "unit_tax",
    "daily_fba_storage_fee",
    "daily_fba_long_term_storage_fee",
    "fba_customer_returns_fee",
    "program_fee",
    "next_season_fba_fee",
    "next_season_fba_storage_fee",
]

#: Promotions / price levers.
DAILY_PROMO_COLS = ["deal_discount", "coupon_discount", "business_discount", "is_deal"]

#: Advertising + organic traffic.
DAILY_MARKETING_COLS = ["ad_cost", "ad_units", "impressions", "clicks", "sessions"]

#: Marketplace visibility.
DAILY_RANK_COLS = ["minor_rank", "master_rank"]

#: Availability.
DAILY_STOCK_COLS = ["is_oos", "oos_threshold"]

#: Pre-computed rolling averages present in the export. They are almost certainly
#: calculated *including* the row's own day, so they leak the target. We recompute our
#: own shifted versions in features.py and drop these.
DAILY_LEAKY_COLS = [
    "avg_same_weekday",
    "avg_daily_weekly",
    "avg_daily_monthly",
    "avg_daily_monthly_last_year",
    "avg_daily_monthly_last_year2",
    "no_oos_demand",
    "base_on_last_month_sales_demand",
]

# --------------------------------------------------------------------------------------
# Column groups in the WEEKLY export
# --------------------------------------------------------------------------------------

WEEKLY_LEAKY_COLS = [
    "avg2weeks",
    "avg4weeks",
    "avg8weeks",
    "stdev2weeks",
    "stdev4weeks",
    "stdev8weeks",
]

#: Comma-packed rank strings, e.g. "0,0,0,0,0,0,0" (one value per day of the week).
WEEKLY_PACKED_COLS = ["master_ranks", "minor_ranks"]

# --------------------------------------------------------------------------------------
# Column groups in the PRODUCT export
# --------------------------------------------------------------------------------------

PRODUCT_KEEP_COLS = [
    "id",
    "sku",
    "name",
    "brand_id",
    "channel_category",
    "type",
    "uom",
    "tier",
    "listing_status",
    "status",
    "material",
    "is_whole_sale",
    "is_pallet_ship",
    "is_new_item",
    "has_arrived",
    "moq",
    "weight",
    "average_cost",
    "default_cost",
    "po_price",
    "sale_price",
    "default_sale_price",
    "percent_of_master_sku",
    "buy_box_share",
    "carton_size",
    "dimension_depth",
    "dimension_lengthx",
    "dimension_width",
    "item_dimension_depth",
    "item_dimension_lengthx",
    "item_dimension_width",
    "carton_dimension_depth",
    "carton_dimension_lengthx",
    "carton_dimension_width",
    "pallethi",
    "palletti",
    "pallet_rest",
    "currency",
    "date_created",
    "date_on_market",
]

PRODUCT_CATEGORICALS = [
    "brand_id",
    "channel_category",
    "type",
    "uom",
    "tier",
    "listing_status",
    "status",
    "material",
    "currency",
]

# --------------------------------------------------------------------------------------
# Modelling defaults
# --------------------------------------------------------------------------------------


@dataclass
class PanelSpec:
    """How to turn the raw event rows into a dense modelling panel."""

    #: Drop a SKU's history before its first-ever sale (pre-launch zeros are not demand).
    trim_leading_zeros: bool = True
    #: Drop a SKU's history after its last sale (delisted / discontinued).
    trim_trailing_zeros: bool = True
    #: A SKU needs at least this many observed days to be modelled at all.
    min_history_days: int = 120
    #: A SKU needs at least this many total units over its life to be modelled.
    min_total_units: int = 30
    #: Cap per-day units at this rolling quantile to tame bulk-order spikes
    #: (only used when ``target == "units_winsorized"``).
    winsor_quantile: float = 0.995
    #: Drop trailing periods whose total volume is below this fraction of the recent
    #: norm. The export's final day is partial (the report was generated mid-day), which
    #: would read as a demand collapse to any model. 0 disables the check.
    incomplete_tail_threshold: float = 0.5


#: Holiday calendar per store (`dev/market.py` names the stores). The store's own
#: country, so a CA model learns Canada Day and Canadian Thanksgiving (October), and an
#: AU model none of the US holidays. `tests/test_market.py` pins that this agrees with
#: `market.MARKETS`.
MARKET_COUNTRY = {"us": "US", "us2": "US", "ca": "CA", "au": "AU"}


def _market_country() -> str:
    key = os.environ.get("DEMANDML_MARKET", "us").strip().lower()
    return MARKET_COUNTRY.get(key, "US")


@dataclass
class FeatureSpec:
    """Which lag/rolling features to generate. All windows are strictly past-only."""

    lags: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 14, 21, 28, 35, 42, 56, 91, 182, 364)
    roll_windows: tuple[int, ...] = (7, 14, 28, 56, 91, 182, 364)
    roll_stats: tuple[str, ...] = ("mean", "std", "median", "max", "zero_rate")
    #: Same-weekday-of-past-N-weeks averages.
    same_dow_windows: tuple[int, ...] = (4, 8, 13, 26, 52)
    #: Exponentially weighted means (alpha values).
    ewm_alphas: tuple[float, ...] = (0.1, 0.3, 0.6)
    #: Exogenous columns to lag/roll alongside the target.
    exog: tuple[str, ...] = (
        "sessions",
        "ad_cost",
        "impressions",
        "clicks",
        "ad_units",
        "unit_price",
        "is_oos",
        "discount_rate",
        "minor_rank",
        "master_rank",
        "order_count",
        "return_rate",
    )
    exog_lags: tuple[int, ...] = (0, 1, 7, 28)
    exog_roll_windows: tuple[int, ...] = (7, 28, 91)
    #: Columns that are genuinely knowable at forecast time (planned promos, calendar).
    known_future: tuple[str, ...] = ("is_deal", "unit_price")
    include_static: bool = True
    country: str = field(default_factory=_market_country)
    #: Shift every dynamic feature back this many periods, to mimic a reporting lag
    #: (if yesterday's sales are not in the warehouse until T+2, set this to 2).
    origin_lag: int = 0
    #: Store features as float32 -- halves the feature table with no measurable
    #: accuracy cost for tree models.
    float32: bool = True


@dataclass
class SplitSpec:
    """Rolling-origin backtest configuration."""

    #: Forecast horizons (days for the daily panel, weeks for the weekly panel).
    horizons: tuple[int, ...] = (1, 7, 14, 28)
    #: Length of each test window.
    test_size: int = 28
    #: How many backtest folds.
    n_folds: int = 4
    #: Gap between train end and test start, to mimic the real data-availability lag.
    gap: int = 0
    #: Optional cap on training window length (None = expanding window).
    max_train_size: int | None = None


DEFAULT_PANEL = PanelSpec()
DEFAULT_FEATURES = FeatureSpec()
DEFAULT_SPLITS = SplitSpec()


@dataclass
class Paths:
    """Convenience bundle, handy in notebooks: ``from demandml.config import Paths``."""

    root: Path = ROOT
    raw: Path = RAW
    interim: Path = INTERIM
    processed: Path = PROCESSED
    experiments: Path = EXPERIMENTS
    reports: Path = REPORTS
    figures: Path = FIGURES
    files: dict[str, Path | None] = field(
        default_factory=lambda: {
            "raw_daily": RAW_DAILY,
            "raw_weekly": RAW_WEEKLY,
            "raw_product": RAW_PRODUCT,
            "pq_daily": PQ_DAILY,
            "pq_weekly": PQ_WEEKLY,
            "pq_product": PQ_PRODUCT,
            "panel_daily": PANEL_DAILY,
            "panel_weekly": PANEL_WEEKLY,
            "features_daily": FEATURES_DAILY,
            "features_weekly": FEATURES_WEEKLY,
        }
    )


# --------------------------------------------------------------------------------------
# Which columns may be handed to a model
# --------------------------------------------------------------------------------------
#
# Feature selection is an ALLOWLIST, not a denylist. A denylist has the wrong polarity
# here: when a future export adds a column, a denylist silently feeds it to the model,
# and if that column happens to be same-day (a cost that is units x unit_cost, say) it
# swamps the real features. With an allowlist a new column is ignored until someone
# deliberately classifies it.

#: Genuinely static (or very slowly varying) per-SKU attributes. Safe at any horizon.
STATIC_FEATURE_COLS = frozenset({
    # product master
    "brand_id", "channel_category", "type", "uom", "tier", "listing_status", "status",
    "material", "currency", "is_whole_sale", "is_pallet_ship", "is_new_item",
    "has_arrived", "moq", "weight", "carton_size",
    # list prices and costs -- catalogue attributes, not same-day measurements
    "average_cost", "default_cost", "po_price", "sale_price", "default_sale_price",
    "list_margin", "percent_of_master_sku", "buy_box_share",
    # physical dimensions
    "dimension_depth", "dimension_lengthx", "dimension_width",
    "item_dimension_depth", "item_dimension_lengthx", "item_dimension_width",
    "carton_dimension_depth", "carton_dimension_lengthx", "carton_dimension_width",
    "item_volume", "carton_volume", "pallethi", "palletti", "pallet_rest",
    # SKU-code hierarchy and age
    "sku_brand", "sku_family", "sku_len", "days_on_market",
})

#: Calendar columns, knowable arbitrarily far ahead.
CALENDAR_FEATURE_COLS = frozenset({
    "dow", "dom", "doy", "iso_week", "month", "quarter", "year", "t_index",
    "is_weekend", "is_month_start", "is_month_end", "is_holiday", "retail_event",
    "days_to_christmas",
})

#: Name patterns produced by features.py. Every one of these is built from a shifted,
#: past-only window (or is a sanctioned known-future ``kf_`` column).
ENGINEERED_PREFIXES = (
    "fourier_", "days_to_", "kf_", "fam_", "adi_", "cv2_", "nonzero_mean_",
    "days_since_",
)
ENGINEERED_SUBSTRINGS = (
    "_lag_", "_mean_", "_std_", "_median_", "_max_", "_zero_rate_", "_ewm_",
    "_ratio_", "_diff_", "_cv_", "_momentum", "_yoy_", "same_dow", "dow_mean",
)
