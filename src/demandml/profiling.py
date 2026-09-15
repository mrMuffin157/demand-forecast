"""Data-quality profiling.

Run this before trusting any model score. It answers the questions that decide whether a
result is real: how sparse is the target, how much of it is stock-out censored, how many
SKUs have enough history, and is the recent data actually complete (a truncated final
week silently drags every backtest down).
"""

from __future__ import annotations

import polars as pl

from . import config as C
from .logging_utils import get_logger

log = get_logger(__name__)


def column_report(lf: pl.LazyFrame, sample: int | None = 500_000) -> pl.DataFrame:
    """Per-column null rate, distinct count and range. The first thing to look at."""
    if sample:
        lf = lf.head(sample)
    df = lf.collect()
    rows = []
    for c, dtype in df.schema.items():
        s = df.get_column(c)
        row = {
            "column": c,
            "dtype": str(dtype),
            "null_pct": round(100.0 * s.null_count() / max(len(s), 1), 2),
            "n_unique": s.n_unique(),
        }
        if dtype.is_numeric():
            row |= {
                "min": _fmt(s.min()),
                "median": _fmt(s.median()),
                "max": _fmt(s.max()),
                "zero_pct": round(100.0 * (s == 0).sum() / max(len(s), 1), 2),
            }
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort("null_pct", descending=True)


def _fmt(v):
    if v is None:
        return None
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return str(v)


def demand_classes(lf: pl.LazyFrame, target: str = C.TARGET) -> pl.DataFrame:
    """Classify every SKU into the Syntetos-Boylan quadrants.

    The cut-offs (ADI 1.32, CV2 0.49) are the standard ones. This matters because the four
    classes want different models: smooth series reward regression, lumpy ones reward
    Croston-style or quantile approaches, and a single average WAPE hides all of it.
    """
    per_sku = (
        lf.group_by(C.SKU)
        .agg(
            pl.len().alias("n_days"),
            pl.col(target).sum().alias("total_units"),
            (pl.col(target) > 0).sum().alias("nonzero_days"),
            pl.when(pl.col(target) > 0).then(pl.col(target)).mean().alias("nonzero_mean"),
            pl.when(pl.col(target) > 0).then(pl.col(target)).std().alias("nonzero_std"),
            pl.col("is_oos").mean().alias("oos_rate"),
        )
        .with_columns(
            (pl.col("n_days") / pl.col("nonzero_days")).alias("adi"),
            (pl.col("nonzero_std") / pl.col("nonzero_mean")).pow(2).alias("cv2"),
        )
        .with_columns(
            pl.when((pl.col("adi") < 1.32) & (pl.col("cv2") < 0.49))
            .then(pl.lit("smooth"))
            .when((pl.col("adi") < 1.32) & (pl.col("cv2") >= 0.49))
            .then(pl.lit("erratic"))
            .when((pl.col("adi") >= 1.32) & (pl.col("cv2") < 0.49))
            .then(pl.lit("intermittent"))
            .otherwise(pl.lit("lumpy"))
            .alias("demand_class")
        )
    )
    return per_sku.collect()


def panel_report(freq: str = "daily", console=None) -> dict:
    """Print the headline data-quality numbers for a built panel."""
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    from .panel import load_panel

    lf = load_panel(freq)
    target = C.TARGET

    agg = lf.select(
        pl.len().alias("rows"),
        pl.col(C.SKU).n_unique().alias("skus"),
        pl.col(C.DATE).min().alias("first_date"),
        pl.col(C.DATE).max().alias("last_date"),
        pl.col(C.DATE).n_unique().alias("dates"),
        pl.col(target).sum().alias("total_units"),
        (pl.col(target) == 0).mean().alias("zero_rate"),
        pl.col("is_oos").mean().alias("oos_rate"),
        (pl.col("is_oos") & (pl.col(target) == 0)).mean().alias("censored_zero_rate"),
        pl.col("is_imputed_row").mean().alias("gap_filled_rate"),
        pl.col(target).max().alias("max_day_units"),
        pl.col(target).quantile(0.99).alias("p99_day_units"),
        pl.col(target).mean().alias("mean_day_units"),
    ).collect()

    t = Table(title=f"{freq} panel -- data quality", header_style="bold cyan")
    t.add_column("metric")
    t.add_column("value", justify="right")
    row = agg.row(0, named=True)
    for k, v in row.items():
        if isinstance(v, float):
            v = f"{v:,.4f}" if abs(v) < 100 else f"{v:,.0f}"
        elif isinstance(v, int):
            v = f"{v:,}"
        t.add_row(k, str(v))
    console.print(t)

    cls = demand_classes(lf)
    summary = (
        cls.group_by("demand_class")
        .agg(
            pl.len().alias("skus"),
            pl.col("total_units").sum().alias("units"),
            pl.col("adi").median().round(2).alias("median_adi"),
            pl.col("cv2").median().round(2).alias("median_cv2"),
            pl.col("oos_rate").mean().round(3).alias("mean_oos_rate"),
        )
        .with_columns(
            (100 * pl.col("units") / pl.col("units").sum()).round(1).alias("units_pct")
        )
        .sort("units", descending=True)
    )
    t2 = Table(title="demand classes (Syntetos-Boylan)", header_style="bold cyan")
    for c in summary.columns:
        t2.add_column(c, justify="right")
    for r in summary.iter_rows():
        t2.add_row(*[f"{v:,}" if isinstance(v, int) else str(v) for v in r])
    console.print(t2)

    # A truncated tail is the most common silent killer of a backtest.
    tail = (
        lf.filter(pl.col(C.DATE) >= pl.col(C.DATE).max() - pl.duration(days=14))
        .group_by(C.DATE)
        .agg(pl.len().alias("skus"), pl.col(target).sum().alias("units"))
        .sort(C.DATE)
        .collect()
    )
    t3 = Table(title="last 15 periods -- check the tail is complete", header_style="bold cyan")
    for c in tail.columns:
        t3.add_column(c, justify="right")
    for r in tail.iter_rows():
        t3.add_row(*[f"{v:,.0f}" if isinstance(v, float) else str(v) for v in r])
    console.print(t3)

    return {"summary": row, "classes": summary, "tail": tail}
