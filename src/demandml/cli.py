"""Command line entry point: ``uv run demandml <command>``.

The whole pipeline is reachable from here, so a fresh clone plus a data extract
(`scripts/make_sample_data.py` for the sample data, your own database extract in
`data/db/`, or a raw CSV drop in `data/raw/`) is three commands away from a leaderboard.
"""

from __future__ import annotations

from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from . import config as C
from .logging_utils import get_logger

app = typer.Typer(add_completion=False, help="Demand prediction research workbench.")
console = Console()
log = get_logger(__name__)


def _show(df: pl.DataFrame, title: str = "") -> None:
    if df.is_empty():
        console.print("[yellow]empty result[/]")
        return
    t = Table(title=title, header_style="bold cyan")
    for c in df.columns:
        t.add_column(c, justify="right" if df.schema[c].is_numeric() else "left")
    for row in df.iter_rows():
        t.add_row(*[f"{v:,.4f}" if isinstance(v, float) else str(v) for v in row])
    console.print(t)


@app.command()
def info() -> None:
    """Show which raw files were found and which build artefacts exist."""
    t = Table(title="demandml paths", header_style="bold cyan")
    t.add_column("artefact")
    t.add_column("path")
    t.add_column("status")
    for name, path in C.Paths().files.items():
        if path is None:
            t.add_row(name, "-", "[red]not found[/]")
            continue
        if path.exists():
            if path.is_dir():
                n = len(list(path.glob("*.parquet")))
                size = sum(p.stat().st_size for p in path.glob("*.parquet"))
                status = f"[green]{n} parts, {size / 1e6:.0f} MB[/]"
            else:
                status = f"[green]{path.stat().st_size / 1e6:.0f} MB[/]"
        else:
            status = "[yellow]missing[/]"
        t.add_row(name, str(path.relative_to(C.ROOT)), status)
    console.print(t)


@app.command()
def ingest(overwrite: bool = typer.Option(
        False, help="Rebuild even if the Parquet is newer than its source")) -> None:
    """Convert the raw sources (DB pull, else the CSV dumps) to typed Parquet."""
    from .ingest import ingest_all

    ingest_all(overwrite=overwrite)


@app.command()
def panel(
    freq: str = typer.Option("daily", help="daily | weekly"),
    min_history: int = typer.Option(120, help="Minimum days of history per SKU"),
    min_units: int = typer.Option(30, help="Minimum lifetime units per SKU"),
) -> None:
    """Clean, trim and densify the sales rows into a modelling panel."""
    from .panel import build_panel

    spec = C.PanelSpec(min_history_days=min_history, min_total_units=min_units)
    df = build_panel(spec=spec, freq=freq)
    console.print(f"[green]panel: {len(df):,} rows x {df.width} cols[/]")


@app.command()
def features(
    freq: str = typer.Option("daily"),
    batch_size: int = typer.Option(250, help="SKUs per Parquet part (controls memory)"),
    sku_limit: Optional[int] = typer.Option(None, help="Only the first N SKUs (fast iteration)"),
    origin_lag: int = typer.Option(0, help="Reporting lag: shift dynamic features back N periods"),
) -> None:
    """Build the leak-free feature table."""
    from .features import build_features

    spec = C.FeatureSpec(origin_lag=origin_lag)
    build_features(freq=freq, spec=spec, batch_size=batch_size, sku_limit=sku_limit)


@app.command()
def profile(freq: str = typer.Option("daily")) -> None:
    """Print the data-quality report for a panel."""
    from .profiling import panel_report

    panel_report(freq=freq, console=console)


@app.command("models")
def models_cmd() -> None:
    """List every registered model."""
    from .models import list_models

    t = Table(title="registered models", header_style="bold cyan")
    t.add_column("name")
    t.add_column("description")
    for n, note in list_models():
        t.add_row(n, note)
    console.print(t)


@app.command()
def train(
    models: str = typer.Option("ma28,croston,lgbm_tweedie", help="Comma-separated model names"),
    horizon: int = typer.Option(7, help="Forecast horizon in periods"),
    freq: str = typer.Option("daily"),
    folds: int = typer.Option(3, help="Number of rolling-origin backtest folds"),
    test_size: int = typer.Option(28, help="Periods per test window"),
    sku_limit: Optional[int] = typer.Option(None),
    max_train_rows: Optional[int] = typer.Option(None, help="Cap training rows (fast iteration)"),
    keep_censored: bool = typer.Option(False, help="Score on stock-out days too"),
    baseline: str = typer.Option("ma28", help="Baseline for the improvement column"),
) -> None:
    """Backtest one or more models and print the leaderboard."""
    from .experiment import RunConfig, run

    cfg = RunConfig(
        models=[m.strip() for m in models.split(",") if m.strip()],
        horizon=horizon,
        freq=freq,
        splits=C.SplitSpec(test_size=test_size, n_folds=folds),
        drop_censored=not keep_censored,
        sku_limit=sku_limit,
        max_train_rows=max_train_rows,
    )
    res = run(cfg)
    _show(res.vs_baseline(baseline), f"leaderboard  (h={horizon}, {freq})")
    console.print(f"[dim]run saved as {res.run_id}[/]")


@app.command()
def sweep(
    models: str = typer.Option("ma28,lgbm_tweedie"),
    horizons: str = typer.Option("1,7,14,28", help="Comma-separated horizons"),
    folds: int = typer.Option(2),
    sku_limit: Optional[int] = typer.Option(None),
) -> None:
    """Backtest across several horizons -- accuracy decays with h, and by how much is
    the number that decides your replenishment cadence."""
    from .experiment import RunConfig, run

    rows = []
    for h in [int(x) for x in horizons.split(",")]:
        cfg = RunConfig(
            models=[m.strip() for m in models.split(",")],
            horizon=h,
            splits=C.SplitSpec(n_folds=folds),
            sku_limit=sku_limit,
        )
        res = run(cfg)
        rows.append(res.leaderboard().with_columns(pl.lit(h).alias("h")))
    _show(pl.concat(rows).select("h", "model", "wape", "bias", "mase"), "horizon sweep")


@app.command()
def pipeline(
    overwrite: bool = typer.Option(False),
    sku_limit: Optional[int] = typer.Option(None),
) -> None:
    """Run the whole thing: ingest -> panel -> features -> a first leaderboard."""
    from .experiment import RunConfig, run
    from .features import build_features
    from .ingest import ingest_all
    from .panel import build_panel

    ingest_all(overwrite=overwrite)
    build_panel(freq="daily")
    build_features(freq="daily", sku_limit=sku_limit)
    res = run(
        RunConfig(
            models=["ma28", "croston", "lgbm_tweedie"],
            horizon=7,
            splits=C.SplitSpec(n_folds=2),
            sku_limit=sku_limit,
        )
    )
    _show(res.vs_baseline("ma28"), "first leaderboard (h=7)")


@app.command()
def report(
    models: str = typer.Option("ma7,ma28,croston,lgbm_tweedie,two_stage",
                               help="Comma-separated model names"),
    horizon: int = typer.Option(7, help="Forecast horizon in days"),
    folds: int = typer.Option(3, help="Rolling-origin backtest folds"),
    sku_limit: Optional[int] = typer.Option(200, help="SKUs to use (None = all)"),
    max_train_rows: Optional[int] = typer.Option(220_000, help="Cap training rows for speed"),
    quick: bool = typer.Option(False, help="Skip the horizon sweep and the quantile figure"),
    open_figures: bool = typer.Option(True, "--open/--no-open",
                                      help="Open the figures when finished (macOS)"),
) -> None:
    """Run the models and write a set of figures to reports/figures/.

    This is the visual entry point: a leaderboard, forecasts against the real series,
    error diagnostics, a per-demand-class breakdown, the horizon decay curve, and
    calibrated prediction intervals. --quick skips the two slowest parts.
    """
    import subprocess
    import sys

    from .report import build_report

    paths = build_report(
        models=[m.strip() for m in models.split(",") if m.strip()],
        horizon=horizon,
        folds=folds,
        sku_limit=sku_limit,
        max_train_rows=max_train_rows,
        skip_sweep=quick,
        skip_quantiles=quick,
    )
    t = Table(title="figures written", header_style="bold cyan")
    t.add_column("figure")
    t.add_column("size", justify="right")
    for p in paths:
        t.add_row(str(p.relative_to(C.ROOT)), f"{p.stat().st_size / 1e3:.0f} KB")
    console.print(t)
    if open_figures and sys.platform == "darwin":
        subprocess.run(["open", *[str(p) for p in paths]], check=False)


@app.command()
def runs(limit: int = typer.Option(20)) -> None:
    """List saved experiment runs."""
    import json

    rows = []
    for d in sorted(C.EXPERIMENTS.glob("*/meta.json"), reverse=True)[:limit]:
        m = json.loads(d.read_text())
        lb = d.parent / "leaderboard.csv"
        best = ""
        if lb.exists():
            t = pl.read_csv(lb)
            if not t.is_empty():
                best = f"{t.get_column('model')[0]} wape={t.get_column('wape')[0]:.4f}"
        rows.append(
            {
                "run_id": m["run_id"],
                "h": m["config"]["horizon"],
                "models": ",".join(m["config"]["models"])[:40],
                "best": best,
                "secs": m["elapsed_s"],
            }
        )
    _show(pl.DataFrame(rows), "saved runs") if rows else console.print("[yellow]no runs yet[/]")


if __name__ == "__main__":
    app()
