"""Render a visual backtest report: run the models, write figures to reports/figures/.

The figures are deliberately the ones that change a decision:

* which model actually won, and whether the margin clears the fold-to-fold noise
* what the forecasts look like against the real series
* how fast accuracy decays with horizon (this sets your replenishment cadence)
* where the error is biased (weekday, demand level) rather than just how big it is
* whether the prediction intervals are calibrated enough to size safety stock

Charting conventions follow one palette with a fixed hue order: magnitude charts use a
single blue; multi-series charts take blue / orange / aqua in that order and are always
both legended and direct-labelled, so identity never rests on colour alone. Signed
quantities (bias) use the diverging blue-red pair around a neutral midpoint. Grid and
axes are recessive. Never a second y-axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from . import config as C
from .logging_utils import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Palette (validated: all-pairs CVD dE 9.2, normal-vision dE 24.0 on the light surface)
# --------------------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua -- fixed order
SEQ = {100: "#cde2fb", 200: "#9ec5f4", 300: "#6da7ec", 450: "#2a78d6", 600: "#184f95"}
DIV_POS, DIV_NEG, DIV_MID = "#2a78d6", "#d03b3b", "#f0efec"
HILITE = "#184f95"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "font.size": 9.5,
        "figure.dpi": 130,
    })


def _date_axis(ax, n_ticks: int = 5) -> None:
    """Thin and format the date ticks.

    Matplotlib's default on a 40-day span emits one tick per week at full ISO width,
    which runs the labels together ("2026-07-012026-07-08"). Cap the count and drop the
    year, which is constant across any single test window.
    """
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=n_ticks))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_horizontalalignment("center")


def _finish(fig, ax_or_axes, path: Path, subtitle: str = "", has_suptitle: bool = False) -> Path:
    if subtitle:
        # Sits under the suptitle when there is one, otherwise at the top of the figure.
        fig.text(0.008, 0.932 if has_suptitle else 0.975, subtitle,
                 fontsize=8.5, color=MUTED, ha="left", va="top")
    top = 0.90 if (subtitle and has_suptitle) else (0.94 if subtitle else 1.0)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.name)
    return path


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_leaderboard(res, out: Path, baseline: str = "ma28") -> Path:
    """Horizontal bars of WAPE, best at top, with the cross-fold spread as error bars.

    One measure, one hue -- colour here would be decoration, so the baseline is called
    out by a reference line and a darker bar instead. The error bars are the point of
    the chart: a bar that beats the baseline by less than its own whisker has not won.
    """
    lb = res.leaderboard().filter(pl.col("model") != "global_mean").sort("wape", descending=True)
    models = lb.get_column("model").to_list()
    wape = np.array(lb.get_column("wape").to_list(), dtype=float)
    sd = np.nan_to_num(np.array(lb.get_column("wape_sd").to_list(), dtype=float))
    base = float(lb.filter(pl.col("model") == baseline).get_column("wape")[0]) if baseline in models else None

    fig, ax = plt.subplots(figsize=(8.8, 0.46 * len(models) + 2.0))
    colors = [HILITE if m == baseline else SEQ[450] for m in models]
    y = np.arange(len(models))
    ax.barh(y, wape, height=0.5, color=colors, zorder=3)
    ax.errorbar(wape, y, xerr=sd, fmt="none", ecolor=INK_2, elinewidth=1.1,
                capsize=3.0, capthick=1.1, zorder=4)

    xmax = float((wape + sd).max()) * 1.26
    if base is not None:
        ax.axvline(base, color=INK_2, lw=1.3, ls=(0, (4, 3)), zorder=2)
        ax.text(base, len(models) - 0.42, f" {baseline} baseline", fontsize=8.5,
                color=INK_2, va="bottom", ha="left")

    # Value labels go past the END of each whisker, not next to the bar -- otherwise they
    # print on top of the error-bar cap, which is the one thing this chart is for.
    for yi, (w, sdv) in enumerate(zip(wape, sd)):
        ax.text(w + sdv + xmax * 0.018, yi, f"{w:.3f}", va="center", ha="left",
                fontsize=8.5, color=INK)

    ax.set_yticks(y, models, fontsize=9.5, color=INK)
    ax.set_xlabel("WAPE  —  total absolute error / total actual units  (lower is better)")
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    # suptitle rather than ax.set_title so the heading sits above the subtitle line.
    fig.suptitle("Which model actually won", fontsize=11.5, fontweight="semibold",
                 x=0.008, ha="left", y=0.995)
    return _finish(fig, ax, out,
                   f"horizon {res.config.horizon} · {len(res.folds)} rolling-origin folds "
                   f"· whiskers = cross-fold sd · a bar that beats the baseline by less "
                   f"than its whisker has not won", has_suptitle=True)


def fig_forecast_vs_actual(res, out: Path, model: str, n: int = 4) -> Path:
    """Small multiples of the highest-volume SKUs: actual against forecast.

    Two series, so a legend is present and both lines are direct-labelled at their right
    edge. Same y-scale is deliberately *not* shared -- these SKUs differ by an order of
    magnitude and a shared scale would flatten three of the four panels.
    """
    p = res.predictions.filter(pl.col("model") == model)
    if p.is_empty():
        raise ValueError(f"no predictions for {model!r}")
    top = (p.group_by(C.SKU).agg(pl.col("y").sum().alias("v"))
             .sort("v", descending=True).head(n).get_column(C.SKU).to_list())

    ncol = 2
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.4, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for i, (sku, ax) in enumerate(zip(top, axes)):
        d = p.filter(pl.col(C.SKU) == sku).sort(C.DATE)
        x = d.get_column(C.DATE).to_list()
        ya = d.get_column("y").to_numpy()
        yh = d.get_column("yhat").to_numpy()
        ax.plot(x, ya, color=SERIES[0], lw=2.0, label="actual", zorder=3)
        ax.plot(x, yh, color=SERIES[1], lw=2.0, label="forecast", zorder=4)
        wape = np.abs(ya - yh).sum() / max(ya.sum(), 1e-9)
        ax.set_title(f"{sku}   ·   WAPE {wape:.3f}", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        _date_axis(ax)
        for lbl, arr, col in (("actual", ya, SERIES[0]), ("forecast", yh, SERIES[1])):
            ax.annotate(lbl, (x[-1], arr[-1]), xytext=(5, 0), textcoords="offset points",
                        color=col, fontsize=8, va="center", fontweight="semibold")
        if i == 0:
            ax.legend(loc="upper left", ncols=2)
    for ax in axes[len(top):]:
        ax.set_visible(False)

    fig.suptitle("Forecasts against the real series  (test windows only)",
                 fontsize=11.5, fontweight="semibold", x=0.008, ha="left", y=0.995)
    return _finish(fig, axes, out,
                   f"{model} · horizon {res.config.horizon} · the 4 highest-volume SKUs "
                   f"in the test set", has_suptitle=True)


def fig_horizon_decay(sweep: pl.DataFrame, out: Path) -> Path:
    """WAPE against forecast horizon, one line per model.

    This is the curve that sets replenishment cadence: if error doubles by day 28, a
    monthly ordering cycle needs materially more safety stock than a weekly one.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    models = sweep.get_column("model").unique(maintain_order=True).to_list()
    for i, m in enumerate(models):
        d = sweep.filter(pl.col("model") == m).sort("h")
        x = d.get_column("h").to_numpy()
        y = d.get_column("wape").to_numpy()
        col = SERIES[i % len(SERIES)]
        ax.plot(x, y, color=col, marker="o", markersize=6.5, label=m,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3 + i)
        ax.annotate(m, (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                    color=col, fontsize=8.5, va="center", fontweight="semibold")
    ax.set_xlabel("forecast horizon (days ahead)")
    ax.set_ylabel("WAPE")
    ax.set_title("Accuracy decays with horizon")
    ax.set_xticks(sorted(sweep.get_column("h").unique().to_list()))
    ax.legend(loc="upper left")
    ax.margins(x=0.16)
    ax.grid(axis="x", visible=False)
    return _finish(fig, ax, out, "each point is a separate direct multi-horizon model")


def fig_error_diagnostics(res, out: Path, model: str) -> Path:
    """Where the error is, not just how big it is.

    Left: mean signed error by weekday, on the diverging blue-red pair around a neutral
    midpoint -- over- and under-forecasting are opposite failures with different costs,
    so they get opposite hues. Right: WAPE by actual-demand bucket, a magnitude chart, so
    one hue.
    """
    p = res.predictions.filter(pl.col("model") == model).with_columns(
        (pl.col("yhat") - pl.col("y")).alias("err"),
        pl.col(C.DATE).dt.weekday().alias("dow"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    dow = p.group_by("dow").agg(pl.col("err").mean().alias("err")).sort("dow")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    vals = dow.get_column("err").to_numpy()
    ax = axes[0]
    ax.bar(np.arange(len(vals)), vals, width=0.62,
           color=[DIV_POS if v >= 0 else DIV_NEG for v in vals], zorder=3)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=4)
    ax.set_xticks(np.arange(len(vals)), [names[i - 1] for i in dow.get_column("dow")])
    ax.set_ylabel("mean signed error (units)")
    ax.set_title("Bias by weekday")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    lim = np.abs(vals).max() * 1.45
    ax.set_ylim(-lim, lim)
    for i, v in enumerate(vals):
        ax.text(i, v + np.sign(v) * lim * 0.06, f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=DIV_POS, label="over-forecast"),
                       Line2D([], [], marker="s", ls="", color=DIV_NEG, label="under-forecast")],
              loc="upper left", ncols=2)

    ax = axes[1]
    b = (p.with_columns(pl.col("y").cut([0.5, 2, 5, 10, 25, 100]).alias("bucket"))
           .group_by("bucket")
           .agg((pl.col("y") - pl.col("yhat")).abs().sum().alias("ae"),
                pl.col("y").sum().alias("act"), pl.len().alias("n"))
           .with_columns(pl.when(pl.col("act") > 0)
                           .then(pl.col("ae") / pl.col("act"))
                           .otherwise(None).alias("wape"))
           .sort("bucket"))
    labels = ["0", "1-2", "3-5", "6-10", "11-25", "26-100", "100+"]
    wv = [v if v is not None else 0.0 for v in b.get_column("wape").to_list()]
    labels = labels[: len(wv)]
    ax.bar(np.arange(len(wv)), wv, width=0.62, color=SEQ[450], zorder=3)
    ax.set_xticks(np.arange(len(wv)), labels)
    ax.set_xlabel("actual units that day")
    ax.set_ylabel("WAPE")
    ax.set_title("Error by demand level")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for i, v in enumerate(wv):
        if np.isfinite(v):
            ax.text(i, v * 1.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)

    fig.suptitle("Error diagnostics", fontsize=11.5, fontweight="semibold",
                 x=0.008, ha="left", y=0.995)
    return _finish(fig, axes, out,
                   f"{model} · a flat weekday panel means the weekly shape is captured; "
                   f"a sloped right panel means scale-dependent bias", has_suptitle=True)


def fig_quantiles(out: Path, sku_limit: int = 120, horizon: int = 7,
                  n_panels: int = 4, n_estimators: int = 300) -> Path:
    """Prediction intervals for the replenishment question.

    Target is ``y_cum7`` -- total demand over the next 7 days, which is what an order
    decision needs, not a single day's point forecast. The band is P10-P90 in one hue;
    the actual is the contrasting slot, direct-labelled. The calibration panel checks
    that the nominal levels are the realised ones before anyone sizes stock off them.
    """
    from .features import feature_columns, load_features
    from .models import get_model
    from .splits import apply_fold, holdout

    lf = load_features("daily")
    fts = feature_columns(lf)
    ycol = f"y_cum{horizon}"
    sub = (lf.filter(pl.col(ycol).is_not_null() & (pl.col("sample_weight") > 0))
             .select([C.SKU, C.DATE, ycol] + fts))
    skus = sub.select(pl.col(C.SKU).unique().sort().head(sku_limit)).collect().get_column(C.SKU)
    df = sub.filter(pl.col(C.SKU).is_in(skus.to_list())).collect()

    fold = holdout(df.get_column(C.DATE).unique(), test_size=42, horizon=horizon)
    tr = apply_fold(df.lazy(), fold, "train").collect()
    te = apply_fold(df.lazy(), fold, "test").collect()
    log.info("quantile fit: train %s, test %s", f"{len(tr):,}", f"{len(te):,}")

    qm = get_model("lgbm_quantile", n_estimators=n_estimators, num_leaves=63)
    qm.fit(tr.select(fts), tr.get_column(ycol).to_numpy())
    q = qm.predict_quantiles(te.select(fts))
    te = te.select(C.SKU, C.DATE, ycol).with_columns(
        pl.Series("lo", q.get_column("q0.1")),
        pl.Series("mid", q.get_column("q0.5")),
        pl.Series("hi", q.get_column("q0.9")),
    )

    top = (te.group_by(C.SKU).agg(pl.col(ycol).sum().alias("v"))
             .sort("v", descending=True).head(n_panels).get_column(C.SKU).to_list())

    fig = plt.figure(figsize=(12.4, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.55, wspace=0.16)

    for i, sku in enumerate(top):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        d = te.filter(pl.col(C.SKU) == sku).sort(C.DATE)
        x = d.get_column(C.DATE).to_list()
        ax.fill_between(x, d.get_column("lo"), d.get_column("hi"), color=SEQ[200],
                        alpha=0.85, lw=0, label="P10-P90", zorder=2)
        ax.plot(x, d.get_column("mid"), color=SERIES[0], lw=2.0, label="P50 forecast", zorder=3)
        ax.plot(x, d.get_column(ycol), color=SERIES[1], lw=2.0, label="actual", zorder=4)
        inside = ((d.get_column(ycol) >= d.get_column("lo")) &
                  (d.get_column(ycol) <= d.get_column("hi"))).mean()
        ax.set_title(f"{sku}   ·   {inside:.0%} of actuals inside the band", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        _date_axis(ax)
        # Nudge the two end labels apart when the series converge, or they overprint.
        ends = [("actual", SERIES[1], float(d.get_column(ycol)[-1])),
                ("P50", SERIES[0], float(d.get_column("mid")[-1]))]
        ends.sort(key=lambda e: e[2])
        span = max(ax.get_ylim()[1], 1.0)
        if abs(ends[1][2] - ends[0][2]) < 0.06 * span:
            offsets = [-7, 7]
        else:
            offsets = [0, 0]
        for (lbl, col, val), dy in zip(ends, offsets):
            ax.annotate(lbl, (x[-1], val), xytext=(6, dy), textcoords="offset points",
                        color=col, fontsize=8, va="center", fontweight="semibold")
        if i == 0:
            ax.legend(loc="upper left", ncols=3)

    ax = fig.add_subplot(gs[2, :])
    y = te.get_column(ycol).to_numpy()
    nominal = [0.1, 0.5, 0.9]
    realised = [float(np.mean(y <= te.get_column(c).to_numpy())) for c in ("lo", "mid", "hi")]
    xs = np.arange(len(nominal))
    ax.bar(xs - 0.17, nominal, width=0.3, color=SEQ[200], label="nominal", zorder=3)
    ax.bar(xs + 0.17, realised, width=0.3, color=SERIES[0], label="realised", zorder=3)
    for j, (nv, rv) in enumerate(zip(nominal, realised)):
        ax.text(j - 0.17, nv + 0.02, f"{nv:.2f}", ha="center", fontsize=8, color=INK_2)
        ax.text(j + 0.17, rv + 0.02, f"{rv:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, ["P10", "P50", "P90"])
    ax.set_ylabel("share of actuals at or below")
    ax.set_ylim(0, 1.12)
    ax.set_title("Are the intervals calibrated?  (bars should match in pairs)")
    ax.legend(loc="upper left", ncols=2)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    fig.suptitle(f"Prediction intervals for cumulative {horizon}-day demand",
                 fontsize=11.5, fontweight="semibold", x=0.008, ha="left", y=0.995)
    return _finish(fig, None, out,
                   "P90 minus P50 is the safety stock; trust it only if the bottom panel "
                   "pairs up", has_suptitle=True)


def fig_by_demand_class(res, out: Path, models: list[str]) -> Path:
    """WAPE per Syntetos-Boylan demand class.

    The single most useful breakdown in the repo: an average WAPE across the whole catalogue
    hides that smooth and lumpy series want different models. If the bars cross, route
    by class instead of picking one global winner.
    """
    from .panel import load_panel
    from .profiling import demand_classes

    cls = demand_classes(load_panel("daily")).select(C.SKU, "demand_class")
    p = res.predictions.join(cls, on=C.SKU, how="left").filter(
        pl.col("model").is_in(models) & pl.col("demand_class").is_not_null()
    )
    agg = (p.group_by(["model", "demand_class"])
             .agg((pl.col("y") - pl.col("yhat")).abs().sum().alias("ae"),
                  pl.col("y").sum().alias("act"))
             .with_columns((pl.col("ae") / pl.col("act")).alias("wape")))

    order = ["smooth", "erratic", "intermittent", "lumpy"]
    classes = [c for c in order if c in agg.get_column("demand_class").unique().to_list()]
    present = [m for m in models if m in agg.get_column("model").unique().to_list()]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    w = 0.78 / max(len(present), 1)
    xs = np.arange(len(classes))
    for i, m in enumerate(present):
        vals = []
        for c in classes:
            r = agg.filter((pl.col("model") == m) & (pl.col("demand_class") == c))
            vals.append(float(r.get_column("wape")[0]) if not r.is_empty() else np.nan)
        off = (i - (len(present) - 1) / 2) * w
        col = SERIES[i % len(SERIES)]
        ax.bar(xs + off, vals, width=w * 0.9, color=col, label=m, zorder=3)
        for x, v in zip(xs + off, vals):
            if np.isfinite(v):
                ax.text(x, v * 1.02, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK)
    ax.set_xticks(xs, classes)
    ax.set_ylabel("WAPE")
    ax.set_title("Where each model wins")
    ax.legend(loc="upper left", ncols=len(present))
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    return _finish(fig, ax, out,
                   "if the ordering flips between classes, route by class rather than "
                   "picking one global model")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def build_report(
    models: list[str] | None = None,
    horizon: int = 7,
    folds: int = 3,
    sku_limit: int | None = 200,
    max_train_rows: int | None = 220_000,
    horizons: tuple[int, ...] = (1, 7, 14, 28),
    out_dir: Path | None = None,
    skip_sweep: bool = False,
    skip_quantiles: bool = False,
) -> list[Path]:
    """Run the backtest and write every figure. Returns the paths written."""
    from .experiment import RunConfig, run

    _style()
    out_dir = out_dir or C.FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    models = models or ["ma7", "ma28", "croston", "lgbm_tweedie", "two_stage"]

    log.info("=== main backtest: %d models, h=%d, %d folds ===", len(models), horizon, folds)
    res = run(RunConfig(
        models=models, horizon=horizon,
        splits=C.SplitSpec(test_size=28, n_folds=folds),
        sku_limit=sku_limit, max_train_rows=max_train_rows,
    ))
    best = res.leaderboard().get_column("model")[0]
    log.info("best model: %s", best)

    paths = [
        fig_leaderboard(res, out_dir / "01_leaderboard.png"),
        fig_forecast_vs_actual(res, out_dir / "02_forecast_vs_actual.png", best),
        fig_error_diagnostics(res, out_dir / "03_error_diagnostics.png", best),
        fig_by_demand_class(res, out_dir / "04_by_demand_class.png",
                            [m for m in models if m != "global_mean"][:3]),
    ]

    if not skip_sweep:
        log.info("=== horizon sweep: %s ===", list(horizons))
        rows = []
        sweep_models = [m for m in ("ma28", "croston", "lgbm_tweedie") if m in models] or models[:2]
        for h in horizons:
            r = run(RunConfig(
                models=sweep_models, horizon=h,
                splits=C.SplitSpec(test_size=28, n_folds=1),
                sku_limit=sku_limit, max_train_rows=max_train_rows,
            ), save=False)
            rows.append(r.leaderboard().with_columns(pl.lit(h).alias("h")))
        paths.append(fig_horizon_decay(pl.concat(rows), out_dir / "05_horizon_decay.png"))

    if not skip_quantiles:
        log.info("=== quantile intervals ===")
        paths.append(fig_quantiles(out_dir / "06_prediction_intervals.png",
                                   sku_limit=min(sku_limit or 120, 120), horizon=horizon))

    log.info("wrote %d figures to %s", len(paths), out_dir)
    return paths
