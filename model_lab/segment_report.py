"""WHERE THE FORECAST FAILS — one headline WAPE, broken into the parts it hides.

    uv run python model_lab/segment_report.py                  # best model on the board
    uv run python model_lab/segment_report.py --model xgboost_gbdt
    uv run python model_lab/segment_report.py --sku-limit 300  # quick answer

`compare_models.py` answers "which model is best". This file answers the two questions
that come straight after it, and that a single number cannot:

    1. best for WHAT?   -- the same model can be several times more accurate on fast
                           movers than on the long tail, and the headline is a
                           volume-weighted average of the two.
    2. at WHAT LEVEL?   -- a planner ordering a container wants the product line, not
                           the SKU. Does the forecast get better if you ask it that way?

It writes:

    model_lab/results/_segments.json     every number below, machine-readable
    model_lab/SEGMENTS.md                the written breakdown
    model_lab/figures/_segments.png      WAPE and bias per segment, per horizon

Nothing here is a new model. It re-fits the model you name on the same three
rolling-origin folds `common.make_folds` gives every other file, keeps the per-row test
predictions, and then slices them. So a number here is directly comparable to the same
model's row on the board.

-----------------------------------------------------------------------------------
WHY BIAS IS PRINTED NEXT TO EVERY WAPE
-----------------------------------------------------------------------------------
WAPE is symmetric: it charges you the same for 100 units too many and 100 too few. The
inventory consequences are not symmetric at all, and neither is the model's error --
it is usually biased one way on one product line and the other way on another. Those
biases cancel in the headline. A catalogue-wide bias near zero can be one line well
under and another well over, which is two separate problems wearing a disguise, and the
disguise comes off the moment you group.

-----------------------------------------------------------------------------------
THE AGGREGATION SECTION, AND THE TRAP IN IT
-----------------------------------------------------------------------------------
Forecasting a category total is an easier question than forecasting a SKU, because SKU
errors partially cancel when you add them up. WAPE therefore falls a long way -- and it
is very tempting to read that as the model having improved. It has not. The question
changed.

Two ways to answer the category question are scored here, on identical actuals:

    bottom_up    sum the SKU-level forecasts inside each category.
    direct       fit the same model on the category's own aggregated series.

and two naive category baselines (`cat_moving_avg`, `cat_last_year`) so the two real
answers have something to beat.

The direct model's features are the SKU-level features aggregated at the same origin --
extensive ones (levels, volumes) summed, intensive ones (rates, prices, ranks) averaged.
That is deliberate: it inherits the project's past-only timing convention exactly rather
than inventing a second feature pipeline that could be wrong in a new way.

Watch the bias column in that table. It is the one that does NOT improve with
aggregation, which is the whole point: cancelling errors cancel variance, not bias.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ACTUAL_C,
    FIGURES,
    GRID_C,
    HORIZON_LABEL,
    HORIZONS,
    LAB,
    PRED_C,
    RESULTS,
    add_top_meaningful_tags,
    load_dataset,
    make_folds,
    split,
)
from demandml import config as C  # noqa: E402

REPORT = LAB / "SEGMENTS.md"
OUT_JSON = RESULTS / "_segments.json"
#: Per-row test predictions from the last run. The fits are the only slow part of this
#: file (a boosted model over 3 horizons x 3 folds takes a while), and every table and
#: figure below is a pure function of these rows -- so they are cached, and
#: `--reuse-preds` redraws everything in seconds. The cache records which model and
#: which fold geometry produced it and refuses to be reused under a different one.
PREDS_CACHE = RESULTS / "_segments_preds.parquet"

#: Levels to aggregate to, coarsest last. `sku_family` is the SKU code's BRAND-FAMILY
#: prefix (variants of one product), `sku_brand` its BRAND prefix (a product line),
#: `channel_category` the marketplace category off the product master.
AGG_LEVELS = ("sku_family", "sku_brand", "channel_category")

#: Extensive columns -- they scale with how many SKUs are in the group, so they SUM.
AGG_SUM = [
    "units_lag_0", "units_lag_364", "units_mean_7", "units_mean_28", "units_mean_91",
    "units_mean_182", "units_mean_364", "units_median_28", "units_max_28",
    "units_std_28", "units_ewm_01", "units_ewm_03", "units_ewm_06", "ly_mean_91",
    "units_same_dow_mean_52w", "sessions_mean_28", "sessions_mean_91", "ad_cost_mean_28",
    "impressions_mean_28", "clicks_mean_28", "order_count_mean_28", "ad_units_mean_28",
    "fam_units_mean_28",
]
#: Intensive columns -- rates, prices, ranks. Adding them up would be meaningless, so
#: they average.
AGG_MEAN = [
    "units_zero_rate_28", "units_zero_rate_91", "units_cv_28", "discount_rate_mean_28",
    "discount_rate_lag_0", "is_oos_mean_28", "kf_price_vs_91d", "kf_is_deal_now",
    "kf_unit_price_now", "unit_price_mean_28", "return_rate_mean_28",
    "master_rank_mean_28", "adi_364", "cv2_364", "days_on_market",
]
#: Calendar columns are identical for every SKU on a date, so one of them is the group's.
AGG_FIRST = [
    "t_index", "doy", "iso_week", "month", "dow", "is_holiday", "days_to_black_friday",
    "days_to_cyber_monday", "days_to_prime_day", "days_to_thanksgiving",
    "days_to_christmas",
] + [f"fourier_y_{t}_{k}" for t in ("sin", "cos") for k in (1, 2, 3, 4)]


# ======================================================================================
# Metrics
# ======================================================================================

def _wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9))


def _bias(y: np.ndarray, p: np.ndarray) -> float:
    return float((p - y).sum() / max(y.sum(), 1e-9))


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


# ======================================================================================
# Segment definitions
# ======================================================================================

def add_segments(df: pl.DataFrame) -> pl.DataFrame:
    """Attach the cuts we slice by. All are computed from PAST-only columns.

    `volume_band` and `syntetos` come off `units_mean_91` / `adi_364` / `cv2_364`, which
    are trailing windows ending on the origin -- so a SKU can move between bands over
    time, which is correct. Classifying a SKU once, on its whole history, would be a
    quiet way of telling each test row how busy its own future was.
    """
    df = df.with_columns(
        pl.when(pl.col("units_mean_91") >= 50).then(pl.lit("A  >=50/day"))
        .when(pl.col("units_mean_91") >= 10).then(pl.lit("B  10-50/day"))
        .when(pl.col("units_mean_91") >= 2).then(pl.lit("C  2-10/day"))
        .when(pl.col("units_mean_91") >= 0.5).then(pl.lit("D  0.5-2/day"))
        .otherwise(pl.lit("E  <0.5/day")).alias("volume_band"),
        # Syntetos-Boylan quadrants. ADI = mean gap between non-zero demands, CV2 =
        # squared coefficient of variation of the non-zero sizes. The 1.32 / 0.49 cuts
        # are the standard ones.
        pl.when(pl.col("adi_364") < 1.32)
        .then(pl.when(pl.col("cv2_364") < 0.49).then(pl.lit("smooth"))
              .otherwise(pl.lit("erratic")))
        .otherwise(pl.when(pl.col("cv2_364") < 0.49).then(pl.lit("intermittent"))
                   .otherwise(pl.lit("lumpy"))).alias("syntetos"),
        pl.when(pl.col("days_on_market") < 365).then(pl.lit("1  <1y"))
        .when(pl.col("days_on_market") < 730).then(pl.lit("2  1-2y"))
        .when(pl.col("days_on_market") < 1460).then(pl.lit("3  2-4y"))
        .otherwise(pl.lit("4  4y+")).alias("age_band"),
    )
    # top_n_band / meaningful are shared with compare_subsets.py, so they live in
    # common.py -- see add_top_meaningful_tags for how each is defined.
    return add_top_meaningful_tags(df)


SEGMENTS = ("sku_brand", "channel_category", "volume_band", "syntetos", "age_band",
            "top_n_band", "meaningful")
SEGMENT_LABEL = {
    "sku_brand": "product line (SKU prefix)",
    "channel_category": "marketplace category",
    "volume_band": "volume band",
    "syntetos": "demand pattern (Syntetos-Boylan)",
    "age_band": "time on market",
    "top_n_band": "top-N items by trailing volume",
    "meaningful": "meaningful vs. thin/new items",
}


# ======================================================================================
# Backtest -> per-row predictions
# ======================================================================================

def load_model(name: str):
    """Import a model file from this folder and hand back its `MODEL`."""
    return importlib.import_module(name).MODEL


def best_on_board() -> str:
    """The lowest mean-WAPE model that has a results JSON. Falls back to LightGBM."""
    best, best_w = "lightgbm_gbdt", float("inf")
    for p in RESULTS.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            r = json.loads(p.read_text())
            hs = [v["wape"] for v in r.get("horizons", {}).values()]
            if hs and float(np.mean(hs)) < best_w:
                best, best_w = r["model"], float(np.mean(hs))
        except (json.JSONDecodeError, KeyError):
            continue
    return best


def backtest(df: pl.DataFrame, feats: list[str], model, horizons: list[int],
             folds_n: int, test_days: int, cap: int | None) -> pl.DataFrame:
    """Fit `model` on every (horizon, fold) and keep the test-row predictions.

    Identical folds, identical cap, identical leakage assertion as `common.run_model` --
    this only differs in keeping the predictions instead of collapsing them to a score.
    """
    rows: list[pl.DataFrame] = []
    for h in horizons:
        ycol = f"y_cum{h}"
        if ycol not in df.columns:
            print(f"  ! {ycol} not in the dataset, skipping")
            continue
        for fold in make_folds(df, h, n_folds=folds_n, test_days=test_days):
            tr, te = split(df, fold, h, max_train_rows=cap, seed=fold.index)
            if not len(tr) or not len(te):
                continue
            y_tr = tr.get_column(ycol).to_numpy().astype(np.float64)
            y_te = te.get_column(ycol).to_numpy().astype(np.float64)
            w_tr = tr.get_column("sample_weight").to_numpy().astype(np.float64)
            X_tr = tr.select(feats) if feats else tr
            X_te = te.select(feats) if feats else te

            t0 = time.time()
            est = model.build(h)
            if model.needs_frame:
                est.fit(X_tr, y_tr, w_tr, frame=tr)
            else:
                est.fit(X_tr, y_tr, w_tr)
            p = np.clip(np.nan_to_num(
                np.asarray(est.predict(X_te), dtype=np.float64).ravel()), 0.0, None)
            print(f"  h={h:3d} split {fold.index}: wape={_wape(y_te, p):.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            rows.append(te.select(C.SKU, C.DATE, *SEGMENTS).with_columns(
                pl.lit(h).alias("h"), pl.lit(fold.index).alias("fold"),
                pl.Series("actual", y_te), pl.Series("predicted", p)))
    if not rows:
        raise SystemExit("no folds produced predictions")
    return pl.concat(rows)


def cached_preds(model_name: str, horizons: list[int], folds_n: int, test_days: int,
                 sku_limit: int | None) -> pl.DataFrame | None:
    """The last run's predictions, if they were produced the same way as this one.

    Refuses a cache built by a different model or a different fold geometry, because
    silently slicing yesterday's model under today's heading is exactly the kind of
    error this folder is careful about everywhere else.
    """
    if not PREDS_CACHE.exists():
        return None
    p = pl.read_parquet(PREDS_CACHE)
    want = {"model": model_name, "folds": folds_n, "test_days": test_days,
            "sku_limit": -1 if sku_limit is None else sku_limit}
    got = {k: p.get_column(f"_{k}")[0] for k in want if f"_{k}" in p.columns}
    if got != want:
        print(f"  ! cached predictions were built as {got}, this run wants {want} "
              f"-- refitting")
        return None
    have = set(p.get_column("h").unique().to_list())
    if not set(horizons) <= have:
        print(f"  ! cache only covers horizons {sorted(have)} -- refitting")
        return None
    print(f"  reusing {len(p):,} cached predictions from {PREDS_CACHE.name} "
          f"(omit --reuse-preds to refit)")
    # Drop the provenance columns so the reused frame is identical in shape to a fresh
    # backtest's -- everything downstream should not be able to tell the difference.
    return (p.filter(pl.col("h").is_in(horizons))
            .drop([c for c in p.columns if c.startswith("_")]))


# ======================================================================================
# Slice
# ======================================================================================

def slice_by(preds: pl.DataFrame, by: str, h: int, min_vol_pct: float = 0.3
             ) -> pl.DataFrame:
    """WAPE / bias / share-of-volume for one segmentation at one horizon.

    Pooled over folds rather than averaged across them: a thin segment can be empty in
    one fold, and averaging three numbers one of which is undefined is worse than
    scoring the rows together.
    """
    g = (preds.filter(pl.col("h") == h)
         .group_by(by)
         .agg((pl.col("actual") - pl.col("predicted")).abs().sum().alias("_ae"),
              (pl.col("predicted") - pl.col("actual")).sum().alias("_se"),
              pl.col("actual").sum().alias("_vol"),
              pl.col(C.SKU).n_unique().alias("skus"),
              pl.len().alias("origins")))
    return (g.with_columns(
        (pl.col("_ae") / pl.col("_vol").clip(1e-9)).alias("wape"),
        (pl.col("_se") / pl.col("_vol").clip(1e-9)).alias("bias"),
        (pl.col("_vol") / pl.col("_vol").sum() * 100).alias("vol_pct"))
        .filter(pl.col("vol_pct") >= min_vol_pct)
        .drop("_ae", "_se", "_vol")
        .sort("vol_pct", descending=True))


# ======================================================================================
# Aggregation: is the category question easier, and how should it be answered?
# ======================================================================================

def build_agg_frame(df: pl.DataFrame, key: str, h: int) -> tuple[pl.DataFrame, list[str]]:
    """One row per (category, origin), features aggregated from the SKU rows.

    The target is the category's own next-h-day total: the sum of the SKU targets over
    exactly the SKUs that have a usable target at that origin. `bottom_up` is scored
    against the same number, which is what makes the two comparable.
    """
    yc = f"y_cum{h}"
    d = df.filter(pl.col(yc).is_not_null() & pl.col(key).is_not_null())
    cols = set(d.columns)
    aggs = (
        [pl.col(c).sum().alias(c) for c in AGG_SUM if c in cols]
        + [pl.col(f"ly_cum{h}").sum().alias("ly_cum_h")]
        + [pl.col(c).mean().alias(c) for c in AGG_MEAN if c in cols]
        + [pl.col(c).first().alias(c) for c in AGG_FIRST if c in cols]
        + [pl.len().alias("n_skus"), pl.col(yc).sum().alias("Y")]
    )
    g = d.group_by([key, C.DATE]).agg(aggs).sort([key, C.DATE])
    # Category-level momentum, the same ratios `features.trend_and_ratio_exprs` builds
    # per SKU. Ratios travel across groups of any size; raw levels do not.
    g = g.with_columns(
        (pl.col("units_mean_7") / pl.col("units_mean_28").clip(1e-6)).alias("c_ratio_7_28"),
        (pl.col("units_mean_28") / pl.col("units_mean_91").clip(1e-6)).alias("c_ratio_28_91"),
        (pl.col("units_mean_28") / pl.col("units_mean_364").clip(1e-6)).alias("c_ratio_28_364"),
    )
    feats = [c for c in g.columns if c not in (C.DATE, "Y")]
    return g, feats


def aggregation_study(df: pl.DataFrame, preds: pl.DataFrame, model,
                      horizons: list[int], folds_n: int, test_days: int,
                      cap: int | None) -> list[dict[str, Any]]:
    """Score four answers to the category question, on identical actuals."""
    out: list[dict[str, Any]] = []
    if not model.uses_features:
        # The per-series baselines read a stored column off the SKU frame (`ly_cum{h}`,
        # `units_mean_28`). An aggregated frame has no such column to read, and the two
        # naive rows below already cover that idea at category level.
        print("  ! this model reads a stored column rather than a feature matrix, "
              "so only bottom_up is meaningful here -- skipping")
        return out
    for key in AGG_LEVELS:
        if key not in df.columns:
            continue
        lut = df.select([C.SKU, key]).unique()
        for h in horizons:
            if f"y_cum{h}" not in df.columns:
                continue
            g, feats = build_agg_frame(df, key, h)
            folds = make_folds(df, h, n_folds=folds_n, test_days=test_days)

            bu = (preds.filter(pl.col("h") == h)
                  .drop(key, strict=False)  # `key` is only in preds for some levels
                  .join(lut, on=C.SKU, how="left")
                  .filter(pl.col(key).is_not_null())
                  .group_by([key, C.DATE, "fold"])
                  .agg(pl.col("actual").sum().alias("Y"),
                       pl.col("predicted").sum().alias("P")))

            acc: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
                "direct": [], "bottom_up": [], "cat_moving_avg": [], "cat_last_year": []}
            for fold in folds:
                tr = g.filter((pl.col(C.DATE) >= fold.train_start)
                              & (pl.col(C.DATE) <= fold.train_end))
                te = g.filter((pl.col(C.DATE) >= fold.test_start)
                              & (pl.col(C.DATE) <= fold.test_end))
                if not len(tr) or not len(te):
                    continue
                ytr = tr.get_column("Y").to_numpy().astype(np.float64)
                yte = te.get_column("Y").to_numpy().astype(np.float64)
                # Always the plain three-argument fit, even for a `needs_frame` model:
                # the `y_cum_m{k}` month slices an additive model wants do not exist on
                # an aggregated frame, so at this level it fits the total directly.
                est = model.build(h)
                est.fit(tr.select(feats), ytr, None)
                p = np.clip(np.nan_to_num(
                    np.asarray(est.predict(te.select(feats)), dtype=np.float64).ravel()),
                    0.0, None)
                acc["direct"].append((yte, p))
                acc["cat_moving_avg"].append(
                    (yte, np.nan_to_num(te.get_column("units_mean_28")
                                        .to_numpy().astype(np.float64)) * h))
                acc["cat_last_year"].append(
                    (yte, np.nan_to_num(te.get_column("ly_cum_h")
                                        .to_numpy().astype(np.float64))))
                b = bu.filter(pl.col("fold") == fold.index)
                acc["bottom_up"].append((b.get_column("Y").to_numpy().astype(np.float64),
                                         b.get_column("P").to_numpy().astype(np.float64)))

            for name, rs in acc.items():
                if not rs:
                    continue
                w, sd = _mean_sd([_wape(y, p) for y, p in rs])
                bi, _ = _mean_sd([_bias(y, p) for y, p in rs])
                out.append({"level": key, "h": h, "method": name, "wape": w,
                            "wape_sd": sd, "bias": bi,
                            "n_groups": int(g.get_column(key).n_unique())})
                print(f"  {key:18s} h={h:3d} {name:16s} wape={w:.4f} +/-{sd:.4f} "
                      f"bias={bi:+.4f}", flush=True)
    return out


# ======================================================================================
# Figure
# ======================================================================================

def segment_figure(preds: pl.DataFrame, horizons: list[int], model_name: str,
                   out: Path) -> Path:
    """One row per segmentation: WAPE bars, with SIGNED bias on a twin axis.

    The sign is the point. Plotting |bias| would hide exactly the thing this file
    exists to show -- that a segment under-forecasting and one over-forecasting are two
    different problems, and that they are what the catalogue-wide
    "near zero" is made of. The twin axis is symmetric and centred on zero so the
    direction is readable at a glance.
    """
    segs = [s for s in ("volume_band", "top_n_band", "meaningful", "syntetos", "sku_brand")
            if s in preds.columns]
    fig, axes = plt.subplots(len(segs), len(horizons),
                             figsize=(5.2 * len(horizons), 3.6 * len(segs)),
                             squeeze=False)
    for r, seg in enumerate(segs):
        for c, h in enumerate(horizons):
            ax = axes[r][c]
            t = slice_by(preds, seg, h).sort(seg)
            if seg == "sku_brand":  # too many to read -- keep the biggest
                t = t.sort("vol_pct", descending=True).head(10).sort(seg)
            lbl = t.get_column(seg).to_list()
            x = np.arange(len(lbl))
            ax.bar(x, t.get_column("wape").to_numpy(), color=ACTUAL_C, alpha=0.85,
                   label="WAPE")
            ax.set_xticks(x)
            ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=7)
            ax.grid(axis="y", color=GRID_C, linewidth=0.6)
            ax.set_axisbelow(True)

            bx = ax.twinx()
            bias = t.get_column("bias").to_numpy()
            bx.axhline(0.0, color=GRID_C, linewidth=1.0)
            bx.plot(x, bias, "o", color=PRED_C, markersize=5,
                    label="bias (signed, right axis)")
            lim = max(0.05, float(np.abs(bias).max()) * 1.25)
            bx.set_ylim(-lim, lim)
            bx.tick_params(axis="y", labelsize=7, colors=PRED_C)
            if c == len(horizons) - 1:
                bx.set_ylabel("bias  (+ over / - under)", fontsize=7, color=PRED_C)

            if c == 0:
                ax.set_ylabel(f"{SEGMENT_LABEL.get(seg, seg)}\nWAPE", fontsize=8)
            if r == 0:
                ax.set_title(HORIZON_LABEL.get(h, f"{h}d"), fontsize=10)
            if r == 0 and c == 0:
                hs, ls = ax.get_legend_handles_labels()
                h2, l2 = bx.get_legend_handles_labels()
                ax.legend(hs + h2, ls + l2, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"{model_name}: where the error actually is "
                 f"(bars = WAPE, dots = signed bias)", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ======================================================================================
# Report
# ======================================================================================

def _md_table(t: pl.DataFrame, by: str) -> str:
    head = f"| {by} | SKUs | % volume | WAPE | bias |\n|---|---|---|---|---|\n"
    body = "".join(
        f"| `{r[by]}` | {r['skus']} | {r['vol_pct']:.1f} | **{r['wape']:.3f}** | "
        f"{r['bias']:+.3f} |\n"
        for r in t.iter_rows(named=True))
    return head + body


def write_report(preds: pl.DataFrame, agg: list[dict[str, Any]], model_name: str,
                 horizons: list[int], fig: Path, meta: dict[str, Any]) -> Path:
    L: list[str] = []
    A = L.append
    A(f"# Where `{model_name}` fails\n")
    A(f"*Generated {meta['finished'][:16].replace('T', ' ')} by "
      f"`model_lab/segment_report.py`.*\n")
    A(f"Same dataset, same {meta['folds']} rolling-origin splits and same metrics as "
      f"`REPORT.md` — the headline numbers there are the volume-weighted average of the "
      f"tables below.\n")
    A(f"![segments]({fig.relative_to(LAB)})\n")

    A("## How to read this\n")
    A("**WAPE** is total absolute error over total units, pooled across folds within "
      "each segment. **bias** is signed: negative means under-forecasting, which looks "
      "respectable in WAPE and is paid for in stockouts; positive means dead stock.\n")
    A("The bias column is the one to read first. It cancels across segments, so a "
      "catalogue-wide bias near zero says nothing about whether any individual line is "
      "being forecast straight.\n")

    for seg in SEGMENTS:
        if seg not in preds.columns:
            continue
        A(f"## By {SEGMENT_LABEL.get(seg, seg)}\n")
        for h in horizons:
            A(f"**{HORIZON_LABEL.get(h, f'{h} days')}**\n")
            A(_md_table(slice_by(preds, seg, h), seg))
            A("")

    if agg:
        A("## Does forecasting the category instead of the SKU help?\n")
        A("`bottom_up` sums the SKU forecasts inside each category. `direct` fits the "
          "same model on the category's own aggregated series. Both are scored against "
          "identical actuals; the two naive rows are there for scale.\n")
        for h in horizons:
            rows = [r for r in agg if r["h"] == h]
            if not rows:
                continue
            A(f"**{HORIZON_LABEL.get(h, f'{h} days')}**\n")
            # No trailing newline: these lines are joined with "\n" below, and a blank
            # line between a table's header and its body stops it rendering as a table.
            A("| level | groups | method | WAPE | ± spread | bias |\n"
              "|---|---|---|---|---|---|")
            for r in sorted(rows, key=lambda r: (AGG_LEVELS.index(r["level"]),
                                                 r["wape"])):
                A(f"| `{r['level']}` | {r['n_groups']} | `{r['method']}` | "
                  f"**{r['wape']:.3f}** | ±{r['wape_sd']:.3f} | {r['bias']:+.3f} |")
            A("")
        A("Two things to take from that table, and only the first is good news.\n")
        A("WAPE falls steeply as you aggregate. That is **not** the model improving — "
          "it is SKU-level errors cancelling against each other inside the group, so "
          "the question has got easier. Compare a category row only against another "
          "category row at the same level.\n")
        A("The bias barely moves. Cancelling errors cancel variance, not systematic "
          "error, so the number a planner would actually buy on is no better than it "
          "was per SKU. A category WAPE quoted without its bias overstates how much "
          "the aggregation bought you.\n")

    REPORT.write_text("\n".join(L))
    return REPORT


# ======================================================================================
# Entry point
# ======================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Break one model's WAPE down by product segment and by "
                    "aggregation level.")
    ap.add_argument("--model", default=None,
                    help="model file stem (default: the best one on the board)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--horizons", type=str, default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--sku-limit", type=int, default=None)
    ap.add_argument("--max-train-rows", type=int, default=300_000,
                    help="cap on training rows per fold, 0 = no cap")
    ap.add_argument("--no-aggregation", action="store_true",
                    help="skip the category-level section (it is the slow half)")
    ap.add_argument("--reuse-preds", action="store_true",
                    help="redraw tables and figures from the last run's cached "
                         "predictions instead of refitting (seconds, not minutes)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    name = args.model or best_on_board()
    model = load_model(name)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    cap = args.max_train_rows or None

    print(f"\n{'=' * 86}\nWHERE IT FAILS — {name}\n{'=' * 86}")
    df, feats = load_dataset(args.sku_limit)
    if not model.uses_features:
        feats = []
    df = add_segments(df)

    print("\n[backtest]")
    preds = (cached_preds(name, horizons, args.folds, args.test_days, args.sku_limit)
             if args.reuse_preds else None)
    if preds is None:
        preds = backtest(df, feats, model, horizons, args.folds, args.test_days, cap)
        if not args.no_save:
            RESULTS.mkdir(parents=True, exist_ok=True)
            preds.with_columns(
                pl.lit(name).alias("_model"), pl.lit(args.folds).alias("_folds"),
                pl.lit(args.test_days).alias("_test_days"),
                pl.lit(-1 if args.sku_limit is None else args.sku_limit)
                .alias("_sku_limit"),
            ).write_parquet(PREDS_CACHE)

    # A cache written before a segment existed won't have its column. Segments are
    # static SKU/date attributes, not model output, so they can be joined back on
    # from `df` instead of forcing a refit just to pick up a new cut.
    missing = [s for s in SEGMENTS if s not in preds.columns]
    if missing:
        preds = preds.join(
            df.select(C.SKU, C.DATE, *missing).unique(subset=[C.SKU, C.DATE]),
            on=[C.SKU, C.DATE], how="left")

    print("\n[segments]")
    for seg in SEGMENTS:
        print(f"\n-- by {SEGMENT_LABEL.get(seg, seg)} --")
        for h in horizons:
            t = slice_by(preds, seg, h)
            print(f"  {HORIZON_LABEL.get(h, str(h))}:")
            for r in t.iter_rows(named=True):
                print(f"     {str(r[seg])[:22]:24s} vol={r['vol_pct']:5.1f}%  "
                      f"wape={r['wape']:.3f}  bias={r['bias']:+.3f}")

    agg: list[dict[str, Any]] = []
    if not args.no_aggregation:
        print("\n[aggregation levels]")
        agg = aggregation_study(df, preds, model, horizons, args.folds,
                                args.test_days, cap)

    if args.no_save:
        return

    meta = {"model": name, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "folds": args.folds, "test_days": args.test_days,
            "sku_limit": args.sku_limit, "n_skus": int(df.get_column(C.SKU).n_unique())}
    payload = {
        "meta": meta,
        "segments": {seg: {str(h): slice_by(preds, seg, h).to_dicts()
                           for h in horizons}
                     for seg in SEGMENTS if seg in preds.columns},
        "aggregation": agg,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    fig = segment_figure(preds, horizons, name, FIGURES / "_segments.png")
    rep = write_report(preds, agg, name, horizons, fig, meta)
    print(f"\nresults -> {OUT_JSON.relative_to(LAB.parent)}")
    print(f"figure  -> {fig.relative_to(LAB.parent)}")
    print(f"report  -> {rep.relative_to(LAB.parent)}")


if __name__ == "__main__":
    main()
