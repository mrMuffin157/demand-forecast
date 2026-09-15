"""THE SCOREBOARD, RESTRICTED TO ONE SLICE OF THE CATALOGUE.

    uv run python model_lab/compare_subsets.py

`compare_models.py` ranks every model on the whole catalogue, pooled. That
pooled number is dominated by whichever SKUs carry the volume, and it says nothing about
two questions a planner actually asks:

    1. How good is this, specifically, on my handful of biggest sellers?
    2. How good is this on the catalogue that's actually worth forecasting, once the
       thin/near-new long tail (which is most of the SKU *count* but almost none of the
       volume) is set aside?

This file answers both, using the exact same fitted models and folds as the main board
-- it does not refit anything. Each model file's `run_model()` (in common.py) already
caches its raw per-row test predictions to `results/_preds/<model>.parquet`; this just
re-pools those rows under a different SKU filter. A model that hasn't been (re-)run since
that cache was added won't have a file there yet and is skipped with a warning --
run it once (e.g. `uv run python model_lab/xgboost_gbdt.py`) to pick it up.

One folder per horizon, one image per scope, every model side by side in each image:

    model_lab/figures/1_month/all.png            model_lab/figures/1_month/top20.png
    model_lab/figures/1_month/meaningful.png
    model_lab/figures/3_month/...
    model_lab/figures/6_month/...

Every image shares ONE y-axis range across its model panels (computed from that same
horizon+scope group, e.g. every "1 month ahead, all SKUs" panel uses the same min/max) --
without that, each panel auto-scales to its own model's error and two models that are
actually very different can end up looking the same. Blue = actual, orange dashed =
predicted, same convention as `common._fig_for_horizon`.

Also writes model_lab/SUBSETS.md (the top20/meaningful WAPE tables + the cross-model bar
charts `_comparison_top20.png` / `_comparison_meaningful.png`; the "all" scope's own
board is `REPORT.md`, not duplicated here).
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ACTUAL_C,
    FIGURES,
    GRID_C,
    HORIZON_LABEL,
    HORIZONS,
    LAB,
    MEANINGFUL_LABEL,
    PRED_C,
    RESULTS,
    _fig_for_horizon,
    add_top_meaningful_tags,
    load_dataset,
)
from compare_models import MODEL_FILES, _table, comparison_figure, mean_wape  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demandml import config as C  # noqa: E402
from demandml.metrics import evaluate  # noqa: E402

REPORT = LAB / "SUBSETS.md"
PREDS_DIR = RESULTS / "_preds"

#: "Top several" = top 20 SKUs by trailing volume, re-ranked at every origin date (a
#: model_lab convention, see `common.add_top_meaningful_tags`). Change this one number
#: to ask about a bigger or smaller handful.
TOP_N = 20

#: `mask=None` means no filter -- the whole catalogue.
SCOPES: dict[str, dict[str, Any]] = {
    "top20": {"label": f"Top {TOP_N} SKUs by trailing volume",
             "mask": pl.col("vol_rank") <= TOP_N},
    "meaningful": {"label": "Meaningful catalogue (>=0.5 units/day trailing, 90d+ on "
                            "market) -- the thin/near-new long tail set aside",
                  "mask": pl.col("meaningful") == MEANINGFUL_LABEL},
    "all": {"label": "All SKUs (whole catalogue, pooled)", "mask": None},
}
#: Only these two get their own WAPE leaderboard in SUBSETS.md -- "all" already has
#: one, REPORT.md, and re-deriving it here would just be a second number that could
#: quietly drift from the first.
SCORED_SCOPES = ("top20", "meaningful")

HOR_DIR = {30: "1_month", 90: "3_month", 180: "6_month"}
ROW_LABEL = {"top20": f"Top {TOP_N}", "meaningful": "Meaningful", "all": "All SKUs"}


def load_model_preds() -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    for f in MODEL_FILES:
        name = Path(f).stem
        p = PREDS_DIR / f"{name}.parquet"
        if not p.exists():
            print(f"  ! no cached predictions for `{name}` yet ({p.name} missing) "
                  f"-- run it once to pick it up, skipping")
            continue
        out[name] = pl.read_parquet(p)
    if not out:
        raise SystemExit(
            "no model has cached predictions yet. Run any model file at least once "
            "(they now write results/_preds/<name>.parquet), then re-run this.")
    return out


def subset_rows(preds: pl.DataFrame, tags: pl.DataFrame, mask: pl.Expr | None,
                h: int) -> pl.DataFrame:
    sub = preds.filter(pl.col("h") == h)
    if mask is not None:
        sub = sub.join(tags, on=[C.SKU, C.DATE], how="left").filter(mask)
    return sub


def score_subset(all_preds: dict[str, pl.DataFrame], tags: pl.DataFrame,
                 mask: pl.Expr) -> list[dict[str, Any]]:
    """Same shape as one model's entry in results/*.json: mean +/- sd over folds."""
    ranked = []
    for name, preds in all_preds.items():
        horizons: dict[str, Any] = {}
        for h in HORIZONS:
            sub = subset_rows(preds, tags, mask, h)
            per_fold = []
            for fold_id in sorted(sub.get_column("fold").unique().to_list()):
                row = sub.filter(pl.col("fold") == fold_id)
                if row.is_empty():
                    continue
                y = row.get_column("actual").to_numpy()
                p = row.get_column("predicted").to_numpy()
                per_fold.append(evaluate(y, p))
            if not per_fold:
                continue
            agg = {k: float(np.mean([f[k] for f in per_fold]))
                   for k in ("wape", "bias", "shortfall")}
            agg["wape_sd"] = (float(np.std([f["wape"] for f in per_fold], ddof=1))
                              if len(per_fold) > 1 else 0.0)
            horizons[str(h)] = agg
        if horizons:
            ranked.append({"model": name, "horizons": horizons})
    ranked.sort(key=mean_wape)
    return ranked


def build_horizon_scope_grid(h: int, scope_key: str, mask: pl.Expr | None,
                             all_preds: dict[str, pl.DataFrame],
                             tags: pl.DataFrame) -> Path | None:
    """Every model's actual-vs-predicted-over-time line, one subplot each, sharing a
    single y-axis range across the whole image -- computed from every model in this
    same (horizon, scope) group, so a flat blue line in one panel and a wildly-off
    orange line in another are directly comparable rather than each auto-scaled to
    look equally "big" on its own axes.
    """
    per_model: dict[str, tuple[pl.DataFrame, float]] = {}
    y_min, y_max = np.inf, -np.inf
    for name, preds in all_preds.items():
        sub = subset_rows(preds, tags, mask, h)
        if sub.is_empty():
            continue
        wape = evaluate(sub.get_column("actual").to_numpy(),
                        sub.get_column("predicted").to_numpy())["wape"]
        by_date = (sub.group_by(C.DATE)
                      .agg(pl.col("actual").mean().alias("actual"),
                           pl.col("predicted").mean().alias("predicted"))
                      .sort(C.DATE))
        per_model[name] = (by_date, wape)
        y_max = max(y_max, float(by_date.get_column("actual").max()),
                    float(by_date.get_column("predicted").max()))
        y_min = min(y_min, float(by_date.get_column("actual").min()),
                    float(by_date.get_column("predicted").min()))
    if not per_model:
        return None
    # Pad both ends by the same fraction of the range -- NOT anchored at 0, since a
    # floor of 0 on a series that never gets near it just wastes vertical space and
    # flattens the very differences this shared axis exists to show.
    pad = (y_max - y_min) * 0.08 or max(abs(y_max), 1.0) * 0.08
    y_lo, y_hi = y_min - pad, y_max + pad

    names = sorted(per_model, key=lambda n: per_model[n][1])  # best WAPE first
    ncols = 3
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.2 * nrows), squeeze=False)
    for i, name in enumerate(names):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        by_date, wape = per_model[name]
        ax.plot(by_date.get_column(C.DATE), by_date.get_column("actual"),
                color=ACTUAL_C, lw=2.0, label="actual")
        ax.plot(by_date.get_column(C.DATE), by_date.get_column("predicted"),
                color=PRED_C, lw=1.6, ls="--", label="predicted")
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"#{i + 1}  {name}  (WAPE {wape:.3f})", fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(True, color=GRID_C, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for j in range(len(names), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=10, frameon=False)
    fig.suptitle(
        f"{HORIZON_LABEL[h]} — {SCOPES[scope_key]['label']}\n"
        f"actual (blue) vs predicted (orange) — same y-axis on every panel here",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    outdir = FIGURES / HOR_DIR[h]
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{scope_key}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def build_model_horizon_card(name: str, h: int, preds: pl.DataFrame, tags: pl.DataFrame,
                             scratch: Path) -> Path | None:
    """One model, one horizon: its top20 / meaningful / all diagnostic (the same
    3-panel figure `run_model()` draws -- scatter, actual-vs-predicted-over-time,
    per-split WAPE) stacked into 3 rows. Each row is already 3 panels wide, so the
    result is a genuine 3x3, not the old model.png's 3x9 (3 scopes x [3 horizons x 3
    panels]) -- this is that same picture cut apart at the horizon boundaries.

    Each row keeps its own y-axis (top20's few, high-volume SKUs are a different scale
    from the whole catalogue's average) -- the cross-model shared axis is a separate,
    deliberate thing, done in `build_horizon_scope_grid` instead.
    """
    row_imgs: dict[str, Path] = {}
    for scope_key, spec in SCOPES.items():
        sub = subset_rows(preds, tags, spec["mask"], h)
        if sub.is_empty():
            return None
        per_fold = []
        for fold_id in sorted(sub.get_column("fold").unique().to_list()):
            row = sub.filter(pl.col("fold") == fold_id)
            if row.is_empty():
                continue
            y = row.get_column("actual").to_numpy()
            p = row.get_column("predicted").to_numpy()
            per_fold.append({"fold": fold_id, **evaluate(y, p)})
        if not per_fold:
            return None
        p_out = scratch / f"{name}_{scope_key}.png"
        _fig_for_horizon(name, h, per_fold, sub, p_out)
        row_imgs[scope_key] = p_out

    fig, axes = plt.subplots(3, 1, figsize=(16, 5.4 * 3))
    for r, scope_key in enumerate(("top20", "meaningful", "all")):
        ax = axes[r]
        ax.imshow(mpimg.imread(row_imgs[scope_key]))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(ROW_LABEL[scope_key], fontsize=13, fontweight="bold")
    fig.suptitle(f"{name} — {HORIZON_LABEL[h]}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    outdir = FIGURES / HOR_DIR[h]
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def write_report(scored: dict[str, list[dict[str, Any]]],
                 grids: dict[int, dict[str, Path]],
                 cards: dict[int, dict[str, Path]]) -> Path:
    L = ["# Model comparison, restricted to a subset of the catalogue", "",
         f"*Generated {datetime.now():%Y-%m-%d %H:%M} by "
         "`model_lab/compare_subsets.py`, from the same fitted models and folds as "
         "`REPORT.md` -- nothing here is refit.*", "",
         "The main board pools every SKU, so it's a volume-weighted average that can "
         "hide how a model does on the handful of items you'd actually reorder by hand, "
         "or on the catalogue worth forecasting at all once the long tail is set aside. "
         "These two boards answer that.", ""]
    for key in SCORED_SCOPES:
        ranked = scored[key]
        label = SCOPES[key]["label"]
        fig = FIGURES / f"_comparison_{key}.png"
        L += [f"## {label}", "", f"![{key}]({fig.relative_to(LAB).as_posix()})", ""]
        for h in HORIZONS:
            if not any(str(h) in r["horizons"] for r in ranked):
                continue
            L += [f"**{HORIZON_LABEL[h]}**", "", _table(ranked, h), ""]

    L += ["## Actual vs. predicted over time, one folder per horizon", "",
          "Every model, one subplot each, same y-axis within each image so the "
          "panels are directly comparable.", ""]
    for h in HORIZONS:
        if h not in grids:
            continue
        L += [f"**{HORIZON_LABEL[h]}**", ""]
        for key in ("top20", "meaningful", "all"):
            p = grids[h].get(key)
            if p:
                L.append(f"- [{key}]({p.relative_to(LAB).as_posix()})")
        L.append("")

    L += ["## Per-model detail, one card per horizon", "",
          "Each card: top20 / meaningful / all stacked top to bottom, one model's "
          "own scatter + actual-vs-predicted-over-time + per-split WAPE alongside "
          "each other -- a 3x3 in the literal sense, not the cross-model comparison "
          "above.", ""]
    for h in HORIZONS:
        if not cards.get(h):
            continue
        L += [f"**{HORIZON_LABEL[h]}**: "
              + ", ".join(f"[{name}]({p.relative_to(LAB).as_posix()})"
                          for name, p in sorted(cards[h].items())),
              ""]
    REPORT.write_text("\n".join(L))
    return REPORT


def main() -> None:
    print(f"\n{'=' * 86}\nSUBSET COMPARISON\n{'=' * 86}")
    df, _ = load_dataset()
    tags = (add_top_meaningful_tags(df)
            .select(C.SKU, C.DATE, "vol_rank", "meaningful")
            .unique(subset=[C.SKU, C.DATE]))

    print("\n[loading cached per-row predictions]")
    all_preds = load_model_preds()
    print(f"  {len(all_preds)}/{len(MODEL_FILES)} models have a cache: "
          f"{', '.join(sorted(all_preds))}")

    scored: dict[str, list[dict[str, Any]]] = {}
    for key in SCORED_SCOPES:
        spec = SCOPES[key]
        print(f"\n[{spec['label']}]")
        ranked = score_subset(all_preds, tags, spec["mask"])
        for r in ranked:
            cells = "  ".join(f"{r['horizons'][str(h)]['wape']:.3f}" if str(h) in r["horizons"]
                              else "  --  " for h in HORIZONS)
            print(f"  {r['model']:<20} mean={mean_wape(r):.3f}   {cells}")
        comparison_figure(ranked, FIGURES / f"_comparison_{key}.png")
        scored[key] = ranked

    print("\n[actual vs predicted over time, one folder per horizon]")
    grids: dict[int, dict[str, Path]] = {}
    for h in HORIZONS:
        grids[h] = {}
        for key, spec in SCOPES.items():
            out = build_horizon_scope_grid(h, key, spec["mask"], all_preds, tags)
            if out is None:
                print(f"  ! nothing to plot for {HORIZON_LABEL[h]} / {key}")
                continue
            grids[h][key] = out
            print(f"  {out.relative_to(LAB.parent)}")

    print("\n[per-model cards, one 3x3 (top20/meaningful/all x 3 panels) per horizon]")
    cards: dict[int, dict[str, Path]] = {h: {} for h in HORIZONS}
    with tempfile.TemporaryDirectory(prefix="compare_subsets_") as tmp:
        scratch = Path(tmp)
        for h in HORIZONS:
            for name, preds in all_preds.items():
                out = build_model_horizon_card(name, h, preds, tags, scratch)
                if out is None:
                    print(f"  ! `{name}` / {HORIZON_LABEL[h]}: no data for one of the "
                          f"three scopes, skipping")
                    continue
                cards[h][name] = out
            print(f"  {HORIZON_LABEL[h]}: {len(cards[h])}/{len(all_preds)} models -> "
                  f"figures/{HOR_DIR[h]}/<model>.png")

    rep = write_report(scored, grids, cards)
    print(f"\nreport -> {rep.relative_to(LAB.parent)}")


if __name__ == "__main__":
    main()
