"""FEATURE IMPORTANCE & THE `channel_category` QUESTION

    uv run python model_lab/feature_importance.py

-----------------------------------------------------------------------------------
WHAT THIS FILE IS FOR
-----------------------------------------------------------------------------------
No other file in this folder looks inside the winning model to see what it actually
uses. The only earlier evidence was accidental: during the categorical-encoding bug hunt
described in xgboost_gbdt.py's header, dropping every categorical column
(channel_category, brand_id, sku_family, sku_brand, type, uom, tier, status, material,
...) on a *single* fold scored marginally *better* than keeping them. That was a side
effect of debugging something else, not a real ablation.

This file runs the real version: the champion model (xgboost_gbdt's XGBoostTweedie),
across every horizon and every fold, twice per fold --

    (a) every feature the dataset offers
    (b) the same, minus `channel_category`

-- plus the model's own gain-based feature importances from every (a) run, so
`channel_category`'s contribution is a number instead of an inference from a debugging
anecdote or from intuition about what "should" matter.

-----------------------------------------------------------------------------------
READING THE OUTPUT
-----------------------------------------------------------------------------------
`delta = wape_without - wape_full`. Positive means removing the column made the
forecast worse (the column was pulling weight); negative or ~0 means removing it did
not hurt, i.e. the model was not relying on it. Compare `delta` against `wape_sd`
(the fold-to-fold spread) the same way the rest of this folder compares any two
numbers -- a delta smaller than the spread has not established anything.

The importance table is gain-based (`get_score(importance_type="gain")`), which only
lists columns the model actually split on. A column missing from it was not used in
that fit at all, not just used lightly.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import polars as pl
from tqdm import tqdm

from common import FIGURES, GRID_C, HORIZONS, HORIZON_LABEL, RESULTS, load_dataset, make_folds, score, split
from xgboost_gbdt import XGBoostTweedie

DROP_COL = "channel_category"
MAX_TRAIN_ROWS = 300_000
N_FOLDS = 3
TEST_DAYS = 90
TOP_N_FEATURES = 15


def fit_and_score(tr: pl.DataFrame, te: pl.DataFrame, feats: list[str], ycol: str, horizon: int):
    y_tr = tr.get_column(ycol).to_numpy().astype(np.float64)
    y_te = te.get_column(ycol).to_numpy().astype(np.float64)
    w_tr = tr.get_column("sample_weight").to_numpy().astype(np.float64)

    model = XGBoostTweedie(horizon)
    model.fit(tr.select(feats), y_tr, w_tr)
    yhat = np.asarray(model.predict(te.select(feats)), dtype=np.float64).ravel()
    yhat = np.clip(np.nan_to_num(yhat), 0.0, None)
    return score(y_te, yhat), model


def main() -> None:
    df, feats = load_dataset()
    if DROP_COL not in feats:
        raise SystemExit(f"'{DROP_COL}' is not in the current feature list -- nothing to ablate")
    feats_without = [c for c in feats if c != DROP_COL]

    jobs = [(h, fold) for h in HORIZONS for fold in make_folds(df, h, n_folds=N_FOLDS, test_days=TEST_DAYS)]
    total_fits = len(jobs) * 2

    fold_rows: list[dict] = []
    gain_totals: dict[str, float] = defaultdict(float)
    gain_appearances: dict[str, int] = defaultdict(int)
    n_full_fits = 0

    bar = tqdm(total=total_fits, desc="feature importance", unit="fit")
    for h, fold in jobs:
        ycol = f"y_cum{h}"
        tr, te = split(df, fold, h, max_train_rows=MAX_TRAIN_ROWS, seed=fold.index)
        if not len(tr) or not len(te):
            bar.update(2)
            continue

        bar.set_postfix_str(f"h={h}d fold={fold.index} full")
        m_full, model_full = fit_and_score(tr, te, feats, ycol, h)
        bar.update(1)
        n_full_fits += 1
        for name, gain in model_full.importances():
            gain_totals[name] += gain
            gain_appearances[name] += 1

        bar.set_postfix_str(f"h={h}d fold={fold.index} without {DROP_COL}")
        m_drop, _ = fit_and_score(tr, te, feats_without, ycol, h)
        bar.update(1)

        fold_rows.append({
            "horizon": h, "fold": fold.index,
            "wape_full": m_full["wape"], "wape_without": m_drop["wape"],
            "delta": m_drop["wape"] - m_full["wape"],
            "bias_full": m_full["bias"], "bias_without": m_drop["bias"],
        })
        print(f"\n  h={h:>3}d fold={fold.index}  "
              f"wape_full={m_full['wape']:.4f}  wape_without={m_drop['wape']:.4f}  "
              f"delta={m_drop['wape'] - m_full['wape']:+.4f}")
    bar.close()

    if not fold_rows:
        raise SystemExit("no folds produced results -- check the dataset / horizons")

    # ---- aggregate the ablation -------------------------------------------------------
    by_horizon = {}
    for h in HORIZONS:
        rows = [r for r in fold_rows if r["horizon"] == h]
        if not rows:
            continue
        by_horizon[str(h)] = {
            "wape_full": float(np.mean([r["wape_full"] for r in rows])),
            "wape_without": float(np.mean([r["wape_without"] for r in rows])),
            "delta": float(np.mean([r["delta"] for r in rows])),
            "delta_sd": float(np.std([r["delta"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "wape_sd_full": float(np.std([r["wape_full"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
    overall = {
        "wape_full": float(np.mean([r["wape_full"] for r in fold_rows])),
        "wape_without": float(np.mean([r["wape_without"] for r in fold_rows])),
        "delta": float(np.mean([r["delta"] for r in fold_rows])),
        "delta_sd": float(np.std([r["delta"] for r in fold_rows], ddof=1)) if len(fold_rows) > 1 else 0.0,
    }

    # ---- aggregate gain importances (averaged over fits where the column was used;
    #      absent = zero gain, i.e. the model never split on it in that fit) -----------
    avg_gain = {name: gain_totals[name] / n_full_fits for name in gain_totals}
    top_features = sorted(avg_gain.items(), key=lambda kv: -kv[1])[:TOP_N_FEATURES]
    drop_col_gain = avg_gain.get(DROP_COL, 0.0)
    drop_col_rank = next((i + 1 for i, (name, _) in enumerate(
        sorted(avg_gain.items(), key=lambda kv: -kv[1])) if name == DROP_COL), None)
    drop_col_appearances = gain_appearances.get(DROP_COL, 0)

    print(f"\n{'=' * 86}\nSUMMARY\n{'=' * 86}")
    print(f"{DROP_COL}: appeared in {drop_col_appearances}/{n_full_fits} fits' importance table, "
          f"avg gain={drop_col_gain:.1f}, rank={drop_col_rank or 'never used'} of "
          f"{len(avg_gain)} features the model ever split on")
    print(f"\noverall (mean over {len(fold_rows)} horizon x fold fits):")
    print(f"  wape with {DROP_COL}:     {overall['wape_full']:.4f}")
    print(f"  wape without {DROP_COL}:  {overall['wape_without']:.4f}")
    print(f"  delta:                   {overall['delta']:+.4f}  (+/-{overall['delta_sd']:.4f} sd)")
    print(f"\nby horizon:")
    for h in HORIZONS:
        if str(h) not in by_horizon:
            continue
        agg = by_horizon[str(h)]
        print(f"  {HORIZON_LABEL[h]:<16} wape_full={agg['wape_full']:.4f}  "
              f"delta={agg['delta']:+.4f} (+/-{agg['delta_sd']:.4f})  "
              f"[fold wape spread +/-{agg['wape_sd_full']:.4f}]")
    print(f"\ntop {TOP_N_FEATURES} features by average gain across {n_full_fits} fits:")
    for i, (name, gain) in enumerate(top_features, 1):
        marker = "  <-- " + DROP_COL if name == DROP_COL else ""
        print(f"  {i:>2}. {name:<28} {gain:>10.1f}{marker}")

    # ---- save results -------------------------------------------------------------
    RESULTS.mkdir(exist_ok=True, parents=True)
    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "xgboost_gbdt",
        "drop_col": DROP_COL,
        "n_full_fits": n_full_fits,
        "overall": overall,
        "by_horizon": by_horizon,
        "folds": fold_rows,
        "drop_col_gain": drop_col_gain,
        "drop_col_rank": drop_col_rank,
        "drop_col_appearances": drop_col_appearances,
        "top_features": top_features,
        "n_features_ever_used": len(avg_gain),
        "n_features_offered": len(feats),
    }
    (RESULTS / "_feature_importance.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / '_feature_importance.json'}")

    # ---- figure ---------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    names = [n for n, _ in reversed(top_features)]
    gains = [g for _, g in reversed(top_features)]
    colors = ["#F76707" if n == DROP_COL else "#4C6EF5" for n in names]
    ax.barh(names, gains, color=colors)
    ax.set_xlabel("average gain (xgboost_gbdt, summed over splits, per fit)")
    ax.set_title(f"top {TOP_N_FEATURES} features by importance\n"
                 f"({DROP_COL} in orange, rank {drop_col_rank or 'n/a'} of {len(avg_gain)})")
    ax.grid(axis="x", color=GRID_C, linewidth=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    hs = [h for h in HORIZONS if str(h) in by_horizon]
    full_v = [by_horizon[str(h)]["wape_full"] for h in hs]
    without_v = [by_horizon[str(h)]["wape_without"] for h in hs]
    x = np.arange(len(hs))
    width = 0.35
    ax.bar(x - width / 2, full_v, width, label="with channel_category", color="#4C6EF5")
    ax.bar(x + width / 2, without_v, width, label="without channel_category", color="#F76707")
    ax.set_xticks(x, [HORIZON_LABEL[h] for h in hs])
    ax.set_ylabel("mean WAPE across folds")
    ax.set_title(f"does dropping {DROP_COL} hurt accuracy?")
    ax.legend()
    ax.grid(axis="y", color=GRID_C, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(f"xgboost_gbdt -- feature importance & the {DROP_COL} ablation", fontweight="bold")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIGURES / "_feature_importance.png", dpi=130)
    print(f"wrote {FIGURES / '_feature_importance.png'}")


if __name__ == "__main__":
    main()
