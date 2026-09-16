"""Is the flat-rate extension past 180 days good enough to buy against?

    uv run python dev/longhorizon.py                      # xgboost_gbdt, all folds
    uv run python dev/longhorizon.py --model random_forest
    uv run python dev/longhorizon.py --folds 3            # quicker

Any order quantity sized on a window longer than the longest trained horizon depends on
the answer to this question.

`policy.py` buys against `lead time + review period` (set in `assumptions.toml`). The
longest horizon anything here is trained on is 180. When the window is longer, the
cumulative-demand curve is extended past its last knot at the 90-to-180 daily rate.
`window_backtest.py` can only show the score DRIFTING as the window grows -- it has no
270-day truth to compare against, so it cannot say whether the drift is the horizon
getting genuinely harder or the extension being wrong.

`model_lab/prepare_data.py` carries `y_cum270` as a VALIDATION target: built like any
other, banned from the feature matrix by the `y_cum*` rule, and deliberately absent from
`common.HORIZONS` so no model trains on it. That makes the comparison possible:

    extrapolated   the serving path. Cached 30/90/180 predictions pushed through
                   `policy.interpolate_cumulative`, exactly as an order is sized.
    direct         the same estimator trained on `y_cum270` and asked the question
                   straight.

Both scored against the same realised 270-day demand, on the same rows, same folds.

-----------------------------------------------------------------------------------
WHAT A RESULT HERE MEANS
-----------------------------------------------------------------------------------
If `direct` wins by a wide margin, 270 belongs in `common.HORIZONS` and every model
should be refitted at a fourth horizon -- a real cost, justified.

If it does not, the extension is doing its job and the honest thing is to say so and stop
treating the extrapolation as an open wound. Either way the flat-rate extension stops
being an untested assumption underneath every purchase order.

This file does NOT change the forecast. It measures. Adopting the result is a separate,
deliberate act. See docs/FINDINGS.md.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import policy as P  # noqa: E402
from common import LAB, load_dataset, make_folds, results_dir, split  # noqa: E402
from demandml import config as C  # noqa: E402

LONG_H = 270


def metrics(y: np.ndarray, p: np.ndarray, cu: float, co: float) -> dict:
    den = float(np.sum(y))
    if den <= 0:
        return {k: float("nan") for k in ("wape", "bias", "fill", "cost")}
    short, over = np.maximum(y - p, 0.0), np.maximum(p - y, 0.0)
    return {"wape": float(np.sum(np.abs(y - p)) / den),
            "bias": float(np.sum(p - y) / den),
            "fill": float(1.0 - np.sum(short) / den),
            "cost": float(np.sum(cu * short + co * over) / den)}


def cached_curve(model: str) -> pl.DataFrame:
    """One row per scored origin: the cached cumulative predictions at 30/90/180.

    These come from the 180-day folds, whose test windows do not line up with the folds a
    270-day target produces -- the gap before each test period is horizon-sized, so it is
    90 days wider here. The two methods are therefore scored on the INTERSECTION of the
    rows both can answer for, not on whatever each happens to cover. Scoring them on
    different rows would compare the samples rather than the methods.
    """
    d = results_dir() / "_preds" / f"{model}.parquet"
    if not d.exists():
        raise SystemExit(f"no cached predictions at {d} -- run the model first")
    wide = (pl.read_parquet(d).filter(pl.col("h").is_in(list(common.HORIZONS)))
            .pivot(on="h", index=[C.SKU, C.DATE], values="predicted"))
    hcols = [str(h) for h in common.HORIZONS]
    missing = [c for c in hcols if c not in wide.columns]
    if missing:
        raise SystemExit(f"cached predictions for {model} lack horizons {missing}")
    return wide.drop_nulls(hcols).select([C.SKU, C.DATE] + hcols)


def extrapolated_at(curve: pl.DataFrame, window: int) -> np.ndarray:
    """Cumulative demand at `window` days, via the SERVING curve.

    Calls `policy.interpolate_cumulative` rather than reimplementing it, so a change to
    the path an order is sized on changes this measurement too -- the same discipline
    `window_backtest.py` uses.
    """
    cum = curve.select([str(h) for h in common.HORIZONS]).to_numpy()
    out, _ = P.interpolate_cumulative(cum, window)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default="xgboost_gbdt")
    ap.add_argument("--folds", type=int, default=12)
    ap.add_argument("--max-train-rows", type=int, default=300_000)
    ap.add_argument("--sku-limit", type=int, default=None)
    ap.add_argument("--tuned", action="store_true")
    args = ap.parse_args()
    common.set_mode("tuned" if args.tuned else "initial")
    cu, co = common.cost_ratio()

    df, feats = load_dataset(args.sku_limit)
    ycol = f"y_cum{LONG_H}"
    if ycol not in df.columns:
        raise SystemExit(
            f"the dataset has no {ycol}. Rebuild it:\n"
            f"  uv run demandml features --freq daily && "
            f"uv run python model_lab/prepare_data.py")

    print(f"\n{'=' * 92}\nLONG HORIZON   [{common.MODE}]   model={args.model}   "
          f"target={ycol}   cu:co = {cu:g}:{co:g}\n{'=' * 92}")
    usable = int(df.get_column(ycol).is_not_null().sum())
    print(f"  {usable:,} of {df.height:,} origins have a complete {LONG_H}-day target")

    folds = make_folds(df, LONG_H, n_folds=args.folds)
    mod = importlib.import_module(args.model).MODEL
    curve = cached_curve(args.model)
    print(f"  {curve.height:,} origins have cached 30/90/180 predictions to extend from")

    ys, direct, extrap, per_fold = [], [], [], []
    for f in folds:
        tr, te = split(df, f, LONG_H, args.max_train_rows)
        if not len(tr) or not len(te):
            continue
        # Both methods must answer for exactly the same rows.
        te = te.join(curve.select(C.SKU, C.DATE), on=[C.SKU, C.DATE], how="semi")
        if not len(te):
            print(f"  fold {f.index}: no overlap with the cached predictions -- skipped")
            continue
        ex = extrapolated_at(
            te.select(C.SKU, C.DATE).join(curve, on=[C.SKU, C.DATE], how="left"), LONG_H)

        est = mod.build(LONG_H)
        X = tr.select(feats) if feats and mod.uses_features else tr
        y = tr.get_column(ycol).to_numpy()
        w = tr.get_column("sample_weight").to_numpy() if "sample_weight" in tr.columns else None
        t0 = time.time()
        if mod.needs_frame:
            est.fit(X, y, w, frame=tr)
        else:
            est.fit(X, y, w)
        Xt = te.select(feats) if feats and mod.uses_features else te
        dp = np.clip(np.nan_to_num(np.asarray(est.predict(Xt), dtype=np.float64).ravel()),
                     0.0, None)

        yt = te.get_column(ycol).to_numpy()
        ys.append(yt)
        direct.append(dp)
        extrap.append(ex)
        md, me = metrics(yt, dp, cu, co), metrics(yt, ex, cu, co)
        per_fold.append({"fold": f.index, "test_start": str(f.test_start),
                         "test_end": str(f.test_end), "n": len(te),
                         "direct": md["cost"], "extrapolated": me["cost"]})
        print(f"  fold {f.index:>2}  test {f.test_start}..{f.test_end}  n={len(te):,}  "
              f"{time.time() - t0:.0f}s   direct cost {md['cost']:.4f}   "
              f"extrapolated {me['cost']:.4f}")

    if not ys:
        raise SystemExit("nothing scored -- no fold had both a fit and cached predictions")

    y = np.concatenate(ys)
    d, e = np.concatenate(direct), np.concatenate(extrap)
    md, me = metrics(y, d, cu, co), metrics(y, e, cu, co)
    gap = (me["cost"] - md["cost"]) / md["cost"] if md["cost"] else float("nan")

    print(f"\n{'-' * 92}")
    print(f"  {'':<16}{'WAPE':>9}{'bias':>9}{'fill':>9}{'cost':>10}")
    for name, m in (("direct", md), ("extrapolated", me)):
        print(f"  {name:<16}{m['wape']:>9.4f}{m['bias']:>+9.4f}{m['fill']:>9.3f}"
              f"{m['cost']:>10.4f}")
    print(f"\n  the flat-rate extension costs {100 * gap:+.2f}% against training the "
          f"horizon directly")
    print(f"{'-' * 92}")

    rows = {"model": args.model, "horizon": LONG_H, "n": int(len(y)),
            "folds": len(ys), "direct": md, "extrapolated": me, "gap_pct": 100 * gap,
            "per_fold": per_fold, "cu": cu, "co": co}
    (results_dir() / "_longhorizon.json").write_text(json.dumps(rows, indent=2))

    L = [
        "# Is the flat-rate extension good enough to buy against?\n",
        f"*Generated by `dev/longhorizon.py` (mode `{common.MODE}`) from "
        f"`lab_dataset.parquet`'s `{ycol}` validation target. `{args.model}`, "
        f"{len(ys)} folds, {len(y):,} scored origins.*\n",
        f"Every order this project sizes is exposed to `lead time + review period`, "
        f"while the longest trained horizon is 180. Wherever that window is longer, the "
        f"cumulative curve is extended past its last knot. This scores that extension "
        f"against realised {LONG_H}-day demand.\n",
        "| method | WAPE | bias | fill | cost |",
        "|---|---|---|---|---|",
        f"| `direct` — trained on `{ycol}` | {md['wape']:.4f} | {md['bias']:+.4f} | "
        f"{md['fill']:.3f} | **{md['cost']:.4f}** |",
        f"| `extrapolated` — the serving path | {me['wape']:.4f} | {me['bias']:+.4f} | "
        f"{me['fill']:.3f} | **{me['cost']:.4f}** |",
        "",
        f"**The flat-rate extension costs {100 * gap:+.2f}% of inventory cost** against "
        f"training the horizon directly.\n",
        "Read that with the WAPE column "
        f"({md['wape']:.4f} against {me['wape']:.4f}). If the two methods are about "
        "equally ACCURATE, what separates them is calibration, and under an asymmetric "
        "cost calibration is what gets paid for.\n",
        "## Fold by fold\n",
        "The aggregate hides the thing that decides how much to believe it.\n",
        "| fold | test window | n | direct | extrapolated | extension costs |",
        "|---|---|---|---|---|---|",
    ]
    flips = 0
    for r in per_fold:
        d_ = (r["extrapolated"] - r["direct"]) / r["direct"] if r["direct"] else float("nan")
        flips += d_ < 0
        L.append(f"| {r['fold']} | {r['test_start']} → {r['test_end']} | {r['n']:,} | "
                 f"{r['direct']:.4f} | {r['extrapolated']:.4f} | **{100 * d_:+.2f}%** |")
    L.append("")
    if flips and flips < len(per_fold):
        L.append(f"**The sign flips.** The extension is cheaper than the direct model in "
                 f"{flips} of {len(per_fold)} folds and dearer in the rest. A mean of "
                 f"{100 * gap:+.2f}% over folds that disagree that strongly is not a "
                 f"margin this project would act on anywhere else -- it is the same "
                 f"pattern `regime.py` looks for in the bias, and for the same reason: "
                 f"a 270-day window spans most of a year, so a fold is close to a single "
                 f"observation of whatever demand regime it landed in.\n")

    if gap > 0.05:
        L.append(f"That is large enough to act on. `{LONG_H}` belongs in "
                 f"`common.HORIZONS`, which means refitting every model at a fourth "
                 f"horizon and rebuilding every board -- a real cost, and on this "
                 f"evidence a justified one.\n")
    elif gap > 0.0:
        L.append("That is a real but modest penalty. Weigh it against refitting every "
                 "model at a fourth horizon and maintaining a fourth column on every "
                 "board; the honest reading is that the extension is doing most of its "
                 "job, not that it is free.\n")
    else:
        L.append("**The extension is not the problem.** Training the horizon directly "
                 "does not beat it, so the flat-rate extrapolation can stop being "
                 "treated as an open wound underneath every purchase order. Note what "
                 "this does and does not say: the long horizon is still harder than the "
                 "short one, and both methods are worse at 270 days than either is at "
                 "90. What is settled is that the EXTENSION is not where that loss "
                 "comes from.\n")

    L += [
        "## What this does not settle\n",
        f"- **One estimator.** Scored with `{args.model}`. A model that extrapolates "
        f"differently -- `random_forest` cannot extrapolate beyond its training range at "
        f"all -- may not behave the same way. Re-run with `--model` to check.\n",
        f"- **{LONG_H} days is one point.** If the coverage window falls between 180 and "
        f"{LONG_H}, a {LONG_H}-day knot would let a real order interpolate rather than "
        f"extend past the last knot. That is strictly a better position, and it is the "
        f"reason to carry the target even if `direct` does not win.\n",
        "- **The sample is older.** A 270-day target needs 270 days of future, so these "
        "origins stop earlier than the 180-day board's. Some of the difference from "
        "the cost board is the sample moving, not the horizon.\n",
        "- **Nothing here changed the forecast.** `policy.py` still extends past 180. "
        "Adopting the knot is a separate, deliberate change.\n",
    ]
    out = LAB / "LONGHORIZON.md"
    out.write_text("\n".join(L))
    print(f"\nwrote {out.relative_to(LAB.parent)}")


if __name__ == "__main__":
    main()
