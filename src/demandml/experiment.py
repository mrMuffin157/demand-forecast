"""The experiment runner: fit N models across M folds, log everything, rank the results.

This is the piece that makes the repo a *research* workbench rather than a script. One
call gives you a leaderboard with identical splits, identical feature sets and identical
metrics across every model, plus a JSON record on disk so a result from last week is
still reproducible and comparable today.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import config as C
from .features import feature_columns, load_features
from .logging_utils import get_logger
from .metrics import evaluate, evaluate_by
from .models import get_model
from .splits import Fold, apply_fold, assert_no_leakage, make_folds

log = get_logger(__name__)


@dataclass
class RunConfig:
    """Everything that defines one experiment. Serialised alongside the results."""

    models: list[str] = field(default_factory=lambda: ["ma28", "lgbm_tweedie"])
    horizon: int = 7
    target: str | None = None  # defaults to f"y_h{horizon}"
    freq: str = "daily"
    splits: C.SplitSpec = field(default_factory=lambda: C.DEFAULT_SPLITS)
    #: Exclude stock-out-censored zero days from training AND scoring. On by default:
    #: scoring a demand model on days when the product could not be bought measures
    #: stock-outs, not demand.
    drop_censored: bool = True
    #: Use the report's bulk-filtered units instead of raw sales as the target.
    use_filtered_target: bool = False
    #: Row cap for fast iteration (stratified by taking the most recent rows).
    max_train_rows: int | None = None
    #: Fraction of the tail of each training window held out for early stopping.
    val_fraction: float = 0.1
    sku_limit: int | None = None
    #: Extra feature columns to exclude.
    drop_features: list[str] = field(default_factory=list)
    tag: str = ""
    seed: int = 0

    def target_col(self) -> str:
        return self.target or f"y_h{self.horizon}"


def _prepare(cfg: RunConfig) -> tuple[pl.DataFrame, list[str], str]:
    """Load the feature table, select columns, drop unusable rows. Returns eagerly --
    one materialisation reused across every model and fold, so the comparison is exact.
    """
    lf = load_features(cfg.freq)
    ycol = cfg.target_col()
    feats = feature_columns(lf, drop_extra=cfg.drop_features)

    keep = [C.SKU, C.DATE, ycol, "sample_weight", "is_censored", C.TARGET]
    if cfg.use_filtered_target:
        alt = ycol.replace("y_h", "y_h")  # targets are built from cfg's panel target
        log.warning("use_filtered_target requires rebuilding features with "
                    "target='units_filtered'; ignoring for this run")
    extra = [c for c in ("sku_family", "adi_91", "units_mean_28") if c in lf.collect_schema().names()]
    cols = list(dict.fromkeys(keep + extra + feats))

    lf = lf.select([c for c in cols if c in lf.collect_schema().names()])
    lf = lf.filter(pl.col(ycol).is_not_null())
    if cfg.sku_limit:
        sub = lf.select(pl.col(C.SKU).unique().sort().head(cfg.sku_limit)).collect()
        # .to_list(): is_in with a Series of the same dtype is deprecated as ambiguous.
        lf = lf.filter(pl.col(C.SKU).is_in(sub.get_column(C.SKU).to_list()))

    df = lf.collect(engine="streaming")
    log.info("loaded %s rows x %d features for target %s", f"{len(df):,}", len(feats), ycol)
    return df, feats, ycol


def _fold_arrays(
    df: pl.DataFrame, fold: Fold, cfg: RunConfig, feats: list[str], ycol: str
):
    tr = apply_fold(df.lazy(), fold, "train").collect()
    te = apply_fold(df.lazy(), fold, "test").collect()

    if cfg.drop_censored:
        tr = tr.filter(pl.col("sample_weight") > 0)
        te = te.filter(pl.col("sample_weight") > 0)

    if cfg.max_train_rows and len(tr) > cfg.max_train_rows:
        tr = tr.tail(cfg.max_train_rows)  # most recent rows: the ones that matter

    assert_no_leakage(tr, te, cfg.horizon)

    n_val = int(len(tr) * cfg.val_fraction)
    fit, val = (tr.head(len(tr) - n_val), tr.tail(n_val)) if n_val > 100 else (tr, None)

    return fit, val, te


def run(cfg: RunConfig | None = None, save: bool = True, **kw: Any) -> "ExperimentResult":
    """Fit every model in ``cfg.models`` on every fold and return a comparable result."""
    cfg = cfg or RunConfig(**kw)
    run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    t_start = time.time()

    df, feats, ycol = _prepare(cfg)
    folds = make_folds(df.get_column(C.DATE).unique(), cfg.splits, horizon=cfg.horizon)
    if not folds:
        raise RuntimeError("no valid folds -- history too short for this SplitSpec")
    for f in folds:
        log.info(f.describe())

    rows: list[dict[str, Any]] = []
    preds: list[pl.DataFrame] = []
    extras: dict[str, Any] = {}

    for fold in folds:
        fit, val, te = _fold_arrays(df, fold, cfg, feats, ycol)
        if not len(te):
            continue
        Xf, yf = fit.select(feats), fit.get_column(ycol).to_numpy()
        Xt, yt = te.select(feats), te.get_column(ycol).to_numpy()
        wf = fit.get_column("sample_weight").to_numpy()
        Xv = val.select(feats) if val is not None else None
        yv = val.get_column(ycol).to_numpy() if val is not None else None

        # Per-SKU naive scale for MASE/RMSSE, computed on the training window only.
        scale = float(np.mean(np.abs(np.diff(fit.get_column(C.TARGET).to_numpy())))) or 1.0

        for name in cfg.models:
            t0 = time.time()
            model = get_model(name)
            try:
                fit_kw = {}
                if Xv is not None and hasattr(model, "params"):
                    fit_kw["eval_set"] = (Xv, yv)
                model.fit(Xf, yf, wf, **fit_kw)
                yhat = model.predict(Xt)
            except Exception as e:  # keep the leaderboard alive if one model explodes
                log.error("model %s failed on fold %d: %s", name, fold.index, e)
                continue
            elapsed = time.time() - t0

            m = evaluate(yt, yhat, scale=scale)
            rows.append({"model": name, "fold": fold.index, "horizon": cfg.horizon,
                         "seconds": round(elapsed, 2), **m})
            preds.append(
                te.select(C.SKU, C.DATE)
                .with_columns(
                    pl.lit(name).alias("model"),
                    pl.lit(fold.index).alias("fold"),
                    pl.Series("y", yt),
                    pl.Series("yhat", yhat),
                )
            )
            log.info(
                "fold %d | %-16s wape=%.4f mae=%.3f bias=%+.4f  (%.1fs)",
                fold.index, name, m["wape"], m["mae"], m["bias"], elapsed,
            )

            if hasattr(model, "importances") and name not in extras:
                try:
                    extras[name] = model.importances().head(40).to_dicts()
                except Exception:
                    pass

    result = ExperimentResult(
        run_id=run_id,
        config=cfg,
        scores=pl.DataFrame(rows),
        predictions=pl.concat(preds) if preds else pl.DataFrame(),
        importances=extras,
        n_features=len(feats),
        features=feats,
        folds=folds,
        elapsed=time.time() - t_start,
    )
    if save:
        result.save()
    return result


@dataclass
class ExperimentResult:
    run_id: str
    config: RunConfig
    scores: pl.DataFrame
    predictions: pl.DataFrame
    importances: dict[str, Any]
    n_features: int
    features: list[str]
    folds: list[Fold]
    elapsed: float

    def leaderboard(self, by: str = "wape") -> pl.DataFrame:
        """Mean and spread of each metric across folds, best first.

        The cross-fold *standard deviation* is the column people skip and then get
        burned by: a model that wins fold 3 by 8% and loses fold 1 by 6% has not beaten
        anything.
        """
        if self.scores.is_empty():
            return self.scores
        return (
            self.scores.group_by("model")
            .agg(
                pl.col("wape").mean().alias("wape"),
                pl.col("wape").std().alias("wape_sd"),
                pl.col("mae").mean().alias("mae"),
                pl.col("rmse").mean().alias("rmse"),
                pl.col("bias").mean().alias("bias"),
                pl.col("shortfall").mean().alias("shortfall"),
                pl.col("mase").mean().alias("mase"),
                pl.col("seconds").mean().round(1).alias("fit_s"),
                pl.len().alias("folds"),
            )
            .sort(by)
        )

    def vs_baseline(self, baseline: str = "ma28") -> pl.DataFrame:
        """Percent improvement in WAPE over a named baseline. The only number that
        answers "was any of this worth it"."""
        lb = self.leaderboard()
        if baseline not in lb.get_column("model").to_list():
            return lb
        base = lb.filter(pl.col("model") == baseline).get_column("wape")[0]
        return lb.with_columns(
            ((1.0 - pl.col("wape") / base) * 100).round(2).alias(f"vs_{baseline}_%")
        )

    def by_group(self, model: str, group: str = "sku") -> pl.DataFrame:
        """Per-SKU (or per-anything) metrics for one model's predictions."""
        p = self.predictions.filter(pl.col("model") == model)
        d = evaluate_by(p.get_column("y"), p.get_column("yhat"), p.get_column(group))
        return pl.DataFrame([{group: k, **v} for k, v in d.items()]).sort("wape")

    def save(self, root: Path | None = None) -> Path:
        root = (root or C.EXPERIMENTS) / self.run_id
        root.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": self.run_id,
            "created": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(self.elapsed, 1),
            "n_features": self.n_features,
            "features": self.features,
            "folds": [asdict(f) | {k: str(v) for k, v in asdict(f).items() if k != "index"}
                      for f in self.folds],
            "config": _jsonable(asdict(self.config)),
            "env": {"python": platform.python_version(), "platform": platform.platform()},
        }
        (root / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        if not self.scores.is_empty():
            self.scores.write_parquet(root / "scores.parquet")
            self.leaderboard().write_csv(root / "leaderboard.csv")
        if not self.predictions.is_empty():
            self.predictions.write_parquet(root / "predictions.parquet")
        if self.importances:
            (root / "importances.json").write_text(json.dumps(self.importances, indent=2, default=str))
        log.info("saved run to %s", root)
        return root


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _jsonable(v)
        elif isinstance(v, (tuple, set)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def load_run(run_id: str) -> dict[str, Any]:
    """Reload a saved run's artefacts for comparison against a new one."""
    root = C.EXPERIMENTS / run_id
    out: dict[str, Any] = {"meta": json.loads((root / "meta.json").read_text())}
    for name in ("scores", "predictions"):
        p = root / f"{name}.parquet"
        if p.exists():
            out[name] = pl.read_parquet(p)
    return out


def compare_runs(*run_ids: str) -> pl.DataFrame:
    """Stack the leaderboards of several saved runs side by side."""
    frames = []
    for rid in run_ids:
        d = load_run(rid)
        if "scores" in d:
            frames.append(d["scores"].with_columns(pl.lit(rid).alias("run_id")))
    return pl.concat(frames) if frames else pl.DataFrame()
