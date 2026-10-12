# demand-forecast

SKU-level demand forecasting and order planning for an e-commerce catalogue.

Given a daily sales history per SKU (units, price, promotions, stock-outs, traffic), it answers
two questions:

1. **How many units of each SKU will sell over the next 30 / 90 / 180 days?** (`dev/forecast.py`)
2. **How many should we order now?** given stock on hand, units on order, lead time and the cost of
   running out versus over-stocking. (`dev/policy.py`)

The repository ships with a **synthetic sample dataset generator**, so everything below runs end
to end without access to any real data.

---

## Approach

```
sales history ─► ingest ─► panel ─► features ─► prepare_data ─► backtest models ─► pick ─► forecast ─► order plan
 (data/db/)      typed      dense    leak-free    weekly origins   rolling-origin     per      30/90/180   lead time +
                 Parquet    SKU×day  lags/rolls   + targets        folds              horizon  days        review window
```

- **Rolling-origin backtesting.** Every model is scored only on forecasts it could have made at
  the time: weekly forecast origins, a horizon-sized gap between train and test, many folds.
  Random k-fold is never used.
- **Strict timing convention.** Features use data up to and including the origin day; targets are
  separate cumulative columns (`y_cum30/90/180`). Leakage tests guard it.
- **Censored demand.** Out-of-stock days are treated as missing demand, not zero sales. Partly
  out-of-stock training windows are lifted by their in-stock share; scoring stays on actual sales.
- **Direct multi-horizon models.** One model per horizon on the cumulative target. The board
  includes a moving average, Croston, random forest, XGBoost (log-target + calibration), several
  LightGBM variants, a two-stage hurdle model and a blend.
- **Cost-aware selection.** Models are reported on WAPE and on an asymmetric inventory cost
  (`CU` per unit short, `CO` per unit left over). The planning forecast is lifted to the
  `CU/(CU+CO)` quantile of demand by a multiplier fitted on backtest errors.
- **Order policy.** The cumulative forecast is interpolated onto each SKU's *lead time + review
  period* window, net of stock on hand and on order, with per-SKU service levels and MOQ / carton
  rounding (always up).
- **Multi-store pooling (optional).** Several stores can be trained together with a store marker
  while each store is still forecast separately.
- **Human overrides, kept honest.** Planned launches, delistings or bulk orders can be applied
  after the model (`dev/events.py`), with an `entered_at` date and a probability, and are scored
  separately once their windows close.

What was learned along the way, including the dead ends, is in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## Results

Measured on the original project's real sales history for its US and Canadian stores, with
rolling-origin backtests, rounded:

- About **20% lower inventory cost** than a moving-average baseline, at every horizon.
- The cost-aware forecast adjustment alone was worth about **a fifth of inventory cost**.
- Switching to a log-target fit with calibration put **2–5 percentage points more SKUs within
  ±40%** of actual sales, with the biggest gains on small sellers.
- Training the stores together beat separate per-store models by **about 1–2 points** for both
  the US and Canadian stores, in every recent fold tested.
- In use, the quarterly forecast drives planning only for SKUs it flags as **confident before
  the quarter starts**. Scored on seven closed quarters:
  - **US, high-confidence SKUs** (about 85% of revenue): **~92% landed within ±50%** of actual
    sales, and ~88% within ±40%.
  - **Canada, high- and medium-confidence SKUs** (about 70% of revenue): **~91% within ±50%**,
    and ~84% within ±40%.
  - The flags were calibrated: the share promised within ±50% matched the share that landed to
    within a couple of points.
- In its first live check after release, the forecast **matched its backtest accuracy** on real
  sales.

> These figures come from the original real-data project and **will not reproduce on the
> sample data**, which is synthetic. Running the pipeline on the sample gives its own numbers.

---

## Tech stack

Python 3.12 · [uv](https://docs.astral.sh/uv/) · Polars (lazy / streaming) · DuckDB · NumPy ·
scikit-learn · XGBoost · LightGBM · statsmodels · Optuna · Matplotlib · Typer + Rich (CLI) ·
pytest

---

## Quick start (sample data)

```bash
uv sync
```

```bash
uv run python scripts/make_sample_data.py
```

That writes a reproducible synthetic dataset under `data/` (120 SKUs, about four years of daily
sales). Then build the modelling tables:

```bash
uv run demandml ingest && uv run demandml panel --freq daily && uv run demandml features --freq daily
```

```bash
uv run python model_lab/prepare_data.py
```

Backtest the models and compare them:

```bash
for m in moving_average xgboost_gbdt random_forest blend; do uv run python dev/$m.py; done
```

```bash
uv run python dev/compare_models.py && uv run python dev/cost_board.py
```

Forecast forward and turn it into an order plan:

```bash
uv run python dev/forecast.py
```

```bash
uv run python dev/policy.py --inventory data/inventory/position_sample.csv
```

Run the tests:

```bash
uv run pytest tests/ -q
```

A self-contained narrative version of the whole idea (CSV → clean → features → several models →
one figure) is in `examples/walkthrough.py`.

---

## Configuration

- **`assumptions.toml`** holds every business assumption in one place: the under/over-stock cost
  ratio, holding / recapture / markdown rates, the default lead time and review period. **The
  shipped values are illustrative examples**, not measured from any business. Point
  `DEMANDML_ASSUMPTIONS` at your own file to change them.
- **Environment variables:**
  - `DEMANDML_DATA`: data directory (default `./data`).
  - `DEMANDML_MODELS`: comma-separated list that narrows the model board, e.g.
    `moving_average,random_forest,xgboost_gbdt,blend`.
  - `DEMANDML_MARKET` / `DEMANDML_POOL`: select a store, or train several stores together.
  - `DEMANDML_MODE`: pin baseline / tuned / cost-trained parameters for a whole process.
- **Your own data.** Replace the files in `data/db/` with an extract that has the same columns:
  `sales_daily`, `product`, and optionally `inventory_daily` and `deals`. The generator in
  `scripts/make_sample_data.py` documents the expected schema. A CSV in `data/raw/daily_sales_*.csv`
  also works as a fallback.

---

## Layout

```
src/demandml/    core library: ingest, cleaning, panel, features, splits, metrics, CLI
model_lab/       first-generation model board + prepare_data.py (builds the modelling tables)
dev/             shortlisted models, model selection, forecast, order policy, diagnostics
scripts/         sample data generator
tests/           leakage, timing, cost-metric, policy and pipeline tests
notebooks/       exploration notebooks (run against the sample data)
docs/            FINDINGS.md
assumptions.toml example business assumptions
```

Generated outputs (results, figures, forecasts, order plans, Markdown reports) are written next to
the scripts in `dev/` and `model_lab/` and are git-ignored.
