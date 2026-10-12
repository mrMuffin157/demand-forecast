# Findings

Lessons from building and stress-testing this forecasting pipeline on a multi-year, SKU-level
e-commerce sales history. Specific figures are deliberately left out: they described one
catalogue and would not transfer. What does transfer is the mechanism behind each result, so
that is what is written down here. Every claim below was measured with rolling-origin
backtests (many folds, each one a forecast made only from data available at its origin), not
argued from first principles.

---

## 1. Evaluation: how to not fool yourself

**Never use random k-fold on a time series panel.** Random splits put tomorrow's rows in the
training set and score several times better than an honest split. `splits.py` is rolling-origin
with a horizon-sized gap between train and test.

**A timing bug makes scores better, so you will not notice it.** Every row is a forecast origin
`(sku, origin_date)`. Features use data up to and including the origin; targets are separate
columns. Anything measured on the target's own day belongs in the banned set, not the feature
matrix. The leakage tests (`tests/test_leakage.py`, `test_tuning_isolation.py`) exist because
nothing else would complain.

**Source data can ship its own leakage.** Reporting tables often carry pre-computed rolling
averages written at row-write time, which include the row's own day. They are dropped and
recomputed with a proper shift.

**The training table is a full horizon stale by construction.** A row can only be trained on once
its target has happened, so the newest training origin is at least one horizon old. Forecasting
from it silently answers a question about the past. The pipeline therefore writes two tables in
one pass: `lab_dataset` (has targets, used for training) and `open_origins` (the newest origins,
no targets, the only rows a forecast may start from). Building serving features anywhere else
is how train/serve skew gets in, and it produces plausible numbers rather than errors.

**Bias measured on a few folds is mostly regime, not model.** Early on, three folds suggested
every model under-forecast systematically. Over many more folds the average bias was close to
zero and the fold-to-fold swing was an order of magnitude larger than the average. The giveaway:
per-fold bias was strongly correlated across *all* models, including simple per-SKU baselines
that share no code with the tree models. When unrelated models are wrong together, the driver
is the demand level, not any estimator. Do not fit a correction on a handful of folds.

**Count folds, not averages.** Several "wins" that looked real on one or three folds disappeared
or reversed over twelve. The per-fold win count is the number to read; refit noise alone moves
headline metrics by a few tenths of a percentage point.

**Score what was actually issued.** `track.py` re-scores forecasts that were really produced, once
their windows close. It is the only check that can catch train/serve skew, because a backtest
stays clean when the serving path breaks.

---

## 2. Metrics and the objective

**WAPE, not MAPE.** A large share of SKU-days are zero, which makes MAPE undefined or explosive.

**WAPE is a reporting metric, not an objective.** It is a sum of absolute errors, so its optimum
is the conditional *median*, which on a right-skewed panel sits below the mean. Selecting or
tuning on WAPE alone quietly pushes forecasts low. Model selection here breaks ties toward low
|bias|.

**Price the asymmetry.** One unit short and one unit of dead stock do not cost the same. With
under-stock cost `CU` and over-stock cost `CO`, the cost-minimising forecast is the
`CU/(CU+CO)` quantile of demand. The ratio lives in `assumptions.toml` (example values only).
At `CU == CO` the cost metric reduces exactly to WAPE, which is what makes the two boards
comparable.

**Under an asymmetric cost, calibration beats accuracy.** Ranking models on inventory cost
reorders the WAPE leaderboard: a model that aims slightly higher can win on cost while losing on
WAPE. Read the cost board before the WAPE board when the output drives purchasing.

**The target quantile is not a hyper-parameter.** When it was left in a WAPE-driven search, the
search moved it toward the median, in effect deciding on the business's behalf that a stock-out
is cheap. It is derived from the cost ratio and a test fails if it reappears in a search space.

**Tuning on WAPE made cost worse.** Hyper-parameters chosen to minimise WAPE improved WAPE
slightly and worsened inventory cost for every model. Do not adopt tuned parameters wholesale.

**The lift multiplier has a closed form.** The cost-optimal scalar on a forecast is the
`CU/(CU+CO)` quantile of `actual / predicted`, weighted by *predicted* (it falls out of the
first-order condition). Weighting by actual looks reasonable and over-orders the whole catalogue
by several percent.

**Raw newsvendor arithmetic can ask for absurd service levels.** On a high-margin catalogue the
per-SKU ratio from prices alone asks for an extreme tail quantile that cannot be measured on a
few folds. `economics.py` anchors per-SKU ratios so prices set the *spread* and the configured
ratio sets the *level*. A side effect: the unsourceable global guesses (holding rate, recapture,
markdown risk) cancel exactly, which a test asserts. A measured per-SKU storage cost breaks that
cancellation, so adopting one is a real decision, not a free upgrade.

---

## 3. Data realities

**Stock-out days are censored, not zero.** An out-of-stock zero is missing demand. Those days get
sample weight 0. Target windows that were partly out of stock sold less than their demand, so
the shipped tree models train on a *lifted* target (units divided by the in-stock share of the
window) but are still *scored* on what actually sold. Scoring on the lift would grade the model
against a guess.

**Siblings absorb some lost demand.** When one variant of a product is out of stock, its sibling
variants pick up a meaningful slice of its demand (a placebo check confirms it). Removing that
double count from the lift cost more precision than it saved.

**The last day of an export is usually partial.** A pull taken mid-day reads as a demand collapse.
`panel.py` detects and drops an incomplete tail.

**A SKU can vanish because it could not be bought.** Forecasting starts from the newest origins,
so a SKU whose rows stopped is *absent*, not zero. `coverage.py` classifies every catalogue SKU
(planned / starved / incoming / dormant / discontinued) using daily stock, because "out of stock
with nothing inbound" is exactly the SKU a reorder is most urgent for.

**Null is not zero.** A SKU with no stock record gets a null order quantity. In Polars `NaN > 0`
evaluates true, so a NaN sentinel would silently order a full window of cover for every
unmatched SKU. A test pins it.

**Check the input, not just the model.** A wrong input produces a plausible forecast, which no
model metric can see. `datacheck.py` fingerprints counts, coverage, null rates and *quantiles*
(not means, which a single bulk order moves) against an accepted build, and fails when the
modelling table falls behind the panel.

---

## 4. Modelling results (qualitative)

**Tree models beat per-SKU baselines** (moving average, Croston, seasonal naive) clearly on both
WAPE and inventory cost; the moving average stays in production as the floor, and the pipeline
flags loudly if it ever wins.

**Direct models per horizon beat summing monthly sub-models.** Additive sub-models' errors lean
the same way and do not cancel.

**Log-target plus one calibration factor beat Tweedie for precision.** Squared error on
`log1p(y)`, exponentiated back and multiplied by `Σactual/Σpredicted` on the training rows, put
more SKUs within ±40% than the Tweedie objective at every horizon, with the largest gains on
small sellers. Without the calibration factor a log fit aims below the mean. Median (absolute
error) loss was worse everywhere.

**A three-model blend helps modestly at longer horizons only.** Averaging models whose errors are
only partly correlated (gradient boosting, the same with in-stock demand features, a random
forest) beat the best single model at 90 and 180 days and was noise at 30 days, so it does not
run there.

**Pool stores, add a store marker.** One model trained on several stores (with a one-hot store
column and each store's own holiday calendar) beat separate per-store models. The marker lets one
model share what the stores have in common (seasonal shape, the big sales events) while still
learning each store's own level.

**Watch near-unique categoricals in gradient boosting.** A product-family categorical with almost
one level per SKU let partition splits memorise per-level offsets that went stale. Forcing
one-vs-rest splits fixed it.

**A specialist model for low sellers did not beat the global model** on those same SKUs. Small
SKUs miss by more in percentage terms because small counts are noisy, not because the model is
biased against them. Beware banding on *actual* volume: it manufactures a large fake bias through
regression to the mean. Band on predicted or trailing volume.

**Per-SKU calibration is capped by evidence, not by code.** Origins are weekly, so a long horizon
gives many rows per SKU but only about one non-overlapping window. Per-SKU correction helped at
short horizons and lost at the longest.

**Extrapolating past the longest trained horizon is cheap but not free.** Purchase windows (lead
time + review period) can exceed the longest backtested horizon; the cumulative forecast is then
extended at a flat rate. Training that horizon directly was barely better on WAPE and the cost
difference flipped sign fold to fold: not worth a fourth horizon.

**There is a precision ceiling, and it is level shifts.** Most large misses were sales doubling or
halving relative to the prior period. A perfect-rate Poisson process would land almost every SKU
within tolerance, so the noise floor is not the barrier. A walk-forward risk classifier can find
the safest slice of the catalogue; ranking by "smallest forecast first" is nearly as good.

---

## 5. Things that did not help (do not retry without a new reason)

- Blending in the previous weeks' forecasts for the same target.
- Correcting by recent realised error (the signal is real but too weak to move precision reliably).
- Removing the sibling double count from the stock-out lift.
- Training on windows that were heavily out of stock.
- Catalogue / category / brand momentum features (one value per date, too few regimes to learn).
- A zero-shot time-series foundation model, alone or as a blend member.
- Reconciling SKU forecasts to line-level totals (line forecasts were worse than summed SKUs).

---

## 6. Human input and known-future signals

**Explicit promo overrides double-count.** Replaying many real promotions through the override
layer moved inventory cost by roughly nothing. The reason: the forecast multiplier was fitted on
history that already contained promo days, so the *average* promotional response is already
inside it. An event resembling something the model has seen many times is priced in; enter only
events with no precedent (a launch, a delisting, a booked bulk order).

**Overrides must be honest about time and uncertainty.** Each event carries `entered_at` (it never
applies to an origin it postdates) and a mandatory `probability` (deals do get cancelled; the
applied effect is the expected one). Overrides are applied after the model, never as features,
because a backfilled event would silently improve every backtest.

**Most "known future" inputs are not known early enough.** Measured as notice period against the
purchase decision window, promotions arrive far too late to matter. One table that appeared to
carry long notice had in fact been bulk-rewritten with hindsight; checking `date_created`
distributions caught it.

**Score the incumbent fairly.** A business's own planning forecast is often edited during the
period it describes. Scored raw it looks impossibly good or bad; only rows written before the
period began are a fair comparison.
