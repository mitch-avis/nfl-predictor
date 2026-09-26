# Modeling, evaluation and feature specification

The full specification for predictions, evaluation and features. `AGENTS.md`, "Prediction,
evaluation and feature rules", restates the rules that must never be missed.

## Prediction & Modeling Logic

### Canonical targets

The primary model predicts:

- `margin = home_score - away_score`
- `total  = home_score + away_score`

Derived scores are computed as:

- `home_score = (total + margin) / 2`
- `away_score = (total - margin) / 2`

Direct home/away score regressors are allowed only as secondary ensemble members.

### Win probability

- Win probability is derived from the margin prediction.
- Win probabilities are calibrated using time-aware calibration data.
- Calibration metrics (Brier, log loss, reliability table) are reported in evaluation.

Calibration methods (canonical names):

- `none`: deterministic margin->prob mapping (baseline)
- `platt`: logistic regression (Platt scaling)
- `isotonic`: isotonic regression
- `elo`: deterministic Elo-style logistic mapping

Notes:

- Prefer time-aware calibration (`platt` or `isotonic`) when enough calibration rows exist.
- If adding new CLI options, keep names stable and document them.

### Market integration

When market lines exist, the system produces market-derived features and supports market anchoring:

- Market transforms produce `market_home_margin`, `market_total_line`, `home_market_prob`,
  `away_market_prob`.
- Market anchoring trains residuals vs market baselines and adds the baseline back at prediction
  time.
- Market probability blending/clamping uses explicit CLI/config values and is validated in
  time-aware evaluation.

Market anchoring details:

- Prefer residual training: `target_resid = target - market_baseline` and `pred = market_baseline +
pred_resid`.

Market probability post-processing (blend/clamp):

- Blending must be explicit and bounded (weights in [0, 1]).
- Clamping must be explicit and bounded (delta in [0, 0.5]).
- If adding "no-vig" market probability options, implement them consistently (home/away normalize to
  sum to 1) and validate in walk-forward.

### Uncertainty

- Predictions include uncertainty intervals for margin and total (p10/p50/p90 or equivalent).
- Interval outputs are evaluated (coverage/width diagnostics) and are part of the run artifacts.

Minimum requirement:

- Output a median plus at least one interval for both margin and total (quantiles preferred).

### Realistic score outputs

- Realistic score outputs are produced as post-processing applied after margin/total predictions are
  generated.
- Realistic score adjustments are used for display and reporting.
- Realistic score adjustments do not alter win probabilities, confidence rankings, pool scoring, or
  tuning objectives.

If implementing score "realism":

- Apply post-processing only after core predictions; rounding/snapping policies must be
  configurable.
- Never change training targets to enforce an "NFL score lattice" unless explicitly designed and
  documented.

### Confidence pool deliverable

- Weekly outputs include a **1..N** unique confidence ranking across that week's games.
- Predicted winner is derived from calibrated win probability.
- Confidence strength is derived from calibrated win probability (default: `abs(p - 0.5)`).

Authoritative pool scoring rules:

- Each week assign unique confidence values `1..N` to the chosen winner in each matchup.
- Max weekly points = `N*(N+1)/2`.
- Realized points = `sum(conf_i * 1[pick_i_correct])`.
- Tie games: treat as incorrect for both sides.
- Picks are submitted before the first game of the week (single-shot; no in-week updates in
  backtests).

## Evaluation

Required evaluation modes:

- **Season-blocked CV** (acceptable baseline; primarily used for hyperparameter tuning).
- **Walk-forward evaluation (authoritative):** for each season and each week `w` (e.g., `3..end`),
  train on all games strictly before week `w` (plus prior seasons if configured), predict week `w`,
  and record metrics.

Required metrics:

- margin MAE
- total MAE
- win probability Brier score
- win probability log loss
- binned reliability summary
- confidence pool point summaries (expected + actual)

Market-relative metrics (when market anchoring is enabled):

- residual MAE vs market baseline for margin/total
- edge vs spread/total as diagnostics only (do not claim profitability)

### Model selection protocol (how to choose "best" settings)

When multiple options exist (calibration method, market integration mode, probability blend/clamp
rules, weighting choices):

- Prefer selecting settings via walk-forward over multiple seasons.
- Pick a primary selection metric (typically Brier/log loss for probability quality) and use
  secondary tie-breakers (confidence pool expected points, then margin/total MAE).
- Report mean and variance across folds; avoid choosing a setting that wins by a hair on one season
  but regresses elsewhere.
- Never use the holdout window to tune hyperparameters.

Required run artifacts:

- saved model artifact
- metadata JSON (see "Model artifact contract")
- metrics report JSON (walk-forward aggregated + per-season/per-week summaries)
- plots are optional and must not block CI

## Leakage Audit (Required)

Maintain a leakage audit tool/mode (`nfl-predictor leakage-audit`,
`nfl_predictor/cli/leakage_audit.py`):

- checks for target/label columns in features
- flags suspiciously predictive columns (e.g., absurd correlations)
- validates season-to-date features exclude the current game row

## Reproducibility & Model Artifact Contract

Every saved model must include adjacent metadata JSON with:

- created timestamp
- git commit hash (if available)
- dataset fingerprint (hash of training CSV and/or stable row ids)
- library versions (xgboost, sklearn, numpy, pandas, polars, scipy)
- training config (CLI args / config object)
- season/week ranges used for train/calibration/holdout
- feature list used
- best params (if tuned) and early-stopping info

Artifacts must be loadable without hidden external state.

## Feature Development Rules

All engineered features apply to **every matchup**, not only end-of-season games.

Feature areas tracked in `.agents/TODO.md` include (examples):

- season-to-date record features (overall, division, conference W-L-T)
- divisional rivalry indicator
- lookahead/trap indicators (next-week opponent strength + rest/travel context)
- motivational asymmetry features (playoff leverage and clinch/elimination context)
- PBP-derived per-snap EPA, success, explosive, and special-teams families
- weekly schedule-adjusted (ridge) offense/defense strength and EPA-based schedule strength, in
  both the ridge form and the one-hop head-to-head-excluded form
- QB per-dropback EPA families for the expected starter

Rules for stat-style features:

- Carry counts and sums through season-to-date aggregation and compute rates afterward (ratio of
  sums), the way `_compute_derived_metrics` already works.
- Name allowed/defensive metrics explicitly and add them to `EXCLUDE_FROM_OPPONENT_STATS` so the
  generic `opponent_` mirror does not duplicate them. Before excluding a stat, grep
  `_compute_derived_metrics` for its `opponent_` mirror: a derived metric that reads it needs
  the stat listed in `OPPONENT_MIRROR_INTERMEDIATES` too, or the derived columns go null at the
  next rebuild (the `0.12.4` sack exclusion did exactly that, fixed in `0.12.6`).
- Cite the formula in the docstring and test each self-computed metric against a hand-built
  fixture.
- Any schedule-adjusted or opponent-adjusted value for week `N` must be solved from games strictly
  before week `N` in that season, with a documented prior-season fallback for early weeks.
- A change that alters feature *values* at ETL time (a prior blend, a regression factor, a new
  fallback) cannot be ablated with `--disable-feature-groups`, which only drops columns. Ablate it
  with two dataset builds behind an ETL flag, back up `data/*.csv` before each rebuild, and keep
  both walk-forward reports under `models/`. Measure early-season changes with `--wf-start-week 1`
  and report weeks 1, 2, and 3-18 separately.
