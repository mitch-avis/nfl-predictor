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

- Win probability is derived from the margin prediction through the deterministic floor,
  `Phi(margin / sigma)`, and nothing else: no fitted calibrator, no Elo-style curve, no market
  blend or clamp.
- Sigma for a game in season `s`, week `w` is the root-mean-square of (actual minus predicted
  margin) over every out-of-fold prediction strictly before `(s, w)`: earlier seasons, and season
  `s` weeks before `w` (`nfl_predictor/ml/floor_sigma.py`, one implementation for every run
  type). It is one value per week, so it never changes a pick or a confidence rank. Until the
  pool spans `FLOOR_SIGMA_MIN_POOL_SEASONS` (`3`) earlier seasons (any weeks), sigma is
  `SCORE_DIFF_STD_DEV` (`14.21`), recorded as the fallback.
- The pool: a walk-forward pools its own earlier weeks with any supplied history (none for a
  standalone `backtest` unless `--floor-sigma-reference-runs` names runs; the reference runs in
  the weekly run's stage 1), a week the run predicts replacing the history's. The production
  final fit pools the reference runs' fold checkpoints (`floor_sigma_reference_runs`, default
  the two GPU reference seeds, averaged per game) with the weekly run's stage-1 weeks of the
  predicted season before the predicted week, estimates sigma for the predicted week, and
  records it in the saved model; prediction from a saved model uses the recorded value, and a
  model saved without one uses the constant. A configured reference run that is missing is an
  error, never a silent fallback.
- Every run type uses it identically: the weekly run, `backtest`, `train` and `predict`. The
  weekly run's stage 1 walks this one production configuration forward to report how it scores;
  it selects nothing.
- The calibration option takes `auto` (the documented value and the default); `none` is an
  accepted second spelling. A saved model that carries a retired calibrator or market blend
  predicts the floor, and loading it logs what is ignored.
- Calibration metrics (Brier, log loss, reliability table) are reported in evaluation.
- Changing how probabilities are formed is a default change for the user to decide, measured on
  the walk-forward instrument against the floor first.

### Market integration

When market lines exist, the system produces market-derived features and supports market anchoring:

- Market transforms produce `market_home_margin`, `market_total_line`, `home_market_prob`,
  `away_market_prob`.
- Market anchoring trains residuals vs market baselines and adds the baseline back at prediction
  time.
- The market-implied home win probability (no-vig moneylines, else the spread) is a scored
  yardstick only: every walk-forward reports it beside the model, with the paired
  deterministic-minus-market intervals, and it never enters the submitted probability. There is no
  market probability blend or clamp.

Market anchoring details:

- Prefer residual training: `target_resid = target - market_baseline` and `pred = market_baseline +
pred_resid`.

No-vig market probabilities normalize the home and away implied probabilities to sum to 1.

### Uncertainty

- Predictions include uncertainty intervals for margin and total (p10/p50/p90 or equivalent).
- The intervals are outputs only: the win probability never reads them (it is the deterministic
  floor of the predicted margin, the margin head's squared-error point prediction).
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
- Predicted winner is derived from the deterministic win probability: home when the unrounded
  `p >= 0.5` (an exact 0.5 picks home), else away (`picks_home` in `nfl_predictor/ml/metrics.py`,
  shared by the walk-forward, the training pool summaries and the weekly output).
- Confidence strength is derived from the deterministic win probability (default:
  `abs(p - 0.5)`), rounded to 12 decimals (`CONFIDENCE_DECIMALS` in `nfl_predictor/ml/metrics.py`)
  so that mathematically equal confidences (a home favorite and a home underdog by the same
  spread) almost always tie instead of differing by floating-point noise; a pair whose noise
  straddles a 12-decimal rounding boundary can still differ.
- Ranks run from least confident (`1`) to most confident (`N`); equal rounded confidences are
  ordered by `game_id`. One shared rule (`confidence_ranks` in `nfl_predictor/ml/metrics.py`)
  ranks the weekly picks, walk-forward pool points, training pool summaries and
  `nfl-predictor compare`.
- Picks, ranks and the published confidence strength come from the unrounded probability; the
  published probability columns are rounded to 4 decimals for display only, so the weekly picks
  choose sides and rank exactly as the walk-forward does.

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
- The production final fit trains the same way: on every eligible completed game except the
  evaluation holdout seasons (`--holdout-seasons`, `0` in the weekly run), the newest completed
  week included. No fit holds weeks out of its trees or hands XGBoost an eval frame, and every
  head runs its full tree budget.

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

When multiple options exist (market integration mode, weighting choices, tuned parameters):

- Prefer selecting settings via walk-forward over multiple seasons.
- Pick a primary selection metric (typically Brier/log loss for probability quality) and use
  secondary tie-breakers (confidence pool expected points, then margin/total MAE).
- Report mean and variance across folds; avoid choosing a setting that wins by a hair on one season
  but regresses elsewhere. The stability view (each week bucket split by season, in
  `metrics_report.json` as `metrics.stability` and at the end of `nfl-predictor compare`'s report,
  per run and per paired contrast) shows this.
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
- the XGBoost device the model trained on (`xgb_device`: `cpu` or `cuda`, never `auto`)
- the seasons used for training and for the evaluation holdout (`splits`)
- the probability floor's sigma (`floor_sigma`): the value the model predicts with, whether it is
  the fallback constant, the week it was estimated for, its pool's game count and seasons, and
  the runs the pool came from
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
