# Copilot Instructions for nfl-predictor

## 0) Mission + Non-Negotiables

This repository predicts NFL outcomes and scores for **Pick 'Em** and **Confidence Pools**.

- Primary objective: calibrated win probabilities and weekly confidence rankings.
- Secondary objective: realistic score outputs for display.

Rules that are always enforced:

- No data leakage. Features, splits, calibration, and evaluation only use information available
  before the predicted games (“known at prediction time” for the specific week, including
  calibration).
- Time-aware evaluation. Hyperparameter tuning and model selection use blocked, time-ordered splits.
- Polars-first ETL. Dataset creation and feature engineering run in Polars; pandas/numpy are
  acceptable inside ML modules only as needed.
- nflreadpy is the data source. NFLverse data is pulled via `nflreadpy`.
- Reproducible artifacts. Training and backtests write run directories with metadata and metrics.
- Tests are required. New functionality includes unit tests and improves or maintains code coverage.

## Engineering Standards (Logic, Docs, Lint, Coverage)

- Docstrings are required for every module, class, and function (including tests).
- Address linter and type-checker findings as they arise; avoid leaving new warnings behind.
- Do not reference temporary planning artifacts in code: do not mention roadmap items, milestone
  numbers, or TODO goal labels in any code, comments, docstrings, or test descriptions.
- Aim for maximum test coverage where practical; prefer small, deterministic unit tests.
- If a Python file grows beyond ~2000 lines, propose a refactor plan to split it into smaller
  focused modules (e.g., helpers/utils), and implement the split when it reduces complexity.
- Keep `TODO.md` accurate: verify items before checking them off.
- Keep `README.md` current: update it when behavior, CLI usage, features, or outputs change.

## Project Shape (Big Picture)

- **Primary Pipeline:** Polars for data processing + `nflreadpy` for NFLverse sources (schedule,
  team stats, Elo, etc.). The pipeline integrates schedule/results, team statistics, Elo ratings,
  TeamRankings stats, and market odds to produce ML-ready datasets.
- **Orchestration Script:** `nfl_predictor/data_collection.py` (run as a module). This
  orchestrator fetches data, applies transformations, and writes output CSVs.
- **Core Data Transforms:** Polars ETL helpers live under `nfl_predictor/utils/polars/`.
  `nfl_predictor/utils/polars_utils.py` is a compatibility facade that forwards imports to the
  split modules.
- **Game-Specific Enrichments:** `nfl_predictor/utils/game_utils.py` contains domain-specific
  calculations and dataset enrichments.
- **External Data Scraping/Caching:** `nfl_predictor/utils/scraping_utils.py` fetches and caches
  external web data used by the pipeline.

ML implementation layout:

- `nfl_predictor/ml/` contains the split ML implementation modules.
- `nfl_predictor/ml_model.py` is a compatibility facade for legacy imports and the CLI entrypoint.
- XGBoost version/build compatibility helpers live in `nfl_predictor/ml/ml_model_xgb_utils.py`.

## Modeling Philosophy (Important Context for Code Generation)

- Implementation is purely in Python.
- Team strength is represented by learned relationships between engineered features and outcomes.
- Feature interactions and weights are learned by the model; feature engineering provides signal,
  not fixed scoring formulas.

## Data Inputs/Outputs (Repo Conventions)

- **Data Directory:** all datasets live under `data/` (see `constants.DATA_PATH` in
  `nfl_predictor/constants.py`).
- **Key Output Files (examples; do not hard-code filenames):**
  - `data/all_data_ml.csv` - master ML dataset (includes engineered features and targets where
    available)
  - `data/all_data.csv` - combined dataset without ML-only targets
  - `data/completed_games_ml.csv` and `data/completed_games.csv` - completed games subsets
  - `data/predict/week_XX_games_to_predict.csv` - upcoming week games with engineered features
- **Caching:** historical weeks use cached files when available; the current week may require live
  refresh for supported sources.

I/O rules:

- Prefer the project’s Polars-based load/save helpers in `nfl_predictor/data_collection.py`.
- `nfl_predictor/utils/csv_utils.py` is legacy (pandas). Do not extend it; migrate call sites toward
  Polars when touching related code.

## Column & Schema Rules (Source of Truth)

- Column names and schema lists are defined in `nfl_predictor/constants.py`.
- Team identifiers are normalized using the canonical mapping in `constants.py`.
- Do not hard-code column lists; use constants to prevent schema drift.
- ETL does not silently drop columns.
- Always normalize team identifiers via `constants.ALIAS_TO_CANONICAL` and (when available)
  `normalize_team_column(df, col)`.

## Season/Week Logic & Edge Cases

- Use `constants.get_regular_season_weeks(season)` to determine regular-season length.
- Week 1 / early-season rows with missing history use a documented fallback consistent with tests
  (regression-to-mean and/or previous-season values + global mean).
- Future games have missing outcomes; the pipeline still outputs a structurally complete row
  suitable for prediction.

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

Calibration methods:

- Support Platt scaling (logistic regression) and isotonic regression.
- A normal-CDF mapping from margin is acceptable as a baseline only; if calibration is enabled,
  calibrated output is the default.

### Market features

When market lines exist, the system produces market-derived features and supports market anchoring:

- Market transforms produce `market_home_margin`, `market_total_line`, `home_market_prob`,
  `away_market_prob`.
- Market anchoring trains residuals vs market baselines and adds the baseline back at prediction
  time.
- Market probability blending/clamping uses explicit CLI/config values and is validated in
  time-aware evaluation.

Market anchoring details:

- Prefer residual training: `target_resid = target - market_baseline` and
  `pred = market_baseline + pred_resid`.

### Uncertainty

- Predictions include uncertainty intervals for margin and total (p10/p50/p90 or equivalent).
- Reports include interval diagnostics.

Minimum requirement:

- Output a median plus at least one interval for both margin and total (quantiles preferred).

### Realistic score outputs

- Realistic score outputs are produced as post-processing applied after margin/total predictions are
  generated.
- Realistic score adjustments are used for display and reporting.
- Realistic score adjustments do not alter win probabilities, confidence rankings, pool scoring, or
  tuning objectives.

If implementing score “realism”:

- Apply post-processing only after core predictions; rounding/snapping policies must be
  configurable.
- Never change training targets to enforce “NFL score lattice” unless explicitly designed and
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

- Season-blocked CV (acceptable baseline).
- Walk-forward evaluation (required): for each season and each week `w` (e.g., `3..end`), train on
  all games strictly before week `w` (plus prior seasons if configured), predict week `w`, and
  record metrics.

Required metrics:

- margin MAE
- total MAE
- win probability Brier score
- win probability log loss
- binned reliability summary
- confidence pool point summaries

Market-relative metrics (when market anchoring is enabled):

- residual MAE vs market baseline for margin/total.
- optionally edge vs spread/total for diagnostics; do not claim profitability.

Required run artifacts:

- saved model artifact
- metadata JSON (see “Model artifact contract”)
- metrics report JSON (walk-forward aggregated + per-season/per-week summaries)
- plots are optional and must not block CI

## ML Implementation Standards

Preprocessing:

- Use `ColumnTransformer` for categorical one-hot + numeric passthrough/impute.
- Avoid densifying large sparse matrices unintentionally.
- Tree-based models (XGBoost): do not use `StandardScaler` unless a non-tree model requires it.
- Missing values: XGBoost can handle them; impute only if required for consistency.

Training:

- Use early stopping and set `eval_metric` explicitly (aligned to objective).
- Tune hyperparameters consistently with the evaluation metric (Optuna optional but supported).
- Use `random_state` everywhere applicable.
- Do not hard-code `n_jobs`; prefer `os.cpu_count()` or a config default.

Blending:

- Prefer explicit, interpretable blends (market anchoring often sufficient).
- If using a blender/regressor, avoid unstable unconstrained weights; prefer non-negative or
  sum-to-1 if implemented.
- Validate blends using time-aware splits.

## Leakage Audit (Required)

Maintain a leakage audit tool/mode (see `scripts/leakage_audit.py`):

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

## Dependency & Environment Hygiene

- Pin key ML dependencies for reproducibility: xgboost, scikit-learn, numpy, pandas, polars,
  scipy, optuna (if used).
- Document supported Python version(s) and CPU/GPU constraints if applicable.
- Avoid optional GPU paths that break CPU-only execution unless explicitly guarded.

## Feature Development Rules

All engineered features apply to **every matchup**, not only end-of-season games.

Feature areas tracked in `TODO.md` include:

- season-to-date record features (overall, division, conference W-L-T)
- divisional rivalry indicator
- team health and injury burden features (team-week and positional aggregations)
- lookahead/trap indicators (next-week opponent strength + rest/travel context)
- motivational asymmetry features (playoff leverage and clinch/elimination context)

## Missing Data Rules

Some sources do not exist for all seasons.

- ETL emits an invariant schema for every run.
- Missing sources become nulls or defined defaults.
- The model path handles nulls without crashing and reports how many rows use fallbacks for each
  feature group.

## constants.py Hygiene

`nfl_predictor/constants.py` remains the canonical reference for:

- file paths
- feature column names/lists
- team mapping tables
- season/week rules
- default parameters used in data collection

Unused constants are removed and the file remains organized into clear sections.

## Dev Workflows (How to Run Things)

- Refresh data:
  - `python -m nfl_predictor.data_collection`
- Testing:
  - `pytest`
  - `pytest --cov=nfl_predictor --cov-report=term-missing`
- Validation:
  - `python scripts/validate_offline.py`
  - `python scripts/validate_live.py`

Training/prediction entrypoints may be updated/replaced, but must remain runnable and documented.

## Logging & Coding Style

- Logging uses the project logger (`from nfl_predictor.utils.logger import log`). No `print`.
- Formatting and linting:
  - Black (line length 100)
  - isort (Black profile)
  - flake8 (E402 ignored)
- Type hints are required for new/modified modules.
- Prefer Polars expressions over Python loops.

## Safety, Scope, and Prohibited Behaviors

- Do not introduce offensive/unsafe content or harmful instructions.
- Do not add unrelated features (dashboards, new scrapers, unrelated pipelines).
- Do not remove/alter existing pipeline behavior without updating tests and documentation.
- Avoid new external services or network dependencies beyond existing scraping utilities.
- Do not claim betting profitability; report metrics and uncertainty honestly.

## Choosing APIs & Libraries

- Polars is used for ETL and feature engineering.
- `nflreadpy` is used for NFLverse data.
- scikit-learn and XGBoost are used for modeling.

## GPT-5.2-Codex Guidance

- Use existing project utilities and constants.
- Implement changes in small, testable increments.
- Keep outputs deterministic under fixed seeds.
- Do not change behavior without updating tests and documentation.
