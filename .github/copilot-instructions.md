# Copilot Instructions for nfl-predictor

## 0) Mission + Non-Negotiables
This repo predicts NFL outcomes and scores for **Confidence Pools / Pick ’Em**. The primary objective is **reliable win probabilities and rankings**; realistic score outputs are secondary but must be coherent.

**Non-negotiables**
- **No data leakage**: never train/evaluate using information from after the game week being predicted.
- **Time-aware evaluation**: backtests must reflect “known at prediction time.”
- **Reproducible artifacts**: models must be loadable and comparable across runs.
- **Polars-first pipeline**: prefer Polars for ETL. Pandas/Numpy are acceptable inside ML modules only as needed.

## 1) Project Shape (Big Picture)
- **Primary Pipeline:** Polars + nflreadpy + multiple sources (schedule, stats, Elo, TeamRankings, odds) to create ML-ready datasets.
- **Orchestration:** `nfl_predictor/data_collection_polars.py` pulls/transforms data; writes CSVs under `data/`.
- **Core Polars transforms:** `nfl_predictor/utils/polars_utils.py`
- **Game enrichments:** `nfl_predictor/utils/game_utils.py`
- **Scraping/caching:** `nfl_predictor/utils/scraping_utils.py` (cache-first)

## 2) Data Inputs/Outputs (Repo Conventions)
- All data lives under `data/` (`constants.DATA_PATH`).
- Key outputs (examples, do not hard-code filenames):
  - `data/all_data_ml.csv` (master ML dataset)
  - `data/completed_games_ml.csv` (training/eval subset)
  - `data/predict/week_XX_games_to_predict.csv` (prediction inputs)

**I/O rules**
- Prefer the project’s Polars-based load/save helpers in `data_collection_polars.py`.
- `utils/csv_utils.py` is legacy (pandas). Do not extend it; migrate call sites toward Polars.

## 3) Column & Schema Rules (Source of Truth)
- Always use `nfl_predictor/constants.py` for column names and schema lists.
- Do not hard-code columns; use constants to prevent schema drift.
- Always normalize team identifiers via `constants.ALIAS_TO_CANONICAL` and `normalize_team_column(df, col)`.

## 4) Season/Week Logic & Edge Cases
- Use `constants.get_regular_season_weeks(season)`; never hard-code week counts.
- Week 1 / no-prior-games: regress to prior season values + global mean (or documented fallback), consistent with tests.
- Historical weeks: use cached data; current week may require live scrape for missing sources.
- Future games: handle missing results gracefully (null targets).

## 5) Modeling Philosophy (What to Build)
### 5.1 Canonical approach (preferred)
**Use a Margin/Total model**:
- Predict:
  - `margin = home_score - away_score`
  - `total = home_score + away_score`
- Derive scores:
  - `home = (total + margin)/2`
  - `away = (total - margin)/2`

Direct home/away score regressors are allowed only as secondary ensemble members.

### 5.2 Market integration (required capabilities)
Market features are **strong priors** but must be explicit and configurable.

Support:
- **Market transforms** (derived columns):
  - `market_home_margin` from spread
  - `market_total_line`
  - `home_market_prob`, `away_market_prob` from moneyline
- **Market anchoring** (preferred):
  - Train residuals: `target_resid = target - market_baseline`
  - Predict: `pred = market_baseline + pred_resid`
- Optional **market clamp/blend** for win probability:
  - Combine calibrated model probability with market implied probability via an explicit weight.
  - Weight must be configurable and validated in time-aware evaluation.

### 5.3 Calibration (required)
Win probabilities must be calibrated and evaluated using:
- **Brier score** and **log loss**
- Calibration methods:
  - Platt scaling (logistic regression)
  - Isotonic regression
- A normal-CDF mapping from margin is allowed as a baseline only; if calibration is enabled, calibrated output is the default.

### 5.4 Uncertainty (high priority)
Add **prediction intervals** for margin and total:
- Train quantile models (e.g., P10/P50/P90) or equivalent approach.
- Outputs must include median and at least one interval.

### 5.5 Realistic score outputs (optional but encouraged)
If implementing score “realism”:
- Apply post-processing only after core predictions:
  - rounding/snapping policies must be configurable
  - never change training targets to enforce “NFL score lattice” unless explicitly designed and documented
- Keep this logic separate from model training and evaluation.

## 6) Evaluation & Backtesting (Required)
### 6.1 Time-aware evaluation modes
Maintain and/or implement:
1) **Season-blocked CV** (acceptable baseline)
2) **Walk-forward evaluation** (required)
   - For each season in an eval range:
     - for each week `w` (e.g., 3..end):
       - train on all games strictly before week `w` (plus all prior seasons if configured)
       - predict games in week `w`
       - record metrics

### 6.2 Required metrics (report all)
- Margin/Total:
  - MAE for margin
  - MAE for total
- Win probabilities:
  - Brier score
  - log loss
  - calibration summary (binned reliability table)
- Pool utility:
  - weekly confidence ranking score:
    - rank by confidence strength (default: `abs(p-0.5)`)
    - compute realized confidence points using Section 7 rules
- Market-relative:
  - residual MAE vs market baseline (if market anchoring enabled)
  - optional: edge metrics vs spread/total (do not claim profit unless robustly backtested and clearly caveated)

### 6.3 Output artifacts (required)
Every training/backtest run must produce:
- a saved model artifact
- a metadata JSON (Section 9)
- a metrics report JSON (walk-forward aggregated + per-season/per-week summaries)
- plots optional (do not block CI)

## 7) Confidence Pool Rules (Authoritative)
- Each week assign unique confidence values `1..N` to the chosen winner in each matchup.
- Max weekly points = `N*(N+1)/2`
- Realized points = `sum(conf_i * 1[pick_i_correct])`
- Tie games: treat as incorrect for both sides.

**Production constraint**
- Picks are submitted **before the first game of the week** (typically TNF).
- Backtests must assume **single-shot picks** (no in-week updates).

## 8) ML Implementation Standards (How to Code It)
### 8.1 Preprocessing rules
- Tree-based models (XGBoost):
  - Do **not** use `StandardScaler` unless a non-tree model requires it.
  - Missing values: XGBoost can handle; impute only if required for consistency.
- Use `ColumnTransformer` for categorical one-hot + numeric passthrough/impute.
- Avoid densifying large sparse matrices unintentionally.

### 8.2 Training rules
- Use early stopping and set `eval_metric` explicitly (aligned to objective).
- Tune hyperparameters consistently with evaluation metric (Optuna optional but supported).
- Use `random_state` everywhere applicable.
- Do not hardcode `n_jobs`; prefer `os.cpu_count()` or a config default.

### 8.3 Blending rules
If blending signals (team model + market model):
- Prefer explicit, interpretable blends:
  - residual model via market anchoring (often sufficient)
  - simple linear blend with regularization
- If using a blender/regressor:
  - avoid unstable unconstrained weights; prefer non-negative or sum-to-1 if implemented
- Validate blends using time-aware splits.

### 8.4 Leakage audit (required)
Add/maintain a leakage audit tool/mode:
- Checks for target/label columns in features
- Flags suspiciously predictive columns (e.g., absurd correlations)
- Validates “season-to-date” features exclude the current game row

## 9) Reproducibility & Model Artifact Contract (Required)
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

## 10) Dependency & Environment Hygiene (Required)
- Pin key ML dependencies for reproducibility:
  - xgboost, scikit-learn, numpy, pandas, polars, scipy, optuna (if used)
- Document supported Python version(s) and GPU/CPU constraints if applicable.
- Avoid optional GPU paths that break CPU-only execution unless explicitly guarded.

## 11) Coding Style, Formatting, and Logging
- Use `from nfl_predictor.utils.logger import log` (no `print`).
- Formatting:
  - Black (line length 100)
  - isort (Black profile)
  - flake8 (E402 ignored)
- Type hints required in new/modified ML modules.
- Keep functions small, testable, and documented.
- Prefer explicit configuration over “magic defaults”.

## 12) Safety, Scope, and Prohibited Behaviors
- Do not introduce offensive/unsafe content or harmful instructions.
- Do not add unrelated features (UI dashboards, new scrapers, unrelated pipelines).
- Do not remove existing pipeline behavior without updating tests and documentation.
- Avoid introducing new external services or network dependencies beyond existing scraping utilities.
- Do not claim betting profitability; report metrics and uncertainty honestly.

## 13) Dev Workflows
- Refresh data: `python -m nfl_predictor.data_collection_polars`
- Tests: `pytest`
- Validation scripts:
  - `scripts/validate_offline.py`
  - `scripts/validate_live.py`
- Training/prediction entrypoints may be updated/replaced, but must remain runnable and documented.
