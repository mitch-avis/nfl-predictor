# Copilot Instructions for nfl-predictor

## Project Shape (Big Picture)
- **Primary Pipeline:** The project uses **Polars** for data processing, combined with **nflreadpy** to fetch raw data (schedule, team stats, Elo ratings, etc.). The pipeline integrates multiple data sources (schedule results, team statistics, ELO ratings, TeamRankings stats, betting odds) to produce a rich, ML-ready dataset for predictions.
- **Orchestration Script:** The main data assembly is done in [`nfl_predictor/data_collection_polars.py`](../nfl_predictor/data_collection_polars.py) (run as a module). This orchestrator pulls the data, applies all transformations, and saves output CSVs.
- **Core Data Transforms:** Polars transformations and helper functions live in [`nfl_predictor/utils/polars_utils.py`](../nfl_predictor/utils/polars_utils.py). This includes functions to normalize team names, compute derived feature columns (offensive/defensive metrics, matchup deltas, rolling stats), and other Polars DataFrame manipulations.
- **Game-Specific Enrichments:** Additional domain-specific calculations (e.g., filling missing QB data, computing SurvivorGrid spreads, adding moneyline odds) are in [`nfl_predictor/utils/game_utils.py`](../nfl_predictor/utils/game_utils.py). These enrich the dataset after core data assembly.
- **External Data Scraping/Caching:** Data from external sites (TeamRankings stats, SurvivorGrid lines, etc.) is handled via [`nfl_predictor/utils/scraping_utils.py`](../nfl_predictor/utils/scraping_utils.py). This module fetches and caches web data so that it can be reused without unnecessary network calls.

## Modeling Philosophy (Important Context for Code Generation)
- This project no longer relies on spreadsheets or spreadsheet formulas for implementation.
- However, the **modeling philosophy** is informed by years of structured analytical work:
  - Teams are represented by **composite offensive and defensive ratings**, derived from aggregated statistics.
  - Matchups are evaluated by **comparing offensive strength vs opponent defensive strength**, and vice versa.
  - Overall team strength (e.g., Elo or power rating) provides a baseline expectation, while matchup-specific deltas capture stylistic advantages.
  - Historical game data is used to **learn relationships** between these features and actual outcomes (scores, spreads, totals).
- These ideas should be implemented **purely in Python**, using statistical or ML methods (e.g., regression, gradient boosting), **not by recreating spreadsheet logic**.
- When generating code, Copilot/GPT should:
  - Translate these concepts into **features and models**
  - Prefer **learned relationships** over hardcoded weights
  - Treat all feature weighting and interactions as data-driven unless explicitly stated otherwise

## Data Inputs/Outputs (Repo Conventions)
- **Data Directory:** All data files reside under the `data/` directory (see `constants.DATA_PATH` in [`nfl_predictor/constants.py`](../nfl_predictor/constants.py)). The pipeline reads from and writes to this location.
- **Key Output Files:**
  - `data/all_data_ml.csv` – Master dataset for machine learning (includes game results with engineered features and target variables such as team scores, point differentials, or totals).
  - `data/all_data.csv` – Combined dataset without ML-only columns.
  - `data/completed_games_ml.csv` and `data/completed_games.csv` – Subsets of completed games used for training, validation, and evaluation.
  - `data/predict/week_XX_games_to_predict.csv` – Upcoming games for the current week with all features prepared for prediction.
- **TeamRankings Cache:** TeamRankings data is cached per season/week:
  - `data/<season>/<season>_week_<WW>_team_rankings.csv` (weekly snapshot)
  - `data/<season>/<season>_team_rankings.csv` (season-to-date)
  - Prefer cached data for historical weeks; scrape only when missing.
- **Loading/Saving Functions (Current Reality):**
  - For CSV IO in the Polars pipeline, prefer `df.write_csv(...)` / `pl.read_csv(...)` via
    `save_dataframe()` / `load_dataframe()` in [`nfl_predictor/data_collection_polars.py`](../nfl_predictor/data_collection_polars.py).
  - `utils/csv_utils.py` exists but is **legacy/outdated** (pandas-based). Do not extend it without first converting it to Polars.

## Column & Schema Rules (Source of Truth)
- **Constants Module:** Always refer to [`nfl_predictor/constants.py`](../nfl_predictor/constants.py) for authoritative definitions of column names and schema:
  - **nflreadpy Columns:** `NFLREADPY_SCHEDULE_COLUMNS`, `NFLREADPY_SCHEDULE_RENAME`
  - **Output Ordering:** `FIRST_COLUMNS`, `LINES_COLUMNS`, `RESULT_COLUMNS`
  - **Polars Dataset Schema:** `POLARS_METADATA_COLUMNS`, `POLARS_LINES_COLUMNS`, `POLARS_RESULT_COLUMNS`, and stat field lists
- **Do not hard-code column names.** Use constants to avoid schema drift.
- **Team Name Normalization:** Always normalize team abbreviations/names via `constants.ALIAS_TO_CANONICAL`. Use `normalize_team_column(df, col)` whenever ingesting team identifiers.

## Season/Week Logic & Edge Cases
- **Season Length Variations:** Use `constants.get_regular_season_weeks(season)` to determine the correct number of regular-season weeks. Do not hard-code week counts.
- **Start-of-Season Edge Case:** In Week 1 (or when a team has no prior games), features requiring historical context must fall back to previous-season values with regression-to-mean. Tests in [`tests/test_data_collection_polars.py`](../tests/test_data_collection_polars.py) assert this behavior.
- **Historical vs Current Week:**
  - Historical weeks should load fully from cached data.
  - The current week may require live scraping for certain sources.
- **Future Weeks:** Upcoming games will have missing outcome data. Code must gracefully handle nulls/NaNs and still output a structurally complete dataset suitable for prediction.

## Prediction & Modeling Logic
- **Primary product goals:** support **Confidence pools** and **Pick ’Em** decisions.
  - Confidence pools require *good win-probability ranking* (separation + calibration).
  - “Realistic scores” matter, but are secondary to expected confidence points.
- **ML Code Status:** `ml_model.py` and `ml_utils.py` are known to be outdated. It is acceptable to pivot to a new modeling approach (regression/classification/both) as long as:
  - it consumes the existing datasets under `data/` (not spreadsheets), and
  - it avoids training on future data (time-aware splits), and
  - it outputs reproducible predictions suitable for confidence ranking.
- **Targets (allowed approaches):**
  - Predict `home_score` & `away_score` directly **or**
  - Predict `margin` and `total` and derive scores (preferred if it improves realism/consistency).
- **Evaluation (project-relevant metrics):**
  - For pick’em / win probability: optimize and report **log loss** or **Brier score** for win probs.
  - For “realistic scores”: report **MAE** on `margin` and `total` (and optionally team scores).
  - For confidence pools: backtest “confidence points” using predicted win probs (ranking by |p-0.5| or implied margin) and report expected/actual points over seasons.
- **Confidence Pool Logic (separation of concerns):**
  - Determine predicted winner via predicted scores or win probability.
  - Compute confidence strength via win probability (preferred) or predicted margin.
  - Rank games by descending confidence; keep this separate from the model training code.

## Dev Workflows (How to Run Things)
- **Refresh Data:**  
  `python -m nfl_predictor.data_collection_polars`
- **Testing:**  
  `pytest`
  - `tests/test_polars_utils.py`
  - `tests/test_data_collection_polars.py`
- **Validation Scripts:**
  - `scripts/validate_offline.py`
  - `scripts/validate_live.py`
- **Train & Predict (Legacy):**  
  `python -m nfl_predictor.ml_model` (legacy/outdated; may be replaced)

## Logging & Coding Style
- **Logging:** Use the project logger (`from nfl_predictor.utils.logger import log`). Avoid `print`.
- **Formatting & Linting:**
  - Black (line length 100)
  - isort (Black profile)
  - flake8 (E402 ignored)
- **Type Hints:** Use type annotations consistently.
- **Performance:** Prefer Polars expressions over Python loops.
- **Avoid pandas in Polars pipeline.** pandas is acceptable only in ML modules where required by libraries.

## Choosing APIs & Libraries
- **Polars First:** All data transformation and feature engineering should use Polars.
- **nflreadpy:** Primary source for schedules, team stats, and Elo.
- **Machine Learning:** scikit-learn and XGBoost are preferred.
  - pandas/numpy usage is acceptable within modeling modules.
- **No Magic Numbers:** Avoid hardcoded assumptions about teams, games, or seasons.

## GPT-5.2 Codex Guidance
- Use existing project utilities whenever possible.
- Generate idiomatic code consistent with this repository.
- Prioritize clarity, correctness, and maintainability over brevity.
- Assume no access to external artifacts beyond this repository and its data outputs.