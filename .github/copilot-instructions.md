# Copilot Instructions for nfl-predictor

## Project Shape (Big Picture)
- **Primary Pipeline:** The project uses **Polars** for data processing, combined with **nflreadpy** to fetch raw data (schedule, team stats, Elo ratings, etc.). The pipeline integrates multiple data sources (schedule results, team statistics, ELO ratings, TeamRankings stats) to produce a rich, ML-ready dataset for predictions.
- **Orchestration Script:** The main data assembly is done in [`nfl_predictor/data_collection_polars.py`](../nfl_predictor/data_collection_polars.py) (run as a module). This orchestrator pulls the data, applies all transformations, and saves output CSVs.
- **Core Data Transforms:** Polars transformations and helper functions live in [`nfl_predictor/utils/polars_utils.py`](../nfl_predictor/utils/polars_utils.py). This includes functions to normalize team names, compute new feature columns (offense/defense metrics, etc.), and any other Polars DataFrame manipulations.
- **Game-Specific Enrichments:** Additional domain-specific calculations (e.g., filling missing QB stats, computing Survivor pool spreads, adding moneyline odds) are in [`nfl_predictor/utils/game_utils.py`](../nfl_predictor/utils/game_utils.py). This is used after core data assembly to enrich the dataset with specialized metrics.
- **External Data Scraping/Caching:** Data from external sites (TeamRankings stats, SurvivorGrid lines, etc.) is handled via [`nfl_predictor/utils/scraping_utils.py`](../nfl_predictor/utils/scraping_utils.py). This module fetches and caches web data so that it can be reused without hitting the network repeatedly.

## Data Inputs/Outputs (Repo Conventions)
- **Data Directory:** All data files reside under the `data/` directory (see `constants.DATA_PATH` in [`nfl_predictor/constants.py`](../nfl_predictor/constants.py)). The pipeline reads from and writes to this location.
- **Key Output Files:**
  - `data/all_data_ml.csv` – Master dataset for machine learning (includes game results with engineered features and target variables, suitable for training models; e.g., with point differentials, totals, etc. as needed).
  - `data/all_data.csv` – Basic combined dataset (similar to above but without ML-specific columns like derived differentials if those are separated).
  - `data/completed_games_ml.csv` and `data/completed_games.csv` – Subsets of data for games already completed (used for training/validation).
  - `data/predict/week_XX_games_to_predict.csv` – Upcoming games (for the current week) with all features prepared, ready for the model to generate predictions.
- **TeamRankings Cache:** TeamRankings data is cached per season/week:
  - Files like `data/<season>/<season>_week_<WW>_team_rankings.csv` (weekly snapshot) and a consolidated `data/<season>/<season>_team_rankings.csv` (season-to-date summary). The pipeline should use these instead of re-scraping if available for historical weeks.
- **Loading/Saving Functions:** Use the utilities in `utils/csv_utils.py` for saving or reading Polars DataFrames to CSV (e.g., `save_df_to_csv` and `read_df_from_csv`) to ensure consistent handling (these wrap Polars I/O with checks).

## Column & Schema Rules (Source of Truth)
- **Constants Module:** Always refer to [`nfl_predictor/constants.py`](../nfl_predictor/constants.py) for authoritative definitions of column names and schema:
  - **nflreadpy Columns:** See `NFLREADPY_SCHEDULE_COLUMNS` and `NFLREADPY_SCHEDULE_RENAME` in constants.py for how raw schedule columns are renamed/selected.
  - **Output Ordering:** Use `FIRST_COLUMNS`, `LINES_COLUMNS`, `RESULT_COLUMNS` from constants.py to order columns in outputs. These define the expected schema for the output CSVs (e.g., first identifying columns, betting lines, result stats).
  - **Polars Dataset Schema:** Constants like `POLARS_METADATA_COLUMNS`, `POLARS_LINES_COLUMNS`, `POLARS_RESULT_COLUMNS` and lists of stat fields dictate the Polars DataFrame structure. Do not hard-code column name strings in code; use these constants to avoid typos and ensure consistency.
- **Team Name Normalization:** Always normalize team abbreviations/names via the mappings in `constants.ALIAS_TO_CANONICAL`. The function `normalize_team_column(df, col)` (found in polars_utils or nfl_utils) should be used wherever team identifiers are ingested (e.g., merging TeamRankings data) to ensure “NE” vs “N.E.” vs “Patriots” all resolve to the same canonical team code. This prevents data misalignment due to naming differences.

## Season/Week Logic & Edge Cases
- **Season Length Variations:** Use `constants.get_regular_season_weeks(season)` to determine the number of regular-season weeks for a given season (e.g., 17 games vs 16 games in older seasons). Don’t hard-code week counts; rely on this utility so that the code can adapt to historical data changes (especially pre-2021 seasons had 17 weeks, post-2021 have 18 weeks).
- **Start of Season Edge Case:** In Week 1 (or whenever a team has no prior games in the dataset), certain features like rolling averages or ELO from previous season need special handling. The pipeline implements a previous-season fallback with regression-to-mean for such cases. (See tests in [`tests/test_data_collection_polars.py`](../tests/test_data_collection_polars.py) for expected behavior.) When writing code that accesses “last week’s” stats, ensure to handle the case where last week doesn’t exist (e.g., use last season’s finale or default values).
- **Real-Time Data vs Historical:** The pipeline should distinguish between historical weeks (where all data including TeamRankings stats are final and can be loaded from CSV) and the current ongoing week:
  - For past weeks, prefer loading from the cached CSV files rather than scraping new data (e.g., use the `data/<season>_week_<WW>_team_rankings.csv` if it exists).
  - For the current week (or if a cache is missing for a past week), the code may fetch live data via `scraping_utils`. Include appropriate logging and caching after scraping to not duplicate work.
- **Future Weeks:** When preparing `week_XX_games_to_predict.csv` for upcoming games, some fields (like actual scores or some TeamRankings stats that require completed games) will be blank or zero. The code should still output the structure with available info (e.g., Vegas odds for future games might be present, team ratings from last week, etc.). Ensure that predictions handle `None`/NaN appropriately (Polars operations should be carefully written to skip or fill missing values if needed for model input).

## Prediction & Modeling Logic
- **Score Prediction Models:** The project uses machine learning to predict game scores. Specifically, `ml_model.py` trains regression models (currently XGBoost) to predict home and away scores separately. Keep this approach unless a unified model is introduced. 
  - When adding features, update the model training code to include them. The list of feature columns (predictors) should come from the dataset’s columns (avoid manual lists that might become outdated; consider deriving it from constants or DataFrame columns minus non-feature columns).
  - Maintain separate models for home and away scores for now, which output `predicted_home_score` and `predicted_away_score`. This ensures predicted totals and point differentials can be derived easily (and keeps consistency with how data is labeled).
- **Feature Engineering for Modeling:** We have introduced new derived features inspired by the Excel workbook to improve predictions:
  - Offense vs Defense mismatches: e.g., `off_vs_def_home = offense_rating_home - defense_rating_away` and vice versa for away. These should be part of the features passed to the model.
  - Combined offensive strength: e.g., `off_sum = offense_rating_home + offense_rating_away`. In the dataset, use a normalized version if appropriate (the Excel uses deviation from mean; in code, we can standardize or leave as is and let the model figure it out).
  - Team rating difference: e.g., `rating_diff = team_rating_home - team_rating_away` (where team_rating could be Elo or a composite). If Elo is in the data, use that; otherwise an offensive/defensive composite can serve.
  - “Fire” metrics: It may not be necessary to include all intermediate Excel metrics explicitly, but ensure the model can capture their effects. For instance, including both `off_vs_def_home` and `off_vs_def_away` allows the model to learn the equivalent of FireHigh/FireLow. You can also include `fire_net = (off_vs_def_home + off_vs_def_away)` and perhaps its absolute or squared value if non-linearity is needed. Simpler: trust the XGBoost model to handle interactions, or include explicit interaction features if using a linear model.
- **Confidence Scoring:** After predicting scores, implement logic to rank games for confidence pools:
  - Determine the predicted winner for each game (compare predicted scores or use predicted point spread).
  - Calculate the predicted point **spread** (difference). Use `abs(spread)` as a confidence metric.
  - Rank games by descending confidence. The highest spread = highest confidence pick.
  - This ranking can be output or logged. If writing an output file (e.g., `week_XX_confidence_picks.csv`), include columns for the game, predicted winner, predicted spread, and a confidence rank or points.
  - Keep this separate from the core model prediction to maintain single responsibility (e.g., have a function `rank_confidence(predict_df)` that takes a DataFrame of games with predictions and adds a confidence rank).
- **Model Training Considerations:** When writing training code:
  - Use proper train/test splits. Do not train on future data. Typically, exclude the current season’s ongoing week from training. The project might use all completed games (`completed_games_ml.csv`) for training and then predict the current week.
  - If using time-series split (training on past seasons to predict current season), ensure code allows filtering by season or week as needed.
  - Log model performance (e.g., mean absolute error on validation set, feature importance). Use `log.info` for outputting metrics and important information.
  - The `ml_utils.py` might contain helper functions for model training or evaluation; use them if available.

## Dev Workflows (How to Run Things)
- **Running Data Pipeline:** To refresh data and features, run `python -m nfl_predictor.data_collection_polars`. This will regenerate the CSVs in `data/` with up-to-date information (requires internet for nflreadpy and any scraping for the current week).
- **Running Predictions:** After data is updated, run `python -m nfl_predictor.ml_model` (or the appropriate function) to train models on `completed_games_ml.csv` and predict the upcoming games. This will output predictions to logs and update the `week_XX_games_to_predict.csv` with new columns like `predicted_home_score` (and potentially write a separate file for confidence rankings if implemented).
- **Testing:** Use `pytest` to run unit tests. Key test modules include:
  - [`tests/test_polars_utils.py`](../tests/test_polars_utils.py) – ensures data transformations (including new feature computations) are correct.
  - [`tests/test_data_collection_polars.py`](../tests/test_data_collection_polars.py) – integration tests for the data pipeline (including edge cases for week 1, season transitions).
  - Ensure new code passes existing tests. If tests need updates due to changed columns or logic, update them accordingly (they serve as specification).
- **Validation Scripts:** The repository provides scripts for validating outputs:
  - `scripts/validate_offline.py`: runs the pipeline and performs some checks on the output data (without needing live data). Use this after modifying data logic.
  - `scripts/validate_live.py`: similar, but may pull live data (useful for a final check against the current week’s info).
  - Run these in a development environment to sanity-check after changes.

## Logging & Coding Style
- **Logger Usage:** Always use the project’s logger for console output, rather than print. Import it via `from nfl_predictor.utils.logger import log`. There are convenience methods: `log.info()`, `log.debug()`, etc. Use `log.debug` for very detailed step logs (which can be turned on in debug mode), and `log.info` for high-level progress messages. This ensures consistency and makes output controllable via log level.
- **Error Handling:** If an expected data file is missing or an external request fails, log a warning or error with context. The pipeline should not silently fail. Use exceptions where appropriate, but catch and log them in the orchestrator so that one game’s data issue doesn’t crash the entire run if possible.
- **Formatting & Linting:** Adhere to the repo’s style configurations:
  - The project uses **Black** with a line length of 100 characters (see `pyproject.toml`). Format code accordingly (usually `black .`).
  - **isort** is used (with Black profile, line length 100) to organize imports. Group imports and avoid unused imports.
  - **flake8** is used for linting. Notably, `E402` (module level import not at top of file) is ignored in `setup.cfg`, but otherwise fix any new lint errors.
  - Type hints: The codebase uses type annotations for functions (see examples in `ml_model.py` and utils). Continue to use type hints for new functions and parameters (this helps with clarity and tools like mypy).
  - Naming: Use descriptive variable and function names. Follow Python conventions (snake_case for functions and variables, PascalCase for classes). For data frames, short names like `df` or `games_df` are fine in context, but avoid single-letter names except in small comprehensions.
- **Performance:** Polars is very fast; prefer Polars operations over Python loops. Use DataFrame expressions and aggregations rather than iterating rows. If you find yourself needing to loop in Python, check Polars docs for a vectorized alternative. This ensures the pipeline runs efficiently even as data grows.
- **Avoid Pandas in Polars Pipeline:** The new pipeline is Polars-centric. Do not introduce pandas usage in data_collection or polars_utils. If some legacy code (e.g. in `ml_model.py` or `csv_utils.py`) uses pandas, that’s fine in that isolated context, but do not mix pandas and Polars in the same data flow. For example, do not convert a Polars DataFrame to pandas just to do a small transformation – implement it in Polars instead, or if it must use pandas, justify it with a comment. The goal is to keep the core pipeline highly performant and consistent.

## Choosing APIs & Libraries
- **Polars First:** When manipulating data tables, use Polars DataFrame methods (`pl.DataFrame` and `lazy` where appropriate). `polars_utils.py` provides some wrappers and might already have common operations (like loading a CSV into Polars with proper dtypes, or adding calculated columns).
- **nflreadpy Data Access:** Use nflreadpy’s provided functions to get NFL schedules, team stats, and Elo. This data likely comes as pandas DataFrames or dictionaries – convert them to Polars DataFrames early in the pipeline. There may be convenience methods in `polars_utils.py` for this.
- **Machine Learning Libraries:** The project uses scikit-learn and XGBoost for modeling. When writing code in `ml_model.py`:
  - Use numpy/pandas for interfacing with scikit-learn if needed (since XGBoost can accept numpy arrays or pandas DataFrames). It’s acceptable that `ml_model.py` uses pandas for the final modeling steps, as it was initially written that way. But ensure the data passed in (from CSV) matches expectations.
  - If adding a new modeling approach (e.g., a linear regression with statsmodels or sklearn), follow similar patterns for splitting data and evaluating. Keep any heavy ML training logic out of the Polars pipeline file to maintain separation of concerns.
- **No Hardcoded Magic Numbers:** Do not hardcode indices or lengths (for example, don’t assume exactly 272 games in a season or 32 teams; use data-driven approaches or constants). The code should be robust to changes (e.g., if NFL adds games, or if we run on a subset of data).
- **GPT-5.2 Codex Specific:** When using these instructions for AI code generation, remember that they are geared to providing context. The assistant should:
  - Use the project’s existing functions whenever possible (don’t reinvent a function that exists; e.g., use `normalize_team_column` instead of writing a new normalization).
  - Write code that is **idiomatic to this project** (e.g., using Polars expressions, using the logger, abiding by constants for column names).
  - Comments in generated code should be clear and project-specific (explain why something is done, if not obvious, especially if it’s a workaround for an edge case).
  - Aim for **readability and maintainability** over short clever code. Other contributors (or tests) will review the output.

By following these guidelines, any new code or refactoring will be consistent with the nfl-predictor project’s style and will integrate smoothly with the existing pipeline and modeling framework. The focus is on maintaining a clear, data-oriented pipeline with Polars, and leveraging ML to improve prediction accuracy for NFL games (particularly for confidence pool strategy).
