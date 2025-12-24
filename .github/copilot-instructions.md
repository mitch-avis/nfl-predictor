# Copilot instructions for nfl-predictor

## Project shape (big picture)
- Primary pipeline is Polars + `nflreadpy`: schedule + team_stats + ELO + TeamRankings → ML-ready dataset.
- Orchestrator: [nfl_predictor/data_collection_polars.py](../nfl_predictor/data_collection_polars.py) (run as a module).
- Core transforms/loaders live in [nfl_predictor/utils/polars_utils.py](../nfl_predictor/utils/polars_utils.py).
- Game-specific enrichments (QB fill, SurvivorGrid spreads, moneyline fill) live in [nfl_predictor/utils/game_utils.py](../nfl_predictor/utils/game_utils.py).
- Scraping/caching for TeamRankings + SurvivorGrid is in [nfl_predictor/utils/scraping_utils.py](../nfl_predictor/utils/scraping_utils.py).

## Data inputs/outputs (repo conventions)
- All data lives under `data/` (`constants.DATA_PATH` in [nfl_predictor/constants.py](../nfl_predictor/constants.py)).
- Running `python -m nfl_predictor.data_collection_polars` writes:
  - `data/all_data_ml.csv` (ML version with diff columns)
  - `data/all_data.csv` (no diff columns)
  - `data/completed_games_ml.csv`, `data/completed_games.csv`
  - `data/predict/week_XX_games_to_predict.csv`
- TeamRankings cache files are per-season/per-week:
  - `data/<season>/<season>_week_XX_team_rankings.csv` and consolidated `data/<season>/<season>_team_rankings.csv`.

## Column/schema rules (don’t guess)
- Treat [nfl_predictor/constants.py](../nfl_predictor/constants.py) as the source of truth for:
  - nflreadpy renames: `NFLREADPY_SCHEDULE_COLUMNS`, `NFLREADPY_SCHEDULE_RENAME`
  - output ordering: `FIRST_COLUMNS`, `LINES_COLUMNS`, `RESULT_COLUMNS`
  - Polars dataset schema: `POLARS_METADATA_COLUMNS`, `POLARS_LINES_COLUMNS`, `POLARS_RESULT_COLUMNS`, plus stats lists.
- Team abbreviations must be normalized via `constants.ALIAS_TO_CANONICAL` / `normalize_team_column()` (used throughout `polars_utils`).

## Season/week edge cases (important)
- Week counts differ pre/post 2021: use `constants.get_regular_season_weeks(season)`.
- Week 1 (and other “no prior games” situations) uses previous-season fallback with regression-to-mean; tests assert this behavior in [tests/test_data_collection_polars.py](../tests/test_data_collection_polars.py).
- TeamRankings for current/future weeks may scrape live; historical weeks should load from cached CSV when present.

## Dev workflows (what to run)
- Tests: `pytest` (see [README.md](../README.md)).
- Offline dataset validation (local-only): `python scripts/validate_offline.py`.
- Live validation (may require network via nflreadpy): `python scripts/validate_live.py`.

## Logging & style
- Use the shared logger: `from nfl_predictor.utils.logger import log` (configured in [nfl_predictor/utils/logger.py](../nfl_predictor/utils/logger.py)).
- Formatting/linting conventions:
  - Black line length 100 (see `pyproject.toml`)
  - isort profile black, line length 100; flake8 ignores `E402` (see `setup.cfg`).

## Choosing APIs in this repo
- Prefer Polars code paths (`data_collection_polars.py`, `utils/polars_utils.py`) for data collection/feature engineering.
- Some older/legacy utilities use pandas (e.g. [nfl_predictor/ml_model.py](../nfl_predictor/ml_model.py), [nfl_predictor/utils/csv_utils.py](../nfl_predictor/utils/csv_utils.py)); avoid mixing pandas into the Polars pipeline unless a module already expects pandas.
