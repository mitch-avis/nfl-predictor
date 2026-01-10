# nfl-predictor

NFL game score and outcome prediction with a margin/total ML model, calibrated win probabilities,
and tooling for Pick 'Em and Confidence Pools.

This repo is geared toward:

- **Pick 'Em** and **Confidence Pools** (primary)
- sports-betting research (secondary; no profit claims)

## What this project produces

- **Predicted margin and total** for each game (canonical targets).
- **Predicted home/away scores** derived from margin/total.
- **Calibrated win probabilities** for confidence ranking.
- **Weekly confidence ranks** (1..N unique values) suitable for pool submission.
- **Backtest summaries** for pool points and probability calibration.

## Setup

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run tests

```bash
python -m pytest
```

By default, pytest runs with coverage enabled (configured in `pyproject.toml`). To disable
coverage for a quick local run:

```bash
python -m pytest --no-cov
```

To run explicitly with coverage (same behavior as the default config):

```bash
python -m pytest --cov=nfl_predictor --cov-report=term-missing
```

To enforce a minimum coverage threshold locally, add `--cov-fail-under`:

```bash
python -m pytest --cov-fail-under=80
```

## Data collection (Polars + nflreadpy)

The authoritative data build pipeline is:

```bash
python -m nfl_predictor.data_collection
```

The default season range is controlled by `constants.MIN_SEASON` (currently 2006).

This writes datasets under `data/` (paths are defined in `nfl_predictor/constants.py`).

Typical outputs:

- `data/all_data.csv` and `data/all_data_ml.csv`
- `data/completed_games.csv` and `data/completed_games_ml.csv`
- `data/predict/week_XX_games_to_predict.csv`

Note: `*_ml.csv` files include model-ready engineered features.

Historical weeks load from cached artifacts where available; the current week may require network
access for external sources handled by `nfl_predictor/utils/scraping_utils.py`.

## Data sources + missing data

This project is designed to keep an invariant output schema across seasons, even when some
sources are missing historically.

Primary sources:

- `nflreadpy` (NFLverse): schedules, results, and team-level stats.
- Local cached CSVs under `data/` for Elo/market data when present.
- TeamRankings web scrape for select rates not available in NFLverse (see ETL logs).

Missing data policy (high level):

- ETL emits all expected columns; missing sources become nulls and/or defined defaults.
- The ML pipeline is expected to tolerate nulls (imputation and/or model-native missing handling).

## Training + prediction

The primary entrypoint is:

```bash
python -m nfl_predictor.ml_model --help
```

### Quickstart (train + predict)

Train a margin/total model and generate predictions for a weekly input file:

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --holdout-seasons 0 \
  --calibration-seasons 0 \
  --win-prob-calibration none \
  --tune \
  --tune-timeout 600 \
  --tune-metric expected_points \
  --xgb-tree-method hist \
  --predict-path data/predict/week_17_games_to_predict.csv
```

This prints a weekly summary and writes `*_predictions.csv` next to the input file.

If you want a minimal run without tuning (and with explicit input paths):

```bash
python -m nfl_predictor.ml_model \
  --data-path data/completed_games_ml.csv \
  --model-kind margin_total \
  --holdout-seasons 1 \
  --calibration-seasons 1 \
  --win-prob-calibration isotonic \
  --predict-path data/predict/week_17_games_to_predict.csv
```

### Splits: train, calibration, holdout

Splits are time-aware by season and (optionally) by in-season week.

By default, training/evaluation uses **regular season** games only when the input data includes a
`game_type` column (i.e., postseason rows are filtered out). You can still generate predictions
for playoff games as long as the feature row exists.

To include postseason games in training, pass `--include-postseason`. To emphasize postseason
games, also set `--postseason-weight` (e.g., `--postseason-weight 1.5`).

- `--holdout-seasons` reserves the most recent seasons for evaluation only.
- `--calibration-seasons` reserves seasons just before the holdout for calibration/blending.
- `--calibration-weeks` reserves the most recent weeks from the latest season for calibration.

Example: hold out the most recent season for evaluation and calibrate on the season before it:

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --holdout-seasons 1 \
  --calibration-seasons 1
```

## Modeling approach

### Margin/Total targets (canonical)

We predict:

- `margin = home_score - away_score`
- `total  = home_score + away_score`

Then derive scores:

- `home = (total + margin) / 2`
- `away = (total - margin) / 2`

This keeps score predictions internally consistent and makes win probability derivation
straightforward.

### Win probability calibration

Raw margin -> win probability mappings tend to be miscalibrated. This repo supports:

- **Platt scaling** (logistic regression)
- **Isotonic regression**

Calibration is time-aware: it fits only on historical data relative to the evaluation window.

### Market integration (optional, recommended)

If spreads/totals/moneylines are present, you can:

- use market-derived features (`--market-transform`)
- train on residuals vs market baselines (`--market-anchor`) so the model learns deviations rather
  than re-learning what the market already priced

Win probability can also be blended or clamped vs market implied probabilities via
`--market-prob-blend` / `--market-prob-clamp`.

## Backtesting

Backtest and produce weekly confidence ranks and summary metrics:

```bash
python scripts/backtest_predictions.py \
  --model-in models/your_model.joblib \
  --model-kind margin_total \
  --data-path data/completed_games_ml.csv \
  --output-dir data/backtest
```

For the most realistic evaluation, use walk-forward (rolling-origin) backtesting:

```bash
python scripts/walk_forward_backtest.py --help
```

## Scripts

Repo utilities under `scripts/`:

- `scripts/betting_pipeline.py`: end-to-end orchestration (walk-forward compare -> resumable
  tuning -> final train -> weekly predictions + betting_report.csv). See `--help`.
- `scripts/objective_compare_models.py`: objective walk-forward comparison of two saved models
  by retraining per fold under identical splits.
- `scripts/betting_report_excel.py`: generate an Excel betting template/report.
- `scripts/golden_command.py`: convenience orchestration for walk-forward + training + prediction
  and artifact stamping.
- `scripts/wf_compare.py`: sweep calibration + market-prob post-processing variants and summarize
  walk-forward metrics.
- `scripts/backtest_predictions.py`: run a backtest using a saved model artifact.

GPU note (XGBoost 2.x): prefer `--xgb-tree-method hist --xgb-device cuda`.

If you see great performance on the exact data a model trained on, that is not evidence the model
generalizes. Prefer holdout and walk-forward metrics.

## Validation

Offline validation:

```bash
python scripts/validate_offline.py
```

Live validation (may require network access):

```bash
python scripts/validate_live.py
```

## Leakage audit

To detect obvious feature leakage patterns:

```bash
python scripts/leakage_audit.py
```

## Artifacts

Training/backtests can write a run directory containing reproducible artifacts.

- Use `--run-dir` to write `model.joblib`, `metadata.json`, and (when evaluated)
  `metrics_report.json`.
- Metadata includes timestamp, dataset fingerprint/hash, key package versions, training config/CLI
  args, feature list, and tuning/early-stopping info (when used).

## Confidence pool rules (implemented)

- Each week assigns unique confidence values `1..N` to each picked winner.
- Max weekly points: `N*(N+1)/2`.
- Realized points: `sum(confidence_value * 1[pick_correct])`.
- Ties count as incorrect.

## Repository layout notes

Some modules are split to keep files under lint `max-module-lines` limits while preserving legacy
import paths.

- Polars ETL helpers live under `nfl_predictor/utils/polars/` with a compatibility facade at
  `nfl_predictor/utils/polars_utils.py`.
- ML implementation lives under `nfl_predictor/ml/` with a compatibility facade at
  `nfl_predictor/ml_model.py`.

## Implemented feature areas

All engineered features are defined so they apply to **every matchup**, not only end-of-season
games.

- Invariant-schema missing-data handling across seasons.
- Season-to-date record features (overall/division/conference).
- Divisional rivalry indicator.
- Lookahead / next-week context features.
- Standings-based motivation proxy features (clinch/elimination proxies).

## Open work

Active tasks are tracked in `TODO.md`.

At the moment, `TODO.md` contains a **guardrails checklist** (time-aware/no-leakage, Polars-first
ETL, reproducible artifacts, and tests/coverage expectations). Completed milestones and past work
live in `ARCHIVE.md`.

## Development notes

- ETL and feature engineering run in Polars.
- All NFLverse data is pulled via `nflreadpy`.
- Logging uses the project logger; avoid `print`.
- Formatting is enforced via Black/isort, and lint is enforced via Ruff.

## Safety and claims

This project outputs statistical forecasts and backtest metrics. It does not guarantee accuracy,
profit, or betting success.
