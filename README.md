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
pytest
```

Run tests with coverage:

```bash
pytest --cov=nfl_predictor --cov-report=term-missing
```

## Data collection (Polars + nflreadpy)

The authoritative data build pipeline is:

```bash
python -m nfl_predictor.data_collection_polars
```

This writes datasets under `data/` (paths are defined in `nfl_predictor/constants.py`).

Typical outputs:

- `data/all_data.csv` and `data/all_data_ml.csv`
- `data/completed_games.csv` and `data/completed_games_ml.csv`
- `data/predict/week_XX_games_to_predict.csv`

Note: `*_ml.csv` files include model-ready engineered features.

Historical weeks load from cached artifacts where available; the current week may require network
access for external sources handled by `nfl_predictor/utils/scraping_utils.py`.

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

- `--holdout-seasons` reserves the most recent seasons for evaluation only.
- `--calibration-seasons` reserves seasons just before the holdout for calibration/blending.
- `--calibration-weeks` reserves the most recent weeks from the latest season for calibration.

Example: hold out the most recent season for evaluation and calibrate on the season before it:

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --holdout-seasons 1 \
  --calibration-seasons 1

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

## Planned feature roadmap

All new features are defined so they apply to **every matchup**, not only end-of-season games.

Planned feature areas (see `TODO.md`):

- season-to-date record features (overall, division, conference W-L-T)
- divisional rivalry indicator
- injury/health burden features (team-week and positional)
- lookahead/trap indicators (next-week opponent strength + rest/travel context)
- motivational asymmetry features (playoff leverage and clinch/elimination context)
- blocked/time-series cross-validation for tuning stability
- missing data policies for partial-coverage sources (consistent schema across seasons)

## Development notes

- ETL and feature engineering run in Polars.
- All NFLverse data is pulled via `nflreadpy`.
- Logging uses the project logger; avoid `print`.
- Formatting is enforced via Black/isort/flake8.

## Safety and claims

This project outputs statistical forecasts and backtest metrics. It does not guarantee accuracy,
profit, or betting success.
