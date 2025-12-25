# nfl-predictor

## ML Model Usage

### Quickstart (train + predict)

Train a margin/total model on all available seasons and generate predictions for the weekly file:

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

### Model kinds

`--model-kind score`

- Predicts `away_score` and `home_score` directly.
- Uses a season holdout for evaluation (if configured).

`--model-kind margin_total` (default)

- Predicts margin and total, then derives scores.
- Optional win-probability calibration (Platt or isotonic).
- Reports confidence-pool metrics on the holdout (if configured).

`--model-kind blend`

- Trains two models: team-feature only and market-only.
- Learns a blending layer on the calibration seasons.
- Requires at least one calibration season.

### Data and feature selection

The model expects `data/completed_games_ml.csv` by default. Feature selection is rule-based:

- All columns between `away_rest` and `home_moneyline` (inclusive) are used as features.
- Columns before `away_rest` are treated as metadata and dropped.
- `away_score` and `home_score` are the target columns.
- `--exclude-market` removes spread/total/moneyline features.

### Splits: train, calibration, holdout

Splits are time-aware by season.

- `--holdout-seasons` keeps the most recent seasons for evaluation only.
- `--calibration-seasons` reserves seasons just before the holdout for calibration/blending.
- `--calibration-weeks` reserves the most recent weeks from the latest season for calibration.
- Use `--min-season` / `--max-season` to bound the dataset.

Example: hold out 2025 for evaluation, calibrate on 2024:

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --holdout-seasons 1 \
  --calibration-seasons 1
```

Example: blended model with in-season calibration from the latest four weeks:

```bash
python -m nfl_predictor.ml_model \
  --model-kind blend \
  --holdout-seasons 0 \
  --calibration-seasons 0 \
  --calibration-weeks 4
```

### Optuna hyperparameter tuning

Optuna is a hyperparameter search library. Use it to optimize for your pool objective.

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --tune \
  --tune-timeout 600 \
  --tune-metric expected_points
```

Available tuning objectives:

- `margin_mae`, `total_mae`, `combined_mae`
- `winner_accuracy`, `brier`, `expected_points`

### Monitoring and checkpoints

Optuna prints a line per trial. For long runs, persist the study and write the best params
to a JSON file as training progresses:

```bash
python -m nfl_predictor.ml_model \
  --model-kind blend \
  --tune \
  --tune-timeout 14400 \
  --tune-storage sqlite:///optuna.db \
  --tune-study-name blend_gpu_4h \
  --tune-best-params-out models/best_params.json
```

You can save a trained model for reuse with `--model-out` and load it later with `--model-in`.

### Persist Optuna across runs

To keep improving across runs, use a persistent study:

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --tune \
  --tune-timeout 600 \
  --tune-storage sqlite:///optuna.db \
  --tune-study-name margin_total_all_2025
```

### Model checkpoints

Save a trained model and reuse it later without retraining:

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --model-out models/margin_total.joblib
```

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --model-in models/margin_total.joblib \
  --predict-path data/predict/week_17_games_to_predict.csv
```

### GPU usage

XGBoost uses the GPU only when built with CUDA support and when you set:

```bash
--xgb-device cuda --xgb-tree-method hist
```

If your XGBoost build is CPU-only, `cuda` will fail and the CLI falls back to CPU `hist`.
Passing `--xgb-tree-method gpu_hist` is also accepted and will be coerced to `hist` + `device=cuda`
for XGBoost 3.x compatibility.

For a Python venv without conda, GPU support requires building XGBoost from source with CUDA
enabled and installing it into the venv.

To ensure GPU support:

- Install NVIDIA drivers and a compatible CUDA toolkit.
- Prefer a GPU-enabled package (e.g., conda-forge `xgboost` with CUDA), or build XGBoost
  from source with CUDA enabled.
- Verify by running a small training run with `--xgb-tree-method gpu_hist`.

XGBoost does not blend CPU and GPU in a single training run; you choose one via `tree_method`.

## Testing

Run the test suite with:

```bash
pytest
```

## Validation

Offline validation against the latest dataset:

```bash
python scripts/validate_offline.py
```

Live validation against the latest schedule (may require network access):

```bash
python scripts/validate_live.py
```
