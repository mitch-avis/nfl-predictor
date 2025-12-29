# ML Overhaul Plan Notes

## Current entrypoints

- `python -m nfl_predictor.data_collection_polars`: collects data via Polars/nflreadpy; writes
  `data/all_data_ml.csv`, `data/completed_games_ml.csv`, and
  `data/predict/week_XX_games_to_predict.csv` (plus non-ML variants).
- `python -m nfl_predictor.ml_model`: trains XGBoost score or margin/total models, optional
  Optuna tuning; can save/load a `joblib` checkpoint and generate weekly predictions.
- `scripts/backtest_predictions.py`: loads a saved model checkpoint, scores a dataset (default
  `data/completed_games_ml.csv`), and writes predictions plus weekly/season summaries under
  `data/backtest/`.
- `scripts/validate_offline.py` / `scripts/validate_live.py`: data validation against
  `data/all_data.csv`.

## Files likely to change during the overhaul

- `nfl_predictor/constants.py`: column/schema source of truth; changes here must be reflected
  across ETL + modeling.
- `README.md`: document the canonical commands and artifact outputs.

## Current artifacts

- Model checkpoint only: `joblib` saved via `--model-out` in `nfl_predictor/ml_model.py`.
- Backtest outputs: CSV/JSON summaries in `data/backtest/` (not run-scoped).
- Missing metadata: no dataset hash, library versions, training config, or season/week ranges
  saved alongside the model.

## Margin/total + calibration

- Margin/total targets, score derivation, and win-prob calibration live in
  `nfl_predictor/ml_model.py` (see `_prepare_margin_total_targets*`,
  `_derive_scores_from_margin_total`, `_fit_win_prob_calibrator`, and
  `_predict_home_win_prob`).

## Dataset producers

- `data/all_data_ml.csv` and `data/completed_games_ml.csv` are produced by
  `python -m nfl_predictor.data_collection_polars`.
- Prediction inputs are written to `data/predict/week_XX_games_to_predict.csv` by the same
  module.
