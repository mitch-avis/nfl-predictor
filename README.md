# nfl-predictor

Predict NFL game outcomes and scores using a **margin/total** ML pipeline + walk-forward evaluation,
with optional market anchoring and probability calibration.

This repo is geared toward:

- **Pick 'Em** and **Confidence Pools** (primary)
- sports-betting research (secondary; no profit claims)

---

## Quickstart

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

### 2) Build/refresh datasets (Polars + nflreadpy)

```bash
python -m nfl_predictor.data_collection_polars
```

Outputs are written under `data/` (CSV). Primary files:

- `data/all_data.csv` and `data/all_data_ml.csv`
- `data/completed_games.csv` and `data/completed_games_ml.csv`
- `data/predict/week_##_games_to_predict.csv`

> Note: `*_ml.csv` files include model-ready engineered features (including differential features).

### 3) Train + predict (simple)

```bash
python -m nfl_predictor.ml_model \
  --data-path data/completed_games_ml.csv \
  --model-kind margin_total \
  --holdout-seasons 1 \
  --win-prob-calibration platt \
  --predict-path data/predict/week_17_games_to_predict.csv
```

This writes:

- a predictions CSV alongside the input predict file
- a run folder under `models/` containing a model artifact + `metadata.json` + (if evaluated) `metrics_report.json`

---

## Modeling approach

### Margin/Total is the canonical target

We predict:

- `margin = home_score - away_score`
- `total  = home_score + away_score`

Then derive scores:

- `home = (total + margin) / 2`
- `away = (total - margin) / 2`

This keeps predictions internally consistent and makes it easy to derive win probabilities from the
margin distribution.

### Win probability calibration

Raw margin → win probability mappings tend to be miscalibrated. This repo supports:

- **Platt scaling** (logistic regression)
- **Isotonic regression**

Calibration must be **time-aware** (fit only on past data relative to the evaluation window).

### Market anchoring (recommended)

If spreads/totals/moneylines are present, you can train on **residuals vs market** (anchor) so the
model learns “how to beat the line” rather than re-learning what the market already priced.

---

## Evaluation: avoid in-sample overfitting traps

The project includes realistic evaluation modes:

1) **Holdout seasons** (`--holdout-seasons N`): reserve the last N seasons for evaluation.
2) **Walk-forward backtest** (`scripts/walk_forward_backtest.py`): week-by-week rolling-origin
   evaluation (best realism).
3) (Planned) **Blocked CV** across seasons/weeks for tuning stability (see TODO).

If you see “great” performance on the exact data the model trained on, that is **not** evidence the
model is good — it’s evidence the model can memorize patterns. Prefer walk-forward metrics.

---

## Core scripts

- `python -m nfl_predictor.data_collection_polars`  
  Build datasets via nflreadpy + Polars

- `python -m nfl_predictor.ml_model`  
  Train/evaluate/predict (main entrypoint)

- `python scripts/walk_forward_backtest.py`  
  Walk-forward evaluation + reports

- `python scripts/backtest_predictions.py`  
  Evaluate a saved model on historical rows

- `python scripts/leakage_audit.py`  
  Detect obvious feature leakage patterns

---

## Artifacts

Runs write to `models/<run_id>/`:

- `model.joblib` (or similar)
- `metadata.json` (required)
- `metrics_report.json` (required for backtests)

Metadata includes:

- timestamp
- dataset fingerprint/hash
- key package versions
- training config / CLI args
- feature list
- tuning / early stopping info (when used)

---

## Feature roadmap (high-level)

This repo already uses:

- team performance stats (season-to-date)
- rest/travel context (where available)
- TeamRankings ratings
- optional betting markets

Planned features to apply **to all matchups** (not only late-season):

- **Team health burden** (injury report aggregation → team-week features)
- **Divisional rivalry flag** (same-division matchups)
- **Lookahead / trap indicators** (next-week opponent strength + travel/rest context)
- **Motivational asymmetry** (playoff leverage and clinch/elimination context, late-season-heavy but
  still generic)

See `TODO.md` for the actionable plan.

---

## Notes on responsible use

This project produces statistical forecasts. It does not guarantee accuracy, profit, or betting
success. Use predictions as one input among many, and treat performance metrics as probabilistic,
not deterministic.
