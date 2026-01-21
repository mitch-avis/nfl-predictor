# nfl-predictor

NFL game score and outcome prediction with a margin/total ML model, calibrated win probabilities,
and tooling for Pick 'Em and Confidence Pools.

This repo is geared toward:

- **Pick 'Em** and **Confidence Pools** (primary)
- sports-betting research (secondary; no profit claims)

## Table of Contents

- [nfl-predictor](#nfl-predictor)
  - [Table of Contents](#table-of-contents)
  - [What this project produces](#what-this-project-produces)
  - [Setup](#setup)
    - [Python environment](#python-environment)
      - [Recommended: uv + pinned requirements](#recommended-uv--pinned-requirements)
      - [Alternative: venv + pip](#alternative-venv--pip)
    - [Run tests](#run-tests)
  - [Data collection (Polars + nflreadpy)](#data-collection-polars--nflreadpy)
  - [Data sources + missing data](#data-sources--missing-data)
  - [Training + prediction](#training--prediction)
    - [Quickstart (train + predict)](#quickstart-train--predict)
    - [Splits: train, calibration, holdout](#splits-train-calibration-holdout)
  - [Modeling approach](#modeling-approach)
    - [Margin/Total targets (canonical)](#margintotal-targets-canonical)
    - [Win probability calibration](#win-probability-calibration)
    - [Market integration (optional, recommended)](#market-integration-optional-recommended)
  - [Backtesting](#backtesting)
  - [Weekly pipeline](#weekly-pipeline)
  - [Scripts](#scripts)
  - [Validation](#validation)
  - [Leakage audit](#leakage-audit)
  - [Artifacts](#artifacts)
  - [Confidence pool rules (implemented)](#confidence-pool-rules-implemented)
  - [Score rounding / realism (optional)](#score-rounding--realism-optional)
  - [Repository layout notes](#repository-layout-notes)
  - [Implemented feature areas](#implemented-feature-areas)
  - [Open work](#open-work)
  - [Development notes](#development-notes)
  - [Safety and claims](#safety-and-claims)

## What this project produces

- **Predicted margin and total** for each game (canonical targets).
- **Predicted home/away scores** derived from margin/total.
- **Calibrated win probabilities** for confidence ranking.
- **Weekly confidence ranks** (1..N unique values) suitable for pool submission.
- **Backtest summaries** for pool points and probability calibration.

## Setup

### Python environment

This project targets **Python 3.13+** (see `pyproject.toml`).

#### Recommended: uv + pinned requirements

This repo uses pinned requirements for reproducible runs:

- `requirements.in` / `requirements-dev.in` are the sources of truth
- `requirements.txt` / `requirements-dev.txt` are generated via `uv pip compile`

```bash
uv venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

uv pip sync requirements.txt requirements-dev.txt
uv pip install -e . --no-deps
```

To regenerate pinned requirements:

```bash
uv pip compile requirements.in -o requirements.txt
uv pip compile requirements-dev.in -o requirements-dev.txt
```

#### Alternative: venv + pip

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps
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

Note: the `data/` directory is gitignored by default; generate it via the data collection step above.
Note: `*_ml.csv` files include model-ready engineered features.

Historical seasons load from cached artifacts where available. nflreadpy outputs are cached per
season under `data/cache/nflreadpy` (schedule + team stats). Current/future seasons are always
refreshed to keep upcoming games and lines current. Use `--min-season`/`--max-season` to override
the default season window (defaults to `constants.MIN_SEASON` through the current NFL season).

TeamRankings data is cached under `data/<season>/` as week-level CSVs; enable debug logging to see
cache hits. Use `--timing` to log per-step runtimes and `--debug-logs` for detailed ETL diagnostics.
Use `--refresh-nflreadpy` to force refresh nflreadpy data even when cache exists.

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
Optional recency weighting is available via `--recency-half-life-weeks` or
`--recency-half-life-seasons` (use only one) to apply exponential decay to training and
calibration samples.

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

Win probabilities are derived from the predicted margin, then optionally calibrated using a
time-aware calibration split (seasons and/or weeks immediately preceding the holdout window).

`--win-prob-calibration` options:

- `none`: deterministic Normal-CDF mapping using `constants.SCORE_DIFF_STD_DEV`.
- `elo`: deterministic Elo-style logistic mapping (no fitting).
- `platt`: Platt scaling via logistic regression fit on the calibration split.
- `isotonic`: isotonic regression fit on the calibration split.
- `auto`: use isotonic when calibration data is large enough; otherwise fall back to Platt.
- `logistic`: alias for `platt`.

Calibration is time-aware: it fits only on historical data relative to the evaluation window.

### Market integration (optional, recommended)

If spreads/totals/moneylines are present, you can:

- use market-derived features (`--market-transform`)
- train on residuals vs market baselines (`--market-anchor`) so the model learns deviations rather
  than re-learning what the market already priced

Win probability can also be blended or clamped vs market-implied home win probability via
`--market-prob-blend` / `--market-prob-clamp` (alias: `--market-prob-weight`). Use
`--market-prob-source raw|novig` to choose implied-prob handling and
`--market-prob-blend-method prob|logit` to blend in probability or log-odds space.

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

This is the **canonical evaluation protocol** for model selection. By default it evaluates the
last N seasons (regular season only) with time-aware calibration from the last K weeks of each
eval season. Use `--include-postseason` if you want postseason folds included.
Optional recency weighting is available via `--recency-half-life-weeks` or
`--recency-half-life-seasons` (use only one).
GPU acceleration is optional: add `--xgb-tree-method hist --xgb-device cuda`.

Trend/season-phase ablation (drop trend + season-phase features while keeping everything else
identical) is available via `--disable-trend-features`. Example 2x2 comparison matrix:

```bash
.venv/bin/python scripts/walk_forward_backtest.py --calibration platt
.venv/bin/python scripts/walk_forward_backtest.py --calibration platt --recency-half-life-seasons 2
.venv/bin/python scripts/walk_forward_backtest.py --calibration platt --disable-trend-features
.venv/bin/python scripts/walk_forward_backtest.py --calibration platt --disable-trend-features \
  --recency-half-life-seasons 2
```

Recent ablation example (2003-2025 seasons, include postseason, calibration=platt,
recency half-life seasons=2):

```text
Setting                         Brier    LogLoss  MarginMAE  TotalMAE  ActualPts
Trends off, recency off         0.2372   0.7772   10.0987    10.0759   210.55
Trends on, recency off          0.2325   0.7525   10.0159    10.1254   213.35
Trends off, recency on          0.2827   1.9775   10.2003    10.0706   211.50
Trends on, recency on           0.2804   1.9525   10.0691    10.0670   213.55
```

Interpretation:

- Trend features improve probability metrics (Brier/log loss) and margin MAE, with a small
  tradeoff in total MAE.
- Recency weighting (half-life seasons=2) hurts probability metrics in this run; keep it off
  unless a future ablation shows improvement.

Evaluation rule:
“Model selection is based on time-aware walk-forward evaluation; random CV is not authoritative.”

## Weekly pipeline

Authoritative weekly workflow (run in order):

1) Refresh data (ETL):

```bash
.venv/bin/python -m nfl_predictor.data_collection
```

1) Canonical evaluation + model selection (walk-forward):

```bash
.venv/bin/python scripts/walk_forward_backtest.py --help
```

1) Train + predict for the upcoming week (writes predictions + artifacts):

```bash
.venv/bin/python -m nfl_predictor.ml_model --help
```

1) Power rankings + projected standings:

```bash
.venv/bin/python scripts/power_rankings.py --help
```

One-command weekly orchestration (refresh + selection + train + reports):

```bash
.venv/bin/python scripts/weekly_run.py --help
```

Outputs and conventions:

- Run artifacts (model/metrics/metadata/feature importance) land under `models/<run_id>/` by default.
- Weekly prediction outputs live next to the input prediction file (e.g., `data/predict/`).
- `metadata.json` includes dataset fingerprint, tuned params, and Optuna summary when tuning runs.
- Power rankings outputs:
  - `power_rankings_season_XXXX_week_YY.csv`
  - `projected_standings_season_XXXX_week_YY.csv`
  - `projected_division_standings_season_XXXX_week_YY.csv`

Model selection hierarchy (default):

- Primary: probability quality (Brier, log loss, reliability).
- Secondary: confidence pool expected points and stability.
- Tertiary: margin/total MAE (plus market-relative residual MAE when anchoring).

Metrics reports include a summary table (with metric priority + direction), plus optional
diagnostics such as season win totals (expected vs actual) and calibration drift by season/week.

## Scripts

Repo utilities under `scripts/`:

- `scripts/betting_pipeline.py`: end-to-end orchestration (walk-forward compare -> resumable
  tuning -> final train -> weekly predictions + betting_report.csv). See `--help`.
- `scripts/objective_compare_models.py`: objective walk-forward comparison of two saved models
  by retraining per fold under identical splits.
- `scripts/betting_report_excel.py`: generate an Excel betting template/report.
- `scripts/golden_command.py`: convenience orchestration for walk-forward + training + prediction
  and artifact stamping.
- `scripts/shap_analysis.py`: optional SHAP feature attribution for a saved model (requires `shap`).
- `scripts/wf_compare.py`: sweep calibration + market-prob post-processing variants and summarize
  walk-forward metrics.
- `scripts/weekly_run.py`: weekly orchestration (refresh -> wf compare -> train -> predictions +
  reports), resumable with optional JSON/YAML config.
- `scripts/backtest_predictions.py`: run a backtest using a saved model artifact.

`wf_compare` examples:

```bash
.venv/bin/python scripts/wf_compare.py \
  --eval-last-n-seasons 3 \
  --market-mode hybrid \
  --market-prob-source raw \
  --market-prob-blend-method prob
```

```bash
.venv/bin/python scripts/wf_compare.py \
  --eval-last-n-seasons 3 \
  --market-mode all \
  --market-prob-source both \
  --market-prob-blend-method both
```

Uncertainty-aware comparison:

```bash
.venv/bin/python scripts/wf_compare.py \
  --eval-last-n-seasons 3 \
  --win-prob-uncertainty both
```

`weekly_run` config example (JSON):

```json
{
  "wf_eval_last_n_seasons": 3,
  "wf_market_mode": "hybrid",
  "wf_market_prob_source": "raw",
  "wf_market_prob_blend_method": "prob",
  "predict_path": "data/predict/week_03_games_to_predict.csv"
}
```

Run it with:

```bash
.venv/bin/python scripts/weekly_run.py --config path/to/weekly_run.json
```

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
- `feature_importance.json` includes XGBoost gain/weight importance per model head.
- Metadata includes timestamp, dataset fingerprint/hash, key package versions, training config/CLI
  args, feature list, and tuning/early-stopping info (when used).
  `models/` and `optuna.db` are gitignored by default, so keep run artifacts local unless you copy
  them elsewhere.

## Confidence pool rules (implemented)

- Each week assigns unique confidence values `1..N` to each picked winner.
- Max weekly points: `N*(N+1)/2`.
- Realized points: `sum(confidence_value * 1[pick_correct])`.
- Ties count as incorrect.

## Score rounding / realism (optional)

When generating predictions, you can optionally post-process **display scores** without changing
training targets, win probabilities, or pool ranking logic:

- `--score-rounding none|int|half|nfl`

Use `nfl` to snap to common NFL score patterns for reporting.

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

Active tasks (milestones + guardrails) are tracked in `TODO.md`. Completed milestones and past
work live in `ARCHIVE.md`.

## Development notes

- ETL and feature engineering run in Polars.
- All NFLverse data is pulled via `nflreadpy`.
- Logging uses the project logger; avoid `print`.
- Formatting is enforced via Black (line length 100).
- Linting and import sorting are enforced via Ruff (includes isort rules).
- Dev tooling (Black/Ruff/pytest/pytest-cov) is installed via `requirements-dev.txt`.

Common local checks:

```bash
ruff check .
black --check .
```

## Safety and claims

This project outputs statistical forecasts and backtest metrics. It does not guarantee accuracy,
profit, or betting success.
