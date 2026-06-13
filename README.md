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
      - [Recommended: uv project workflow](#recommended-uv-project-workflow)
      - [Alternative: create the venv manually](#alternative-create-the-venv-manually)
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
  - [Weekly workflow (canonical)](#weekly-workflow-canonical)
    - [High-level stages](#high-level-stages)
    - [Authoritative weekly workflow (runs, in this order)](#authoritative-weekly-workflow-runs-in-this-order)
    - [Outputs and conventions](#outputs-and-conventions)
  - [Scripts](#scripts)
  - [Validation](#validation)
  - [Leakage audit](#leakage-audit)
  - [Changelog](#changelog)
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

This project targets **Python 3.14+** (see `pyproject.toml`).

#### Recommended: uv project workflow

This repo uses `pyproject.toml` plus `uv.lock` for reproducible runs:

- `pyproject.toml` is the source of truth for runtime and development dependencies
- `uv.lock` is the lockfile used to sync environments reproducibly

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

To refresh the lockfile and sync the active virtual environment:

```bash
./update_requirements.sh
```

The helper expects `uv` on your `PATH` and an activated project virtual environment. If `.venv` is
missing, it offers to create one with `uv venv .venv` and then exits so you can activate the
environment before re-running it.

For a manual upgrade without the helper script:

```bash
uv lock --upgrade
uv sync
```

#### Alternative: create the venv manually

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

uv sync
```

### Run tests

```bash
python -m pytest
```

By default, pytest runs with coverage enabled (configured in `pyproject.toml`). To disable coverage
for a quick local run:

```bash
python -m pytest --no-cov
```

To run explicitly with coverage (same behavior as the default config):

```bash
python -m pytest --cov=nfl_predictor --cov-report=term-missing
```

Pytest now enforces the repo coverage floor of **90%** by default via `pyproject.toml`. To try a
stricter local target, add `--cov-fail-under` with a higher value. The preseason hardening target
remains **90% or higher**, with **100%** as the aspirational ceiling:

```bash
python -m pytest --cov-fail-under=95
```

Current validated local baseline as of 2026-06-13: `406 passed` with `90.01%` coverage.

## Data collection (Polars + nflreadpy)

The authoritative data build pipeline is:

```bash
python -m nfl_predictor.data_collection
```

The default season range is controlled by `constants.MIN_SEASON` (currently 1999).

This writes datasets under `data/` (paths are defined in `nfl_predictor/constants.py`).

Typical outputs:

- `data/all_data.csv` and `data/all_data_ml.csv`
- `data/completed_games.csv` and `data/completed_games_ml.csv`
- `data/predict/week_XX_games_to_predict.csv`

Note: the `data/` directory is gitignored by default; generate it via the data collection step
above. Note: `*_ml.csv` files include model-ready engineered features.

Historical seasons load from cached artifacts where available. nflreadpy outputs are cached per
season under `data/cache/nflreadpy` (schedule + team stats). Current/future seasons are always
refreshed to keep upcoming games and lines current. Use `--min-season`/`--max-season` to override
the default season window (defaults to `constants.MIN_SEASON` through the current NFL season).

TeamRankings data is cached under `data/<season>/` as week-level CSVs; enable debug logging to see
cache hits. Use `--timing` to log per-step runtimes and `--debug-logs` for detailed ETL diagnostics.
Use `--refresh-nflreadpy` to force refresh nflreadpy data even when cache exists.

## Data sources + missing data

This project is designed to keep an invariant output schema across seasons, even when some sources
are missing historically.

Primary sources:

- `nflreadpy` (NFLverse): schedules, results, and team-level stats.
- Local cached CSVs under `data/` for Elo/market data when present.
- TeamRankings web scrape for select ratings and stats not available in NFLverse (see ETL logs).

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
`game_type` column (i.e., postseason rows are filtered out). You can still generate predictions for
playoff games as long as the feature row exists.

To include postseason games in training, pass `--include-postseason`. To emphasize postseason games,
also set `--postseason-weight` (e.g., `--postseason-weight 1.5`). Optional recency weighting is
available via `--recency-half-life-weeks` or `--recency-half-life-seasons` (use only one) to apply
exponential decay to training and calibration samples.

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
`--market-prob-source raw|novig` to choose implied-prob handling and `--market-prob-blend-method
prob|logit` to blend in probability or log-odds space.

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

This is the **canonical evaluation protocol** for model selection. By default it evaluates the last
N seasons (regular season only) with time-aware calibration from the last K weeks of each eval
season. Use `--include-postseason` if you want postseason folds included. Optional recency weighting
is available via `--recency-half-life-weeks` or `--recency-half-life-seasons` (use only one). GPU
acceleration is optional: add `--xgb-tree-method hist --xgb-device cuda`. If the latest season is
incomplete, either pass `--exclude-incomplete-seasons` or specify `--eval-seasons` explicitly; the
metrics report includes the evaluated window and any exclusions. Walk-forward calibration uses the
last K weeks strictly before the eval week; if insufficient weeks or outcomes are available,
calibration is skipped for that fold.

Trend/season-phase ablation (drop trend + season-phase features while keeping everything else
identical) is available via `--disable-trend-features`. Example 2x2 comparison matrix:

```bash
python scripts/walk_forward_backtest.py
python scripts/walk_forward_backtest.py --recency-half-life-seasons 2
python scripts/walk_forward_backtest.py --disable-trend-features
python scripts/walk_forward_backtest.py --disable-trend-features --recency-half-life-seasons 2
```

Recent ablation example (2003-2025 seasons, include postseason, calibration=platt, recency half-life
seasons=2):

```text
Setting                         Brier    LogLoss  MarginMAE  TotalMAE  ActualPts
Trends on, recency off          0.2325   0.7525   10.0159    10.1254   213.35
Trends on, recency on           0.2804   1.9525   10.0691    10.0670   213.55
Trends off, recency off         0.2372   0.7772   10.0987    10.0759   210.55
Trends off, recency on          0.2827   1.9775   10.2003    10.0706   211.50
```

Interpretation:

- Trend features improve probability metrics (Brier/log loss) and margin MAE, with a small tradeoff
  in total MAE.
- Recency weighting (half-life seasons=2) hurts probability metrics in this run; keep it off unless
  a future ablation shows improvement.

Evaluation rule: "Model selection is based on time-aware walk-forward evaluation; random CV is not
authoritative." Season-blocked CV is used for hyperparameter tuning only; walk-forward remains the
source of truth.

## Weekly workflow (canonical)

The canonical "do everything for this week" entrypoint is:

```bash
python scripts/weekly_run.py --help
```

### High-level stages

1. (optional) refresh data (`python -m nfl_predictor.data_collection`)
2. (optional) walk-forward compare to choose market/calibration/prob-postprocess variants
3. train + calibrate the selected configuration
4. generate weekly predictions + betting outputs + (optional) power rankings

Notes:

- `--wf-*` flags control **walk-forward comparison** behavior (model selection).
- `--train-*` flags control **final training/calibration** for the model used to produce weekly
  outputs.
- `--xgb-*` flags control XGBoost runtime (GPU/CPU), and should be used for both comparison and
  final training.
- Outputs are written under the run directory (default: `models/<run_id>/`) unless `--output-dir` is
  provided.
- Stage 1 walk-forward comparison is resumable and writes `wf_compare/` artifacts under the run
  directory (including `wf_summary.csv` and per-candidate results).

### Authoritative weekly workflow (runs, in this order)

1. Refresh data (ETL)

   ```bash
   python -m nfl_predictor.data_collection
   ```

2. Canonical evaluation + model selection (walk-forward)

   ```bash
   python scripts/walk_forward_backtest.py --help
   ```

3. Train + predict for the upcoming week (writes predictions + artifacts)

   ```bash
   python -m nfl_predictor.ml_model --help
   ```

4. Power rankings + projected standings

   ```bash
   python scripts/power_rankings.py --help
   ```

### Outputs and conventions

- Run artifacts (model/metrics/metadata/feature importance) land under `models/<run_id>/` by
  default.
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
Walk-forward reports also record the evaluation window, calibration window, and any excluded
incomplete seasons.

## Scripts

Repo utilities under `scripts/`:

- `scripts/betting_pipeline.py`: end-to-end orchestration (walk-forward compare -> resumable tuning
  -> final train -> weekly predictions + betting_report.csv). If `--predict-path` is omitted, the
  newest `data/predict/week_XX_games_to_predict.csv` file is selected automatically. `--dry-run`
  previews the planned paths/stages even in a clean checkout before local `data/` files exist. See
  `--help`.
- `scripts/objective_compare_models.py`: objective walk-forward comparison of two saved models by
  retraining per fold under identical splits.
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
python scripts/wf_compare.py \
  --eval-last-n-seasons 3 \
  --market-mode hybrid \
  --market-prob-source raw \
  --market-prob-blend-method prob
```

```bash
python scripts/wf_compare.py \
  --eval-last-n-seasons 3 \
  --market-mode all \
  --market-prob-source both \
  --market-prob-blend-method both
```

Uncertainty-aware comparison:

```bash
python scripts/wf_compare.py \
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
python scripts/weekly_run.py --config path/to/weekly_run.json
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

Both validation scripts exit non-zero when the input data file is missing or the validation fails,
so they are safe to use in shell automation.

Canonical local validation sequence:

```bash
ruff format --check .
ruff check .
pyright .
ty check .
python -m pytest
markdownlint .
uv lock --check
uv sync --check --active
```

GitHub Actions mirrors this gate in `.github/workflows/validation.yml` and also runs the
editable-install smoke check plus `--help` smoke checks for `nfl_predictor.ml_model`,
`scripts/weekly_run.py`, and `scripts/power_rankings.py`.

## Leakage audit

To detect obvious feature leakage patterns:

```bash
python scripts/leakage_audit.py
```

## Changelog

Release history lives in `CHANGELOG.md` and follows the [Common
Changelog](https://common-changelog.org/) format. The historical baseline is `0.1.0` from `main`.

When preparing the next release, add a new `## VERSION - YYYY-MM-DD` entry at the top of the file
and keep the change groups in this order:

- `Changed`
- `Added`
- `Removed`
- `Fixed`

Keep each change to a single imperative line, link the most relevant commit or PR, and skip routine
formatting noise. Update `CHANGELOG.md` whenever user-facing behavior, tooling expectations, or the
operating workflow changes. Pushing a `0.x.y` or `v0.x.y` tag triggers
`.github/workflows/release.yml`, which extracts the matching `CHANGELOG.md` section and creates or
updates the GitHub release. Keep git tags aligned with changelog versions.

## Artifacts

Training/backtests can write a run directory containing reproducible artifacts.

- Use `--run-dir` to write `model.joblib`, `metadata.json`, and (when evaluated)
  `metrics_report.json`.
- `feature_importance.json` includes XGBoost gain/weight importance per model head.
- Metadata includes timestamp, dataset fingerprint/hash, key package versions, training config/CLI
  args, feature list, and tuning/early-stopping info (when used). `models/` and `optuna.db` are
  gitignored by default, so keep run artifacts local unless you copy them elsewhere.

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
- Stadium metadata features (roof/surface/type, elevation, venue location).
- Head coach prior record features (career and team-specific).

## Open work

Active tasks (milestones + guardrails) are tracked in `TODO.md`. Completed milestones and past work
live in `ARCHIVE.md`.

## Development notes

- ETL and feature engineering run in Polars.
- All NFLverse data is pulled via `nflreadpy`.
- Logging uses the project logger; avoid `print`.
- Formatting is enforced via Ruff format (line length 100).
- Linting and import sorting are enforced via Ruff (includes isort rules).
- Type checking runs through both Pyright and Ty; both are required local validation gates.
- Development dependencies are declared in `pyproject.toml` and synced via `uv.lock`.

For users reading this documentation: commands are shown assuming your project virtual environment
is already activated. Agent-specific files keep the fully qualified `.venv/bin/...` forms for
automation reliability.

See the Validation section above for the canonical local validation sequence.

## Safety and claims

This project outputs statistical forecasts and backtest metrics. It does not guarantee accuracy,
profit, or betting success.
