# TODO — ML Upgrades for nfl-predictor

This file breaks the ML overhaul into small, checkable tasks. Treat each section as a mini-milestone:

- implement
- add/adjust tests
- run `pytest`
- update docs/scripts if behavior changes

---

## Agent Guardrails (must follow)

- No leakage: time-aware splits only; never train/evaluate on same-week or future games.
- Polars-first ETL: avoid pandas except inside ML modules where necessary.
- Keep changes in scope: ML modules, scripts, tests, and documentation only.
- No new external services or scraping beyond existing utilities.
- Do not claim betting profitability; report metrics and uncertainty honestly.
- Use constants for schema/column names; do not hard-code column lists.

---

## Defaults & Config Knobs (do not invent)

Unless explicitly overridden by CLI/config:

- Walk-forward start week: `3`
- Walk-forward eval seasons: last `N` seasons available (default `N=3`), OR explicit
  `--eval-seasons` list
- Walk-forward data path: `data/completed_games_ml.csv` via `--data-path`
- Calibration method: `platt` (options: `platt`, `isotonic`, `none`)
- Walk-forward calibration weeks: `4` via `--wf-calibration-weeks`
- Walk-forward report path: `models/<run_id>/metrics_report.json` via `--out-json`
- Market usage:
  - market transforms: `enabled` if odds columns exist (`--market-transform/--no-market-transform`)
  - market anchoring: `enabled` by default (`--market-anchor/--no-market-anchor`)
  - market prob blend/clamp: `disabled` by default (configurable via `--market-prob-weight`)
- Quantiles: `p10/p50/p90` for margin and total
- Random seed: fixed default (`42`) via `--random-seed` and configurable
- CPU-only must work; optional GPU paths must be guarded
- XGBoost parallelism: `os.cpu_count()` (or 1) default, configurable via `--xgb-n-jobs`

---

## Artifact “Done” Definition (Milestone 9 target state)

A completed training/backtest run must produce a single run directory (location configurable,
default shown):

- `models/<run_id>/model.joblib`
- `models/<run_id>/metadata.json`
- `models/<run_id>/metrics_report.json`

Where:

- `<run_id>` is unique and deterministic-friendly (e.g., timestamp + short hash).
- `metadata.json` contains:
  - created timestamp
  - git hash (if available)
  - dataset fingerprint/hash
  - library versions
  - training config
  - season/week ranges
  - feature list
  - tuned params (if any)
  - early stopping info
- `metrics_report.json` contains:
  - walk-forward per-week rows
  - per-season aggregates
  - overall aggregates
  - probability calibration summary

---

## Milestone 0 — Repo scan & plan (no behavior change)

- [x] Identify current ML entrypoints (train, predict, backtest scripts).
- [x] Identify where margin/total and win prob calibration currently live.
- [x] List the current artifact outputs and what metadata is missing.
- [x] Confirm where `data/all_data_ml.csv`, `data/completed_games_ml.csv`, and
  week prediction inputs are produced.
- [ ] Primary files likely touched:
  - `nfl_predictor/ml_model.py`
  - `scripts/backtest_predictions.py`
  - `nfl_predictor/data_collection_polars.py`
  - `nfl_predictor/constants.py`
  - `README.md`

### Milestone 0 — Acceptance

- A short developer note summarizing current state + planned file touch list.

---

## Milestone 1 — Canonical Margin/Total pipeline (core correctness)

- [x] Ensure margin/total modeling is the primary path:
  - predict `margin = home - away`
  - predict `total = home + away`
  - derive home/away scores from those
- [x] Ensure score derivation is numerically stable and well-tested.
- [x] If direct score models exist, demote them to optional/secondary ensemble components.
- [ ] Primary files likely touched:
  - `nfl_predictor/ml_model.py`
  - `tests/test_ml_model_margin_total.py`

### Milestone 1 — Tests

- [x] Unit test: converting (margin, total) -> (home, away) round-trips for synthetic values.
- [x] Unit test: prediction outputs contain required columns.

### Milestone 1 — Acceptance

- Prediction outputs always include margin/total + derived scores.

---

## Milestone 2 — Preprocessing cleanup (XGBoost-friendly)

- [x] Remove/avoid `StandardScaler` for XGBoost paths.
- [x] Ensure `ColumnTransformer` does not densify sparse matrices unintentionally.
- [x] Ensure missing values behavior is intentional (XGB supports missing natively).
- [ ] Primary files likely touched:
  - `nfl_predictor/ml_model.py`

### Milestone 2 — Tests

- [x] Unit test: training pipeline produces sparse matrix when categoricals are present (if applicable).
- [x] Unit test: training works with missing numeric values.

### Milestone 2 — Acceptance

- Training runtime/memory do not regress; pipeline is simpler.

---

## Milestone 3 — Training improvements (early stopping + aligned metrics)

- [x] Use early stopping for XGB models (margin and total).
- [x] Set `eval_metric` explicitly and align it with optimization target (MAE if optimizing MAE).
- [x] Avoid hard-coded `n_jobs`; use `os.cpu_count()` or config.
- [ ] Optional: extend Optuna tuning to any remaining untuned core models.
- [ ] Primary files likely touched:
  - `nfl_predictor/ml_model.py`
  - `tests/test_ml_model_training_smoke.py`

### Milestone 3 — Tests

- [x] Unit test: early stopping triggers with a small dataset (smoke test).
- [ ] Unit test: model config is serialized into metadata (Milestone 9).

### Milestone 3 — Acceptance

- Training logs show early stopping + eval_metric; artifacts include config.

---

## Milestone 4 — Walk-forward evaluation (required realism)

Implement a true walk-forward backtest:

- For each season in eval range:
  - for each week `w` (configurable start week, default 3):
    - train on games strictly before (season, week)
    - predict games in week `w`
    - record metrics
- Must expose config knobs:
- [x] `--wf-start-week` (default `3`)
- [x] `--eval-seasons` or `--eval-last-n-seasons` (default `3`)
- [ ] Primary files likely touched:
  - `scripts/walk_forward_backtest.py`
  - `nfl_predictor/ml/walk_forward.py`

### Milestone 4 — Metrics Required

- margin MAE
- total MAE
- win prob Brier + log loss (Milestone 5)
- market-relative residual MAE (if market anchoring enabled)
- per-week and per-season summaries

### Milestone 4 — Tests

- [x] Unit test: walk-forward split never includes same-week games in training.
- [x] Unit test: walk-forward deterministic with fixed random_state.

### Milestone 4 — Acceptance

- Single command generates walk-forward JSON with per-week rows + aggregates.

---

## Milestone 5 — Probability calibration + diagnostics (required reliability)

- [x] Implement calibration options:
  - Platt scaling (logistic regression)
  - Isotonic regression
  - None (no calibration)
- [x] Derive raw win prob from predicted margin baseline (normal-CDF mapping ok as baseline).
- [x] If calibration enabled, calibrated probability is the default output.
- [x] Add diagnostics to reports:
  - Brier score
  - log loss
  - binned calibration summary (reliability table)
- Must expose config knobs:
- [x] `--calibration` (default `platt`; options `platt|isotonic|none`)
- [ ] Primary files likely touched:
  - `nfl_predictor/ml_model.py`
  - `nfl_predictor/ml/metrics.py`

### Milestone 5 — Tests

- [x] Unit test: probabilities always in [0, 1].
- [x] Unit test: calibration uses time-aware splits only.

### Milestone 5 — Acceptance

- Walk-forward report includes probability metrics + calibration summary.

---

## Milestone 6 — Quantile intervals (uncertainty outputs)

- [x] Add quantile regressors for margin and total:
  - at least P10, P50, P90
- [x] Add interval columns to prediction outputs and reports:
  - `predicted_margin_p10`, `predicted_margin_p50`, `predicted_margin_p90`
  - `predicted_total_p10`, `predicted_total_p50`, `predicted_total_p90`
- [x] Primary files likely touched:
  - model training code (to train quantiles)
  - prediction output schema/report schema

### Milestone 6 — Tests

- [x] Unit test: p10 <= p50 <= p90 for margin/total.
- [x] Unit test: interval columns exist.

### Milestone 6 — Acceptance

- Outputs include uncertainty intervals; (optional) add coverage diagnostics.

---

## Milestone 7 — Market transforms + anchoring + (optional) clamp/blend

- [x] Ensure market transforms are explicit and configurable.
- [x] Implement/verify market anchoring:
  - train on residuals vs market baseline
  - add baseline back at prediction time
- [x] Optional: implement market probability clamp/blend:
  - `p_final = w * p_model + (1 - w) * p_market`
  - `w` configurable; validated via walk-forward.
- Must expose config knobs:
  - [x] `--market-anchor` (default `true`)
  - [x] `--market-prob-weight` (default disabled; when set, enable blending)
- [ ] Primary files likely touched:
  - market feature transform utilities
  - training target transform logic
  - probability blending logic

### Milestone 7 — Tests

- [x] Unit test: anchoring math correct (baseline + residual).
- [x] Unit test: market blend weight boundaries and behavior.

### Milestone 7 — Acceptance

- Models run with/without market info; report market-relative metrics.

---

## Milestone 8 — Leakage audit tool/mode (required safety)

- [x] Add a leakage audit mode that:
  - asserts target columns are not used as features
  - flags suspicious predictors (e.g., extreme correlations)
  - validates season-to-date features exclude the current game row (when possible)
- [x] Output a structured audit report (JSON) with clear pass/fail.
- [ ] Primary files likely touched:
  - `scripts/leakage_audit.py`
  - `tests/`

### Milestone 8 — Tests

- [x] Unit test: intentionally leaked column is detected.
- [x] Unit test: audit runs on a small fixture dataset.

### Milestone 8 — Acceptance

- Audit yields pass/fail summary + flagged columns list.

---

## Milestone 9 — Artifact contract + metadata (required reproducibility)

Every saved model must produce:

- model file (e.g., `.joblib`)
- `metadata.json` adjacent
- `metrics_report.json` adjacent (for backtests)

Metadata keys must include:

- created timestamp
- git commit hash (if available)
- dataset fingerprint/hash
- library versions (xgboost, sklearn, numpy, pandas, polars, scipy)
- training config/CLI args
- season/week ranges (train/calibration/holdout)
- feature list used
- best params and early stopping info

- [ ] Primary files likely touched:
  - `nfl_predictor/ml/artifacts.py`
  - `nfl_predictor/ml_model.py`
  - `scripts/walk_forward_backtest.py`

### Milestone 9 — Tests

- [ ] Unit test: metadata exists and contains required keys.
- [ ] Unit test: artifact loads without external state.

### Milestone 9 — Acceptance

- Artifacts are self-describing and reproducible.

---

## Milestone 10 — “Golden command” entrypoint (developer ergonomics)

- [ ] Provide a single runnable command (script/module) to:
  - train
  - run walk-forward backtest
  - generate current-week predictions
  - write artifacts + reports
- [ ] Primary files likely touched:
  - `scripts/golden_command.py`
  - `README.md`

### Milestone 10 — Acceptance

- README snippet includes exact commands + expected outputs.

---

## Milestone 11 — Dependency pinning & documentation (reproducibility)

- [ ] Pin ML dependencies in the repo’s dependency system (requirements/pyproject/lockfile).
- [ ] Document supported Python version(s) and CPU/GPU notes.
- [ ] Ensure CPU-only path works.
- [ ] Primary files likely touched:
  - `requirements.txt`
  - `pyproject.toml`
  - `README.md`

### Milestone 11 — Acceptance

- Fresh venv install produces stable training/backtest outputs (within tolerance).

---

## Optional Enhancements (after required milestones)

- [ ] Score realism post-processing (configurable rounding/snapping) after predictions.
- [ ] Remove market-only model if anchoring suffices (simplify).
- [ ] Constrain blending weights (non-negative or sum-to-1) if blender remains.
- [ ] Coverage diagnostics: how often true margin/total falls inside P10–P90.
