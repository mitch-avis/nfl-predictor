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
- Calibration method: `platt` (options: `platt`, `isotonic`, `none`)
- Market usage:
  - market transforms: `enabled` if odds columns exist
  - market anchoring: `enabled` by default (configurable)
  - market prob blend/clamp: `disabled` by default (configurable via `--market-prob-weight`)
- Quantiles: `p10/p50/p90` for margin and total
- Random seed: fixed default (e.g., `42`) and configurable
- CPU-only must work; optional GPU paths must be guarded

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

- [ ] Identify current ML entrypoints (train, predict, backtest scripts).
- [ ] Identify where margin/total and win prob calibration currently live.
- [ ] List the current artifact outputs and what metadata is missing.
- [ ] Confirm where `data/all_data_ml.csv`, `data/completed_games_ml.csv`, and
  week prediction inputs are produced.
- [ ] Primary files likely touched:
  - ML code (e.g., `nfl_predictor/ml_model.py` or equivalent)
  - scripts for training/backtesting/prediction in `scripts/`
  - constants/config locations (e.g., `nfl_predictor/constants.py`)

### Milestone 0 — Acceptance

- A short developer note summarizing current state + planned file touch list.

---

## Milestone 1 — Canonical Margin/Total pipeline (core correctness)

- [ ] Ensure margin/total modeling is the primary path:
  - predict `margin = home - away`
  - predict `total = home + away`
  - derive home/away scores from those
- [ ] Ensure score derivation is numerically stable and well-tested.
- [ ] If direct score models exist, demote them to optional/secondary ensemble components.
- [ ] Primary files likely touched:
  - core model implementation module(s)
  - prediction output formatting/serialization code

### Milestone 1 — Tests

- [ ] Unit test: converting (margin, total) -> (home, away) round-trips for synthetic values.
- [ ] Unit test: prediction outputs contain required columns.

### Milestone 1 — Acceptance

- Prediction outputs always include margin/total + derived scores.

---

## Milestone 2 — Preprocessing cleanup (XGBoost-friendly)

- [ ] Remove/avoid `StandardScaler` for XGBoost paths.
- [ ] Ensure `ColumnTransformer` does not densify sparse matrices unintentionally.
- [ ] Ensure missing values behavior is intentional (XGB supports missing natively).
- [ ] Primary files likely touched:
  - preprocessing/pipeline construction in ML module(s)
  - any shared feature encoding utilities

### Milestone 2 — Tests

- [ ] Unit test: training pipeline produces sparse matrix when categoricals are present (if applicable).
- [ ] Unit test: training works with missing numeric values.

### Milestone 2 — Acceptance

- Training runtime/memory do not regress; pipeline is simpler.

---

## Milestone 3 — Training improvements (early stopping + aligned metrics)

- [ ] Use early stopping for XGB models (margin and total).
- [ ] Set `eval_metric` explicitly and align it with optimization target (MAE if optimizing MAE).
- [ ] Avoid hard-coded `n_jobs`; use `os.cpu_count()` or config.
- [ ] Optional: extend Optuna tuning to any remaining untuned core models.
- [ ] Primary files likely touched:
  - training routine(s) for margin/total
  - config parsing / CLI args for training

### Milestone 3 — Tests

- [ ] Unit test: early stopping triggers with a small dataset (smoke test).
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
  - [ ] `--wf-start-week` (default `3`)
  - [ ] `--eval-seasons` or `--eval-last-n-seasons` (default `3`)
- [ ] Primary files likely touched:
  - backtest script(s) in `scripts/`
  - split/build-fold logic in ML module(s)

### Milestone 4 — Metrics Required

- margin MAE
- total MAE
- win prob Brier + log loss (Milestone 5)
- market-relative residual MAE (if market anchoring enabled)
- per-week and per-season summaries

### Milestone 4 — Tests

- [ ] Unit test: walk-forward split never includes same-week games in training.
- [ ] Unit test: walk-forward deterministic with fixed random_state.

### Milestone 4 — Acceptance

- Single command generates walk-forward JSON with per-week rows + aggregates.

---

## Milestone 5 — Probability calibration + diagnostics (required reliability)

- [ ] Implement calibration options:
  - Platt scaling (logistic regression)
  - Isotonic regression
  - None (no calibration)
- [ ] Derive raw win prob from predicted margin baseline (normal-CDF mapping ok as baseline).
- [ ] If calibration enabled, calibrated probability is the default output.
- [ ] Add diagnostics to reports:
  - Brier score
  - log loss
  - binned calibration summary (reliability table)
- Must expose config knobs:
  - [ ] `--calibration` (default `platt`; options `platt|isotonic|none`)
- [ ] Primary files likely touched:
  - probability computation + calibration code
  - backtest reporting code

### Milestone 5 — Tests

- [ ] Unit test: probabilities always in [0, 1].
- [ ] Unit test: calibration uses time-aware splits only.

### Milestone 5 — Acceptance

- Walk-forward report includes probability metrics + calibration summary.

---

## Milestone 6 — Quantile intervals (uncertainty outputs)

- [ ] Add quantile regressors for margin and total:
  - at least P10, P50, P90
- [ ] Add interval columns to prediction outputs and reports:
  - `margin_p10`, `margin_p50`, `margin_p90`
  - `total_p10`, `total_p50`, `total_p90`
- [ ] Primary files likely touched:
  - model training code (to train quantiles)
  - prediction output schema/report schema

### Milestone 6 — Tests

- [ ] Unit test: p10 <= p50 <= p90 for margin/total.
- [ ] Unit test: interval columns exist.

### Milestone 6 — Acceptance

- Outputs include uncertainty intervals; (optional) add coverage diagnostics.

---

## Milestone 7 — Market transforms + anchoring + (optional) clamp/blend

- [ ] Ensure market transforms are explicit and configurable.
- [ ] Implement/verify market anchoring:
  - train on residuals vs market baseline
  - add baseline back at prediction time
- [ ] Optional: implement market probability clamp/blend:
  - `p_final = w * p_model + (1 - w) * p_market`
  - `w` configurable; validated via walk-forward.
- Must expose config knobs:
  - [ ] `--market-anchor` (default `true`)
  - [ ] `--market-prob-weight` (default disabled; when set, enable blending)
- [ ] Primary files likely touched:
  - market feature transform utilities
  - training target transform logic
  - probability blending logic

### Milestone 7 — Tests

- [ ] Unit test: anchoring math correct (baseline + residual).
- [ ] Unit test: market blend weight boundaries and behavior.

### Milestone 7 — Acceptance

- Models run with/without market info; report market-relative metrics.

---

## Milestone 8 — Leakage audit tool/mode (required safety)

- [ ] Add a leakage audit mode that:
  - asserts target columns are not used as features
  - flags suspicious predictors (e.g., extreme correlations)
  - validates season-to-date features exclude the current game row (when possible)
- [ ] Output a structured audit report (JSON) with clear pass/fail.
- [ ] Primary files likely touched:
  - new audit module or new subcommand in an existing script
  - reporting utilities

### Milestone 8 — Tests

- [ ] Unit test: intentionally leaked column is detected.
- [ ] Unit test: audit runs on a small fixture dataset.

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
  - artifact save/load utilities
  - training/backtest entrypoints (to emit run directory outputs)

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
  - a single new script entrypoint (or consolidation of existing scripts)
  - README/docs snippet for usage

### Milestone 10 — Acceptance

- README snippet includes exact commands + expected outputs.

---

## Milestone 11 — Dependency pinning & documentation (reproducibility)

- [ ] Pin ML dependencies in the repo’s dependency system (requirements/pyproject/lockfile).
- [ ] Document supported Python version(s) and CPU/GPU notes.
- [ ] Ensure CPU-only path works.
- [ ] Primary files likely touched:
  - dependency files (`requirements.txt` / `pyproject.toml` / lockfile)
  - README/docs

### Milestone 11 — Acceptance

- Fresh venv install produces stable training/backtest outputs (within tolerance).

---

## Optional Enhancements (after required milestones)

- [ ] Score realism post-processing (configurable rounding/snapping) after predictions.
- [ ] Remove market-only model if anchoring suffices (simplify).
- [ ] Constrain blending weights (non-negative or sum-to-1) if blender remains.
- [ ] Coverage diagnostics: how often true margin/total falls inside P10–P90.
