# ARCHIVE - Completed Milestones

This file contains completed milestones and optional enhancements that were previously tracked in
`TODO.md`. Keep this as the audit trail. If future changes regress behavior, re-run the acceptance
checks from the relevant section.

---

## Completed core milestones (0-11)

> These items were completed and verified in prior work. They are archived here to keep `TODO.md`
> focused on active work.

### Milestone 0 - Repo scan & plan

- [x] Identify ML entrypoints (train, predict, backtest scripts).
- [x] Identify where margin/total and win prob calibration live.
- [x] List current artifact outputs and missing metadata.
- [x] Confirm where ML datasets and week prediction inputs are produced.

### Milestone 1 - Canonical Margin/Total pipeline

- [x] Margin/total modeling is the primary path.
- [x] Score derivation is stable and unit-tested.
- [x] Outputs include margin/total and derived scores.

### Milestone 2 - Preprocessing cleanup

- [x] XGBoost path avoids scaling and avoids accidental densification.
- [x] Missing values are handled intentionally.

### Milestone 3 - Training improvements

- [x] Early stopping is enabled.
- [x] `eval_metric` aligns to the optimization target.
- [x] Parallelism is configurable.
- [x] Training config is serialized into metadata.

### Milestone 4 - Walk-forward evaluation

- [x] Walk-forward backtest exists and is time-aware.
- [x] Per-week and per-season metrics are emitted.

### Milestone 5 - Probability calibration + diagnostics

- [x] Calibration options are implemented.
- [x] Brier/log loss and reliability summaries are reported.

### Milestone 6 - Quantile intervals

- [x] Margin and total include p10/p50/p90 outputs.
- [x] Interval columns exist and are validated.

### Milestone 7 - Market transforms + anchoring

- [x] Market transforms are explicit and configurable.
- [x] Market anchoring is supported.
- [x] Market probability blending/clamping exists as configured.

### Milestone 8 - Leakage audit

- [x] Leakage audit mode exists and emits a JSON report.
- [x] Tests verify detection of leaked columns.

### Milestone 9 - Artifact contract + metadata

- [x] Run directories include model + metadata + metrics report.
- [x] Artifacts are loadable without hidden state.

### Milestone 10 - Golden command entrypoint

- [x] One command runs train + backtest + weekly predictions and writes artifacts.

### Milestone 11 - Dependency pinning + documentation

- [x] ML dependencies are pinned.
- [x] CPU-only path works and is documented.

---

## Completed milestones (12-19)

### Milestone 12 - Documentation + repository cleanup (Polars-only narrative)

- [x] Remove documentation references to deprecated data collection and utility modules.
- [x] Ensure all docs describe `nfl_predictor/data_collection.py` as the authoritative ETL entrypoint.
- [x] Add a short "Data sources + missing data" section describing fallbacks and season coverage
  limits.

Acceptance:

- [x] Docs reference only the Polars+nflreadpy pipeline and current ML entrypoints.

### Milestone 13 - constants.py cleanup and organization

- [x] Audit `nfl_predictor/constants.py` for unused constants and remove them.
- [x] Group constants into clear sections (paths, season/week rules, team mappings, feature names,
  defaults).
- [x] Ensure schema/feature lists are centralized and used everywhere (no hard-coded columns).

Tests:

- [x] Team alias mapping resolves to canonical abbreviations.
- [x] Schema lists contain no duplicates.
- [x] Required output columns exist in the ML datasets.

Acceptance:

- [x] `constants.py` is organized, minimal, and referenced consistently across ETL/ML/docs.

### Milestone 14 - Missing data handling across seasons

- [x] Inventory sources with limited historical coverage (injuries, markets, etc.).
- [x] Define a per-feature-group missing-data policy: null, default, or carry-forward.
- [x] Implement ETL fallbacks so output schema is invariant across seasons.
- [x] Ensure ML preprocessing handles nulls explicitly and logs fallback usage counts.

Tests:

- [x] ETL produces the same columns for a season with missing sources and one without.
- [x] Model train/predict completes when market fields are null.
- [x] Fallback counters appear in metrics/report outputs.

Acceptance:

- [x] Pipeline and ML runs succeed across the full historical range with consistent schema.

### Milestone 15 - Season-to-date record features (W-L-T, division, conference)

- [x] Implement record features for away and home teams (prefix columns `away_` and `home_`).
- [x] Record columns are defined in `constants.py` and included in the ML feature range.

Tests:

- [x] Computed season-to-date records match known records for a small fixture season/week range.
- [x] Divisional records reconcile with overall when a team's prior games are divisional.
- [x] Week 1 records are zero for all teams.

Acceptance:

- [x] Datasets include record features and they are available for training and prediction.

### Milestone 16 - Divisional rivalry feature

- [x] Add a `is_divisional_matchup` feature for each game.
- [x] Implement using a division mapping table in `constants.py`.
- [x] Ensure this applies to all seasons and teams.

Tests:

- [x] Known divisional pairings are flagged correctly.
- [x] Cross-division pairings are not flagged.

Acceptance:

- [x] All game rows contain the divisional indicator and it is stable across seasons.

### Milestone 17 - Lookahead / trap indicators

- [x] Build next-week opponent features using the schedule.
- [x] Add per-team lookahead features and join to games for away/home teams.

Tests:

- [x] Next-week opponent lookup is correct for a fixed season/week range.
- [x] Missing next-week opponent (end of season) yields null/default.

Acceptance:

- [x] Lookahead features exist in the ML dataset for all games with defined fallbacks.

### Milestone 18 - Motivational asymmetry features

- [x] Create a playoff-incentive feature set computed from standings and tiebreak proxies.
- [x] Integrate into ETL as season-to-date features available prior to each game.

Tests:

- [x] Incentive state features do not use future games.
- [x] Motivation feature join is schema-invariant when schedule scores are missing.

Acceptance:

- [x] Motivation/standings proxy features are available for all games without leakage.

---

## Completed optional enhancements

- [x] Realistic score post-processing for display outputs.
- [x] Market-only model removed when anchoring sufficed.
- [x] Blending weights constrained where applicable.
- [x] Interval coverage diagnostics implemented.

---

### Milestone 19 - Blocked/time-series cross-validation for tuning

- [x] Implement blocked CV at the season-week level for hyperparameter tuning and model selection.
- [x] Ensure folds are strictly time-ordered (train < validation).
- [x] Integrate CV into Optuna objectives so tuning does not overfit a single season holdout.
- [x] Report CV mean/std metrics in tuning CV summary (stored under `metrics_report.json`).

Tests:

- [x] CV fold generation is strictly time-ordered.
- [x] CV fold generation is deterministic.

Acceptance:

- [x] Optuna tuning evaluates parameters using time-series CV over season-week timepoints.

Primary files:

- [x] `nfl_predictor/ml/ml_model_core.py`
- [x] `tests/test_time_series_cv.py`

### Milestone 20 - Unit tests and code coverage hardening

- [x] `pytest-cov` is configured and coverage is reported by default.
- [x] Coverage threshold is enforced (current floor: 80%).
- [x] Tests exist across ETL joins and feature derivations introduced in prior milestones.

Acceptance:

- [x] `pytest --cov=nfl_predictor --cov-report=term-missing --cov-fail-under=80` passes.

Primary files:

- [x] `setup.cfg`
- [x] `tests/`

### Milestone 21 - Remove deprecated modules from the import surface

- [x] No code or docs reference deprecated modules.
- [x] Compatibility facades import cleanly.

Acceptance:

- [x] The package imports cleanly and no deprecated modules are referenced.

Primary files:

- [x] `tests/test_imports.py`
