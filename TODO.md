# TODO - Next Milestones for nfl-predictor

This file contains **active** work only. Completed milestones live in `ARCHIVE.md`.

Execution loop for each milestone:

- implement
- add/adjust tests (increase coverage)
- run `pytest` (and coverage)
- update docs if behavior changes

---

## Guardrails checklist

- [ ] Confirm all new features apply to all matchups (no end-of-season-only logic).
- [ ] Confirm no leakage: all splits, features, and labels are time-aware.
- [ ] Confirm ETL is Polars-only and reads NFLverse via `nflreadpy`.
- [ ] Confirm artifacts remain reproducible (run folder, metadata, metrics).
- [ ] Confirm new code includes unit tests and does not reduce coverage.

---

## Milestone 20 - Blocked/time-series cross-validation for tuning

- [x] Implement blocked CV at the season-week level for hyperparameter tuning and model selection.
- [x] Ensure folds are strictly time-ordered (train < validation).
- [x] Integrate CV into Optuna objectives so tuning does not overfit a single season holdout.
- [x] Report CV mean/std metrics in `metrics_report.json`.
- [ ] Primary files likely touched:
  - [x] `nfl_predictor/ml_model.py`
  - [x] `tests/test_time_series_cv.py`

### Tests (Milestone 20)

- [x] Unit test: CV split generator never places a later week in training for an
  earlier-week validation fold.
- [x] Unit test: CV is deterministic under a fixed random seed.

### Acceptance (Milestone 20)

- [x] Tuning uses time-series CV and produces more stable out-of-sample results.

---

## Milestone 21 - Unit tests and code coverage hardening

- [x] Add `pytest-cov` configuration to report coverage locally and in CI.
- [x] Set and enforce a coverage threshold (target: 80%+; raise over time).
- [x] Add tests for critical ETL joins and feature derivations introduced in Milestones 14-20.
- [ ] Primary files likely touched:
  - [x] `tests/`
  - [x] `pyproject.toml` and/or `setup.cfg`
  - [ ] CI config (if present)

### Acceptance (Milestone 21)

- [x] Running `pytest --cov=nfl_predictor --cov-report=term-missing` passes and meets the coverage threshold.

---

## Milestone 22 - Remove deprecated modules from the import surface

- [x] Ensure no imports reference deprecated modules.
- [x] Remove deprecated modules from packaging/exports if they remain unused.
- [ ] Primary files likely touched:
  - [x] `nfl_predictor/__init__.py`
  - [x] `setup.cfg` / `setup.py`
  - [x] `tests/test_imports.py`

### Acceptance (Milestone 22)

- [x] The package imports cleanly and no deprecated modules are referenced in code or docs.
