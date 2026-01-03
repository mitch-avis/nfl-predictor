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

## Milestone 12 - Documentation + repository cleanup (Polars-only narrative)

- [x] Remove documentation references to deprecated data collection and utility modules.
- [x] Ensure all docs describe `nfl_predictor/data_collection.py` as the authoritative ETL entrypoint.
- [x] Add a short "Data sources + missing data" section describing fallbacks and season coverage limits.
- [ ] Primary files likely touched:
  - [x] `.github/copilot-instructions.md`
  - [x] `README.md`
  - [x] `TODO.md`
  - [ ] `ARCHIVE.md`

### Acceptance (Milestone 12)

- [x] Docs reference only the Polars+nflreadpy pipeline and current ML entrypoints.

---

## Milestone 13 - constants.py cleanup and organization

- [x] Audit `nfl_predictor/constants.py` for unused constants and remove them.
- [x] Group constants into clear sections (paths, season/week rules, team mappings, feature names, defaults).
- [x] Ensure schema/feature lists are centralized and used everywhere (no hard-coded columns).
- [ ] Primary files likely touched:
  - [x] `nfl_predictor/constants.py`
  - [x] `tests/test_constants.py`

### Tests (Milestone 13)

- [x] Unit test: team alias mapping resolves to canonical abbreviations.
- [x] Unit test: schema lists contain no duplicates.
- [x] Unit test: required output columns exist in the ML datasets.

### Acceptance (Milestone 13)

- [x] `constants.py` is organized, minimal, and referenced consistently across ETL/ML/docs.

---

## Milestone 14 - Missing data handling across seasons

- [x] Inventory sources with limited historical coverage (injuries, markets, etc.).
- [x] Define a per-feature-group missing-data policy: null, default, or carry-forward.
- [x] Implement ETL fallbacks so output schema is invariant across seasons.
- [x] Ensure ML preprocessing handles nulls explicitly and logs fallback usage counts.
- [ ] Primary files likely touched:
  - [x] `nfl_predictor/data_collection.py`
  - [x] `nfl_predictor/utils/polars_utils.py`
  - [x] `nfl_predictor/ml_model.py`
  - [x] `tests/test_missing_data_policy.py`

### Tests (Milestone 14)

- [x] Unit test: ETL produces the same columns for a season with missing sources and one without.
- [x] Unit test: model train/predict completes when injury/market fields are null.
- [x] Unit test: fallback counters appear in metrics/report outputs.

### Acceptance (Milestone 14)

- [x] Pipeline and ML runs succeed across the full historical range with consistent schema.

---

## Milestone 15 - Season-to-date record features (W-L-T, division, conference)

Add the following season-to-date features for each team at each game:

- overall: `wins`, `losses`, `ties`
- division: `division_wins`, `division_losses`, `division_ties`
- conference: `conference_wins`, `conference_losses`, `conference_ties`

Rules:

- values are computed using games strictly before the current game
- week 1 values are zero
- postseason is excluded unless explicitly enabled

- [ ] Implement record features for away and home teams (prefix columns `away_` and `home_`).
- [ ] Add schema entries in `constants.py` and include in the ML feature set.
- [ ] Primary files likely touched:
  - [ ] `nfl_predictor/data_collection.py`
  - [ ] `nfl_predictor/utils/polars_utils.py`
  - [ ] `nfl_predictor/constants.py`
  - [ ] `tests/test_record_features.py`

### Tests (Milestone 15)

- [ ] Unit test: computed season-to-date records match known records for a small fixture season/week
  range.
- [ ] Unit test: `division_* + non_division_*` reconciles to `overall_*` when applicable.
- [ ] Unit test: week 1 records are zero for all teams.

### Acceptance (Milestone 15)

- [ ] Datasets include record features and they are available for training and prediction.

---

## Milestone 16 - Divisional rivalry feature

- [ ] Add a `is_divisional_matchup` feature for each game.
- [ ] Implement using a division mapping table in `constants.py`.
- [ ] Ensure this applies to all seasons and teams.
- [ ] Primary files likely touched:
  - [ ] `nfl_predictor/constants.py`
  - [ ] `nfl_predictor/utils/polars_utils.py`
  - [ ] `tests/test_divisional_matchups.py`

### Tests (Milestone 16)

- [ ] Unit test: known divisional pairings are flagged correctly.
- [ ] Unit test: cross-division pairings are not flagged.

### Acceptance (Milestone 16)

- [ ] All game rows contain the divisional indicator and it is stable across seasons.

---

## Milestone 17 - Team health and injury burden features

- [ ] Pull injury data via `nflreadpy` (NFLverse).
- [ ] Define team-week aggregates and positional aggregates (QB/RB/WR/TE/OL/DL/LB/DB).
- [ ] Implement a numeric "health burden" score per team-week.
- [ ] Join team-week health into each game row for away/home teams.
- [ ] Provide explicit missing-data behavior for pre-coverage seasons (null/default + logged fallback).
- [ ] Primary files likely touched:
  - [ ] `nfl_predictor/data_collection.py`
  - [ ] `nfl_predictor/utils/polars_utils.py`
  - [ ] `nfl_predictor/constants.py`
  - [ ] `tests/test_injury_features.py`

### Tests (Milestone 17)

- [ ] Unit test: injury pipeline produces deterministic outputs for a fixed historical week.
- [ ] Unit test: injury features are null/default for seasons before injury coverage begins.
- [ ] Unit test: joining team-week injury aggregates produces the expected columns per game.

### Acceptance (Milestone 17)

- [ ] Injury features exist for all games (null/default where unavailable) and are included in the ML
  dataset.

---

## Milestone 18 - Lookahead / trap indicators

Trap-style indicators are computed for every game:

- current opponent strength vs next-week opponent strength
- rest/travel and short-week context
- game location changes (home/away) between weeks

- [ ] Build next-week opponent features using the schedule.
- [ ] Add per-team "lookahead pressure" features and join to games for away/home teams.
- [ ] Primary files likely touched:
  - [ ] `nfl_predictor/data_collection.py`
  - [ ] `nfl_predictor/utils/polars_utils.py`
  - [ ] `tests/test_lookahead_features.py`

### Tests (Milestone 18)

- [ ] Unit test: next-week opponent lookup is correct for a fixed season/week range.
- [ ] Unit test: missing next-week opponent (end of season) yields null/default.

### Acceptance (Milestone 18)

- [ ] Lookahead features exist in the ML dataset for all games with defined fallbacks.

---

## Milestone 19 - Motivational asymmetry features

Implement features that quantify each team's incentive level for a given game:

- playoff leverage (division/conference race)
- clinch/elimination states
- seeding leverage
- late-season weighting where appropriate, but features exist for all matchups

- [ ] Create a playoff-incentive feature set computed from standings and tiebreak proxies.
- [ ] Integrate into ETL as season-to-date features available prior to each game.
- [ ] Primary files likely touched:
  - [ ] `nfl_predictor/data_collection.py`
  - [ ] `nfl_predictor/utils/polars_utils.py`
  - [ ] `tests/test_motivation_features.py`

### Tests (Milestone 19)

- [ ] Unit test: incentive state features do not use future games.
- [ ] Unit test: known clinch/elimination scenarios in a fixture season produce expected flags.

### Acceptance (Milestone 19)

- [ ] Motivation features are available for all games and improve walk-forward metrics without leakage.

---

## Milestone 20 - Blocked/time-series cross-validation for tuning

- [ ] Implement blocked CV at the season-week level for hyperparameter tuning and model selection.
- [ ] Ensure folds are strictly time-ordered (train < validation).
- [ ] Integrate CV into Optuna objectives so tuning does not overfit a single season holdout.
- [ ] Report CV mean/std metrics in `metrics_report.json`.
- [ ] Primary files likely touched:
  - [ ] `nfl_predictor/ml_model.py`
  - [ ] `tests/test_time_series_cv.py`

### Tests (Milestone 20)

- [ ] Unit test: CV split generator never places a later week in training for an
  earlier-week validation fold.
- [ ] Unit test: CV is deterministic under a fixed random seed.

### Acceptance (Milestone 20)

- [ ] Tuning uses time-series CV and produces more stable out-of-sample results.

---

## Milestone 21 - Unit tests and code coverage hardening

- [ ] Add `pytest-cov` configuration to report coverage locally and in CI.
- [ ] Set and enforce a coverage threshold (target: 80%+; raise over time).
- [ ] Add tests for critical ETL joins and feature derivations introduced in Milestones 14-20.
- [ ] Primary files likely touched:
  - [ ] `tests/`
  - [ ] `pyproject.toml` and/or `setup.cfg`
  - [ ] CI config (if present)

### Acceptance (Milestone 21)

- [ ] Running `pytest --cov=nfl_predictor --cov-report=term-missing` passes and meets the coverage threshold.

---

## Milestone 22 - Remove deprecated modules from the import surface

- [ ] Ensure no imports reference deprecated modules.
- [ ] Remove deprecated modules from packaging/exports if they remain unused.
- [ ] Primary files likely touched:
  - [ ] `nfl_predictor/__init__.py`
  - [ ] `setup.cfg` / `setup.py`
  - [ ] `tests/test_imports.py`

### Acceptance (Milestone 22)

- [ ] The package imports cleanly and no deprecated modules are referenced in code or docs.
