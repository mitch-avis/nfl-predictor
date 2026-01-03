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
- [x] Model train/predict completes when injury/market fields are null.
- [x] Fallback counters appear in metrics/report outputs.

Acceptance:

- [x] Pipeline and ML runs succeed across the full historical range with consistent schema.

Note:

- NFLverse participation/injury data for in-progress seasons does not update during the season;
  recent seasons are published after postseason. In-season runs emit injury columns as nulls and
  the ML pipeline should not rely on them by default.

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

### Milestone 17 - Team health and injury burden features

- [x] Pull injury data via `nflreadpy` (NFLverse).
- [x] Define team-week aggregates and positional aggregates (QB/RB/WR/TE/OL/DL/LB/DB).
- [x] Implement a numeric "health burden" score per team-week.
- [x] Join team-week health into each game row for away/home teams.
- [x] Provide explicit missing-data behavior for pre-coverage seasons (null/default + logged
  fallback).

Tests:

- [x] Injury pipeline produces deterministic outputs for a fixed historical week.
- [x] Injury features are null/default for seasons before injury coverage begins.
- [x] Joining team-week injury aggregates produces the expected columns per game.

Acceptance:

- [x] Injury features exist for all games (null/default where unavailable) and are included in the
  ML dataset.

### Milestone 18 - Lookahead / trap indicators

- [x] Build next-week opponent features using the schedule.
- [x] Add per-team lookahead features and join to games for away/home teams.

Tests:

- [x] Next-week opponent lookup is correct for a fixed season/week range.
- [x] Missing next-week opponent (end of season) yields null/default.

Acceptance:

- [x] Lookahead features exist in the ML dataset for all games with defined fallbacks.

### Milestone 19 - Motivational asymmetry features

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
