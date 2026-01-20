# TODO - Active Work for nfl-predictor

This file contains **active** work only. Completed milestones live in `ARCHIVE.md`.

Execution loop for each milestone:

- implement
- add/adjust tests (increase coverage)
- run `.venv/bin/uv run pytest` (and coverage)
- run `.venv/bin/uv run ruff check .` and `.venv/bin/uv run black .`
- update docs if behavior changes

---

## Guardrails checklist (must stay true)

- [ ] Confirm all new features apply to all matchups (no end-of-season-only logic).
- [ ] Confirm no leakage: all splits, features, and labels are time-aware.
- [ ] Confirm ETL is Polars-first and pulls NFLverse via `nflreadpy`.
- [ ] Confirm artifacts remain reproducible (run folder, metadata, metrics).
- [ ] Confirm new code includes unit tests and does not reduce coverage.
- [ ] Confirm historical data is cached (TeamRankings + nflreadpy) and re-scrapes are minimized.

---

## Milestone 31 - Recency + trend features (non-linearity and drift)

Goal: add leakage-safe trend/recency signals plus optional time-weighted training.

### Feature design + audit

- [ ] Inventory existing recency signals (TR last_5/last_10 ratings, lookahead, motivation).
- [ ] Finalize minimal trend feature set and confirm they are time-safe.

### Trend features (time-safe)

- [ ] Rating trend: `last_5_games_rating - last_10_games_rating` for away/home + diff.
- [ ] Elo trend: `elo_pre - rolling_4wk_mean(elo_pre)` for away/home + diff.
- [ ] QB Elo trend: `qb_elo_pre - rolling_4wk_mean(qb_elo_pre)` for away/home + diff.
- [ ] Performance trend (select 1-2 stats): recent 4-week mean vs season-to-date mean
  (e.g., scoring margin, EPA, or turnover margin) for away/home + diff.
- [ ] Season-phase features: normalized `week_in_season` plus early/mid/late bucket flags.

### ETL + schema

- [ ] Implement rolling aggregates in Polars (per team, per season, prior weeks only).
- [ ] Add derived columns to `constants.py` and enforce schema ordering.
- [ ] Ensure missing-data policy is consistent for early weeks and short seasons.

### Recency weighting (exponential half-life)

- [ ] Add optional exponential half-life sample-weighting for training + calibration.
- [ ] Add CLI/config flags for half-life (weeks or seasons) in training + walk-forward.
- [ ] Keep default off and ensure weights are deterministic.

### Tests

- [x] Unit tests verifying trend features only use prior weeks.
- [x] Unit tests for recency weights (monotonic decay, boundary cases).
- [x] Unit tests for season-phase buckets and normalization.

### Evaluation

- [ ] Walk-forward comparisons with/without trend features and with/without weights.
- [ ] Track Brier/log loss first; pool points as tie-breakers; MAE third.

Acceptance:

- [ ] New features are leakage-safe and schema-invariant.
- [ ] Walk-forward results show a clear improvement or a documented tradeoff.

---

## Milestone 32 - Weather + venue effects (consistent, non-leaky)

- [ ] Extend stadium metadata beyond city/state:
  - stadium type (open / dome / retractable)
  - altitude (Denver, Mexico City, etc.)
- [ ] Pull historical weather where available (ideally from NFLverse via `nflreadpy`):
  - temperature, wind, precipitation flags
- [ ] Define strict missing-data policy and ensure invariant schema across seasons.
- [ ] Add tests for missing-weather fallbacks and schema invariance.

Acceptance:

- [ ] Weather/venue features exist for all games with safe fallbacks and improve metrics.

---

## Milestone 33 - Head coach features (if data is robust)

- [ ] Determine availability/coverage of head coach information (ideally via NFLverse).
- [ ] If reliable:
  - encode coach identity (categorical) and/or tenure/experience features
  - add coach prior record (career and with current team) computed strictly to date
- [ ] Add tests that verify no leakage in coach-derived features.

Acceptance:

- [ ] Coach features do not leak and demonstrate value in walk-forward.

---

## Milestone 34 - Referee features (if data is robust)

- [ ] Determine availability/coverage of referee assignments historically.
- [ ] If reliable:
  - encode referee identity (categorical) and/or per-ref historical tendencies computed to date
    (penalties, home bias proxies, etc.)
- [ ] Add tests that verify no leakage in ref-derived features.

Acceptance:

- [ ] Ref features do not leak and improve at least one primary metric.

---

## Explicitly out of scope (unless the world changes)

- Injury/practice participation features.
  - Prior work showed the data is not reliably updated in-season.
  - QB Elo + manual verified starters remain the preferred approach.
