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

## Milestone 35 - Pandas to Polars audit/refactor

- [ ] Inventory pandas usage across the repo and classify by module (ETL vs ML vs reporting).
- [ ] Identify pandas usage outside ML/reporting that can move to Polars safely.
- [ ] Refactor candidate modules to Polars-first implementations.
- [ ] Document any pandas usage that must remain (e.g., sklearn pipelines, calibration).
- [ ] Add/update tests to confirm schema parity and no leakage.

Acceptance:

- [ ] ETL and feature engineering are fully Polars-first with minimal pandas use.
- [ ] Remaining pandas usage is justified and documented.

---

## Explicitly out of scope (unless the world changes)

- Injury/practice participation features.
  - Prior work showed the data is not reliably updated in-season.
  - QB Elo + manual verified starters remain the preferred approach.
