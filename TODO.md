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

## Milestone 32B - Stadium metadata + venue features (non-leaky)

Context: Weather and referee data are not reliable pre-kickoff, so focus is stadium metadata
and coach features only.

Tasks:

- [x] Expand `STADIUMS` to include `name` and `elevation` (and keep city/state).
- [ ] Update stadium feature derivation to use the new `STADIUMS` fields and drop any
  legacy altitude map if redundant.
- [ ] Keep stadium type/surface features derived from NFLverse schedule fields.
- [ ] Add/adjust tests for stadium metadata parsing and safe fallbacks.
- [ ] Update README feature list to reflect stadium-only (no weather/ref).

Acceptance:

- [ ] Stadium metadata features are present for all games with safe defaults.

---

## Milestone 33 - Head coach features (time-safe)

Tasks:

- [ ] Keep coach prior record features (career + team-specific) computed strictly to date.
- [ ] Add/adjust tests that verify no leakage in coach-derived features.
- [ ] Run walk-forward ablation to confirm effect.

Acceptance:

- [ ] Coach features are leakage-safe and show documented impact in walk-forward.

---

## Milestone 34 - Remove weather + referee features (post-game data)

Tasks:

- [ ] Remove weather fields from constants, ETL, and tests (including any parsing logic).
- [ ] Remove referee features and any ref-derived aggregates from ETL and tests.
- [ ] Ensure feature ordering/schema stays invariant after removal.
- [ ] Update README + ARCHIVE notes to reflect the rollback.

Acceptance:

- [ ] No weather/ref features exist in datasets or feature specs; tests remain green.

---

## Explicitly out of scope (unless the world changes)

- Injury/practice participation features.
  - Prior work showed the data is not reliably updated in-season.
  - QB Elo + manual verified starters remain the preferred approach.
