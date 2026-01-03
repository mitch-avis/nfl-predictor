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
