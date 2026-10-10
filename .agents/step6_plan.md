# Roadmap step 6 work plan (absorbing `nfl-sos-ratings`)

Planning file for the delegating session; plan labels are allowed here and nowhere else. The
task text is `.agents/TODO.md`, "Roadmap Status", step 6. The decision log for the user is
`.agents/sos_port/DECISIONS.md`.

## Authority (the user, 2026-10-10, about 02:00)

The user asked for step 6 to run autonomously to completion or near-completion, with final
approvals batched at the end, and deferred every difference between the two repos to the
session's judgment, provided each difference, the decision, the exact reasoning and objective
data are logged for review. Read as:

- Proceeds without asking: all code on branches; decisions on differences (logged); candidate
  dataset builds outside the main checkout's `data/`; the walk-forward runs phase 4 needs, under
  rules 3, 4, 9, 12 and 13 (hypothesis, decision rule and run cap written before launch, one run
  at a time, two seeds, independent rescore before any number enters a doc).
- Held for the user's final approval, in one list: merging to `main` and pushing; replacing
  `data/` with a new build; any default that changes what the weekly run produces; the
  `../nfeloqb` fix; the unapproved feature ideas (referee, weather, attendance, team health),
  which are not started.
- Never: the user's web server (port 8001), writes to `../nfl-sos-ratings` (read at `9081bce`
  only) or `../nfeloqb`.

## Layout

- The main checkout stays on `main`: the user's web server and weekly runs import its code, so
  no step-6 code lands there before approval.
- Integration worktree `~/scratch/step6` on `feat/step6-sos-port` (off `main` at `ad07198`, plus
  the plan commits). Phase branches from `implementer` worktrees merge into it with `--no-ff`
  after an independent `reviewer` pass; the delegating session writes `CHANGELOG.md`, the version,
  `TODO.md`, `ARCHIVE.md`, `AGENTS.md`, `benchmarks.md`, the handoff prompt and the decision
  index there, and runs `scripts/gate.sh` there.
- Implementer worktrees start from `main`; each brief tells them to fast-forward to
  `feat/step6-sos-port` first.
- Candidate builds: `~/scratch/step6/data/` (a copy of the main checkout's upstream inputs; the
  ETL writes there). Walk-forward artifacts: new directories under the main checkout's
  `models/step6/` (`--out-json`, `--checkpoint-dir`), launched from the integration worktree at a
  recorded commit.

## Streams

| stream | phase | depends on | branch / state |
| --- | --- | --- | --- |
| S1 inventory and parity scripts | 1 | none | implementer running (from `cacd301`) |
| S2 new nflverse sources, `MODELS_PATH`, `compare` tie fix | 2 | none | implementer running |
| S3 own Elo (538 framework, 2019 additions, QB value, win totals) | 4 candidate | none | implementer running |
| S4 ratings core port (ridge, team, prior, sos, QB, histories, ranges) | 3 | none for the algorithms | implementer running |
| S8a web: pool helpers, margin toggle, live-betting design | 6 (58.1, 58.6, 58.3) | none | implementer running |
| S5 ETL integration: play-by-play fixes, ratings inputs and outputs, pre-week features, follow-ups | 3-4 | S1, S2, S4 | after those merge |
| S6 candidate build and measurement | 4 | S3, S5 | after S5 |
| S7 power rankings and projected standings | 5 | S5, S6 | after S6 |
| S8b web: team, quarterback, schedule, rank-range, head-to-head pages (58.2) | 6 | S5 output schema | after S5 |

## Measurement ladder (phase 4)

Pre-registered on 2026-10-10 before any step-6 code merged: the main checkout's
`models/step6/HYPOTHESIS.md` (arms, cap of 8 runs plus 2 conditional and one build, instrument,
decision rules). Do not edit it after the launch record is written.

## For the morning report (running list)

- Process incident: the own-Elo reviewer's first test request to Wikipedia's API (checking the
  stadium coordinates) sent the user's email address in its User-Agent header; its later
  requests used a generic agent. Disclose plainly.
- The web implementer used system `python3` once to edit two web files (forbidden form; the edit
  was harmless).
- Held for the user: the live-betting design's three choices (score source, odds source, odds
  snapshots under `data/live/`); calibrating the tiebreaker's top-score estimate (runs about 2
  points low); the reviewed pool-interval labels in `benchmarks.md` (lines 459, 503, 577 need
  their resampling unit confirmed from each run's own artifacts before labeling).
- The `compare` tie fix changes market-line pick accuracy in future reports (170 pick-time and 25
  closing-line coin flips; 0.6401 to 0.6579 and 0.6648 to 0.6670), no model number.
- Modeling observation (not changed): completed games carry the actual starting quarterback in
  every quarterback feature (the `qb_elos.csv` convention, and now the own Elo's), while
  upcoming games carry the projected one. The own-Elo evaluation's diagnostic puts the cost of
  knowing only the previous starter at +0.00101 Brier [-0.00027, +0.00232] and +0.00331 log loss
  [+0.00035, +0.00631] for the market-free preset (`.agents/sos_port/ELO_EVAL.md`, "Starters
  known before kickoff"; second review pending), so training rows hold slightly better
  quarterback information than prediction rows. A question for the user, not a step-6 change.

## Snapshots and timings

- Upstream inputs for the candidate build were copied into `~/scratch/step6/data/` on 2026-10-10
  at about 02:30 (main checkout's `data/` without backups, old cuts, `predict/`, `web/`); manifest
  `~/scratch/step6_data_snapshot_20261010.sha256` (715 files).
- A full ETL took 153 s on 2026-10-09; a GPU walk-forward of 2007-2025 from week 1 about 34
  minutes (`models/step4_reference/*/run.log`).
- Output location for the ported ratings files, shared by the ETL integration and the web pages:
  `data/ratings/{season}_{kind}.parquet`, the kinds named as in `nfl-sos-ratings`.
