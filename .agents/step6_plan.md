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
| S1 inventory and parity scripts | 1 | none | review fixes (rerun on the play-by-play snapshot) |
| S2 new nflverse sources, `MODELS_PATH`, `compare` tie fix | 2 | none | merged, `0.46.0` |
| S3 own Elo (538 framework, 2019 additions, QB value, win totals) | 4 candidate | none | merged, `0.47.0` |
| S4 ratings core port (ridge, team, prior, sos, QB, histories, ranges) | 3 | none for the algorithms | review fixes |
| S8a web: pool helpers, margin toggle, live-betting design | 6 (58.1, 58.6, 58.3) | none | merged, `0.48.0` |
| S5a play-by-play cache columns, stadium table | 3-4 | none | in review |
| S5b integration: play-by-play corrections, one starter, own Elo in the ETL | 3-4 | S5a | implementer running (on S5a's branch) |
| S5c integration: ratings in the ETL (published tables, pre-week features) | 3-4 | S4, S5a | implementer running (on S4's and S5a's branches) |
| S10 season statistics tables and metric registry | 6 (data for the pages) | S2 | implementer running |
| S5 ETL integration: play-by-play fixes, ratings inputs and outputs, pre-week features, follow-ups | 3-4 | S1, S2, S4 | after those merge |
| S6 candidate build and measurement | 4 | S3, S5 | after S5 |
| S7 power rankings and projected standings | 5 | S5, S6 | after S6 |
| S8b web: team, quarterback, schedule, rank-range, head-to-head pages (58.2) | 6 | S5 output schema | after S5 |

## Measurement ladder (phase 4)

Pre-registered on 2026-10-10 before any step-6 code merged: the main checkout's
`models/step6/HYPOTHESIS.md` (arms, cap of 8 runs plus 2 conditional and one build, instrument,
decision rules). Do not edit it after the launch record is written.

## Integration design (decided 2026-10-10, before the integration streams start)

Three integration streams start once the ratings port and the play-by-play column extension are
merged; they all touch the ETL orchestration, so each keeps its `data_collection.py` edits
small and local.

- **Play-by-play corrections (this repo's own play-by-play feature family)**, adopted from the
  phase-1 differences: a run play flagged `qb_scramble` counts as a dropback, not a carry (fixes
  the pre-2006 coding gap and the scrambles coded as runs); a play flagged both `qb_dropback` and
  `rush` counts once, as a dropback; the Jacksonville 2001-2002 play-by-play player-team repair is
  wired in; third-down, fourth-down and two-point flags are read on every play; a down flagged
  both converted and failed counts once. Kept as they are (definitional, both defensible):
  carries on the `rush` flag (penalty-wiped runs stay carries, symmetric with penalty-wiped
  dropbacks), two-point tries out of dropbacks, first downs as passing plus rushing, fumbles on
  plays that are not special teams, both-sides special-teams play counts, the explosive and
  stuffed definitions, Washington as `WSH`. Follow-ups in the same stream: quarterback scrambles
  credited to the scrambler (`rusher_player_id`), not the team-game's plurality passer; the
  current season's play-by-play requested once before kickoff; the unread `PBP_COUNT_COLUMNS`
  and quarterback sums dropped; `strength_games_played_diff` dropped.
- **One starter definition for every quarterback feature**: a completed game's starter comes
  from its play-by-play (the source's rule: the first first-quarter snap by a passer or rusher
  who is a quarterback that season), an upcoming game's from the projected starter the ETL
  already resolves (`away_qb`/`home_qb`). The schedule's `away_qb_id`/`home_qb_id` are wrong in
  42 team-games of 2006-2025 (the source's finding), so they are not used. Measure agreement with
  the current identity chain before switching and log it.
- **Ratings integration**: an adapter from this repo's play-by-play to the ratings package's
  input frames, using the source's definitions internally (so its ratings reproduce the published
  files), checked against the source's game logs and published ratings for 1999-2025; the
  published tables written to `data/ratings/{season}_{kind}.parquet`; pre-week features as a new
  feature group (offense, defense, special-teams and team ratings, `sos`, and the expected
  starter's pre-week quarterback rating, with `_diff` companions), group markers checked against
  every column name for collisions; one BLAS thread pinned in code (`threadpoolctl`), fixed row
  order, a two-run byte-identity test, `ratings` in the season cache's source directories. The
  pre-week quarterback identity uses the players file and roster weeks strictly before the week.
- **Own Elo wiring**: both presets built by the ETL (`own_elo`, `own_elo_wt`); upcoming games'
  projected starters from the ETL's expected starter; designed quarterback runs added to VALUE
  from the play-by-play rusher ids, then `elo_eval.py` rerun under its unchanged rule; the
  win-total preset fails loudly when the win totals cover none of the build's seasons, and the
  build metadata records the win-total file's origin and sha256.
- **Later, cheap measurements without walk-forwards**: the ported quarterback rating as the own
  Elo's quarterback adjustment (`elo_eval.py`); the source's pre-registered quarterback all-plays
  test and kneel-and-spike test, and its approved botched-snap rule (logged as departures from
  the published files).

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

- nflverse lists the 2026 Melbourne, Munich and Paris venues as domes (none has a roof over the
  field) and leaves Madrid blank; this feeds `stadium_type`. Recommendation: no override now,
  because the current season's schedule is downloaded again on every ETL run; recheck before the
  Paris game (2026-10-25) and add a roof override then if it is still wrong
  (`.agents/sos_port/decisions/pbp_columns.md`).

## Snapshots and timings

- Upstream inputs for the candidate build were copied into `~/scratch/step6/data/` on 2026-10-10
  at about 02:30 (main checkout's `data/` without backups, old cuts, `predict/`, `web/`); manifest
  `~/scratch/step6_data_snapshot_20261010.sha256` (715 files).
- A full ETL took 153 s on 2026-10-09; a GPU walk-forward of 2007-2025 from week 1 about 34
  minutes (`models/step4_reference/*/run.log`).
- Output location for the ported ratings files, shared by the ETL integration and the web pages:
  `data/ratings/{season}_{kind}.parquet`, the kinds named as in `nfl-sos-ratings`.
