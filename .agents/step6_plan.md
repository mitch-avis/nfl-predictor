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
| S2 new nflverse sources, `MODELS_PATH` | 2 | none | to launch |
| S3 own Elo (538 framework, 2019 additions, QB value, win totals) | 4 candidate | none | to launch |
| S4 ratings core port (ridge, team, prior, sos, QB, histories, ranges) | 3 | none for the algorithms | to launch |
| S8a web: pool helpers, margin toggle, live-betting design | 6 (58.1, 58.6, 58.3) | none | to launch |
| S5 ETL integration: play-by-play fixes, ratings inputs and outputs, pre-week features, follow-ups | 3-4 | S1, S2, S4 | after those merge |
| S6 candidate build and measurement | 4 | S3, S5 | after S5 |
| S7 power rankings and projected standings | 5 | S5, S6 | after S6 |
| S8b web: team, quarterback, schedule, rank-range, head-to-head pages (58.2) | 6 | S5 output schema | after S5 |

## Measurement ladder (phase 4; written before any run, cap 8 runs plus one build)

Arms on one candidate build, 2007-2025 from week 1, seeds 42 and 7, the reference's command
otherwise (see `models/step4_reference/driver.sh`):

1. Ported ratings anchored against the adopted-settings reference (`models/step4_reference/
   adopted_seed*`).
2. Ported ratings market-free (no betting-line, no Elo columns) against the test bench
   (`models/step4_reference/marketfree_all_seed*`).
3. Own Elo anchored (replacing the `qb_elos.csv` columns) against arm 1.
4. Own Elo market-free (without win totals) against arm 2.

The decision rules are written into `models/step6/HYPOTHESIS.md` before launch.
