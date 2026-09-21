# Next Agent Session Prompt

You are the orchestrating agent for a session in the `nfl-predictor` workspace
(`/home/mitch/workspace/nfl-predictor`). The user has delegated the remaining roadmap to a
sequence of agent sessions like this one. Your job is to move the project toward one goal, one
task at a time, and to stop for the user whenever a decision is theirs.

**The goal.** The user wants to use this project every week of the regular season to make picks
and bets on that week's games, and to trust what it produces. Every task is judged against that:
does it make the weekly run more correct, more reproducible, more honest about its uncertainty,
or easier to operate. Work that cannot be measured on the walk-forward instrument, or that adds
operational risk during the season, ranks below operational work.

**Read `AGENTS.md` first, all of it, and treat its "Delegation guardrails" section as binding.**
The short form: `scripts/gate.sh` decides "done"; narrowing is never a checkbox; a number goes
into the docs only after a separate rescore names its run directory (the two-key rule); two
walk-forward runs per task before you ask, unless they are rungs of an accepted ladder; the
must-ask list means stop and wait.

## Calendar

- Today's date is in your environment. The 2026 season is in progress. Week 2 ended with the
  Monday 2026-09-21 game; the Week 3 weekly run is due before the Thursday 2026-09-24 kickoff,
  and every later week follows the same rhythm (Thursday kickoff, Monday finish).
- The user runs `scripts/weekly_run.py` themselves; they confirmed this again on 2026-09-20 for
  Week 3. The next weekly run should use the `0.13.0` defaults once they land, so land the task
  55.7 and 56.2 config chunk before Thursday if you can, and tell the user when it is in.
  Measured 2026-09-20: with `--skip-data-refresh` the weekly run took 32 minutes (Stage 1, the
  walk-forward compare, 30 of them while the web API's reload watcher loaded the machine); the
  ETL adds about 10 minutes from the nflreadpy cache. Refresh the lines first (`python -m
  nfl_predictor.lines_refresh --season 2026 --week N`) when the dataset is fresh but the odds
  are hours old. It must not overlap any walk-forward: `pgrep -af walk_forward` and `uptime`
  before anything heavy.
- Walk-forward durations, one at a time: from week 1 over three eval seasons about 50 minutes
  idle, about 110 minutes when anything else (the web API with `--reload`, a browser session)
  loads the machine, in which case relaunch with `OMP_WAIT_POLICY=PASSIVE` and resume; six
  seasons about 100 minutes idle at `200` trees and about 4.8 hours at `598` under load. Launch
  runs through a small `launch.sh` in the run directory with `nohup setsid`, never through a
  harness-bound shell (they have a 10-minute limit), and never `pkill -f` a pattern that matches
  your own shell. Any edit under `nfl_predictor/ml/` changes every checkpoint fingerprint; get
  the code stable before you measure.

## Starting state (2026-09-21, the user's answers in hand)

- **`main` is at `3729006` with everything through `0.12.14` merged and pushed.** The current
  branch is `docs/handoff-next-steps`, off that commit, carrying the documentation-only
  `0.12.15` chunk (this file, `.agents/TODO.md`, `CHANGELOG.md`, `pyproject.toml`, `uv.lock`),
  which is uncommitted until the orchestrator commits it. No code changed in it. The last full
  `scripts/gate.sh` exited `0` on the `0.12.13` tree (`862 passed`, coverage `92.68%`); the last
  `--web` run was `862 passed`, coverage `92.66%`, on `ebb1f8e`.
- **The user answered every open question on 2026-09-21, so your first check-in is a report, not
  a question.** The decisions are recorded on the tasks in `.agents/TODO.md` and summarized in
  the `0.12.15` changelog entry: adopt `200` trees as the shared default and align
  `config/weekly_run.yaml` with it; turn the weekly run's `tune` off; set the postseason flags
  to `false`; proceed with task 54.0 including the box-score repair and its rebuild; add task
  55.9, the Optuna re-tune, after Milestone 54; and the order of work below.
- **No walk-forward is running.** The tree-budget ladder (task 55.7) is complete: three
  six-season rungs of the reference configuration on the rebuilt build, differing only in
  `--n-estimators`, each with a `HYPOTHESIS.md` written before its launch and an independent
  `REVIEW.md`: `models/wf_m55_7_2020_2025_trees598/` (checkpoints
  `models/wf_checkpoints/8431001f74f2766a8c44/`, the six-season reference, which reproduces the
  three-season reference arm's 2023-2025 folds bit for bit),
  `models/wf_m55_7_2020_2025_trees200/` (`a5e76d54187e27ca7370`) and
  `models/wf_m55_7_2020_2025_trees400/` (`849c353f7074fedb0538`). The numbers and the pairwise
  intervals are in `AGENTS.md` under "Tree-budget ladder". The `1200` rung stays pre-written and
  unlaunched; the `100` rung is the one still to run, as the plateau check.
- **The parked branch `feat/m54-0-schedule-skeleton`** is two commits off `main` and not merged:
  `ae687dd feat(etl): build the team-game frame from the schedule skeleton` (task 54.0) and
  `d34b0ab fix(etl): null the box score of a team-stats row that covers both teams`. The repair
  came out of the audit of the skeleton: for all 16 JAX home games of 2001-2002 the surviving
  nflverse team-stats row carries both teams' production, and the skeleton alone would push that
  doubled box score into JAX's opponent mirrors. The user approved both commits and the rebuild
  that follows them; the details are on task 54.0 in `.agents/TODO.md`.
- Data on disk: the 2026-09-20 04:31 rebuild on the `0.12.6` schema, `data/completed_games_ml.csv`
  `db6a78a3...` (`7278` rows, `513` columns), lines refreshed 2026-09-20 10:13 for Week 2. The
  walk-forward input for every arm on this build is the through-2025 cut
  `data/completed_games_ml.m59_through_2025.csv` (`cf42ec55...`, `7261` rows). The previous
  build is in `data/backup_pre_m59_rebuild/`. Leakage audit `models/audit_m59_rebuild/`: `463`
  features, `0` flags.
- **Reference arms.** On the rebuilt build at `598` trees:
  `models/wf_m59_rebuild_2023_2025_from_week1/` (three seasons; table in `AGENTS.md` under
  "Reference arm on the 2026-09-20 rebuild") and `models/wf_m55_7_2020_2025_trees598/` (six
  seasons). Once `200` is the default, the comparable references are
  `models/wf_m55_7_2020_2025_trees200/` and its 2023-2025 folds. **Fit-noise floor**: the
  reference arm rerun with `--random-seed 7`,
  `models/wf_m59_rebuild_2023_2025_from_week1_seed7/`; the paragraph beside it in `AGENTS.md`
  states what a three-season arm cannot resolve (Brier under about `0.002`, pick accuracy under
  about `0.01`, margin MAE under about `0.06`). The six-season floor has never been measured.
  Every run directory holds a `HYPOTHESIS.md`, a `REVIEW.md` and
  `compare_to_benchmark.py <candidate_ckpt> <reference_ckpt>`; reuse that script and that
  layout for every arm.
- What the model trains on: every walk-forward fold and the production fit train on every game
  strictly before the predicted week, all seasons from 1999 (a fold in 2025 trains on about
  `6,900` games across 27 seasons). `--eval-last-n-seasons` sets how many seasons are scored,
  not trained on. Every training row carries the same weight; recency weighting exists
  (`--recency-half-life-seasons`) and is off by default, which is what task 55.8 measures.
- The Week 2 package for the Sunday and Monday games is `models/weekly_2026_week_02_refresh/`
  (config `hybrid_raw_prob_base_elo_blend0.20_clamp0.10`, `elo` calibration, 15 games).
- `auto` calibration is the deterministic floor; production defaults to `elo`; no in-season
  early stopping since `0.12.3`; the tree budget is `598` until the `0.13.0` chunk moves it to
  `200`. `--n-estimators` overrides it since `0.12.10`.
- Machine state: no walk-forward running; check with `pgrep -af walk_forward` and `uptime`
  before anything heavy. The web API runs from the worktree `../nfl-predictor-web` on port 8765
  (the user starts it with `--reload`; its watcher takes half a core continuously and does load
  the machine); never restart it unasked. `../nfeloqb` and `../nfl-sos-ratings` are the user's
  and read-only.

## Read first, in this order

1. `AGENTS.md`: mission, the standing yardstick, the benchmark table, the reference arm and the
   fit-noise floor paragraphs, the "Tree-budget ladder" paragraph, the delegation guardrails,
   the changelog and commit rules, the walk-forward operating notes.
2. `.agents/TODO.md`: the execution loop, the roadmap order (set 2026-09-21), Milestone 55 tasks
   55.7, 55.8 and 55.9 in full, Milestone 54 tasks 54.0 to 54.2, task 56.2, Milestone 60, and
   the "From Milestone 59" follow-ups.
3. The tree-budget ladder, in this order:
   `models/wf_m55_7_2020_2025_trees598/HYPOTHESIS.md` (the ladder rule and stopping rule), then
   the three reviews `models/wf_m55_7_2020_2025_trees598/REVIEW.md`,
   `models/wf_m55_7_2020_2025_trees200/REVIEW.md` and
   `models/wf_m55_7_2020_2025_trees400/REVIEW.md`, whose "Ladder summary" section is the source
   of the `AGENTS.md` tables.
4. `.agents/ARCHIVE.md`, Milestone 59 ("Rebuild" and "Audit"), then Milestone 52.
5. `CHANGELOG.md` entries `0.12.0` to `0.12.15`.
6. `README.md` sections "Win probability calibration", "Backtesting" (including the superseded
   recency ablation note), "Validation".
7. `models/wf_m59_rebuild_2023_2025_from_week1_seed7/HYPOTHESIS.md` and `REVIEW.md`, as the
   model for how a run is written up.

## First check-in (before any code or run)

1. `git status --short | wc -l`, `git log --oneline -3`, `scripts/gate.sh --quick`. A dirty
   quick gate is the first task; ask about nothing else until it is clean.
2. `pgrep -af walk_forward`, `uptime`, and `ps -eo pcpu,args --sort=-pcpu | head -4` to see
   whether the web API or a weekly run is loading the machine.
3. Report to the user, in one message: the state of the tree, the plan for the `0.13.0` chunk
   and whether it can land before Thursday's kickoff, and the `100` rung you intend to launch.
   This is a report, not a question: the user answered everything on 2026-09-21. They also said
   on 2026-09-20 that they run the Week 3 weekly run themselves and that the machine is free for
   walk-forward runs for 12 hours or more, so do not re-ask either.

## Order of work (decided with the user on 2026-09-21)

Each item is one or more versioned chunks with its own changelog entry, gate run, commit on a
branch named for the task, handoff rewrite, and a check-in. New work starts on a fresh branch
off `main`; merging and pushing are must-ask, every time.

1. **Task 55.7 close-out, version `0.13.0` (a minor bump: it is a default change).** Adopt `200`
   as the shared default `n_estimators` in walk-forward and production together. In the same
   chunk make `config/weekly_run.yaml` consistent: set `tune: false` (today's `tune: true`
   re-runs a one-hour Optuna study on every weekly run, which the user did not intend) and align
   the walk-forward stage's XGBoost params (`wf_n_estimators`, `wf_max_depth`,
   `wf_learning_rate`) with the production defaults, so both stages fit the same model. Then run
   the `100` rung as the plateau check: six seasons, the same ladder configuration, a
   `HYPOTHESIS.md` naming its windows and columns, an independent reviewer rescore. If `100`
   ties `200`, keep `200`; if `100` is better beyond the fit-noise floor, report it and ask.
2. **Task 56.2 config flags.** Set `include_postseason: false` and `wf_include_postseason: false`
   in `config/weekly_run.yaml`, leaving `postseason_weight` in place but inert. It can share the
   `0.13.0` chunk with item 1, since both are config defaults.
3. **Task 54.0.** Merge `feat/m54-0-schedule-skeleton` (both commits, the repair included) into
   the working branch, changelog entry, gate; then the approved rebuild (back up `data/*.csv` to
   `data/backup_pre_m54_0/`, rerun the ETL from the cache, rerun the leakage audit, cut
   `data/completed_games_ml.m54_0_through_2025.csv`); then one three-season from-week-1 arm at
   the `200` default against the 2023-2025 folds of `models/wf_m55_7_2020_2025_trees200/`, read
   as a no-breakage tie check. Expect 9 coverage-gap warnings and 16 repair warnings from the
   ETL.
4. **Tasks 54.1 and 54.2**: the eight situational rates from the play-by-play counts (fix the
   `red_zone_tds` attribution first), then the box-score stats from play-by-play behind
   `--team-stats-source`.
5. **Task 55.8, season weighting**: half-lives of about `4`, `8` and `16` seasons via
   `--recency-half-life-seasons` as six-season arms against the unweighted reference at the
   `200` default on whichever build is then current. Replace the README's superseded ablation
   with the new measurement and its run directories whichever way it goes. A winning weighting
   is a default change: must-ask.
6. **Task 55.9, the Optuna re-tune**: after Milestone 54, ideally on a bye week or in the
   off-season. Its three prerequisites (no early stopping inside trials, a deterministic-Brier
   objective, and plumbing so a tuned set reaches the walk-forward) are each their own tested
   chunk before any trial runs. Details on the task.
7. **Milestone 60, the CLI inventory**: read-only, so it can run in parallel with any
   walk-forward as a subagent task. The removals land only after the user signs off.

Later, unchanged: Milestone 53 task 53.7, tasks 54.3 and 54.4, the rest of Milestone 55 and
Milestone 56, Milestone 58 phases 4-6 whenever the user asks, and Milestone 57 stays parked.

## Documentation debts to clear as you go

- `README.md`, "Backtesting": the recency ablation table is marked superseded; task 55.8
  replaces it with the new measurement. Still open. The duration line was corrected in
  `0.12.13` with the ladder's measured timings. Done.
- `README.md` and `AGENTS.md` both name `598` as the tree budget in several places; the `0.13.0`
  chunk has to move every one of them to `200` along with the code default.
- `AGENTS.md`, walk-forward operating notes: the `launch.sh` / `nohup setsid` convention and the
  load-driven `PASSIVE` relaunch are recorded (`345c398`, `0.12.10`). Done.
- `.agents/TODO.md`, "Current validated baseline": restated at `0.12.15`. Restate it again at
  each landed chunk so it does not lag.
- Any place that still says the benchmark is "the" reference: on the rebuilt build the
  reference is `models/wf_m59_rebuild_2023_2025_from_week1/` at `598` trees and
  `models/wf_m55_7_2020_2025_trees200/` at the new default, and the `AGENTS.md` benchmark table
  is the record of the previous build.
- `CHANGELOG.md`: one entry per landed chunk, new version each time, never `[Unreleased]`.

## Open questions waiting on the user

**None blocking.** The user answered all seven open questions on 2026-09-21; the answers are on
the tasks in `.agents/TODO.md` and in the `0.12.15` changelog entry. What remains must-ask under
guardrail rule 5 inside this work:

1. Merging any branch into `main`, and pushing: must-ask, every time, however small the chunk.
2. The ETL rebuild is pre-approved **for task 54.0 only**, with the backup to
   `data/backup_pre_m54_0/` first. Any other rebuild, and any deletion or overwrite under
   `data/` or `models/`, is a fresh ask.
3. The `100` rung is an accepted rung of the tree-budget ladder, so it needs no new approval.
   Any further rung (the pre-written `1200`, a second seed, anything outside the ladder as
   written) does.
4. Moving the default away from `200` on the strength of the `100` rung, or any other default
   change (55.8's weighting, 55.9's tuned parameters), is a fresh ask with the numbers in hand.
5. Anything else on the rule 5 list: reopening Milestone 57, reordering the roadmap, touching
   `../nfeloqb`, `../nfl-sos-ratings` or the web API on port 8765.

## How each chunk runs

- TDD: characterization or failing tests first for every production line you touch; no new
  `noqa`, `type: ignore` or `pragma: no cover` without a reason in the code.
- One versioned changelog entry per landed chunk, `pyproject.toml` to the same version, `uv
  lock`, `uv sync`. Never a tag.
- `scripts/gate.sh` exits `0` on the final tree before the chunk is reported done; markdownlint
  covers the `.md` notes under `models/` too, so keep `HYPOTHESIS.md` and `REVIEW.md` clean.
- Walk-forward runs: `HYPOTHESIS.md` with the decision rule, the exact windows and columns it
  reads, and the exact command, written before the launch; a `launch.sh`; one run at a time; a
  reviewer subagent rescores from the checkpoints and writes `REVIEW.md` before any number
  reaches the docs; the check-in after every run names the directory, the deterministic and
  market numbers, the floor comparison and the decision.
- Subagents for read-only parallel work and mechanical edits only; never over anything on the
  must-ask list.
- Rewrite this file at every landed chunk: branch, version, uncommitted state, the next task,
  open questions.

## Final report for a session

1. What landed, by version, with the commit subjects.
2. Every walk-forward run started: directory, hypothesis, result on the deterministic and
   market columns against the fit-noise floor, and the decision it produced.
3. Anything in this file or in `AGENTS.md` that turned out to be wrong.
4. The state of the tree (branch, version, uncommitted files) and the next task, also written
   into this file.
5. The questions waiting on the user.
