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
  Week 3. The `0.13.1` defaults are now on `main`, and the reviewed `100`-tree plateau check
  tied `200`, so that default now stands. The next priority is task 54.0.
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

## Starting state (2026-09-21, after task 54.0 on `feat/m54-0-landing`)

- **The working branch is `feat/m54-0-landing`, unmerged, at version `0.14.0`.** It is based on
  `main` at `d6795ca`; `main` and `origin/main` still carry `0.13.1`. `scripts/gate.sh` exits
  `0` on this branch after the reviewed 54.0 doc sync. The shared `n_estimators` default remains
  `200`; `scripts/weekly_run.py` Stage 1 evaluates the shared production XGBoost defaults; and the
  shipped `config/weekly_run.yaml` keeps `tune: false`, `wf_include_postseason: false`,
  `include_postseason: false`, `wf_max_depth: 5` and `wf_learning_rate: 0.0165`.
- **Task 55.7 is closed.** The accepted `100`-tree rung completed under
  `models/wf_m55_7_2020_2025_trees100/` (checkpoints `models/wf_checkpoints/09a60441e87d86c894ba/`)
  and was independently rescored in `REVIEW.md` there against both the governing `200` rung and
  the ladder reference `598`. By the written rule, `100` ties `200`, so `200` stays the shared
  default and no new user decision is needed.
- **Task 54.0 is complete on this branch.** The parked commits `ae687dd` and `d34b0ab` were
  cherry-picked cleanly, the top-level CSVs were backed up to `data/backup_pre_m54_0/`, the ETL
  rebuild from cache logged the expected 9 coverage-gap warnings and 16 repair warnings, leakage
  audit `models/audit_m54_0_rebuild/leakage_audit.json` passed (`463` features, `0` flags), and
  the through-2025 cut `data/completed_games_ml.m54_0_through_2025.csv` is `e914eadf...`
  (`7261` rows, `513` columns). `data/completed_games_ml.csv` is now `db8b8ff4...`
  (`7292` completed rows, `513` columns).
- **No walk-forward is running.** The tree-budget ladder (task 55.7) is complete: four
  six-season rungs of the reference configuration on the rebuilt build, differing only in
  `--n-estimators`, each with a `HYPOTHESIS.md` written before its launch and an independent
  `REVIEW.md`: `models/wf_m55_7_2020_2025_trees598/` (checkpoints
  `models/wf_checkpoints/8431001f74f2766a8c44/`, the six-season reference, which reproduces the
  three-season reference arm's 2023-2025 folds bit for bit),
  `models/wf_m55_7_2020_2025_trees200/` (`a5e76d54187e27ca7370`) and
  `models/wf_m55_7_2020_2025_trees400/` (`849c353f7074fedb0538`), plus
  `models/wf_m55_7_2020_2025_trees100/` (`09a60441e87d86c894ba`). The numbers and the pairwise
  intervals are in `AGENTS.md` under "Tree-budget ladder" and in the `100` rung's `REVIEW.md`.
  The `1200` rung stays pre-written and unlaunched.
- **The 54.0 no-breakage arm is the current-build three-season reference.**
  `models/wf_m54_0_2023_2025_from_week1/` (checkpoints
  `models/wf_checkpoints/d112ebcba3115bafe9d9/`, with `HYPOTHESIS.md`, `compare_output.txt` and
  `REVIEW.md` in the run directory) tied the accepted `200`-tree reference slice
  `models/wf_checkpoints/a5e76d54187e27ca7370_2023_2025/` on the governing weeks 3-18 window:
  deterministic Brier `0.2097` vs `0.2090`, diff `+0.0007` `[-0.0011, +0.0024]`; margin MAE
  `9.9166` vs `9.9044`, diff `+0.0122` `[-0.0566, +0.0801]`. The rebuild moved 836 of 855 scored
  2023-2025 rows in at least one feature, chiefly in the `sos_*` and `opponent_*` EPA families,
  so future current-build arms compare against this run, not the earlier `0.12.6` reference.
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
  early stopping since `0.12.3`; the shared tree budget is now `200`. `--n-estimators`
  overrides it since `0.12.10`.
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
4. `.agents/ARCHIVE.md`, Milestone 54 (partial), then Milestone 59 ("Rebuild" and "Audit"),
   then Milestone 52.
5. `CHANGELOG.md` entries `0.12.0` to `0.13.1`.
6. `README.md` sections "Win probability calibration", "Backtesting" (including the superseded
   recency ablation note), "Validation".
7. `models/wf_m54_0_2023_2025_from_week1/HYPOTHESIS.md` and `REVIEW.md`, then
  `models/wf_m59_rebuild_2023_2025_from_week1_seed7/HYPOTHESIS.md` and `REVIEW.md` as the fit-
  noise-floor model.

## First check-in (before any code or run)

1. `git status --short | wc -l`, `git log --oneline -3`, `scripts/gate.sh --quick`. A dirty
   quick gate is the first task; ask about nothing else until it is clean.
2. `pgrep -af walk_forward`, `uptime`, and `ps -eo pcpu,args --sort=-pcpu | head -4` to see
   whether the web API or a weekly run is loading the machine.
3. Report to the user, in one message: the current branch state, that task 54.0 is complete on
  `feat/m54-0-landing`, the reviewed no-breakage result, and the plan for tasks 54.1 and 54.2.
  This is still a report unless the user explicitly asks about merge or push.

## Order of work (decided with the user on 2026-09-21)

Each item is one or more versioned chunks with its own changelog entry, gate run, commit on a
branch named for the task, handoff rewrite, and a check-in. New work starts on a fresh branch
off `main`; merging and pushing are must-ask, every time.

1. **Tasks 54.1 and 54.2**: the eight situational rates from the play-by-play counts (fix the
   `red_zone_tds` attribution first), then the box-score stats from play-by-play behind
   `--team-stats-source`.
2. **Task 55.8, season weighting**: half-lives of about `4`, `8` and `16` seasons via
   `--recency-half-life-seasons` as six-season arms against the unweighted reference at the
   `200` default on whichever build is then current. Replace the README's superseded ablation
   with the new measurement and its run directories whichever way it goes. A winning weighting
   is a default change: must-ask.
3. **Task 55.9, the Optuna re-tune**: after Milestone 54, ideally on a bye week or in the
   off-season. Its three prerequisites (no early stopping inside trials, a deterministic-Brier
   objective, and plumbing so a tuned set reaches the walk-forward) are each their own tested
   chunk before any trial runs. Details on the task.
4. **Milestone 60, the CLI inventory**: read-only, so it can run in parallel with any
   walk-forward as a subagent task. The removals land only after the user signs off.

Later, unchanged: Milestone 53 task 53.7, tasks 54.3 and 54.4, the rest of Milestone 55 and
Milestone 56, Milestone 58 phases 4-6 whenever the user asks, and Milestone 57 stays parked.

## Documentation debts to clear as you go

- `README.md`, "Backtesting": the recency ablation table is marked superseded; task 55.8
  replaces it with the new measurement. Still open. The duration line was corrected in
  `0.12.13` with the ladder's measured timings. Done.
- The `0.13.0` / `0.13.1` tree-budget closeout is done. Keep later docs aligned to `200` as the
  standing shared default unless a future measured default change lands.
- `AGENTS.md`, walk-forward operating notes: the `launch.sh` / `nohup setsid` convention and the
  load-driven `PASSIVE` relaunch are recorded (`345c398`, `0.12.10`). Done.
- `.agents/TODO.md`, "Current validated baseline": restated at `0.14.0`. Restate it again at
  each landed chunk so it does not lag.
- Any place that still says the reference is the `0.12.6` rebuild arm: on the current 54.0 build,
  later current-build arms compare against `models/wf_m54_0_2023_2025_from_week1/`; the older
  `models/wf_m59_rebuild_2023_2025_from_week1/` and the accepted six-season `200` rung remain the
  previous-build records.
- `CHANGELOG.md`: one entry per landed chunk, new version each time, never `[Unreleased]`.

## Open questions waiting on the user

**Immediate question if the user asks for it:** whether to merge and push `feat/m54-0-landing`
(`0.14.0`). That is must-ask every time. Otherwise none block continuing into 54.1 and 54.2.
What remains must-ask under guardrail rule 5 inside this work:

1. Merging any branch into `main`, and pushing: must-ask, every time, however small the chunk.
2. Task 54.0 has already spent the one approved rebuild. Any later rebuild, and any deletion or
  overwrite under `data/` or `models/`, is a fresh ask.
3. Any further rung of the closed tree-budget ladder (the pre-written `1200`, a second seed, or
  anything outside the accepted ladder as written) is a fresh ask.
4. Any future default change (`55.8`'s weighting, `55.9`'s tuned parameters, or a later revisit
  of tree budget) is a fresh ask with the numbers in hand.
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
