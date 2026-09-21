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
walk-forward runs per task before you ask; the must-ask list means stop and wait.

## Calendar

- Today's date is in your environment. The 2026 season is in progress. Week 2 ends with the
  Monday 2026-09-21 game; the Week 3 weekly run is due before the Thursday 2026-09-24 kickoff,
  and every later week follows the same rhythm (Thursday kickoff, Monday finish).
- The user runs `scripts/weekly_run.py` themselves; they confirmed this again on 2026-09-20 for
  Week 3. Measured 2026-09-20: with `--skip-data-refresh` it took 32 minutes (Stage 1, the
  walk-forward compare, 30 of them while the web API's reload watcher loaded the machine); the
  ETL adds about 10 minutes from the nflreadpy cache. Refresh the lines first (`python -m
  nfl_predictor.lines_refresh --season 2026 --week N`) when the dataset is fresh but the odds
  are hours old. It must not overlap any walk-forward: `pgrep -af walk_forward` and `uptime`
  before anything heavy.
- Walk-forward durations, one at a time: from week 1 over three eval seasons about 50 minutes
  idle, about 110 minutes when anything else (the web API with `--reload`, a browser session)
  loads the machine, in which case relaunch with `OMP_WAIT_POLICY=PASSIVE` and resume; six
  seasons about 100 minutes idle. Launch runs through a small `launch.sh` in the run directory
  with `nohup setsid`, never through a harness-bound shell (they have a 10-minute limit), and
  never `pkill -f` a pattern that matches your own shell. Any edit under `nfl_predictor/ml/`
  changes every checkpoint fingerprint; get the code stable before you measure.

## Starting state (2026-09-21, the tree-budget ladder complete)

- Branch `feat/m55-7-tree-budget`, version `0.12.13`. The `0.12.13` docs chunk (this file, the
  `AGENTS.md` ladder record, the `README.md` timings, the task 55.7 progress note, the
  changelog entry and the `pyproject.toml` bump) is uncommitted until the orchestrator commits
  it; everything before it is committed, the tip being `9b9a7c8`. The branch sits off
  `docs/handoff-m55-first`, which is one commit ahead of `main` at `7ea8e39` and is **not
  merged**; merging both branches into `main` and pushing is on the question list (must-ask).
  The last full `scripts/gate.sh --web` exited `0` on `ebb1f8e`: `862 passed`, coverage
  `92.66%`, frontend gate with `22` vitest tests.
- **The tree-budget ladder (task 55.7) is complete and no walk-forward is running.** Three
  six-season rungs of the reference configuration on the rebuilt build, differing only in
  `--n-estimators`, each with a `HYPOTHESIS.md` written before its launch and an independent
  `REVIEW.md`: `models/wf_m55_7_2020_2025_trees598/` (checkpoints
  `models/wf_checkpoints/8431001f74f2766a8c44/`, the six-season reference, which reproduces the
  three-season reference arm's 2023-2025 folds bit for bit),
  `models/wf_m55_7_2020_2025_trees200/` (`a5e76d54187e27ca7370`) and
  `models/wf_m55_7_2020_2025_trees400/` (`849c353f7074fedb0538`). The numbers and the pairwise
  intervals are in `AGENTS.md` under "Tree-budget ladder"; the ladder stopped on its "report all
  three and ask" branch, the default stays `598`, and the `1200` rung stays pre-written and
  unlaunched. Nothing about the default changes without the user.
- **A parked branch: `feat/m54-0-schedule-skeleton`**, two commits off `main` and not merged:
  `ae687dd feat(etl): build the team-game frame from the schedule skeleton` (task 54.0) and
  `d34b0ab fix(etl): null the box score of a team-stats row that covers both teams`. The repair
  came out of the audit of the skeleton: for all 16 JAX home games of 2001-2002 the surviving
  nflverse team-stats row carries both teams' production, and the skeleton alone would push
  that doubled box score into JAX's opponent mirrors. The repair alters feature values beyond
  the task text, so it needs the user's yes on its own. Both commits need the must-ask ETL
  rebuild (back up `data/*.csv` first), the leakage audit rerun and a three-season from-week-1
  tie check against the reference arm before anything lands.
- Data on disk: the 2026-09-20 04:31 rebuild on the `0.12.6` schema, `data/completed_games_ml.csv`
  `db6a78a3...` (`7278` rows, `513` columns), lines refreshed 2026-09-20 10:13 for Week 2. The
  walk-forward input for every new arm is the through-2025 cut
  `data/completed_games_ml.m59_through_2025.csv` (`cf42ec55...`, `7261` rows). The previous
  build is in `data/backup_pre_m59_rebuild/`. Leakage audit `models/audit_m59_rebuild/`: `463`
  features, `0` flags.
- **Reference arm on the rebuilt build**: `models/wf_m59_rebuild_2023_2025_from_week1/`
  (checkpoints `models/wf_checkpoints/34c17e508ab015a80662/`; table in `AGENTS.md` under
  "Reference arm on the 2026-09-20 rebuild"). **Fit-noise floor**: the same arm with
  `--random-seed 7`, `models/wf_m59_rebuild_2023_2025_from_week1_seed7/`; the paragraph beside
  the reference arm in `AGENTS.md` states what a three-season arm cannot resolve (Brier under
  about `0.002`, pick accuracy under about `0.01`, margin MAE under about `0.06`). The
  six-season floor has never been measured. Every run directory holds a `HYPOTHESIS.md`, a
  `REVIEW.md` and `compare_to_benchmark.py <candidate_ckpt> <reference_ckpt>`; reuse that
  script and that layout for every arm.
- What the model trains on: every walk-forward fold and the production fit train on every game
  strictly before the predicted week, all seasons from 1999 (a fold in 2025 trains on about
  `6,900` games across 27 seasons). `--eval-last-n-seasons` sets how many seasons are scored,
  not trained on. Every training row carries the same weight; recency weighting exists
  (`--recency-half-life-seasons`) and is off by default, which is what task 55.8 measures.
- The Week 2 package for the Sunday and Monday games is `models/weekly_2026_week_02_refresh/`
  (config `hybrid_raw_prob_base_elo_blend0.20_clamp0.10`, `elo` calibration, 15 games).
- `auto` calibration is the deterministic floor; production defaults to `elo`; no in-season
  early stopping since `0.12.3`; every head runs the `598`-tree budget, which the ladder above
  measured but did not change. `--n-estimators` overrides it since `0.12.10`.
- Machine state: no walk-forward running; check with `pgrep -af walk_forward` and `uptime`
  before anything heavy. The web API runs from the worktree `../nfl-predictor-web` on port 8765
  (the user starts it with `--reload`; its watcher takes half a core continuously); never
  restart it unasked. `../nfeloqb` and `../nfl-sos-ratings` are the user's and read-only.

## Read first, in this order

1. `AGENTS.md`: mission, the standing yardstick, the benchmark table, the reference arm and the
   fit-noise floor paragraphs, the delegation guardrails, the changelog and commit rules, the
   walk-forward operating notes.
2. `.agents/TODO.md`: the execution loop, the roadmap order (reordered 2026-09-20), Milestone 55
   tasks 55.7 and 55.8 in full, task 54.0, and the "From Milestone 59" follow-ups.
3. The tree-budget ladder, in this order:
   `models/wf_m55_7_2020_2025_trees598/HYPOTHESIS.md` (the ladder rule and stopping rule), then
   `models/wf_m55_7_2020_2025_trees598/REVIEW.md`,
   `models/wf_m55_7_2020_2025_trees200/REVIEW.md` and
   `models/wf_m55_7_2020_2025_trees400/REVIEW.md` (whose "Ladder summary" section is the source
   of the `AGENTS.md` tables).
4. `.agents/ARCHIVE.md`, Milestone 59 ("Rebuild" and "Audit"), then Milestone 52.
5. `CHANGELOG.md` entries `0.12.0` to `0.12.13`.
6. `README.md` sections "Win probability calibration", "Backtesting" (including the superseded
   recency ablation note), "Validation".
7. `models/wf_m59_rebuild_2023_2025_from_week1_seed7/HYPOTHESIS.md` and `REVIEW.md`, as the
   model for how a run is written up.

## First check-in (before any code or run)

1. `git status --short | wc -l`, `git log --oneline -3`, `scripts/gate.sh --quick`. A dirty
   quick gate is the first task; ask about nothing else until it is clean.
2. `pgrep -af walk_forward`, `uptime`, and `ps -eo pcpu,args --sort=-pcpu | head -4` to see
   whether the `598` rung, the web API or a weekly run is loading the machine.
3. Report to the user, in one message: the state of the `598` rung, what the ladder does next,
   and anything on the question list below. The user already said on 2026-09-20 that they run
   the Week 3 weekly run themselves and that the machine is free for walk-forward runs for 12
   hours or more, so do not re-ask either.

## Order of work (agreed with the user on 2026-09-20)

Each item is one or more versioned chunks with its own changelog entry, gate run, commit on a
branch named for the task, handoff rewrite, and a check-in. New work starts on a fresh branch
off `main`; merging and pushing are must-ask, every time.

1. **Task 55.7, the tree budget: ladder done, decision pending.** Three six-season rungs
   (`598`, `200`, `400`) ran and were reviewed; the record is in `AGENTS.md` under "Tree-budget
   ladder" and the progress note is on the task in `.agents/TODO.md`. The task stays `[ ]`
   because the outcome is a question, not a change: the default `n_estimators` stays `598`
   until the user picks. Do not launch a `100` rung, a second seed or the pre-written `1200`
   rung without being asked, and do not change the default, which is shared by walk-forward and
   production; mid-season the user may prefer a bye week for it.
2. **Task 55.8, season weighting: next, once the user answers on the budget.** Half-lives of
   about `4`, `8` and `16` seasons via `--recency-half-life-seasons` as six-season arms against
   the unweighted reference, same discipline. Replace the README's superseded ablation with the
   new measurement and its run directories whichever way it goes. A winning weighting is a
   default change: must-ask.
3. **Task 54.0** (schedule skeleton and coverage check): **code parked** on
   `feat/m54-0-schedule-skeleton` (`ae687dd`, plus the both-teams box-score repair `d34b0ab`,
   which needs its own yes). It changes feature values at ETL time, so it needs a rebuild
   (must-ask, back up `data/*.csv` to `data/backup_pre_<tag>/` first, rerun the leakage audit,
   cut a through-2025 copy) and one three-season from-week-1 arm against the reference arm,
   read as a no-breakage check: every margin will move (training rows changed), and a tie is
   the only claimable outcome.
4. **Milestone 56** (weekly orchestration residuals): 56.1 data-refresh pass-through **done**
   in `0.12.11`; 56.2 postseason handling **narrowed** in the same version, waiting on the
   user's default (clamp plus README;
   the default is the user's call); 56.3 still waits on the sweep. The "From task 56.4"
   follow-up about early stopping on the calibration frame is **closed** in `.agents/TODO.md`:
   `0.12.3` removed in-season early stopping, and the remaining tree-budget question is 55.7.
5. **Task 58.4** (web housekeeping): **narrowed** in `0.12.12`. The `web` extra is dropped and
   the three job templates pass `--data-dir`; the ETL's upstream input paths are the open
   remainder, and whether to chase them is a question.
6. **The rest of Milestone 55** (sweep schema and runner, `PRIOR_BLEND_GAMES`, `xgb_device`,
   `ScoreModel`, the stability view), including the out-of-fold calibration pool ("From
   Milestone 59") as an arm if a fitted calibrator is reconsidered at all.
7. **Milestone 54, tasks 54.1 to 54.4**, then **task 53.7**: feature work, six seasons per arm.
8. **Milestone 58 phases 4 to 6** only when the user asks; **Milestone 57** stays parked.

## Documentation debts to clear as you go

- `README.md`, "Backtesting": the recency ablation table is marked superseded; task 55.8
  replaces it with the new measurement. Still open. The duration line was corrected in
  `0.12.13` with the ladder's measured timings. Done.
- `AGENTS.md`, walk-forward operating notes: the `launch.sh` / `nohup setsid` convention and the
  load-driven `PASSIVE` relaunch are recorded (`345c398`, `0.12.10`). Done.
- `.agents/TODO.md`, "Current validated baseline": restated at `0.12.13`, with the gate
  result pointing at the session check-in until the orchestrator's gate lands. Restate it
  again at each landed chunk so it does not lag.
- Any place that still says the benchmark is "the" reference: on the rebuilt build the
  reference is `models/wf_m59_rebuild_2023_2025_from_week1/`, and the `AGENTS.md` benchmark
  table is the record of the previous build.
- `CHANGELOG.md`: one entry per landed chunk, new version each time, never `[Unreleased]`.

## Open questions waiting on the user

1. Merge `docs/handoff-m55-first` and `feat/m55-7-tree-budget` into `main` and push (must-ask).
2. The tree budget: keep `598`, adopt `200`, or run `100` and/or a second seed on `200` first?
   Weeks 3-18 over six seasons, deterministic Brier `0.2103` (`200`) / `0.2111` (`400`) /
   `0.2117` (`598`) against market `0.2095`, margin MAE `9.9281` / `9.9978` / `10.0240`, the
   `200 - 598` Brier interval `[-0.0029, +0.0001]` covering zero by `+0.0000791` and its margin
   MAE `-0.0958` `[-0.1583, -0.0345]` beyond the fit-noise floor. Also: should any default
   change wait for a bye week, since the default is shared with production?
3. Two procedural deviations of the ladder session, for the record. The `400` rung was the third
   walk-forward on task 55.7, which the guardrails list as must-ask; it was launched on the
   user's "machine is free, go wild" authorization. And it was launched reading the all-weeks
   interval and margin MAE, while the `200` rung's own decision rule, stated on weeks 3-18, read
   as a tie.
4. Task 54.0: rebuild with the schedule skeleton alone (`ae687dd`), with the both-teams
   box-score repair too (`d34b0ab`), or not yet?
5. Task 56.2: should postseason games be included in weekly runs by default? The code defaults
   exclude them, `config/weekly_run.yaml` includes them at weight `1.3`, and one of the two has
   to change.
6. Task 58.4: is the remainder wanted at all, that is, should the ETL's upstream inputs
   (`qb_elos.csv`, the QB identity file, the TeamRankings and nflreadpy caches) also follow
   `NFLP_DATA_DIR`, or is the configured tree only meant for the datasets the jobs write?
7. From the 54.0 audit: `blend_with_prior_stats` weights box-score columns by the count of
   scheduled games while their means rest only on the games actually covered (JAX 2001-2002:
   `8` of `16`). Is that weighting intended, or should the blend use the covered-game count?

## How each chunk runs

- TDD: characterization or failing tests first for every production line you touch; no new
  `noqa`, `type: ignore` or `pragma: no cover` without a reason in the code.
- One versioned changelog entry per landed chunk, `pyproject.toml` to the same version, `uv
  lock`, `uv sync`. Never a tag.
- `scripts/gate.sh` exits `0` on the final tree before the chunk is reported done; markdownlint
  covers the `.md` notes under `models/` too, so keep `HYPOTHESIS.md` and `REVIEW.md` clean.
- Walk-forward runs: `HYPOTHESIS.md` with the decision rule and the exact command written
  before the launch; a `launch.sh`; one run at a time; a reviewer subagent rescores from the
  checkpoints and writes `REVIEW.md` before any number reaches the docs; the check-in after
  every run names the directory, the deterministic and market numbers, the floor comparison and
  the decision.
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
