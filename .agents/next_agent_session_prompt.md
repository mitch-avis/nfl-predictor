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
- The user runs `scripts/weekly_run.py` themselves unless they ask you to. Measured 2026-09-20:
  with `--skip-data-refresh` it took 32 minutes (Stage 1, the walk-forward compare, 30 of them
  while the web API's reload watcher loaded the machine); the ETL adds about 10 minutes from the
  nflreadpy cache. Refresh the lines first (`python -m nfl_predictor.lines_refresh --season
  2026 --week N`) when the dataset is fresh but the odds are hours old. It must not overlap any
  walk-forward: `pgrep -af walk_forward` and `uptime` before anything heavy.
- Walk-forward durations, one at a time: from week 1 over three eval seasons about 50 minutes
  idle, about 110 minutes when anything else (the web API with `--reload`, a browser session)
  loads the machine, in which case relaunch with `OMP_WAIT_POLICY=PASSIVE` and resume; six
  seasons about 100 minutes idle. Launch runs through a small `launch.sh` in the run directory
  with `nohup setsid`, never through a harness-bound shell (they have a 10-minute limit), and
  never `pkill -f` a pattern that matches your own shell. Any edit under `nfl_predictor/ml/`
  changes every checkpoint fingerprint; get the code stable before you measure.

## Starting state (2026-09-20, end of the rebuild session)

- `main` is at the merge of `docs/m59-noise-floor` (`c1fff6a`) plus whatever landed the
  `0.12.9` handoff (this file); version `0.12.9`. Everything through `0.12.8` is merged and
  pushed. Check `git status --short | wc -l` and `git log --oneline -3` first; if this file's
  branch is not yet merged, say so in the check-in.
- Data on disk: the 2026-09-20 04:31 rebuild on the `0.12.6` schema, `data/completed_games_ml.csv`
  `db6a78a3...` (`7278` rows, `513` columns), lines refreshed at 10:13 for Week 2. The
  walk-forward input for every new arm is the through-2025 cut
  `data/completed_games_ml.m59_through_2025.csv` (`cf42ec55...`, `7261` rows). The previous
  build is in `data/backup_pre_m59_rebuild/`. Leakage audit `models/audit_m59_rebuild/`: `463`
  features, `0` flags.
- **Reference arm on the rebuilt build**: `models/wf_m59_rebuild_2023_2025_from_week1/`
  (checkpoints `models/wf_checkpoints/34c17e508ab015a80662/`; table in `AGENTS.md` under
  "Reference arm on the 2026-09-20 rebuild"). **Fit-noise floor**: the same arm with
  `--random-seed 7`, `models/wf_m59_rebuild_2023_2025_from_week1_seed7/`; the paragraph beside
  the reference arm in `AGENTS.md` states what a three-season arm cannot resolve (Brier under
  about `0.002`, pick accuracy under about `0.01`, margin MAE under about `0.06`). Every run
  directory holds a `HYPOTHESIS.md` (rule and command, written before the run), a `REVIEW.md`
  (the independent rescore) and `compare_to_benchmark.py <candidate_ckpt> <reference_ckpt>`,
  which rescores two checkpoint directories and is validated against the `AGENTS.md` table.
  Reuse that script and that layout for every arm.
- What the model trains on: every walk-forward fold and the production fit train on every game
  strictly before the predicted week, all seasons from 1999 (a fold in 2025 trains on about
  `6,900` games across 27 seasons). `--eval-last-n-seasons` sets how many seasons are scored,
  not trained on. Every training row carries the same weight; recency weighting exists
  (`--recency-half-life-seasons`) and is off by default.
- The Week 2 package for the Sunday and Monday games is `models/weekly_2026_week_02_refresh/`
  (config `hybrid_raw_prob_base_elo_blend0.20_clamp0.10`, `elo` calibration, 15 games).
- `auto` calibration is the deterministic floor; production defaults to `elo`; no in-season
  early stopping since `0.12.3`; every head runs the untuned `598`-tree budget (task 55.7).
- The web API runs from the worktree `../nfl-predictor-web` on port 8765 (the user starts it
  with `--reload`; its watcher takes half a core continuously); never restart it unasked.
  `../nfeloqb` and `../nfl-sos-ratings` are the user's and read-only.

## Read first, in this order

1. `AGENTS.md`: mission, the standing yardstick, the benchmark table, the reference arm and the
   fit-noise floor paragraphs, the delegation guardrails, the changelog and commit rules, the
   walk-forward operating notes.
2. `.agents/TODO.md`: the execution loop, the roadmap order (reordered 2026-09-20), Milestone 55
   tasks 55.7 and 55.8 in full, task 54.0, and the "From Milestone 59" follow-ups.
3. `.agents/ARCHIVE.md`, Milestone 59 ("Rebuild" and "Audit"), then Milestone 52.
4. `CHANGELOG.md` entries `0.12.0` to `0.12.9`.
5. `README.md` sections "Win probability calibration", "Backtesting" (including the superseded
   recency ablation note), "Validation".
6. `models/wf_m59_rebuild_2023_2025_from_week1_seed7/HYPOTHESIS.md` and `REVIEW.md`, as the
   model for how a run is written up.

## First check-in (before any code or run)

1. `git status --short | wc -l`, `git log --oneline -3`, `scripts/gate.sh --quick`. A dirty
   quick gate is the first task; ask about nothing else until it is clean.
2. `pgrep -af walk_forward`, `uptime`, and `ps -eo pcpu,args --sort=-pcpu | head -4` to see
   whether the web API or a weekly run is loading the machine.
3. Confirm with the user, in one message: the Week 3 weekly run (they said they will likely run
   it themselves; ask whether that still holds and whether the machine is free for a six-season
   walk-forward now), and the order of work below.

## Order of work (agreed with the user on 2026-09-20)

Each item is one or more versioned chunks with its own changelog entry, gate run, commit on a
branch named for the task, handoff rewrite, and a check-in. New work starts on a fresh branch
off `main`; merging and pushing are must-ask, every time.

1. **Task 55.7, the tree budget.** `598` trees is an old Optuna value that was never measured,
   and every fit now runs the full budget by construction. Design before running: a ladder of
   budgets (for example `200`, `400`, `598`, `800`, `1200`) as separate six-season arms of the
   reference configuration on the rebuilt build (`--eval-last-n-seasons 6 --wf-start-week 1
   --calibration auto --wf-calibration-weeks 4 --market-anchor --market-transform`, plus the
   budget flag), one `HYPOTHESIS.md` per arm with the decision rule written against the
   fit-noise floor, one arm at a time, a reviewer rescore of each before any number enters the
   docs. Six seasons cost about 100 minutes idle each; write the whole ladder's plan and the
   stopping rule in the check-in before the first launch, and stop to ask after two arms if no
   decision has emerged (guardrail 4). The winner becomes the shared default in walk-forward and
   production together, which is a default change: must-ask before changing it, and mid-season
   the user may prefer to wait for a bye week or the off-season.
2. **Task 55.8, season weighting.** Half-lives of about `4`, `8` and `16` seasons via
   `--recency-half-life-seasons` as six-season arms against the unweighted reference, same
   discipline. Replace the README's superseded ablation with the new measurement and its run
   directories whichever way it goes. A winning weighting is a default change: must-ask.
3. **Task 54.0** (schedule skeleton and coverage check): small, data-correctness, JAX
   2001-2002. It changes feature values at ETL time, so it needs a rebuild (must-ask, back up
   `data/*.csv` to `data/backup_pre_<tag>/` first, rerun the leakage audit, cut a through-2025
   copy) and one three-season from-week-1 arm against the reference arm, read as a no-breakage
   check: every margin will move (training rows changed), and a tie is the only claimable
   outcome.
4. **Milestone 56** (weekly orchestration residuals): 56.1 data-refresh pass-through, 56.2
   postseason handling, then 56.3 after the sweep. Also close or rewrite the "From task 56.4"
   follow-up about early stopping on the calibration frame: superseded by `0.12.3`, remainder
   is 55.7.
5. **Task 58.4** (web housekeeping): the `web` extra duplication and the job templates that
   ignore `NFLP_DATA_DIR`; run `scripts/gate.sh --web`.
6. **The rest of Milestone 55** (sweep schema and runner, `PRIOR_BLEND_GAMES`, `xgb_device`,
   `ScoreModel`, the stability view), including the out-of-fold calibration pool ("From
   Milestone 59") as an arm if a fitted calibrator is reconsidered at all.
7. **Milestone 54, tasks 54.1 to 54.4**, then **task 53.7**: feature work, six seasons per arm.
8. **Milestone 58 phases 4 to 6** only when the user asks; **Milestone 57** stays parked.

## Documentation debts to clear as you go

- `README.md`, "Backtesting": the recency ablation table is marked superseded; task 55.8
  replaces it with the new measurement. The "about 75 minutes on a 24-core machine" duration
  line predates the load findings above; correct it when 55.7's runs give fresh timings.
- `AGENTS.md`, walk-forward operating notes: add the `launch.sh` / `nohup setsid` convention
  and the load-driven `PASSIVE` relaunch observed on 2026-09-20 if they are not there yet.
- `.agents/TODO.md`, "Current validated baseline": restate the version and gate counts at each
  landed chunk (they lag by one version at the moment).
- Any place that still says the benchmark is "the" reference: on the rebuilt build the
  reference is `models/wf_m59_rebuild_2023_2025_from_week1/`, and the `AGENTS.md` benchmark
  table is the record of the previous build.
- `CHANGELOG.md`: one entry per landed chunk, new version each time, never `[Unreleased]`.

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
