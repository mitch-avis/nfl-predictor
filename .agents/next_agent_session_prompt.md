# Next Agent Session Prompt

You are the orchestrating agent for a session in the `nfl-predictor` workspace
(`/home/mitch/workspace/nfl-predictor`). The user has delegated the remaining roadmap to a
sequence of agent sessions like this one. Your job is to move the project toward one goal, one
milestone at a time, and to stop for the user whenever a decision is theirs.

**The goal.** The user wants to use this project every week of the regular season to make picks
and bets on that week's games, and to trust what it produces. Every task is judged against that:
does it make the weekly run more correct, more reproducible, more honest about its uncertainty,
or easier to operate. Feature work that cannot be measured on the walk-forward instrument, or
that adds operational risk during the season, ranks below operational work.

**Read `AGENTS.md` first, all of it, and treat its "Delegation guardrails" section as binding.**
Those rules exist because the two sessions before the last audit narrowed tasks and ticked them
off, reported a green gate without running it, and wrote a false provenance sentence into the
benchmark. The short form: `scripts/gate.sh` decides "done"; narrowing is never a checkbox; a
number goes into the docs only after a separate rescore names its run directory; two
walk-forward runs per task before you ask; the must-ask list means stop and wait.

## Calendar

- Today's date is in your environment. The 2026 season is in progress. Week 2 ends with the
  Monday 2026-09-21 game; the Week 3 weekly run is due before the Thursday 2026-09-24 kickoff,
  and every later week follows the same rhythm (Thursday kickoff, Monday finish).
- The user runs `scripts/weekly_run.py` themselves unless they ask you to. It takes about 30
  minutes cold (ETL about 10 minutes from the nflreadpy cache, walk-forward compare 3 to 18
  minutes, training and reports a few minutes) and must not overlap with any walk-forward you
  are running. Check `pgrep -af walk_forward` and `uptime` before starting anything heavy.
- Walk-forward durations on this machine, one at a time: a from-week-1 run over three eval
  seasons about 60 to 140 minutes depending on load; six seasons about 100 minutes idle. Any
  edit under `nfl_predictor/ml/` changes every checkpoint fingerprint, so a rerun after a code
  change retrains from scratch: get the code stable before you measure.

## Starting state (updated 2026-09-20 during the rebuild session)

- `main` is at `85e4522`, the `--no-ff` merge of `feat/m59-benchmark-instrument` (`0.12.1` to
  `0.12.5`, committed on 2026-09-20 as five per-version commits `e5ee418`, `7189792`,
  `db6e892`, `532fd38`, `d9453cb` plus `084f857`; the split is a hunk-level reconstruction and
  the first commit's body names the two `walk_forward.py` hunks that carry later material).
  **Nothing is pushed**; pushing is a must-ask item the user has not answered yet.
- Active branch `chore/m59-etl-rebuild` off `main`, version `0.12.6`: `2d46083` fixes the
  `0.12.4` defect that the first ETL rebuild exposed (the sack exclusion starved
  `opponent_points_per_play`; six derived columns went null), `fec17d2` adds the matching rule
  to `AGENTS.md`. `scripts/gate.sh` exited `0` on `fec17d2` (`846 passed`, coverage `92.67%`).
- Data on disk (2026-09-20 04:31 rebuild, second pass, on the `0.12.6` code):
  `data/completed_games_ml.csv` `db6a78a3...` (`7278` rows, `513` columns); the through-2025
  cut `data/completed_games_ml.m59_through_2025.csv` `cf42ec55...` (`7261` rows). The build it
  replaced (`8bacad41...`, `519` columns) is in `data/backup_pre_m59_rebuild/`. Against it,
  exactly the six sack mirrors are gone and only the 27 division and conference derived columns
  move, all in 1999-2001 rows (plus one 2026 stadium surface filled in by the current-season
  refresh). Leakage audit `models/audit_m59_rebuild/leakage_audit.json`: `463` features, `0`
  flags. ETL logs and the cut script are in `models/etl_m59_rebuild/`.
- **Walk-forward tie check in progress or finished**: `models/wf_m59_rebuild_2023_2025_from_week1/`
  (`HYPOTHESIS.md` has the hypothesis, decision rule and command; `run.log` the progress;
  `compare_to_benchmark.py <candidate_ckpt> <reference_ckpt>` rescores two checkpoint
  directories, validated to reproduce the `AGENTS.md` table on a self-comparison). Reference:
  `models/wf_checkpoints/f6ff076066674127b163/`. If the report exists, the next step is the
  two-key rescore (a separate reviewer runs the compare script), then the `AGENTS.md` data-state
  and benchmark-input bullets, `.agents/TODO.md` ("From Milestone 59": the rebuild follow-up),
  `CHANGELOG.md`, gate, commit, and the merge question to the user.
- Milestone 59 is closed and archived with two parts reopened: the fitted-calibration pool is
  in-sample ("From Milestone 59" in `TODO.md`), and `n_estimators` is untuned (task 55.7).
  `auto` is the deterministic floor; production defaults to `elo`; no in-season early stopping
  since `0.12.3`.
- The user answered the first check-in on 2026-09-20: merge now (done locally), run the ETL
  rebuild when needed (done), the proposed order of work is confirmed, and the user will likely
  run the Week 3 weekly run themselves (hold off unless asked).
- The web API runs from the worktree `../nfl-predictor-web` on port 8765 against this
  checkout's `data/`, `models/` and `reports/`; never restart it unasked. `../nfeloqb` and
  `../nfl-sos-ratings` are the user's and read-only.

## Read first, in this order

1. `AGENTS.md` (mission, the standing yardstick, the benchmark table and its provenance
   paragraph, the delegation guardrails, the changelog and commit rules, the walk-forward
   operating notes).
2. `.agents/TODO.md`: the execution loop, the roadmap order, Milestones 54 to 58, and every
   "Open follow-ups" group (the "From Milestone 59" and "From task 56.4" groups first).
3. `.agents/ARCHIVE.md`, Milestone 59 (what landed, what was narrowed, the measurements, the
   audit), then Milestone 52 (the total head) and Milestone 56 (partial).
4. `CHANGELOG.md` entries `0.12.0` to `0.12.5`.
5. `README.md` sections "Win probability calibration", "Walk-forward backtesting", "Validation".
6. `.agents/web_ui_plan.md` only when you reach Milestone 58.

## First check-in (before any code)

Do these, then stop and put the questions to the user in one message:

1. `git status --short | wc -l`, `git log --oneline -3`, `scripts/gate.sh --quick`. If the quick
   gate is not clean, that is your first task; do not ask about anything else until it is.
2. Group the uncommitted tree into logical commits by version (`0.12.1` instrument, `0.12.2`
   calibration, `0.12.3` fit parity, `0.12.4` divisions/rare-events/sack mirrors, `0.12.5` audit
   fixes and gate script) and commit them on this branch (rule 6 allows this). Conventional
   Commits subjects, a body with what and why, the attribution line. Do not push.
3. Ask, in this order:
   - Merge `feat/m59-benchmark-instrument` to `main` now (via a PR or a local merge), or first
     wait for the Week 3 run to confirm nothing in the weekly path changed behavior?
   - When may the pending ETL rebuild run (it takes about 10 minutes plus a from-week-1
     walk-forward tie check of one to two hours, and must not overlap the weekly run)?
   - Confirm or change the proposed order of work below.
   - Should you run the Week 3 weekly run, or will the user?

## Proposed order of work (confirm with the user first)

The order is chosen for the goal above: correctness and operability of the weekly run first,
measurable modelling work second, UI phases when asked. Each item is one or more versioned
chunks with its own changelog entry, gate run, commit and handoff rewrite.

1. **Merge `0.12.5`** once the user says how. New work starts on a fresh branch off `main`,
   named for the milestone.
2. **ETL rebuild for `0.12.4`** (must-ask): back up `data/*.csv` (the repo convention is a
   `data/backup_pre_<tag>/` folder), rebuild, rerun `scripts/leakage_audit.py`, cut a
   seasons-`<= 2025` copy, run one from-week-1 walk-forward on it against the standing benchmark
   as the tie check, and record fingerprints and counts in `AGENTS.md` under the two-key rule.
3. **Task 54.0** (schedule skeleton and coverage check): small, data-correctness, directly
   affects JAX 2001-2002 rows and any team a future nflverse gap hits. Acceptance is in the task.
4. **Milestone 56** (weekly orchestration residuals): 56.1 data-refresh pass-through, 56.2
   postseason handling (the playoff weeks arrive in January and the current behavior is
   undocumented), 56.3 after Milestone 55. Also close or rewrite the "From task 56.4" follow-up
   about early stopping on the calibration frame: it is superseded by `0.12.3` (no in-season
   early stopping), and its remaining substance is task 55.7.
5. **Task 58.4** (web housekeeping): the `web` extra duplication and the job templates that
   ignore `NFLP_DATA_DIR`. Operational, small, touches the API so run `scripts/gate.sh --web`.
6. **Milestone 55** (configuration sweep and defaults), starting with 55.7 (`n_estimators`
   time-aware) because every later measurement depends on the fit being a measured choice.
   Changing a default mid-season is a must-ask; the sweep itself can run any time the machine is
   free. Include the out-of-fold calibration pool ("From Milestone 59") as an arm of the sweep
   if a fitted calibrator is to be reconsidered at all.
7. **Milestone 54, tasks 54.1 to 54.4** (PBP-first stats): feature work, measured on the
   instrument over six seasons, one arm per hypothesis.
8. **Task 53.7** (opponent-adjusted quarterback rate): the cheap step first; the ridge only if
   the cheap step shows signal.
9. **Milestone 58 phases 4 to 6**: only when the user asks, on their own branch, with the
   frontend gate.
10. **Milestone 57** stays parked unless the user reopens it.

The remaining "Open follow-ups" groups in `TODO.md` are picked up when their area is touched;
promote one to a task only with the user's agreement.

## How each chunk runs

- TDD: characterization or failing tests first for every production line you touch; no new
  `noqa`, `type: ignore` or `pragma: no cover` without a reason in the code.
- One versioned changelog entry per landed chunk (`Changed`, `Added`, `Removed`, `Fixed` in that
  order; patch for fixes, minor for a new family, default change or schema change), then
  `pyproject.toml` to the same version, `uv lock`, `uv sync`. Never `[Unreleased]`, never a tag.
- `scripts/gate.sh` exits `0` on the final tree before the chunk is reported done.
- Commit the chunk (rule 6). Rewrite this file (rule 8): branch, version, uncommitted state,
  the next task, open questions.
- Walk-forward runs: hypothesis and decision rule written in the check-in before the run;
  artifacts under `models/<descriptive_name>/`; numbers into the docs only after a separate
  rescore under the two-key rule, with the run directory beside them.
- Subagents: use them for parallel read-only work (a reviewer that rescores an artifact, a
  search across the tree) and for mechanical edits with a clear spec. Never run two walk-forward
  backtests at once, and do not give a subagent authority over anything on the must-ask list.
- Check in (a short message) at every landed version and after every walk-forward run: what
  landed, run directory, gate result, open questions. When a must-ask item comes up, ask and
  stop; do not fill the wait with unrelated work that changes state.

## Final report for a session

1. What landed, by version, with the commit subjects.
2. Every walk-forward run started this session: directory, hypothesis, result on the
   deterministic and market columns, and the decision it produced.
3. Anything in this file or in `AGENTS.md` that turned out to be wrong.
4. The state of the tree (branch, version, uncommitted files) and the next task, also written
   into this file.
5. The questions waiting on the user.
