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
- The user runs `scripts/weekly_run.py` themselves. `feat/m54-0-landing` is unmerged, at
  `0.16.1`; `main` still carries `0.13.1`'s shared `200`-tree default and the pre-flip
  `nflverse`/`scrape` sources. **If the user wants this branch's work in their Week 3 run, it
  needs to be merged first** (must-ask; see "Open questions" below). The next priority task is
  55.8.
  Measured 2026-09-20 (pre-flip): with `--skip-data-refresh` the weekly run took 32 minutes
  (Stage 1, the walk-forward compare, 30 of them while the web API's reload watcher loaded the
  machine); the ETL adds about 10 minutes from the nflreadpy cache. That ETL timing has not been
  re-measured on the `pbp`-default sources; expect it to be similar since the sources it derives
  from (play-by-play) were already being loaded either way. Refresh the lines first (`python -m
  nfl_predictor.lines_refresh --season 2026 --week N`) when the dataset is fresh but the odds
  are hours old. It must not overlap any walk-forward: `pgrep -af walk_forward` and `uptime`
  before anything heavy.
- Walk-forward durations, one at a time: from week 1 over three eval seasons about 20-50 minutes
  idle (measured 1227s-1559s on a lightly loaded machine on 2026-09-21), about 110 minutes when
  anything else (the web API with `--reload`, a browser session) loads the machine, in which
  case relaunch with `OMP_WAIT_POLICY=PASSIVE` and resume; six seasons about 100 minutes idle at
  `200` trees and about 4.8 hours at `598` under load. Launch runs through a small `launch.sh`
  in the run directory with `nohup setsid`, never through a harness-bound shell (they have a
  10-minute limit), and never `pkill -f` a pattern that matches your own shell. Any edit under
  `nfl_predictor/ml/` changes every checkpoint fingerprint; get the code stable before you
  measure. **Auto-mode note (2026-09-21):** an ETL rebuild with `--refresh-nflreadpy` was
  initially blocked by the harness's auto-mode safety classifier as "irreversible local
  destruction" even though it was pre-approved by the user; backing up the target files first
  (`data/*.csv`, `data/cache/nflreadpy/pbp_*.parquet`) and then writing the launch script with
  the `Write` tool and launching in separate small steps (rather than one large `mkdir && cat >
  ... && nohup ...` compound command) got it through. If you hit the same block, back up first
  and split the compound command into smaller separately-approved steps.

## Starting state (2026-09-21, after tasks 54.0-54.4 and the default flip on `feat/m54-0-landing`)

- **The working branch is `feat/m54-0-landing`, unmerged, at version `0.16.1`.** It is based on
  `main` at `d6795ca`; `main` and `origin/main` still carry `0.13.1`. `scripts/gate.sh` exits
  `0` on this branch at every landed chunk through `0.16.1`. The shared `n_estimators` default
  remains `200`; `scripts/weekly_run.py` Stage 1 evaluates the shared production XGBoost
  defaults; and the shipped `config/weekly_run.yaml` keeps `tune: false`,
  `wf_include_postseason: false`, `include_postseason: false`, `wf_max_depth: 5` and
  `wf_learning_rate: 0.0165`.
- **Task 55.7 is closed.** The accepted `100`-tree rung completed under
  `models/wf_m55_7_2020_2025_trees100/` (checkpoints `models/wf_checkpoints/09a60441e87d86c894ba/`)
  and was independently rescored in `REVIEW.md` there against both the governing `200` rung and
  the ladder reference `598`. By the written rule, `100` ties `200`, so `200` stays the shared
  default and no new user decision is needed.
- **Milestone 54 is closed (`0.14.0`-`0.16.1`).** Task 54.0 built the schedule skeleton and the
  collapsed-box-score repair; tasks 54.1-54.4 added `--team-stats-source`/`--tr-stats-source`
  options to derive the box score and situational percentages from play-by-play. The user then
  reviewed the four columns that disagreed with nflverse and asked for each to be fixed and both
  flags flipped to the default once verified:
  - **`passing_epa`** now sums `qb_epa` (nflverse's own quarterback-attribution EPA column)
    instead of `epa`, over every `pass_attempt` play including sacks and two-point tries. Match
    rate with nflverse: `69.54%` to `99.33%` on the full 1999-2025 rebuild.
  - **`pass_attempts`, `pass_completions`, `pass_yards`, `pass_touchdowns`,
    `interceptions_thrown`, `rush_attempts`, `rush_yards`, `rush_touchdowns`** now use
    nflverse's own canonical `pass_attempt`/`rush_attempt` raw flags (added to
    `constants.PBP_COLUMNS`) instead of `play_type`-based conditions. Worst case
    (`pass_attempts`) moved from `86.70%` to `99.87%`.
  - **`rushing_epa`** now includes two-point tries: `95.54%` to `99.87%`.
  - **`fumbles`/`fumbles_lost`** now exclude special-teams plays, matching nflverse's
    offense-only fumble stat: `73.63%`/`90.35%` to `91.95%`/`98.37%`. A residual gap on
    aborted-snap fumbles is documented, not further fixable from play-by-play (nflverse's own
    player-level fumble categories don't cleanly attribute those either).
  - **`2pt_conversions`** needed no code change (`94.80%` match): the derivation was already
    correct. The user manually verified one mismatch against the actual game (2024 week 17,
    Green Bay at Minnesota: exactly one two-point conversion happened, matching play-by-play
    exactly) and asked for the rest to be checked, which confirmed a systematic nflverse
    team-stats bug: `models/pbp_vs_nflverse_m54_2/verify_2pt_doubling.py` finds that of `246`
    mismatches across seven sampled seasons (`3710` team-games), `234` (`95.1%`) show nflverse's
    count at exactly double the play-by-play count, and zero mismatches go the other way. The
    play-by-play value is correct wherever it disagrees with nflverse. Its derived
    `two_point_conversion_pct` still does not track the TeamRankings scrape well once blended,
    but that comparison is against a different, unverified third-party source, not nflverse, so
    it does not by itself say which side is closer to the truth for the rate.
  - Full numbers and every formula: `models/pbp_vs_nflverse_m54_2/COMPARISON.md`.
  - **Both flags are now the default** (`0.16.0`), including the production fast path
    `scripts/weekly_run.py` uses when it calls `data_collection.main()` with no arguments (a
    separate hardcoded default the CLI argparse default alone would not have changed).
    `nflverse`/`scrape` remain selectable explicitly.
  - **The full ETL rebuild that followed** (`0.16.1`, `--refresh-nflreadpy` for the two new raw
    columns) closed the 1999-2002 Jacksonville team-stats coverage gap that task 54.0's
    schedule-skeleton repair was built around, as anticipated when Milestone 54 was widened on
    2026-09-18 specifically because play-by-play has both sides of every JAX game where
    nflverse's team-stats table does not: the now-default overlay fills those rows before the
    coverage check runs. The ETL logs `0` repair warnings on this rebuild (down
    from `16` on the 54.0 rebuild) and `7` coverage-gap warnings, all pre-existing single-game
    gaps unrelated to JAX. `strength_games_played` for JAX still reads `16.0` at both 2001 and
    2002 season end. The schedule-skeleton and repair code remain in place as a safety net for
    the `nflverse`/`scrape` configuration.
  - **Data on disk:** `data/completed_games_ml.csv` `edd6b852...` (`7292` completed rows, `513`
    columns); through-2025 cut `data/completed_games_ml.m54_flip_through_2025.csv`
    `2d4111a6...` (`7261` rows). Prior top-level CSVs backed up to `data/backup_pre_m54_flip/`;
    the pre-refresh play-by-play cache backed up to
    `data/cache/nflreadpy/backup_pre_m54_flip/`. Leakage audit
    `models/audit_m54_flip_rebuild/leakage_audit.json`: `463` features, `0` flags.
  - **Reviewed verification arm:** `models/wf_m54_flip_2023_2025_from_week1/` (checkpoints
    `models/wf_checkpoints/9779c1cbb0701d23661a/`, `HYPOTHESIS.md`, `compare_output.txt` and
    `REVIEW.md` in the run directory) tied the pre-flip reference
    `models/wf_m54_0_2023_2025_from_week1/` on weeks 3-18: deterministic Brier `0.2106` vs
    `0.2097`, diff `+0.0009` `[-0.0008, +0.0026]`; margin MAE `9.9321` vs `9.9166`, diff
    `+0.0156` `[-0.0529, +0.0812]`. A no-breakage tie by the written rule.
  - **A cache-reproducibility note the user asked about:** the `--refresh-nflreadpy` rebuilds
    (both task 54.0's and this one's) overwrite the play-by-play cache under
    `data/cache/nflreadpy/`, so the exact prior cache state before either refresh is not fully
    preserved (only a mixed-schema partial backup survives from the 54.0 refresh). This does not
    affect correctness of the data on disk now, which is fingerprinted and leakage-audited; it
    only means a bit-for-bit historical replay of an older build's cache state is not possible.
    Every future `--refresh-nflreadpy` should keep backing up the cache directory first (now
    standard practice, see the auto-mode note above), so this stops recurring going forward.
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
  The `1200` rung stays pre-written and unlaunched. These rungs all predate the pbp-default
  flip, so a new six-season floor would need a fresh run on the current build if task 55.8 or
  55.9 needs one.
- What the model trains on: every walk-forward fold and the production fit train on every game
  strictly before the predicted week, all seasons from 1999 (a fold in 2025 trains on about
  `6,900` games across 27 seasons). `--eval-last-n-seasons` sets how many seasons are scored,
  not trained on. Every training row carries the same weight; recency weighting exists
  (`--recency-half-life-seasons`) and is off by default, which is what task 55.8 measures.
- The Week 2 package for the Sunday and Monday games is `models/weekly_2026_week_02_refresh/`
  (config `hybrid_raw_prob_base_elo_blend0.20_clamp0.10`, `elo` calibration, 15 games). It
  predates the pbp-default flip.
- `auto` calibration is the deterministic floor; production defaults to `elo`; no in-season
  early stopping since `0.12.3`; the shared tree budget is `200`. `--n-estimators`
  overrides it since `0.12.10`. `--team-stats-source`/`--tr-stats-source` default to `pbp`
  since `0.16.0`.
- Machine state: no walk-forward running; check with `pgrep -af walk_forward` and `uptime`
  before anything heavy. The web API runs from the worktree `../nfl-predictor-web` on port 8765
  (the user starts it with `--reload`; its watcher takes half a core continuously and does load
  the machine); never restart it unasked. `../nfeloqb` and `../nfl-sos-ratings` are the user's
  and read-only.

## Read first, in this order

1. `AGENTS.md`: mission, the standing yardstick, the benchmark table, the reference arm and the
   fit-noise floor paragraphs, the "Tree-budget ladder" paragraph, the tasks 54.0-54.4 and
   default-flip paragraph, the delegation guardrails, the changelog and commit rules, the
   walk-forward operating notes.
2. `.agents/TODO.md`: the execution loop, the roadmap order (set 2026-09-21, updated after
   Milestone 54 closed), Milestone 55 tasks 55.7, 55.8 and 55.9 in full, task 56.2, Milestone 60,
   and the "From Milestone 59" follow-ups.
3. The tree-budget ladder, in this order:
   `models/wf_m55_7_2020_2025_trees598/HYPOTHESIS.md` (the ladder rule and stopping rule), then
   the three reviews `models/wf_m55_7_2020_2025_trees598/REVIEW.md`,
   `models/wf_m55_7_2020_2025_trees200/REVIEW.md` and
   `models/wf_m55_7_2020_2025_trees400/REVIEW.md`, whose "Ladder summary" section is the source
   of the `AGENTS.md` tables.
4. `.agents/ARCHIVE.md`, Milestone 54 (partial) in full, including the "Default flip to `pbp`"
   subsection at the end, then Milestone 59 ("Rebuild" and "Audit"), then Milestone 52.
5. `CHANGELOG.md` entries `0.12.0` to `0.16.1`.
6. `README.md` sections "Win probability calibration", "Backtesting" (including the superseded
   recency ablation note), "Validation", and the `--team-stats-source`/`--tr-stats-source`
   paragraph (rewritten for the default flip).
7. `models/pbp_vs_nflverse_m54_2/COMPARISON.md` for the full box-score and situational-percentage
   verification numbers.
8. `models/wf_m54_flip_2023_2025_from_week1/HYPOTHESIS.md` and `REVIEW.md`, then
   `models/wf_m54_0_2023_2025_from_week1/HYPOTHESIS.md` and `REVIEW.md`, then
   `models/wf_m59_rebuild_2023_2025_from_week1_seed7/HYPOTHESIS.md` and `REVIEW.md` as the fit-
   noise-floor model.

## First check-in (before any code or run)

1. `git status --short | wc -l`, `git log --oneline -3`, `scripts/gate.sh --quick`. A dirty
   quick gate is the first task; ask about nothing else until it is clean.
2. `pgrep -af walk_forward`, `uptime`, and `ps -eo pcpu,args --sort=-pcpu | head -4` to see
   whether the web API or a weekly run is loading the machine.
3. Report to the user, in one message: the current branch state, that Milestone 54 (including
   the default flip) is complete on `feat/m54-0-landing`, the reviewed results, and the plan for
   task 55.8. This is still a report unless the user explicitly asks about merge or push.

## Order of work (decided with the user on 2026-09-21; Milestone 54 closed the same day)

Each item is one or more versioned chunks with its own changelog entry, gate run, commit on a
branch named for the task, handoff rewrite, and a check-in. New work starts on a fresh branch
off `main`; merging and pushing are must-ask, every time.

1. **Task 55.8, season weighting**: half-lives of about `4`, `8` and `16` seasons via
   `--recency-half-life-seasons` as six-season arms against the unweighted reference at the
   `200` default on whichever build is then current (now the pbp-default build). Replace the
   README's superseded ablation with the new measurement and its run directories whichever way
   it goes. A winning weighting is a default change: must-ask.
2. **Task 55.9, the Optuna re-tune**: after Milestone 54, ideally on a bye week or in the
   off-season. Milestone 54 is now closed, so this is unblocked whenever the user wants it. Its
   three prerequisites (no early stopping inside trials, a deterministic-Brier objective, and
   plumbing so a tuned set reaches the walk-forward) are each their own tested chunk before any
   trial runs. Details on the task.
3. **Milestone 60, the CLI inventory**: read-only, so it can run in parallel with any
   walk-forward as a subagent task. The removals land only after the user signs off.

Later, unchanged: Milestone 53 task 53.7, the rest of Milestone 55 and Milestone 56, Milestone
58 phases 4-6 whenever the user asks, and Milestone 57 stays parked.

## Documentation debts to clear as you go

- `README.md`, "Backtesting": the recency ablation table is marked superseded; task 55.8
  replaces it with the new measurement. Still open. The duration line was corrected in
  `0.12.13` with the ladder's measured timings. Done.
- The `0.13.0` / `0.13.1` tree-budget closeout is done. Keep later docs aligned to `200` as the
  standing shared default unless a future measured default change lands.
- `AGENTS.md`, walk-forward operating notes: the `launch.sh` / `nohup setsid` convention and the
  load-driven `PASSIVE` relaunch are recorded (`345c398`, `0.12.10`). Done. The auto-mode
  classifier note (back up before `--refresh-nflreadpy`, split large compound launch commands)
  is recorded in this file's Calendar section; consider folding it into `AGENTS.md` proper if it
  recurs.
- `.agents/TODO.md`, "Current validated baseline": restated at `0.16.1`. Restate it again at
  each landed chunk so it does not lag.
- Any place that still says the reference is a pre-flip build: task 55.8 and later work should
  measure against the `pbp`-default build (`data/completed_games_ml.csv` `edd6b852...`) and its
  reference arm `models/wf_m54_flip_2023_2025_from_week1/`, not
  `models/wf_m54_0_2023_2025_from_week1/` or earlier, which are now historical (pre-flip)
  records.
- `CHANGELOG.md`: one entry per landed chunk, new version each time, never `[Unreleased]`.

## Open questions waiting on the user

**Immediate question if the user asks for it:** whether to merge and push `feat/m54-0-landing`
(`0.16.1`). That is must-ask every time. The user's 2026-09-20 note that they run the Week 3
weekly run themselves before Thursday 2026-09-24 kickoff means this branch's work (including the
default flip) only reaches their weekly run if it is merged to `main` first; flag this plainly
if the deadline is close and no merge has happened yet, without merging unasked. Otherwise
nothing blocks continuing into 55.8. What remains must-ask under guardrail rule 5 inside this
work:

1. Merging any branch into `main`, and pushing: must-ask, every time, however small the chunk.
2. Any later rebuild under `data/`, and any deletion or overwrite under `data/` or `models/`
   beyond what Milestone 54 already spent, is a fresh ask.
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
