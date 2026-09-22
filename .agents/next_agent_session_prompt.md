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
  `0.15.1`; `main` still carries `0.13.1`'s shared `200`-tree default. The next priority is
  task 55.8.
  Measured 2026-09-20: with `--skip-data-refresh` the weekly run took 32 minutes (Stage 1, the
  walk-forward compare, 30 of them while the web API's reload watcher loaded the machine); the
  ETL adds about 10 minutes from the nflreadpy cache. Refresh the lines first (`python -m
  nfl_predictor.lines_refresh --season 2026 --week N`) when the dataset is fresh but the odds
  are hours old. It must not overlap any walk-forward: `pgrep -af walk_forward` and `uptime`
  before anything heavy.
- Walk-forward durations, one at a time: from week 1 over three eval seasons about 20-50 minutes
  idle (measured 1227s on an idle machine 2026-09-21), about 110 minutes when anything else (the
  web API with `--reload`, a browser session) loads the machine, in which case relaunch with
  `OMP_WAIT_POLICY=PASSIVE` and resume; six seasons about 100 minutes idle at `200` trees and
  about 4.8 hours at `598` under load. Launch runs through a small `launch.sh` in the run
  directory with `nohup setsid`, never through a harness-bound shell (they have a 10-minute
  limit), and never `pkill -f` a pattern that matches your own shell. Any edit under
  `nfl_predictor/ml/` changes every checkpoint fingerprint; get the code stable before you
  measure.

## Starting state (2026-09-21, after tasks 54.0-54.4 on `feat/m54-0-landing`)

- **The working branch is `feat/m54-0-landing`, unmerged, at version `0.15.1`.** It is based on
  `main` at `d6795ca`; `main` and `origin/main` still carry `0.13.1`. `scripts/gate.sh` exits
  `0` on this branch at every landed chunk through `0.15.1`. The shared `n_estimators` default
  remains `200`; `scripts/weekly_run.py` Stage 1 evaluates the shared production XGBoost
  defaults; and the shipped `config/weekly_run.yaml` keeps `tune: false`,
  `wf_include_postseason: false`, `include_postseason: false`, `wf_max_depth: 5` and
  `wf_learning_rate: 0.0165`.
- **Task 55.7 is closed.** The accepted `100`-tree rung completed under
  `models/wf_m55_7_2020_2025_trees100/` (checkpoints `models/wf_checkpoints/09a60441e87d86c894ba/`)
  and was independently rescored in `REVIEW.md` there against both the governing `200` rung and
  the ladder reference `598`. By the written rule, `100` ties `200`, so `200` stays the shared
  default and no new user decision is needed.
- **Task 54.0 (`0.14.0`) is complete on this branch.** The parked commits `ae687dd` and
  `d34b0ab` were cherry-picked cleanly, the top-level CSVs were backed up to
  `data/backup_pre_m54_0/`, the ETL rebuild from cache logged the expected 9 coverage-gap
  warnings and 16 repair warnings, leakage audit `models/audit_m54_0_rebuild/leakage_audit.json`
  passed (`463` features, `0` flags), and the through-2025 cut
  `data/completed_games_ml.m54_0_through_2025.csv` is `e914eadf...` (`7261` rows,
  `513` columns). `data/completed_games_ml.csv` is now `db8b8ff4...` (`7292` completed rows,
  `513` columns). Note: the ETL rebuild ran with `--refresh-nflreadpy`, which overwrote the
  play-by-play cache under `data/cache/nflreadpy/` on new raw columns, so the `db8b8ff4...`
  build is no longer bit-reproducible from a from-scratch cache rebuild (a mixed-schema backup
  of the old cache sits in `data/cache/nflreadpy/backup_pre_m54_complete_20260921_152747/`, but
  it is not a clean pre-image). This does not affect the built CSVs already on disk, which are
  intact and verified by fingerprint.
- **Tasks 54.1-54.4 (`0.15.0`-`0.15.1`) are complete on this branch.** Two new source flags,
  both defaulting to the prior behavior: `--tr-stats-source {scrape,pbp}` derives the eight
  situational percentages from play-by-play counts instead of the TeamRankings scrape;
  `--team-stats-source {nflverse,pbp}` derives the per-team-game box score from play-by-play,
  overlaid on nflverse. Three correctness fixes landed alongside: `red_zone_tds` now requires
  `td_team == posteam` (a defensive score no longer credits the offense); the derived
  `red_zone_td_pct` now divides touchdown drives by red-zone trips (`fixed_drive`) rather than
  touchdowns by red-zone snaps (was `0.187` vs the scrape's `56.0` on the old, wrong basis; is
  `58.81` vs `56.02` now); and all eight percentages are now on the scraped 0-100 scale rather
  than 0-1. A separate, later fix (`0.15.1`) corrected a sign error in derived `total_yards`
  (nflverse subtracts an already-negative sack-yardage column; the derivation was subtracting a
  positive one — matched only `14.13%` of nflverse team-games before the fix, `96.62%` after).
  The task 54.2 comparison lives in `models/pbp_vs_nflverse_m54_2/COMPARISON.md`
  (reproduce with `compare_sources.py` and `compare_tr_situational.py` beside it): seventeen of
  twenty-one box-score columns agree with nflverse on `94%`+ of `13912` team-games of
  `1999-2025`; four are open exceptions (`passing_epa` `69.54%`, `fumbles` `73.63%`,
  `2pt_conversions` `94.80%`, `pass_attempts` `86.70%`), and a second section records that the
  published, season-to-date `two_point_conversion_pct` does not track the scrape well once
  blended (pbp mean `~47%` vs scrape `~32%` over 2010-2025), because rare attempts amplify the
  small `2pt_conversions` under-count. Neither flag is the default. **Flipping either default is
  a fresh must-ask decision**, argued for by play-by-play filling `1026` of `1029` completed
  games of 1999-2002 that the TeamRankings scrape (which starts in 2003) leaves null on the
  situational columns, and weighed against the four open exceptions above.
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
- **The 54.0 no-breakage arm is the current-build three-season reference for the default
  sources.** `models/wf_m54_0_2023_2025_from_week1/` (checkpoints
  `models/wf_checkpoints/d112ebcba3115bafe9d9/`, with `HYPOTHESIS.md`, `compare_output.txt` and
  `REVIEW.md` in the run directory) tied the accepted `200`-tree reference slice
  `models/wf_checkpoints/a5e76d54187e27ca7370_2023_2025/` on the governing weeks 3-18 window:
  deterministic Brier `0.2097` vs `0.2090`, diff `+0.0007` `[-0.0011, +0.0024]`; margin MAE
  `9.9166` vs `9.9044`, diff `+0.0122` `[-0.0566, +0.0801]`. The rebuild moved 836 of 855 scored
  2023-2025 rows in at least one feature, chiefly in the `sos_*` and `opponent_*` EPA families,
  so current-build arms on the default sources compare against this run, not the earlier
  `0.12.6` reference.
- **The tasks 54.1-54.4 arm** (`--team-stats-source pbp --tr-stats-source pbp`) is
  `models/wf_m54_12_2023_2025_from_week1/` (checkpoints
  `models/wf_checkpoints/4e729a9c5751ba978a71/`, `HYPOTHESIS.md` and `REVIEW.md` in the run
  directory), tied against the 54.0 reference immediately above on weeks 3-18: deterministic
  Brier `0.2096` vs `0.2097`, diff `-0.0001` `[-0.0017, +0.0016]`; margin MAE `9.9090` vs
  `9.9166`, diff `-0.0075` `[-0.0760, +0.0611]`; pick accuracy identical at `0.6833`. An earlier
  launch on an uncorrected build (before the `total_yards` sign fix) was stopped after 6 of 54
  folds once the source comparison exposed the defect; its checkpoints were discarded.
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
- The tasks 54.1-54.4 candidate build lives at `data_m54_candidate/` at the repo root (untracked
  and gitignored, like all data). Its through-2025 cut is
  `data/completed_games_ml.m54_12_through_2025.csv` (`e0b68a0e...`). Leave both alone unless
  you are extending the source-flag comparison work; they are not the production dataset.
- Machine state: no walk-forward running; check with `pgrep -af walk_forward` and `uptime`
  before anything heavy. The web API runs from the worktree `../nfl-predictor-web` on port 8765
  (the user starts it with `--reload`; its watcher takes half a core continuously and does load
  the machine); never restart it unasked. `../nfeloqb` and `../nfl-sos-ratings` are the user's
  and read-only.

## Read first, in this order

1. `AGENTS.md`: mission, the standing yardstick, the benchmark table, the reference arm and the
   fit-noise floor paragraphs, the "Tree-budget ladder" paragraph, the tasks 54.0-54.4 paragraph,
   the delegation guardrails, the changelog and commit rules, the walk-forward operating notes.
2. `.agents/TODO.md`: the execution loop, the roadmap order (set 2026-09-21, updated after
   54.1-54.4 landed), Milestone 55 tasks 55.7, 55.8 and 55.9 in full, Milestone 54's remaining
   default-flip decision, task 56.2, Milestone 60, and the "From Milestone 59" follow-ups.
3. The tree-budget ladder, in this order:
   `models/wf_m55_7_2020_2025_trees598/HYPOTHESIS.md` (the ladder rule and stopping rule), then
   the three reviews `models/wf_m55_7_2020_2025_trees598/REVIEW.md`,
   `models/wf_m55_7_2020_2025_trees200/REVIEW.md` and
   `models/wf_m55_7_2020_2025_trees400/REVIEW.md`, whose "Ladder summary" section is the source
   of the `AGENTS.md` tables.
4. `.agents/ARCHIVE.md`, Milestone 54 (partial) in full (both the 54.0 and the 54.1-54.4
   sections), then Milestone 59 ("Rebuild" and "Audit"), then Milestone 52.
5. `CHANGELOG.md` entries `0.12.0` to `0.15.1`.
6. `README.md` sections "Win probability calibration", "Backtesting" (including the superseded
   recency ablation note), "Validation", and the new `--team-stats-source`/`--tr-stats-source`
   paragraph.
7. `models/wf_m54_12_2023_2025_from_week1/HYPOTHESIS.md` and `REVIEW.md`, then
   `models/wf_m54_0_2023_2025_from_week1/HYPOTHESIS.md` and `REVIEW.md`, then
   `models/wf_m59_rebuild_2023_2025_from_week1_seed7/HYPOTHESIS.md` and `REVIEW.md` as the fit-
   noise-floor model.

## First check-in (before any code or run)

1. `git status --short | wc -l`, `git log --oneline -3`, `scripts/gate.sh --quick`. A dirty
   quick gate is the first task; ask about nothing else until it is clean.
2. `pgrep -af walk_forward`, `uptime`, and `ps -eo pcpu,args --sort=-pcpu | head -4` to see
   whether the web API or a weekly run is loading the machine.
3. Report to the user, in one message: the current branch state, that tasks 54.0-54.4 are
   complete on `feat/m54-0-landing`, the reviewed results, and the plan for task 55.8. This is
   still a report unless the user explicitly asks about merge or push.

## Order of work (decided with the user on 2026-09-21; tasks 54.1-54.4 landed the same day)

Each item is one or more versioned chunks with its own changelog entry, gate run, commit on a
branch named for the task, handoff rewrite, and a check-in. New work starts on a fresh branch
off `main`; merging and pushing are must-ask, every time.

1. **Task 55.8, season weighting**: half-lives of about `4`, `8` and `16` seasons via
   `--recency-half-life-seasons` as six-season arms against the unweighted reference at the
   `200` default on whichever build is then current. Replace the README's superseded ablation
   with the new measurement and its run directories whichever way it goes. A winning weighting
   is a default change: must-ask.
2. **Task 55.9, the Optuna re-tune**: after Milestone 54, ideally on a bye week or in the
   off-season. Its three prerequisites (no early stopping inside trials, a deterministic-Brier
   objective, and plumbing so a tuned set reaches the walk-forward) are each their own tested
   chunk before any trial runs. Details on the task.
3. **Milestone 60, the CLI inventory**: read-only, so it can run in parallel with any
   walk-forward as a subagent task. The removals land only after the user signs off.

Later, unchanged: Milestone 53 task 53.7, the rest of Milestone 55 and Milestone 56, Milestone
58 phases 4-6 whenever the user asks, and Milestone 57 stays parked. Also later: the Milestone
54 default-flip decision itself, whenever the user wants to make it.

## Documentation debts to clear as you go

- `README.md`, "Backtesting": the recency ablation table is marked superseded; task 55.8
  replaces it with the new measurement. Still open. The duration line was corrected in
  `0.12.13` with the ladder's measured timings. Done.
- The `0.13.0` / `0.13.1` tree-budget closeout is done. Keep later docs aligned to `200` as the
  standing shared default unless a future measured default change lands.
- `AGENTS.md`, walk-forward operating notes: the `launch.sh` / `nohup setsid` convention and the
  load-driven `PASSIVE` relaunch are recorded (`345c398`, `0.12.10`). Done.
- `.agents/TODO.md`, "Current validated baseline": restated at `0.15.1`. Restate it again at
  each landed chunk so it does not lag.
- Any place that still says the reference is the `0.12.6` rebuild arm: on the current 54.0 build,
  later current-build arms on the default sources compare against
  `models/wf_m54_0_2023_2025_from_week1/`; arms on the pbp-source build compare against
  `models/wf_m54_12_2023_2025_from_week1/`. The older `models/wf_m59_rebuild_2023_2025_from_week1/`
  and the accepted six-season `200` rung remain the previous-build records.
- `CHANGELOG.md`: one entry per landed chunk, new version each time, never `[Unreleased]`.

## Open questions waiting on the user

**Immediate question if the user asks for it:** whether to merge and push `feat/m54-0-landing`
(`0.15.1`). That is must-ask every time. Otherwise none block continuing into 55.8.
What remains must-ask under guardrail rule 5 inside this work:

1. Merging any branch into `main`, and pushing: must-ask, every time, however small the chunk.
2. Whether to flip `--team-stats-source`/`--tr-stats-source` to `pbp` as the shared default. The
   case for it is the `1999-2002` coverage gain; the case against is the four open exceptions in
   `models/pbp_vs_nflverse_m54_2/COMPARISON.md`. Not yet asked.
3. Any later rebuild under `data/`, and any deletion or overwrite under `data/` or `models/`
   beyond what tasks 54.0-54.4 already spent, is a fresh ask.
4. Any further rung of the closed tree-budget ladder (the pre-written `1200`, a second seed, or
   anything outside the accepted ladder as written) is a fresh ask.
5. Any future default change (`55.8`'s weighting, `55.9`'s tuned parameters, or a later revisit
   of tree budget) is a fresh ask with the numbers in hand.
6. Anything else on the rule 5 list: reopening Milestone 57, reordering the roadmap, touching
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
