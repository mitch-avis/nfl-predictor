# Benchmarks, reference arms and data state

Benchmark tables, reference arms, fit-noise floors and dataset records for walk-forward work.
Every number here follows guardrail rule 3 in `AGENTS.md` (two keys, a run directory beside each
number). The roadmap order is in `AGENTS.md` and `.agents/TODO.md`.

## Workstream history

Run directories this section names under `models/` (`audit_m54_0_rebuild`,
`wf_m54_0_2023_2025_from_week1`, `wf_m54_12_2023_2025_from_week1`,
`wf_m54_flip_2023_2025_from_week1`, `wf_m55_8_2020_2025_half_life4`,
`wf_m55_8_2020_2025_half_life8`) were deleted on 2026-10-16 (inventory in
`.agents/models_cleanup/`); the numbers stand as reviewed.

- The active workstream is feature engineering for true team strength: PBP-derived per-snap EPA
  families, weekly schedule-adjusted (ridge) team strength, continuous early-season shrinkage of
  every season-to-date family, power rankings on the adjusted composite, and the total-head fix
  (version `0.6.2`: each XGBoost fit gets its own early stopping) have landed. The fixed total
  head still trails the market's total line in walk-forward, so the betting report labels totals
  `diagnostic_only`. QB per-dropback EPA families landed in version `0.7.0`, tied in walk-forward
  with the group switched off, and the user decided 2026-09-11 to keep them (`.agents/ARCHIVE.md`,
  Milestone 53). Their schedule lenses (task 53.6) landed in version `0.9.0`, measured no gain
  in walk-forward, and were dropped from the schema in `0.10.0` by the user's decision on
  2026-09-17 (`.agents/ARCHIVE.md`, Milestone 53, "53.6"). The Milestone 49
  `games_played` follow-up closed in `0.11.0`, and the 2026-09-18 audit pruned 20 dead-weight and
  duplicate columns at training time in `0.12.0` (`482` to `462` features; a walk-forward tie).
  Milestone 59 closed in `0.12.4` and was audited and corrected in `0.12.5`: the benchmark
  instrument now carries configured, deterministic and market-implied probability views; `auto`
  is explicitly the deterministic floor; production and walk-forward fit through the same
  helpers with no in-season early stopping and `best_iteration` recorded per head; 1999-2001
  division context is historically aligned; `def_sacks` / `times_sacked` are excluded from
  opponent mirrors at the ETL source; and the six-season noise-family follow-up found no lift.
  Two parts were narrowed and are reopened as follow-ups (`.agents/TODO.md`, "From Milestone
  59"): the fitted-calibration pool is in-sample rather than out-of-fold, and `n_estimators` was
  not re-tuned (task 55.7). The ETL rebuild for 59.4 and 59.6 ran on 2026-09-20 (data state
  below); its first pass exposed a `0.12.4` defect (the sack exclusion starved
  `opponent_points_per_play`), fixed in `0.12.6`. The fit-noise floor of the instrument was
  measured the same day (see "Fit-noise floor on the same build" below), and on that evidence
  the user reordered the roadmap: Milestone 55 tasks 55.7 (the tree budget) and 55.8 (season
  weighting) come first, then task 54.0 as a no-breakage check, then the rest of Milestone 54
  (PBP-first, with the schedule skeleton as task 54.0). The analysis, crosswalk, and
  prioritized shortlist live in `.agents/feature_crosswalk.md`; the ordered milestones live in
  `.agents/TODO.md`. Task 55.7's tree-budget ladder ran in `0.12.10`-`0.12.13` (three
  six-season rungs, "Tree-budget ladder" below); on 2026-09-21 the user approved adopting
  `200` as the shared default in `0.13.0`, and the accepted `100` rung later tied that default
  in `0.13.1`, so `200` stays in place.
  Task 56.1 landed in `0.12.11`; tasks 56.2 (the postseason default) and 58.4 (the ETL's
  upstream input paths) were narrowed in `0.12.11` and `0.12.12`. The web UI
  (FastAPI + React, Milestone 58 phases 0-3) merged into `main` as version `0.8.0` on
  2026-09-11; its open phases are Milestone 58 in `.agents/TODO.md`. Task 54.0 landed on the
  feature branch `feat/m54-0-landing` as version `0.14.0` on 2026-09-21: the schedule now
  supplies the completed regular-season team-game rows, the collapsed 2001-2002 Jacksonville home
  rows have their box-score columns nulled, the cache rebuild produced
  `data/completed_games_ml.csv` `db8b8ff4...` (`7292` completed rows, `513` columns) and the
  through-2025 cut `data/completed_games_ml.m54_0_through_2025.csv` `e914eadf...` (`7261` rows),
  leakage audit `models/audit_m54_0_rebuild/leakage_audit.json` passed (`463` features, `0`
  flags), and the reviewed three-season no-breakage arm
  `models/wf_m54_0_2023_2025_from_week1/` (checkpoints `d112ebcba3115bafe9d9`) tied the accepted
  `200`-tree reference slice on the governing weeks 3-18 window: deterministic Brier `0.2097`
  vs `0.2090`, diff `+0.0007` `[-0.0011, +0.0024]`; margin MAE `9.9166` vs `9.9044`, diff
  `+0.0122` `[-0.0566, +0.0801]`. The rebuild moved 836 of 855 scored 2023-2025 rows in at
  least one feature, chiefly in the `sos_*` and `opponent_*` EPA families, so later current-build
  arms compare against the 54.0 reference, not the earlier `0.12.6` reference arm. Tasks
  54.1-54.4 landed as `0.15.0`-`0.15.1` on the same day and branch: the eight situational
  percentages and the per-team-game box score can now be derived from play-by-play behind
  `--tr-stats-source pbp` and `--team-stats-source pbp` (both default to the prior source), with
  `red_zone_tds` fixed to require `td_team == posteam` and the derived `red_zone_td_pct` fixed to
  divide touchdown drives by red-zone trips (`fixed_drive`) instead of touchdowns by red-zone
  snaps. The comparison task 54.2 requires
  (`models/pbp_vs_nflverse_m54_2/COMPARISON.md`) caught a sign error in the derived `total_yards`
  (nflverse subtracts an already-negative sack-yardage column; the derivation was subtracting a
  positive one, matching only 14.13% of nflverse team-games before the fix and 96.62% after) and
  records four open exceptions (`passing_epa`, `fumbles`, `2pt_conversions`, `pass_attempts`) and
  one derived-rate caveat (`two_point_conversion_pct` does not track the scrape once blended,
  because rare attempts amplify the `2pt_conversions` under-count). The reviewed three-season
  arm `models/wf_m54_12_2023_2025_from_week1/` (checkpoints `4e729a9c5751ba978a71`) tied the
  54.0 reference above on weeks 3-18: deterministic Brier `0.2096` vs `0.2097`, diff `-0.0001`
  `[-0.0017, +0.0016]`; margin MAE `9.9090` vs `9.9166`, diff `-0.0075` `[-0.0760, +0.0611]`.
  Neither flag was the default at that point; the user then reviewed the four exceptions and
  approved fixing each and flipping both flags once verified. All four landed as `0.16.0` on the
  same branch: `passing_epa` sums `qb_epa` (nflverse's own quarterback-attribution EPA column)
  instead of `epa`, matching nflverse on `99.33%` of the full 1999-2025 rebuild (up from
  `69.54%`); `pass_attempts`/`pass_completions`/`pass_yards`/`pass_touchdowns`/
  `interceptions_thrown`/`rush_attempts`/`rush_yards`/`rush_touchdowns` use nflverse's own
  `pass_attempt`/`rush_attempt` raw flags instead of `play_type` (worst case `pass_attempts`
  `86.70%` to `99.87%`); `rushing_epa` now includes two-point tries (`95.54%` to `99.87%`);
  `fumbles`/`fumbles_lost` exclude special-teams plays (`73.63%`/`90.35%` to `91.95%`/`98.37%`,
  a residual gap on aborted-snap fumbles documented as not further fixable from play-by-play).
  `2pt_conversions` (`94.80%`) needed no code change: the user manually verified one mismatch
  against the actual game (2024 week 17, Green Bay at Minnesota — exactly one two-point
  conversion happened, matching play-by-play) and asked for the rest to be checked, which found
  a confirmed, systematic nflverse team-stats bug: of `246` mismatches across seven sampled
  seasons (`3710` team-games), `234` (`95.1%`) show nflverse's count at exactly double the
  play-by-play count, and zero mismatches go the other way
  (`models/pbp_vs_nflverse_m54_2/verify_2pt_doubling.py`). The play-by-play value is correct
  wherever it disagrees with nflverse. Both flags are now the default, including the production
  fast path the weekly run (`nfl-predictor weekly`) uses when it calls `data_collection.main()`
  with no arguments; `nflverse`/`scrape` remain selectable explicitly. A full ETL rebuild followed
  (`0.16.1`, `--refresh-nflreadpy` for the two new raw columns), which closed the JAX 1999-2002
  coverage gap that task 54.0's schedule-skeleton repair was built for, as anticipated when
  Milestone 54 was widened on 2026-09-18 specifically because play-by-play has both sides of
  every JAX game where nflverse's team-stats table does not: the now-default overlay fills those
  rows before the coverage check runs, and the ETL logs `0` repair warnings on this rebuild (down
  from `16`). The reviewed verification arm
  `models/wf_m54_flip_2023_2025_from_week1/` (checkpoints `9779c1cbb0701d23661a`) tied the 54.0
  pre-flip reference on weeks 3-18: deterministic Brier `0.2106` vs `0.2097`, diff `+0.0009`
  `[-0.0008, +0.0026]`; margin MAE `9.9321` vs `9.9166`, diff `+0.0156` `[-0.0529, +0.0812]`.
  Milestone 54 is closed. `feat/m54-0-landing` merged into `main` with no conflicts and was
  pushed on 2026-09-22 (merge commit `295d4c4`, version `0.16.2`); `main` and `origin/main` now
  carry the `pbp`-default sources and the shared `200`-tree default with no further merge
  needed. A 2026-09-22 review found every other branch except `feat/web-ui` (the live worktree
  behind the web API on port 8765) was a fully-merged, zero-commit ancestor of `main`, and the
  user approved deleting all ten of them, locally and on `origin`. Task 55.8 (season weighting)
  was closed on `feat/m55-8-season-weighting` as version `0.17.0` on 2026-09-23 and reopened the
  same day by the user after a second-key review: the four six-season arms
  (`models/wf_m55_8_2020_2025_{unweighted,half_life4,half_life8,half_life16}/`) are valid, but the
  close-out kept the shipped half-life `4`, the weakest arm on every probability and error metric,
  and its reviews were written by the agent that produced the runs. The redo added a second seed
  and longer half-lives (nine arms in all) and an independent review
  (`models/wf_m55_8_review/INDEPENDENT_REVIEW.md`), and on 2026-09-24 the user chose
  **unweighted** training for production: version `0.18.0` removes both
  `train_recency_half_life_seasons: 4` and `wf_recency_half_life_seasons: 4` from
  `config/weekly_run.yaml`. The numbers are under "Season weighting and the six-season fit-noise
  floor" below.

## Data state and ETL rebuild records

Run directories this section names under `models/` (`audit_m49_games_played`, `audit_m59_rebuild`,
`etl_m59_rebuild`, `wf_m59_rebuild_2023_2025_from_week1`) were deleted on 2026-10-16 (inventory in
`.agents/models_cleanup/`); the numbers stand as reviewed.

- ETL rebuilt on 2026-10-14 at 20:12-20:18 MDT on `0.59.0` (`119a016`, the step-6 merge) with
  `nfl-predictor data` and its defaults, after the user approved the merge and the rebuild:
  `7340` completed rows, `541` columns, fingerprint `f3eca626...`, and `534` ported-ratings
  files under `data/ratings/`. The build it replaced (`ca2f7bd3...`, `0.45.4` with the
  2026-10-14 `qb_elos.csv`) is in `~/scratch/data_backups/2026-10-14_pre_step6_merge/` (`1641`
  files, `SHA256SUMS.txt`, verified). The leakage audit passed on it (`490` features, `0`
  flags, `models/leakage_audit.json`). The driver and its log are
  `~/scratch/week06_step6/launch.sh` and `run.log` beside it.
- ETL was rerun on 2026-09-20 at 04:31 from the cache for `1999-2026` on the `0.12.6` schema
  (pre-2002 division and conference alignment from `0.12.4`, the six `*_opponent_def_sacks` /
  `*_opponent_times_sacked` mirrors gone, the times-sacked mirror kept as an unpublished
  intermediate by `0.12.6`): `7278` completed rows, `513` columns, fingerprint `db6a78a3...`.
  Against the build it replaced (`8bacad41...`, kept in `data/backup_pre_m59_rebuild/`) only
  the 27 division and conference derived columns moved, all in 1999-2001 rows, plus one 2026
  `stadium_surface` filled in by the current-season refresh. The leakage audit passed on it
  (`463` features, `0` flags, `models/audit_m59_rebuild/leakage_audit.json`). The first pass
  of the rebuild, on the `0.12.5` code, published six derived columns as nulls
  (`models/etl_m59_rebuild/etl_defective_first_pass.log`) and was discarded. The through-2025
  cut for walk-forward comparisons is `data/completed_games_ml.m59_through_2025.csv` (`7261`
  rows, `cf42ec55...`, cut by `models/etl_m59_rebuild/cut_through_2025.py`); its tie check
  against the benchmark below is `models/wf_m59_rebuild_2023_2025_from_week1/`.
- The previous record: ETL rerun on 2026-09-17 at 22:28 from the cache for `1999-2026` on
  the `0.11.0` schema
  (schedule lenses removed in `0.10.0`, `games_played` corrected in `0.11.0`): `7278` completed
  rows (all of `1999-2025` plus the 2026 Week 1 games and the Week 2 Thursday game, which
  finished during the rebuild), `519` columns, fingerprint `8bacad41...`. Against the build it
  replaced (`0cecc2e3...`, kept in `data/backup_pre_m49_games_played/`) exactly two columns
  moved: `away_games_played` and `home_games_played`. The leakage audit passed on it (`483`
  features, `0` flags, `models/audit_m49_games_played/leakage_audit.json`). Earlier inputs kept
  as records: `data/completed_games_ml.m49_through_2025.csv` (`7261` rows, `07971269...`, the
  `0.11.0` walk-forward input), `data/completed_games_ml.m53_6_through_2025.csv` (`525`
  columns, `940cbbf4...`, the lens measurement's input), and the pre-lens builds in
  `data/backup_pre_m53_6/` and `data/backup_pre_m53_6_drop/`. The
  benchmark below was measured on the earlier `498`-column build
  `data/completed_games_ml.m49_on_through_2025.csv`, which a data cleanup removed; its numbers
  stay auditable from the fold checkpoints named below.

## Walk-forward benchmark and reference arms

Run directories this section names under `models/` (`wf_m54_0_2023_2025_from_week1`,
`wf_m55_7_2020_2025_trees100`, `wf_m55_7_2020_2025_trees200`, `wf_m55_7_2020_2025_trees400`,
`wf_m55_7_2020_2025_trees598`, `wf_m59_2020_2025_auto_floor_baseline`,
`wf_m59_2023_2025_from_week1`, `wf_m59_2023_2025_from_week1_auto_floor`,
`wf_m59_rebuild_2023_2025_from_week1`, `wf_m59_rebuild_2023_2025_from_week1_seed7`,
`wf_shrink_2023_2025_off`, `wf_strength_2023_2025_from_week1`,
`wf_m59_2023_2025_from_week1_auto_sigma`, `wf_m59_2023_2025_from_week1_auto_sigma_centered`,
`wf_m59_2023_2025_from_week1_auto_selected`, `wf_m55_8_2020_2025_half_life4`,
`wf_m55_8_2020_2025_half_life4_seed7`, `wf_m55_8_2020_2025_half_life8`,
`wf_m55_8_2020_2025_half_life32`, `wf_m55_8_2020_2025_half_life32_seed7`) were deleted on 2026-10-16
(inventory in `.agents/models_cleanup/`); the numbers stand as reviewed.

- **Current walk-forward benchmark**, measured 2026-09-19 on the `0.12.3` fit-parity code
  (no in-season early stopping) with `data/completed_games_ml.m49_through_2025.deadweight_cut.csv`,
  seasons `2023-2025`, from week 1, `market_anchor` on: run
  `models/wf_m59_2023_2025_from_week1_auto_floor/` (checkpoints
  `models/wf_checkpoints/f6ff076066674127b163/`). The five `models/wf_m59_2023_2025_from_week1*/`
  arms differ only in the configured calibrator and share these deterministic and market columns
  exactly, and the 2023-2025 folds of the six-season `models/wf_m59_2020_2025_auto_floor_baseline/`
  reproduce them bit for bit. A true rescore of the `0.12.0` arm's checkpoints
  (`bae0e56db951d1a890d4`, the previous fit with early stopping) sits in the fourth decimal:
  weeks 3-18 log loss `0.6072` and margin MAE `9.9600`, all weeks `0.6096` and `9.8291`, because
  the fit-parity change moved 247 of 816 predicted margins by up to `0.34` points. Treat the
  deterministic and market-implied columns below as the standing probability instrument; the
  configured calibrator is diagnostic only unless it can beat them on the same games.

  | window | games | model Brier | model log loss | model pick acc | market Brier | market log loss | market pick acc | margin MAE | total MAE | market total MAE |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | week 1 only | 48 | `0.2066` | `0.6029` | `0.7500` | `0.2099` | `0.6090` | `0.7083` | `9.1501` | `10.2129` | `10.3333` |
  | week 2 only | 48 | `0.2306` | `0.6526` | `0.6250` | `0.2330` | `0.6585` | `0.5833` | `8.5436` | `9.6833` | `10.4479` |
  | weeks 3-18 (headline) | 720 | `0.2099` | `0.6073` | `0.6861` | `0.2086` | `0.6042` | `0.6861` | `9.9608` | `10.3025` | `10.0847` |
  | all weeks | 816 | `0.2109` | `0.6097` | `0.6863` | `0.2102` | `0.6077` | `0.6814` | `9.8298` | `10.2608` | `10.1207` |

  Before the fix the one-tree total head scored `9.9426` / `9.8727` / `10.1257` / `10.1000` in the
  same windows (weeks 1-2 skip calibration, so they have no eval set and never had the bug). The
  healthy anchored head trails the market line in weeks 3-18; the comparison and its bootstrap
  intervals are under Milestone 52 in `.agents/TODO.md`.

  **Reference arm on the 2026-09-20 rebuild** (`0.12.6` schema, `data/completed_games_ml.m59_through_2025.csv`
  `cf42ec55...`, same config and code path, git `fec17d2`):
  `models/wf_m59_rebuild_2023_2025_from_week1/` (checkpoints
  `models/wf_checkpoints/34c17e508ab015a80662/`; hypothesis and decision rule written before the
  run in its `HYPOTHESIS.md`; rescored independently by a reviewer subagent in its `REVIEW.md`,
  which also reproduces the table above from the `f6ff0760...` checkpoints). New arms on the
  rebuilt build compare against this run, not the table above, unless they are on the later 54.0
  schedule-skeleton build, in which case compare against the 54.0 reference below.

  | window | games | model Brier | model log loss | model pick acc | market Brier | market log loss | market pick acc | margin MAE | total MAE |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | week 1 only | 48 | `0.2071` | `0.6038` | `0.7500` | `0.2099` | `0.6090` | `0.7083` | `9.0976` | `10.0247` |
  | week 2 only | 48 | `0.2266` | `0.6440` | `0.5833` | `0.2330` | `0.6585` | `0.5833` | `8.4281` | `9.7688` |
  | weeks 3-18 (headline) | 720 | `0.2096` | `0.6072` | `0.6764` | `0.2086` | `0.6042` | `0.6861` | `9.9722` | `10.2389` |
  | all weeks | 816 | `0.2105` | `0.6091` | `0.6752` | `0.2102` | `0.6077` | `0.6814` | `9.8299` | `10.1986` |

  Paired against the benchmark above (rebuild minus benchmark, 5000 resamples, seed 0), weeks
  3-18: deterministic Brier `-0.0003` `[-0.0028, +0.0023]`, margin MAE `+0.0113`
  `[-0.0890, +0.1142]`; all weeks `-0.0004` `[-0.0028, +0.0019]` and `+0.0001`. A tie by the
  rule written before the run. It is an aggregate tie, not a reproduction: all 816 margins moved
  (up to `5.16` points, because the 1999-2001 training rows changed), and deterministic pick
  accuracy in weeks 3-18 fell from `0.6861` to `0.6764` (7 games), so the model no longer
  matches the market's pick accuracy there. Every fold ran the full `598`-tree budget.

  **Reference arm on the 2026-09-21 schedule-skeleton rebuild** (`0.14.0` branch,
  `data/completed_games_ml.m54_0_through_2025.csv` `e914eadf...`, same benchmark config with the
  shared `200`-tree default, git `5afe30f`):
  `models/wf_m54_0_2023_2025_from_week1/` (checkpoints
  `models/wf_checkpoints/d112ebcba3115bafe9d9/`; hypothesis and decision rule in its
  `HYPOTHESIS.md`; reviewed from `compare_output.txt` in its `REVIEW.md` against the reference
  slice `models/wf_checkpoints/a5e76d54187e27ca7370_2023_2025/`, which is the 2023-2025 subset of
  the accepted six-season `200` rung). This is the current-build reference for later work on the
  54.0 branch and its descendants.

  | window | games | model Brier | model log loss | model pick acc | market Brier | market log loss | market pick acc | margin MAE | total MAE |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | week 1 only | 48 | `0.2046` | `0.5987` | `0.7500` | `0.2099` | `0.6090` | `0.7083` | `8.8055` | `10.1284` |
  | week 2 only | 48 | `0.2292` | `0.6501` | `0.5833` | `0.2330` | `0.6585` | `0.5833` | `8.4435` | `9.9878` |
  | weeks 3-18 (headline) | 720 | `0.2097` | `0.6072` | `0.6833` | `0.2086` | `0.6042` | `0.6861` | `9.9166` | `10.1890` |
  | all weeks | 816 | `0.2105` | `0.6093` | `0.6814` | `0.2102` | `0.6077` | `0.6814` | `9.7645` | `10.1736` |

  Paired against the accepted `200`-tree reference slice (candidate minus reference, 5000
  resamples, seed 0), weeks 3-18: deterministic Brier `+0.0007` `[-0.0011, +0.0024]`, margin MAE
  `+0.0122` `[-0.0566, +0.0801]`; all weeks `+0.0005` `[-0.0011, +0.0021]` and `+0.0166`
  `[-0.0477, +0.0801]`. By the written rule, a no-breakage tie. Unlike the narrower expectation in
  the pre-run hypothesis, 836 of 855 scored 2023-2025 rows moved in at least one feature between
  the `0.12.6` and `0.14.0` cuts, chiefly in the `sos_*` and `opponent_*` EPA families, so later
  arms on this build should treat the 54.0 reference as the current baseline rather than the older
  `0.12.6` reference arm.

  **Fit-noise floor on the same build** (2026-09-20): the reference arm rerun with only
  `--random-seed 7` (`models/wf_m59_rebuild_2023_2025_from_week1_seed7/`, checkpoints
  `models/wf_checkpoints/89dc69c3f18d74ad205b/`, hypothesis and rule in its `HYPOTHESIS.md`,
  rescored independently in its `REVIEW.md`). XGBoost runs with `subsample 0.6354` and
  `colsample_bytree 0.6098`, so the seed alone changes every fit. Weeks 3-18 against the
  reference arm: every margin moved (median `0.91`, p90 `2.34`, max `5.58` points), 46 picks
  flipped, deterministic Brier `0.2116` against `0.2096` (`+0.0020` `[-0.0006, +0.0045]`), pick
  accuracy `0.6847` against `0.6764` (`+0.0083` `[-0.0097, +0.0264]`), margin MAE `10.0271`
  against `9.9722`. The rebuild arm's differences from the benchmark above (median move `0.89`,
  45 flips, pick accuracy `-0.0097`) are the same size, so they are fit noise, not a
  division-correction effect. Consequence for reading any single three-season arm: a Brier
  difference under about `0.002`, a pick-accuracy difference under about `0.01` (7 games in 720)
  and a margin MAE difference under about `0.06` are indistinguishable from re-seeding; a claimed
  effect of that size needs six seasons or several seeds before it is a result.

  **Tree-budget ladder (task 55.7, 2026-09-20/21)**: three six-season rungs of the reference
  configuration on the rebuilt build (`data/completed_games_ml.m59_through_2025.csv`
  `cf42ec55...`, seasons 2020-2025 from week 1, `auto`, `market_anchor` on, four calibration
  weeks, seed `42`, git `9b9a7c8`), differing only in `--n-estimators`:
  `models/wf_m55_7_2020_2025_trees598/` (checkpoints `models/wf_checkpoints/8431001f74f2766a8c44/`),
  `models/wf_m55_7_2020_2025_trees200/` (`a5e76d54187e27ca7370`) and
  `models/wf_m55_7_2020_2025_trees400/` (`849c353f7074fedb0538`). Each rung carries a
  `HYPOTHESIS.md` written before its launch and a `REVIEW.md` from an independent rescore. The
  `598` rung reproduces the three-season reference arm's 2023-2025 folds bit for bit (identical
  `game_id` sets, maximum absolute difference exactly `0.0` on `predicted_margin`,
  `predicted_total` and `deterministic_home_win_prob` over all 816 rows), so it is the six-season
  reference on this build; every fold of every rung spent its whole budget with no early
  stopping, and `auto` resolved to the deterministic floor in all 107 folds.

  Weeks 3-18, `1423` games; all weeks, `1615` games. Market Brier on those rows is `0.2095` and
  `0.2104` respectively for every rung, because the market view depends on the rows and not on
  the fit.

  | window | rung | det Brier | market Brier | pick acc | margin MAE | total MAE | s/fold |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | weeks 3-18 | 200 | `0.2103` | `0.2095` | `0.6732` | `9.9281` | `10.3151` | `56` |
  | weeks 3-18 | 400 | `0.2111` | `0.2095` | `0.6676` | `9.9978` | `10.3712` | `111` |
  | weeks 3-18 | 598 | `0.2117` | `0.2095` | `0.6648` | `10.0240` | `10.4114` | `161` |
  | all weeks | 200 | `0.2110` | `0.2104` | `0.6693` | `9.8040` | `10.3219` | `56` |
  | all weeks | 400 | `0.2119` | `0.2104` | `0.6625` | `9.8806` | `10.3645` | `111` |
  | all weeks | 598 | `0.2125` | `0.2104` | `0.6607` | `9.9186` | `10.3940` | `161` |

  Pairwise differences (candidate minus reference, 5000 paired resamples, seed 0):

  | pair | window | Brier diff | Brier interval | MAE diff | MAE interval |
  | --- | --- | --- | --- | --- | --- |
  | 200 - 598 | weeks 3-18 | `-0.0014` | `[-0.0029, +0.0001]` | `-0.0958` | `[-0.1583, -0.0345]` |
  | 200 - 598 | all weeks | `-0.0015` | `[-0.0029, -0.0002]` | `-0.1146` | `[-0.1724, -0.0564]` |
  | 400 - 598 | weeks 3-18 | `-0.0006` | `[-0.0014, +0.0002]` | `-0.0262` | `[-0.0605, +0.0080]` |
  | 400 - 598 | all weeks | `-0.0007` | `[-0.0014, +0.0001]` | `-0.0380` | `[-0.0698, -0.0058]` |
  | 400 - 200 | weeks 3-18 | `+0.0008` | `[-0.0002, +0.0018]` | `+0.0697` | `[+0.0267, +0.1129]` |
  | 400 - 200 | all weeks | `+0.0009` | `[-0.0001, +0.0018]` | `+0.0766` | `[+0.0370, +0.1160]` |

  The aggregate order is monotone in the budget and points at fewer trees, but the per-season
  picture is not: margin MAE falls monotonically from `598` to `400` to `200` in five of six
  seasons (2025 is the exception), while deterministic Brier is monotone in only two of six
  (2020 and 2023). The monotone reading is an aggregate tilt, not a per-season law. Reproduce any
  rung with `.venv/bin/python models/wf_m55_7_2020_2025_trees400/compare_to_benchmark.py
  <rung_ckpt> <598_ckpt>` (the same script sits in each rung directory).

  Outcome: the ladder stopped on the `400` rung's "report all three and ask" branch. `400` ties
  `598` on Brier, `200` beats `400` on margin MAE beyond the fit-noise floor but not on Brier,
  and the weeks 3-18 `200 - 598` Brier interval covers zero by `+0.0000791`. On 2026-09-21 the
  user chose `200` as the shared default for the `0.13.0` code change. The accepted `100` rung
  later tied the new default in its independent review (`models/wf_m55_7_2020_2025_trees100/`
  and `REVIEW.md` there): governing weeks 3-18 against `200`, deterministic Brier `0.2103`
  against `0.2103`, diff `+0.0000` `[-0.0008, +0.0009]`; margin MAE `9.9237` against `9.9281`,
  diff `-0.0044` `[-0.0389, +0.0296]`. So `200` stays the default and task 55.7 is closed. The
  `1200` rung stays pre-written and never launched. The six-season fit-noise floor this ladder
  lacked was measured by task 55.8 (below). Open question: whether any further default change
  should wait for ample downtime if run mid-season. Timings, which are scheduling facts and
  not clean speed measurements: `598` about `161` s/fold (4h46m) under load from gates and the
  web API watcher, `200` about `56` s/fold (1h40m) and `400` about `111` s/fold (3h18m) on a
  machine quiet apart from that watcher.

  **Season weighting and the six-season fit-noise floor (task 55.8, 2026-09-23/24)**: nine
  six-season arms of the `200`-tree reference configuration on the `pbp`-default build
  (`data/completed_games_ml.m54_flip_through_2025.csv` `2d4111a6...`, seasons 2020-2025 from week
  1, `auto`, `market_anchor` on, four calibration weeks, CPU), differing only in
  `--recency-half-life-seasons` (none, `4`, `8`, `16`, `32`) and `--random-seed` (`42`, `7`;
  half-life `8` at seed 42 only). Run directories `models/wf_m55_8_2020_2025_*/`, each with its
  `HYPOTHESIS.md` or a pointer to it; the second key for all nine is
  `models/wf_m55_8_review/INDEPENDENT_REVIEW.md` (a separate session that produced none of the
  runs), reproduced with `.venv/bin/python models/wf_m55_8_review/independent_rescore.py`. The
  half-life `4`, `4` seed 7, `8`, `32` and `32` seed 7 arms were deleted on 2026-10-16 with their
  fold checkpoints (`.agents/models_cleanup/deleted_2026-10-16.txt`), so that rescore can no longer
  be fully rerun: its JSON names the deleted checkpoints `2b9b6a6c`, `359abe3f`, `41cb4904`,
  `92e351e4` and `f8efe644`; the numbers stand as reviewed. Every arm spent the full `200` trees in
  all 107 folds, and `auto` resolved to the deterministic floor. Weeks 3-18, `1423` games, market
  Brier `0.20948`:

  | arm | det Brier s42 / s7 | margin MAE s42 / s7 | pool pts s42 / s7 |
  | --- | --- | --- | --- |
  | unweighted | `0.21073` / `0.20959` | `9.9361` / `9.9105` | `8276` / `8346` |
  | half-life 4 | `0.21149` / `0.21180` | `10.0051` / `10.0121` | `8272` / `8246` |
  | half-life 8 | `0.21072` / - | `9.9673` / - | `8304` / - |
  | half-life 16 | `0.21016` / `0.20994` | `9.9522` / `9.9299` | `8277` / `8317` |
  | half-life 32 | `0.21100` / `0.20975` | `9.9571` / `9.9304` | `8251` / `8325` |

  Two seeds combined per game (candidate minus unweighted, averaged over the seeds, 5000 game
  resamples, seed 0), weeks 3-18: half-life `4` deterministic Brier `+0.00148`
  `[-0.00012, +0.00313]` and margin MAE `+0.0853` `[+0.0188, +0.1535]`; half-life `16` Brier
  `-0.00011` `[-0.00104, +0.00088]`; half-life `32` Brier `+0.00021` `[-0.00061, +0.00102]`. Over
  all weeks half-life `4` is worse beyond its intervals on Brier (`+0.00171`), log loss
  (`+0.00381`) and margin MAE (`+0.0916`). By the rules written before the runs, half-life `4`
  loses and neither `16` nor `32` is adopted; the user chose unweighted on 2026-09-24. Read it as
  "weighting at `16` or `32` shows no benefit, and weighting at `4` hurts", not as unweighted
  beating mild weighting. The one consistent signal in weighting's favor is week-2 total MAE
  (`0.2`-`0.4` points better for every weighted two-seed contrast; 96 games, non-governing,
  diagnostic-only head). The seed-42 ladder's own rule reached its "flat, keep the shipped `4`"
  fallback, a rule-12 case that the two-seed rules superseded.

  The same setting re-seeded (seed 7 minus seed 42, weeks 3-18) moved deterministic Brier by up to
  `0.00125`, pick accuracy by up to `0.0084` and pool points by up to `74` over the six seasons
  (32-50 picks flipped, median margin move `0.65`-`0.78` points), and some of those game-resampled
  intervals exclude zero: unweighted pick accuracy `+0.0084` `[+0.0007, +0.0162]` and pool points
  `+70` `[+14, +128]`; half-life `32` Brier `-0.00125` `[-0.00250, -0.00007]`. So on six seasons, a
  single-seed difference under about `0.0013` Brier, `0.008` pick accuracy, `75` pool points or
  `0.03` margin MAE is within re-seeding noise, whatever its interval says. This is the evidence
  behind rule 13.

  **How to read it now.** Walk-forward reports and `wf_compare` now carry three probability views:
  the configured calibrator, the deterministic map, and market-implied home win probability from
  the same rows (no-vig moneyline with a spread fallback). Feature work is ranked on the
  deterministic columns and on the paired deterministic-minus-market intervals; the default `auto`
  calibration path stays on the deterministic floor until a fitted calibrator proves it can beat
  that floor on the shared validation logic. Revalidated 2026-09-19 by
  `models/wf_m59_2020_2025_auto_floor_baseline/`: filtering its 2023-2025 folds reproduces the
  table above exactly, configured `auto` equals the deterministic floor in every 2023-2025
  window, no probability leaves `[0.02, 0.98]` unless `|predicted_margin| > 14`, and all 107
  six-season folds ran the full `598`-tree budget for both heads (`best_iteration = 597`,
  `early_stopped = false`; the budget itself is not tuned, see task 55.7).

  The same config on the pre-change build (`models/wf_shrink_2023_2025_off/`) gave week 2
  `0.2434` / `0.6799` / `0.5417` and weeks 3-18 `0.2293` / `0.7541` / `0.6847`; the comparison and
  its bootstrap intervals are in `.agents/ARCHIVE.md` under Milestone 49. To compare new feature
  work, run the reference arm on the same build and code version, with `--wf-start-week 1`
  whenever early-season handling could move.
- Older reference, measured on a superseded `0.4.0` build (dataset hash `668368d8...`) with
  `--wf-start-week 3`, kept for the strength-family ablation it records. Reports are under
  `models/wf_strength_2023_2025_{both_on,strength_off,both_off,prior_off}/`. Do not compare new
  arms against these numbers directly.

  | arm | Brier | log loss | pick acc | margin MAE | total MAE | ECE |
  | --- | --- | --- | --- | --- | --- | --- |
  | strength + play-by-play on | `0.2277` | `0.7431` | `0.6958` | `9.9006` | `10.1074` | `0.1321` |
  | strength off, play-by-play on | `0.2312` | `0.7495` | `0.6819` | `9.8698` | `10.1025` | `0.1430` |
  | both off | `0.2320` | `0.7493` | `0.6736` | `9.9772` | `10.0823` | `0.1237` |
  | strength on, prior blend off | `0.2277` | `0.7492` | `0.6847` | `9.8838` | `10.1043` | `0.1315` |

  The benchmark starts at week 3 (`--wf-start-week 3`). A separate run from week 1
  (`models/wf_strength_2023_2025_from_week1/`) measured the two weeks it skips, over 48 games
  each:

  | window | Brier | log loss | pick acc | margin MAE |
  | --- | --- | --- | --- | --- |
  | week 1 only | `0.2134` | `0.6151` | `0.6042` | `9.3640` |
  | week 2 only | `0.2445` | `0.6846` | `0.5208` | `8.7838` |
  | weeks 3-18 | `0.2277` | `0.7431` | `0.6958` | `9.9006` |

  That run exposed the early-season defect: week 2 (`0.5208` accuracy) ran on unshrunk one-game
  means. The stat prior blend fixed it on 2026-09-10 (current benchmark above). Week 1 remains the
  best-calibrated week because it runs entirely on the regressed prior season.

  What these arms still tell you: the strength group is the first family in this workstream to
  improve Brier and log loss rather than trade them for margin MAE; margin MAE and ECE do **not**
  improve alongside them. The early-season
  prior blend is a **tie on Brier** (`0.2277` either way) and earns its place only on log loss
  (`0.7431` vs `0.7492`) and pick accuracy (`0.6958` vs `0.6847`).
- The older reference (Brier `0.2312`, log loss `0.7352`, pick accuracy `0.6833`, margin MAE
  `9.8954`) is **not reproducible**: re-running the default config against the untouched pre-change
  dataset gives Brier `0.2300`, log loss `0.7501`, pick accuracy `0.6833`, margin MAE `9.9705`. Do
  not treat a gap against those old numbers as a regression. Compare arms only within a single
  dataset build and code version.

## Step 3 evidence (2026-09-27, no retraining beyond one fold)

Run directories this section names under `models/` (`step3_l0_gpu_check`, `step3_open_lines`,
`step3_prob_paths`) were deleted on 2026-10-16 (inventory in `.agents/models_cleanup/`); the numbers
stand as reviewed.

Each item below has its hypothesis and rule (written before scoring), the producer's script, and
an independent reviewer's own rescore and `REVIEW.md` in the run directory named.

- **GPU check, rung L0** (`models/step3_l0_gpu_check/`; reproduce with
  `.venv/bin/python models/step3_l0_gpu_check/independent_check.py`). One fold, 2025 week 18
  (16 games), of the unweighted six-season reference configuration, seed `42`. Two identical
  `--xgb-device cuda` legs were equal in every one of 520 numeric columns (GPU fits are
  deterministic for a fixed seed on this machine: RTX 4070 Ti, XGBoost 3.4.1). GPU minus CPU
  moved predicted margins by a median `0.61` points (p90 `1.67`, max `2.40`) and flipped one
  pick, the same order as re-seeding; 16 games cannot say it is smaller. The CPU leg reproduced
  the reference arm's fold bit for bit. Fold time `64` s on the CPU and `13` s on the GPU, with
  another process loading the GPU, so not a clean speed result.
- **Probability paths, part A2** (`models/step3_prob_paths/`; reproduce with
  `.venv/bin/python models/step3_prob_paths/independent_rescore.py`). Fixed and out-of-fold
  probability paths rescored on the two unweighted task 55.8 arms (seeds `42` and `7`,
  2020-2025, 1615 games each), path minus the deterministic floor, two seeds combined per game,
  all weeks: floor blended `0.2` with the raw moneyline probability and clamped at `0.1` Brier
  `-0.00052` `[-0.00102, -0.00003]`, log loss `-0.00136` `[-0.00249, -0.00024]`, the only path
  that beats the floor under the rule; floor clamp `0.1` alone `-0.00017` `[-0.00037, +0.00002]`;
  `elo` `-0.00061` `[-0.00205, +0.00086]`; the Week 2 production path (`elo`, blend `0.2`, clamp
  `0.1`) `-0.00089` `[-0.00231, +0.00055]`; out-of-fold Platt `-0.00000` `[-0.00136, +0.00139]`;
  out-of-fold isotonic `+0.00228` `[-0.00029, +0.00486]` with log loss `+0.083`
  `[+0.019, +0.163]`, pick accuracy `-0.0152` and pool points `-105` beyond their intervals
  (isotonic emits exact 0 and 1 probabilities).
- **Opening lines, part A3** (`models/step3_open_lines/`; reproduce with
  `.venv/bin/python models/step3_open_lines/independent_rescore.py`). The stored nflverse spread
  equals `data/nfl_lines.csv`'s last spread in `61%`-`80%` of 2020-2025 games (mean gap
  `0.14`-`0.33` points), a late snapshot. Spreads mapped to probabilities through one normal map:
  opening-line Brier minus stored-line Brier, all weeks, `+0.00481` `[+0.00255, +0.00714]`
  (2020-2025) and `+0.00462` `[+0.00163, +0.00755]` (2023-2025), so openers are measurably worse
  than the lines the backtests anchor to (the 2024-2025 "openers" are dated Tuesday or Wednesday
  scrapes, closer to pick time than a true open). The A2 blend rebuilt
  from the opening spread minus the floor: `+0.00088` `[-0.00033, +0.00209]` (2020-2025),
  `+0.00041` `[-0.00075, +0.00156]` (2023-2025); it does not beat the floor at pick-time lines.
  The model is anchored to the stored spread, so any "model minus opening line" gap flatters it.
  On 2007-2025 (market only, `data/completed_games_ml.m54_flip_through_2025.csv`, 4930 regular-
  season games with a real opener; reproduce with
  `.venv/bin/python models/step3_open_lines/independent_rescore_market_2007.py`), opening-line
  minus stored-line Brier, all weeks: `+0.00297` `[+0.00175, +0.00420]`; 2007-2019 `+0.00208`
  `[+0.00068, +0.00351]`; without 2022 `+0.00297` `[+0.00172, +0.00423]`. Weeks 1 and 2 alone
  (303 games each) include zero.

## GPU reference (2026-09-28, rungs L1-L2; superseded on 2026-10-09)

Run directories this section names under `models/` (`step3_gpu_reference`) were deleted on
2026-10-16 (inventory in `.agents/models_cleanup/`); the numbers stand as reviewed.

The reference for every arm from 2026-09-28 to 2026-10-09, replaced by the adopted-settings
reference (below): `models/step3_gpu_reference/l1_seed42/` and `l2_seed7/`
(checkpoints named in each `metadata.json`), `nfl-predictor backtest` on
`data/completed_games_ml.m54_flip_through_2025.csv` (`2d4111a6...`), seasons 2007-2025 from week
1 (328 folds, 4943 games), the floor, `market_anchor` and `market_transform` on, 200 trees,
unweighted, device `auto` (the GPU: an RTX 4070 Ti, every fold `cuda`), seeds `42` and `7`, code
`0.33.0` (commits `43d2a7d`/`42afe0d`, docs-only apart). About 32 minutes per seed. Rules and
launch record: `HYPOTHESIS.md` there; second key: `REVIEW.md` there (a separate reviewer), reproduce
with `.venv/bin/python models/step3_gpu_reference/independent_rescore.py`.

- Determinism: two GPU runs of the whole 2025 season (18 folds) are identical in every column.
- Against the CPU task 55.8 unweighted pair on 2020-2025, all weeks, two seeds (GPU minus CPU):
  deterministic Brier `+0.00015` `[-0.00069, +0.00098]`, pick accuracy `+0.0046`
  `[-0.0000, +0.0096]`, pool points (week-block bootstrap) `-2.5` `[-45.5, +39.5]`, margin MAE
  `+0.0213` `[-0.0121, +0.0550]`, all inside the six-season noise floors; weeks 3-18 pick accuracy
  `+0.0060` `[+0.0004, +0.0116]` is the one interval that excludes zero. The per-game GPU-vs-CPU
  margin move (median `0.62`-`0.66` points) is about the size of a reseed on either device. The
  CPU pair ran on code 246 commits older; the changes in between are documented as
  prediction-neutral.
- The retired market blend (`0.2` toward the raw moneyline, clamp `0.1`) minus the floor, 2007-2025
  with an opener (4930 games), all weeks, two seeds: at the stored moneyline Brier `-0.00072`
  `[-0.00105, -0.00041]`, log loss `-0.00173` `[-0.00244, -0.00104]`; built from the opening spread
  Brier `+0.00010` `[-0.00051, +0.00071]`, pick accuracy `-0.0041` `[-0.0078, -0.0006]`.
- The floor against the market, same games: against the stored spread Brier `+0.0003`
  `[-0.0007, +0.0013]` (a tie); against the opening spread `-0.00267` `[-0.00425, -0.00107]`
  (flattered: the model is anchored to the stored spread). The market-view pool-point columns are
  left out here: exact spread ties are ranked by float noise in `nfl-predictor compare`'s pool
  definition.
- Later code: `0.34.1` edited `nfl_predictor/ml/ml_model_training.py` (the final fit only), so
  runs on newer code cannot reuse these checkpoints; walk-forward predictions are unchanged (the
  walk-forward never calls the edited function; its reviewer's before/after synthetic run was
  hash-identical), so the reference stays comparable.

## The spread in the probability floor (2026-09-28, rescoring the GPU reference)

Run directories this section names under `models/` (`step3_sigma`) were deleted on 2026-10-16
(inventory in `.agents/models_cleanup/`); the numbers stand as reviewed.

`Phi(margin / sigma)` with the fixed `SCORE_DIFF_STD_DEV = 14.21` against sigmas estimated from the
model's own earlier errors (root-mean-square of actual minus predicted margin over strictly
earlier folds), scored on 2010-2025 (4175 games per seed, seeds 42 and 7), all weeks, candidate
minus fixed: `expanding` (all earlier folds, from 2007) Brier `-0.00021` `[-0.00039, -0.00004]`,
log loss `-0.00058` `[-0.00104, -0.00012]`; `trailing3` (the three seasons before plus the
current season's earlier weeks) Brier `-0.00028` `[-0.00059, +0.00002]`, log loss `-0.00078`
`[-0.00159, +0.00003]`; `trailing3` minus `expanding` includes zero everywhere. Picks and pool
points cannot change (one sigma per week keeps every side and every rank). The expanding sigma
falls from `13.93` (2010) to `13.27` (2025), the trailing one to `12.39`. Rules and data:
`models/step3_sigma/HYPOTHESIS.md`; second key `models/step3_sigma/REVIEW.md` (a separate
reviewer); reproduce with `.venv/bin/python models/step3_sigma/independent_rescore.py`.

Adopted by the user on 2026-09-28 as `0.35.0`: the floor uses the `expanding` sigma
(`src/nfl_predictor/ml/floor_sigma.py`); production pools the GPU reference runs with stage 1's weeks
the reference lacks (2026 week 1 sigma `13.248`, both seeds averaged per game).

## Market transform on against off (2026-09-28, task 56.7(b))

Run directories this section names under `models/` (`step3_market_transform`) were deleted on
2026-10-16 (inventory in `.agents/models_cleanup/`); the numbers stand as reviewed.

`models/step3_market_transform/`: the GPU reference's command on code `368570c` (`0.35.1`),
2007-2025 from week 1 (328 folds, 4943 games), seeds `42` and `7`, `--market-transform`
(`r_seed42`, `r_seed7`) against `--no-market-transform` (`t_seed42`, `t_seed7`); every fold on
the GPU, 200 trees in every head. Off minus on, all weeks, both seeds averaged per game:
deterministic Brier `+0.00003` `[-0.00022, +0.00028]`, log loss `-0.00001`
`[-0.00056, +0.00058]`, pick accuracy `-0.0019` `[-0.0043, +0.0005]`, pool points (week-block
bootstrap) `-31.0` `[-85.5, +20.0]`, margin MAE `+0.0013` `[-0.0093, +0.0120]`, total MAE
`+0.0024` `[-0.0087, +0.0135]`. No two-seed interval excludes zero in any window (all weeks,
week 1, week 2, weeks 3-18); on ranks first on every all-weeks column but log loss (by `0.00001`).
The Brier difference is about a seventeenth of the seed-7-minus-seed-42 move. Rules:
`HYPOTHESIS.md` there; second key `REVIEW.md` there (a separate reviewer); reproduce with
`.venv/bin/python models/step3_market_transform/independent_rescore.py`.

Decided by the user on 2026-09-28: a tie, `market_transform` stays `auto` (on whenever lines
exist).

The reference re-runs `r_seed42`/`r_seed7` have the GPU reference's margins and totals bit for
bit (`l1_seed42`/`l2_seed7`), which confirms that `0.34.1` left walk-forward predictions
unchanged. Their probabilities differ (up to `0.017`) because the GPU reference's recorded
Brier and log loss were scored at the fixed sigma, before `0.35.0`; `r_seed42`/`r_seed7` are the
reference's current-code equivalent (the pooled sigma, checkpoints named in `REVIEW.md`).

## Step 4, Phase B feature-value arms (2026-10-02, `models/step4/`)

Run directories this section names under `models/` (the contents of `step4/`) were deleted on
2026-10-16 (inventory in `.agents/models_cleanup/`); the numbers stand as reviewed.

Seven arms on code `c4dd24a` (`0.42.1`), each the GPU reference's command (2007-2025 from week 1,
328 folds, 4943 games, the floor with each run's own sigma pool, market anchor and transform on,
200 trees, unweighted, every fold `cuda`), seeds `42` and `7`, on five cold full-history builds
(`models/step4/inputs/<arm>.csv`, through-2025 cuts with `.sha256`; leakage audits all ok):
R0 `1fe38b9f...` (also QB and NO), S2 `fd82ddfa...`, S8 `4a41eac2...`, T2 `59720451...`, PT
`b24dde79...`. Every arm except QB passes `--disable-feature-groups qb_def_adj`. Rules:
`HYPOTHESIS.md` there (from `.agents/step4_plan.md`); second key `REVIEW.md` there (a separate
reviewer, recomputed from the fold checkpoints, `compare` matched exactly); reproduce with the
commands in `REVIEW.md` (`review/compare_<arm>.json`).

All weeks, two seeds averaged per game, candidate minus R0 [95%]: deterministic Brier / log loss /
pick accuracy / pool points / margin MAE.

- QB `-0.00001` `[-0.00051, +0.00048]` / `-0.00019` `[-0.00134, +0.00095]` / `-0.0020`
  `[-0.0053, +0.0014]` / `+23.5` `[-55.0, +100.5]` / `-0.0025` `[-0.0228, +0.0181]`: a tie.
- NO `+0.00002` `[-0.00051, +0.00054]` / `-0.00008` `[-0.00128, +0.00109]` / `+0.0001`
  `[-0.0034, +0.0036]` / `+35.5` `[-50.5, +122.0]` / `+0.0008` `[-0.0210, +0.0223]`: a tie.
- S2 `+0.00003` `[-0.00056, +0.00059]` / `-0.00009` `[-0.00144, +0.00120]` / `+0.0007`
  `[-0.0029, +0.0044]` / `+21.0` `[-76.5, +119.0]` / `-0.0027` `[-0.0279, +0.0199]`: a tie.
- S8 `+0.00033` `[-0.00030, +0.00098]` / `+0.00064` `[-0.00080, +0.00215]` / `-0.0025`
  `[-0.0062, +0.0011]` / `-20.5` `[-119.5, +77.0]` / `+0.0053` `[-0.0206, +0.0320]`: a tie.
- T2 `+0.00020` `[-0.00018, +0.00059]` / `+0.00030` `[-0.00057, +0.00119]` / `-0.0017`
  `[-0.0048, +0.0014]` / `-38.5` `[-104.0, +28.0]` / `+0.0091` `[-0.0066, +0.0254]`: a tie.
- PT `+0.00276` `[+0.00152, +0.00402]` / `+0.00623` `[+0.00346, +0.00902]` / `-0.0097`
  `[-0.0168, -0.0028]` / `-343.5` `[-525.0, -169.0]` / `+0.0861` `[+0.0367, +0.1339]`, worse on
  every column, as expected for the pick-time line. Against its own market PT is `+0.00073`
  `[-0.00050, +0.00193]` (R0 against the close `+0.00104` `[-0.00017, +0.00221]`); against the
  close (`compare --market-from` R0) `+0.00380` `[+0.00211, +0.00552]`, where the pick-time line
  itself trails the close by `+0.00307` `[+0.00173, +0.00449]`. No season's gap to its own
  market excludes zero. Stored-line fallback in the evaluation window: 2022 (all 284) and 13
  games of 2024.

Other intervals excluding zero (weeks 1, 2, 3-18, all weeks): T2 week-1 pool points `-17.5`
`[-34.5, -2.0]` (only under the week-block bootstrap; game-level `[-48, +13.5]`) and PT week-1
total MAE `-0.0641` `[-0.1198, -0.0087]` (PT better). Week-2 total MAE for the `K` arms includes
zero everywhere. All-weeks ranks (best first): Brier QB, R0, NO, S2, T2, S8, PT; log loss QB, S2,
NO, R0, T2, S8, PT; margin MAE S2, QB, R0, NO, S8, T2, PT; pool NO, QB, S2, R0, S8, T2, PT; pick
accuracy S2, NO, R0, T2, QB, S8, PT. The T2 build also moves the `qb_def_adj` values (they read
the strength snapshot), so a QB-plus-new-`K` combination would differ from what QB measured.

## Adopted-settings reference and market-free baselines (2026-10-09, superseded on 2026-10-14)

Run directories this section names under `models/` (the contents of `step4/`) were deleted on
2026-10-16 (inventory in `.agents/models_cleanup/`); the numbers stand as reviewed.

The GPU reference from 2026-10-09: `models/step4_reference/adopted_seed42/` and
`adopted_seed7/`, on code `2522cb6` (`0.44.0`), input `inputs/adopted.csv` (`855a9088...`, the
verbatim through-2025 cut of the rebuilt `data/`: pick-time lines, no defense-adjusted
quarterback columns, the next-opponent identity pair pruned), the GPU reference's command
(2007-2025 from week 1, 328 folds, 4943 games, the floor with each run's own error pool, market
anchor and transform on, 200 trees, unweighted, every fold `cuda`). Rules: `HYPOTHESIS.md`
there; second key: `REVIEW.md` there (a separate reviewer, recomputed from the fold checkpoints,
`compare` matched exactly); reproduce with the commands at the top of `REVIEW.md`
(`review/compare_*.json`). Two seeds averaged per game, candidate minus reference, 95% intervals.

- The reference against the pick-time lines test (`models/step4/PT_seed*`, the same games and
  inputs apart from the removed columns), all weeks: deterministic Brier `+0.00032`
  `[-0.00021, +0.00085]` (the rule's tie, so it was adopted), log loss `+0.00080`
  `[-0.00041, +0.00200]`, pick accuracy `+0.0029` `[-0.0007, +0.0066]`, pool points (week-block
  bootstrap) `+14.0` `[-68.0, +95.0]`, margin MAE `+0.0094` `[-0.0118, +0.0301]`. Week 1 favors
  the pick-time lines test (Brier `+0.00213` `[+0.00031, +0.00398]`, log loss `+0.00520`
  `[+0.00119, +0.00922]`), about the size of that arm's own seed-7-minus-seed-42 week-1 move
  (`+0.00217`). The reference's own seed-7-minus-seed-42 all-weeks Brier is `+0.00066`
  `[-0.00013, +0.00144]`.
- The market yardstick, all weeks: the reference minus the pick-time line `+0.00105`
  `[-0.00016, +0.00222]`, minus the closing line (`compare --market-from models/step4/R0_seed42`)
  `+0.00412` `[+0.00246, +0.00586]`; the pick-time line itself trails the close by `+0.00307`
  `[+0.00173, +0.00447]`.

The market-free baselines, same build and code (`marketfree_lines_seed*`: no betting-line column,
no anchor, no transform, 456 features; `marketfree_all_seed*`: also no Elo column, 438 features),
all weeks: deterministic Brier / log loss / pick accuracy / pool points (week-block bootstrap) /
margin MAE.

- Without betting lines minus the reference: `+0.00311` `[+0.00137, +0.00479]` / `+0.00717`
  `[+0.00335, +0.01090]` / `-0.0106` `[-0.0189, -0.0023]` / `-379.5` `[-636.5, -125.0]` /
  `+0.0459` `[-0.0231, +0.1137]`. Against the pick-time line `+0.00415` `[+0.00205, +0.00627]`,
  against the close `+0.00723` `[+0.00512, +0.00934]`.
- Without betting lines or Elo minus the reference: `+0.00394` `[+0.00208, +0.00574]` /
  `+0.00880` `[+0.00468, +0.01284]` / `-0.0110` `[-0.0195, -0.0024]` / `-524.0`
  `[-788.0, -253.5]` / `+0.0796` `[+0.0056, +0.1507]`. Against the pick-time line `+0.00498`
  `[+0.00276, +0.00719]`, against the close `+0.00806` `[+0.00584, +0.01032]`.
- Without Elo minus with Elo (both without lines): `+0.00083` `[+0.00002, +0.00161]` / `+0.00163`
  `[-0.00019, +0.00340]` / `-0.0004` `[-0.0053, +0.0048]` / `-144.5` `[-259.5, -30.0]` (by game
  `[-336.0, +42.0]`) / `+0.0337` `[+0.0020, +0.0642]`.
- Weeks 1 and 2: neither market-free run separates from the reference on any column except total
  MAE for the one without Elo. By season, the run without lines trails in 14 of 19 seasons and
  the one without lines or Elo in 15 (largest gaps 2013, 2007 and 2022; in 2022 every pick-time
  line is the stored close).
- The Elo probabilities in `data/qb_elos.csv` (partly market-informed: `../nfeloqb` pulls each
  preseason Elo toward betting win totals, 2003-2023) on the same 4943 games: Brier QB Elo
  `0.21620`, 538-style Elo `0.22001`, the pick-time line `0.21286`, the closing line `0.20979`,
  the reference `0.21391`. The reference minus QB Elo `-0.00229` `[-0.00457, -0.00001]`; the run
  without lines `+0.00081` `[-0.00131, +0.00295]` and the one without lines or Elo `+0.00164`
  `[-0.00065, +0.00393]`, both ties with QB Elo.

## Step 6 reference arms (2026-10-12 and 2026-10-13, `models/step6/`)

The reference production adopts on 2026-10-14: `models/step6/ratings_anchored_seed42/` and
`ratings_anchored_seed7/` (checkpoints `d97fab54e7261f18e349`, `2f7442e89fd6ff3981f4`), the ported
ratings in place of the strength snapshot with the own Elo off (`--disable-feature-groups
strength,own_elo,own_elo_wt`, production's default until `0.60.0`, 449 features). They also
supplied the floor's sigma (`constants.FLOOR_SIGMA_REFERENCE_RUNS`) until `0.62.0`, when the
new-file runs of "`qb_elos.csv` measurement" below replaced them. Input
`models/step6/inputs/candidate.csv` (`74b21113d167...`, the through-2025 cut of the step-6 build:
the ported ratings, the own Elo and the play-by-play corrections), code `d6064eb`, and otherwise
the 2026-10-09 reference's command
(2007-2025 from week 1, 328 folds, 4943 games, market anchor and transform on, 200 trees,
unweighted, every fold `cuda`). Rules: `HYPOTHESIS.md` there (sha256
`e027793932a519e26963210370f27861d0743fbc9b59951318883edc8c396ed9`, the amended text; the checksum
before the amendment, `5b91d1d4a0b60810b1ace9db1ad7586f17778596301ccd118db083f5099ba976`, is in
`HYPOTHESIS.original.sha256`, and the original text is not on disk) and
`nogamelines_wt/HYPOTHESIS.md` (sha256
`08bc8c7fdd902e730c0f9de31f41cf657f35aaf11b8a75eeb94959a649970caf`); both matched their `.sha256`
files on 2026-10-16. Second key: `REVIEW.md` in each, by reviewers that did not produce the runs.
Two seeds averaged per game, 95% intervals, all weeks unless named.

The other reference arms on the same build and code:

- Without betting lines or `qb_elos.csv`, with the own Elo (no line column, no anchor, no
  transform; `--disable-feature-groups strength,elo,own_elo_wt`, 431 features):
  `elo_marketfree_seed{42,7}` (`d8462a7de8d460c12491`, `54ec3f119f2d50ad6b14`).
- Without betting lines or any Elo (`strength,elo,own_elo,own_elo_wt`, 427 features):
  `ratings_marketfree_seed{42,7}` (`75186da854af9d4eb24c`, `b074c98e1196a1da8af2`).

**The `qb_adj` columns and these references (approved by the user on 2026-10-15, rule 11).**
Since `0.60.0` production also leaves out `qb_adj` (`strength,own_elo,own_elo_wt,qb_adj`). The
step-6 build has no `qb_adj` column, so the reference runs trained on the same columns production
does, and the only effect of the longer list is a difference listed in `compare`'s
production-settings section; nothing in the weekly run, `train` or `backtest` stops or warns on
it (independent review, 2026-10-15). The reference production reads from `0.62.0` was measured on
a build with the columns, with `qb_adj` left out as production does ("`qb_elos.csv` measurement"
below). On such a build, each list above reproduces with `qb_adj` added.

Results (deterministic Brier and log loss, other columns where an interval excludes zero):

- The ported ratings in place of the strength snapshot, anchored, minus the 2026-10-09 reference:
  `-0.00084` `[-0.00144, -0.00024]` and `-0.00191` `[-0.00326, -0.00057]`; pool points `+77.5`
  `[-15.0, +171.0]`; weeks 1 and 2 ties. Adopted.
- That split by the new build with the strength snapshot (`newref_anchored_seed*`): the
  play-by-play corrections' share is a tie on every column in every window (Brier `+0.00011`
  `[-0.00040, +0.00062]`), kept as correctness fixes; the ratings' share wins (`-0.00095`
  `[-0.00153, -0.00039]`, `-0.00218` `[-0.00349, -0.00092]`, margin MAE `-0.0257`
  `[-0.0492, -0.0029]`).
- The ported ratings without betting lines or Elo, minus the 2026-10-09 test bench without them:
  a tie (`-0.00032` `[-0.00114, +0.00049]`), with a week-1 loss flagged (`+0.00321`
  `[+0.00045, +0.00596]`).
- The own Elo in place of the `qb_elos.csv` columns, anchored (`elo_anchored_seed*`): a loss
  (`+0.00069` `[+0.00017, +0.00122]`, `+0.00154` `[+0.00035, +0.00275]`, pool points `-114.0`
  `[-199.0, -29.5]`), so `qb_elos.csv` stays and the own Elo is off in production.
- The own Elo added without betting lines or `qb_elos.csv`: a win (`-0.00093`
  `[-0.00177, -0.00005]`, `-0.00224` `[-0.00409, -0.00034]`), so that configuration keeps it.
- Its win-total variant there (`nogamelines_wt/elo_wt_nogamelines_seed*`, `own_elo_wt` in place
  of `own_elo`): a tie on every column (`-0.00012` `[-0.00080, +0.00055]`, `-0.00030`
  `[-0.00179, +0.00119]`), so `own_elo` stays.
- Against the pick-time line on the same rows: production's reference `+0.00021`
  `[-0.00102, +0.00138]` (the 2026-10-09 reference `+0.00105` `[-0.00014, +0.00223]`), with week 1
  `+0.00506` `[+0.00001, +0.01012]` and margin MAE `+0.0500` `[+0.0087, +0.0932]` against it; the
  arm with the own Elo and no lines `+0.00373` `[+0.00161, +0.00580]`; the arm with no lines or
  Elo `+0.00466` `[+0.00249, +0.00683]`.

**The kickoff starter, a shared convention.** Every quarterback-dependent column (the
`qb_elos.csv` columns, the ported quarterback rating, the own Elo and the quarterback families)
rates a completed game for its actual starter from the play-by-play, which is known only at
kickoff. The weekly run rates upcoming games for the starters the user projects by hand before
running `../nfeloqb`. Every backtest arm alike therefore knows each past game's starter; the user
ruled on 2026-10-12 that this is the project's convention, not a caveat on any adoption. Knowing
only the previous starter costs the own Elo without lines `+0.00101` `[-0.00027, +0.00232]` Brier
and `+0.00331` `[+0.00035, +0.00631]` log loss (`.agents/sos_port/decisions/elo.md`, entry 15).

**`qb_elos.csv` after the upstream Elo fix (2026-10-14, approved by the user; measured on
2026-10-15).** Since Week 6 of 2026, `data/qb_elos.csv` comes from `../nfeloqb` with upstream's
`90e19d9` merged (fork commit `27709b5`). That commit enters the quarterback adjustment in the
QB-adjusted probability as home minus away (the fork added both) and tracks the plain Elo and the
QB-adjusted Elo separately, as 538's original file does. The rows before the 2023-02-12 cutover
are unchanged; from 2023 on, `elo1_pre` moves by 12.3 Elo points on average (110 at most),
`qbelo1_pre` by 25.2 (142 at most), and `qb1_value_pre` not at all (`verify.py` and the file
hashes in `~/scratch/data_backups/2026-10-14_pre_week6/`, beside the replaced files). Every step-6
reference arm above was measured on the earlier file, so production differs from them in these
four columns from 2023 on. "`qb_elos.csv` measurement" below measures the effect with two
dataset builds. The stale win-total file (`../nfeloqb` `Manual Data/wt_ratings.csv` ends at 2023)
has its own measurement, "Upstream win totals in the quarterback-adjusted Elo" below.

## `qb_elos.csv` measurement (2026-10-15, `models/qb_elos_measure/`)

Eight runs, four arms on two seeds each, all in `models/qb_elos_measure/` (checkpoint directories
for seeds 42 and 7):

- The old file with production's features (`--disable-feature-groups
  strength,own_elo,own_elo_wt,qb_adj`, 449 features): `old_file_seed{42,7}`
  (`c11316698f8b83a57544`, `988441baadcd2d3e2634`).
- The new file with production's features (the same groups, 449 features):
  `new_file_seed{42,7}` (`3bd2b79b36c6ed70e0e5`, `98696fa36a5ae245740b`).
- The new file with the quarterback adjustment (`strength,own_elo,own_elo_wt`, 452 features):
  `qb_adj_seed{42,7}` (`a6acc6d874d9b3bdb00b`, `90d8d9374d2d9cc3d55c`).
- The new file with the strength snapshot in place of the ported ratings
  (`ratings,own_elo,own_elo_wt,qb_adj`, 460 features): `strength_seed{42,7}`
  (`ab173f4fb240abfe577d`, `be38b8d37d9ab698570c`).

Inputs: `inputs/new_file.csv` (`b9772e3e...49f0`, the through-2025 cut with `data/qb_elos.csv` as
copied on 2026-10-14) and `inputs/old_file.csv` (`b59366e2...0af3`, the same cut with the replaced
copy). Both have 7,261 rows and 544 columns; they differ in 12 columns, all in the `elo` group,
all in the 855 rows of 2023-2025, and the `qb_adj` columns are identical. Code `c637fc0`
(`0.60.0`) in all eight runs. Command (`driver.sh`): `nfl-predictor backtest --calibration auto
--eval-seasons 2007..2025 --wf-start-week 1 --random-seed {42,7} --market-anchor
--market-transform --disable-feature-groups <arm's groups>`: 328 folds, 4943 games, 200 trees
with no early stop, calibration `none` in every fold, unweighted, every fold `cuda`, each run's
own floor-sigma pool. Rules: `HYPOTHESIS.md` there (checksummed before the builds, sha256
`d682a0c0e4330614df0e866e62a3e628990ea1595d96c419ac26ac8edc1406fb`, which matched its `.sha256`
file on 2026-10-16); second key:
`REVIEW.md` there, by a reviewer that did not produce the runs. The old-file runs reproduce the
step-6 reference (`models/step6/ratings_anchored_seed{42,7}`) bit for bit, and the two files'
runs are identical game for game before 2023. Two seeds averaged per game, candidate minus
reference, 95% intervals.

Results, by rule:

- The new file minus the old, governing window seasons 2023-2025, all weeks (816 games): Brier
  `-0.00026` `[-0.00082, +0.00030]`, log loss `-0.00070` `[-0.00195, +0.00055]`. A tie: the new
  file stays (already in production; nothing changes).
- The quarterback adjustment minus the new file, all weeks: Brier `+0.00025`
  `[-0.00026, +0.00076]`, log loss `+0.00056` `[-0.00056, +0.00170]`. A tie: the `qb_adj` group
  stays off (the simpler setting). The direction does not hold across seeds.
- The strength snapshot minus the new file (the ported ratings re-checked on the new file), all
  weeks: Brier `+0.00102` `[+0.00045, +0.00159]`, log loss `+0.00238` `[+0.00111, +0.00366]`,
  pool points `-95.5` `[-187.5, -4.0]`, margin MAE `+0.0280` `[+0.0049, +0.0511]`. The swap
  loses, on each seed alone too: the ported ratings stay.
- Weeks 1 and 2: no two-seed interval excludes zero in either week in any comparison.

Every paired interval `REVIEW.md` lists as excluding zero (rule 9):

- The new file minus the old: none, on two seeds or on either seed alone.
- The quarterback adjustment minus the new file, two seeds: weeks 3-18 pool points `-82.0`
  `[-156.5, -8.0]`; 2023-2025 week 2 (outside the instrument) pool points `-6.5`
  `[-13.5, -3.0]`. Seed 42 alone: week 1 Brier `-0.00254` `[-0.00500, -0.00009]` and log loss
  `-0.00538` `[-0.01077, -0.00002]`, for the quarterback adjustment. Seed 7 alone: weeks 3-18
  Brier `+0.00090` `[+0.00013, +0.00167]` and log loss `+0.00210` `[+0.00036, +0.00386]`;
  2023-2025 week 2 (outside the instrument) pool points `-10.0` `[-15.0, -6.0]`.
- The strength snapshot minus the new file, two seeds, all against the strength snapshot: weeks
  3-18 Brier `+0.00123` `[+0.00063, +0.00185]`, log loss `+0.00283` `[+0.00148, +0.00422]`,
  margin MAE `+0.0344` `[+0.0094, +0.0589]`, pool points `-97.5` `[-178.5, -13.0]`; all weeks
  Brier `+0.00102` `[+0.00045, +0.00159]`, log loss `+0.00238` `[+0.00111, +0.00366]`, margin MAE
  `+0.0280` `[+0.0049, +0.0511]`, pool points `-95.5` `[-187.5, -4.0]`; 2007-2022 all weeks
  (outside the instrument) Brier `+0.00098` `[+0.00034, +0.00161]`, log loss `+0.00226`
  `[+0.00086, +0.00368]`, margin MAE `+0.0291` `[+0.0030, +0.0551]`.
- The same, seed 42 alone: weeks 3-18 Brier `+0.00124` `[+0.00041, +0.00208]`, log loss
  `+0.00287` `[+0.00099, +0.00475]`; all weeks Brier `+0.00105` `[+0.00029, +0.00182]`, log loss
  `+0.00250` `[+0.00079, +0.00423]`; 2007-2022 all weeks (outside the instrument) Brier `+0.00126`
  `[+0.00039, +0.00211]`, log loss `+0.00303` `[+0.00106, +0.00496]`.
- The same, seed 7 alone: weeks 3-18 Brier `+0.00122` `[+0.00038, +0.00206]`, log loss `+0.00280`
  `[+0.00091, +0.00470]`, margin MAE `+0.0370` `[+0.0029, +0.0713]`; all weeks Brier `+0.00100`
  `[+0.00022, +0.00180]`, log loss `+0.00226` `[+0.00048, +0.00405]`, margin MAE `+0.0327`
  `[+0.0009, +0.0644]`; outside the instrument, 2023-2025 all weeks Brier `+0.00258`
  `[+0.00084, +0.00435]`, log loss `+0.00613` `[+0.00233, +0.01003]`, margin MAE `+0.0756`
  `[+0.0094, +0.1431]`, and 2023-2025 weeks 3-18 Brier `+0.00250` `[+0.00065, +0.00437]`, log
  loss `+0.00603` `[+0.00194, +0.01013]`, margin MAE `+0.0789` `[+0.0068, +0.1520]`.
- Against the pick-time line on each arm's own rows (two seeds), all for the line: the old file
  week 1 margin MAE `+0.2037` `[+0.0326, +0.3761]`, all weeks `+0.0500` `[+0.0072, +0.0924]`; the
  new file week 1 Brier `+0.00511` `[+0.00007, +0.01030]`, week 1 margin MAE `+0.2107`
  `[+0.0399, +0.3824]`, all weeks `+0.0484` `[+0.0057, +0.0909]`; the quarterback adjustment week 1
  margin MAE `+0.1673` `[+0.0005, +0.3360]`, weeks 3-18 margin MAE `+0.0532` `[+0.0076, +0.0985]`
  and total MAE `+0.0503` `[+0.0025, +0.0979]`, all weeks margin MAE `+0.0534`
  `[+0.0113, +0.0958]`; the strength snapshot week 1 margin MAE `+0.2034` `[+0.0292, +0.3786]`,
  weeks 3-18 `+0.0738` `[+0.0281, +0.1198]`, all weeks log loss `+0.00273` `[+0.00002, +0.00544]`
  and margin MAE `+0.0764` `[+0.0344, +0.1198]`. No arm beats the line on any column in any
  window.
- Seed 7 minus seed 42 within an arm: a tie in every instrument window; in seasons 2023-2025, the
  old file margin MAE `-0.0711` `[-0.1354, -0.0068]`, the new file log loss `-0.00360`
  `[-0.00710, -0.00016]`, the quarterback adjustment total MAE `-0.0813` `[-0.1510, -0.0132]`.

Ranks, all weeks, two-seed means with the line: Brier and log loss: the line 1, the new file 2,
the old file 3, the quarterback adjustment 4, the strength snapshot 5. Pool points: the new file
1, the old file 2, the quarterback adjustment 3, the line 4, the strength snapshot 5. Margin MAE:
the line 1, the new file 2, the old file 3, the quarterback adjustment 4, the strength snapshot 5.
The rank table for every window and column is in `REVIEW.md`.

**The floor-sigma reference.** Since `0.62.0`, `new_file_seed42` and `new_file_seed7` are
production's floor-sigma reference (`constants.FLOOR_SIGMA_REFERENCE_RUNS`), by the user's
decision on 2026-10-15. They replace the step-6 runs, which trained on the `qb_elos.csv` before
its refresh. From `REVIEW.md`: "the sigma moves by under `0.005` points (final
`13.3184`/`13.3182` against `13.3195`/`13.3227`), so probabilities barely change".

## The quarterback adjustment without game lines (2026-10-15, `models/qb_adj_no_lines/`)

Whether the run without game lines gains from the quarterback adjustment group (`qb_adj`:
`away_qb_adj`, `home_qb_adj`, `qb_adj_diff`), which the market-anchored model measured as a tie
above. Four runs, two arms on two seeds each, all in `models/qb_adj_no_lines/` (checkpoint
directories for seeds 42 and 7):

- The reference, the default without game lines (`--disable-feature-groups
  strength,elo,own_elo_wt,qb_adj`, 431 features): `no_lines_seed{42,7}`
  (`524400106ab720ccfb3e`, `6e8ec19833318631e24d`).
- The candidate, the same with the quarterback adjustment kept (`strength,elo,own_elo_wt`, 434
  features): `no_lines_qb_adj_seed{42,7}` (`2108d71fbe1e0b36f5df`, `7e1e9343fb8596950136`).

Input: `models/qb_elos_measure/inputs/new_file.csv` (`b9772e3e...49f0`, the new-file cut of the
measurement above, no new build); its `qb_adj` columns have no nulls from 2007 on. Code `d791846`
(`0.62.0`) in all four runs. Command (`driver.sh`): `nfl-predictor backtest --calibration auto
--eval-seasons 2007..2025 --wf-start-week 1 --random-seed {42,7} --exclude-market
--no-market-anchor --no-market-transform --disable-feature-groups <arm's groups>`: 328 folds, 4943
games, 200 trees with no early stop, calibration `none` in every fold, unweighted, every fold
`cuda`, each run's own floor-sigma pool. Rules: `HYPOTHESIS.md` there (sha256
`8c1028438a66594c8ecb1445ed8fcf4403ccf0e7a8c2cd3a1f1ed4c34519ce06`, committed on `main` in
`7b1560b` before the launch, unamended, and matching its `.sha256` file on 2026-10-16); second
key: `REVIEW.md` there, by a reviewer that did not
produce the runs. Two seeds averaged per game, candidate minus reference, 95% intervals.

Provenance (`REVIEW.md`): each candidate differs from its reference only in the disabled groups,
and its features only by the three `qb_adj` columns; every run computed all 328 folds and restored
none, the four ran back to back with no overlap, and the reference runs reproduce the step-6 runs
without game lines (`models/step6/elo_marketfree_seed{42,7}`) bit for bit on all 4,943 games. The
reviewer's own rescore from the fold checkpoints matches `nfl-predictor compare` on every point
estimate and on which intervals exclude zero.

Results, by rule (governing window all weeks; Brier selects, log loss must not lose, then pool
points and margin MAE break ties):

- All weeks: Brier `+0.00007` `[-0.00050, +0.00064]`, log loss `+0.00027` `[-0.00100, +0.00151]`,
  pool points `+19.0` `[-84.0, +120.5]`, margin MAE `-0.0053` `[-0.0275, +0.0174]`. A tie, the
  rule's second branch and not between branches: `qb_adj` stays in
  `constants.NO_GAME_LINES_DISABLED_FEATURE_GROUPS` (the simpler setting), and no default changes.
  The two seeds' all-weeks Brier contrasts (`-0.00004`, `+0.00018`) straddle zero.
- Weeks 1 and 2: no two-seed interval excludes zero on any column, so nothing is flagged. The point
  estimates lean the hypothesised way: Brier week 1 `-0.00049` `[-0.00255, +0.00162]`, week 2
  `-0.00063` `[-0.00289, +0.00167]` (weeks 3-18 `+0.00016` `[-0.00045, +0.00075]`); log loss
  `-0.00084` `[-0.00522, +0.00364]` and `-0.00137` `[-0.00624, +0.00352]`; pool points `+11.0`
  `[-19.0, +40.0]` and `+8.0` `[-29.5, +46.5]`; margin MAE `-0.0200` `[-0.1014, +0.0618]` and
  `-0.0366` `[-0.1233, +0.0519]`; pick accuracy `+0.0066` in both.

Every paired interval `REVIEW.md` lists as excluding zero (rule 9):

- The candidate minus the reference: none, in any window, on two seeds or on either seed alone.
- Against the pick-time line on each arm's own rows (two seeds), all for the line: the candidate
  weeks 3-18 Brier `+0.00370` `[+0.00149, +0.00592]`, log loss `+0.00846` `[+0.00342, +0.01358]`,
  pool points `-316.5` `[-576.5, -52.5]`, margin MAE `+0.1139` `[+0.0393, +0.1888]`, total MAE
  `+0.1863` `[+0.1114, +0.2602]`; all weeks Brier `+0.00381` `[+0.00172, +0.00586]`, log loss
  `+0.00865` `[+0.00385, +0.01341]`, pool points `-362.5` `[-661.5, -71.5]`, margin MAE `+0.1143`
  `[+0.0417, +0.1832]`, total MAE `+0.1944` `[+0.1237, +0.2651]`. The reference weeks 3-18 Brier
  `+0.00354` `[+0.00133, +0.00577]`, log loss `+0.00800` `[+0.00293, +0.01312]`, pool points
  `-316.5` `[-580.0, -61.0]`, margin MAE `+0.1160` `[+0.0405, +0.1914]`, total MAE `+0.1831`
  `[+0.1062, +0.2573]`; all weeks Brier `+0.00373` `[+0.00162, +0.00579]` (the step-6 record's
  `+0.00373` `[+0.00161, +0.00580]`), log loss `+0.00839` `[+0.00351, +0.01313]`, pool points
  `-381.5` `[-677.0, -92.5]`, margin MAE `+0.1197` `[+0.0469, +0.1901]`, total MAE `+0.1923`
  `[+0.1204, +0.2632]`. Neither arm's interval against the line excludes zero in week 1 or week 2,
  or on pick accuracy in any window.
- Seed 7 minus seed 42 within an arm (outside the instrument): the candidate's total MAE in week 1
  `-0.1428` `[-0.2698, -0.0136]`, weeks 3-18 `-0.0386` `[-0.0762, -0.0011]` and all weeks
  `-0.0449` `[-0.0800, -0.0099]`, for seed 7; the reference's all-weeks margin MAE `+0.0360`
  `[+0.0020, +0.0692]`, for seed 42. A seed moves the error columns about as much as the arms
  differ.

Ranks, two-seed means with the line: the line ranks first on every column in every window. All
weeks, the reference leads on Brier, log loss and total MAE, the candidate on pick accuracy, pool
points and margin MAE. Weeks 3-18 the same, except that the arms tie on pool points (24,590.5
each). Weeks 1 and 2, the candidate leads on every column except week 2 total MAE. The rank table
for every window and column is in `REVIEW.md`.

**The floor sigma in a weekly run without game lines: a rule-11 gap, fixed in `0.63.0`.** The
reviewer's one
rule-11 finding (severity P2). The reference arm is what `nfl-predictor weekly --wf-market-mode
none` (and `train` and `backtest` without market features or anchor) does, except for the floor
sigma's source. The weekly run loads its floor-sigma reference runs in every market mode
(`weekly_run/pipeline.py`, `floor_sigma.load_reference_pool(args.floor_sigma_reference_runs)`;
`ml/floor_sigma.py` has no market-mode branch). Those are the market-anchored
`models/qb_elos_measure/new_file_seed{42,7}` (`config/weekly_run.yaml`,
`constants.FLOOR_SIGMA_REFERENCE_RUNS`), so a weekly run with `--wf-market-mode none` would take its
sigma from the anchored model's out-of-fold errors (final pooled `13.3184` and `13.3182`), while
every walk-forward without game lines, these runs and step 6's, pools its own (final `13.3969` and
`13.4341` for the reference). That weekly run's probabilities would be slightly sharper than the
ones measured here (a sigma about `0.1` lower). The user chose the fix on 2026-10-15: since
`0.63.0` a run without game lines defaults to `constants.NO_GAME_LINES_FLOOR_SIGMA_REFERENCE_RUNS`,
this measurement's reference arm (`no_lines_seed{42,7}`). The weekly run with game lines is
unaffected, and so is this verdict: both arms pooled their own errors.

The reviewer's other findings, all minor (P3): `compare`'s production-settings section always
compares with the weekly config with game lines, so for runs without them it lists the expected
market and group differences and misses real gaps such as the floor sigma's source (suggested:
compare with the config resolved for `wf_market_mode: none` and include the sigma's source); the
driver's check for other runs ran only at each run's start and its pattern misses other launch
forms (no effect here); and the seeds disagree in sign on pool points (`+92.0` against `-54.0`, all
weeks) and on total MAE, which is why the two-seed mean is the instrument. Full tables,
provenance, findings and reproduction scripts: `models/qb_adj_no_lines/REVIEW.md`.

## Upstream win totals in the quarterback-adjusted Elo (2026-10-16, `models/qb_wt_measure/`)

Whether the quarterback-adjusted Elo in `qb_elos.csv` (`qbelo1_pre`, the source of the
`*_qb_elo_pre` features) measures better with upstream's maintained win-total file
(`greerreNFL/nfelosrs` `wt_ratings.csv`, 2003-2026) in place of the fork's (`../nfeloqb`
`Manual Data/wt_ratings.csv`, 2003-2023). At each season start the fork pulls `qbelo` 66% toward
the season's win-total rating, so from 2024 on the fork's file leaves it only regressing to the
mean. Four runs, two arms on two seeds each (checkpoint directories for seeds 42 and 7):

- The candidate, upstream win totals: `models/qb_wt_measure/upstream_wt_seed{42,7}`
  (`60c64046f68b798af669`, `2b29e91c047c82fd228e`), on `models/qb_wt_measure/inputs/upstream_wt.csv`
  (sha256 `1061d14130f867d9206a71cbd374f938d495ae3abba4f6d740a67437a763d8cc`), the through-2025 cut
  built from the fork's `qb_elos.csv` regenerated at `27709b5` with only the win-total file
  replaced (fork branch `exp/upstream-wt-ratings` `847b6b9`, upstream `nfelosrs` `3c6d4ae`).
- The reference, fork win totals: `models/qb_elos_measure/new_file_seed{42,7}`
  (`3bd2b79b36c6ed70e0e5`, `98696fa36a5ae245740b`), on `models/qb_elos_measure/inputs/new_file.csv`
  (`b9772e3e...49f0`): the new-file runs of "`qb_elos.csv` measurement" above and production's
  floor-sigma reference, not rerun.

The two cuts (7,261 rows, 544 columns) differ in 6 columns, all in the `elo` group
(`{home,away}_qb_elo_pre`, `{home,away}_qb_elo_4wk_trend`, `qb_elo_pre_diff`,
`qb_elo_4wk_trend_diff`), only in the 855 rows of 2023-2025: every `qb_elos.csv` row before
2023-02-12 is copied from 538's historical file, so the win totals reach the file only through the
state the fork's Elo carries into 2023. Code `c637fc0` (`0.60.0`) in all four runs. Command
(`driver.sh`): `nfl-predictor backtest --data-path inputs/upstream_wt.csv --calibration auto
--eval-seasons 2007..2025 --wf-start-week 1 --random-seed {42,7} --market-anchor
--market-transform --disable-feature-groups strength,own_elo,own_elo_wt,qb_adj`: 449 features, 328
folds, 4943 games, 200 trees with no early stop, calibration `none` in every fold, unweighted,
every fold `cuda`, each run's own floor-sigma pool. Rules: `HYPOTHESIS.md` there with its
Amendment 1 (sha256 `f26039b61ebe6639d3d717821b2198863a0dfde60fff91d00999ba3fd4a6d7a7`, written
before the build and the launch and matching its `.sha256` file at the review). Amendment 1 moved
the governing window from all weeks of 2007-2025 to seasons 2023-2025, all weeks (816 games),
once the candidate's check showed that only those rows change. Second key: `REVIEW.md` there, by
a reviewer that did not produce the runs. Two seeds averaged per game, upstream minus fork, 95%
intervals.

Provenance (`REVIEW.md`): a pass on every check. Each candidate differs from its reference only in
the input; the leakage audit passed; all 4,127 games before 2023 are bit-identical between the
arms in both seeds, including seed 7, which shared the GPU with another process and took 9,934 s
against 5,165 s for seed 42. The reviewer's rescore from the fold checkpoints matches
`nfl-predictor compare` on every run level and point estimate and on which intervals exclude zero.
`compare` has no seasons filter, so the season windows rest on the rescore alone.

Upstream minus fork, two seeds (loss columns and MAEs: negative favours upstream; pick accuracy
and pool points: positive favours upstream):

| window | games | det Brier | log loss | pick acc | pool pts | margin MAE | total MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **2023-2025 all weeks (governing)** | 816 | -0.00006 [-0.00057, +0.00044] | -0.00009 [-0.00124, +0.00104] | +0.0018 [-0.0025, +0.0061] | +2.5 [-17.0, +21.5] | +0.0029 [-0.0171, +0.0228] | +0.0062 [-0.0147, +0.0276] |
| 2024-2025 all weeks | 544 | -0.00041 [-0.00103, +0.00021] | -0.00080 [-0.00219, +0.00059] | +0.0000 [-0.0055, +0.0055] | +6.5 [-8.0, +21.0] | -0.0064 [-0.0310, +0.0183] | +0.0145 [-0.0120, +0.0418] |
| 2007-2025 all weeks | 4943 | -0.00001 [-0.00009, +0.00007] | -0.00002 [-0.00020, +0.00017] | +0.0003 [-0.0004, +0.0010] | +2.5 [-17.0, +22.0] | +0.0005 [-0.0027, +0.0037] | +0.0010 [-0.0024, +0.0046] |
| week 1 (2007-2025) | 303 | +0.00004 [-0.00021, +0.00031] | +0.00007 [-0.00044, +0.00063] | -0.0033 [-0.0083, +0.0000] | +0.5 [+0.0, +1.5] | -0.0022 [-0.0106, +0.0063] | +0.0058 [-0.0042, +0.0154] |
| week 2 (2007-2025) | 303 | -0.00015 [-0.00047, +0.00016] | -0.00033 [-0.00100, +0.00035] | +0.0000 [-0.0050, +0.0050] | -1.0 [-9.5, +7.0] | -0.0032 [-0.0150, +0.0084] | +0.0066 [-0.0050, +0.0186] |

The 2007-2025 rows are the 2023-2025 effect diluted about six-fold, since 2007-2022 contributes
exactly zero. Arm levels on the governing window (two-seed means): Brier `0.21378` upstream,
`0.21384` fork, `0.21490` the pick-time line.

Results, by rule (deterministic Brier selects, log loss must not lose, pick accuracy never
selects; a loss in week 1, week 2 or 2024-2025 is flagged without overriding the rule):

- Governing window: Brier `-0.00006` `[-0.00057, +0.00044]`, a tie; log loss `-0.00009`
  `[-0.00124, +0.00104]`, no loss. The tie branch, not between branches: the fork's win-total file
  stays as it is (the simpler setting, no change), and the result goes to the user marked as a
  tie.
- 2024-2025, the seasons the fork's file does not cover: Brier `-0.00041` `[-0.00103, +0.00021]`,
  log loss `-0.00080` `[-0.00219, +0.00059]`. It leans to upstream, as hypothesised, but is not a
  win.
- Weeks 1 and 2 and 2024-2025: no two-seed interval excludes zero, so nothing is flagged.

Every paired interval `REVIEW.md` lists as excluding zero (rule 9):

- Upstream minus fork, two seeds: none, in any window or column. Three touch zero without
  excluding it: week 1 pool points `+0.5` `[+0.0, +1.5]`, 2023-2025 week 1 pool points `+0.5`
  `[+0.0, +1.5]` and week 1 pick accuracy `-0.0033` `[-0.0083, +0.0000]`.
- Upstream minus fork, one seed (outside the instrument): seed 42 total MAE in week 1 `+0.0152`
  `[+0.0020, +0.0294]` and in 2023-2025 week 1 `+0.0960` `[+0.0137, +0.1809]`, both for the fork;
  seed 7 none.
- Each arm against the pick-time line on its own rows (two seeds): for the line, all weeks margin
  MAE, week 1 Brier and week 1 margin MAE (upstream `+0.0489`, `+0.00515`, `+0.2085`; fork
  `+0.0484`, `+0.00511`, `+0.2107`); for the arm, 2023-2025 week 1 pool points (upstream `+10.0`,
  fork `+9.5`) and 2023-2025 week 2 total MAE (upstream `-0.3629`, fork `-0.4043`); and for the
  fork alone, 2023-2025 week 2 pool points `+12.0` `[+1.5, +27.0]`.
- Seed 7 minus seed 42 within an arm, in 2023-2025 and 2024-2025: Brier, log loss and margin MAE
  move by up to `0.0028`, `0.0068` and `0.11`, and the fork's 2024-2025 pool points by `+45.0`
  `[+12.0, +75.0]`. The within-arm seed movement on the governing window's 816 games (Brier
  `0.0015`-`0.0016`) is about 25 times the arm difference.

Ranks, two-seed means with the line. Governing window: upstream first on Brier, log loss and pool
points, the fork first on margin MAE, the line first on pick accuracy and total MAE. 2024-2025:
upstream first on Brier, log loss, pool points and margin MAE, all three tied on pick accuracy, the
line first on total MAE. Every upstream-minus-fork difference is far inside the three-season
fit-noise floors above. The rank table for every window and column is in `REVIEW.md`.

The reviewer's findings, both minor (P3, informational): `build.sh` reran the candidate's check
and overwrote `inputs/qb_elos_upstream_wt_check.json`, the file Amendment 1 cites, 13 s after the
hypothesis was hashed (the check is deterministic and the copy on disk shows the amendment's
figures; suggested: write the check to a new file or check its hash); and `nfl-predictor compare`
still has no seasons filter. Whether to adopt upstream's file anyway, as maintenance with no
measured cost, is the user's call (`.agents/TODO.md`, step 7): it would change a production input
between game weeks and make `upstream_wt_seed{42,7}` the floor-sigma reference. Full tables,
provenance, findings and reproduction scripts: `models/qb_wt_measure/REVIEW.md`.
