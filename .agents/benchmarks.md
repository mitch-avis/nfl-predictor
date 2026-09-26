# Benchmarks, reference arms and data state

Benchmark tables, reference arms, fit-noise floors and dataset records for walk-forward work.
Every number here follows guardrail rule 3 in `AGENTS.md` (two keys, a run directory beside each
number). The roadmap order is in `AGENTS.md` and `.agents/TODO.md`.

## Workstream history

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
  runs), reproduced with `.venv/bin/python models/wf_m55_8_review/independent_rescore.py`. Every
  arm spent the full `200` trees in all 107 folds, and `auto` resolved to the deterministic floor.
  Weeks 3-18, `1423` games, market Brier `0.20948`:

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
