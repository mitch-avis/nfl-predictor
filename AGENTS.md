# Agent Instructions for nfl-predictor

## 0) Mission + Non-Negotiables

This repository predicts NFL outcomes and scores for **Pick 'Em** and **Confidence Pools**.

- **Primary objective:** calibrated win probabilities and weekly confidence rankings.
- **Standing yardstick:** the closing market line. Every walk-forward report compares the model's
  win probability and margin against the market-implied probability and the spread on the same
  games; the model is measured by that difference, and "not worse than the market with better
  early-season calibration" is the current bar. Beating the closing line is a stretch goal, never a
  claim.
- **Secondary objective:** realistic score outputs for display and reporting.

Rules that are always enforced:

- **No data leakage.** Features, splits, calibration, blending, and evaluation only use information
  available before the predicted games ("known at prediction time" for the specific week).
- **Time-aware evaluation.** Hyperparameter tuning and model selection use blocked, time-ordered
  splits (season/week aware).
- **Polars-first ETL.** Dataset creation and feature engineering run in Polars; pandas/numpy are
  acceptable inside ML modules only as needed.
- **NFLverse via nflreadpy is the core data source.**
- **Reproducible artifacts.** Training and backtests write run directories with metadata and
  metrics.
- **Tests are required.** New functionality includes unit tests and improves or maintains coverage.
- **TDD is the default for executable code changes.** Before changing production code, verify that
  the exact behavior and lines you plan to touch are covered; if not, add focused characterization
  or failing tests first, then edit the production code.

## Current Focus (2026 season start)

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
  fast path `scripts/weekly_run.py` uses when it calls `data_collection.main()` with no
  arguments; `nflverse`/`scrape` remain selectable explicitly. A full ETL rebuild followed
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
  floor" below. The order agreed with the user on 2026-09-24, each step on its own branch and
  merged before the next, with no deadline:
  (1) close 55.8;
  (2) Milestone 60, CLI and entrypoint consolidation widened to every file under `scripts/`,
  behavior-preserving, with task 55.5;
  (3) production/benchmark parity: GPU as the default device (55.4), how production
  probabilities are formed (56.5), pick-time lines (56.6), and the out-of-fold calibration pool;
  (4) rebuild reproducibility, then every feature-value change (55.3, 53.7 and the feature
  follow-ups);
  (5) the Optuna re-tune with wiring into production (55.9 with 56.3);
  (6) the web UI, Milestone 58 phases 4-6.
  Tasks 55.1 and 55.2 are retired. The reasoning and the follow-up assignments are under "Roadmap
  Status" in `.agents/TODO.md`.
- XGBoost margin/total stays the primary model and benchmark. Do not build alternative model
  families or run large tuning campaigns unless the user asks.
- Borrow proven methodology from `../nfl-sos-ratings` before inventing new metrics; treat that
  repo as read-only reference material. The method being ported is its head-to-head-excluded
  opponent profiling and the simultaneous ridge that generalizes it (see
  `.agents/feature_crosswalk.md` section 3.1).
- Validated baseline on 2026-09-24 (`main` at `703ea25`, version `0.18.0`, re-run on the fresh
  `feat/m60-cli-consolidation` branch): `scripts/gate.sh` exits `0` (`889 passed`, coverage
  `92.45%` against the enforced `90%` floor; ruff format, ruff, ty, pyright, markdownlint,
  `uv lock --check`, `uv sync --check --active` and the CLI help smoke checks all clean).
  `feat/m55-8-season-weighting` (the task 55.8 close-out, the 2026-09-24 roadmap and the
  guardrail rules 9-14) is merged into `main` and pushed. The
  frontend gate was last verified at the `0.8.0` merge; run `scripts/gate.sh --web` whenever
  `web/` or `nfl_predictor/api/` changes.
  - `.agents/skills/` is a separate git clone of agent skills: gitignored, excluded from ruff
    (`pyproject.toml`) and markdownlint (`.markdownlintignore`; pass `"#.agents/skills"` to
    `markdownlint-cli2`). Never edit it as part of this repo's work.
  - Re-run a plain `uv sync` after every version bump (including after merging a branch that
    bumped the version), or `uv sync --check --active` fails on the stale installed package.
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

### Readiness behaviors that must not regress

- Pre-kickoff current-week detection resolves to the new season's Week 1, not the prior season's
  final playoff week.
- Current-season nflreadpy team-stat 404s are non-fatal so ETL still runs when the schedule exists
  before weekly stats are published. Any new nflreadpy source (PBP included) must follow the same
  cache-then-degrade pattern.
- Default weekly prediction-file resolution uses the CSV `season`/`week` values, not the filename
  week, so a stale `week_22` file never wins over a new `week_01` file.

## Delegation guardrails (every agent session)

Added 2026-09-19 after the audit of the two sessions that closed Milestone 59
(`.agents/ARCHIVE.md`, Milestone 59, "Audit"). Those sessions narrowed the task text and ticked
the box anyway, reported a green gate without running the whole gate, wrote a benchmark
provenance sentence that was false, and spent about twelve CPU-hours on calibrators that were
then discarded. An autonomous session has no other supervision, so these rules are not advisory.
Rules 4 and 5 were extended on 2026-09-21, after the tree-budget ladder's `400` rung was launched
as a third run on one task and its result was read on the all-weeks interval while the written
rule named weeks 3-18; the user accepted both and asked for the two amendments below.
Rules 13 and 14 were added on 2026-09-24 with the roadmap above, after the same review's
second-seed pair showed that re-seeding alone can move six-season pick accuracy and pool points by
amounts whose intervals exclude zero.
Rules 3 and 7 were tightened and rules 9-12 added on 2026-09-23, after the second-key review of
the session that closed task 55.8 as `0.17.0`. Both of its reviewer subagents failed, and the
producing agent wrote the four `REVIEW.md` files and every number in the docs itself. It closed
the task by keeping the shipped setting, which was the weakest arm on every probability and error
metric, and never reported the two intervals that excluded zero (a later count; the review
first recorded one). It also committed a CLI inventory
whose largest section had been filled in by guesswork after its code searches failed. The user
accepted these amendments.

1. **One gate.** `scripts/gate.sh` is the definition of "checks pass". No task, chunk or version
   is reported done, and no changelog entry is written as landed, until it exits `0` on the
   final tree. Reporting a subset of checks as the gate is a defect; `--quick` is for iteration,
   never for the report.
2. **Narrowing is never a checkbox.** If what landed differs from the task text (smaller scope, a
   substitute method, a skipped acceptance criterion), the task stays `[ ]` with a `Narrowed:`
   note giving the difference, the reason and where the remainder now lives, and the difference
   goes on the user's question list. Rewriting the acceptance text to fit the delivery is not
   allowed.
3. **Two keys on every number.** The agent that produced a run never writes its numbers into
   `AGENTS.md`, `.agents/ARCHIVE.md`, `.agents/TODO.md` or `CHANGELOG.md`. A reviewer (a separate
   subagent, or the next session) rescores the artifact from disk, writes the run directory and
   the reproduction command beside the number, and only then may the docs change. A number in
   the docs without a run directory is a defect. The review is independent only if (a) its
   `REVIEW.md` names who reviewed and states that the reviewer did not produce the run; (b) it
   recomputes the metrics from the fold checkpoints (`models/wf_checkpoints/<fingerprint>/`), not
   from `metrics_report.json` or only through the producer's comparison script; and (c) it checks
   provenance: dataset hash, git commit, fold count, `best_iteration`/`early_stopped`, and that
   only the intended setting differs between arms. If a reviewer subagent fails or returns
   without numbers, the producing agent reports that and stops; it never writes the review
   itself.
4. **Compute budget.** Before each walk-forward run, write the hypothesis and the decision rule
   (which result changes what). The decision rule names the exact window or windows it reads
   (week 1, week 2, weeks 3-18, all weeks), the exact columns (deterministic Brier, log loss,
   pick accuracy, margin MAE, the paired deterministic-minus-market interval), and how the
   windows combine when they disagree, so a rung cannot read as a tie on one window and a win
   on another without the rule saying which one governs. After two runs on one task without a
   decision, stop and ask. A **ladder** (several runs of one hypothesis family, with a stated
   cap on the number of rungs and a stopping rule) written into the check-in and accepted by
   the user counts as approval for every rung up to that cap; only a rung beyond the cap, or
   outside the ladder as written, needs a fresh ask.
   One walk-forward at a time; `uptime` and `pgrep -af walk_forward` first; OpenMP policy by
   load. Any edit under `nfl_predictor/ml/` changes every checkpoint fingerprint, so a rerun
   after a code change retrains from scratch; plan runs after the code is stable.
5. **Must ask first** (stop and wait; never assume):
   - rebuilding anything under `data/` (an ETL rerun), or deleting or overwriting any file under
     `data/` or `models/`;
   - changing a default (CLI, config, constants) that alters what a weekly run produces, or any
     change that alters feature values at ETL time;
   - closing a milestone, reopening a parked one (Milestone 57), or reordering the roadmap;
   - merging to `main` or pushing (tags and releases: never, see above);
   - touching `../nfeloqb`, `../nfl-sos-ratings`, or the web API on port 8765;
   - a third walk-forward run on one task, or any six-season run, unless it is a rung of a
     ladder the user has already accepted under rule 4 and is within that ladder's cap;
   - anything the task text says to decide with the user.
6. **May proceed without asking:**
   - commits on the working feature branch after each versioned chunk (Conventional Commits, one
     logical change per commit, the attribution line the harness provides);
   - fixes with a failing test first, and doc updates that restate numbers already verified
     under rule 3;
   - one walk-forward run per written hypothesis, within rule 4;
   - creating the milestone's feature branch off `main` when none exists.
7. **Check-ins.** Report at every landed version and after every walk-forward run: what landed,
   the run directory, the gate result, the open questions. Stop for a question whenever rule 5
   triggers; a session that ends blocked on a question has done the right thing. The check-in
   after a walk-forward run carries its governing-window numbers (not only "it finished"), and it
   is written before the next rung is launched. A session that launches long runs says in the
   same check-in how they will be sequenced and supervised: either one driver script that runs
   the accepted rungs back to back, or a watcher that wakes the session when the run exits.
   Never end a turn while promising to "keep monitoring" with nothing actually watching.
8. **Handoff hygiene.** Rewrite `.agents/next_agent_session_prompt.md` at every landed chunk
   (branch, version, uncommitted state, the next task, open questions) so a restart after a
   closed terminal or a context summary resumes without re-deriving anything.
9. **Read every interval; the incumbent gets no benefit of the doubt.** A decision or close-out
   lists every paired interval that excludes zero, in every window reported and in either
   direction, plus where each arm ranks on every named column and tie-breaker (deterministic
   Brier, log loss, margin MAE, total MAE, pick accuracy, confidence-pool points). "No arm beat
   X" is incomplete without how X compares to the others. A shipped setting that was never
   measured is an assumption, not a baseline: when a measurement ties, the simpler setting (fewer
   knobs, fewer assumptions) is the recommendation, and the tie goes to the user as a question
   rather than closing in the incumbent's favor.
10. **Inventories and audits are generated, not written.** Any inventory of code (flags, files,
    columns, consumers, call sites) is produced by a script checked in beside it, and its
    headline counts are reproduced by that script. A search or tool call that fails, errors, or
    returns "No matches found" where matches must exist is reported as a failure. The gap is
    never filled by inference, and a subagent's summary is input to verify, not a source to
    copy.
11. **The benchmark measures what production does.** Any setting that differs between the
    reference walk-forward configuration (the benchmark arms under "Current Focus") and the
    production weekly run (`config/weekly_run.yaml` and the weekly stage-1 selection) is a
    defect until the user approves it and it is recorded here. The season-weighting gap closed in
    `0.18.0` (both train unweighted, task 55.8). Open gaps, all scheduled:
    - the probability path: production submits the stage-1 winner, currently `elo` with a market
      blend and clamp, while the benchmark scores the deterministic map (task 56.5);
    - the device: the weekly run trains on the GPU, standalone walk-forwards on the CPU (task
      55.4; the user chose the GPU for everything, 2026-09-23);
    - line timing: backtests anchor to and score against the stored, probably closing, lines,
      while production anchors to mid-week lines and picks are made before Thursday (task 56.6).
12. **A decision rule favors no outcome after the fact.** The rule written under rule 4 is
    applied as written. If the result falls between its branches, or its first condition fails
    and a fallback branch rescues the preferred outcome, the result goes to the user as a
    question with the numbers.
13. **Two seeds before any default changes.** A walk-forward result that would change a default
    (a setting, a feature family kept or dropped, a tuned parameter) holds on two seeds. Combine
    them per game: the candidate-minus-reference loss difference averaged over the seeds (same
    seed paired with same seed), bootstrapped over games. Game-resampled intervals hold the fit
    fixed, so they leave out seed-to-seed variance: a single-seed six-season interval that
    excludes zero is not enough on its own, above all for pick accuracy and pool points. A new
    reference arm (a new build, device or code path) is measured on two seeds too, which also
    records that reference's own noise floor.
14. **Order of work.** Restructure before changing behavior (a behavior-preserving move is
    pinned by a characterization test and lands before any output-changing task that touches the
    same code). Close benchmark/production parity gaps before new measurements (rule 11). Change
    feature values before tuning. Group changes that invalidate walk-forward checkpoints so they
    share one new reference. Production-changing steps land between game weeks. The step order
    in `.agents/TODO.md` ("Roadmap Status") follows these principles; changing it is must-ask.

## Source of truth for work

- Active milestones and tasks live in `.agents/TODO.md` (authoritative active worklist).
- Completed milestones live in `.agents/ARCHIVE.md`; archived numbers never change. Active
  milestones in `.agents/TODO.md` are numbered in execution order (renumbered once on 2026-09-10;
  the old-to-new map is at the top of `ARCHIVE.md`), and new milestones take the next number after
  the highest one in either file. When priorities change, move a section instead of renumbering.
- The cross-repo feature review and prioritized shortlist live in `.agents/feature_crosswalk.md`.
- The handoff prompt for the next session lives in `.agents/next_agent_session_prompt.md`.
- `CHANGELOG.md` is the authoritative release history.
- Before starting any task: read `.agents/TODO.md` and work only on the highest-priority blocking
  items.
- When a task is completed: move it from `.agents/TODO.md` to `.agents/ARCHIVE.md` with a short
  completion note.

## Engineering Standards (Logic, Docs, Lint, Coverage)

- Docstrings are required for every module, class, and function (including tests).
- Type hints are required for new/modified code.
- Fix linter findings introduced by your changes. Do not leave new warnings behind.
- Avoid adding new `noqa`, `type: ignore`, or `pragma: no cover` suppressions unless they are truly
  necessary, narrowly scoped, and justified by the code rather than convenience.
- Do not reference temporary planning artifacts in code: do not mention roadmap items, milestone
  numbers, or TODO goal labels in any code, comments, docstrings, or test descriptions.
- Prefer small, deterministic unit tests.
- Load and apply relevant skills before acting. Default to `python` for Python work; add
  `test-driven-development`, `clean-code`, `systematic-debugging`, `code-review`, `observability`,
  `task-orchestrator`, and the `python-*` skills when their domains apply.
- If a Python file grows beyond ~2000 lines, propose a refactor plan to split it into smaller,
  focused modules (helpers/utils) and implement the split if it reduces complexity.
- Keep `.agents/TODO.md` accurate: verify items before checking them off.
- Keep `.agents/TODO.md`, `.agents/ARCHIVE.md`, `.agents/feature_crosswalk.md`, and
  `CHANGELOG.md` synchronized with the actual repo state after meaningful progress, completed
  tasks, or validation changes.
- Keep `README.md` current: update it when behavior, CLI usage, features, or outputs change.
- Keep project documentation current after significant changes. When a subsystem outgrows the
  top-level `README.md`, add or update nested module README files and link them from the top-level
  README.

## Changelog and Commit Workflow

- `CHANGELOG.md` follows Common Changelog: latest version first, `## [VERSION] - YYYY-MM-DD`,
  then `Changed`, `Added`, `Removed`, and `Fixed` in that order.
- **Update the changelog as you go**, not at the end of a session: add an entry each time a
  milestone, a task, or a sizable chunk of one lands (a fix, a feature group, a changed default).
- **Every entry gets its own incremented version. Never write `[Unreleased]`.** Bump the patch
  (`0.6.1` to `0.6.2`) for fixes and small additions, and the minor (`0.6.x` to `0.7.0`) for a new
  feature family, a changed default, or a schema change. Date the entry with the day the change
  lands.
- **Set `pyproject.toml` to the newest changelog version in the same change**, then run `uv lock`
  and `uv sync`, or `uv lock --check` and `uv sync --check --active` fail on the stale version.
- The project is private and not ready for releases: **never create or push a git tag or a GitHub
  release.** Versions live only in `CHANGELOG.md`, `pyproject.toml`, and `uv.lock`;
  `.github/workflows/release.yml` stays idle because no tag is ever pushed.
- The historical baseline is `0.1.0` on `main`.
- Keep changelog entries focused on notable user-facing, tooling, or workflow changes; skip routine
  formatting-only noise.
- When committing work, prefer one logical change per commit for multi-file schema work and one file
  per commit for all other changes, including deletions, unless the user explicitly asks for
  different commit granularity.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): a subject
  of the form `type(scope): imperative summary`, at most 72 characters, followed by a blank line
  and a body that explains what changed and why. Types: `feat`, `fix`, `docs`, `test`, `refactor`,
  `perf`, `build`, `ci`, `chore`. The scope is optional and names the area (`etl`, `ml`,
  `rankings`, `reporting`, `agents`, `walk-forward`). Mark breaking changes with `!` after the
  type or scope. Examples: `feat(etl): blend early-season stats toward the regressed prior`,
  `docs(agents): regenerate the handoff prompt`, `fix(reporting): balance the workbook formulas`.

## CI Direction

- GitHub Actions is the first CI target and currently stays validation-only.
- `.github/workflows/validation.yml` provisions `.venv` with `uv` and runs Ruff format/check,
  Pyright, Ty, pytest, markdownlint, `uv lock --check`, `uv sync --check --active`, and the existing
  editable-install plus primary CLI help smoke checks.
- `.github/workflows/release.yml` publishes or updates GitHub releases for `0.x.y` and `v0.x.y` tags
  by extracting the matching `CHANGELOG.md` entry.

## Command execution rules (non-negotiable)

This project uses a **local virtual environment located at `.venv/`**.

When running any commands, you MUST invoke tools from the virtual environment explicitly. Do NOT
rely on shell activation, PATH inference, or system-installed binaries.

### Required command forms

Use these forms **at all times**:

- Python:
  - `.venv/bin/python`
- uv:
  - `uv`
- pytest:
  - `.venv/bin/python -m pytest` **or** `.venv/bin/pytest` **or** `uv run pytest`
- ruff:
  - `.venv/bin/ruff`
- pyright:
  - `.venv/bin/pyright`
- ty:
  - `.venv/bin/ty`

### Explicitly forbidden

- `python`, `pip`, `pytest`, `ruff`, `pyright`, or `ty` **without a `.venv/` prefix**
- assuming an activated shell or implicit virtualenv
- using system Python, Conda, pyenv, or global tools

For Python-based tooling, assume the `.venv/bin/` prefix is required even if it is not written in a
doc example. `uv` is expected to come from `PATH` as an external project manager.

### Formatting, linting, and style

- **Ruff** formatting (line length 100).
- **Ruff** linting (including import sorting, docstrings, security, simplify, NumPy, and
  pygrep-hooks rules).
- **Pyright** and **Ty** are both mandatory local gates today.
- **Pyright** remains the more mature signal in pandas-heavy code, so keep both green rather than
  replacing one with the other.
- PEP 8 / PEP 257 conventions unless explicitly overridden by repo tooling.

Recommended local commands (`scripts/gate.sh` runs all of them the way CI does and is the only
form that counts as "the gate"; the individual commands are for iteration):

- `scripts/gate.sh` (add `--web` when `web/` changed, `--quick` to skip pytest while iterating)
- `.venv/bin/ruff format .`
- `.venv/bin/ruff check .` (and optionally `.venv/bin/ruff check . --fix`)
- `.venv/bin/pyright .`
- `.venv/bin/ty check .`
- `.venv/bin/python -m pytest`
- `uv lock --check`
- `uv sync --check --active`

## Project Shape (Big Picture)

- **Primary Pipeline:** Polars for data processing + `nflreadpy` for NFLverse sources (schedule,
  team stats, etc.). The pipeline integrates schedule/results, team statistics, Elo/QB ratings,
  TeamRankings stats, and market odds to produce ML-ready datasets.
- **Orchestration Script (ETL):** `nfl_predictor/data_collection.py` (run as a module). This
  orchestrator fetches data, applies transformations, and writes output CSVs.
- **Core Data Transforms:** Polars ETL helpers live under `nfl_predictor/utils/polars/`.
  `nfl_predictor/utils/polars_utils.py` is a compatibility facade that forwards imports to the split
  modules.
- **Game-Specific Enrichments:** `nfl_predictor/utils/game_utils.py` contains domain-specific
  calculations and dataset enrichments.
- **External Data Scraping/Caching:** `nfl_predictor/utils/scraping_utils.py` fetches and caches
  external web data used by the pipeline.

### ML implementation layout

- `nfl_predictor/ml/` contains the split ML implementation modules.
- `nfl_predictor/ml_model.py` is a compatibility facade for legacy imports and a primary CLI
  entrypoint.
- XGBoost version/build compatibility helpers live in `nfl_predictor/ml/ml_model_xgb_utils.py`.

### Repo scripts (operational entrypoints)

- `scripts/weekly_run.py`: canonical weekly orchestration (data refresh -> compare -> tune/train ->
  predict -> reports).
- `scripts/walk_forward_backtest.py`: walk-forward evaluation utility.
- `scripts/wf_compare.py`: sweep calibration + market-prob variants and summarize metrics.
- `scripts/power_rankings.py`: power rankings + projected standings. Since 2026-09-11 the default
  (`--method composite`) ranks on the ETL's schedule-adjusted composite, read from
  `data/strength_snapshots.csv` for the week after `--through-week`. `--method bradley_terry` keeps
  the 2026-09-09 current-season fit (two-season window, prior seasons weighted `0.25`, margin
  targets, future model probabilities excluded), and `--legacy-franchise-fit` restores the old
  all-seasons equal-weight fit and implies it. `scripts/weekly_run.py` exposes the same options
  and calls the same `compute_power_rankings` (`nfl_predictor/reporting/power_rankings.py`).
- Retired in `0.19.0`: `golden_command.py`, `betting_pipeline.py`, `backtest_predictions.py`,
  `objective_compare_models.py` and the Excel betting workbook (`betting_report_excel.py`).
  Old launchers that name them reproduce from their run's recorded git commit.

### Web UI (FastAPI + React)

- `nfl_predictor/api/` is the FastAPI backend (`python -m nfl_predictor.api`): auth (argon2,
  JWT cookie, `viewer`/`admin`), a run index over `models/*/metadata.json` with one **active**
  run, readers and a column registry for predictions, betting, power rankings, model and data
  status, and a job runner (`nfl_predictor/api/jobs/`) that launches the repo CLIs as
  subprocesses with streamed logs; walk-forward jobs share one worker so two never overlap.
  `nfl_predictor/lines_refresh.py` and `nfl_predictor/week_builder.py` are the CLIs it added.
- `web/` is the Vite + React 19 + Tailwind app; it is excluded from ruff, pyright and ty.
  Its gate (`source ~/.nvm/nvm.sh`, then in `web/`: `npm run lint`, `npm run typecheck`,
  `npx vitest run`, `npm run build`) runs as the `web` job in CI; run it whenever `web/` changes.
- Tests for the backend live in `tests/api/`; the design, decisions and phase status live in
  `.agents/web_ui_plan.md`, and `web/README.md` documents the runtime configuration.

## Modeling Philosophy (Important Context)

- Implementation is purely in Python.
- Team strength is represented by learned relationships between engineered features and outcomes.
- Feature interactions and weights are learned by the model; feature engineering provides signal,
  not fixed scoring formulas.

## Data Inputs/Outputs (Repo Conventions)

- **Data directory:** all datasets live under `data/` (see `constants.DATA_PATH` in
  `nfl_predictor/constants.py`).
- **Key output files (examples; do not hard-code filenames):**
  - `data/all_data_ml.csv` - master ML dataset (includes engineered features and targets where
    available)
  - `data/all_data.csv` - combined dataset without ML-only targets
  - `data/completed_games_ml.csv` and `data/completed_games.csv` - completed games subsets
  - `data/predict/week_XX_games_to_predict.csv` - upcoming week games with engineered features
  - `data/strength_snapshots.csv` - pre-week adjusted strength per `(season, week, team)` for
    every scheduled team, bye teams included; the source of the default power rankings

### I/O rules

- Prefer the project’s Polars-based load/save helpers in `nfl_predictor/data_collection.py`.
- Any pandas-based CSV I/O utilities are legacy. Do not add new pandas-based I/O helpers; prefer the
  Polars ETL helpers when touching related code.

## Neighboring Repos and Cross-Repo Contracts

- `../nfeloqb` produces `qb_elos.csv` (538-style schema; `team1` is the home team).
  `nfl-predictor` consumes a manually copied `data/qb_elos.csv`. Treat that file as a downstream
  contract: do not propose schema changes there casually, and never edit `../nfeloqb` outputs from
  this repo. `../nfeloqb/Other Data/meta_data.csv` maps Elo QB names to GSIS ids and is the
  intended identity bridge for QB-level PBP features.
- `../nfl-sos-ratings` is the reference implementation for the strength-of-schedule method this
  repo is porting: for each subject (team or QB) and each opponent it faced, build that opponent's
  statistical profile from only its games against the rest of the league, excluding every
  head-to-head game with the subject, so subject and opponent profiles are independent for every
  matchup; then compare the subject to that adjusted schedule. Its simultaneous ridge solve
  (`simultaneous_adjustment.py`) is the all-hops generalization of that one-hop method and is its
  published backbone; the one-hop profiles remain for descriptive views. It is also the reference
  for PBP-derived per-snap EPA, success and explosive rates, and special-teams EPA. Read its
  `README.md`, `AGENTS.md`, and `docs/` (`methodology.md`, `validation-report.md`, both stats
  catalogs) before designing any adjusted feature, and note that its own walk-forward puts the
  within-season ridge at parity with SRS and raw EPA and behind prior-carrying Elo. Port ideas and
  formulas into this repo's Polars ETL; do not import it as a dependency and do not modify it from
  here. The repo-root symlink `nfl-sos-ratings -> ../nfl-sos-ratings/` is gitignored and exists
  only for convenient reading; always state which repo you are inspecting.
- Leave untracked local files in neighboring repos alone (for example `../nfeloqb/.bash_history`).
- Play-by-play comes from `nflreadpy.load_pbp`. nflreadpy caches only in memory, so cache selected
  columns per season as Parquet under `data/cache/nflreadpy/` with the same current-season refresh
  and non-fatal failure behavior as schedules and team stats. Filter to the regular season for
  feature inputs and normalize `posteam`, `defteam`, `home_team`, and `away_team`.

## Column & Schema Rules (Source of Truth)

- Column names and schema lists are defined in `nfl_predictor/constants.py`.
- Team identifiers are normalized using the canonical mapping in `constants.py`.
- Do not hard-code column lists; use constants to prevent schema drift.
- ETL does not silently drop columns.
- Always normalize team identifiers via `constants.ALIAS_TO_CANONICAL` and (when available)
  `normalize_team_column(df, col)`.

## Season/Week Logic & Edge Cases

- Use `constants.get_regular_season_weeks(season)` to determine regular-season length.
- Week 1 / early-season rows with missing history use the previous regular season regressed by
  `constants.WEEK1_REGRESSION_FACTOR`. From week 2 on, season-to-date stats blend toward that same
  prior with in-season weight `games / (games + constants.PRIOR_BLEND_GAMES)`, and rates are
  recomputed from the blended sums (`polars_utils.blend_with_prior_stats`). The adjusted-strength
  family blends its own previous-season snapshot the same way. New season-to-date families inherit
  the stat blend automatically if they flow through `team_stats_df`.
- Future games have missing outcomes; the pipeline still outputs a structurally complete row
  suitable for prediction.
- Postseason rows may exist. Training/evaluation defaults should be explicit about whether
  postseason is included and (if included) how it is weighted.

## Prediction & Modeling Logic

### Canonical targets

The primary model predicts:

- `margin = home_score - away_score`
- `total  = home_score + away_score`

Derived scores are computed as:

- `home_score = (total + margin) / 2`
- `away_score = (total - margin) / 2`

Direct home/away score regressors are allowed only as secondary ensemble members.

### Win probability

- Win probability is derived from the margin prediction.
- Win probabilities are calibrated using time-aware calibration data.
- Calibration metrics (Brier, log loss, reliability table) are reported in evaluation.

Calibration methods (canonical names):

- `none`: deterministic margin->prob mapping (baseline)
- `platt`: logistic regression (Platt scaling)
- `isotonic`: isotonic regression
- `elo`: deterministic Elo-style logistic mapping

Notes:

- Prefer time-aware calibration (`platt` or `isotonic`) when enough calibration rows exist.
- If adding new CLI options, keep names stable and document them.

### Market integration

When market lines exist, the system produces market-derived features and supports market anchoring:

- Market transforms produce `market_home_margin`, `market_total_line`, `home_market_prob`,
  `away_market_prob`.
- Market anchoring trains residuals vs market baselines and adds the baseline back at prediction
  time.
- Market probability blending/clamping uses explicit CLI/config values and is validated in
  time-aware evaluation.

Market anchoring details:

- Prefer residual training: `target_resid = target - market_baseline` and `pred = market_baseline +
pred_resid`.

Market probability post-processing (blend/clamp):

- Blending must be explicit and bounded (weights in [0, 1]).
- Clamping must be explicit and bounded (delta in [0, 0.5]).
- If adding "no-vig" market probability options, implement them consistently (home/away normalize to
  sum to 1) and validate in walk-forward.

### Uncertainty

- Predictions include uncertainty intervals for margin and total (p10/p50/p90 or equivalent).
- Interval outputs are evaluated (coverage/width diagnostics) and are part of the run artifacts.

Minimum requirement:

- Output a median plus at least one interval for both margin and total (quantiles preferred).

### Realistic score outputs

- Realistic score outputs are produced as post-processing applied after margin/total predictions are
  generated.
- Realistic score adjustments are used for display and reporting.
- Realistic score adjustments do not alter win probabilities, confidence rankings, pool scoring, or
  tuning objectives.

If implementing score "realism":

- Apply post-processing only after core predictions; rounding/snapping policies must be
  configurable.
- Never change training targets to enforce an "NFL score lattice" unless explicitly designed and
  documented.

### Confidence pool deliverable

- Weekly outputs include a **1..N** unique confidence ranking across that week's games.
- Predicted winner is derived from calibrated win probability.
- Confidence strength is derived from calibrated win probability (default: `abs(p - 0.5)`).

Authoritative pool scoring rules:

- Each week assign unique confidence values `1..N` to the chosen winner in each matchup.
- Max weekly points = `N*(N+1)/2`.
- Realized points = `sum(conf_i * 1[pick_i_correct])`.
- Tie games: treat as incorrect for both sides.
- Picks are submitted before the first game of the week (single-shot; no in-week updates in
  backtests).

## Evaluation

Required evaluation modes:

- **Season-blocked CV** (acceptable baseline; primarily used for hyperparameter tuning).
- **Walk-forward evaluation (authoritative):** for each season and each week `w` (e.g., `3..end`),
  train on all games strictly before week `w` (plus prior seasons if configured), predict week `w`,
  and record metrics.

Required metrics:

- margin MAE
- total MAE
- win probability Brier score
- win probability log loss
- binned reliability summary
- confidence pool point summaries (expected + actual)

Market-relative metrics (when market anchoring is enabled):

- residual MAE vs market baseline for margin/total
- edge vs spread/total as diagnostics only (do not claim profitability)

### Model selection protocol (how to choose "best" settings)

When multiple options exist (calibration method, market integration mode, probability blend/clamp
rules, weighting choices):

- Prefer selecting settings via walk-forward over multiple seasons.
- Pick a primary selection metric (typically Brier/log loss for probability quality) and use
  secondary tie-breakers (confidence pool expected points, then margin/total MAE).
- Report mean and variance across folds; avoid choosing a setting that wins by a hair on one season
  but regresses elsewhere.
- Never use the holdout window to tune hyperparameters.

Required run artifacts:

- saved model artifact
- metadata JSON (see "Model artifact contract")
- metrics report JSON (walk-forward aggregated + per-season/per-week summaries)
- plots are optional and must not block CI

## ML Implementation Standards

Preprocessing:

- Use `ColumnTransformer` for categorical one-hot + numeric passthrough/impute.
- Avoid densifying large sparse matrices unintentionally.
- Tree-based models (XGBoost): do not use `StandardScaler` unless a non-tree model requires it.
- Missing values: XGBoost can handle them; impute only if required for consistency.

Training:

- Use early stopping and set `eval_metric` explicitly (aligned to objective).
- Tune hyperparameters consistently with the evaluation metric (Optuna supported).
- Use `random_state` everywhere applicable.
- Do not hard-code `n_jobs`; prefer `os.cpu_count()` or a config default.

Blending:

- Prefer explicit, interpretable blends (market anchoring often sufficient).
- If using a blender/regressor, avoid unstable unconstrained weights; prefer non-negative and/or
  sum-to-1 when appropriate.
- Validate blends using time-aware splits.

## Leakage Audit (Required)

Maintain a leakage audit tool/mode (see `scripts/leakage_audit.py`):

- checks for target/label columns in features
- flags suspiciously predictive columns (e.g., absurd correlations)
- validates season-to-date features exclude the current game row

## Reproducibility & Model Artifact Contract

Every saved model must include adjacent metadata JSON with:

- created timestamp
- git commit hash (if available)
- dataset fingerprint (hash of training CSV and/or stable row ids)
- library versions (xgboost, sklearn, numpy, pandas, polars, scipy)
- training config (CLI args / config object)
- season/week ranges used for train/calibration/holdout
- feature list used
- best params (if tuned) and early-stopping info

Artifacts must be loadable without hidden external state.

## Dependency & Environment Hygiene

- Pin key ML dependencies for reproducibility: xgboost, scikit-learn, numpy, pandas, polars, scipy,
  optuna (if used).
- Document supported Python version(s) and CPU/GPU constraints if applicable.
- Avoid optional GPU paths that break CPU-only execution unless explicitly guarded.

### Dependency management (uv + pinned requirements)

- Declare runtime and development dependencies in `pyproject.toml`.
- Treat `uv.lock` as the lockfile source of truth for reproducible environments.
- Preferred install/update flow is `uv lock` / `uv sync`.
- `update_requirements.sh` is the convenience wrapper for refreshing the lockfile and syncing the
  active project environment.

## Feature Development Rules

All engineered features apply to **every matchup**, not only end-of-season games.

Feature areas tracked in `.agents/TODO.md` include (examples):

- season-to-date record features (overall, division, conference W-L-T)
- divisional rivalry indicator
- lookahead/trap indicators (next-week opponent strength + rest/travel context)
- motivational asymmetry features (playoff leverage and clinch/elimination context)
- PBP-derived per-snap EPA, success, explosive, and special-teams families
- weekly schedule-adjusted (ridge) offense/defense strength and EPA-based schedule strength, in
  both the ridge form and the one-hop head-to-head-excluded form
- QB per-dropback EPA families for the expected starter

Rules for stat-style features:

- Carry counts and sums through season-to-date aggregation and compute rates afterward (ratio of
  sums), the way `_compute_derived_metrics` already works.
- Name allowed/defensive metrics explicitly and add them to `EXCLUDE_FROM_OPPONENT_STATS` so the
  generic `opponent_` mirror does not duplicate them. Before excluding a stat, grep
  `_compute_derived_metrics` for its `opponent_` mirror: a derived metric that reads it needs
  the stat listed in `OPPONENT_MIRROR_INTERMEDIATES` too, or the derived columns go null at the
  next rebuild (the `0.12.4` sack exclusion did exactly that, fixed in `0.12.6`).
- Cite the formula in the docstring and test each self-computed metric against a hand-built
  fixture.
- Any schedule-adjusted or opponent-adjusted value for week `N` must be solved from games strictly
  before week `N` in that season, with a documented prior-season fallback for early weeks.
- A change that alters feature *values* at ETL time (a prior blend, a regression factor, a new
  fallback) cannot be ablated with `--disable-feature-groups`, which only drops columns. Ablate it
  with two dataset builds behind an ETL flag, back up `data/*.csv` before each rebuild, and keep
  both walk-forward reports under `models/`. Measure early-season changes with `--wf-start-week 1`
  and report weeks 1, 2, and 3-18 separately.

## Missing Data Rules

Some sources do not exist for all seasons.

- ETL emits an invariant schema for every run.
- Missing sources become nulls or defined defaults.
- The model path handles nulls without crashing and reports how many rows use fallbacks for each
  feature group.

## constants.py Hygiene

`nfl_predictor/constants.py` remains the canonical reference for:

- file paths
- feature column names/lists
- team mapping tables
- season/week rules
- default parameters used in data collection

Unused constants are removed and the file remains organized into clear sections.

## Dev Workflows (How to Run Things)

- Refresh data:
  - `.venv/bin/python -m nfl_predictor.data_collection`
- Train/predict (CLI):
  - `.venv/bin/python -m nfl_predictor.ml_model --help`
- Walk-forward evaluation:
  - `.venv/bin/python scripts/walk_forward_backtest.py --help`
- Weekly orchestration:
  - `.venv/bin/python scripts/weekly_run.py --help`
- Testing:
  - `.venv/bin/python -m pytest`

Training/prediction entrypoints may be updated/replaced, but must remain runnable and documented.

## Logging & Coding Style

- Logging uses the project logger (`from nfl_predictor.utils.logger import log`). No `print`.
- Prefer Polars expressions over Python loops in ETL.

## Safety, Scope, and Prohibited Behaviors

- Do not add unrelated features (new scrapers, unrelated pipelines). The web UI under
  `nfl_predictor/api/` and `web/` is in scope; extend it by phase per `.agents/web_ui_plan.md`.
- Do not remove/alter existing pipeline behavior without updating tests and documentation.
- Avoid new external services or network dependencies beyond existing scraping utilities.
- Do not claim betting profitability; report metrics and uncertainty honestly.

## Choosing APIs & Libraries

- Polars is used for ETL and feature engineering.
- `nflreadpy` is used for NFLverse data.
- scikit-learn and XGBoost are used for modeling.

## Assistant guidance

- Use existing project utilities and constants.
- Implement changes in small, testable increments.
- Keep outputs deterministic under fixed seeds.
- Do not change behavior without updating tests and documentation.
- Do not delete `models/<run_id>/wf_compare/` during active walk-forward runs; those artifacts power
  resume behavior.
- Keep walk-forward comparison artifacts under `models/` (do not point `--out-json` at a temporary
  directory). Any number reported in `.agents/` or `AGENTS.md` must be auditable from disk.
- Walk-forward runs checkpoint every finished week (`models/wf_checkpoints/<fingerprint>/` by
  default; `wf_compare/wf_folds/` inside `weekly_run` runs). After a stop, re-run the
  identical command and it resumes at the next unfinished week; `--no-resume` retrains
  everything. Watch progress in the run's log: every finished week prints a
  `Walk-forward fold N/M done` line with elapsed and remaining time. For `weekly_run` comparisons,
  `wf_compare/wf_summary.csv` still shows per-candidate results.
- Run **one walk-forward at a time**. XGBoost uses every core, and on 2026-09-10 two concurrent
  from-week-1 runs each burned more CPU than a whole solo run (42 CPU-hours against a solo run's
  roughly 30) without finishing, so both were stopped and rerun in sequence. Measured 2026-09-20
  on an idle machine: a from-week-1 run over `--eval-last-n-seasons 3` takes about 50 minutes and
  over `--eval-last-n-seasons 6` about 100 minutes. The earlier observation, kept as a record, put
  a from-week-1 three-season run at about 75 minutes alone and a week-3 start at about 40.
- Launch every walk-forward through a small `launch.sh` in its own run directory (see
  `models/wf_m59_rebuild_2023_2025_from_week1_seed7/launch.sh` for the shape) with
  `nohup setsid`, never through a harness-bound shell, which stops at 10 minutes. Never
  `pkill -f` a pattern that can match your own shell. Several accepted rungs run back to back
  through one driver script. Two lessons from 2026-09-23:
  - Any logic added to a `launch.sh` (a load probe, for example) is tested under the script's own
    strict-mode header, because `IFS=$'\n\t'` changes how `read` splits. A probe written without
    it failed and stopped the whole queue overnight.
  - Never execute a fragment cut from a `launch.sh`: a cut that includes the
    `walk_forward_backtest.py` line starts a second walk-forward with default settings.
  Check the driver's log after its first run-to-run transition, not only at the end.
- Choose the OpenMP wait policy by machine load at launch. Under other load, use
  `OMP_WAIT_POLICY=PASSIVE`: with the default policy XGBoost's threads spin while a preempted peer
  catches up (on 2026-09-10 one week took `730s` by default and `185s` with `PASSIVE`). On an idle
  machine keep the default, because sleeping threads cost more to wake than they save on this
  small dataset (an idle `PASSIVE` week took `~142s` against `~82s` for the default). The setting
  changes scheduling only, so it neither alters results nor invalidates fold checkpoints; switching
  mid-run means stop, relaunch with the other policy, and resume.
- Load observation from 2026-09-20: with the web API running `--reload` (its file watcher takes
  about half a core continuously) a three-season from-week-1 run took about 110 minutes instead of
  about 50 idle, so the seed-7 arm was relaunched with `OMP_WAIT_POLICY=PASSIVE` and resumed from
  its checkpoints. Check `uptime` and `ps -eo pcpu,args --sort=-pcpu | head -4` before a launch.
