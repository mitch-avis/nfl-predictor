# Changelog

## [0.17.0] - 2026-09-23

### Changed

- Close task 55.8 after a reviewed four-arm six-season season-weighting ladder on the current
  `pbp`-default build (`models/wf_m55_8_2020_2025_{unweighted,half_life4,half_life8,half_life16}/`,
  each with a `REVIEW.md`). Weeks 3-18 deterministic Brier came in at `0.2107` / `0.2115` /
  `0.2107` / `0.2102` for unweighted / half-life `4` / half-life `8` / half-life `16`; the
  paired intervals against the unweighted reference and against half-life `4` all still covered
  zero, so the ladder reads as flat within noise rather than as a default-changing win for a
  different value.
- Keep the shipped production season weighting at `train_recency_half_life_seasons: 4`, because no
  reviewed arm beat it beyond the paired intervals on the governing weeks 3-18 window.
- Align `config/weekly_run.yaml` so the weekly walk-forward comparison stage finally measures the
  same season weighting the final training stage already uses: `wf_recency_half_life_seasons: 4`.
- Replace the README's superseded Platt-based recency ablation with the reviewed six-season ladder
  and its outcome.

## [0.16.2] - 2026-09-21

### Added

- Confirm that the `2pt_conversions` disagreement with nflverse recorded in `0.16.0`
  (`models/pbp_vs_nflverse_m54_2/COMPARISON.md`) is a systematic nflverse team-stats bug, not a
  play-by-play derivation defect, so no code change was needed. The user manually verified one
  mismatch against the actual game (2024 week 17, Green Bay at Minnesota: exactly one two-point
  conversion happened that game, matching the play-by-play row exactly, while nflverse's
  team-stats table reports two) and asked for the rest to be checked.
  `models/pbp_vs_nflverse_m54_2/verify_2pt_doubling.py` finds that across seven sampled seasons
  (`2010`, `2015`, `2020`, `2022`, `2023`, `2024`, `2025`; `3710` team-games), `246` team-games
  disagree, and `234` of those (`95.1%`) show nflverse's count at exactly double the play-by-play
  count; zero mismatches go the other way. `COMPARISON.md` is updated accordingly, and the
  `two_point_conversion_pct` rate's disagreement with the TeamRankings scrape is now described as
  unresolved rather than as a pbp-side limitation, since the scrape is a different, unverified
  third-party source.

## [0.16.1] - 2026-09-21

### Changed

- Rebuild the production dataset from a refreshed play-by-play cache
  (`--refresh-nflreadpy`, needed for the new `pass_attempt`/`rush_attempt` raw columns) on the
  `0.16.0` code: `data/completed_games_ml.csv` is now `edd6b852...` (`7292` completed rows,
  `513` columns), with the prior build backed up to `data/backup_pre_m54_flip/` (top-level CSVs)
  and `data/cache/nflreadpy/backup_pre_m54_flip/` (the pre-refresh play-by-play cache). The
  through-2025 cut is `data/completed_games_ml.m54_flip_through_2025.csv` (`2d4111a6...`).
  Leakage audit `models/audit_m54_flip_rebuild/leakage_audit.json` passed (`463` features, `0`
  flags, same shape as every prior 54.x build).
- Re-verified the four corrected box-score columns against the full rebuild: `passing_epa`
  matches nflverse on `99.33%` of the `13912` overlapping 1999-2025 team-games (up from
  `69.54%` before `0.16.0`), `pass_attempts` on `99.87%` (from `86.70%`), `rushing_epa` on
  `99.87%` (from `95.54%`), and `fumbles`/`fumbles_lost` on `91.95%`/`98.37%` (from
  `73.63%`/`90.35%`); seventeen of twenty-one derivable columns now match on `98%` or more.
  `models/pbp_vs_nflverse_m54_2/COMPARISON.md` carries the full table.
- Reviewed the default-flip walk-forward arm `models/wf_m54_flip_2023_2025_from_week1/`
  (checkpoints `models/wf_checkpoints/9779c1cbb0701d23661a/`, reviewed in its `REVIEW.md`): it
  ties the pre-flip reference `models/wf_m54_0_2023_2025_from_week1/` on the governing weeks
  3-18 window: deterministic Brier `0.2106` vs `0.2097`, diff `+0.0009` `[-0.0008, +0.0026]`;
  margin MAE `9.9321` vs `9.9166`, diff `+0.0156` `[-0.0529, +0.0812]`. A no-breakage tie by the
  written rule.

### Fixed

- The full rebuild incidentally closed the 1999-2002 Jacksonville team-stats coverage gap that
  task 54.0's schedule skeleton and box-score repair were built around: because play-by-play has
  both sides of every JAX game even where nflverse's team-stats table does not, the now-default
  `pbp` box-score overlay fills those rows before the schedule-skeleton coverage check runs, so
  the ETL logs `0` repair warnings on this rebuild instead of the `16` the 54.0 rebuild logged
  (verified: `strength_games_played` for JAX still reads `16.0` at both 2001 and 2002 season
  end). The schedule-skeleton and repair code remain in place as a safety net for the
  `nflverse`/`scrape` source configuration, which a caller can still select explicitly.

## [0.16.0] - 2026-09-21

### Changed

- Flip the default box-score and situational-percentage sources to play-by-play
  (`--team-stats-source pbp`, `--tr-stats-source pbp`), including the production fast path
  `scripts/weekly_run.py` uses when it calls `data_collection.main()` with no arguments.
  `nflverse`/`scrape` remain selectable explicitly. The change is justified by the
  `1999-2002` situational-percentage coverage gain (the TeamRankings scrape starts in 2003)
  and by fixing four columns that previously disagreed with nflverse (below), measured on a
  reviewed walk-forward no-breakage arm (see the ETL rebuild entry below).
- `passing_epa` now sums `qb_epa` (nflverse's own quarterback-attribution EPA column) instead
  of `epa`, over every `pass_attempt` play including sacks and two-point tries. Verified against
  nflverse team stats: match rate rose from `69.54%` to `98.84%` over a four-season sample
  (`100%` on 2024 alone).
- `pass_attempts`, `pass_completions`, `pass_yards`, `pass_touchdowns`, `interceptions_thrown`,
  `rush_attempts`, `rush_yards` and `rush_touchdowns` now use nflverse's own canonical
  `pass_attempt`/`rush_attempt` flags instead of `play_type`-based conditions. A sack carries
  `pass_attempt = 1` in the raw data (nflverse's own `pass_attempts` excludes it explicitly) and
  a kneel carries `rush_attempt = 1` despite `rush = 0`; the flag-based derivation now matches
  nflverse on `99.89%`-`100%` of team-games for every one of these columns, up from as low as
  `86.70%` (`pass_attempts`).
- `rushing_epa` now includes two-point tries (`99.78%` match, from `95.54%`).
- `fumbles`/`fumbles_lost` now exclude special-teams plays, matching nflverse's offense-only
  fumble stat (the sum of a player's sack, rushing and receiving fumbles). Match rate rose from
  `73.63%`/`90.35%` to `87.49%`/`98.03%`; a residual gap remains for fumbles on aborted snaps,
  which nflverse's own player-level fumble categories also do not cleanly attribute, and is
  documented rather than further chased.
- `2pt_conversions` is unchanged (`94.35%` match); tracing individual mismatches found
  nflverse's own team-stats table disagreeing with its own play-by-play on rare plays, which is
  not fixable from this side. Recorded in `models/pbp_vs_nflverse_m54_2/COMPARISON.md`.

### Added

- `pass_attempt` and `rush_attempt` join the cached raw play-by-play columns
  (`constants.PBP_COLUMNS`), needed for the corrected derivations above.

## [0.15.1] - 2026-09-21

### Added

- Record the play-by-play against nflverse team-game comparison that task 54.2 asks for, in
  `models/pbp_vs_nflverse_m54_2/` (`compare_sources.py`, `comparison.json` and
  `COMPARISON.md`). Seventeen of twenty-one derivable columns agree on `94%` or more of the
  `13912` overlapping team-games of `1999-2025` with a median difference of zero; the four
  exceptions (`passing_epa`, `fumbles`, `2pt_conversions`, `pass_attempts`) are recorded with
  what causes each.

### Fixed

- The play-by-play `total_yards` had its sack term inverted. nflverse defines
  `total_yards = pass_yards + rush_yards - yards_lost_from_sacks` and stores the sack losses as
  a negative number, so the yardage is added back rather than deducted; that identity holds on
  `13418` of `13418` nflverse team-games of 2000-2025. The derived column matched only `14.13%`
  of team-games and ran `31` yards low (`335.236` against `366.264`); it now matches `96.62%`
  with a mean of `365.515`.

## [0.15.0] - 2026-09-21

### Added

- Derive the eight situational percentages (third down, fourth down, red zone and two-point,
  for and allowed) from the play-by-play counts, behind `--tr-stats-source pbp`. The scraped
  TeamRankings columns remain the default, and TeamRankings still supplies its ratings either
  way.
- Aggregate per-team-game box-score stats from play-by-play behind `--team-stats-source pbp`,
  overlaid on the nflverse rows so nflverse still fills what play-by-play cannot derive.
- Count red-zone trips and touchdown drives per team-game (`red_zone_trips`,
  `red_zone_td_drives` and their `_allowed` mirrors) from `fixed_drive` and
  `fixed_drive_result`.
- Carry the raw play-by-play columns the new families need: `fumble`, `penalty`,
  `penalty_team`, `penalty_yards`, `first_down_pass` and `first_down_rush`.

### Fixed

- `red_zone_tds` now counts only touchdowns scored by the team in possession. It previously
  counted any touchdown on a red-zone play, so a defensive score credited the offense.
- The play-by-play `red_zone_td_pct` divides touchdown drives by red-zone trips rather than
  touchdowns by red-zone snaps. The snap denominator measured a different statistic under the
  scraped column's name: on 2023-2025 it read `0.187` where the scrape reads `56.0`, and the
  trip denominator puts 2024 at `60.4%` against `19.1%` per snap.
- The eight play-by-play situational percentages are emitted on the scraped columns' 0-100
  scale instead of 0-1, so switching `--tr-stats-source` no longer changes what the column
  means by a factor of 100. `third_down_pct` and `fourth_down_pct` now agree with the scrape
  (`39.33` against `39.07`, `54.25` against `53.79` over 2023-2025).
- `load_pbp` normalizes `td_team` and `penalty_team` to canonical abbreviations alongside the
  other team columns, so legacy aliases no longer invent team-game rows the schedule has no
  place for.
- Build the play-by-play box-score frame by joining only the perspectives that produced rows,
  instead of seeding the result from one of them and joining that same frame again.

## [0.14.0] - 2026-09-21

### Changed

- Bump the project version to `0.14.0` and keep `uv.lock` aligned.
- Task 54.0 now builds the per-team-game frame from the schedule skeleton instead of from the
  nflverse team-stats rows, so every completed regular-season game keeps its two team rows even
  when the stats source misses one side. Team stats and play-by-play counts are left-joined onto
  that skeleton, coverage gaps are logged per `(season, team)`, and schedule-driven game counts
  no longer disappear with missing team-stat rows.
- Rebuild the top-level datasets from cache on the 54.0 ETL code: `data/completed_games_ml.csv`
  is now `db8b8ff4...` (`7292` completed rows, `513` columns) with the prior top-level CSVs in
  `data/backup_pre_m54_0/`; the walk-forward input for later current-build arms is
  `data/completed_games_ml.m54_0_through_2025.csv` `e914eadf...` (`7261` rows), cut by
  `models/etl_m54_0_rebuild/cut_through_2025.py`. Leakage audit
  `models/audit_m54_0_rebuild/leakage_audit.json` passed (`463` features, `0` flags).
- The reviewed no-breakage arm `models/wf_m54_0_2023_2025_from_week1/` (checkpoints
  `models/wf_checkpoints/d112ebcba3115bafe9d9/`, reviewed in its `REVIEW.md`) ties the accepted
  `200`-tree reference slice `models/wf_checkpoints/a5e76d54187e27ca7370_2023_2025/` on the
  governing weeks 3-18 window: deterministic Brier `0.2097` vs `0.2090`, diff `+0.0007`
  `[-0.0011, +0.0024]`; margin MAE `9.9166` vs `9.9044`, diff `+0.0122`
  `[-0.0566, +0.0801]`. The rebuild moved 836 of 855 scored 2023-2025 rows in at least one
  feature, mostly in the `sos_*` and `opponent_*` EPA families, but the decision remains a
  performance-based tie.

### Fixed

- The 2001-2002 Jacksonville home rows where the surviving nflverse team-stats row carried both
  teams' box score are now repaired by nulling only the box-score columns on those one-row games,
  leaving identity, schedule scores and play-by-play counts intact so the duplicated totals do not
  contaminate season-to-date mirrors.

## [0.13.1] - 2026-09-21

### Changed

- Bump the project version to `0.13.1` and keep `uv.lock` aligned.
- Task 55.7 closes on the reviewed `100`-tree plateau check:
  `models/wf_m55_7_2020_2025_trees100/` (checkpoints
  `models/wf_checkpoints/09a60441e87d86c894ba/`, reviewed independently in its `REVIEW.md`).
  By the written governing rule, `100` ties the current default `200`, so `200` stays the
  shared default. Governing weeks 3-18, `100` vs `200`: deterministic Brier `0.2103` vs
  `0.2103`, diff `+0.0000` with 95% interval `[-0.0008, +0.0009]`; margin MAE `9.9237` vs
  `9.9281`, diff `-0.0044` with 95% interval `[-0.0389, +0.0296]`.
- `AGENTS.md`, `.agents/TODO.md` and `.agents/next_agent_session_prompt.md` now treat task 55.7
  as closed, keep `200` as the shared default after the plateau check, and move the next active
  work item to task 54.0.

## [0.13.0] - 2026-09-21

### Changed

- Bump the project version to `0.13.0` and keep `uv.lock` aligned.
- Adopt `200` as the shared default `n_estimators` in walk-forward and production together.
  `nfl_predictor/ml/ml_model_core.py` now carries the `200`-tree default, and the bare
  `scripts/weekly_run.py` parser falls back to the same `n_estimators`, `max_depth` and
  `learning_rate` as the production XGBoost defaults.
- `scripts/weekly_run.py` Stage 1 no longer forces `subsample` and `colsample_bytree` to `0.9`,
  so the weekly walk-forward comparison now evaluates the same XGBoost parameter set the final
  fit uses by default.
- `config/weekly_run.yaml` now matches the approved weekly defaults: `tune: false`,
  `wf_include_postseason: false`, `include_postseason: false`, `wf_n_estimators: 200`,
  `wf_max_depth: 5` and `wf_learning_rate: 0.0165`. `postseason_weight: 1.3` stays in place
  but is inert unless postseason training is explicitly enabled.
- `README.md` and `AGENTS.md` now describe the `200`-tree default and the shipped weekly config
  accurately, while keeping the ladder measurements and benchmark history intact.

## [0.12.15] - 2026-09-21

### Changed

- Bump the project version to `0.12.15` and keep `uv.lock` aligned. Documentation only: this
  entry records the user's decisions of 2026-09-21 on the tasks they belong to, and no
  executable code changes.
- `.agents/TODO.md`, task 55.7: the user adopts `200` as the shared default `n_estimators` in
  walk-forward and production together (an approved default change, to land as `0.13.0`). The
  same chunk makes `config/weekly_run.yaml` consistent: `tune: false`, because today's
  `tune: true` re-runs a one-hour Optuna study on every weekly run and was not intended, and the
  walk-forward stage's XGBoost params aligned with the production defaults, so both stages fit
  the same model. One more rung, `100`, then runs as the plateau check; if it ties `200`, `200`
  stays, and if it is better beyond the fit-noise floor, it is reported and asked about.
- `.agents/TODO.md`: new task 55.9, "Optuna re-tune", to run after Milestone 54 lands, ideally
  on a bye week or in the off-season. Three prerequisites land first, each as its own tested
  chunk: trials must fit the way production fits (`_score_margin_total_fold` still passes
  `early_stopping_rounds`, which `0.12.3` removed from production), the objective must score the
  deterministic Brier the walk-forward instrument scores rather than the configured
  calibrator's, and a tuned parameter set needs plumbing to reach the walk-forward. The study
  itself keeps the current search space, TPE seed `42`, SQLite storage and `300`-`500` trials,
  and its winner is confirmed on six seasons and a second seed. The studies under
  `models/weekly_2025_week_21/` and `models/weekly_2025_week_22/` are void.
- `.agents/TODO.md`, task 56.2: the user approves setting `include_postseason: false` and
  `wf_include_postseason: false` in `config/weekly_run.yaml`, leaving `postseason_weight` in
  place but inert, so the weekly run matches the rule that a regular-season model never trains
  on playoff games. The verified facts are recorded with it: season-to-date stats, strength,
  records and the prior-season blend are regular-season only, and Elo, QB Elo and the
  TeamRankings playoff-week snapshots are the only features carrying playoff results, as
  pre-game ratings with no leakage. The playoff-week design stays deferred.
- `.agents/TODO.md`, task 54.0: the user approves proceeding with the parked branch
  `feat/m54-0-schedule-skeleton` including the box-score repair (`ae687dd`, `d34b0ab`), and the
  ETL rebuild that follows it (back up `data/*.csv` to `data/backup_pre_m54_0/` first, rerun the
  leakage audit, cut `data/completed_games_ml.m54_0_through_2025.csv`). The tie check is one
  three-season from-week-1 arm at the new `200` default against the 2023-2025 folds of
  `models/wf_m55_7_2020_2025_trees200/`, which are the `200` reference on the previous build.
  The blend-weight follow-up from the 54.0 audit is resolved as "do nothing": task 54.2 erases
  it.
- `.agents/TODO.md`, "Roadmap Status": the execution order set by the user on 2026-09-21, from
  the 55.7 close-out through 56.2, 54.0, 54.1 and 54.2, 55.8, 55.9 and the Milestone 60 CLI
  inventory; and "Current validated baseline" restated at `0.12.15` on `main` at `3729006`.
- `.agents/next_agent_session_prompt.md`: rewritten starting state, order of work and open
  questions. `main` carries everything through `0.12.14`, no walk-forward is running, the web
  API watcher still loads the machine, nothing blocks the next session, and the first check-in
  is a report rather than a question.

## [0.12.14] - 2026-09-21

### Changed

- Bump the project version to `0.12.14` and keep `uv.lock` aligned.
- `AGENTS.md`, "Delegation guardrails" rule 4: a walk-forward decision rule must now name the
  exact window or windows it reads, the exact columns, and how the windows combine when they
  disagree, so a rung cannot read as a tie on one window and a win on another without the rule
  saying which governs. A ladder (several runs of one hypothesis family with a stated cap and
  stopping rule) written into the check-in and accepted by the user now counts as approval for
  every rung up to that cap.
- `AGENTS.md`, rule 5: the must-ask item "a third walk-forward run on one task, or any
  six-season run" excludes rungs of a ladder already accepted under rule 4 and within its cap.
  The section intro records why both amendments were asked for.
- `.agents/TODO.md`, task 56.2: the user's direction that postseason matchups stay in the data
  but are kept separate from regular-season matchups for training and prediction, that a
  regular-season model should not train on playoff games, that the prior-season blend should
  draw on the previous regular season only, and that the design discussion is deferred until
  the rest runs smoothly, in time for the 2026 playoffs.
- `.agents/TODO.md`, task 58.4: closed by the user's decision. `NFLP_DATA_DIR` stays, because
  the web API resolves its paths through it and it lets the app run against a copied data tree,
  but the ETL's upstream inputs will not follow it; the `0.12.12` narrowing is the final shape.
- `.agents/TODO.md`: new Milestone 60, "CLI consolidation", to audit every `add_argument` across
  the entrypoints, list the inert, duplicated and config-only flags, and propose removals plus a
  shared options module; the removals land only after the user signs off. Task 55.7 carries a
  note that the user asked whether a fresh Optuna run should choose `n_estimators` rather than
  adopting `200` directly.

## [0.12.13] - 2026-09-21

### Changed

- Bump the project version to `0.12.13` and keep `uv.lock` aligned.
- `AGENTS.md` records the task 55.7 tree-budget ladder: three six-season rungs of the reference
  configuration on the rebuilt build that differ only in `--n-estimators`
  (`models/wf_m55_7_2020_2025_trees598/`, the reference, which reproduces the three-season
  reference arm's 2023-2025 folds bit for bit, plus
  `models/wf_m55_7_2020_2025_trees200/` and `models/wf_m55_7_2020_2025_trees400/`), each one
  independently rescored into its `REVIEW.md`. Weeks 3-18 over `1423` games: deterministic
  Brier `0.2103` (`200`), `0.2111` (`400`) and `0.2117` (`598`) against market Brier `0.2095`,
  with margin MAE `9.9281`, `9.9978` and `10.0240`; the `200 - 598` paired difference is
  `-0.0014` `[-0.0029, +0.0001]` on Brier and `-0.0958` `[-0.1583, -0.0345]` on margin MAE. The
  ladder stopped on its "report all three and ask" branch: no default change, and the choice of
  budget is with the user.
- `README.md`, "Backtesting": fresh walk-forward durations measured during the ladder (three
  seasons from week 1 about 50 minutes idle and about 110 loaded; six seasons about 100 minutes
  idle at `200` trees, about 3.3 hours at `400`, about 4.8 hours at the default `598` under
  load), replacing the older "about 75 minutes" line, and a pointer from `--n-estimators` to the
  ladder record.
- `.agents/TODO.md`: task 55.7 stays open with a progress note carrying the ladder result, the
  three run directories and the decision pending with the user.

## [0.12.12] - 2026-09-20

### Changed

- Bump the project version to `0.12.12` and keep `uv.lock` aligned.
- The `etl_full`, `validate_offline` and `validate_live` job templates pass `--data-dir` from
  `NFLP_DATA_DIR`, so the datasets those jobs read and write follow the configured data tree
  instead of the checkout's own `data/`. The ETL's upstream inputs (`qb_elos.csv`, the
  quarterback identity file, the TeamRankings and nflreadpy caches) still resolve from
  `constants.DATA_PATH`; task 58.4 stays open for that remainder.
- `python -m nfl_predictor.data_collection`, `scripts/validate_offline.py` and
  `scripts/validate_live.py` accept `--data-dir`; omitting it keeps the packaged data directory,
  so every existing invocation behaves as before.

### Removed

- The `web` optional-dependency extra in `pyproject.toml`, whose seven packages are all core
  dependencies already. `scripts/gate.sh`, `.github/workflows/validation.yml`, `web/README.md`
  and the agent docs now say plain `uv sync` / `uv sync --check --active`.

## [0.12.11] - 2026-09-20

### Changed

- Bump the project version to `0.12.11` and keep `uv.lock` aligned.
- The default power-rankings through-week in `scripts/weekly_run.py` is clamped to the last
  regular-season week when the prediction week is postseason (the ETL writes strength snapshots
  only through the week after the regular season, so later playoff weeks skipped the rankings).
  An explicit `--power-rankings-through-week` still wins.

### Added

- `--data-collection-args` on `scripts/weekly_run.py` (config key `data_collection_args`, also a
  parameter of the `weekly_run` API job): one shell-quoted string passed through to the data
  refresh, so `--min-season`, `--max-season` and the `--stat-prior-blend*` flags can be set from
  the weekly command. Unset, the refresh runs exactly as before.
- README: how postseason games enter training, evaluation and the rankings today (the code
  defaults exclude them; the shipped `config/weekly_run.yaml` includes them at weight `1.3`).

## [0.12.10] - 2026-09-20

### Changed

- Bump the project version to `0.12.10` and keep `uv.lock` aligned.
- `AGENTS.md` walk-forward operating notes carry the `launch.sh` / `nohup setsid` launch
  convention, the 2026-09-20 idle durations (three seasons about 50 minutes, six about 100) and
  the load-driven `OMP_WAIT_POLICY=PASSIVE` relaunch; `.agents/TODO.md` restates the validated
  baseline at `0.12.9`.

### Added

- `--n-estimators` on `scripts/walk_forward_backtest.py`: an XGBoost tree-budget override that
  flows through `WalkForwardConfig.xgb_params_overrides` like `--gamma`, so the budget ladder of
  task 55.7 runs as separate arms without touching `nfl_predictor/ml/` or any default.

## [0.12.9] - 2026-09-20

### Changed

- Bump the project version to `0.12.9` and keep `uv.lock` aligned.
- Roadmap reordered by the user's decision: Milestone 55 tasks 55.7 (the tree budget) and the
  new 55.8 (season weighting, re-measured on the deterministic instrument over six seasons) run
  before Milestone 54; task 54.0 is read as a no-breakage check on three seasons. The feature
  rules in `.agents/TODO.md` now require six seasons for any arm that claims an improvement.
- The README's recency-weighting ablation is marked superseded (measured through the
  noise-dominated Platt calibrator on an earlier build); task 55.8 replaces it.
- `.agents/next_agent_session_prompt.md` rewritten for the next session.

## [0.12.8] - 2026-09-20

### Changed

- Bump the project version to `0.12.8` and keep `uv.lock` aligned.
- Record the fit-noise floor of the walk-forward instrument in `AGENTS.md`: the reference arm
  rerun with only `--random-seed 7` (`models/wf_m59_rebuild_2023_2025_from_week1_seed7/`,
  rescored in its `REVIEW.md`) moves every margin (median `0.91` points), flips 46 of 720 picks
  and shifts weeks-3-18 deterministic Brier by `+0.0020` `[-0.0006, +0.0045]` and pick accuracy
  by `+0.0083`. The 2026-09-20 rebuild's 7-game pick-accuracy drop is therefore fit noise, and
  single three-season arms cannot resolve effects below those sizes.
- The Week 2 package was refreshed for the Sunday and Monday games
  (`models/weekly_2026_week_02_refresh/`): market lines refreshed first, then the weekly run
  on the rebuilt dataset with the data refresh skipped.

## [0.12.7] - 2026-09-20

### Changed

- Bump the project version to `0.12.7` and keep `uv.lock` aligned.
- The dataset on disk is the 2026-09-20 rebuild on the `0.12.6` schema (`7278` rows, `513`
  columns, `db6a78a3...`; previous build kept in `data/backup_pre_m59_rebuild/`): the six sack
  mirrors are gone and the 1999-2001 division and conference derived columns use the pre-2002
  alignment. Leakage audit `463` features, `0` flags. The from-week-1 walk-forward on its
  through-2025 cut (`models/wf_m59_rebuild_2023_2025_from_week1/`, rescored in its `REVIEW.md`)
  ties the standing benchmark (weeks 3-18 deterministic Brier `0.2096` against `0.2099`, paired
  `[-0.0028, +0.0023]`; margin MAE `+0.0113`) and becomes the reference arm for new work on the
  rebuilt build; `AGENTS.md` carries its table beside the benchmark.

## [0.12.6] - 2026-09-20

### Changed

- Bump the project version to `0.12.6` and keep `uv.lock` aligned.

### Fixed

- The `0.12.4` exclusion of `times_sacked` from the opponent mirror starved
  `opponent_points_per_play`, which divides `points_allowed` by the opponent's pass attempts,
  rush attempts and times sacked: the first ETL rebuild on that schema (2026-09-20) published
  `away_/home_opponent_points_per_play`, `away_/home_points_per_play_margin` and their two
  diffs as all-null columns (`models/etl_m59_rebuild/etl_defective_first_pass.log`, the schema
  enforcement line). The ETL now builds the mirrors named in
  `constants.OPPONENT_MIRROR_INTERMEDIATES` (`times_sacked` only) per game for the derivation
  and drops them when the final schema is selected, so `opponent_times_sacked` stays
  unpublished and the six derived columns keep the values the previous build had.

## [0.12.5] - 2026-09-19

### Changed

- Bump the project version to `0.12.5` and keep `uv.lock` aligned.
- Walk-forward fold metrics record `<head>.early_stopped` beside `<head>.best_iteration`, and
  the `at_cap` warning fires only when early stopping ran and never triggered. A head that used
  its whole `n_estimators` budget by configuration is recorded, not warned about (before this,
  every fold of every `0.12.3`-`0.12.4` run logged `at_cap` for every head). `resolved_settings`
  records `in_season_early_stopping: false`, and the `--wf-early-stopping-rounds` /
  `--train-early-stopping-rounds` help text says what the flags still do (the run fingerprint
  and Optuna trials only).
- Rewrite the `0.12.1` to `0.12.4` entries below into Common Changelog sections and correct
  their claims after the 2026-09-19 audit of the two sessions that produced them: the benchmark
  table comes from the retrained fit-parity arms, not from a rescore of the `0.12.0`
  checkpoints; the fitted calibration pool is in-sample, not out-of-fold; and `n_estimators` was
  not re-tuned. The audit record is in `.agents/ARCHIVE.md`, Milestone 59, "Audit".
- Reopen the narrowed parts of tasks 59.2 and 59.3 as follow-ups in `.agents/TODO.md` ("From
  Milestone 59") and as task 55.7.

### Added

- `scripts/gate.sh`: the single validation gate. It runs every check that
  `.github/workflows/validation.yml` runs, in CI's order, and reports all of them before exiting
  non-zero; `--web` adds the frontend gate and `--quick` skips pytest.
- "Delegation guardrails" in `AGENTS.md`: the gate rule, the narrowing rule, the two-key rule
  for numbers written into docs, the compute budget, and the must-ask / may-proceed lists for
  autonomous sessions.
- `sigma` (residual-scale Normal-CDF map) is selectable by name in every calibration choice
  list; before, it was reachable only as the fallback for undersized isotonic requests.

### Removed

- The unreachable fitted-`auto` selector (`_select_auto_calibration_method`) and its floor
  check (`_sigma_calibrator_improves_on_floor`), left behind when `auto` was locked to the
  deterministic floor in `0.12.2`, and the undocumented `deterministic` alias for `sigma`.

### Fixed

- `nfl_predictor/constants.py` failed `ruff format --check` (one missing blank line after
  `conference_map_for_season`), so CI would have failed on the `0.12.4` tree.
- The `AGENTS.md` benchmark table states which run its numbers come from and how far a true
  rescore of the `0.12.0` checkpoints sits from them.

## [0.12.4] - 2026-09-19

### Changed

- Bump the project version to `0.12.4` and keep `uv.lock` aligned.
- `def_sacks` and `times_sacked` join `EXCLUDE_FROM_OPPONENT_STATS`, so the next ETL rebuild
  drops the six `*_opponent_def_sacks` / `*_opponent_times_sacked` mirrors that `0.12.0` prunes
  at training time. The dataset on disk (2026-09-17 build) predates this change; the
  training-time prune tolerates the columns disappearing.
- Remove the dead fitted-selector call sites from production and walk-forward so `auto` reads as
  the deterministic floor in code as well as in the docs.

### Added

- `PRE_2002_TEAM_TO_DIVISION`, `PRE_2002_TEAM_TO_CONFERENCE`, `division_map_for_season` and
  `conference_map_for_season` in `constants`, and season-aware division and conference
  expressions in `features.py`.
- The `rare_events` feature group (`special_teams_tds`, `def_fumbles`, `fumble_recovery_tds`,
  `2pt_conversions`, `def_safeties`, `def_tds`) for `--disable-feature-groups`, and
  `--min-child-weight` / `--gamma` on `scripts/walk_forward_backtest.py`.
- Three six-season arms (`--eval-last-n-seasons 6`, from week 1, `auto`, `market_anchor` on,
  `data/completed_games_ml.m49_through_2025.deadweight_cut.csv`), run one at a time:

  | arm | deterministic Brier | market Brier | paired 95% CI | margin MAE |
  | --- | --- | --- | --- | --- |
  | `models/wf_m59_2020_2025_auto_floor_baseline/` | `0.2116` | `0.2104` | `[-0.0016, +0.0041]` | `9.9015` |
  | `models/wf_m59_2020_2025_rare_events_off/` | `0.2123` | `0.2104` | `[-0.0010, +0.0049]` | `9.9101` |
  | `models/wf_m59_2020_2025_regularized_gamma5_mcw5/` | `0.2114` | `0.2104` | `[-0.0018, +0.0039]` | `9.8903` |

  Every interval covers zero: the rare-event family is not measurable noise at this sample
  size and stronger regularization is a tie. The baseline's 2023-2025 folds are bit-identical
  to the `0.12.3` from-week-1 arms and reproduce the `AGENTS.md` table; its configured `auto`
  probabilities equal the deterministic floor, none leaves `[0.02, 0.98]` unless
  `|predicted_margin| > 14`, and every head ran the full `598`-tree budget.

### Fixed

- 1999-2001 rows used today's division map. `is_divisional_matchup`, the division and
  conference records, the lookahead context and the standings proxies now use the pre-2002
  alignment, and the divisional flag agrees with nflverse `division` in every season (it
  disagreed on 61, 67 and 61 rows in 1999, 2000 and 2001). The dataset on disk predates this
  fix; the corrected values arrive with the next ETL rebuild.

## [0.12.3] - 2026-09-19

### Changed

- Bump the project version to `0.12.3` and keep `uv.lock` aligned.
- Remove per-fold early stopping from the in-season production and walk-forward fits. Both call
  the same shared XGBoost fit helpers and run the full `n_estimators` budget (`598`), which is
  what the walk-forward folds already did in practice (patience never fired) and what the Week 2
  production run did not (its margin head stopped at iteration `0` on a 64-game window).
  `metadata.json` records a fallback `best_iteration` for every head even without early
  stopping, and walk-forward fold metrics carry per-head iteration fields. Not done, and
  reopened as task 55.7: choosing `n_estimators` from a season-sized time-aware tuning.
- Measured on the deadweight cut from week 1, seasons 2023-2025
  (`models/wf_m59_2023_2025_from_week1*/`, five arms that differ only in the configured
  calibrator): against the `0.12.0` arm, 247 of 816 predicted margins move, by at most `0.34`
  points; deterministic weeks-3-18 Brier / log loss / margin MAE are `0.2099` / `0.6073` /
  `9.9608` against `0.2098` / `0.6072` / `9.9600`, a tie.

## [0.12.2] - 2026-09-19

### Changed

- The calibration frame for fitted calibrators is the previous two seasons plus the completed
  weeks of the current season (`select_calibration_data` in walk-forward,
  `_pooled_calibration_frame` in production) instead of the last four weeks of the eval season.
  The rows are in-sample: the model that predicts them was trained on them. The out-of-fold pool
  the task asked for is an open follow-up (`.agents/TODO.md`, "From Milestone 59").
- `auto` resolves to the deterministic floor (`none`). Five three-season arms measured fitted
  alternatives on the pooled frame and none beat the floor on weeks 3-18 (configured Brier:
  isotonic `0.2378` with log loss `2.02`, sigma `0.2119`, centered sigma `0.2119`, sigma with a
  floor fallback `0.2119`, validation-selected `0.2347`; the floor `0.2099`). Run directories and
  the table are in `.agents/ARCHIVE.md`, Milestone 59.

### Added

- `sigma` calibration: one residual standard deviation estimated on the calibration frame, used
  in `Phi(margin / sigma)`. An explicit `isotonic` request below the 200-row threshold falls
  back to it.
- Platt scaling chooses `C` from `{0.01, 0.1, 1, 10}` on the latest pre-eval season of the
  calibration frame instead of using the unregularized default.

## [0.12.1] - 2026-09-19

### Changed

- Candidate ranking in `weekly_run`, `betting_pipeline`, `wf_compare` and the API reader sorts by
  deterministic Brier then deterministic log loss instead of the configured columns.
- The `AGENTS.md` benchmark table is rebuilt on the deterministic and market-implied columns with
  the market row beside the model's.

### Added

- Three probability views on the same games in every walk-forward report: the configured
  calibrator, the deterministic `Phi(margin / SCORE_DIFF_STD_DEV)` map, and the market-implied
  home win probability (no-vig moneyline, `Phi(spread / sigma)` when moneylines are missing).
  Per-window rows (week 1, week 2, weeks 3-18, all) carry paired deterministic-minus-market Brier
  and log-loss differences with 5000-sample bootstrap intervals, in `metrics.windows` and the
  summary table. `deterministic_*` and `market_*` columns are written to `wf_compare.csv` by
  `scripts/wf_compare.py`, `scripts/weekly_run.py` and `scripts/betting_pipeline.py`, and the API
  column registry and metrics reader expose them.

## [0.12.0] - 2026-09-18

### Changed

- Bump the project version to `0.12.0` and keep `uv.lock` aligned.
- Prune 20 more columns at training time (`constants.PRUNED_FEATURE_COLUMNS`; `482` to `462`
  features, no schema change, no rebuild): `is_divisional_matchup` (equal to the schedule's
  `division` flag from 2002 on and derived from today's division map before that), the three
  `season_phase_*` one-hots (repeats of `week_in_season_norm`), `away_ties`,
  `home_division_eliminated_proxy` and `away_division_rank` (the last unpruned members of
  families whose other sides were already out), `strength_games_played_diff` (a constant zero
  outside bye weeks), the six next-game flags `*_next_is_home`, `*_next_is_divisional_matchup`
  and `*_days_to_next_game`, and the six sack mirrors `*_opponent_def_sacks` /
  `*_opponent_times_sacked` plus their diffs (`times_sacked` / `def_sacks` seen from the other
  sideline, `r = 0.998`). Selection evidence: gain aggregated over 18 retrained walk-forward
  folds (`models/feature_audit_2026_09_18/feature_ranking.json`), where each of the first 14
  carries under `0.05%` of total gain, and the near-duplicate scan in the same directory.
  Walk-forward on `data/completed_games_ml.m49_through_2025.csv`, benchmark config from week 1,
  `models/wf_deadweight_2023_2025_pruned/` (checkpoints `bae0e56db951d1a890d4`) against the
  `0.11.0` arm `models/wf_m49_gp_2023_2025_pruned/`, both scored through the benchmark's Platt
  calibrator and through the deterministic map `Phi(margin / SCORE_DIFF_STD_DEV)`:

  | window | games | Platt Brier after / before | deterministic Brier after / before | margin MAE after / before |
  | --- | --- | --- | --- | --- |
  | week 1 only | 48 | `0.2066` / `0.2024` | `0.2066` / `0.2024` | `9.1501` / `9.0399` |
  | week 2 only | 48 | `0.2306` / `0.2282` | `0.2306` / `0.2282` | `8.5436` / `8.4502` |
  | weeks 3-18 | 720 | `0.2295` / `0.2324` | `0.2098` / `0.2106` | `9.9600` / `9.9647` |
  | all weeks | 816 | `0.2283` / `0.2304` | `0.2111` / `0.2111` | `9.8291` / `9.8212` |

  Paired bootstrap (5000 resamples, seed 0), after minus before, weeks 3-18: deterministic Brier
  `-0.0007` `[-0.0032, +0.0016]`, Platt Brier `-0.0029` `[-0.0077, +0.0022]`, margin MAE
  `-0.0046` `[-0.1003, +0.0924]`; every window's interval covers zero. The pruned arm's
  deterministic weeks-3-18 Brier equals the market spread's through the same map (`0.2098`
  against `0.2099`). A tie is the expected outcome for removing columns the trees did not use;
  the change is a cleanup, not a gain. The audit that produced it, including the finding that
  the benchmark's Platt columns are noise-dominated, is Milestone 59 in `.agents/TODO.md`.

## [0.11.0] - 2026-09-17

### Changed

- Bump the project version to `0.11.0` and keep `uv.lock` aligned.
- Prune `away_games_played` as well as `home_games_played`, so neither side reaches the model.
  Both duplicate `away_/home_strength_games_played`, which already publish the same count per
  side with a diff: in the 2023-2025 evaluation window, weeks 3-18, the two agreed in 759 of 759
  rows. Measured against the Milestone 53 baseline arm (`models/wf_qbsched_2023_2025_off/`, the
  same benchmark config from week 1, 483 features against 482): week 1, where the fallback lie
  lived, improves on every metric (Brier `0.2024` against `0.2051`, log loss `0.5945` against
  `0.5996`, pick accuracy `0.7917` against `0.7708`); weeks 3-18, where the column was an exact
  duplicate, drift slightly the other way (Brier `0.2324` against `0.2282`) with every interval
  covering zero. Since no information can be lost by dropping a value-identical column, that
  drift is a `colsample_bytree = 0.6098` sampling artifact: two copies of one fact reach a tree
  with probability `0.85`, one copy with `0.61`. Full table in `.agents/ARCHIVE.md`, Milestone 49.
- Rebuild `data/completed_games_ml.csv` on the corrected column (`7278` rows, `519` columns,
  `8bacad41...`); the previous build is in `data/backup_pre_m49_games_played/`.

### Fixed

- `away_games_played` / `home_games_played` now carry the record-feature count they are
  declared as (`constants.RECORD_FEATURE_COLUMNS`): the team's completed games this season,
  `wins + losses + ties`. The season-to-date stat frame produces a column of the same name
  holding the row count behind its means, and the records join silently suffixed the record
  values away, so a prior-season fallback row published the _previous_ season's total: `17` in
  week 1 (and `8` / `16` for the postponed first games of JAX 2001, JAX 2002 and MIA 2017)
  beside the `wins = 0, losses = 0, ties = 0` it should have agreed with. The stat-frame copies
  are dropped before the join, so the record values keep the names.

## [0.10.0] - 2026-09-17

### Changed

- Bump the project version to `0.10.0` and keep `uv.lock` aligned.
- Rebuild `data/completed_games_ml.csv` from the nflreadpy cache on the new schema (`519`
  columns); the previous `525`-column build is in `data/backup_pre_m53_6_drop/`.

### Removed

- Drop the quarterback schedule lenses (`qb_faced_pass_def_adj`, `qb_faced_pass_def_raw`, per
  side plus `_diff`; six columns) from the schema, by the user's decision after the `0.9.0`
  walk-forward measured no gain. The reasoning, recorded in `.agents/ARCHIVE.md` Milestone 53
  ("53.6"): a schedule faced is a nuisance parameter for estimating the quarterback's skill, not a
  predictor of the next game; as standalone columns the lenses left the subtraction
  `production - expected production given schedule` to the model, in a season-to-date window
  that matches neither the career nor the last-8 rates; and the team-level ridge pass offense is
  already opponent-adjusted. Gone: `constants.QB_SCHEDULE_STATS`, the `qb_schedule` feature
  group, the `_schedule_lenses` machinery in `qb_stats.py`, the `defense_games` / `snapshots`
  arguments of `attach_qb_features` and `data_collection._attach_qb_features`, and their tests.
  The quarterback-game rows keep `opponent_abbr`, which a defense-adjusted quarterback rate (the
  form the idea takes if revisited; see `.agents/TODO.md` task 53.7) would need. The `qb` group
  is the seven per-dropback stats again.

## [0.9.0] - 2026-09-11

### Changed

- Bump the project version to `0.9.0` and keep `uv.lock` aligned.
- Quarterback games (`qb_stats.aggregate_qb_game_stats`) now record the defense faced
  (`opponent_abbr`). The quarterback module reuses the play-by-play expression helpers and
  `calculate_stat_differentials` instead of its own copies; the existing quarterback columns are
  unchanged on a rebuild (every 1999-2025 value identical).
- The `qb` feature group now includes the schedule lenses below; the new `qb_schedule` group
  drops only the lenses.

### Added

- Add the quarterback schedule lenses (`constants.QB_SCHEDULE_STATS`, per side plus `_diff`), the
  head-to-head-excluded opponent-profiling method of the `nfl-sos-ratings` project applied to the
  expected starter. Both average his games earlier in the row's season, weighted by his dropbacks
  in each. `qb_faced_pass_def_adj` is the ridge form (the sos `QSoS` construct): each faced
  defense's `adj_def_pass_epa_snap` from the strength snapshot of the week it was faced (higher is
  tougher). `qb_faced_pass_def_raw` is the one-hop form, like `sos_played_raw`: each faced
  defense's EPA per dropback allowed in its games before the row's week, excluding its games
  against the quarterback's team (higher is easier). The ETL feeds them the play-by-play
  team-game counts and the weekly strength snapshots it already builds. Measured in walk-forward
  (benchmark config, 2023-2025, `--disable-feature-groups qb_schedule` for the off arm): no gain,
  weeks 3-18 Brier `0.2312` on against `0.2282` off (paired difference `+0.0030`
  `[-0.0019, +0.0080]`); whether to keep them is open.

## [0.8.0] - 2026-09-11

### Changed

- Bump the project version to `0.8.0` and keep `uv.lock` aligned.
- Make the web server libraries (`fastapi`, `uvicorn[standard]`, `pyjwt`, `argon2-cffi`,
  `sse-starlette`, `pydantic-settings`, `python-multipart`) core dependencies, so a plain
  `uv sync` installs everything `nfl_predictor.api` and its tests import; the `web` extra still
  exists and now adds nothing. Add `httpx` to the dev group for the API test client.
- Exclude `web/` from ruff and pyright, and `web/node_modules/`, `web/dist/` and `data/web/` from
  git and markdownlint.

### Added

- Add a web UI for the project: a FastAPI backend (`nfl_predictor/api/`, run with
  `.venv/bin/python -m nfl_predictor.api`) and a Vite + React 19 + Tailwind single-page app
  (`web/`, built into `web/dist/` and served by the backend). Sign-in is username/password with
  argon2 hashes and a signed JWT cookie, with `viewer` and `admin` roles and a bootstrap CLI
  (`python -m nfl_predictor.api.auth.cli`). The backend indexes `models/*/metadata.json`, lets an
  admin mark one run **active**, and serves that run's predictions, confidence picks, betting
  table (derived from the predictions with the workbook formulas; totals are not actionable),
  power rankings with week-over-week movement, model metadata, metrics, feature importance and
  calibration, and data/ETL status, through a column registry (`nfl_predictor/api/registry/`)
  that drives labels, tooltips and heatmaps. Pages: Overview, Predictions, Power Rankings,
  Betting, Data & ETL, Model, Runs, Users, Jobs and Glossary, on desktop and at phone width.
- Run the project's CLIs as background jobs from the UI (`nfl_predictor/api/jobs/`): a subprocess
  runner with a SQLite job table, persisted logs streamed over server-sent events, progress from
  the walk-forward candidate log line, cancel, and one worker for the walk-forward group so two
  walk-forward runs never overlap. Templates: `etl_full`, `lines_refresh`, `weekly_run`,
  `train`, `predict`, `predict_week`, `power_rankings`, `betting_xlsx`, `leakage_audit`,
  `validate_offline`, `validate_live`, `walk_forward_backtest` and `shap_analysis`.
- Refresh market lines without a full ETL (`nfl_predictor/lines_refresh.py`): reload the
  nflverse schedule for one season, fill missing moneylines from the spreads, and update the
  line columns of that season's rows in `data/all_data_ml.csv`, `data/all_data.csv` and the
  week's `games_to_predict` file in place, then chain a predict job on the active run.
- Build prediction inputs for a future week (`nfl_predictor/week_builder.py`, the `predict_week`
  job): extract the week's unplayed games from `data/all_data_ml.csv` with the ETL's own
  upcoming-game rule and predict them with the active run's model, so the week selector can
  offer any unplayed week of the current season.
- Add a `web` job to `.github/workflows/validation.yml` (Node 26: `npm ci`, lint, typecheck,
  vitest, build) and sync the Python job with the `web` extra.

## [0.7.1] - 2026-09-11

### Changed

- Bump the project version to `0.7.1` and keep `uv.lock` aligned.
- Load the quarterback family's history seasons (the play-by-play seasons before the ones an ETL
  run processes) from the per-season cache even under `--refresh-nflreadpy`. The flag now
  refreshes only the seasons being processed, as before the family existed; a one-season
  refresh no longer re-downloads every season back to 1999. A full-range run still refreshes
  every season it processes.
- Say in the missing-identity-file warning what actually happens: without
  `data/qb_meta_data.csv` quarterbacks are matched through the abbreviated passer-name
  fallback alone, so most starters still get features and only ambiguous or unknown names are
  null.

### Fixed

- Accept a training pool whose only seasons outside the rolling calibration window are the
  requested whole calibration seasons. The season-count guard predated the rolling window and
  still demanded a spare season for training, although the window seasons' remaining weeks
  always train; with `--train-calibration-seasons 1` on a three-season pool, weeks 2-4 raised
  `Not enough seasons` where week 5 succeeded. The guard now raises only when no season is left
  to train on. Pools without a window split exactly as before.
- Log the whole in-season calibration window. The `Calibration weeks` line printed only the
  newest season's weeks (`season 2026 weeks [1]` for a window that also held 2025 weeks
  16-18); it now appends the `[season, week]` pairs that the metadata records.

## [0.7.0] - 2026-09-11

### Changed

- Bump the project version to `0.7.0` and keep `uv.lock` aligned.
- Grow the invariant output schema from `498` to `519` columns with the quarterback family.

### Added

- Add quarterback per-dropback features for the expected starter (`constants.QB_PBP_STATS`,
  `nfl_predictor/utils/polars/qb_stats.py`), for `away_qb` and `home_qb` plus a `_diff` each:
  career EPA per dropback, CPOE (2006+), sack rate and ANY/A, each shrunk toward the league rate
  with `300` pseudo-dropbacks; EPA per dropback and ANY/A over the last `8` games, shrunk toward
  the career rate; and career dropbacks, so the model sees the sample size. Every value comes from
  that quarterback's regular-season dropbacks in earlier weeks, across teams and seasons, never
  the game's own week. Names map to play-by-play passer ids through `data/qb_meta_data.csv`, a
  read-only copy of the nfeloqb metadata, with three aliases and an abbreviated-name fallback;
  every 1999-2026 row matched. The family is the `qb` feature group
  (`--disable-feature-groups qb`), disjoint from `pbp` and `strength`. In the 2023-2025
  walk-forward (benchmark config, one build) it is a statistical tie with the group switched off:
  weeks 3-18 Brier `0.2327` against `0.2302` and log loss `0.7708` against `0.7577`, both inside
  their 95% bootstrap intervals, while weeks 1 and 2 lean better (week 2 pick accuracy `0.6458`
  against `0.5833`, 48 games). The weekly model trains on it by default.

## [0.6.3] - 2026-09-11

### Changed

- Bump the project version to `0.6.3` and keep `uv.lock` aligned.

### Added

- Label the total (over/under) columns of the betting report as diagnostic-only: every row of
  `build_betting_report` (the weekly run's `*_betting_report.csv` and `scripts/betting_pipeline.py`)
  carries `total_signal = diagnostic_only` next to `total_edge_points`, and the README says so for
  the report and the workbook. With the total head fixed, the 2023-2025 walk-forward still puts it
  behind the market's own total line: weeks 3-18 total MAE `10.3152` in the production
  configuration (no market anchoring; 95% bootstrap interval of the gap to the line
  `[+0.07, +0.39]`) and `10.2295` with anchoring, against `10.0847` for the line. Its deviation
  from the line is uncorrelated with the actual deviation (`-0.012`), so an over/under lean is not a
  betting signal. The fix in `0.6.2` restores the head's behaviour but not its walk-forward
  accuracy: the pre-fix head scores `10.3009` in the same configuration, a statistical tie.
  Spreads, moneylines and win probabilities are unaffected.

## [0.6.2] - 2026-09-11

### Changed

- Bump the project version to `0.6.2` and keep `uv.lock` aligned.

### Fixed

- Give every XGBoost fit its own early-stopping state, so the total (over/under) head learns
  again. On XGBoost versions whose `fit()` no longer takes `early_stopping_rounds` (every model in
  `models/` trained with early stopping, on `3.1.3` and `3.4.1`), the compatibility helper put one
  `xgb.callback.EarlyStopping` object into the params dict, and the margin and total heads were
  both built from that dict. The total head inherited the margin head's best score and spent
  patience and stopped after one round, so the unanchored production model predicted `43.9-44.2`
  for every 2026 Week 1 game and anchored runs predicted roughly the market line plus a constant.
  The helper now sets only the `early_stopping_rounds` init parameter, from which XGBoost builds a
  fresh callback for every fit. The margin head was always fit first, so margin predictions, win
  probabilities, Brier, log loss and pick accuracy do not change; total predictions, total MAE and
  the Optuna `combined_mae` objective do. Retrained on the same build and config, the Week 1 total
  head keeps `235` trees instead of one and its totals span `39.4-48.8` against market lines of
  `38.5-50.5`, while the margin head is identical. Models trained before this version keep the
  one-tree total head until they are retrained, and tuned parameters from earlier Optuna runs were
  chosen on the margin head alone.

## [0.6.1] - 2026-09-11

### Changed

- Bump the project version from `0.5.0` to `0.6.1` and keep `uv.lock` aligned. `0.6.0` names the
  power-rankings work below, which landed on `main` without a version bump.

### Added

- Record the whole in-season calibration window in the training metadata:
  `splits.calibration_inseason` keeps `season` and `weeks` (the newest season in the window and
  its weeks) and adds `pairs`, every `[season, week]` the window used.

### Fixed

- Roll the in-season calibration window back across the season boundary. Training took its
  calibration weeks only from the newest season and raised `Not enough weeks in season 2026 for
  calibration` when that season had fewer completed weeks than requested, so every weekly run for
  weeks 2-4 of a season stopped at the training stage (the weekly default asks for four). The
  window is now the newest completed `(season, week)` pairs in time order: for Week 2 of 2026,
  week 1 plus 2025 weeks 16-18. Training excludes exactly those pairs, whole-season calibration
  never takes a season the window touched, and the split raises only when the whole training
  pool has fewer weeks than requested. When the newest season already has enough weeks the split
  is unchanged; on the live dataset it matches the previous code in all 126 configurations that
  code accepted.

## [0.6.0] - 2026-09-11

### Changed

- Rank teams on the ETL's schedule-adjusted strength composite by default in
  `scripts/power_rankings.py` (`--method composite`). A ranking through week N reads the week
  N+1 snapshot, which is solved only from earlier games and has a row for every scheduled team,
  so teams on a bye are ranked exactly and no later result leaks into a historical rerun. The
  composite is converted to points with that week's SRS slope and to a win probability against
  an average team through the model's margin curve before it is placed on the 1-10 and 0-10
  scales; the composite, `points_vs_average` and the components are published next to the rank.
  `--method bradley_terry` reproduces the previous default exactly (pinned by a test), and
  `--legacy-franchise-fit` now implies it.
- Expose the ranking options in `scripts/weekly_run.py` (`--power-rankings-method`,
  `--power-rankings-strength-snapshots`, `--ratings-window-seasons`,
  `--ratings-prior-season-weight`, `--ratings-target`, `--ratings-include-future`,
  `--legacy-franchise-fit`). It now calls the same `compute_power_rankings` as the script,
  includes the options in its reports-stage reuse hash, and skips the rankings with a warning
  when the snapshot for the requested week is missing.
- Write the model-rating table from `scripts/golden_command.py` as `model_rating_rankings.csv`
  instead of `power_rankings.csv`, so the power rankings from `scripts/power_rankings.py` are the
  one canonical ranking artifact.

### Added

- Write `data/strength_snapshots.csv` from the ETL: pre-week adjusted strength for every team on
  each season's schedule and every processed week, teams on a bye included, plus the week after
  the regular season before the playoff schedule exists. Values equal the `away_`/`home_`
  strength columns on the game rows (one solve feeds both) and add the league-wide home-field
  term `adj_hfa`. Training rows are unchanged.

### Fixed

- Compare scores as numbers when building power-ranking records and projected standings. The ETL
  writes the newest games first, so once unplayed games led the file Polars inferred the score
  columns as text and compared them alphabetically: for 2024 through week 17, 26 of 32 records
  were wrong (DET showed 11-5 instead of 14-2). The Bradley-Terry ratings were unaffected.
- List every team in projected standings before its first game. Standings were built on the
  record rows, which do not exist before week 1, so the Week-1 projected standings came out empty;
  teams with no games yet now start from a zero record.

## [0.5.0] - 2026-09-10

### Changed

- Bump the project version to `0.5.0` for the early-season shrinkage and resumable walk-forward
  release and keep `uv.lock` aligned.
- Renumber the active worklist in `.agents/TODO.md` so milestones run in execution order (51-57);
  finished parts moved to `.agents/ARCHIVE.md`, which records the old-to-new map. Archived numbers
  are unchanged.
- Blend season-to-date team stats toward the regressed previous season in early weeks instead of
  switching from the full prior in week 1 to a single unshrunk game in week 2. Each team's per-game
  means are weighted `games / (games + 4)` in-season and the rest prior, and every rate is
  recomputed from the blended sums. Over 2023-2025 from week 1, week-2 Brier fell from `0.2434` to
  `0.2268` and pick accuracy rose from `0.5417` to `0.6042`, with weeks 3-18 unchanged within noise
  (`models/wf_shrink_2023_2025_{off,on}/`). On by default; `--no-stat-prior-blend` and
  `--stat-prior-blend-games` on `nfl_predictor.data_collection` ablate or tune it at ETL time.
  `PRIOR_BLEND_GAMES` moved to `constants.py` and is shared with the adjusted-strength blend.
- Rank teams on current-season strength by default in `scripts/power_rankings.py`. The
  Bradley-Terry fit now sees a two-season window with prior-season games weighted `0.25`
  (`--ratings-window-seasons`, `--ratings-prior-season-weight`), scores completed games by margin
  through the model's win-probability curve (`--ratings-target`), and excludes future
  model-probability rows from the strength fit (`--ratings-include-future`). The previous
  all-seasons equal-weight fit ranked a 4-13 team first for 2024; `--legacy-franchise-fit`
  reproduces it exactly and is pinned by a test. `scripts/weekly_run.py` inherits the new defaults
  without exposing the flags yet.

### Added

- Make walk-forward runs resumable. Every finished week is saved under
  `models/wf_checkpoints/<fingerprint>/` (data, config, modelling code, and library versions), and
  re-running an identical command restores those weeks and trains only the rest, with results
  identical to an uninterrupted run. Available as `--resume` / `--checkpoint-dir` on
  `scripts/walk_forward_backtest.py` and `scripts/wf_compare.py`, `--wf-resume` /
  `--wf-checkpoint-dir` on `scripts/golden_command.py`, and through the existing `--resume` of
  `scripts/weekly_run.py` and `scripts/betting_pipeline.py`, which now also resume partway through a
  candidate. Metrics reports record how many weeks were restored.
- Log one progress line per finished walk-forward week with elapsed time and an estimate of the time
  remaining; previously a run printed nothing between its first weeks and its final report.
- Document walk-forward operating practice in `README.md` and `AGENTS.md`: run one XGBoost-heavy job
  at a time, and choose the OpenMP wait policy by machine load. Under CPU contention the default
  policy made one week take `730s`; `OMP_WAIT_POLICY=PASSIVE` cut that to `185s`, but on an idle
  machine the default is faster (`75s` against about `142s`). The policy changes scheduling only,
  never results or fold checkpoints.
- Exclude the optional, gitignored `.agents/skills/` clone from ruff (`pyproject.toml`) and
  markdownlint (new `.markdownlintignore`).
- Add optional per-game sample weights and a margin-based target to
  `nfl_predictor.reporting.power_rankings.fit_bradley_terry_ratings`; uniform weights reproduce
  the unweighted fit exactly.

### Fixed

- Fix the six recommendation formulas in the betting workbook, which closed one more parenthesis
  than they opened, so Excel reported the file as corrupt and stripped every action cell. The
  generated workbook is now validated by tokenizing each formula.

## [0.4.0] - 2026-09-09

### Changed

- Bump the project version to `0.4.0` for the schedule-adjusted strength release and keep
  `uv.lock` aligned.
- Grow the invariant output schema from `465` to `498` columns with the schedule-adjusted team
  strength family.
- Emit `is_home` on the play-by-play team-game frame, derived from `posteam_type` on either
  perspective. It is a context flag rather than a statistic: it is excluded from
  `constants.PBP_COUNT_COLUMNS` and `constants.PBP_STATS`, added to
  `constants.EXCLUDE_FROM_OPPONENT_STATS` so no `opponent_is_home` inverse is generated, and
  dropped by season-to-date aggregation, so it never reaches the published schema.

### Added

- Add `nfl_predictor/utils/polars/adjusted_strength.py`: a NumPy simultaneous ridge
  (`solve_team_ridge`) estimating one offense and one defense coefficient per team plus a shared
  home-field term, centered independently per side, with `solve_srs` and an offline
  `tune_ridge_lambda`. The design is ported from the read-only `nfl-sos-ratings` reference and
  reproduces it numerically on identical inputs.
- Add `nfl_predictor/utils/polars/schedule_strength.py`: `sos_played_adj` / `sos_remaining_adj`
  from opponents' pre-week composite, and `sos_played_raw`, the one-hop companion that profiles
  each faced opponent from only its games against the rest of the league, excluding every
  head-to-head game with the subject.
- Add `nfl_predictor/utils/polars/strength_snapshot.py`: the pre-week snapshot builder, with a
  frozen ridge penalty, an early-season blend of the previous season's final snapshot regressed by
  `constants.WEEK1_REGRESSION_FACTOR`, and a standardized display composite using the
  `nfl-sos-ratings` published weights.
- Publish the strength family per team (`constants.ADJUSTED_STRENGTH_STATS`) as
  `away_`/`home_`/`_diff`, ablatable as the `strength` feature group.
- Add `--strength-prior-blend` / `--no-strength-prior-blend` to
  `nfl_predictor.data_collection` so the early-season prior can be ablated independently of the
  rest of the family.

### Fixed

- Restrict both schedule-strength lenses to the regular season. `sos_remaining_adj` previously
  averaged the whole remaining schedule including the postseason, so which playoff games a team
  would play -- an outcome of the season being predicted -- reached its regular-season features.
- Take the strength snapshot's team universe from the season schedule rather than from games
  already played, so a Week-1 row before any game has kicked off publishes the regressed prior
  instead of nulls. Historical Week-1 rows were unaffected; the live prediction slate was not.
- Filter non-finite solve responses alongside nulls. `NaN` is not null, so a single bad cell
  reached the normal equations and returned `NaN` for every team's coefficient rather than for the
  row that caused it.
- Add `is_home` to the null-fill used when no play-by-play is available at all, so the invariant
  schema claim holds for that column too.

## [0.3.0] - 2026-09-09

### Changed

- Bump the project version to `0.3.0` for the play-by-play feature release and keep `uv.lock`
  aligned.
- Publish `rushing_epa` alongside `passing_epa` and grow the invariant output schema from `384` to
  `465` columns.
- Recompute derived ratio metrics after the Week-1 regression rewrites their summed components, so
  the fallback no longer publishes regressed counts alongside unregressed ratios. This also changes
  the Week-1 values of `yards_per_point`, `points_per_play`, `penalty_yards_per_penalty`, and their
  opponent and margin variants.
- Exclude the gitignored `nfl-sos-ratings` reference symlink from Pyright so `.venv/bin/pyright .`
  checks this project only.

### Added

- Cache play-by-play per season as `data/cache/nflreadpy/pbp_<season>_<reg|all>.parquet` with the
  same current-season refresh and non-fatal degrade behavior already used for schedules and team
  stats.
- Add `nfl_predictor/utils/polars/pbp.py`, aggregating play-by-play into one row per
  `(season, week, team_abbr, opponent_abbr)` of counts and sums, including the situational counts
  previously produced by the unused `loaders.aggregate_pbp_stats`.
- Publish 25 play-by-play stats (`constants.PBP_STATS`): offensive and allowed EPA per snap, EPA per
  dropback and per carry, success rates, explosive pass and rush rates, stuffed-run rate, early-down
  pass rate, snap volume, and special-teams EPA margin per play. Every rate is a ratio of
  season-to-date sums, and defensive metrics are named explicitly rather than mirrored.
- Add `--disable-feature-groups` to `scripts/walk_forward_backtest.py` and `scripts/wf_compare.py`,
  resolved through `constants.FEATURE_GROUP_COLUMN_MARKERS`, with the group also honored inside
  `run_walk_forward_backtest` so the config alone determines the ablation.

### Removed

- Remove the unused, uncached `loaders.aggregate_pbp_stats` in favor of the new play-by-play module.

### Fixed

- Treat an unavailable current season as non-fatal in the play-by-play loader. nflreadpy raises
  `ValueError` rather than `ConnectionError` for a season it cannot serve, so a pre-kickoff ETL run
  aborted instead of degrading.
- Drop play-by-play rows whose possession team is an empty string. nflverse uses `""` rather than
  null in 1999 and 2000, which created phantom team-game rows, duplicated the
  `(season, week, team_abbr)` join key, and multiplied team-stat rows (1999: `495` to `526`;
  2000: `492` to `526`), understating snap volumes and games played for those seasons. The join now
  also collapses duplicate keys with a warning so it can never change the row count.
- Exclude two-point conversion tries from dropbacks and carries so they cannot contaminate the
  per-attempt EPA and success denominators.

## [0.2.6] - 2026-06-13

### Changed

- Bump the project version to `0.2.6` and package the completed Milestone 44 hardening baseline
  for a clean handoff to Milestone 39 ([`287cfa9`])

### Added

- Add focused metric, checkpoint, and prediction coverage plus the tested changelog/release
  workflow helpers, lifting the validated suite to `406 passed` and `90.01%` coverage
  ([`287cfa9`])

### Fixed

- Enforce the repo-wide `90%` pytest coverage floor in `pyproject.toml` and keep `uv.lock`
  aligned with version `0.2.6` ([`287cfa9`])

## [0.2.5] - 2026-06-13

### Changed

- Bump the project version to `0.2.5` and align the hardening docs around the canonical validation
  path plus the rule that `uv` remains an external `PATH` tool while Python tooling stays under
  `.venv/bin/...` ([`ac95abd`])
- Update the preseason hardening docs and active plan so release automation is no longer tracked as
  deferred work.
- Enforce the repo-wide `90%` coverage floor in `pyproject.toml`, complete Milestone 44, and move
  the active roadmap back to Milestone 39.

### Added

- Add `.github/workflows/validation.yml` to mirror the local validation gate in GitHub Actions,
  including Ruff format/check, Pyright, Ty, pytest, markdownlint, lockfile/environment checks,
  editable install, and primary CLI help smoke checks ([`ac95abd`])
- Add a deterministic regression test for `scripts/betting_pipeline.py --dry-run` when the default
  dataset path is absent in a clean checkout ([`8d6f49f`])
- Add `.github/workflows/release.yml` plus a tested changelog extractor so `0.x.y` and `v0.x.y` tags
  publish or refresh GitHub releases from the matching `CHANGELOG.md` entry.
- Add focused helper and prediction tests that lift the validated suite to `406 passed` and `90.01%`
  coverage.

### Fixed

- Refresh `uv.lock` so `uv lock --check` agrees with project version `0.2.5` and the new CI gate
  ([`ac95abd`])
- Allow `scripts/betting_pipeline.py --dry-run` to exit successfully without
  `data/completed_games_ml.csv`, keeping GitHub Actions green in repos that do not commit local
  `data/` artifacts ([`8d6f49f`])
- Remove dead facade-only `TYPE_CHECKING` placeholders so coverage reflects executable behavior
  instead of no-op lines.

## [0.2.4] - 2026-06-13

### Changed

- Raise the validated pytest baseline to `396 passed` / `90%` coverage, and confirm editable install
  plus primary CLI help smoke checks on Python 3.14 ([`05f2931`])

### Added

- Add focused coverage across sample weights, validation utilities, CLI orchestration, feature-spec
  helpers, feature-importance fallbacks, data-collection ETL control flow, `ml_model_core`
  helper/split branches, `ml_model_training` orchestration branches, leakage-audit helpers,
  game/scraping utility fallbacks, TeamRankings helper paths, validation-script entrypoints, and
  walk-forward helper branches, bringing several support modules to `100%` coverage and lifting
  `nfl_predictor/data_collection.py` to `89%`, `nfl_predictor/ml/ml_model_core.py` to `76%`,
  `nfl_predictor/ml/ml_model_training.py` to `88%`, `nfl_predictor/ml/leakage_audit.py` to `93%`,
  `nfl_predictor/utils/game_utils.py` to `94%`, `nfl_predictor/utils/scraping_utils.py` to `96%`,
  `nfl_predictor/utils/polars/teamrankings.py` to `90%`, and `nfl_predictor/ml/walk_forward.py` to
  `94%` ([`05f2931`])

### Fixed

- Propagate non-zero exit codes from `scripts/validate_offline.py` and `scripts/validate_live.py`
  via `SystemExit(main())`, and route validation output through the project logger for safer shell
  automation ([`05f2931`])

## [0.2.3] - 2026-06-12

### Changed

- Raise the validated pytest baseline to `326 passed` / `86%` coverage after the latest preseason
  coverage hardening slices ([`2351b2b`])

### Added

- Add focused coverage across sample weights, validation utilities, CLI orchestration, feature-spec
  helpers, feature-importance fallbacks, data-collection ETL control flow, and walk-forward helper
  branches, bringing several support modules to `100%` coverage and lifting
  `nfl_predictor/data_collection.py` to `89%` plus `nfl_predictor/ml/walk_forward.py` to `94%`
  ([`2351b2b`])

## [0.2.2] - 2026-06-12

### Changed

- Auto-select the newest weekly prediction input in `scripts/betting_pipeline.py` when
  `--predict-path` is omitted, and make the sample `config/weekly_run.yaml` rely on runtime
  inference instead of checked-in 2025 postseason defaults ([`47402e2`])
- Refresh README and preseason hardening guidance to reflect the current `248 passed` baseline and a
  coverage-first next-session priority ([`47402e2`])

### Added

- Add regression tests for unset `--predict-path` handling and newest-week file selection in
  `tests/test_betting_pipeline_script.py` ([`47402e2`])

### Fixed

- Fix stale week-specific CLI examples in `scripts/betting_pipeline.py` and
  `scripts/betting_report_excel.py` ([`47402e2`])

## [0.2.1] - 2026-06-12

### Changed

- Resolve all 444 pyright type errors: add `pandas-stubs` dev dependency and introduce
  `_fit_transform_matrix`/`_transform_matrix` helpers in `ml_model_xgb_utils.py` to narrow sklearn's
  broad `ColumnTransformer` return type; update 23 call sites across 6 files ([`586ffb6`])
- Tighten the repo's agent workflow around TDD-first execution and relevant skill usage in
  `AGENTS.md`, `TODO.md`, and `docs/next_agent_session_prompt.md` ([`586ffb6`])
- Promote `ty` to a mandatory peer gate alongside `pyright` and update repo guidance to reflect the
  passing dual-checker baseline ([`586ffb6`])

### Added

- Add regression tests for typed transform wrappers and the core margin/total prediction helper
  paths that now rely on them ([`586ffb6`])
- Add focused characterization tests for `_coerce_score_value`, `get_team_name`, and `_import_shap`
  before tightening their typing implementations ([`586ffb6`])

### Fixed

- Fix `schedule_df` type-narrowing after try/except in `validation_utils.py` ([`586ffb6`])
- Fix the remaining `ty` diagnostics in walk-forward metrics, optional SHAP loading, feature
  importance coercion, TeamRankings name resolution, Polars trend schema typing, and walk-forward
  test config reconstruction ([`586ffb6`])
- Fix `None` placeholder arguments in checkpoint and model-compare test fixtures using `cast()`
  ([`586ffb6`])

## [0.2.0] - 2026-06-12

_This release backfills the changelog from the historical `0.1.0` baseline on `main`._

### Changed

- Raise the runtime baseline to Python 3.14 and adopt `uv.lock` workflows ([`240f814`], [`78c732e`])
- Standardize model selection around walk-forward evaluation and resumable runs ([`4f33dd6`],
  [`83ba2a7`])
- Replace dynamic facades with explicit exports for better IDE support ([`b98fe0a`], [`6dc8910`])
- Expand repo guidance so changelog and planning files stay synchronized ([`8f7be46`], [`155088e`])
- Extend data refresh defaults for 1999+ history, caching, and incomplete seasons ([`12ae377`],
  [`41b2df0`])

### Added

- Add postseason-aware training, evaluation, and power rankings ([`db96be6`], [`a6d9efd`])
- Add `auto`/`logistic` calibration and uncertainty-aware win probabilities ([`6e6ba43`],
  [`7536001`])
- Add market probability source selection and logit/probability blending ([`e0f75ae`], [`77e9ede`])
- Add weekly orchestration, resumable checkpoints, and run configuration files ([`a363fd1`],
  [`33bcbd5`])
- Add Optuna summaries, feature importance, SHAP analysis, and richer reports ([`a870897`],
  [`f4a7d66`])
- Add trend, season-phase, recency, stadium, and head-coach features ([`febafda`], [`843170c`])
- Add `update_requirements.sh` and CI/changelog planning docs ([`bfb53d4`], [`445c9fd`])

### Removed

- Remove weather and referee features after confirming the data is not pre-kickoff safe
  ([`51f0e23`])
- Remove legacy requirements files and duplicate Copilot instructions ([`38ccc2a`], [`e21f408`])
- Remove obsolete versioning scaffolding and unused QB ID constants ([`689fd0e`], [`c2ef2b2`])

### Fixed

- Fix regular-season record loading and required-feature validation ([`e1397a6`])
- Fix minimum-season guards and cache behavior for data collection ([`baef49a`], [`b2570ad`])
- Fix CLI exit code propagation in the `ml_model` entrypoints ([`a41e68c`], [`b98fe0a`])
- Fix validation-script exits and stale weekly defaults ([`9705fd1`], [`67d13e2`])
- Fix Python 2-3 exception syntax in CLI compatibility paths ([`fce74d3`], [`d58b678`])
- Fix walk-forward checkpoint typing and summary validation ([`aa91404`], [`2d856b3`])
- Fix XGBoost logging and callback plumbing for cleaner diagnostics ([`ec51afa`], [`a0ccdf9`])

## [0.1.0] - 2026-01-16

_Historical baseline on `main` before changelog adoption. Add a matching git tag before automating
releases._

[0.2.6]: https://github.com/mitch-avis/nfl-predictor/compare/e9456f7...287cfa9
[0.2.5]: https://github.com/mitch-avis/nfl-predictor/compare/05f2931...ac95abd
[0.2.4]: https://github.com/mitch-avis/nfl-predictor/compare/3a9ca82...05f2931
[0.2.3]: https://github.com/mitch-avis/nfl-predictor/compare/47402e2...2351b2b
[0.2.2]: https://github.com/mitch-avis/nfl-predictor/compare/586ffb6...47402e2
[0.2.1]: https://github.com/mitch-avis/nfl-predictor/compare/9c04dc4...586ffb6
[0.2.0]: https://github.com/mitch-avis/nfl-predictor/compare/7a4d4c7...main
[0.1.0]: https://github.com/mitch-avis/nfl-predictor/commit/7a4d4c7
[`ac95abd`]: https://github.com/mitch-avis/nfl-predictor/commit/ac95abd
[`287cfa9`]: https://github.com/mitch-avis/nfl-predictor/commit/287cfa9
[`8d6f49f`]: https://github.com/mitch-avis/nfl-predictor/commit/8d6f49f
[`05f2931`]: https://github.com/mitch-avis/nfl-predictor/commit/05f2931
[`2351b2b`]: https://github.com/mitch-avis/nfl-predictor/commit/2351b2b
[`47402e2`]: https://github.com/mitch-avis/nfl-predictor/commit/47402e2
[`586ffb6`]: https://github.com/mitch-avis/nfl-predictor/commit/586ffb6
[`240f814`]: https://github.com/mitch-avis/nfl-predictor/commit/240f814
[`78c732e`]: https://github.com/mitch-avis/nfl-predictor/commit/78c732e
[`4f33dd6`]: https://github.com/mitch-avis/nfl-predictor/commit/4f33dd6
[`83ba2a7`]: https://github.com/mitch-avis/nfl-predictor/commit/83ba2a7
[`b98fe0a`]: https://github.com/mitch-avis/nfl-predictor/commit/b98fe0a
[`6dc8910`]: https://github.com/mitch-avis/nfl-predictor/commit/6dc8910
[`8f7be46`]: https://github.com/mitch-avis/nfl-predictor/commit/8f7be46
[`155088e`]: https://github.com/mitch-avis/nfl-predictor/commit/155088e
[`12ae377`]: https://github.com/mitch-avis/nfl-predictor/commit/12ae377
[`41b2df0`]: https://github.com/mitch-avis/nfl-predictor/commit/41b2df0
[`db96be6`]: https://github.com/mitch-avis/nfl-predictor/commit/db96be6
[`a6d9efd`]: https://github.com/mitch-avis/nfl-predictor/commit/a6d9efd
[`6e6ba43`]: https://github.com/mitch-avis/nfl-predictor/commit/6e6ba43
[`7536001`]: https://github.com/mitch-avis/nfl-predictor/commit/7536001
[`e0f75ae`]: https://github.com/mitch-avis/nfl-predictor/commit/e0f75ae
[`77e9ede`]: https://github.com/mitch-avis/nfl-predictor/commit/77e9ede
[`a363fd1`]: https://github.com/mitch-avis/nfl-predictor/commit/a363fd1
[`33bcbd5`]: https://github.com/mitch-avis/nfl-predictor/commit/33bcbd5
[`a870897`]: https://github.com/mitch-avis/nfl-predictor/commit/a870897
[`f4a7d66`]: https://github.com/mitch-avis/nfl-predictor/commit/f4a7d66
[`febafda`]: https://github.com/mitch-avis/nfl-predictor/commit/febafda
[`843170c`]: https://github.com/mitch-avis/nfl-predictor/commit/843170c
[`bfb53d4`]: https://github.com/mitch-avis/nfl-predictor/commit/bfb53d4
[`445c9fd`]: https://github.com/mitch-avis/nfl-predictor/commit/445c9fd
[`51f0e23`]: https://github.com/mitch-avis/nfl-predictor/commit/51f0e23
[`38ccc2a`]: https://github.com/mitch-avis/nfl-predictor/commit/38ccc2a
[`e21f408`]: https://github.com/mitch-avis/nfl-predictor/commit/e21f408
[`689fd0e`]: https://github.com/mitch-avis/nfl-predictor/commit/689fd0e
[`c2ef2b2`]: https://github.com/mitch-avis/nfl-predictor/commit/c2ef2b2
[`e1397a6`]: https://github.com/mitch-avis/nfl-predictor/commit/e1397a6
[`baef49a`]: https://github.com/mitch-avis/nfl-predictor/commit/baef49a
[`b2570ad`]: https://github.com/mitch-avis/nfl-predictor/commit/b2570ad
[`a41e68c`]: https://github.com/mitch-avis/nfl-predictor/commit/a41e68c
[`9705fd1`]: https://github.com/mitch-avis/nfl-predictor/commit/9705fd1
[`67d13e2`]: https://github.com/mitch-avis/nfl-predictor/commit/67d13e2
[`fce74d3`]: https://github.com/mitch-avis/nfl-predictor/commit/fce74d3
[`d58b678`]: https://github.com/mitch-avis/nfl-predictor/commit/d58b678
[`aa91404`]: https://github.com/mitch-avis/nfl-predictor/commit/aa91404
[`2d856b3`]: https://github.com/mitch-avis/nfl-predictor/commit/2d856b3
[`ec51afa`]: https://github.com/mitch-avis/nfl-predictor/commit/ec51afa
[`a0ccdf9`]: https://github.com/mitch-avis/nfl-predictor/commit/a0ccdf9
