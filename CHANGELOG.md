# Changelog

## [Unreleased]

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
