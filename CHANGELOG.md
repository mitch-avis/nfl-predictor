# Changelog

## [0.2.4] - 2026-06-13

### Changed

- Raise the validated pytest baseline to `391 passed` / `90%` coverage, and confirm editable install
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

[0.2.4]: https://github.com/mitch-avis/nfl-predictor/compare/3a9ca82...05f2931
[0.2.3]: https://github.com/mitch-avis/nfl-predictor/compare/47402e2...2351b2b
[0.2.2]: https://github.com/mitch-avis/nfl-predictor/compare/586ffb6...47402e2
[0.2.1]: https://github.com/mitch-avis/nfl-predictor/compare/9c04dc4...586ffb6
[0.2.0]: https://github.com/mitch-avis/nfl-predictor/compare/7a4d4c7...main
[0.1.0]: https://github.com/mitch-avis/nfl-predictor/commit/7a4d4c7
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
