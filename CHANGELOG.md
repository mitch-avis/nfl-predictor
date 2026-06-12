# Changelog

## [0.2.0] - 2026-06-12

_This release backfills the changelog from the historical `0.1.0` baseline on `main`._

### Changed

- Raise the runtime baseline to Python 3.14 and adopt `uv.lock` workflows ([`240f814`],
  [`78c732e`])
- Standardize model selection around walk-forward evaluation and resumable runs ([`4f33dd6`],
  [`83ba2a7`])
- Replace dynamic facades with explicit exports for better IDE support ([`b98fe0a`],
  [`6dc8910`])
- Expand repo guidance so changelog and planning files stay synchronized ([`8f7be46`],
  [`155088e`])
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

[0.2.0]: https://github.com/mitch-avis/nfl-predictor/compare/7a4d4c7...main
[0.1.0]: https://github.com/mitch-avis/nfl-predictor/commit/7a4d4c7
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
