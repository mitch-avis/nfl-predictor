# Milestone 60 CLI Flag Audit

Date: 2026-09-22

> **Superseded on 2026-09-23. Do not use this file as a source.** A second-key review compared
> every section against the real parsers. The `scripts/golden_command.py` section lists 27 flags
> that do not exist and misses 20 that do; the `scripts/weekly_run.py` section lists a nonexistent
> `--report-save-input-copy` and misses `--betting-template-path` and `--ratings-min-season`;
> about 10 of the 239 flags are classified; and no consumer (web job templates, tests, CI,
> `models/*/launch.sh`) was checked. Task 60.1 in `TODO.md` replaces it with a script-generated
> inventory and lists the findings that were verified. The file stays only as a record.

## Scope

This audit covers every `add_argument(...)` surface in the entrypoints named by task 60.1:

| Entrypoint | `add_argument` count | Notes |
| --- | ---: | --- |
| `nfl_predictor/ml/ml_model_cli.py` | 45 | Primary training and prediction CLI |
| `scripts/weekly_run.py` | 67 | Largest surface; mixes run, walk-forward, training, and rankings |
| `scripts/walk_forward_backtest.py` | 31 | Canonical walk-forward backtest |
| `scripts/betting_pipeline.py` | 31 | Orchestrated compare/tune/train/predict flow |
| `scripts/golden_command.py` | 36 | Older orchestration surface |
| `nfl_predictor/data_collection.py` | 11 | ETL / feature publication entrypoint |
| `scripts/validate_offline.py` | 1 | Validation helper |
| `scripts/validate_live.py` | 1 | Validation helper |
| `scripts/power_rankings.py` | 16 | Rankings-only surface |

Total audited flags: `239`

The goal here is not to remove useful flexibility blindly. It is to identify the parts of the
surface that are inert, duplicated, aliased, or effectively config-managed so phase 2 can remove or
consolidate them with user sign-off.

## Method

- Read each targeted parser surface directly.
- Trace suspect flags into the consuming code to distinguish active behavior from
  config-or-fingerprint-only storage.
- Classify flags as `active`, `inert`, `tuning-only`, `duplicate`, `alias`, or
  `config-skewed`.
- Prefer verified code-path evidence over historical assumptions.

## Executive Summary

The surface has three clear problems:

1. Early-stopping flags survived the `0.12.3` change that removed in-season early stopping from
   walk-forward and final training, so some flags are now fingerprint or Optuna controls rather than
   fit controls.
2. The same concepts are declared multiple times with different names. Calibration, recency,
   market-probability blending, and XGBoost runtime knobs all drift by entrypoint.
3. `scripts/weekly_run.py` has become the union of several smaller CLIs plus its own config layer,
   so a meaningful part of its 67-flag surface is effectively config-managed rather than operated
   directly on the command line.

The highest-confidence removal candidates are the walk-forward early-stopping flag and the
`--market-prob-weight` alias. The highest-confidence consolidation targets are calibration naming,
recency naming, market-probability post-processing, XGBoost runtime overrides, and the shared
power-rankings options.

## Raw Surface by Entrypoint

This section records the full audited surface without repeating every help string.

### `nfl_predictor/ml/ml_model_cli.py` (45)

- Model and dataset selection: `--model-kind`, `--data-path`, `--holdout-seasons`,
  `--min-season`, `--max-season`
- Training inclusion and weighting: `--include-postseason`, `--postseason-weight`,
  `--recency-half-life-weeks`, `--recency-half-life-seasons`
- Market features and post-processing: `--exclude-market`, `--market-transform`,
  `--market-anchor`, `--market-prob-weight`, `--market-prob-blend`,
  `--market-prob-clamp`, `--market-prob-source`, `--market-prob-blend-method`
- Feature selection and preprocessing: `--feature-start`, `--feature-end`,
  `--max-cardinality-ratio`
- Calibration: `--calibration-seasons`, `--calibration-weeks`,
  `--win-prob-calibration`, `--win-prob-uncertainty`
- Tuning and XGBoost runtime: `--tune`, `--tune-timeout`, `--tune-trials`,
  `--tune-metric`, `--cv-splits`, `--early-stopping-rounds`, `--xgb-tree-method`,
  `--xgb-device`, `--xgb-n-jobs`, `--tune-scope`, `--tune-storage`,
  `--tune-study-name`, `--tune-best-params-out`
- Artifacts and prediction output: `--model-in`, `--model-out`, `--run-dir`, `--run-id`,
  `--predict-path`, `--output-path`, `--pretty-output`, `--score-rounding`

### `scripts/weekly_run.py` (67)

- Run/config shell: `--config`, `--data-path`, `--predict-path`, `--run-id`, `--run-dir`,
  `--output-dir`, `--resume`, `--dry-run`, `--skip-data-refresh`, `--data-collection-args`,
  `--score-rounding`
- Walk-forward window and inclusion: `--wf-eval-last-n-seasons`, `--wf-start-week`,
  `--wf-calibration-weeks`, `--wf-include-postseason`, `--wf-checkpoint-per-fold`,
  `--wf-exclude-incomplete-seasons`
- Walk-forward recency and market selection: `--wf-recency-half-life-weeks`,
  `--wf-recency-half-life-seasons`, `--wf-market-mode`, `--wf-market-prob-source`,
  `--wf-market-prob-blend-method`, `--wf-win-prob-uncertainty`,
  `--wf-include-quantiles`
- Walk-forward XGBoost overrides: `--wf-n-estimators`, `--wf-max-depth`,
  `--wf-learning-rate`, `--wf-n-jobs`, `--wf-early-stopping-rounds`
- Final-training surface: `--holdout-seasons`, `--train-calibration-seasons`,
  `--train-calibration-weeks`, `--include-postseason`, `--postseason-weight`,
  `--train-recency-half-life-weeks`, `--train-recency-half-life-seasons`,
  `--market-transform`, `--max-cardinality-ratio`, `--feature-start`, `--feature-end`,
  `--train-early-stopping-rounds`
- Tuning/runtime: `--tune`, `--tune-timeout`, `--tune-n-trials`, `--tune-cv-splits`,
  `--tune-objective`, `--tune-storage`, `--tune-study-name`, `--xgb-tree-method`,
  `--xgb-device`, `--xgb-n-jobs`
- Power-rankings/reporting surface: `--skip-power-rankings`, `--power-rankings-season`,
  `--power-rankings-through-week`, `--power-rankings-data-ml`,
  `--power-rankings-data-schedule`, `--power-rankings-out-dir`,
  `--power-rankings-include-postseason`, `--power-rankings-method`,
  `--power-rankings-strength-snapshots`, `--ratings-window-seasons`,
  `--ratings-prior-season-weight`, `--ratings-target`, `--ratings-include-future`,
  `--legacy-franchise-fit`, `--report-save-input-copy`

### `scripts/walk_forward_backtest.py` (31)

- Data and evaluation window: `--data-path`, `--eval-last-n-seasons`, `--eval-seasons`,
  `--wf-start-week`, `--wf-calibration-weeks`, `--random-seed`, `--include-postseason`,
  `--exclude-incomplete-seasons`
- Calibration and recency: `--calibration`, `--recency-half-life-weeks`,
  `--recency-half-life-seasons`
- Market and probability handling: `--market-anchor`, `--market-transform`,
  `--market-prob-weight`, `--market-prob-blend`, `--market-prob-clamp`,
  `--market-prob-source`, `--market-prob-blend-method`, `--win-prob-uncertainty`
- Feature ablation: `--disable-pruning`, `--disable-trend-features`,
  `--disable-feature-groups`
- XGBoost/runtime/checkpointing: `--xgb-tree-method`, `--xgb-device`, `--xgb-n-jobs`,
  `--min-child-weight`, `--gamma`, `--n-estimators`, `--resume`, `--checkpoint-dir`,
  `--out-json`

### `scripts/betting_pipeline.py` (31)

- Run and artifact shell: `--data-path`, `--predict-path`, `--run-id`, `--run-dir`,
  `--output-predictions`, `--resume`, `--dry-run`
- Walk-forward compare stage: `--wf-eval-last-n-seasons`, `--wf-start-week`,
  `--wf-calibration-weeks`, `--market-prob-source`, `--market-prob-blend-method`,
  `--wf-n-estimators`, `--wf-max-depth`, `--wf-learning-rate`, `--wf-xgb-n-jobs`
- Shared runtime: `--xgb-tree-method`, `--xgb-device`, `--xgb-n-jobs`
- Tuning stage: `--tune-timeout`, `--tune-metric`, `--cv-splits`, `--tune-storage`,
  `--tune-study-name`, `--tune-best-params-out`
- Training and prediction: `--include-postseason`, `--postseason-weight`,
  `--final-win-prob-calibration`, `--score-rounding`, `--market-transform`,
  `--market-anchor`

### `scripts/golden_command.py` (36)

- Run shell: `--data-path`, `--predict-path`, `--run-id`, `--run-dir`, `--resume`,
  `--dry-run`, `--output-predictions`
- Walk-forward controls: `--eval-last-n-seasons`, `--wf-start-week`,
  `--calibration-weeks`, `--wf-resume`, `--wf-checkpoint-dir`
- Training and market controls: `--holdout-seasons`, `--calibration-seasons`,
  `--include-postseason`, `--postseason-weight`, `--market-transform`,
  `--market-anchor`, `--market-prob-weight`, `--market-prob-clamp`,
  `--market-prob-source`, `--market-prob-blend-method`
- Tuning/runtime/output: `--tune`, `--tune-timeout`, `--tune-trials`, `--tune-metric`,
  `--cv-splits`, `--train-early-stopping-rounds`, `--xgb-tree-method`, `--xgb-device`,
  `--xgb-n-jobs`, `--score-rounding`, `--write-power-rankings`,
  `--power-rankings-season`, `--power-rankings-through-week`,
  `--power-rankings-include-postseason`, `--power-rankings-method`,
  `--power-rankings-strength-snapshots`, `--ratings-window-seasons`,
  `--ratings-prior-season-weight`, `--ratings-target`, `--ratings-include-future`,
  `--legacy-franchise-fit`

### `nfl_predictor/data_collection.py` (11)

- Season/output/logging shell: `--min-season`, `--max-season`, `--data-dir`, `--timing`,
  `--debug-logs`, `--refresh-nflreadpy`
- ETL behavior and sources: `--strength-prior-blend`, `--stat-prior-blend`,
  `--stat-prior-blend-games`, `--team-stats-source`, `--tr-stats-source`

### Validation and rankings-only entrypoints

- `scripts/validate_offline.py`: `--data-dir`
- `scripts/validate_live.py`: `--data-dir`
- `scripts/power_rankings.py` (16): model input, season/week selection, input/output paths,
  postseason inclusion, rankings method, strength-snapshot path, ratings window, prior-season
  weight, target, include-future flag, and legacy fit switch

## Verified Findings

### 1. Inert or tuning-only flags

| Flag | Entrypoint | Classification | Evidence |
| --- | --- | --- | --- |
| `--wf-early-stopping-rounds` | `scripts/weekly_run.py` | Inert | The parser help already says it is recorded only in config. `WalkForwardConfig` stores `early_stopping_rounds`, but `run_walk_forward_backtest` sets `in_season_early_stopping: false` and passes `early_stopping_rounds=None` into every in-season fit and quantile fit. |
| `--train-early-stopping-rounds` | `scripts/weekly_run.py` | Tuning-only | The parser help now states it applies to Optuna tuning trials only. Final production fits pass `None` for early stopping inside `train_margin_total_model_with_report`. |
| `--early-stopping-rounds` | `nfl_predictor/ml/ml_model_cli.py` | Tuning-only | The flag feeds `OptunaConfig.early_stopping_rounds`. Final non-Optuna fits run the full tree budget; the final training path passes `None` to `_fit_margin_total_models`. |
| `early_stopping_rounds` field | `WalkForwardConfig` | Fingerprint-only for walk-forward fits | Not a CLI flag on its own, but it is still part of the stored config and fingerprint even though walk-forward fits ignore it. This is the follow-up already noted in `.agents/TODO.md`. |

Implication: the early-stopping surface is no longer a model-fit surface. It is now a tuning-only or
fingerprint-only surface and should be renamed or removed to match reality.

### 2. Duplicate concepts with inconsistent names

| Concept | Current names | Why it is a problem | Best candidate for canonical name |
| --- | --- | --- | --- |
| Win-prob calibration | `--calibration`, `--win-prob-calibration`, `--final-win-prob-calibration` | Same domain concept, three names, one of them stage-specific | `--win-prob-calibration` |
| Recency half-life | Unprefixed in `ml_model_cli.py` and `walk_forward_backtest.py`; `--wf-*` and `--train-*` in `weekly_run.py` | Same setting family, three namespace styles | Unprefixed in single-purpose CLIs, prefixed in multi-stage orchestration |
| Market blend weight | `--market-prob-weight` and `--market-prob-blend` | Alias pair requires fallback code and invites conflicting mental models | `--market-prob-blend` |
| XGBoost runtime overrides | `--xgb-*`, `--wf-xgb-n-jobs`, `--wf-n-jobs`, `--wf-n-estimators` | Similar knobs split between stage-specific and script-specific naming | Shared helpers with explicit stage prefix only when a script exposes multiple stages |
| Power-rankings options | Duplicated between `scripts/weekly_run.py` and `scripts/power_rankings.py` | Same surface declared twice with the risk of future drift | Shared argument group |

### 3. Aliases and drift that should be simplified

| Flag family | Current behavior | Audit read |
| --- | --- | --- |
| `--market-prob-weight` / `--market-prob-blend` | `walk_forward_backtest.py` and `ml_model_cli.py` both accept the alias pair and manually prefer `weight` when present, then fall back to `blend` | Keep one name, deprecate the alias |
| `--wf-market-mode` vs no equivalent in smaller CLIs | Only `weekly_run.py` exposes market-mode selection directly because it drives a matrix sweep | Keep it orchestration-only, but do not mirror it elsewhere |
| `--data-collection-args` | `weekly_run.py` forwards one shell-quoted string into `data_collection.main()` | Valid for orchestration, but it is a passthrough escape hatch rather than a clean first-class interface |

### 4. Config-skewed surface

These flags are active, but in practice they are more config-managed or inferred than operator-run.
That does not make them removal candidates by itself. It does make them good candidates for shared
config handling and for a smaller default CLI story.

| Entrypoint | Flags or families | Why this is config-skewed |
| --- | --- | --- |
| `scripts/weekly_run.py` | `--wf-market-mode`, `--wf-win-prob-uncertainty`, `--wf-include-quantiles` | Stage-1 comparison settings are usually chosen once per workflow and then carried by config |
| `scripts/weekly_run.py` | `--power-rankings-season`, `--power-rankings-through-week`, path overrides, ratings window knobs | The normal flow infers these values from the prediction file or uses shipped defaults |
| `scripts/weekly_run.py` | `--data-collection-args` | Indirectly manages `data_collection.py` rather than exposing a typed subset |
| `nfl_predictor/data_collection.py` | `--stat-prior-blend-games`, source-selection flags | Often reached through `weekly_run.py` rather than typed directly by an operator |
| `scripts/betting_pipeline.py` | `--tune-study-name` | Realistically stable per workflow; rarely an interactive CLI choice |

### 5. Defaults that deserve a decision rather than silent drift

| Setting | Current defaults | Audit read |
| --- | --- | --- |
| `postseason_weight` | `1.0` in `ml_model_cli.py`, `1.0` fallback in `weekly_run.py`, `1.15` in `betting_pipeline.py`, `1.3` in `config/weekly_run.yaml` | This is not one coherent default story. The differing values may be intentional, but the repo should decide whether they are policy differences or drift. |
| Score rounding | `none` in `ml_model_cli.py` and `weekly_run.py`, `nfl` in `betting_pipeline.py` | This is active and used, so it is not an inert-flag issue. It is still a default-policy difference worth documenting. |
| Recency season weighting | `weekly_run.yaml` currently sets `train_recency_half_life_seasons: 4` with no matching walk-forward key | Already in task 55.8; included here because it is also CLI/config surface debt. |

## Proposed Removal List

These are the highest-confidence removals or deprecations from the audited surface.

1. Remove `--wf-early-stopping-rounds` from `scripts/weekly_run.py` after a deprecation note.
   It no longer changes any walk-forward fit behavior.
2. Rename or remove `--train-early-stopping-rounds` in `scripts/weekly_run.py` and
   `--early-stopping-rounds` in `ml_model_cli.py` unless the project explicitly wants a
   tuning-only early-stopping control. If kept, the name should say `tune`, not `train`.
3. Deprecate `--market-prob-weight` in favor of `--market-prob-blend` anywhere both exist.
4. Make `--calibration` in `scripts/walk_forward_backtest.py` an explicit compatibility alias for
   `--win-prob-calibration`, then phase the shorter name out.

These removals should wait for user sign-off, per Milestone 60's acceptance rule.

## Shared Options Module Proposal

Create a shared module, for example `nfl_predictor/cli_args.py`, with reusable argument-group
builders. The aim is to keep one declaration per surviving concept and let entrypoints compose only
what they need.

Recommended groups:

1. `add_data_path_args(parser)`
   Includes dataset paths, predict paths, run id/dir, output path, and resume or dry-run knobs.
2. `add_postseason_and_recency_args(parser, prefix=None)`
   Includes postseason inclusion, postseason weight, and recency half-life flags.
3. `add_market_probability_args(parser, allow_alias=False)`
   Includes blend, clamp, source, and blend method. If backward compatibility is needed, keep the
   old alias behind one code path instead of redeclaring it per CLI.
4. `add_calibration_args(parser, option_name="--win-prob-calibration")`
   Keeps the same choices and normalization everywhere.
5. `add_xgb_runtime_args(parser, prefix=None)`
   Includes tree method, device, thread count, and optionally tree-budget overrides.
6. `add_power_rankings_args(parser)`
   Shared between `scripts/weekly_run.py` and `scripts/power_rankings.py`.
7. `add_data_collection_source_args(parser)`
   Includes ETL source-selection and prior-blend settings for the ETL and any orchestration layer
   that wants typed forwarding.

This would not eliminate every script-specific option, but it would pull the high-drift families
out of nine separate parsers.

## Recommended Follow-up Order

1. Get user sign-off on the removal list and the canonical names.
2. Land a shared options module for calibration, market post-processing, recency, XGBoost runtime,
   and power-rankings options.
3. Remove or rename the early-stopping flags so the surface reflects the current fit behavior.
4. Replace the raw `--data-collection-args` passthrough with a typed forwarding strategy only if
   the added maintenance cost is worth it.

## Open Questions for the User

1. Should the surviving canonical calibration flag be `--win-prob-calibration` everywhere?
2. Should tuning-only early-stopping remain exposed, and if so should it be renamed to say
   `tune` explicitly?
3. Should `postseason_weight` be unified across entrypoints, or are the current differences part
   of intentionally different workflows?
4. Is `--data-collection-args` an acceptable long-term escape hatch, or do you want the weekly
   runner to expose a typed subset of ETL flags instead?
5. Should the power-rankings options remain inline in `scripts/weekly_run.py`, or should that
   stage accept a smaller shared argument group and lean on inference or config for the rest?

## Status

This file satisfies task 60.1's read-only inventory requirement.
No removals are proposed as landed work yet.
The next Milestone 60 step is user sign-off on the removal list and canonical naming.
