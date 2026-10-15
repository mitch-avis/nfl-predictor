# Decisions in leaving feature groups out of production (roadmap step 6, adoption mechanism)

Every judgment call made while letting production drop feature groups exactly as
`nfl-predictor backtest --disable-feature-groups` does, so that a walk-forward arm that disables
groups (the `strength`, `elo`, `own_elo` and `own_elo_wt` arms of the main checkout's
`models/step6/HYPOTHESIS.md`) can be adopted without a gap between the benchmark and production
(`AGENTS.md` rule 11). The setting this adds, `disable_feature_groups` in `config/weekly_run.yaml`,
is empty, so no weekly run changes; its value is set after the measurement, with the user's
approval.

Commands run from this repository's root with the worktree's environment. "The group fixture" is
`tests/feature_group_fixture.py`: the weekly test fixture with hand-written columns of the four
groups added inside the feature range.

## 1. Where the one definition of a group's columns lives

- **What differs**: the backtest turned group names into columns with
  `resolve_feature_group_columns` in `src/nfl_predictor/ml/walk_forward.py`, which the training
  module cannot import. Training had no group handling at all.
- **Options**: (a) move the function unchanged into `src/nfl_predictor/ml/feature_spec.py`, which
  every fit already imports, and import it back into `walk_forward`; (b) copy the logic into
  training; (c) a new module under `ml/` for it; (d) make leaving groups out part of
  `build_feature_spec` (a `FeatureSelection` field), so every feature spec applies it.
- **Decision**: (a).
- **Reasoning**: one definition that every fit can reach, with no import cycle. `walk_forward`
  keeps the name, so every existing caller keeps working, and the existing tests of the function
  served unedited as the characterization pins of the move. (b) is the duplication the task rules
  out; (c) adds a file for one function where feature selection already lives; (d) changes how
  every walk-forward fold builds its feature spec (the columns would move into
  `FeatureSpec.dropped_columns` and the drop point would move), a larger change to the benchmark
  instrument just before the feature-group arms run.
- **Data**: with `walk_forward` added to the training module's imports,
  `.venv/bin/python -c "import nfl_predictor.ml.ml_model_training"` raises
  `ImportError: cannot import name 'train_margin_total_model_with_report' from partially
  initialized module 'nfl_predictor.ml.ml_model_training' (most likely due to a circular import)`:
  `walk_forward.py` line 32 imports `ml_model`, and `ml_model.py` line 39 imports
  `ml_model_training`.
  `grep -rnE --include=*.py "walk_forward(\.| import )resolve_feature_group_columns" tests .agents`
  prints 18 references in 8 files, one of them `.agents/market_free/market_columns.py`. With the
  resolver broken on purpose (a prefix match instead of a substring match), 14 tests in those
  test files fail; after the move every one passes unedited (commit `0e26b3e`).
- **How to reverse it**: move the function back into `walk_forward.py`; training then needs
  another route to it.
- **Review**: not yet reviewed.

## 2. Where the final fit drops the columns

- **What differs**: a walk-forward drops a disabled group's columns from the whole dataset first,
  before it filters to the regular season, finds the targets and builds its folds
  (`walk_forward._prepare_run`). The final fit had no drop.
- **Options**: (a) drop in `ml_model_training._load_training_games`, right after the games load
  and before anything reads the columns, the walk-forward's point; (b) drop inside
  `build_feature_spec`; (c) drop in each command before it calls the training function.
- **Decision**: (a), through `TrainingOptions.disabled_feature_groups`.
- **Reasoning**: the same function at the same point in both paths, so a group removes the same
  columns from a fold and from the final fit by construction. Everything in a training run reads
  the loaded frame (the Optuna trials, the holdout scoring, the missing-data summary, the SHAP
  rows), so all of it follows, and `nfl-predictor train` and the weekly run share the one code
  path. (b) is option (d) of decision 1; (c) would leave a direct caller of the training function
  without the drop.
- **Data**: `tests/test_feature_group_parity.py` runs the backtest and then the weekly run or
  `nfl-predictor train` with the same groups on the group fixture, for `elo,own_elo_wt` and for
  `strength,elo,own_elo`. Each drops exactly the fixture's hand-written columns of those groups,
  and the final fit's, stage 1's first week's and train's feature lists equal the backtest's.
  With the drop disabled on purpose, 4 of its 5 tests fail. With no group disabled,
  `tests/test_final_fit_characterization.py`, whose snapshots were written on the code before
  this change (commit `2e1eed3`), passes unchanged: the same feature lists, predictions and
  stage-1 summary row from `nfl-predictor train` and the weekly run.
- **How to reverse it**: remove the drop from `_load_training_games` and the field from
  `TrainingOptions`.
- **Review**: not yet reviewed.

## 3. The setting's form in the config file and on the command line

- **What differs**: the backtest takes `--disable-feature-groups a,b`, one comma-separated
  string. The weekly config's only other list setting, `floor_sigma_reference_runs`, is a YAML
  list, and the config loader refuses a string there.
- **Options**: (a) a YAML list in the file, the backtest's comma-separated string on the command
  line, and a string in the file refused; (b) a comma-separated string in both; (c) a list in
  both (`nargs`).
- **Decision**: (a).
- **Reasoning**: an arm's option can be copied to `nfl-predictor train` or `nfl-predictor weekly`
  as it is, and the file reads like its other list setting. Refusing a string keeps one way to
  write the file, as `floor_sigma_reference_runs` does; a key with no value (YAML null) reads as no
  group. On the command line an empty value clears the file's list.
- **Data**: `_normalize_config_defaults` in `src/nfl_predictor/weekly_run/config.py` refuses a
  string for both keys ("must be a list of paths", "must be a list of feature group names").
  `tests/test_weekly_run_feature_groups.py` covers the list, the comma-separated option, the
  override (including the empty value) and the refused string.
- **How to reverse it**: accept a string in `_normalize_config_defaults` by splitting it with
  `cli.options.parse_feature_groups`.
- **Review**: not yet reviewed.

## 4. When an unknown group fails

- **What differs**: the backtest raises a `ValueError` naming the unknown group when it resolves
  the groups, after the dataset loads. The weekly run refreshes the data (an ETL run) and starts
  stage 1 before anything would resolve.
- **Options**: (a) `train` and the weekly run check the names when they parse their options (a
  usage error with exit status 2, before any data loads and before the weekly data refresh), the
  backtest unchanged; (b) the same parse-time check in the backtest too; (c) every command fails
  only when it resolves.
- **Decision**: (a).
- **Reasoning**: a typo in the weekly config should not cost a data refresh before it fails. The
  backtest already fails loudly, and a parse-time check there would change what its option
  parses to, which an existing test pins with a placeholder group. One check serves every
  command: `feature_spec.validate_feature_groups`, which the resolver also calls, so the message
  is the same everywhere.
- **Data**: `tests/test_cli_backtest.py` line 238 parses `--disable-feature-groups pbp,other` and
  expects the raw string. `tests/test_weekly_run_feature_groups.py`,
  `test_an_unknown_group_stops_the_run_before_the_data_refresh`, gets exit status 2, the message
  `Unknown feature group(s): ['not_a_group']` and no refresh call;
  `tests/test_cli_train_feature_groups.py`,
  `test_an_unknown_group_stops_the_command_and_names_the_group`, gets the same for `train`.
- **How to reverse it**: drop the two parse-time checks; the resolver still raises when a run
  reaches it.
- **Review**: not yet reviewed.

## 5. The weekly stages' resume hashes

- **What differs**: a weekly run reuses a finished stage when the hash of that stage's settings
  matches its marker. A new setting in the hashed settings changes the hash of every run, so a
  run directory written before this change would retrain both stages on resume even with no
  group left out.
- **Options**: (a) leave an empty list out of both stage hashes and hash a non-empty one; (b)
  always hash it.
- **Decision**: (a), in `pipeline._marker_settings`.
- **Reasoning**: leaving no group out is exactly how every run trained before the setting
  existed, so the default keeps its resume behavior as well as its outputs, and a non-empty list
  changes both hashes, so no stage trained on other columns is reused. The records stay explicit:
  the final fit's `train_config` (in the model's `metadata.json`) always carries the list, and
  stage 1's result file always records its configuration's groups.
- **Data**: `tests/test_weekly_run_settings.py` pins the shipped configuration's hashes
  `ca6a81c8` (stage 1) and `082f6915` (final fit), unchanged by this work, and
  `test_disabled_feature_groups_reach_both_stages_and_change_both_hashes` shows that a non-empty
  setting changes both.
- **How to reverse it**: hash the settings directly instead of through `_marker_settings`;
  existing run directories then retrain once when resumed.
- **Review**: not yet reviewed.

## 6. What a saved model records

- **What differs**: the backtest records `disabled_feature_groups` and
  `dropped_feature_group_columns` in its config. A trained model's `metadata.json` had its feature
  list but nothing about groups.
- **Options**: (a) top-level `disabled_feature_groups` and `dropped_feature_group_columns` in every
  trained model's `metadata.json` (empty lists when none), with the command's own option in its
  config; (b) also a field on the pickled model; (c) the command's option only.
- **Decision**: (a).
- **Reasoning**: the model's feature spec already holds the columns it trained on and prediction
  reindexes to them, so a saved model predicts with no group setting (no hidden state). The
  metadata says what it was trained without, under the backtest's key names, so an arm's list and
  a model's list compare directly. (b) changes the pickled class for no behavior; (c) leaves the
  removed columns unrecorded.
- **Data**: `tests/test_training_feature_groups.py`,
  `test_a_saved_model_predicts_without_reading_the_dropped_columns`, sets the left-out group's
  columns to -999 in the week to predict and gets identical predictions from the reloaded model.
- **How to reverse it**: remove the two fields from `artifacts.TrainedModelDetails` and
  `TrainingResult`.
- **Review**: not yet reviewed.

## 7. How the settings-versus-production section compares groups

- **What differs**: the section took the final fit's groups to be always empty, and it compared
  groups as ordered lists, so `elo,own_elo_wt` against `[own_elo_wt, elo]` would read as a
  difference.
- **Options**: (a) read the final fit's groups from production's `train_config` and compare both
  halves as sorted sets; (b) read them but compare in order.
- **Decision**: (a).
- **Reasoning**: the same groups in any order, or named twice, remove the same columns, so an
  ordering must not show as a gap between the benchmark and production.
- **Data**: `tests/test_production_settings.py`,
  `test_a_run_leaving_out_production_groups_matches_both_halves` (`elo,own_elo_wt` and
  `own_elo_wt,elo,elo` against `[elo, own_elo_wt]`: no difference in either half), and, with a
  real backtest, `tests/test_feature_group_parity.py`,
  `test_a_backtest_and_production_with_the_same_groups_show_no_group_difference`.
- **How to reverse it**: replace `_feature_groups` in `reporting/production_settings.py` with
  `list`.
- **Review**: not yet reviewed.

## 8. Stage 1's candidate key

- **What differs**: stage 1 names its result file and summary row with a key built from the
  model kind, the feature range, the market mode, the quantile setting and the XGBoost overrides
  (`ml/wf_compare_utils.build_candidate_key`). The groups are not in it.
- **Options**: (a) leave the key as it is (the groups are in the result file's configuration and
  in stage 1's run fingerprint); (b) add the groups to the key.
- **Decision**: (a).
- **Reasoning**: each weekly run has its own directory, so keys never collide, and (b) would
  change the default key in the summary row that the characterization snapshots pin, for a label.
- **Data**: `tests/test_feature_group_parity.py` reads the groups from
  `wf_compare/wf_candidate_*.json`; `tests/test_weekly_run_feature_groups.py`,
  `test_stage1s_run_fingerprint_covers_the_groups`, shows the fingerprint changes with them.
- **How to reverse it**: add the groups to `CandidateSpec` and to the key's payload.
- **Review**: not yet reviewed.

## 9. `nfl-predictor predict` and the option

- **What differs**: `predict` runs the same module with the same options as `train`, so it
  accepts `--disable-feature-groups` too.
- **Options**: (a) accept it and log that a saved model ignores it; (b) refuse it with
  `--model-in`.
- **Decision**: (a).
- **Reasoning**: this is how `predict` treats `--tune` (logged and ignored), and the model
  predicts with its own columns either way.
- **Data**: `tests/test_cli_train_feature_groups.py`,
  `test_predicting_with_a_saved_model_says_the_option_does_not_apply`.
- **How to reverse it**: call `parser.error` in `cli/train.py` when both options are given.
- **Review**: not yet reviewed.

## 10. Where the warning for a group that matches no column comes from

- **What differs**: decided by the coordinator after review: a known group that matches no column
  must log a warning from `feature_spec.resolve_feature_group_columns`, so every caller warns. The
  backtest resolves twice, once in `_apply_ablations`, which drops the columns, and again in the
  walk-forward's `_prepare_run` on the frame already without them, where every group matches
  nothing.
- **Options**: (a) the resolver warns by default; the walk-forward's own drop, a safety net that
  may find the columns already gone, resolves with `warn_unmatched=False`; stage 1 of the weekly
  run drops the groups before its walk-forward, as the backtest does, so each entry point warns
  once against the data it loaded; (b) a sixth parameter on `run_walk_forward_backtest` telling it
  the caller already dropped; (c) the warning in the resolver with no opt-out.
- **Decision**: (a).
- **Reasoning**: (c) logs a false warning naming every group on every backtest that disables any.
  (b) breaks the lint rule on parameter count (PLR0913, 6 > 5), and the repo does not suppress
  rules inline. Under (a) the backtest, `train`, stage 1 and the final fit each warn once; only a
  direct caller of `run_walk_forward_backtest` gets no warning from the safety net.
- **Data**: `tests/test_feature_group_columns.py`:
  `test_a_backtest_whose_groups_match_columns_logs_no_warning` failed with the false warning before
  the safety net went silent; `test_stage1_warns_about_a_group_the_data_lacks` fails when stage 1's
  own drop is removed; `tests/test_training_feature_groups.py`,
  `test_a_group_the_data_lacks_drops_nothing_and_warns`, covers the final fit.
- **How to reverse it**: drop the keyword and stage 1's drop; then move the backtest's group drop
  into the walk-forward so it resolves once.
- **Review**: not yet reviewed.
