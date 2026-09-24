# CLI and scripts consolidation: proposed dispositions for sign-off

Written 2026-09-24 on `feat/m60-cli-consolidation` for task 60.3. Every fact below comes from
the generated inventory beside this file and can be reproduced:

```bash
.venv/bin/python .agents/m60/inventory.py         # INVENTORY.md + inventory.json, exits 0
.venv/bin/python .agents/m60/scripts_coverage.py  # SCRIPTS_COVERAGE.md (about 2 minutes)
```

`inventory.py` builds each parser by intercepting `ArgumentParser.parse_args` inside the
entrypoint's own `_build_parser`/`_parse_args` and reads the parser's actions. Read sites and
their sinks come from the AST. The web job commands come from calling every catalog template's
build function, and those commands are then parsed by the target's real parser. Hand judgments
live in `annotations.yaml`, and the script re-checks each one's evidence on every run. The
headline count was cross-checked against every entrypoint's real `--help` output in a
subprocess: the long options match for all 19 entrypoints. The only differences are regex
artifacts: options split across wrapped help lines, and `weekly_run`'s help quoting two
`data_collection` options in an example. `.agents/m60_cli_flag_audit.md` was not used.

Nothing here has been moved. Everything below is a proposal until you sign off.

## What the inventory found

- 19 entrypoints, 309 parser actions (347 long option strings), 14 files under `scripts/`.
- Mechanical classes: 288 flags are read at least once for behavior; 21 reach only the tuning
  fields of `OptunaConfig`. 24 flags carry a hand judgment, all with evidence holding.
- Of the findings recorded under 60.1 and 60.2, 11 are confirmed. One count is overturned:
  eleven test modules import from `scripts`, not ten (listed in INVENTORY.md). The
  market-weight alias is two separate argparse actions (dests `market_prob_weight`, default
  `None`, and `market_prob_blend`, default `0.0`), reconciled in code, not one action with two
  spellings. `betting_pipeline`'s copy of the
  stage-1 matrix has drifted: 7 rows against `weekly_run`'s 9 (no `none_base`, no `auto_base`).
- New findings, not in the recorded list:
  - **Web model-kind vocabulary (a behavior defect).** `ml_model` and the training metadata say
    `blend`; `power_rankings` and the web catalog's `MODEL_KINDS` say `blended_margin_total`.
    Three web launches fail in argparse: `train` with `blended_margin_total`, `predict` for a
    run recorded as `blended_margin_total`, and `power_rankings` for a run recorded as `blend`.
  - `wf_compare --early-stopping-rounds` is inert: it reaches the same `to_dict`-only
    `WalkForwardConfig` field as `--wf-early-stopping-rounds`.
  - `weekly_run --wf-n-jobs` never decides anything in production: `--xgb-n-jobs` overrides it
    whenever it is set, and the config sets both to 12.
  - `leakage_audit --include-market` is a no-op (`store_true`, default true, sharing its dest
    with `--exclude-market`).
  - `walk_forward_backtest --calibration` defaults to `platt`, while all 19 benchmark launchers
    pass `--calibration auto` (the recorded instrument), so a bare run does not reproduce the
    benchmark.
  - Eleven top-level helpers are defined in more than one place. Six are shared between
    `betting_pipeline` and `weekly_run`, four of them with identical bodies. Three exist only
    among scripts slated to retire (`_resolve_team_columns` also has a diverged copy in
    `nfl_predictor/ml/walk_forward.py`). Two, `_market_modes` and `_parse_feature_groups`, are
    identical copies between scripts slated to move, and move into the package once.
  - Two commands in docs name flags their entrypoint lacks: the Milestone 55 acceptance text
    (`weekly_run --defaults-path`, never built) and a historical `CHANGELOG.md` line
    (`walk_forward_backtest --wf-resume`). Both are records, so I propose leaving them.
- `scripts/*.py`: 6,486 lines, all outside the coverage measure (confirmed). Today's suite
  covers 54.0% of their 2,140 statements (`SCRIPTS_COVERAGE.md`, per file). Moving the nine
  files proposed below with today's tests would take package statement coverage from 95.1% to
  about 90.9%, right at the gate's 90% floor. So each move lands with its tests (60.4), and the
  thin ones with no tests today (`leakage_audit`, `betting_report_excel`) need tests first.

## 1. Script dispositions

| file | lines | proposal | why (from the inventory) |
| --- | --- | --- | --- |
| `weekly_run.py` | 1,917 | move, split into a `nfl_predictor/weekly_run/` package | produces the picks; web template; CI smoke; imports two scripts |
| `power_rankings.py` | 868 | move: computation into `nfl_predictor/reporting/power_rankings.py`, thin CLI | `weekly_run` calls `compute_power_rankings`, `resolve_ranking_options`, `_write_outputs`; web; CI |
| `walk_forward_backtest.py` | 400 | move, thin CLI | the benchmark: 19 launchers, web, 1 test module |
| `wf_compare.py` | 436 | move, thin CLI | 1 test module, README |
| `leakage_audit.py` | 107 | move, CLI beside `nfl_predictor/ml/leakage_audit.py` | required tool; web |
| `validate_offline.py`, `validate_live.py` | 85, 79 | move (two modules, or one with two modes) | web templates |
| `betting_report_excel.py` | 69 | move, CLI beside `nfl_predictor/reporting/betting_excel.py` | web template |
| `shap_analysis.py` | 188 | move, dropping its `ScoreModel` branch | web template, 1 test module |
| `betting_pipeline.py` | 1,108 | retire once `build_betting_report` and its helpers move into `nfl_predictor/reporting/` | no web, launcher or CI; `weekly_run` uses only `build_betting_report`; drifted matrix and defaults |
| `golden_command.py` | 681 | retire, with its 2 test modules | agreed 2026-09-23; no web, launcher or CI |
| `backtest_predictions.py` | 389 | retire | no test, web, launcher or CI; one README line; replaced by walk-forward |
| `objective_compare_models.py` | 159 | retire, with `nfl_predictor/ml/model_compare.py` and `tests/test_model_compare.py` | no script test, web, launcher or CI; `model_compare.py` has no other importer |
| `gate.sh` | 121 | keep | the one gate; mirrors CI |

The four retirements take 85 of the 309 flags with them. Once `weekly_run` stops importing
`scripts`, no production code imports from `scripts/`.

## 2. Flag removals (beyond the retired scripts)

Each is behavior-preserving for every weekly output, since production either does not set the
flag or sets it to a value that is never applied.

1. `weekly_run --wf-early-stopping-rounds` and `wf_compare --early-stopping-rounds` (inert), with
   the `wf_early_stopping_rounds: 50` key in `config/weekly_run.yaml`, and the
   `WalkForwardConfig.early_stopping_rounds` field (the step-2 follow-up). Removing the field
   changes every checkpoint fingerprint, so it lands in the same chunk as the other edits
   under `nfl_predictor/ml/`, and no reference run is planned during this milestone.
2. `leakage_audit --include-market` (no-op); `--exclude-market` stays.
3. `weekly_run --wf-n-jobs` and its config key; `--xgb-n-jobs` already governs both stages.
4. The week-based recency half-life, never measured by any run: `ml_model` and
   `walk_forward_backtest --recency-half-life-weeks`, `weekly_run --wf-recency-half-life-weeks`
   and `--train-recency-half-life-weeks`, plus their plumbing. **Your call:** removal is my
   recommendation (rule 9: fewer knobs when nothing measured says otherwise).
5. The `score` choice of `--model-kind` (task 55.5, below).
6. `postseason_weight: 1.3` in `config/weekly_run.yaml` is inert while `include_postseason` is
   false. The flag stays live. **Your call:** drop the key, or keep it ready for the playoff
   design task 56.2 deferred.

## 3. Surviving canonical names

Proposed rule: **walk-forward settings always carry `--wf-`, tuning settings always carry
`--tune-`, the XGBoost runtime always carries `--xgb-`, and final-training settings are bare.**
This keeps every key in `config/weekly_run.yaml` and every flag spelling the 19 benchmark
launchers use. Each renamed flag keeps its old spelling as a second option string on the same
argument, so old commands still parse. Survivors, after the retirements:

| concept | canonical | old spellings kept as aliases |
| --- | --- | --- |
| tuning trials | `--tune-trials` | `--tune-n-trials` (weekly_run) |
| tuning objective | `--tune-objective` | `--tune-metric` (ml_model) |
| tuning CV splits | `--tune-cv-splits` | `--cv-splits` (ml_model) |
| tuning early stopping | `--tune-early-stopping-rounds` | `--train-early-stopping-rounds` (weekly_run), `--early-stopping-rounds` (ml_model) |
| win-probability calibration | `--win-prob-calibration` | `--calibration` (walk_forward_backtest; 19 launchers) |
| market blend weight | `--market-prob-weight` (one action) | `--market-prob-blend` |
| walk-forward window | `--wf-eval-last-n-seasons`, `--wf-start-week`, `--wf-calibration-weeks`, `--wf-exclude-incomplete-seasons` | the bare forms in `walk_forward_backtest` and `wf_compare` |
| walk-forward trees | `--wf-n-estimators`, `--wf-max-depth`, `--wf-learning-rate` | `--n-estimators`, `--max-depth`, `--learning-rate` |
| market mode | `--wf-market-mode` | `--market-mode` (wf_compare) |
| XGBoost threads | `--xgb-n-jobs` | `--n-jobs` (wf_compare) |
| saved model input | `--model-in` | `--model-path` (shap_analysis) |

Notes: `ml_model --calibration-weeks` and `--calibration-seasons` configure final training,
not walk-forward, so under the rule they stay bare. The `--tune-early-stopping-rounds` rename
moves the `train_early_stopping_rounds: 80` config key to `tune_early_stopping_rounds`. No
saved web job config exists (`data/web/job_configs/` does not exist), so nothing stored
breaks.
`--win-prob-uncertainty` is a boolean in `ml_model`, `walk_forward_backtest` and `wf_compare`
but `off`/`on`/`both` in `weekly_run`; I propose leaving the two types alone, since unifying
them changes parsing. Output-location flags (`--out`, `--out-json`, `--output-path`,
`--out-dir`, `--output-dir`, `--run-dir`) differ in meaning (a file against a directory), so I
propose no unification.

## 4. Command style for moved entrypoints

Recommendation: **`python -m nfl_predictor.<name>`**, keeping today's names
(`nfl_predictor.weekly_run`, `.power_rankings`, `.walk_forward_backtest`, `.wf_compare`,
`.leakage_audit`, `.validate_offline`, `.validate_live`, `.betting_report_excel`,
`.shap_analysis`). Six entrypoints already run this way (`data_collection`, `ml_model`, `api`,
`api.auth.cli`, `lines_refresh`, `week_builder`), and the web catalog already builds `-m`
commands for four of them. There is no `[project.scripts]` table today; console commands can be
added later without renaming anything.

## 5. Old `scripts/<name>.py` paths in `models/*/launch.sh`

Facts: 19 launchers call `scripts/walk_forward_backtest.py` and one calls
`scripts/weekly_run.py`; `models/` is not in git.

Recommendation: **rely on each run's recorded git commit**, with a short "Reproducing an old
run" note in the README (`git worktree add <commit>`, then the launcher as written). Re-running
an old launcher on new code would not reproduce it anyway: any edit under `nfl_predictor/ml/`
changes the checkpoint fingerprint, and rescoring reads the checkpoints on disk, not the
launcher. The alternatives are permanent two-line shims (the `scripts/` directory survives as
aliases), or shims with a deprecation period.

## 6. Task 55.5: `ScoreModel`

Evidence (INVENTORY.md, "ScoreModel and the `score` model kind"): it is reachable only through
`--model-kind score` (`ml_model`, `backtest_predictions`, `power_rankings`) and the web `train`
template. `weekly_run` never uses it. No run on disk records `model_kind` `score`, and all 10
saved model artifacts under `models/` are `MarginTotalModel`.

Recommendation: **deprecate cleanly by removing it**: `ScoreModel`, `train_score_model*`, the
score predict path, the `score` choice everywhere (including the catalog), and the score
branches in `power_rankings`, `shap_analysis` and `feature_importance`, with their tests. No
weekly output changes. The alternative is to keep it and document it as experimental.

## 7. Step-2 follow-ups

| follow-up | inventory evidence | proposal |
| --- | --- | --- |
| `WalkForwardConfig.early_stopping_rounds` | read only by `to_dict` | remove (item 2.1) |
| `wf_compare.py` pick-accuracy column | the display shows configured Brier and log loss but only `deterministic_pick_accuracy` | show both pick-accuracy columns (a printed-summary fix with a test) |
| automatic OpenMP wait policy | no code sets it; 13 launchers set it by hand | the walk-forward entrypoints set `OMP_WAIT_POLICY=PASSIVE` when unset, before XGBoost loads (the measured stalls came from mid-run load); scheduling only |
| pruning `models/wf_checkpoints/` | no pruning code; 38 directories, 298.6 MB | a read-only listing of checkpoint directories no run directory references, plus a README note; deleting stays must-ask |
| `_build_xgb_fit_kwargs` / `LogEvalCallback` | XGBoost 3.4.1 `fit()` has no `callbacks`, so the callback is dropped; in-season fits also pass no eval set | delete the dead callback plumbing rather than revive it |

## Questions for sign-off

1. The script dispositions in section 1, including retiring `betting_pipeline`,
   `backtest_predictions` and `objective_compare_models` (with `model_compare.py`), and moving
   rather than retiring `shap_analysis`.
2. The removal list in section 2, and separately items 2.4 (week-based half-life) and 2.6 (the
   inert `postseason_weight` key).
3. The naming rule and the canonical names in section 3.
4. `python -m nfl_predictor.<name>` as the command style.
5. The reproduction policy for old launcher paths (recorded commit, shims, or deprecation).
6. Task 55.5: remove `ScoreModel`, or keep it as experimental.
7. The model-kind vocabulary defect: fix it inside this milestone when the web templates move
   (60.7, test first; proposed canonical `blend`, accepting `blended_margin_total` as an alias),
   or as a separate step. It changes which web launches succeed, not any prediction.
8. The `walk_forward_backtest --calibration` default (`platt`, against the benchmark's `auto`):
   leave it, or change it to `auto` as a separate, clearly labelled default change. It changes
   bare runs only, not the launchers.
9. The step-2 follow-up proposals in section 7, the OpenMP policy above all.
