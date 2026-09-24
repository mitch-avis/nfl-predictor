# CLI and scripts consolidation: proposed dispositions for sign-off

Written 2026-09-24 on `feat/m60-cli-consolidation` for task 60.3, revised the same day after
the user's questions (ScoreModel, `gate.sh`, the two kinds of tools, simplification, and how
success is measured). Every fact below comes from the generated inventory beside this file and
can be reproduced:

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
  spellings. `betting_pipeline`'s copy of the stage-1 matrix has drifted: 7 rows against
  `weekly_run`'s 9 (no `none_base`, no `auto_base`).
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
  - **ScoreModel has never been evaluated.** The walk-forward harness trains margin/total only
    (`walk_forward.py` has no model-kind switch), so no backtest of ScoreModel exists. No saved
    run or model on disk used it (all 10 saved models are `MarginTotalModel`).
  - The Excel betting workbook is optional already: the weekly run writes it only when
    `betting_template_path` is set, and `config/weekly_run.yaml` does not set it. The web
    betting page recomputes its report from the predictions file with its own formulas
    (`nfl_predictor/api/readers/market.py`) and reads neither the workbook nor the CSV. Only
    the Excel code uses the `openpyxl` dependency.
  - CI (`.github/workflows/validation.yml`) lists the same checks as `scripts/gate.sh` step by
    step, so the two lists can drift apart.
- `scripts/*.py`: 6,486 lines, all outside the coverage measure (confirmed). Today's suite
  covers 54.0% of their 2,140 statements (`SCRIPTS_COVERAGE.md`, per file). Moving the eight
  files proposed below with today's tests would take package statement coverage from 95.1% to
  about 91.1%, close to the gate's 90% floor (the projection leaves out code the retirements
  remove, such as the tested Excel module). So each move lands with its tests (60.4), and the
  ones with no tests today need tests first.

## 0. Target shape: one front door, three kinds of work

Today there are 19 separate entrypoints. Proposed: **one command, `nfl-predictor`, with
subcommands**, installed by `uv sync` as `.venv/bin/nfl-predictor` (a `[project.scripts]` entry)
and also runnable as `python -m nfl_predictor`. `nfl-predictor --help` lists everything, grouped:

| group | subcommand | today's entrypoint |
| --- | --- | --- |
| weekly (your run) | `weekly` | `scripts/weekly_run.py` |
| research | `backtest` | `scripts/walk_forward_backtest.py` |
| research | `compare` (new: paired comparison of two backtests from their checkpoints) | replaces `objective_compare_models.py` and the per-run `compare_to_benchmark.py` copies under `models/` |
| research | `sweep` (revisit after step 3) | `scripts/wf_compare.py` |
| research | `explain` | `scripts/shap_analysis.py` |
| data | `data` | `python -m nfl_predictor.data_collection` |
| data | `validate` (`--live` for the schedule check) | `scripts/validate_offline.py`, `scripts/validate_live.py` |
| data | `leakage-audit` | `scripts/leakage_audit.py` |
| data | `lines` | `python -m nfl_predictor.lines_refresh` |
| data | `build-week` | `python -m nfl_predictor.week_builder` |
| models by hand | `train`, `predict`, `rankings` | `python -m nfl_predictor.ml_model`, `scripts/power_rankings.py` |
| web | `web`, `users` | `python -m nfl_predictor.api`, `python -m nfl_predictor.api.auth.cli` |

The weekly run stays driven by `config/weekly_run.yaml`, so for you it is
`.venv/bin/nfl-predictor weekly --config config/weekly_run.yaml` (or the config becomes the
default and it is just `nfl-predictor weekly`). Each subcommand keeps its own flags; the naming
rule in section 3 makes shared concepts spell the same way everywhere.

## 1. Script dispositions

| file | lines | proposal | why (from the inventory) |
| --- | --- | --- | --- |
| `weekly_run.py` | 1,917 | move, split into a `nfl_predictor/weekly_run/` package (`weekly`) | produces the picks; web template; CI smoke; imports two scripts |
| `power_rankings.py` | 868 | move: computation into `nfl_predictor/reporting/power_rankings.py`, thin CLI (`rankings`) | `weekly_run` calls `compute_power_rankings`, `resolve_ranking_options`, `_write_outputs`; web; CI |
| `walk_forward_backtest.py` | 400 | move (`backtest`) | the benchmark: 19 launchers, web, 1 test module |
| `wf_compare.py` | 436 | move (`sweep`); decide after step 3 whether it is still needed | 1 test module, README; overlaps the weekly stage-1 sweep that task 56.5 will decide |
| `leakage_audit.py` | 107 | move (`leakage-audit`) | required tool (AGENTS.md "Leakage Audit"); web |
| `validate_offline.py`, `validate_live.py` | 85, 79 | merge into one `validate` with a `--live` switch | web templates; both check `data/all_data.csv` |
| `shap_analysis.py` | 188 | move (`explain`), keeping its ScoreModel branch | web template, 1 test module |
| `betting_report_excel.py` | 69 | **retire**, with `nfl_predictor/reporting/betting_excel.py` (1,359 lines), the `openpyxl` dependency, `weekly_run --betting-template-path`, the web `betting_xlsx` job, the workbook download routes and button, and their tests | you no longer need the Excel version; the web page does not read it |
| `betting_pipeline.py` | 1,108 | retire once `build_betting_report` and its helpers move into `nfl_predictor/reporting/` | no web, launcher or CI; `weekly_run` uses only `build_betting_report`; drifted matrix and defaults |
| `golden_command.py` | 681 | retire, with its 2 test modules | agreed 2026-09-23; no web, launcher or CI |
| `backtest_predictions.py` | 389 | retire | no test, web, launcher or CI; one README line; runs one saved model over every completed game, mostly games it trained on; walk-forward already reports confidence-pool points out of sample |
| `objective_compare_models.py` | 159 | retire, with `nfl_predictor/ml/model_compare.py` and `tests/test_model_compare.py`; its idea lives on as `compare` | no script test, web, launcher or CI; `model_compare.py` has no other importer |
| `gate.sh` | 121 | keep, and make CI call it | see "Why keep `gate.sh`" below |

The retirements take 87 of the 309 flags with them (85 in the four large scripts, 2 in
`betting_report_excel`), plus `--betting-template-path`. Once `weekly_run` stops importing
`scripts`, no production code imports from `scripts/`.

**Why keep `gate.sh`.** It runs, in one command, every check GitHub CI runs (lockfile, formatting,
lint, two type checkers, the tests with the 90% coverage floor, Markdown lint and the CLI smoke
checks), in CI's order, and reports all failures at once. Agents run it before calling anything
done (AGENTS.md rule 1); you never need to. If it ends up as the only file in `scripts/`, that
is the right signal: `scripts/` holds developer tooling, and everything you run is in the
package. Proposed improvement: make the CI workflow call `scripts/gate.sh` instead of repeating
its steps, so there is one definition of "the checks pass".

**What the weekly run does, and what it does not do yet.** Today `weekly` refreshes the data
(ETL), runs a stage-1 walk-forward over nine probability candidates, trains the final model,
predicts the week, writes predictions, the pick'em winners and 1..N confidence picks (one
file), the betting CSV, power rankings and projected standings. Survivor picks do not exist
yet: the survivor optimizer is web-UI Phase 4 (task 58.1). Two later simplifications depend on
decisions outside this milestone: stage 1 retrains nine candidates every week only because
production picks a probability path from them (task 56.5, step 3), and the betting CSV and the
web page compute the same report with two separate sets of formulas (one should serve both,
checked by a test that they agree).

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
   and `--train-recency-half-life-weeks`, plus their plumbing. Recommended (rule 9: fewer knobs
   when nothing measured says otherwise).
5. `postseason_weight: 1.3` in `config/weekly_run.yaml` is inert while `include_postseason` is
   false. The flag stays live for the playoff design task 56.2 deferred; recommended: drop the
   key, since the value was never measured and a playoff design will choose its own.

The `score` choice of `--model-kind` is **no longer on this list** (section 6).

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
breaks. `--win-prob-uncertainty` is a boolean in `ml_model`, `walk_forward_backtest` and
`wf_compare` but `off`/`on`/`both` in `weekly_run`; I propose leaving the two types alone,
since unifying them changes parsing. Output-location flags (`--out`, `--out-json`,
`--output-path`, `--out-dir`, `--output-dir`, `--run-dir`) differ in meaning (a file against a
directory), so I propose no unification.

## 4. Command style

Recommendation: the single front door of section 0, **`.venv/bin/nfl-predictor <command>`**,
with `python -m nfl_predictor <command>` as the equivalent that needs no install step. The old
per-module forms (`python -m nfl_predictor.data_collection` and the others) keep working, since
the web job runner and existing notes use them; the docs and the web catalog move to the front
door. The fallback, if you prefer fewer changes, is `python -m nfl_predictor.<name>` per
module, keeping today's names.

## 5. Old `scripts/<name>.py` paths in `models/*/launch.sh`

Facts: 19 launchers call `scripts/walk_forward_backtest.py` and one calls
`scripts/weekly_run.py`; `models/` is not in git.

Recommendation: **rely on each run's recorded git commit**, with a short "Reproducing an old
run" note in the README (`git worktree add <commit>`, then the launcher as written). Re-running
an old launcher on new code would not reproduce it anyway: any edit under `nfl_predictor/ml/`
changes the checkpoint fingerprint, and rescoring reads the checkpoints on disk, not the
launcher. The alternatives are permanent two-line shims (the `scripts/` directory survives as
aliases), or shims with a deprecation period.

## 6. Task 55.5: `ScoreModel` gets a fair test, not a removal

What exists (INVENTORY.md, "ScoreModel and the `score` model kind"): two XGBoost regressors,
one for the home score and one for the away score, trained on the same features and default
settings as the margin/total model. The win probability comes from the same fixed normal curve
on (home minus away). It trains, predicts, saves and loads, and has unit tests for each of
those. It is reachable through `--model-kind score` (`ml_model`, `backtest_predictions`,
`power_rankings`) and the web `train` template.

What it lacks, compared with the production margin/total model:

- **Walk-forward support.** The harness cannot backtest it, so it has never been measured.
- **Market anchoring.** The margin/total model learns only the correction to the market spread
  and total; ScoreModel learns scores from scratch. Tested as it stands, it would lose for that
  reason alone. The fair version anchors each score to the market's implied team score
  (total line / 2 plus or minus spread / 2).
- **Uncertainty ranges** (p10/p90), which AGENTS.md requires for every model's predictions.
- **Tuning.** It has no Optuna path; it uses the margin/total model's default settings.

What to expect honestly: margin = home − away and total = home + away, so a perfect score model
and a perfect margin/total model give identical answers. With real data, predicting the margin
directly tends to win, because the model spends all its capacity on the difference between the
teams, while two score models spend much of theirs on the overall scoring level (pace, weather,
era), which cancels in the difference, and their errors add. That is a prior, not a result;
only a measurement settles it.

Proposed plan, as a new milestone ("Model-family comparison"), so nothing is removed first:

1. **Parity work (test-first):** walk-forward support for model kinds; market anchoring for
   ScoreModel; the same probability path as the benchmark; p10/p90 ranges. The blended model
   (`--model-kind blend`: a team-only model, a market-only model and a linear blend layer), also
   never backtested, joins as a third candidate at little extra cost.
2. **Screen, after roadmap step 3** (GPU reference, settled probability path, pick-time lines):
   six seasons, two seeds, same data build, same folds, each candidate against the
   margin/total reference and against the market, with a hypothesis and decision rule written
   first. At shared default settings this still slightly favors margin/total, since the
   defaults were tuned for it, and the check-in will say so.
3. **Fair final, inside step 5:** tune every surviving candidate with the same Optuna budget,
   then confirm on six seasons and two seeds. The winner becomes the default only by your
   decision (rule 13).

In this milestone ScoreModel and `blend` stay, move with the rest, and keep their tests.

## 7. Step-2 follow-ups

| follow-up | inventory evidence | proposal |
| --- | --- | --- |
| `WalkForwardConfig.early_stopping_rounds` | read only by `to_dict` | remove (item 2.1) |
| `wf_compare.py` pick-accuracy column | the display shows configured Brier and log loss but only `deterministic_pick_accuracy` | show both pick-accuracy columns (a printed-summary fix with a test) |
| automatic OpenMP wait policy | no code sets it; 13 launchers set it by hand | the walk-forward entrypoints set `OMP_WAIT_POLICY=PASSIVE` when unset, before XGBoost loads (the measured stalls came from mid-run load); scheduling only |
| pruning `models/wf_checkpoints/` | no pruning code; 38 directories, 298.6 MB | a read-only listing of checkpoint directories no run directory references, plus a README note; deleting stays must-ask |
| `_build_xgb_fit_kwargs` / `LogEvalCallback` | XGBoost 3.4.1 `fit()` has no `callbacks`, so the callback is dropped; in-season fits also pass no eval set | delete the dead callback plumbing rather than revive it |

## 8. How success is measured (input to steps 3 and 5, not a change in this milestone)

Three layers, each with its own measure:

| layer | what it is | measure |
| --- | --- | --- |
| 1. prediction | XGBoost predicts margin and total (a regression), trained with squared error | margin MAE and total MAE, each against the market line's MAE |
| 2. probability | a curve turns the predicted margin into a win probability | Brier (primary) and log loss, against the market's own probabilities on the same games |
| 3. decisions | picks, confidence order, survivor, bets | pool points (and pick accuracy, reported but never used to choose); for bets, edge against the closing line, diagnostic only |

Facts from the code: `none`, `auto`, `sigma` and `elo` are symmetric curves that always rise
and pass through 50% at a margin of 0. So none of them can change a pick or the confidence
order; they change only the stated probabilities. `platt` (fitted logistic, with an offset) and
`isotonic` (fitted steps, which can tie games) can move a close pick. The market blend in
production (weight 0.2, clamp 0.1) is the one setting that can reorder picks. Evidence so far
(AGENTS.md): no fitted calibrator has beaten the fixed curve, and the closing line beats the
model slightly on weeks 3-18 Brier.

Recommendations for later steps: select on Brier of the probabilities actually submitted,
require no loss on log loss, break ties on pool points and then margin MAE; settle on one
probability path in step 3 (the fixed curve with its width fitted out of fold is the
simplest candidate) and retire the calibrators it does not use; test the training objective
(squared error against pseudo-Huber) as one choice inside the step-5 tune rather than a separate
campaign; keep betting a separate decision layer (bet only where the model's probability beats
the market's by a margin, sized conservatively) rather than training on an odds-weighted loss.

## Questions for sign-off

See the check-in message of 2026-09-24 for the full reasoning. In short:

1. The front door (`nfl-predictor <command>`) and the grouping in section 0.
2. The script dispositions in section 1, including retiring the Excel workbook end to end.
3. The flag removals in section 2.
4. The naming rule and canonical names in section 3.
5. The reproduction policy for old launcher paths (section 5).
6. ScoreModel: keep it, and add the model-family comparison (section 6) to the roadmap.
7. The model-kind vocabulary defect: fix it when the web templates move (60.7, test first).
8. The `walk_forward_backtest --calibration` default (`platt` against the benchmark's `auto`).
9. The step-2 follow-ups in section 7.
10. CI calls `scripts/gate.sh`.
11. A new `compare` command for paired comparisons of two backtests.
