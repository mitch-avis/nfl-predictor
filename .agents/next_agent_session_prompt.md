# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status" and Milestone 60) and this file.

## State (written 2026-09-24, Milestone 60 at 60.4)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.18.2`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh` exits `0` on the final tree (the
  check-in carries the numbers).
- Tasks 60.1-60.3 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)"). The user
  signed off on 2026-09-24; every decision is in `.agents/m60/PROPOSAL.md`, "Sign-off", and
  in the amended 60.4-60.9 texts in `TODO.md`. Headlines:
  - one `nfl-predictor <command>` front door (`[project.scripts]` plus `python -m
    nfl_predictor`); subcommands grouped as weekly, research, data, models by hand, web;
  - retire `golden_command`, `backtest_predictions`, `objective_compare_models` (with
    `nfl_predictor/ml/model_compare.py`), `betting_pipeline` (after `build_betting_report`
    moves) and the whole Excel betting workbook (`betting_report_excel`,
    `nfl_predictor/reporting/betting_excel.py`, `openpyxl`, `--betting-template-path`, the web
    `betting_xlsx` job, the download routes and button);
  - remove ScoreModel and everything only it uses (task 55.5);
  - the signed-off flag removals; `postseason_weight: 1.3` becomes a commented-out example in
    `config/weekly_run.yaml` (the power rankings never read it; training does when
    `include_postseason: true`);
  - the naming rule (`--wf-`, `--tune-`, `--xgb-`, bare final-training flags; old spellings kept
    as aliases); old launchers reproduce from their recorded git commit;
  - CI calls `scripts/gate.sh`; a new `compare` command (60.9); the model-kind vocabulary fix in
    60.7; the step-2 follow-ups (automatic `OMP_WAIT_POLICY=PASSIVE` included);
  - the `--calibration` default is deferred to task 56.5 (the user wants one calibration used
    the same way by every run type); survivor picks wait for task 58.1;
  - the measures of success for steps 3 and 5 are recorded under "Roadmap Status".
- 60.4 landed in `0.18.2` except the SHAP tests: the weekly-run characterization test, the
  entrypoint characterization tests, the parser-surface snapshot, the synthetic fixture
  `tests/weekly_fixture.py` and the shared helpers `tests/snapshots.py`. These snapshots must
  not change during a move; an intended output change rewrites them with
  `NFLP_UPDATE_SNAPSHOTS=1` in the same commit and says so in the changelog.
- Housekeeping done: the stray checkpoint folder was already gone; the stale
  `../nfl-predictor-web` worktree record is pruned. The fully merged `feat/web-ui` branch still
  exists locally and on `origin`; nobody asked to delete it.
- `.agents/gpt-5-4_task_55-8_transcript.md` is the user's untracked file; leave it alone.

## Your task

1. Get the user's answers to the two open questions below, then continue with 60.5 (move
   library code into the package, scripts calling it, no `from scripts import` left), keeping
   every characterization test green without touching its snapshots.
2. Then 60.6-60.9 in order, each chunk small, gated and committed; group every edit under
   `nfl_predictor/ml/` into one chunk (it changes every checkpoint fingerprint).
3. Rewrite this file at every landed chunk (rule 8).

## Open questions for the user

- SHAP: `shap` is not a declared dependency, so `scripts/shap_analysis.py` and the web
  `shap_analysis` job can only exit "not installed". Retire the command and its web job
  (gain-based feature importance is already written with every model), or add `shap` as an
  optional dependency and keep `explain`.
- The paired bootstrap in `walk_forward._bootstrap_probability_differences` takes 98% of a
  stage-1 run (5,000 scikit-learn metric calls per window). Rewrite it with numpy, proven to
  give identical values by a test, inside the `nfl_predictor/ml/` chunk?
- How production probabilities should be formed (task 56.5, roadmap step 3).

## Notes

- The Week 3 weekly run can go whenever the user wants; do not run it yourself unless asked,
  and never alongside another walk-forward.
- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
