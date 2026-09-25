# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestone 60, task 56.7 and "From the 2026 Week 3
weekly run") and this file.

## State (written 2026-09-24 night, Milestone 60 in 60.6)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.19.0`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh --web` exits `0` on the final
  tree (see the `0.19.0` commits). The working tree is clean. (The user deleted their
  untracked LightGBM session transcript on purpose.)
- Tasks 60.1-60.5 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)"); every
  sign-off decision is in `.agents/m60/PROPOSAL.md`, "Sign-off", and in the 60.6-60.9 texts.
- 60.6 is in progress; its "Progress" note in `TODO.md` lists what landed and what is next.
  `0.19.0` retired `golden_command`, `betting_pipeline`, `backtest_predictions`, the
  `objective_compare_models` script and the Excel workbook end to end (web job, routes and
  button included). `nfl_predictor/ml/model_compare.py` and `tests/test_model_compare.py`
  remain, for the `nfl_predictor/ml/` chunk.
- The characterization snapshots must not change during a move; an intended output change
  rewrites them with `NFLP_UPDATE_SNAPSHOTS=1` in the same commit and says so in the changelog.
  `tests/fixtures/cli_surface.json` changes with every flag removal or rename (intended; audit
  the diff: `0.19.0` removed exactly 88 actions, 309 to 221).
- `.agents/m60/INVENTORY.md` is the signed-off 60.1 record; do not regenerate it over the
  signed-off copy (it now reports fewer entrypoints because files were retired).
- Decisions made 2026-09-24 (all recorded in `TODO.md` or `ARCHIVE.md`):
  - keep `explain` and `shap`; the numpy bootstrap rewrite goes in the `nfl_predictor/ml/`
    chunk; `postseason_weight` stays; LightGBM stays installed and parked (Milestone 57, CPU
    only when it reopens);
  - remove `shap_analysis --component` (the user never asked for a market model);
  - the 13 test modules' `from scripts import` lines go with 60.6, as the entrypoints move;
  - task 56.7(a): the weekly run reads `config/weekly_run.yaml` by default, with the YAML
    rewritten to today's code defaults (quote `off` as `"off"`), in 60.6. That re-points the
    weekly characterization snapshot at what production runs: an intended snapshot rewrite,
    stated in the changelog. The YAML's values still need optimizing (task 56.7(b), 55.4);
  - how many seasons weekly stage 1 scores is settled in task 56.7.

## How to launch a weekly run today (until 60.6 lands)

`.venv/bin/python scripts/weekly_run.py --run-id <id> --xgb-device cuda` through a `launch.sh`
with `nohup setsid` (see `models/weekly_2026_week_03_full/launch.sh`). Do **not** pass
`--config config/weekly_run.yaml` until 60.6 has rewritten it: its current values are untested.
Without `--xgb-device cuda`, stage 1 runs on the CPU at about six times the time. Do not run a
weekly run or any walk-forward unless the user asks.

## Your task

1. Continue 60.6 with the `nfl_predictor/ml/` chunk: ScoreModel (task 55.5, PROPOSAL.md
   section 6), `nfl_predictor/ml/model_compare.py` and its test,
   `WalkForwardConfig.early_stopping_rounds` with `weekly_run --wf-early-stopping-rounds`,
   `wf_compare --early-stopping-rounds` and the `wf_early_stopping_rounds` config key, the
   week-based recency half-life flags and plumbing, `_build_xgb_fit_kwargs` /
   `LogEvalCallback`, and the numpy rewrite of
   `walk_forward._bootstrap_probability_differences` (a test first proves identical values).
   One chunk, because every edit there changes every checkpoint fingerprint.
2. Then the entrypoint moves behind `nfl-predictor <command>` (weekly run split into a
   package; the `scripts/` files stay as thin shims until 60.7 repoints the web jobs), then
   the shared options module, renames and remaining flag removals, the YAML default, and the
   step-2 follow-ups. When the entrypoints have moved, `grep -rl "from scripts import" tests`
   must be empty.
3. Then 60.7-60.9. Rewrite this file at every landed chunk (rule 8).

## Open questions for the user

- None blocking.

## Notes

- The frontend's catch-all route answers unknown `/api/...` paths with `index.html` and 200
  (logged under 60.7); an API test that checks a removed route must look at `app.routes`.
- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
- Never run two walk-forwards at once; check `uptime` and `pgrep -af walk_forward` first.
