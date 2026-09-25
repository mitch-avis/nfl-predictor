# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestone 60, task 56.7 and "From the 2026 Week 3
weekly run") and this file.

## State (written 2026-09-24 night, Milestone 60 in 60.6)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.24.0`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh --web` exits `0` on the final
  tree (see the `0.24.0` commits). The working tree is clean. (The user deleted their
  untracked LightGBM session transcript on purpose.)
- Tasks 60.1-60.5 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)"); every
  sign-off decision is in `.agents/m60/PROPOSAL.md`, "Sign-off", and in the 60.6-60.9 texts.
- 60.6 is in progress; its "Progress" note in `TODO.md` lists what landed and what is next.
  `0.19.0` retired `golden_command`, `betting_pipeline`, `backtest_predictions`, the
  `objective_compare_models` script and the Excel workbook end to end (web job, routes and
  button included). `0.20.0` was the one `nfl_predictor/ml/` chunk: ScoreModel (task 55.5,
  archived), `model_compare.py`, the inert early-stopping settings, the week-based half-life,
  the dead callback, `shap_analysis --component`, and the numpy bootstrap. Every walk-forward
  checkpoint fingerprint changed with it; no further `nfl_predictor/ml/` edit is planned in
  this milestone. `0.21.0` added the front door `nfl-predictor <command>`
  (`nfl_predictor/cli/main.py`), moved the weekly run into `nfl_predictor/weekly_run/` and the
  other entrypoints into `nfl_predictor/cli/`, merged the validate scripts (`validate
  --live`), and left `scripts/<name>.py` as thin shims for the web jobs. `0.22.0` renamed
  options under one naming rule with every old spelling kept as an alias, merged the market
  weight options, renamed the weekly tuning config keys (old keys still load), removed
  `leakage_audit --include-market`, and moved `ml_model_cli.py` to `nfl_predictor/cli/train.py`
  (the last `nfl_predictor/ml/` edit; the fingerprint changed once more). `0.23.0` made the
  weekly run read `config/weekly_run.yaml` by default and rewrote it to the code defaults
  (production output unchanged); the weekly snapshot was rewritten on purpose to pin the
  production configuration. `0.24.0` finished the step-2 follow-ups (`nfl-predictor
  checkpoints`, the automatic `OMP_WAIT_POLICY=PASSIVE`, the `sweep` summary's pick-accuracy
  columns). 60.6 has landed apart from `--wf-n-jobs`, which waits on the user.
- The characterization snapshots must not change during a move; an intended output change
  rewrites them with `NFLP_UPDATE_SNAPSHOTS=1` in the same commit and says so in the changelog.
  `tests/fixtures/cli_surface.json` changes with every flag removal or rename (intended; audit
  the diff: `0.19.0` removed exactly 88 actions, 309 to 221; `0.20.0` 7 more, to 214;
  `0.21.0` renamed module keys and added the front door and `--live`, to 216; `0.22.0` merged
  two option pairs and removed `--include-market`, to 213).
- `.agents/m60/INVENTORY.md` is the signed-off 60.1 record; do not regenerate it over the
  signed-off copy (it now reports fewer entrypoints because files were retired).
- Decisions made 2026-09-24 (all recorded in `TODO.md` or `ARCHIVE.md`):
  - keep `explain` and `shap`; the numpy bootstrap rewrite goes in the `nfl_predictor/ml/`
    chunk; `postseason_weight` stays; LightGBM stays installed and parked (Milestone 57, CPU
    only when it reopens);
  - remove `shap_analysis --component` (the user never asked for a market model; done in
    `0.20.0`);
  - the 13 test modules' `from scripts import` lines go with 60.6, as the entrypoints move;
  - task 56.7(a), done in `0.23.0`: the weekly run reads `config/weekly_run.yaml` by default,
    rewritten to today's code defaults. The YAML's values still need optimizing (task
    56.7(b), 55.4);
  - how many seasons weekly stage 1 scores is settled in task 56.7.

## How to launch a weekly run today

`.venv/bin/nfl-predictor weekly --run-id <id> --xgb-device cuda` (the old
`.venv/bin/python scripts/weekly_run.py ...` form still works) through a `launch.sh` with
`nohup setsid` (see `models/weekly_2026_week_03_full/launch.sh`). It reads
`config/weekly_run.yaml`, which holds the code defaults, so it produces what earlier 2026 runs
did. Without `--xgb-device cuda`, stage 1 runs on the CPU at about six times the time. Do not run a
weekly run or any walk-forward unless the user asks.

## Your task

1. Settle `--wf-n-jobs` with the user (question below), then close 60.6 (archive it with a
   summary of `0.19.0`-`0.24.0`).
2. Then 60.7 (web job templates onto the front door, the model-kind vocabulary fix, the API
   catch-all returning JSON 404s, then removing the `scripts/` shims), 60.8 (CI calls
   `scripts/gate.sh`, the gate's smoke checks use `nfl-predictor`, the docs) and 60.9 (the
   `compare` command). Avoid edits under `nfl_predictor/ml/` (they change every checkpoint
   fingerprint). Rewrite this file at every landed chunk (rule 8).

## Open questions for the user

- `weekly_run --wf-n-jobs` (signed off for removal) was kept: production never read the
  YAML, so with `--xgb-n-jobs` unset it sets stage 1's CPU threads (default 1). Remove it
  (stage 1 then uses the XGBoost default threads), or keep it? This also bears on the YAML
  rewrite, which records today's defaults (`wf_n_jobs: 1`, `xgb_n_jobs` unset).

## Notes

- The frontend's catch-all route answers unknown `/api/...` paths with `index.html` and 200
  (logged under 60.7); an API test that checks a removed route must look at `app.routes`.
- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
- Never run two walk-forwards at once; check `uptime` and `pgrep -af walk_forward` first.
