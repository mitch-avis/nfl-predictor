# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestone 60 and its 60.6 "Progress" note, task
56.7, Milestone 58 task 58.5) and this file.

## State (written 2026-09-25, Milestone 60 finishing 60.6)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.24.0`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh` exits `0` on the final tree
  (946 passed, coverage 91.96%; `--web` passed on every chunk that touched `web/` or
  `nfl_predictor/api/`). The working tree is clean.
- Tasks 60.1-60.5 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)"). Sign-off
  decisions: `.agents/m60/PROPOSAL.md`, "Sign-off". `.agents/m60/INVENTORY.md` is the signed-off
  60.1 record; do not regenerate it over the signed-off copy.
- Characterization snapshots must not change during a move; an intended output change rewrites
  them with `NFLP_UPDATE_SNAPSHOTS=1` in the same commit and says so in the changelog.
  `tests/fixtures/cli_surface.json` changes with every flag change (intended; audit the diff).
  It held 309 actions when the milestone started and holds 217 in 15 parsers after `0.24.0`.

## What the last session did (2026-09-24/25), as reported to the user

- `0.18.4`: finished 60.4 (the SHAP characterization test). Deleted, with the user's approval,
  `data/qb_meta_data.csv`, `models/weekly_2026_week_03/` and the two Week 3 data backups.
- `0.18.5`: 60.5, library code out of `scripts/` (`nfl_predictor/reporting/power_rankings.py`,
  `nfl_predictor/reporting/betting_report.py`). The user moved the remaining 13 test imports of
  `scripts/` into 60.6; they are now all gone.
- `0.19.0`: retired `golden_command`, `betting_pipeline`, `backtest_predictions`,
  `objective_compare_models` and the Excel workbook end to end (web job, routes and Betting-page
  button included).
- `0.20.0`: the one planned `nfl_predictor/ml/` chunk: ScoreModel removed (task 55.5),
  `model_compare.py`, the inert early-stopping settings, the week-based half-life, the dead
  `LogEvalCallback`, `explain --component`, and the numpy paired bootstrap (identical to 1e-12,
  146x faster; the suite now runs in about 75 s).
- `0.21.0`: the front door `nfl-predictor <command>` (`nfl_predictor/cli/main.py`, also
  `python -m nfl_predictor`); the weekly run in `nfl_predictor/weekly_run/` (`config`,
  `inputs`, `stage1`, `pipeline`; definitions AST-identical), the other entrypoints in
  `nfl_predictor/cli/`, `validate --live`, and `scripts/<name>.py` as thin shims for the web jobs.
- `0.22.0`: option renames under one rule (`--wf-`, `--tune-`, `--xgb-`) with every old spelling
  kept; the merged `--market-prob-weight`; weekly config keys `tune_trials` /
  `tune_early_stopping_rounds` (old keys still load); `leakage-audit --include-market` removed;
  `ml_model_cli.py` moved to `nfl_predictor/cli/train.py` (outside the checkpoint fingerprint).
- `0.23.0`: task 56.7(a). The weekly run reads `config/weekly_run.yaml` by default and the file
  holds the code defaults (production output unchanged; `postseason_weight: 1.3` kept, inert).
  The weekly snapshot was rewritten on purpose to pin the production configuration.
- `0.24.0`: `nfl-predictor checkpoints` (read-only listing; all 38 directories referenced), the
  walk-forward commands default to `OMP_WAIT_POLICY=PASSIVE` unless set, and the `sweep`
  summary shows each probability view's own pick accuracy.
- Checkpoint fingerprints changed in `0.20.0` and `0.22.0`; no walk-forward has run since, so
  the next reference run retrains from scratch.
- Reported open items: `--wf-n-jobs` was kept because its removal's premise was wrong; the web
  API catch-all bug; the dormant `market_model` field.

## The user's answers (2026-09-25)

1. **Thread counts.** The user had assumed `--wf-n-jobs` and `--xgb-n-jobs` were the same flag.
   They set the same thing (XGBoost's CPU threads) in different places: `--xgb-n-jobs`, when
   set, overrides both stage 1 and final training; when it is unset, stage 1 uses `--wf-n-jobs`
   (default 1 thread) and final training uses every core (`DEFAULT_XGB_PARAMS["n_jobs"] =
   os.cpu_count()`). The user decided: keep only `--xgb-n-jobs`, have it set the CPU threads
   everywhere, and default it to the number of CPU cores instead of unset or 1. The GPU default
   (task 55.4) will make this mostly moot later.
2. **Web bug.** Document it accurately in 60.7 and in the web UI milestone. Done in the handoff
   chunk: TODO task 60.7, new TODO task 58.5, and `web_ui_plan.md` ("Milestone 60 impact" and a
   "Known defect" line under Status). The facts: the history-API fallback `GET /{path:path}` in
   `nfl_predictor/api/routers/static.py` (`mount_frontend`) does not exclude `api/`, so an
   unknown `/api/...` GET returns `index.html` with 200 instead of a JSON 404 (its docstring says
   otherwise).
3. **Market model.** The user never asked for a market model and asked why it was not removed
   when they first said so. The previous session read the answer as covering only the
   `--component` flag; that was too narrow. Remove it entirely now (details below).

## Your task

1. Finish 60.6 with the user's two decisions, then close it:
   - **One thread flag** (test first): remove `--wf-n-jobs` and the `wf_n_jobs` config key (map
     or reject an old config that sets it, clearly), make `--xgb-n-jobs` govern stage 1 and final
     training, and default it to `os.cpu_count()`; update `config/weekly_run.yaml`, its comment,
     the weekly fixture (`tests/weekly_fixture.py` sets both keys), and the CLI snapshot. Before
     landing, check whether the weekly fixture's outputs are identical at 1 thread and at N; if
     they differ, stop and report to the user (the snapshot pins 1 thread).
   - **Remove the market model**: the `BlendedMarginTotalModel.market_model` field and every
     branch that reads it (`ml_model_core`: `_early_stopping_info`,
     `_ensure_backward_compatible_model`, `_with_market_prob_config`; `ml_model_predict`;
     `ml_model_training`, including the `market_optuna` config behind `--tune-scope`
     `market`/`both`, then `--tune-scope` itself once you confirm it tunes nothing;
     `feature_importance`'s `market` component; the power-ranking pipeline), plus the tests
     that fake a market model. This edits `nfl_predictor/ml/`, so the fingerprint changes once
     more; confirm no saved blend model exists under `models/` (all are `MarginTotalModel`).
     The blended model itself (team model plus market line through the blend layer) stays.
   - Then archive 60.6 in `ARCHIVE.md` (a summary of `0.19.0` onward and these two) and remove
     it from `TODO.md`.
2. Task 60.7: move the web job templates (`nfl_predictor/api/jobs/catalog.py`) onto
   `nfl-predictor <command>` with canonical option names; keep the progress lines the runner
   parses (`Walk-forward fold N/M`, `WF candidate N/M`, `PROGRESS_RE` in
   `nfl_predictor/api/jobs/runner.py`) and the `weekly_run` template's config keys valid; fix the
   model-kind vocabulary (canonical `blend`, `blended_margin_total` accepted as an alias; three
   web launches fail today) test first; fix the API catch-all bug (task 58.5) test first; then
   delete the `scripts/` shims (only `scripts/gate.sh` stays). Update `tests/api/`, run
   `scripts/gate.sh --web`, amend `web_ui_plan.md` and `web/README.md`, and close 58.5.
3. Task 60.8 (CI calls `scripts/gate.sh`; the gate's smoke checks use `nfl-predictor`; README,
   AGENTS.md "Repo scripts"/"Project Shape"/"Dev Workflows"/launch instructions, a "Reproducing
   an old run" note) and 60.9 (the `compare` command; it must reproduce an existing `REVIEW.md`
   rescore exactly before it is trusted).
4. Each chunk small, versioned, gated (`--web` when the web code changes) and committed; rewrite
   this file at every landed chunk (rule 8).

## How to launch a weekly run today

`.venv/bin/nfl-predictor weekly --run-id <id> --xgb-device cuda` through a `launch.sh` with
`nohup setsid` (see `models/weekly_2026_week_03_full/launch.sh`); the old
`scripts/weekly_run.py` path works until 60.7 removes the shims. It reads
`config/weekly_run.yaml`, which holds the code defaults. Without `--xgb-device cuda`, stage 1
runs on the CPU at about six times the time. Do not run a weekly run or any walk-forward unless
the user asks.

## Open questions for the user

- None blocking. Merging `feat/m60-cli-consolidation` into `main` is must-ask once Milestone 60
  closes (or earlier, if the user wants the chunks on `main` between game weeks).

## Notes

- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
- Never run two walk-forwards at once; check `uptime` and `pgrep -af walk_forward` first.
- Editing under `nfl_predictor/ml/` changes every walk-forward checkpoint fingerprint (the
  fingerprint hashes every file there); the command-line code now lives in `nfl_predictor/cli/`.
