# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestone 60 tasks 60.7-60.9, task 56.7, and
Milestone 58 task 58.5) and this file.

## State (written 2026-09-25, Milestone 60 after closing 60.6)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.26.0`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh` exits `0` on the final tree
  (956 passed, coverage 92.14%). The working tree is clean after the `0.26.0` commits.
- Tasks 60.1-60.6 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)", which now has
  a 60.6 section summarizing `0.19.0`-`0.26.0`). Sign-off decisions: `.agents/m60/PROPOSAL.md`,
  "Sign-off". `.agents/m60/INVENTORY.md` is the signed-off 60.1 record; do not regenerate it
  over the signed-off copy.
- Characterization snapshots must not change during a move; an intended output change rewrites
  them with `NFLP_UPDATE_SNAPSHOTS=1` in the same commit and says so in the changelog.
  `tests/fixtures/cli_surface.json` changes with every flag change (intended; audit the diff).
  It held 309 actions when the milestone started and holds 215 in 15 parsers after `0.26.0`.
- Checkpoint fingerprints changed in `0.20.0`, `0.22.0` and `0.26.0`; no walk-forward has run
  since, so the next reference run retrains from scratch.

## What this session did (2026-09-25)

- `0.25.0`: one thread option. `--wf-n-jobs` and the `wf_n_jobs` config key removed (a config
  that sets it fails with a message naming `xgb_n_jobs`); `--xgb-n-jobs` sets XGBoost's CPU
  threads for stage 1 and the final fit, default every core, resolved in
  `nfl_predictor/weekly_run/config.py` (`xgb_thread_count`). The parser default stays unset so
  the CLI snapshot does not record the machine's core count. Checked before landing: the weekly
  fixture at 1 and 24 threads gave identical outputs in all eight pinned files, apart from the
  thread count in the stage-1 candidate key and its hash. The fixture keeps pinning 1 thread.
- `0.26.0`: the market model removed entirely: the `BlendedMarginTotalModel.market_model`
  field and every branch that read it, the market-only feature selection (`market_only` in the
  feature spec and the Optuna search), the `market` feature-importance component, and
  `train --tune-scope`. Findings reported to the user: `--tune-scope market` tuned nothing, but
  the default `both` halved the blend's team study budget and renamed a stored study
  `<name>_team`; a tuned blend now gets the whole timeout and keeps its name (non-production
  path; the weekly run never read the scope). All twelve saved models under `models/` are
  margin/total models. New pins in `tests/test_blended_model_paths.py`.
- 60.6 archived and removed from `TODO.md`.

## Your task

1. Task 60.7: move the web job templates (`nfl_predictor/api/jobs/catalog.py`) onto
   `nfl-predictor <command>` with canonical option names; keep the progress lines the runner
   parses (`Walk-forward fold N/M`, `WF candidate N/M`, `PROGRESS_RE` in
   `nfl_predictor/api/jobs/runner.py`) and the `weekly_run` template's config keys valid; fix the
   model-kind vocabulary (canonical `blend`, `blended_margin_total` accepted as an alias; three
   web launches fail today) test first; fix the API catch-all bug (task 58.5) test first; then
   delete the `scripts/` shims (only `scripts/gate.sh` stays). Update `tests/api/`, run
   `scripts/gate.sh --web`, amend `web_ui_plan.md` and `web/README.md`, and close 58.5.
   Wait for the user's answer on the blend power-rankings question below before touching that
   branch.
2. Task 60.8 (CI calls `scripts/gate.sh`; the gate's smoke checks use `nfl-predictor`; README,
   AGENTS.md "Repo scripts"/"Project Shape"/"Dev Workflows"/launch instructions, a "Reproducing
   an old run" note) and 60.9 (the `compare` command; it must reproduce an existing `REVIEW.md`
   rescore exactly before it is trusted).
3. Each chunk small, versioned, gated (`--web` when the web code changes) and committed; rewrite
   this file at every landed chunk (rule 8).

## How to launch a weekly run today

`.venv/bin/nfl-predictor weekly --run-id <id> --xgb-device cuda` through a `launch.sh` with
`nohup setsid` (see `models/weekly_2026_week_03_full/launch.sh`); the old
`scripts/weekly_run.py` path works until 60.7 removes the shims. It reads
`config/weekly_run.yaml`, which holds the code defaults. Stage 1 on the CPU now uses every core
by default (it used one thread before `0.25.0`); with `--xgb-device cuda` it runs on the GPU. Do
not run a weekly run or any walk-forward unless the user asks.

## Open questions for the user

- Blend power rankings (found 2026-09-25, recorded under task 60.7): `_predict_future_games`
  reads `model.feature_spec`, which a blend keeps on `team_model`, so ranking with a blend run
  has never worked, even once the vocabulary is fixed. Read the team model's spec, or reject
  blend runs for rankings with a clear message?
- Merging `feat/m60-cli-consolidation` into `main` is must-ask once Milestone 60 closes (or
  earlier, if the user wants the chunks on `main` between game weeks).

## Notes

- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
- Never run two walk-forwards at once; check `uptime` and `pgrep -af walk_forward` first.
- Editing under `nfl_predictor/ml/` changes every walk-forward checkpoint fingerprint (the
  fingerprint hashes every file there); the command-line code now lives in `nfl_predictor/cli/`.
