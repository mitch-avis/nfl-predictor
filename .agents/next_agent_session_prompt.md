# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestone 60, task 56.7 and "From the 2026 Week 3
weekly run") and this file.

## State (written 2026-09-24 late evening, Milestone 60 at 60.5)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.18.4`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh` exits `0` on the final tree
  (see the `0.18.4` commit). The working tree is clean apart from the user's untracked
  `.agents/GPT-5-4_LightGBM_CUDA_session_transcript.md` (leave it untracked; markdownlint skips
  `.agents/*transcript*.md`).
- Tasks 60.1-60.4 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)"); every
  sign-off decision is in `.agents/m60/PROPOSAL.md`, "Sign-off", and in the 60.5-60.9 texts.
- `0.18.4` finished 60.4 with `tests/test_shap_analysis_characterization.py` (snapshots under
  `tests/fixtures/shap_analysis_characterization/`). The characterization snapshots must not
  change during a move; an intended output change rewrites them with
  `NFLP_UPDATE_SNAPSHOTS=1` in the same commit and says so in the changelog.
- Decisions made 2026-09-24 (all recorded in `TODO.md` or `ARCHIVE.md`):
  - keep `explain` and `shap`; the numpy bootstrap rewrite goes in the `nfl_predictor/ml/`
    chunk; `postseason_weight` stays; LightGBM stays installed and parked (Milestone 57, CPU
    only when it reopens);
  - task 56.7(a): the weekly run reads `config/weekly_run.yaml` by default, with the YAML
    rewritten to today's code defaults (quote `off` as `"off"`), in the 60.6 front-door chunk.
    That re-points the weekly characterization snapshot at what production runs: an intended
    snapshot rewrite, stated in the changelog. The user stressed that the YAML's settings still
    need optimizing; that is task 56.7(b) and 55.4 in step 3, not part of the rewrite;
  - how many seasons weekly stage 1 scores is settled in task 56.7, not by widening it now;
  - `data/qb_meta_data.csv`, `models/weekly_2026_week_03/` and the backups
    `data/backup_pre_2026_week_03{,_full}/` were deleted with the user's approval.

## How to launch a weekly run today (until 60.6 lands)

`.venv/bin/python scripts/weekly_run.py --run-id <id> --xgb-device cuda` through a `launch.sh`
with `nohup setsid` (see `models/weekly_2026_week_03_full/launch.sh`). Do **not** pass
`--config config/weekly_run.yaml` until 60.6 has rewritten it: its current values are untested.
Without `--xgb-device cuda`, stage 1 runs on the CPU at about six times the time. Do not run a
weekly run or any walk-forward unless the user asks.

## Your task

1. Task 60.5: move library code out of `scripts/` into the package (power-rankings computation
   and output writing into `nfl_predictor/reporting/`, the betting-report builder likewise),
   with the scripts temporarily calling the package, and drop every `from scripts import`
   (production and tests; `grep -rn "from scripts import\|import scripts" nfl_predictor tests`
   lists them). Every characterization snapshot stays unchanged.
2. Then 60.6-60.9, each chunk small, gated and committed. Group every edit under
   `nfl_predictor/ml/` into one chunk, because it changes every checkpoint fingerprint; the
   bootstrap rewrite goes in that chunk.
3. Rewrite this file at every landed chunk (rule 8).

## Open questions for the user

- (Found in 60.4) `shap_analysis --component market` is dead: `train_blended_margin_total_model`
  always stores `market_model=None`, so the request silently analyzes the team model. Remove
  the flag in 60.6, or make it fail when no market model exists? Not blocking 60.5.

## Notes

- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
- Never run two walk-forwards at once; check `uptime` and `pgrep -af walk_forward` first.
