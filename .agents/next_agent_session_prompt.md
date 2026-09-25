# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestone 60, task 56.7 and "From the 2026 Week 3
weekly run") and this file.

## State (written 2026-09-24 late evening, Milestone 60 at 60.6)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.18.5`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh --web` exits `0` on the final
  tree (see the `0.18.5` commits). The working tree is clean apart from the user's untracked
  `.agents/GPT-5-4_LightGBM_CUDA_session_transcript.md` (leave it untracked; markdownlint skips
  `.agents/*transcript*.md`).
- Tasks 60.1-60.4 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)"); every
  sign-off decision is in `.agents/m60/PROPOSAL.md`, "Sign-off", and in the 60.5-60.9 texts.
- `0.18.4` finished 60.4 with `tests/test_shap_analysis_characterization.py` (snapshots under
  `tests/fixtures/shap_analysis_characterization/`).
- `0.18.5` landed 60.5 (still `[ ]`, narrowed): the ranking pipeline is in
  `nfl_predictor/reporting/power_rankings.py` (`write_ranking_outputs` is the old private
  `_write_outputs`), the betting report builder in `nfl_predictor/reporting/betting_report.py`,
  and no production code imports `scripts/`. 13 test modules still import entrypoint modules
  from `scripts/`; that remainder moves with 60.6. The characterization snapshots must not
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

1. Task 60.6, in small chunks, each gated and committed: the `nfl-predictor <command>` front
   door, the retirements, the shared options module, the flag removals, the weekly run reading
   `config/weekly_run.yaml` by default (rewritten to today's code defaults, an intended weekly
   snapshot rewrite), and the step-2 follow-ups. Group every edit under `nfl_predictor/ml/`
   into one chunk, because it changes every checkpoint fingerprint; the bootstrap rewrite goes
   in that chunk. When the entrypoints have moved, `grep -rl "from scripts import" tests` must
   be empty (the 60.5 remainder).
2. Then 60.7-60.9.
3. Rewrite this file at every landed chunk (rule 8).

## Open questions for the user

- (Found in 60.4) `shap_analysis --component market` is dead: `train_blended_margin_total_model`
  always stores `market_model=None`, so the request silently analyzes the team model. Remove
  the flag in 60.6, or make it fail when no market model exists?
- (60.5, narrowed) Confirm that the test modules' `from scripts import` lines go with 60.6,
  when their entrypoints move, rather than counting as unfinished 60.5 work.

## Notes

- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
- Never run two walk-forwards at once; check `uptime` and `pgrep -af walk_forward` first.
