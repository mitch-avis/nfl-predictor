# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestone 60, task 56.7 and "From the 2026 Week 3
weekly run") and this file.

## State (written 2026-09-24 evening, Milestone 60 at 60.4)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.18.3`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh` exits `0` on the final tree
  (916 passed, coverage 92.55%, every step ok). The working tree is clean apart from the
  user's untracked `.agents/GPT-5-4_LightGBM_CUDA_session_transcript.md` (leave it untracked;
  markdownlint now skips `.agents/*transcript*.md`).
- Tasks 60.1-60.3 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)"); every
  sign-off decision is in `.agents/m60/PROPOSAL.md`, "Sign-off", and in the 60.4-60.9 texts.
- 60.4 landed in `0.18.2` except the SHAP test. The characterization snapshots must not change
  during a move; an intended output change rewrites them with `NFLP_UPDATE_SNAPSHOTS=1` in the
  same commit and says so in the changelog.
- `0.18.3` (today) added `lightgbm` and `shap` as dependencies, the as-needed LightGBM CUDA
  build (`nfl-lightgbm-cuda-install install | status | uv-args`; the gate's strict sync check
  gets the `uv-args` flags), and fixed `constants.QB_META_DATA_NAME` to `meta_data` (the user
  renamed the file to nfeloqb's name). NVIDIA's NCCL 2.31.2 for CUDA 13.3 is installed
  system-wide (it replaced Ubuntu's CUDA 12 NCCL), so the CUDA build needs no shim.
- Decisions made 2026-09-24:
  - keep `explain` and `shap`;
  - the numpy bootstrap rewrite is approved for the `nfl_predictor/ml/` chunk;
  - `postseason_weight` stays;
  - LightGBM stays installed (CUDA build too) and parked in Milestone 57. When that milestone
    reopens, LightGBM runs on the CPU: on this data its CUDA learner is 13x slower and not
    reproducible (`.agents/m57/lightgbm_device_check.py`).

## Week 3 weekly run (2026-09-24), for context

- The user's picks came from `models/weekly_2026_week_03_fast/`. The rerun
  `models/weekly_2026_week_03_full/` (fresh ETL and lines, GPU, default stage-1 window) agreed
  on every pick. `models/weekly_2026_week_03/` was stopped during stage 1.
- Data backups: `data/backup_pre_2026_week_03/` and `data/backup_pre_2026_week_03_full/`.
  `data/qb_meta_data.csv` is a temporary copy of `data/meta_data.csv`, no longer read since
  `0.18.3`. Deleting any of these is must-ask.
- How to launch a weekly run today, until task 56.7 lands: `.venv/bin/python
  scripts/weekly_run.py --run-id <id> --xgb-device cuda` through a `launch.sh` with `nohup
  setsid` (see `models/weekly_2026_week_03_full/launch.sh`).
  - Do **not** pass `--config config/weekly_run.yaml`: production has never read that file,
    and its values are untested.
  - Without `--xgb-device cuda`, stage 1 runs on the CPU at about six times the time.

## Your task

1. Ask the user the open questions below before anything they block.
2. Finish 60.4: the `shap_analysis` characterization test (fixture model plus fixture rows, a
   snapshot of the SHAP summary it writes).
3. Continue with 60.5 (move library code into the package, no `from scripts import` left),
   then 60.6-60.9, each chunk small, gated and committed. Group every edit under
   `nfl_predictor/ml/` into one chunk, because it changes every checkpoint fingerprint; the
   bootstrap rewrite goes in that chunk.
4. Rewrite this file at every landed chunk (rule 8).

## Open questions for the user

- Task 56.7(a): as part of the 60.6 front door, should the weekly run read
  `config/weekly_run.yaml` by default, with the YAML rewritten to today's code defaults so no
  output changes (and `off` quoted, since YAML reads a bare `off` as `false`)? The weekly
  characterization test currently loads the YAML's untested values, so this also re-points
  that snapshot at what production really runs (an intended snapshot rewrite, stated in the
  changelog). The GPU default belongs with it if the user wants it before step 3 (task 55.4).
- Whether the weekly stage 1 should score more seasons. The default `3` scores two, because
  the current season counts. The user asked about it on 2026-09-24; the recommendation is to
  settle it in task 56.7 (possibly retiring the weekly re-selection), not by widening it now.
- Deleting `data/qb_meta_data.csv`, the stopped run `models/weekly_2026_week_03/`, and the two
  data backups once the user no longer needs them.

## Notes

- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
- Never run two walk-forwards at once; check `uptime` and `pgrep -af walk_forward` first.
