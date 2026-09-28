# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", step 3, Milestones 55 and 56, and "Open follow-ups
from completed milestones") and this file.

## State (written 2026-09-28, roadmap step 3 in progress)

- Branch `feat/step3-parity` (off `main` at `624d54f`), version `0.33.0`, not merged and not
  pushed. Merging and pushing stay must-ask. `scripts/gate.sh --web` exits `0` on `43d2a7d`
  (1092 passed).
- How the session works: each code chunk goes to an `implementer` subagent in its own worktree,
  then to a separate `reviewer` subagent, back to the implementer for fixes, then merged with
  `--no-ff`; the delegating session writes the changelog, version, `TODO.md`, `ARCHIVE.md`,
  `AGENTS.md` and this file, lints the Markdown it edits (`markdownlint` on the files), and runs
  the gate in `.claude/worktrees/step3-gate` (detached at the branch tip), because the `--web`
  gate rebuilds `web/dist`, which the user's running server serves.
- Landed on the branch (all reviewed): `0.29.0` total-gain importance; `0.29.1` the per-season
  stability view (task 55.6, closed); `0.30.0` `--xgb-device auto` default; `0.30.1` the
  calibration-window follow-ups; `0.31.0` SHAP headline importance (XGBoost TreeSHAP);
  `0.32.0` the one probability path (task 56.5 code: every run type submits the deterministic
  floor; the stage-1 matrix, fitted and Elo calibrators, market blend and clamp, the `blend`
  kind, `sweep` and uncertainty-aware probabilities are retired); `0.32.1` the `shap` dependency
  dropped; `0.33.0` the final fit trains on every completed game (no calibration hold-out), and
  the walk-forward's unused calibration frame and options are retired.
- The user runs `nfl-predictor web` (without `--reload`) from the main checkout; its Python is
  older code in memory. Do not rebuild `web/dist` there, never stop or restart the server.

## Running when this was written

- The GPU reference (rungs L1-L2), `models/step3_gpu_reference/` (`HYPOTHESIS.md` with the
  rules R-ref, R-5.5, R-6.6 and the launch record, `driver.sh`, `driver.log`), launched
  2026-09-28 00:12 on commit `43d2a7d`: determinism legs `det_a`/`det_b` (the 2025 season), then
  `l1_seed42` and `l2_seed7` (2007-2025 from week 1). About 6 s per fold on the GPU. Do not merge
  anything into the main checkout until `driver.log` says `driver done` (L2 would load different
  code than L1). If the driver stopped, rerun `driver.sh` (the L1/L2 arms resume from their fold
  checkpoints; the determinism legs use `--no-resume`).

## Next

1. When the driver finishes: check `det_a` against `det_b` (identical on every fold), then
   write the check-in with the governing numbers of R-ref, R-5.5 and R-6.6 (rule 7), and have an
   independent `reviewer` rescore from the fold checkpoints (rule 3) before any number enters the
   docs. Compare against the CPU task 55.8 unweighted pair on 2020-2025 with `nfl-predictor
   compare` (it lists the dropped `calibration_weeks` key and the device as config differences).
2. Close task 55.4 (the GPU reference) and task 56.5 (if R-5.5 confirms the floor) with the
   user; 56.6(c) on 2007-2025.
3. Remaining step 3: task 56.7(b) (the "settings versus production" report section and the
   remaining settings: `wf_eval_last_n_seasons`, `market_transform`), the Milestone 60 per-template
   live web checks (must-ask launches, on a server without `--reload`), then the step-3
   close-out and the merge question for the user.

## Open questions for the user

- None pending. Recent answers (2026-09-27/28): floor everywhere; 55.6 closed with a "settings
  versus production" section moved to 56.7(b); SHAP headline with total gain secondary, combined
  ranking, margin toggle later; 56.6(c) on 2007-2025 with new runs; the pool tie-break stays
  as is; GPU free (ComfyUI closed); production stops holding weeks out; `shap` dropped.

## Notes

- Never run two walk-forwards at once; check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- Re-run a plain `uv sync` after every version bump and after every merge that bumps the version.
