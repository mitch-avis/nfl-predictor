# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", step 3, Milestones 55 and 56, and "Open follow-ups
from completed milestones") and this file.

## State (written 2026-09-28, roadmap step 3 nearly done)

- Branch `feat/step3-parity` (off `main` at `624d54f`), version `0.35.0`, not merged and not
  pushed. The user said on 2026-09-28 they are fine merging when step 3 is done (picks are
  submitted Thursdays). `scripts/gate.sh --web` exits `0` on `85e96e1`.
- How the session works: each code chunk goes to an `implementer` subagent in its own worktree,
  then a separate `reviewer`, back for fixes, then merged with `--no-ff`; the delegating session
  writes the changelog, version and records, lints the Markdown it edits, and runs the gate in
  `.claude/worktrees/step3-gate`. Watchers poll a PID, never `pgrep -f` a pattern matching
  themselves.
- Landed this step (all reviewed): importance by SHAP; the stability view; `--xgb-device auto`;
  the one probability path (`0.32.0`); no calibration hold-out (`0.33.0`); settings versus
  production (`0.34.0`); the final fit uses stage 1's tree settings (`0.34.1`); the shared
  confidence-rank rule (`0.34.2`); the expanding sigma for the floor, unrounded picks and ranks,
  stage 1 from week 1 (`0.35.0`). Tasks 55.4, 55.6 and 56.5 are closed; 56.6(c) is done.
- `nfl-predictor weekly` and `train` now read the GPU reference's fold checkpoints
  (`models/step3_gpu_reference/l1_seed42`, `l2_seed7`) for the floor's sigma; a checkout without
  them must set `floor_sigma_reference_runs: []`.
- No weekly run has executed on the step-3 code yet.
- The user runs `nfl-predictor web` (without `--reload`) from the main checkout on older code in
  memory. Do not rebuild `web/dist` there; never stop or restart the server.

## Next

1. The user's call on timing (asked 2026-09-28): verify with a weekly run on the step-3 code before
   merging, and whether to merge before or after this Thursday's picks.
2. Step-3 remainder: 56.7(b) measurement part (`wf_eval_last_n_seasons`, `market_transform`;
   hypothesis and rule first, must-ask defaults); the Milestone 60 per-template live web checks
   (must-ask launches, a server without `--reload`); then the step-3 close-out.
3. Then step 4, starting with the rebuild-reproducibility follow-ups; 56.6(d) belongs there.

## Notes

- Never run two walk-forwards at once; check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- Re-run a plain `uv sync` after every version bump and after every merge that bumps the version.
