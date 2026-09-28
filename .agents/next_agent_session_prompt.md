# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", step 3, Milestones 55 and 56, and "Open follow-ups
from completed milestones") and this file.

## State (written 2026-09-28, roadmap step 3 nearly done)

- Branch `feat/step3-parity` (off `main` at `624d54f`), version `0.34.1`, not merged and not
  pushed. Merging and pushing stay must-ask. `scripts/gate.sh --web` exits `0` on `7c96531`.
- How the session works: each code chunk goes to an `implementer` subagent in its own worktree,
  then to a separate `reviewer`, back for fixes, then merged with `--no-ff`; the delegating
  session writes the changelog, version and records, lints the Markdown it edits, and runs the
  gate in `.claude/worktrees/step3-gate` (the `--web` gate rebuilds `web/dist`, which the user's
  running server serves). Watchers for long runs must poll a PID, not `pgrep -f` a pattern that
  matches the watcher's own command line (that stalled the 2026-09-28 check-in for 7 hours).
- Landed (all reviewed): `0.29.0`-`0.31.0` importance (total gain, then SHAP headline), the
  stability view (55.6 closed), `--xgb-device auto`, the calibration-window follow-ups;
  `0.32.0` the one probability path (the floor; stage-1 matrix, calibrators, market blend,
  `blend` kind, `sweep` retired); `0.32.1` `shap` dependency dropped; `0.33.0` no calibration
  hold-out; `0.34.0` the settings-versus-production report section (56.7(b) code);
  `0.34.1` the final fit uses stage 1's tree settings.
- The GPU reference (`models/step3_gpu_reference/`, 2007-2025, seeds 42 and 7) is recorded and
  reviewed (`.agents/benchmarks.md`, "GPU reference"); task 55.4 is closed.
- The user runs `nfl-predictor web` (without `--reload`) from the main checkout on older code in
  memory. Do not rebuild `web/dist` there; never stop or restart the server.

## Open question for the user (asked 2026-09-28)

- Task 56.5, rule R-5.5: on 2007-2025 the retired blend toward the stored moneyline beats the
  floor (Brier `-0.00072` `[-0.00105, -0.00041]`), the same blend built from the opening spread
  does not (`+0.00010` `[-0.00051, +0.00071]`). Recommendation (both the session and the
  reviewer): keep the floor, close 56.5 on it, and revisit with 56.6(d) in step 4.

## Next

1. The user's R-5.5 answer: close 56.5 (floor) or restore a fixed blend (a new chunk).
2. 56.6: record (c) as done on 2007-2025 (reviewed numbers in `benchmarks.md`); (d) is step 4.
3. 56.7(b) measurement part: the remaining output-changing settings (`wf_eval_last_n_seasons`,
   `market_transform`) need a hypothesis and rule first, and are must-ask as default changes.
4. The Milestone 60 per-template live web checks (must-ask launches, a server without
   `--reload`).
5. Step-3 close-out: every step-3 item in `TODO.md` resolved or moved, then ask the user to
   merge `feat/step3-parity` into `main` (between game weeks).

## Notes

- Never run two walk-forwards at once; check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- Re-run a plain `uv sync` after every version bump and after every merge that bumps the version.
