# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", step 3, Milestones 55 and 56, and "Open follow-ups
from completed milestones") and this file.

## State (written 2026-09-27, roadmap step 3 in progress)

- Branch `feat/step3-parity` (off `main` at `624d54f`, the user's `chore: update deps`), version
  `0.30.0`, not merged and not pushed. Merging and pushing stay must-ask.
- Landed on the branch, each implemented by an `implementer` subagent in its own worktree,
  reviewed by a separate `reviewer` subagent, fixed where the review asked, and merged with
  `--no-ff`; the delegating session wrote the changelog, version and records:
  - `0.29.0`: feature importance ranks base features by total gain; `GET /model` returns
    `{measure, rows}`; the chart names its measure (SHAP not built: a question).
  - `0.29.1`: the per-season stability view in `nfl-predictor backtest` (`metrics.stability`)
    and `nfl-predictor compare` ("Stability by season"); the "recommended defaults" half of
    task 55.6 is a question.
  - `0.30.0`: `--xgb-device auto` is the default for every XGBoost run, resolved before the
    checkpoint fingerprint and recorded in run and model metadata; a walk-forward stops rather
    than checkpoint a week whose models fell back to another device.
- Gate: `scripts/gate.sh` exits `0` on `0.29.1` (1047 passed, 92.26%); `--web` exits `0` on
  `0.29.0`. The `0.30.0` `--web` gate was running when this was written; rerun it on the final
  tree before reporting anything as done. Run gates in a separate worktree
  (`.claude/worktrees/step3-gate`, detached at the branch tip): the `--web` gate rebuilds
  `web/dist`, which the user's running server serves.
- The user runs `nfl-predictor web` (without `--reload`, PID seen 32934) from the main checkout.
  Its Python is the `0.28.6` code in memory; the checkout now has `0.30.0` code, and the rebuilt
  `web/dist` would expect the new `/model` response shape, so do not rebuild `web/dist` in the
  main checkout. Never stop or restart the server (rule 5).
- Every checkpoint fingerprint changed (edits under `nfl_predictor/ml/`); no walk-forward has run
  on the new code.

## In flight when this was written

- An implementer is working on the four calibration-window follow-ups (from the 2026-09-11
  review) plus `fitted_xgb_device` reading every head. Item 1 (walk-forward rolls the
  calibration window back across the season boundary, like production) changes early-season
  walk-forward folds; the postseason item comes back as a question. Review, merge, version
  `0.30.1` or `0.31.0` (item 1 changes benchmark output), gate.

## Step-3 evidence so far (reviewed, recorded in `.agents/benchmarks.md`, "Step 3 evidence")

- L0 (`models/step3_l0_gpu_check/`): GPU fits are bit-deterministic; GPU vs CPU moves margins
  about as much as re-seeding; fold `13` s GPU vs `64` s CPU.
- A2 (`models/step3_prob_paths/`): by the pre-written rule, only the floor blended `0.2` with
  the raw moneyline and clamped at `0.1` beats the deterministic floor (all weeks, two seeds).
  Out-of-fold isotonic is harmful; out-of-fold Platt ties the floor.
- A3 (`models/step3_open_lines/`): opening lines are measurably worse than the stored lines
  (so 56.6(d) stays in step 4), and the A2 blend does not beat the floor when its market input
  comes from the opening line (so the 56.5 choice is a question for the user).

## Next

1. Finish the calibration-window chunk (review, merge, record, gate).
2. Get the user's answers to the open questions below; 56.5's answer decides the rest of phase B
   (the production probability path, whether the stage-1 re-selection and fitted calibrators
   are retired, whether the out-of-fold pool is built, the `blend` power rankings).
3. Then phase C: the two-seed GPU reference (rungs L1-L2), six seasons from week 1, through one
   driver script, with the rule written in `.agents/TODO.md`'s step-3 plan; add the reviewer's
   suggestions (report the per-game CPU-vs-GPU move over the full window next to the GPU's own
   seed-to-seed move, and repeat the determinism check on an early-week fold).

## Open questions for the user (asked 2026-09-27, see the check-in)

- 56.5: which probability path production submits (the evidence and recommendation are in the
  check-in and in `models/step3_open_lines/REVIEW.md`).
- 55.6: what the "recommended defaults" section should be.
- Feature importance: SHAP as the headline measure, and combined vs margin-head ranking.
- 56.6(c) was scored on 2020-2025 only (the checkpoints that exist), not 2007-2025.
- `nfl-predictor compare`'s pool-point tie-break relies on float equality (A3 review, F1): a fix
  would change pool numbers slightly.
- The device chunk's two small choices (`sweep --xgb-device` added; an explicit `cuda` without a
  GPU falls back to the CPU with a warning).

## How to launch a weekly run today

`.venv/bin/nfl-predictor weekly --run-id <id>` through a `launch.sh` in the run directory, started
with `nohup setsid`; the GPU is now the default. Do not run a weekly run or any walk-forward
unless the user asks or it is a rung of the accepted step-3 ladder.

## Notes

- Never run two walk-forwards at once; check `uptime`, `nvidia-smi` (the user also runs ComfyUI on
  the GPU) and `pgrep -af "nfl-predictor (backtest|weekly|sweep)"` first.
- Re-run a plain `uv sync` after every version bump and after every merge that bumps the version.
