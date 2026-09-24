# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status" and Milestone 60) and this file.

## State (written 2026-09-24, Milestone 60 stopped at 60.3 for sign-off)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25` (version `0.18.0`, the
  merged task 55.8 close-out). Version `0.18.1` on the branch adds the inventory chunk only; no
  production code, config or test has changed. The branch is not pushed; merging and pushing
  are must-ask.
- `scripts/gate.sh` exited `0` on `main` at the start (`889 passed`, coverage `92.45%`) and on
  the branch's final tree before the commits (see the check-in for the numbers).
- Roadmap step 2 (Milestone 60 with task 55.5) is in progress. Tasks 60.1 and 60.2 are done as
  generated inventories and **left unchecked** until the user signs off at 60.3:
  - `.agents/m60/inventory.py` generates `.agents/m60/INVENTORY.md` and `inventory.json`
    (deterministic; exits non-zero on any failed search or overturned judgment). Hand
    judgments live in `.agents/m60/annotations.yaml` and are re-checked on every run.
  - `.agents/m60/scripts_coverage.py` generates `.agents/m60/SCRIPTS_COVERAGE.md` (runs the
    whole suite with coverage over `scripts/`, about 2 minutes).
  - `.agents/m60/PROPOSAL.md` holds the proposed dispositions and the eleven sign-off
    questions. It was revised the same day after the user's questions: ScoreModel stays and
    gets a fair model-family comparison after step 3 (section 6); the Excel betting workbook
    retires end to end; one `nfl-predictor <command>` front door (section 0); CI to call
    `scripts/gate.sh`; a new `compare` command; how success is measured (section 8).
- `.agents/m60_cli_flag_audit.md` is superseded; never use it as a source.
- `models/` and `data/` are untouched. `.agents/gpt-5-4_task_55-8_transcript.md` is the user's
  untracked file; leave it alone.

## Your task

1. If the user has answered the sign-off questions in `.agents/m60/PROPOSAL.md`, record each
   answer in `annotations.yaml` (dispositions) and `TODO.md` (60.3), regenerate the inventory,
   tick 60.1-60.3, and start 60.4: the characterization tests, test-first, against the current
   layout, green before anything moves. The weekly-run output test pins predictions, confidence
   picks, the betting report, power rankings and standings on fixed fixture inputs; add a
   `--help` snapshot per surviving entrypoint and tests for the library functions about to move.
   Budget coverage per move from `SCRIPTS_COVERAGE.md`: moving the proposed files with today's
   tests would take the package close to its `90%` floor.
2. If the user has not answered, do not move code. Answer questions about the proposal from
   the generated files only.
3. Rewrite this file at every landed chunk (rule 8).

Constraints:

- Behavior-preserving: no move may change a prediction, probability, pick, ranking or report.
  Behavior defects found are listed in `PROPOSAL.md` and decided separately; never fold one
  into a move (the model-kind vocabulary defect and the `walk_forward_backtest --calibration`
  default are questions 7 and 8 there; the stage-1 list-order selection is task 56.5, step 3).
- Mid-season: the weekly run must work every week while the milestone is open; land in small
  chunks.
- No walk-forward runs in this milestone. Edits under `nfl_predictor/ml/` change every
  checkpoint fingerprint; group them into one chunk.
- Changing any default, merging, pushing, deleting under `data/` or `models/`, and touching the
  web API on port 8765 are must-ask.

## Open questions for the user (carry forward)

- The eleven sign-off questions in `.agents/m60/PROPOSAL.md`, with the recommendations given in
  the 2026-09-24 check-in.
- Where the model-family comparison goes in the roadmap (a new milestone after step 3, with the
  tuned final inside step 5, is the recommendation; reordering the roadmap is must-ask).
- Where survivor picks should live (task 58.1 plans them for the web UI; the user wants the
  weekly run to produce every pick type).
- Delete the unreferenced checkpoint folder `models/wf_checkpoints/ca3e6892daacd42980f4/` (left
  by a stray default-config walk-forward on 2026-09-23)?
- Prune the stale `../nfl-predictor-web` worktree record (`git worktree list` marks it
  prunable), and where the web app will run from.
- How production probabilities should be formed (task 56.5, roadmap step 3).

## Notes

- The Week 3 weekly run can go whenever the user wants; do not run it yourself unless asked,
  and never alongside another walk-forward.
- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
