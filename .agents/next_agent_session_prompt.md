# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then this file,
`.agents/TODO.md` ("Roadmap Status", steps 6-9), `.agents/benchmarks.md` before any walk-forward,
and `models/qb_elos_measure/HYPOTHESIS.md` (main checkout; `models/` is not in git) before the
builds. `.agents/step6_plan.md` and `.agents/sos_port/DECISIONS.md` hold step 6's history.

## State (written 2026-10-15, 00:30 MDT)

- `main` is at `0.59.0` plus docs commits (not pushed): the step-6 port, adopted in production.
  `data/` was rebuilt on 2026-10-14 (record in `.agents/benchmarks.md`, "Data state and ETL
  rebuild records"; backups in `~/scratch/data_backups/2026-10-14_pre_step6_merge/` and
  `2026-10-14_pre_week6/`). The Week 6 run is `models/weekly_20261015_022416_9a4c092c`.
- The user submits Week 6 picks before SEA@DEN (2026-10-15 20:15 ET). Production-changing merges
  land after the Monday game (2026-10-19), with the user's approval (rule 14).
- The user's web server runs on `:8001` from the main checkout (no `--reload`, the `0.45.4` code
  it loaded, the old `web/dist/`). Never stop or restart it, and don't build `web/dist/` on `main`
  while it runs; `scripts/gate.sh --web` and a browser look at the new pages wait until the user
  restarts it.
- `../nfeloqb` fork `main` is at `9bf32f2` (pushed): next-week projections now find a team stored
  under a legacy code (`OAK` for Las Vegas), which had inflated Las Vegas's `qb_adj` to `+160`.
  nfeloqb was not rerun; the user runs it each week (memory `nfeloqb-weekly-process`).
- `feat/qb-adj-features` (`0.60.0`, three commits on `e576b91`: `a403f57`, `d4ef695`,
  `c637fc0`), worktree `.claude/worktrees/agent-afad3cf465d5bcba7` (the harness's; inspect it
  with `git -C`). The implementer reports `scripts/gate.sh` exit `0` (3,626 tests, 95.01%
  coverage), not yet re-run by anyone else. The contract is `.agents/qb_adj_task.md`. What it
  did: `away_qb_adj`, `home_qb_adj`, `qb_adj_diff` from `qb_elos.csv`, group `qb_adj` (not
  market-informed), in `PRODUCTION_DISABLED_FEATURE_GROUPS`; the diff comes from
  `teamrankings.calculate_stat_differentials`. Its notes for the review:
  - beyond the contract's files it edited help text in `cli/options.py`, `cli/backtest.py`,
    `cli/train.py`, `ml/feature_spec.py` (a message string; any `ml/` edit changes checkpoint
    fingerprints), `.agents/guardrails.md` and `.agents/walk_forward_runbook.md` (the group list);
  - the collision test passed before the change (a guard, not red-then-green);
  - the weekly stage hashes changed, so a weekly run directory retrains stage 1 rather than
    resuming;
  - until `data/` is rebuilt, every run warns that `qb_adj` matches no column;
  - `compare`'s production-settings section (`reporting/production_settings.py`) compares group
    lists by name, so the floor-sigma reference runs (`models/step6/ratings_anchored_seed*`, the
    three-group list) now show a difference from production although their data had no
    `qb_adj` column. Proposed: record it in `.agents/benchmarks.md` as harmless (rule 11 needs
    the user's approval), and replace the reference after the measurement.
- The GPU showed 30% use and 795 MiB at 00:21 with no project job running; check `nvidia-smi`
  before a launch.

## Next, in order

1. **Review `feat/qb-adj-features`** with a `reviewer` in a fresh context: the contract in
   `.agents/qb_adj_task.md`, the notes above, re-run the gate in a detached scratch worktree with
   its own synced venv (`VIRTUAL_ENV` set to it), adversarial tests (uncommitted). Also check
   whether anything in the weekly run, `train` or `backtest` stops or warns on the
   production-settings difference with the floor-sigma reference runs. Findings go back to an
   implementer (test first), then a re-check. Don't merge: merging is must-ask and waits until
   after the Monday game.
2. **The grouped measurement** (accepted by the user 2026-10-15; `HYPOTHESIS.md` checksum in
   `HYPOTHESIS.md.sha256`, `d682a0c0...`). Once the review approves:
   - write `models/qb_elos_measure/build.sh`, `driver.sh` and `launch.sh` after
     `models/step6/` (its `build.sh`, `driver.sh`, `launch.sh`): a detached worktree such as
     `~/scratch/qb_measure` at the reviewed commit, with its own synced venv and its own `data/`
     (the main checkout's upstream inputs copied in with a sha256 manifest; the main checkout's
     `data/` is never touched). Build the new file first, cut it to `inputs/new_file.csv`, then
     put `~/scratch/data_backups/2026-10-14_pre_week6/qb_elos.csv` in place, build again
     (full builds, not `--incremental`), cut to `inputs/old_file.csv`; each with its sha256, the
     pick-time line check and the leakage audit. Script the check that the two cuts differ only
     in 2023-2025 rows of the `elo` and `qb_adj` columns;
   - the eight runs, one at a time through `driver.sh`, launched with `nohup setsid` (check
     `uptime`, `nvidia-smi` and `pgrep -af "nfl-predictor (backtest|weekly|data)"` first);
     write the launch record before the launch and check the driver log after its first
     run-to-run transition;
   - afterwards an independent `reviewer` rescores from the fold checkpoints and writes
     `REVIEW.md` (rule 3) before any number reaches the docs; then the check-in (rule 7) and the
     decisions the rules hand to the user.
3. **The "no game lines" groups change** (approved 2026-10-14): the configuration without game
   lines (`--wf-market-mode none` and the three market options) leaves out
   `strength,elo,own_elo_wt`, the reviewed list of `models/step6/elo_marketfree_seed{42,7}`. Test
   first through an implementer and a reviewer, on a branch cut after `feat/qb-adj-features`
   merges (both edit the group lists in `constants.py`) or stacked on it. `qb_adj` is
   market-free, so it is not in that list.
4. **`../nfeloqb`, a backup's first start** (approved 2026-10-15, test first): the next-week
   projection rates the starter with `current_value`, while the played row uses
   `get_value(team)`, which applies the backup rule (Baltimore's Week 5 projection was off by
   `0.8`). Read `get_value` first and confirm it uses only pre-game information; then a failing
   test in `tests/test_elo_qb_adjustment.py`, the fix in `elo_file_constructor.py`, the fork's
   gates (`.venv/bin/ruff format .`, `ruff check .`, `ty check .`, `pyright .`,
   `markdownlint-cli2 "**/*.md" "#.venv"`, `.venv/bin/python -m pytest`), a `fix:` commit, and a
   push to the fork (`GUARD_OK=1`). Before the user's next nfeloqb run (Week 7).
5. **The fork's win-total file** (approved 2026-10-15 as its own measurement after the grouped
   one, landing between game weeks): `nfeloqb/Manual Data/wt_ratings.csv` ends at 2023, so the
   QB-adjusted Elo has had no win-total pull since 2024 (`elo.py` line ~136). Hypothesis and
   rule first, a fork change, a third build against the grouped measurement's new-file build.
6. **The step-6 docs close-out and the roadmap tidy** (approved 2026-10-15):
   - regenerate `RANKINGS_EVAL.md` under `models/step6/rankings/` (a `models/` write: ask);
   - rename "market-free" to "no game lines" where it means the configuration with Elo,
     including `weekly_run/stage1.py:63,68`;
   - refresh the stale "Review:" fields in `.agents/sos_port/decisions/*.md`; commit the
     hypothesis checksums;
   - record the 2026-10-11, 2026-10-12 and 2026-10-14 decisions in `step6_plan.md` and
     `TODO.md`; they and the open P3s from the integration reviews are in the previous handoff,
     `git show 5b73686:.agents/next_agent_session_prompt.md` ("Decisions on record", "Open for
     the user" and the integration-progress bullets);
   - in `TODO.md`: refresh "Current validated baseline" (it still says `0.28.1`); mark step 6's
     phases done; retire the follow-ups that concern only the strength snapshot if the grouped
     measurement keeps the ported ratings (rule 5: the user approves each closure);
   - build the approvals list and the user's final HTML report for step 6.
7. `scripts/gate.sh --web` on `main` and a browser look once the user restarts the server; the
   2026 roof recheck before the Paris game on 2026-10-25
   (`.agents/sos_port/decisions/pbp_columns.md`).

## Decisions on record (2026-10-15)

- `../nfeloqb` Las Vegas fix: approved and done (`9bf32f2`).
- Adopt `qb1_adj`/`qb2_adj` as features, off in production until measured: approved.
- The grouped measurement: approved, with its ladder (two builds, eight runs).
- The backup first-start fork fix: approved, test first, after the picks.
- The stale win-total file: its own measurement after the grouped one, between game weeks.
- The roadmap: steps 7 (the Elo inputs) and 8 (the feature ideas, explored quickly, then
  accepted or dropped) go before the re-tune, now step 9 (`TODO.md`, "Roadmap Status").
- Week 6: the top-decile ranking is the two-seed mean (2026-10-14, in DECISIONS.md).

## Open for the user

- The stats-only model: proposed 2026-10-15, not decided. A named track rather than one more
  step at the end, in three parts:
  - in step 8, measure each feature idea on the configuration without game lines as well as on
    production's, since the market already prices injuries and weather;
  - in step 9, tune it as its own study, since its best settings likely differ from the
    anchored model's;
  - a weekly shadow prediction (logged, never submitted), scored against the closing line, so it
    builds a live record.
  On the step-6 references it trails the pick-time line by `+0.00373` Brier with the own Elo and
  `+0.00466` without any Elo (`.agents/benchmarks.md`, "Step 6 reference arms").
- 58.1's exact survivor planner in place of the plan's beam search (acceptance pending).
- Merging `feat/qb-adj-features` after the review, after the Monday game.
- The production-settings difference with the floor-sigma reference runs (above).

## How the last sessions worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer` in a
  fresh context (adversarial tests, uncommitted), fixes back to an implementer, a re-check by a
  reviewer. A `--no-ff` merge (the commit hook needs a Conventional Commits subject on merge
  commits; `git merge` needs `GUARD_OK=1`, only after the user's approval). The delegating
  session writes `TODO.md`, `ARCHIVE.md`, `AGENTS.md`, `.agents/benchmarks.md` and this file,
  and lints the Markdown it edits.
- Reviewers get a detached scratch worktree with a synced venv, cap pytest with
  `PYTEST_ADDOPTS="-n 8"` when several run at once, and write only their report; at most four at
  once. Every measured number goes through an independent rescore before it enters a doc.
- Run a worktree's gate with `VIRTUAL_ENV` set to that worktree's `.venv`. A backgrounded gate
  wrapper must `exit` the gate's code, never echo after it.
- The bash-guard hook blocks commands containing `git merge` in the main checkout (including
  `merge-tree`) and bare `python3`; use `git diff A...B`, `git rev-list --count A..B` and
  `.venv/bin/python`.
- Inspect another worktree with `git -C <path>` or absolute paths. Run
  `.venv/bin/pre-commit run --all-files` before ending a turn. Stop at a natural point well
  before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once. Check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- A backtest that judges probabilities runs from 2007 week 1 or passes
  `--floor-sigma-reference-runs` with the GPU reference, and passes `--wf-start-week 1`.
- Worktrees: `~/scratch/step6` (`feat/step6-sos-port`, merged) and `~/scratch/step6_runs`
  (detached at `d6064eb`, the step-6 runs' code); the `.claude/worktrees/agent-*` worktrees are
  the harness's, leave them.
- Elo research is in `~/scratch/elo_research/`; `nfl-sos-ratings` is frozen at `9081bce`, read
  only as `git -C ../nfl-sos-ratings show 9081bce:<path>`.
