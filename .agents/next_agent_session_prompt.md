# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then this file,
`.agents/TODO.md` ("Roadmap Status", steps 6-9), `.agents/benchmarks.md` before any walk-forward,
and `models/qb_elos_measure/HYPOTHESIS.md` (main checkout; `models/` is not in git) before reading
the measurement. `.agents/step6_plan.md` and `.agents/sos_port/DECISIONS.md` hold step 6's history.

## State (written 2026-10-15, 02:00 MDT; the user asleep until about 09:30)

- `main` is at `0.59.0` plus docs commits (not pushed): the step-6 port, adopted in production.
  `data/` was rebuilt on 2026-10-14 (record in `.agents/benchmarks.md`, "Data state and ETL
  rebuild records"; backups in `~/scratch/data_backups/2026-10-14_pre_step6_merge/` and
  `2026-10-14_pre_week6/`). The Week 6 run is `models/weekly_20261015_022416_9a4c092c`.
- The user submits Week 6 picks before SEA@DEN (2026-10-15 20:15 ET). Production-changing merges
  land after the Monday game (2026-10-19), with the user's approval (rule 14). Nothing is merged
  tonight.
- The user's web server runs on `:8001` from the main checkout (no `--reload`, the `0.45.4` code
  it loaded, the old `web/dist/`). Never stop or restart it, and don't build `web/dist/` on `main`
  while it runs; `scripts/gate.sh --web` and a browser look at the new pages wait until the user
  restarts it.
- `../nfeloqb` fork `main` is at `9bf32f2` (pushed): next-week projections now find a team stored
  under a legacy code (`OAK` for Las Vegas), which had inflated Las Vegas's `qb_adj` to `+160`.
  nfeloqb was not rerun; the user runs it each week (memory `nfeloqb-weekly-process`).
- **The `qb_adj` review is done.** `feat/qb-adj-features` (`0.60.0`, `c637fc0`, harness worktree
  `.claude/worktrees/agent-afad3cf465d5bcba7`) passed an independent review against
  `.agents/qb_adj_task.md` with minor findings, all docs and tests. They are fixed on
  `fix/qb-adj-review-findings` at `6bacd5e` (nine commits on `c637fc0`, worktree
  `~/scratch/qb_adj_docs`) and went through a re-check reviewer, whose one question (is the
  rule-11 approval on record?) is answered under "Decisions on record":
  - the reproduction commands in `.agents/guardrails.md` and `.agents/walk_forward_runbook.md`
    add `qb_adj` to the group lists recorded before `0.60.0`;
  - AAA labels; a season-build test carrying `qb_adj` (`tests/` fixture regenerated);
  - the changelog's order and its reproduction note; the rule-11 paragraph in
    `.agents/benchmarks.md` ("The `qb_adj` columns and these references").
  `src/`, `pyproject.toml` and `uv.lock` are identical to `c637fc0`, so the measurement below
  measures the reviewed code. `scripts/gate.sh` on `6bacd5e` (detached in `~/scratch/qb_adj_review`,
  its own venv, `-n 8`): exit `0`, 3,627 tests, 95.01% coverage
  (`~/scratch/gate_qb_adj_6bacd5e.log`). `feat/qb-adj-features` stays at `c637fc0`: moving it to
  `6bacd5e` is a fast-forward, which needs `git merge` and so waits for the user's approval.
- The review's adversarial tests stay untracked in
  `~/scratch/qb_adj_review/tests/test_qb_adj_review_adversarial.py`: 16 pass, 3 fail.
  - Two fail in `fill_future_qb_data` (the inherited `game_utils._fill_side_qb` behaviour): a
    played row with a named starter and a null adjustment gets the latest starter's value, and
    `qb_adj_diff` does not follow the filled sides. Fixing it changes upcoming-week feature values
    (rule 5), so it is on the user's list.
  - One: `week_rows._merge_elo` leaves `_right` duplicates when the frame already carries the
    columns. Its only call site (`utils/polars/week_rows.py:552`) never passes them, and the Elo
    columns behave the same way: no action proposed.
- **The grouped measurement is running** (`models/qb_elos_measure/`: `build.sh`, `check_cuts.py`,
  `driver.sh`, `launch.sh`, `build_record.txt`, `launch_record.txt`). Worktree
  `~/scratch/qb_measure` is detached at `c637fc0` with its own venv and `data/`; the main
  checkout's `data/` was only read.
  - Builds done: new cut `b9772e3e...`, old cut `b59366e2...`, 7,261 rows and 544 columns each,
    leakage audits ok. `inputs/cut_diff.json` is ok: 12 `elo` columns differ, only in 2023-2025
    rows (from 2023-09-07); the `qb_adj` columns are identical in the two builds. The hypothesis
    stands unamended (sha256 `d682a0c0...`).
  - The driver started 2026-10-15 01:37:13 (pid `903662`): old file, new file, `qb_adj`, strength,
    each with seeds 42 then 7, one at a time on the GPU. Each run writes
    `models/qb_elos_measure/<arm>_seed<seed>/` (`run.log`, `metrics_report.json`); `driver.log`
    has a `start` and an `end ... exit ... seconds` line per run and `driver done` at the end. The
    first run took about 32 minutes (fold 53 of 328 at 307 s), so all eight should end near 06:00.
  - A failed run is counted and the driver moves on; `./launch.sh driver c637fc0` (from
    `models/qb_elos_measure/`) reruns the ladder and finished runs resume from their checkpoints.
    To stop it: `kill 903662`, then `pgrep -af "nfl-predictor backtest"` and `kill` that PID
    (never `pkill -f`, which matches the watching shells).
- **Step 3, the "no game lines" groups change**, is with an `implementer` subagent in its own
  harness worktree, stacked on `c7eb30f` (an earlier commit of `fix/qb-adj-review-findings`), as
  `0.61.0`, pytest at `-n 8`. The contract: an explicit group list (command line or the config
  key) always wins; otherwise the configuration without game lines (`--wf-market-mode none`, or
  the three market options off) leaves out `strength,elo,own_elo_wt,qb_adj`, and the shipped
  `config/weekly_run.yaml` stops pinning the key. Production's default run does not change.

## Next, in order

1. **Watch the measurement.** Check `driver.log` after each transition; on a failed run, read its
   `run.log` before resuming. Never start another walk-forward or ETL while it runs.
2. **Step 3:** when the implementer reports, rebase its branch onto `6bacd5e` (the commits since
   `c7eb30f` are docs only), run its gate (`VIRTUAL_ENV` set to its venv, `-n 8`), then a
   `reviewer` in a fresh context (adversarial tests, uncommitted), fixes through an implementer
   (test first) and a re-check. Nothing merged.
3. **After the eight runs:** a `reviewer` rescores from the fold checkpoints and writes
   `models/qb_elos_measure/REVIEW.md` (rule 3) before any number reaches a doc. Then the
   check-in (rule 7) with the three comparisons of `HYPOTHESIS.md`, applied as written (rule 12):
   - the new file against the old (seasons 2023-2025, all weeks): no Brier or log-loss loss keeps
     it;
   - the `qb_adj` group against the new file (all weeks): a Brier win with no log-loss loss
     recommends turning it on (the user's call, between game weeks);
   - the strength snapshot against the ported ratings (all weeks): a Brier or log-loss win for
     the snapshot goes to the user.
   Every interval that excludes zero, with each arm's rank on every column (rule 9).
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
     `TODO.md`; they and the open P3s from the integration reviews are in an earlier handoff,
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
- Overnight session rules (2026-10-15, 01:20 MDT, the user asleep for about eight hours):
  - a review finding that needs the user holds the measurement only if its answer could change
    the `qb_adj` column values, the default feature groups or the backtest command; anything
    else goes on the morning question list and the runs proceed;
  - if the build check finds old-file/new-file differences beyond the 2023-2025 rows of the
    `elo` and `qb_adj` columns, an amendment to `HYPOTHESIS.md` (new checksum) proceeds when
    every extra difference traces to `qb_elos.csv`; anything unrelated (pre-2023 rows, market
    lines, non-Elo stats) holds;
  - step 3 (the "no game lines" groups change) may start during the runs, stacked on
    `feat/qb-adj-features`, pytest capped at `-n 8`, nothing merged;
  - the production-settings difference with the floor-sigma reference runs is recorded as
    harmless in `.agents/benchmarks.md` until the reference is replaced (rule 11 approved).

## Open for the user

- Approve fast-forwarding `feat/qb-adj-features` to `6bacd5e`, together with its merge after the
  Monday game.
- Step 3's list includes `qb_adj` (`strength,elo,own_elo_wt,qb_adj`), where the 2026-10-14
  approval named `strength,elo,own_elo_wt`. Recommended: keep it, since `qb_adj` is off in
  production until measured and the `--wf-market-mode` help already advises this list; dropping
  it is a one-line change. If the measurement turns `qb_adj` on, the list follows.
- The inherited `_fill_side_qb` behaviour (above): a fix changes upcoming-week feature values.
- After the measurement: replace the floor-sigma reference runs (a default change, per
  `HYPOTHESIS.md`).
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
  `pgrep -af "nfl-predictor (backtest|weekly|data)"` first.
- A backtest that judges probabilities runs from 2007 week 1 or passes
  `--floor-sigma-reference-runs` with the GPU reference, and passes `--wf-start-week 1`.
- Worktrees: `~/scratch/qb_measure` (detached at `c637fc0`, the measurement's code and `data/`),
  `~/scratch/qb_adj_docs` (`fix/qb-adj-review-findings`), `~/scratch/qb_adj_review` (detached at
  `6bacd5e`, the review's adversarial file), `~/scratch/step6` (`feat/step6-sos-port`, merged)
  and `~/scratch/step6_runs` (detached at `d6064eb`, the step-6 runs' code); the
  `.claude/worktrees/agent-*` worktrees are the harness's, leave them.
- Elo research is in `~/scratch/elo_research/`; `nfl-sos-ratings` is frozen at `9081bce`, read
  only as `git -C ../nfl-sos-ratings show 9081bce:<path>`.
