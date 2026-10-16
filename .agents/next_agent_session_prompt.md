# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then this file,
`.agents/TODO.md` ("Roadmap Status", steps 6-9) and `.agents/benchmarks.md` before any
walk-forward ("`qb_elos.csv` measurement" is the newest section). `.agents/step6_plan.md` and
`.agents/sos_port/DECISIONS.md` hold step 6's history.

## State (written 2026-10-15, 23:40 MDT)

- **Branch `fix/no-game-lines-sigma-reference` (`0.63.0`, head `611945b`) is ready and waits
  for the user's approval to merge** (`GUARD_OK=1 git merge --no-ff`, then the CUDA sync and the
  gate on `main`). The main checkout is on this branch. What it does (the user chose this fix
  2026-10-15): a run without game lines (the weekly run's `--wf-market-mode none`, or `train`
  without market features and anchor) defaults its floor-sigma reference to
  `constants.NO_GAME_LINES_FLOOR_SIGMA_REFERENCE_RUNS`
  (`models/qb_adj_no_lines/no_lines_seed{42,7}`) instead of the market-anchored production
  runs; a listed value still wins in every mode;
  `options.resolve_floor_sigma_reference_runs` mirrors the feature-group default, and
  `config/weekly_run.yaml` leaves the key out so it follows the mode. Production (mode hybrid)
  resolves to the same runs and settings as before; `constants.py` changed, so every
  walk-forward checkpoint fingerprint moved. Tests first (they fail on `3bde4dd`); independent
  review approved with four P3 doc findings, all fixed on the branch; the reviewer's 25
  adversarial tests (all pass, uncommitted) are in
  `~/scratch/sigma_review/test_adversarial_sigma_reference.py`. Gate exit `0` on `611945b`,
  4,406 tests, `src/` coverage 96.32%. The branch also closes the follow-up (`TODO.md`,
  `ARCHIVE.md`, `.agents/benchmarks.md`).
- **`main` is at `0.62.2`** (`3bde4dd`, not pushed, no tags): python-template `0.1.4` (tagged by
  the user) through `copier update` on `build/python-template-0.1.4`, merged `--no-ff` with the
  user's approval. The gate's separate "src coverage" step is gone (pytest's `fail_under`
  measures `src/` alone since `0.62.1`); `tests/test_gate_script.py` pins that, and failed
  against the `0.1.3` gate first. Gate exit `0` on the branch, 4,392 tests, `src/` coverage
  96.32%; the merged tree is identical to the gated one, and the CUDA sync ran.
- **`qb_adj` in the run without game lines: done and closed.** Four runs in
  `models/qb_adj_no_lines/`, reviewed in a fresh context (`REVIEW.md`: approve, provenance clean,
  a tie, so `qb_adj` stays in `constants.NO_GAME_LINES_DISABLED_FEATURE_GROUPS` and no default
  changes) and recorded by an agent that did not produce them (`b810142`, `.agents/benchmarks.md`
  "The quarterback adjustment without game lines"; `1c24b8c` closes the `TODO.md` step 7 item).
  Gate exit `0` on `1c24b8c`. The review's rule-11 finding (runs without game lines took the
  market-anchored sigma reference) is fixed on the branch above.
- `0.62.1` (`eedfa70`, merged with the user's approval): coverage measures `src/` alone
  (`[tool.coverage.run]` `source = ["src"]`), the user's edit minus its `omit` line (`tests/*`
  was redundant; `src/*/__init__.py` matched only the one-line `src/nfl_predictor/__init__.py`).
- Before that, `main` took four docs commits (step 7's finished items closed, the cross-module
  imports follow-up archived, and AGENTS.md rule 6 plus `.agents/guardrails.md` letting
  reviewed, finished items close unasked) after five branches were merged `--no-ff` on
  2026-10-15 evening with the user's approval (the user's Week 6 picks were already in, which
  waived the wait for the Monday game):
  - `fix/round-count-narrowing` (`0.59.7`): `_boosted_rounds` (`ml/ml_model_core.py`) and
    `_estimator_best_iteration` (`ml/walk_forward.py`) read a round count only from a Python or
    numpy integer again (the pyright review's finding);
  - `fix/qb-adj-review-findings` (`0.60.0` plus `0.60.1`): the `qb_adj` group from `qb_elos.csv`,
    off in production, with the first review's fixes; `0.60.1` makes
    `game_utils.fill_future_qb_data` fill quarterbacks only on unplayed games, give a starter's
    values only to the side it filled and recompute every unplayed row's `*_diff` columns. An
    investigation before the fix found no played row filled in the 1999-2025 build and
    consistent diffs everywhere, so no current value changes; the fix closes a path for later
    values into played rows;
  - `feat/no-game-lines-groups` (`0.61.0`): a run without game lines leaves out
    `strength,elo,own_elo_wt,qb_adj` by default; it counts as one when its market features and
    anchor are off, whatever the market transform (the user dropped the transform from the check;
    `ml/feature_spec._resolve_market_columns` adds no line column without market features).
    `train` takes `--no-market-anchor`/`--no-market-transform` (accepted by the user). Production
    is pinned unchanged (groups, stage-1 settings, stage hash, run fingerprint);
  - `feat/new-file-sigma-reference` (`0.62.0`): `constants.FLOOR_SIGMA_REFERENCE_RUNS` and
    `config/weekly_run.yaml` read `models/qb_elos_measure/new_file_seed42` and `new_file_seed7`
    (production's settings on the current `qb_elos.csv`), replacing the step-6 runs;
  - `docs/qb-elos-measurement-record`: `.agents/benchmarks.md` "`qb_elos.csv` measurement" and
    step 7's status in `TODO.md`, written from `models/qb_elos_measure/REVIEW.md` by an agent that
    did not produce the runs.
- Every walk-forward checkpoint fingerprint moved (`ml/*.py` in `0.59.7`, `constants.py` in
  `0.61.0` and `0.62.0`); the user accepted reruns. `scripts/gate.sh` on `f8d6a4e` after the
  merges and the CUDA sync: exit `0`, 4,392 tests, 98.12% coverage.
- Independent review of the four deltas (round count, fill, the two merges' conflict
  resolutions, the transform drop, the reference swap): every one approved, gate exit `0`, 4,392
  tests; every test commit failed on its parent. Report `~/scratch/review_stack_report.md`;
  adversarial tests uncommitted in `~/scratch/review_stack/tests/test_review_stack_adversarial.py`,
  29 pass and 1 fails (a Python `bool` reads as one round: the old check accepted it too, neither
  library returns one; not fixed). The fill docstring nit is fixed (`2dace35`); the stale
  benchmarks line is fixed by the docs branch.
- `data/` in the main checkout is the 2026-10-14 build: no `qb_adj` columns, which production
  leaves out anyway. The weekly run's default data refresh rebuilds every season, so the user's
  Week 7 run writes the columns and applies the fill fix; no separate rebuild is needed.
- The `0.59.6` strict-pyright work on `main` stands: 0 pyright errors, no inline suppressions,
  `tests/fakes.returning`, `utils/narrowing`, every test in Arrange-Act-Assert form,
  python-template `0.1.3`. Week 6 re-predicted from the production run's saved model
  (`models/weekly_20261015_022416_9a4c092c`) gave identical predictions, picks, ranks and floor
  sigma (`13.3049`). The stopped parity run's `models/weekly_parity_0596_week06/` was deleted
  with the user's approval.
- `../nfeloqb` fork `main` is at `9bf32f2` (pushed; the Las Vegas `OAK` lookup fix). The user runs
  it each week (memory `nfeloqb-weekly-process`).
- No web server was listening at 19:05 MDT (`ss -ltnp`); check before a `web/dist/` build or an
  edit, and never stop or restart one the user started.

## Next, in order

0. **Merge `fix/no-game-lines-sigma-reference`** once the user approves (see State), then the
   CUDA sync, the gate on `main`, and delete the branch.
1. **`../nfeloqb`, a backup's first start** (approved 2026-10-15, test first): the next-week
   projection rates the starter with `current_value`, while the played row uses
   `get_value(team)`, which applies the backup rule (Baltimore's Week 5 projection was off by
   `0.8`). Read `get_value` first and confirm it uses only pre-game information; then a failing
   test in `tests/test_elo_qb_adjustment.py`, the fix in `elo_file_constructor.py`, the fork's
   gates (`.venv/bin/ruff format .`, `ruff check .`, `ty check .`, `pyright .`,
   `markdownlint-cli2 "**/*.md" "#.venv"`, `.venv/bin/python -m pytest`), a `fix:` commit, and a
   push to the fork (`GUARD_OK=1`). Before the user's next nfeloqb run (Week 7).
2. **The fork's win-total file** (approved 2026-10-15 as its own measurement, landing between
   game weeks): `nfeloqb/Manual Data/wt_ratings.csv` ends at 2023, so the QB-adjusted Elo has had
   no win-total pull since 2024 (`elo.py` line ~136). Hypothesis and rule first, a fork change, a
   third build against the grouped measurement's new-file build. If it changes production's
   input, the floor-sigma reference moves with it.
3. **The step-6 docs close-out and the roadmap tidy** (approved 2026-10-15):
   - regenerate `RANKINGS_EVAL.md` under `models/step6/rankings/` (a `models/` write: ask);
   - rename "market-free" to "no game lines" where it means the configuration with Elo,
     including `weekly_run/stage1.py:63,68`;
   - the old roadmap labels in user docs: `README.md:898` ("task 55.8"), `README.md:1068` and
     `config/weekly_run.yaml:148` ("decision 12");
   - refresh the stale "Review:" fields in `.agents/sos_port/decisions/*.md`; commit the
     hypothesis checksums;
   - record the 2026-10-11, 2026-10-12 and 2026-10-14 decisions in `step6_plan.md` and
     `TODO.md`; they and the open P3s from the integration reviews are in an earlier handoff,
     `git show 5b73686:.agents/next_agent_session_prompt.md`;
   - in `TODO.md`: refresh "Current validated baseline" (it still says `0.28.1`); mark step 6's
     phases done; retire the follow-ups that concern only the strength snapshot, since the
     grouped measurement kept the ported ratings (rule 5: the user approves each closure);
   - build the approvals list and the user's final HTML report for step 6.
4. `scripts/gate.sh --web` on `main` and a browser look once the user restarts the server; the
   2026 roof recheck before the Paris game on 2026-10-25
   (`.agents/sos_port/decisions/pbp_columns.md`).

## Decisions on record (2026-10-15)

- Floor-sigma reference: `models/qb_elos_measure/new_file_seed{42,7}` (done, `0.62.0`).
- No-game-lines check: market features and anchor off, the transform not counted (done).
- `train`'s `--no-market-anchor`/`--no-market-transform` and its `--exclude-market` default list:
  accepted.
- The no-game-lines list keeps `qb_adj` (measured: a tie, `models/qb_adj_no_lines/REVIEW.md`).
- Runs without game lines take their own floor-sigma reference,
  `models/qb_adj_no_lines/no_lines_seed{42,7}` (chosen by the user; `0.63.0` on the branch).
- The `_fill_side_qb` fix: investigate first, then fix (done, `0.60.1`).
- `.agents/m60/INVENTORY.md` and `PROPOSAL.md` stay frozen with their old helper names.
- Merges no longer wait for the Monday game this week (the picks are in).
- Finished, independently reviewed roadmap items close without asking (AGENTS.md rule 6);
  reopening, reordering or retiring undone items still asks.
- Merged scratch worktrees removed 2026-10-15; the reviewers' adversarial tests are in
  `~/scratch/review_archive/`. Kept: `~/scratch/qb_measure` (the new-file build, for the
  win-total measurement), `~/scratch/qb_adj_nolines` (the code of the `qb_adj` measurement,
  whose reference runs are now the default sigma reference without game lines) and
  `~/scratch/step6/data` (step 6's candidate build, cited by `.agents/sos_port/*.md`).
- Earlier the same day: the Las Vegas fork fix (done); `qb_adj` as features, off until measured;
  the grouped measurement and its ladder; the backup first-start fork fix; the stale win-total
  file as its own measurement; steps 7 and 8 before the re-tune (step 9); the Week 6 top-decile
  ranking is the two-seed mean.

## Open for the user

- Merging `fix/no-game-lines-sigma-reference` (`0.63.0`) into `main`: reviewed, gate green;
  production's settings unchanged.
- The test suite: the user asked about pruning; recommended no blanket review (4,392 tests run
  in about 57 seconds on 8 workers, and coverage of `tests/` never drove their number).
- The harness worktrees `.claude/worktrees/agent-*` (including `feat/qb-adj-features` and
  `docs/qb-elos-measurement-record`, both merged) are left alone.
- The stats-only model: proposed 2026-10-15, not decided. A named track in three parts: in step 8
  measure each feature idea without game lines as well as with them; in step 9 tune it as its own
  study; a weekly shadow prediction, logged and never submitted, scored against the closing line.
- 58.1's exact survivor planner in place of the plan's beam search (acceptance pending).

## How the last sessions worked (keep doing this)

- Each code chunk: test first, one file per commit (a signature change lands with its callers),
  `chore: release X` with `CHANGELOG.md`, `pyproject.toml` and `uv.lock`; a separate `reviewer`
  in a fresh context (adversarial tests, uncommitted); a non-producer agent writes measured
  numbers into the docs from `REVIEW.md`; a `--no-ff` merge with a Conventional Commits subject
  after the user's approval.
- The bash-guard hook blocks any command containing "merge" (`git merge-base` included) unless it
  starts with `GUARD_OK=1`, and bare `python3`; use `git rev-list --count A..B` for ancestry and
  `.venv/bin/python`.
- A worktree needs its own synced venv
  (`VIRTUAL_ENV=$PWD/.venv uv sync $(<main>/.venv/bin/nfl-lightgbm-cuda-install uv-args)`) for
  the gate and the pre-commit ruff hooks. A backgrounded gate keeps its exit code
  (`...; rc=$?; echo "exit=$rc" >> log; exit $rc`), and its summary is read before any commit:
  `f78e27f` was committed on a failed markdownlint step because a trailing `echo` masked the code.
- Stage resolved files before committing a merge (`git add`), and never `git stash` with a
  pathspec while other files are staged: the stash takes the index too.
- Strict pyright in tests: `tests/fakes.returning(value)` for stubs, a typed nested
  `def _capture(message: str, *args: object)` for log capture, typed empty containers
  (`set[tuple[str, ...]]()`).
- Run `.venv/bin/pre-commit run --all-files` before ending a turn. Stop at a natural point well
  before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once. Check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly|data)"` first.
- A backtest that judges probabilities runs from 2007 week 1 or passes
  `--floor-sigma-reference-runs` with the reference, and passes `--wf-start-week 1`.
- Elo research is in `~/scratch/elo_research/`; `nfl-sos-ratings` is frozen at `9081bce`, read
  only as `git -C ../nfl-sos-ratings show 9081bce:<path>`.
