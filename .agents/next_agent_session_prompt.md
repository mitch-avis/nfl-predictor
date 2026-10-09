# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-09 01:05, step 4 close-out runs in progress)

- `feat/step4-feature-values` (local only, not pushed) is checked out in the main checkout at
  `0.44.0` (`2522cb6`): the decisions chunk merged (`2f90f6c`, two review rounds), with the
  changelog and the docs pass. `scripts/gate.sh --web` exits `0` on `2522cb6` (1571 passed, 1
  skipped, coverage 93.29%). `main` is at `0.39.3`. The user runs `nfl-predictor web --port 8001`
  from this checkout (no `--reload`); it needs a restart to load `0.44.0`, which the user does.
- Running since 2026-10-09 01:00 as one detached job: `models/step4_reference/phase.sh 2522cb6`
  (log `phase.log`, PID in `phase.pid`, commit in `launch_record.txt`, rules in `HYPOTHESIS.md`).
  `build.sh` backed up `data/` to `data/backup_pre_step4_adopted_20261009/` (761 MB), rebuilds
  `data/` with the new defaults (the user approved the backed-up rebuild 2026-10-09), cuts
  `inputs/adopted.csv` and runs the leakage audit; then `driver.sh` runs six GPU walk-forwards
  on it: the adopted-settings reference (`adopted_seed42`, `adopted_seed7`), the market-free run
  without betting-line columns (`marketfree_lines_seed*`) and the one without line or Elo columns
  (`marketfree_all_seed*`). **Do not edit `src/`, `pyproject.toml` or `uv.lock`, and do not
  commit, while it runs.** Expected end about 04:45.
- The implementer is fixing the second review's test-isolation findings on its worktree branch
  (`worktree-agent-aae52e93d671a8c39`, after `e75c2bd`): session-wide data and network
  isolation in `tests/conftest.py`, a scan for captured data paths, the Elo real-build test
  removed, an optional warning in `weekly_run/stage1.py`. It merges as `0.44.1` after the runs,
  after a reviewer.
- Names (guardrail 16): the current-settings baseline (`R0`), the defense-adjusted quarterback
  test (`QB`), the next-opponent identity test (`NO`), the stat-blend tests (`S2`, `S8`), the
  strength-blend test (`T2`), the pick-time lines test (`PT`), the adopted-settings reference
  (`adopted_*`), the two market-free baselines (`marketfree_lines_*`, `marketfree_all_*`). Never
  use the codes alone with the user.

## Decisions

- 2026-10-02: drop the defense-adjusted quarterback rate; drop the next-opponent identity pair
  from the model; adopt pick-time lines as the default.
- 2026-10-08: the season-to-date stat blend keeps `K = 4`; the strength-blend `K` question is
  superseded by the preseason prior that `nfl-sos-ratings` fits inside its ridge (ported in
  step 6). The roadmap in `.agents/TODO.md` was extended: step 5 the market-free baseline, step 6
  absorbing `nfl-sos-ratings` fully (it finishes its in-flight work, the port starts from a
  frozen commit, then it becomes read-only), step 7 the Optuna re-tune. Power rankings combine
  the ported ratings with the trained model (how: decided by measurement in step 6, phase 5).
  The user approved finishing step 4 (including the merge into `main`) and running the
  market-free baseline right after it. Feature ideas to evaluate after the port (referee,
  weather, attendance, team health) are listed under step 6, not yet approved.
- The `nfl-sos-ratings` session is `nfl-sos-ratings-dc`; it was told (2026-10-08) to message
  this repo's session with the frozen commit when its fixes are done. Its repo stays read-only
  from here.

## Next

1. When `phase.sh` ends: confirm every fold ran on `cuda`, then an independent reviewer rescores
   all six runs from the fold checkpoints against `models/step4_reference/HYPOTHESIS.md` and
   writes `REVIEW.md` there (the reference against the pick-time lines test; the market-free
   arms against the reference, against each other, against the pick-time and closing markets via
   `compare --market-from models/step4/R0_seed42`, and against the Elo probabilities in
   `data/qb_elos.csv`). Then the check-in with the numbers.
2. If the reference passes its rule: point `config/weekly_run.yaml` `floor_sigma_reference_runs`
   at `models/step4_reference/adopted_seed42` and `adopted_seed7`, close rule 11's line-timing
   gap in `AGENTS.md`, regenerate `.agents/market_free/` on `inputs/adopted.csv` (the committed
   inventory was made on the old build with the removed columns), record the numbers in
   `.agents/benchmarks.md` (rule 3), merge the `0.44.1` test fixes, then merge into `main`
   (approved by the user 2026-10-08). Ask before pushing.
3. Then roadmap step 6 (absorb `nfl-sos-ratings` from `9081bce`, `.agents/sos_port_source.md`).
4. Batch with step 6's first edit to `constants.py` (it changes every checkpoint fingerprint):
   a `constants.MODELS_PATH` read at call time by the five places that build `ROOT_DIR /
   "models"` (`walk_forward.DEFAULT_CHECKPOINT_DIR`, `cli/backtest.py`, `ml/artifacts.py`,
   `weekly_run/pipeline.py`, `cli/checkpoints.py`) and the API settings, so the test guard can
   cover `models/` too (today it covers `data/` only; the implementer's recommendation,
   2026-10-09).

## How the last sessions worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer`, fixes
  back to the implementer, and a `--no-ff` merge into the feature branch (the commit hook needs a
  Conventional Commits subject on merge commits too). The delegating session writes:
  - `CHANGELOG.md` and the version (`pyproject.toml`, `uv lock`, `uv sync`);
  - `TODO.md`, `ARCHIVE.md` (step-4 resolved items under "Roadmap step 4 (in progress)"),
    `AGENTS.md`, `.agents/benchmarks.md` and this file.

  It lints the Markdown it edits with `markdownlint-cli2` before committing.
- Small review-fix deltas (docs, a flag) may be verified by the delegating session from the diff;
  anything with logic goes back to the reviewer (a second-round reviewer caught a `total_yards`
  regression in `0.37.2`).
- Every measured number goes through an independent reviewer's rescore before it enters the docs
  (rule 3). Decision rules are written into the run directory before scoring.
- Run the gate in a separate worktree (`git worktree add --detach ~/scratch/gate_step4 HEAD`,
  `uv sync`, `VIRTUAL_ENV=<it>/.venv scripts/gate.sh`). Watch long runs by polling a PID, never
  with `pgrep -f` on a pattern that matches the watcher's own command line.
- Lint work: fix the code rather than suppress. Run `.venv/bin/pre-commit run --all-files`
  yourself before ending a turn (the Stop hook runs it).
- Stop at a natural point well before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once. Check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- A backtest that judges probabilities must run from 2007 week 1 or pass
  `--floor-sigma-reference-runs` with the GPU reference, and must pass `--wf-start-week 1`
  (rule 11).
- Re-run `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)` after every version bump and
  after every merge or checkout that changes the version or the layout; a plain `uv sync`
  replaces the local CUDA build of LightGBM with the CPU wheel.
- The user runs `nfl-predictor web` from this checkout. Never stop or restart it yourself
  (rule 5).
