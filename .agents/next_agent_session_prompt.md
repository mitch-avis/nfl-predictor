# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-09 05:10, step 4 finished and merged)

- `main` is at `0.45.0` (`db98665`, merged 2026-10-09 in a temporary worktree, not pushed; the
  user approved the merge 2026-10-08, pushing is still must-ask). Its tree equals
  `feat/step4-feature-values` (`66e4508`), which is still checked out in the main checkout.
  `scripts/gate.sh --web` exits `0` on that tree (1580 passed, coverage 93.29%). The environment
  is synced with the LightGBM CUDA flags. The user runs `nfl-predictor web --port 8001` from the
  main checkout (no `--reload`); it needs a restart to load the new code.
- Landed since `0.43.0`: `0.44.0` the decisions chunk (defense-adjusted quarterback rate removed,
  next-opponent pair pruned, pick-time lines by default, `nfl-predictor lines` follows the line
  source, market-free backtests via `--exclude-market` and the `elo` feature group); `0.44.1`
  session-wide test data and network isolation; `0.45.0` production's floor-sigma pool switched
  to the adopted-settings reference. All reviewed independently.
- `data/` was rebuilt 2026-10-09 with the new defaults (backup in
  `data/backup_pre_step4_adopted_20261009/`). The adopted-settings reference and both market-free
  baselines are in `models/step4_reference/` (reviewed in `REVIEW.md`; numbers in
  `.agents/benchmarks.md`). The reference tied the pick-time lines test, so it became the new
  reference and production's floor-sigma pool.
- `nfl-sos-ratings` is frozen at `9081bce` and its session archived; its full handoff is in
  `.agents/sos_port_source.md`.
- Names (guardrail 16): the current-settings baseline (`R0`), the pick-time lines test (`PT`),
  the adopted-settings reference (`adopted_*`), the market-free run without betting-line columns
  (`marketfree_lines_*`), the market-free run without betting-line or Elo columns
  (`marketfree_all_*`). Never use the codes alone with the user.

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

1. Open questions for the user (asked 2026-10-09): which market-free variant is step 6's test
   bench (recommended: without betting-line or Elo columns); close roadmap steps 4 and 5 into
   `ARCHIVE.md` (must-ask; recommended yes); push `main` (must-ask); confirm the three decisions
   the `nfl-sos-ratings` session reported (QB all-plays tie-break order, the botched-snap charge
   rule, a pre-registered kneel and spike test).
2. Then roadmap step 6, phase 1, on a new branch off `main`: the scripted inventory of
   `nfl-sos-ratings` at `9081bce` against this repo, and the play-by-play parity check (with its
   pre-2006 scramble rule).
3. Batch with step 6's first edit to `constants.py` (it changes every checkpoint fingerprint):
   a `constants.MODELS_PATH` read at call time by the five places that build `ROOT_DIR /
   "models"` (`walk_forward.DEFAULT_CHECKPOINT_DIR`, `cli/backtest.py`, `ml/artifacts.py`,
   `weekly_run/pipeline.py`, `cli/checkpoints.py`) and the API settings, so the test guard can
   cover `models/` too.
4. Small follow-ups from the 2026-10-09 review (`models/step4_reference/REVIEW.md`): `compare`
   scores an exact 0.5 probability as a wrong pick while `ml/metrics.picks_home` picks home (only
   market lines are affected); name the resampling unit wherever pool-point intervals are quoted.

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
