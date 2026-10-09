# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-08, finishing step 4)

- Roadmap step 4 runs on `feat/step4-feature-values` (local only, not pushed), checked out in the
  main checkout at `0.43.0` (the maintenance session's tooling changes are committed; read
  `.agents/branch_changes_2026-10-08.md`). The environment was re-synced with the LightGBM CUDA
  flags on 2026-10-08. `main` is at `0.39.3`. The user runs `nfl-predictor web --port 8001` from
  this checkout (no `--reload`).
- Only on the branch: the step-4 code (`0.40.0`-`0.42.1`: the defense-adjusted quarterback
  rate, `--strength-prior-blend-games`, `--line-source`, `compare --market-from`) and the
  measurement arms in `models/step4/` (reviewed in `REVIEW.md` there; numbers in
  `.agents/benchmarks.md`, "Step 4, Phase B feature-value arms").
- Names (guardrail 16): the current-settings baseline (`R0`), the defense-adjusted quarterback
  test (`QB`), the next-opponent identity test (`NO`), the stat-blend tests (`S2`, `S8`), the
  strength-blend test (`T2`), the pick-time lines test (`PT`), the adopted-settings reference
  build (`F` in `.agents/step4_plan.md`). Never use the codes alone with the user.

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

1. The decisions chunk (`0.44.0`), delegated to an implementer on 2026-10-08: remove the
   defense-adjusted quarterback rate, exclude the next-opponent identity pair from the model,
   pick-time lines by default, `nfl-predictor lines` follows the line source, and a market-free
   option on `nfl-predictor backtest` with a scripted inventory of market columns. Then a
   reviewer, the changelog and version, the gate.
2. The adopted-settings reference build (cold, full history, scratch tree, through-2025 cut under
   `models/step4_reference/inputs/`, leakage audit) and one driver for four GPU walk-forwards on
   it, all on one commit: the reference on seeds `42` and `7`, then the market-free baseline on
   the same seeds. Hypotheses and rules written into the run directory first; a PID watcher.
3. An independent rescore; the check-in; point `config/weekly_run.yaml`'s
   `floor_sigma_reference_runs` at the new reference; close rule 11's line-timing gap in
   `AGENTS.md`; merge into `main` (approved); ask before pushing and before any `data/` rebuild.
4. Then step 6.

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
