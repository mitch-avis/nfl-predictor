# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then
`.agents/TODO.md` ("Roadmap Status", above all step 6, and "Open follow-ups from completed
milestones"), `.agents/sos_port_source.md` (the source and open items of the port),
`.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-11, instruction files and lint rule merged; step 6 in progress)

- `main` holds `0.45.2`, ahead of `origin/main` and not pushed. `0.45.1` is the instruction-file
  restructure: `AGENTS.md` at 177 lines, the guardrails' long form in `.agents/guardrails.md`,
  path-scoped rules under `.claude/rules/`, the season and week conventions in
  `.agents/modeling_spec.md`, the launch lessons in `.agents/walk_forward_runbook.md`, and a
  `SessionStart` hook that injects this file at startup and after compaction. `0.45.2` forbids
  inline lint and type-checker suppressions: a genuine false positive goes in Ruff's per-file
  ignores with a one-line reason, and every existing one has its reason. The user's global
  `~/.agents/AGENTS.md` and skills say the same. Both docs branches are merged and deleted.
- Step 6 work from 2026-10-10 lives on `feat/step6-sos-port` (worktree `~/scratch/step6`, 520
  commits ahead of `main`) and `feat/sos-port-inventory`. Four `worktree-agent-*` branches hold
  commits that never reached the step-6 branch and await the user's triage: `a2071e4c` (9: the
  even-blend ranking default, a breaking weekly-run change), `a5c59090` (19: rating-definition
  tests and their records), `aa56c16b` (23: betting-edge review scripts and sos-port tests),
  `ab92fdae` (6: the refreshed port audit). Their worktrees were removed; the other 17 agent
  branches were contained in the step-6 branch and were deleted.
- The user's main model is Opus 5.5 at the default medium effort (`/effort high` for debugging
  and verification work); subagents run Opus 5.5 at medium with turn caps. Fable 5.1 is used
  only when the week's credits are exhausted.

- The main checkout is on `main`. Other branches: `feat/web-ui` (leave it alone), and the step-6
  branches and four agent branches above. Worktrees: `~/scratch/step6` (`feat/step6-sos-port`)
  and `~/scratch/step6_runs` (detached). `scripts/gate.sh` exits `0` on `main`, and the
  environment is synced with the LightGBM CUDA flags.
- The user runs `nfl-predictor web --port 8001` from the main checkout (no `--reload`). Never stop
  or restart it yourself (rule 5); it loads new code only when the user restarts it.
- Roadmap step 4 (feature values, `0.37.2`-`0.45.0`) and step 5 (market-free baseline) are closed
  and archived (`ARCHIVE.md`). Production since `0.44.0`-`0.45.0`: pick-time market lines by
  default, the defense-adjusted quarterback rate removed, the next-opponent identity pair pruned,
  market-free backtests available (`--exclude-market --no-market-anchor --no-market-transform`,
  plus `--disable-feature-groups elo` for no market information at all), tests isolated from
  `data/` and the network, and production's floor-sigma pool is the adopted-settings reference.
- `data/` was rebuilt 2026-10-09 with these defaults (backup in
  `data/backup_pre_step4_adopted_20261009/`). The current GPU reference is
  `models/step4_reference/adopted_seed42` and `adopted_seed7`; the step-6 test bench is
  `models/step4_reference/marketfree_all_seed42` and `marketfree_all_seed7`. Both are reviewed
  (`models/step4_reference/REVIEW.md`; numbers in `.agents/benchmarks.md`, "Adopted-settings
  reference and market-free baselines").
- `nfl-sos-ratings` is frozen at `9081bce` and its session is archived; everything still open
  there is this repo's work, recorded in `.agents/sos_port_source.md`. Read it at that commit,
  read-only (`git -C ../nfl-sos-ratings show 9081bce:<path>`).
- Names (guardrail 16, and the user's global rule): the current-settings baseline (`R0`), the
  pick-time lines test (`PT`), the adopted-settings reference (`adopted_*`), the market-free run
  without betting-line columns (`marketfree_lines_*`), the market-free run without betting-line or
  Elo columns, which is the test bench (`marketfree_all_*`). Never use a code alone with the
  user; say what a task or step is, not just its number.

## Decisions on record

- 2026-10-02: drop the defense-adjusted quarterback rate; drop the next-opponent identity pair
  from the model; adopt pick-time lines.
- 2026-10-08: the season-to-date stat blend keeps `K = 4`; the strength-blend `K` is superseded
  by the preseason prior `nfl-sos-ratings` fits inside its ridge. Roadmap: step 5 the market-free
  baseline, step 6 absorbing `nfl-sos-ratings` fully, step 7 the Optuna re-tune. Power rankings
  combine the ported ratings with the trained model, the combination chosen by measurement in
  step 6, phase 5. Feature ideas (referee, weather, attendance, team health) are listed under
  step 6 and not yet approved.
- 2026-10-09: measure both market-free variants; back up and rebuild `data/`; the market-free
  option as flags instead of a feature group (accepted). Reported by the `nfl-sos-ratings` session
  as the user's (treated as standing): the quarterback all-plays test's tie-break order (every
  quarterback play, then plus scrambles, then passes and sacks), the botched-snap charge rule, and
  a pre-registered test before kneel-downs and spikes leave the team ratings.
- 2026-10-10: the step-6 test bench is the market-free run without betting-line or Elo columns;
  steps 4 and 5 closed, step 4's leftovers moved to step 6; this repo builds its own 538-style
  Elo as a step 6, phase 4 candidate (plan in `TODO.md`); the user's `../nfeloqb` fork reads a
  stale local win-total file (2003-2023) and gets its fix only after that Elo replaces the
  `qb_elos.csv` columns, with the user's go-ahead then.

## Next

1. Roadmap step 6, phase 1, on a new branch off `main` (for example `feat/sos-port-inventory`;
   creating it needs no ask): a scripted inventory (rule 10) of `nfl-sos-ratings` at `9081bce`
   against this repo (data sources, metric registry columns, ratings, outputs, web pages), and a
   check that this repo's play-by-play aggregates match its game logs for the same seasons, with
   its pre-2006 scramble rule and the nflverse quirks listed in `.agents/sos_port_source.md`.
   Plan the phase with the user before code (what the inventory covers, how parity is judged).
2. Batch with step 6's first edit to `constants.py` (it changes every checkpoint fingerprint):
   a `constants.MODELS_PATH` read at call time by the five places that build `ROOT_DIR /
   "models"` (`walk_forward.DEFAULT_CHECKPOINT_DIR`, `cli/backtest.py`, `ml/artifacts.py`,
   `weekly_run/pipeline.py`, `cli/checkpoints.py`) and the API settings, so the test guard can
   cover `models/` too.
3. Small follow-ups from the 2026-10-09 review (`models/step4_reference/REVIEW.md`): `compare`
   scores an exact 0.5 probability as a wrong pick while `ml/metrics.picks_home` picks home (only
   market lines are affected); name the resampling unit wherever pool-point intervals are quoted.
4. Weekly runs and picks are the user's. Production-changing work lands between game weeks
   (after Monday night, before Thursday).

## How the last sessions worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer`, fixes
  back to the implementer (a second review round for anything with logic), and a `--no-ff` merge
  into the step's branch (the commit hook needs a Conventional Commits subject on merge commits
  too). The delegating session writes `CHANGELOG.md` and the version (`pyproject.toml`,
  `uv lock`, then `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`), `TODO.md`,
  `ARCHIVE.md`, `AGENTS.md`, `.agents/benchmarks.md` and this file, and lints the Markdown it
  edits with `markdownlint-cli2` before committing.
- Every measured number goes through an independent reviewer's rescore before it enters the docs
  (rule 3). Hypotheses and decision rules are written into the run directory before launch; long
  runs go through one detached driver script with a PID watcher, and nothing is committed while
  they run (each run records the commit).
- Merges into `main` happen in a temporary worktree (`git worktree add ~/scratch/merge_main
  main`), so the user's checkout and running web server are untouched. Run the gate on the final
  tree, in the main checkout or a scratch worktree.
- Inspect an agent's worktree with `git -C <path>` or absolute paths; a `cd` into it moves this
  session's working directory there.
- Fix lint in the code rather than suppress it. Run `.venv/bin/pre-commit run --all-files`
  before ending a turn (the Stop hook runs it).
- Stop at a natural point well before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once. Check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- A backtest that judges probabilities runs from 2007 week 1 or passes
  `--floor-sigma-reference-runs` with the GPU reference, and passes `--wf-start-week 1`
  (rule 11).
- Re-run `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)` after every version bump and
  every merge or checkout that changes the version or the layout; a plain `uv sync` replaces the
  local CUDA build of LightGBM with the CPU wheel.
- Elo research (538's original code and methodology, `nfelo`, upstream `nfeloqb`, the free
  win-total file) is in `~/scratch/elo_research/`.
