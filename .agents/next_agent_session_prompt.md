# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then
`.agents/TODO.md` ("Roadmap Status", above all step 6, and "Open follow-ups from completed
milestones"), `.agents/sos_port_source.md` (the source and open items of the port),
`.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-10, roadmap steps 4 and 5 closed, step 6 running autonomously)

- The user asked on 2026-10-10 (about 02:00) for step 6 to run autonomously to completion, with
  final approvals batched at the end and every difference between the two repos decided by the
  session and logged in `.agents/sos_port/DECISIONS.md`. What that authorizes and what stays held
  for the user, the streams, their dependencies and the measurement ladder are in
  `.agents/step6_plan.md`: read it first.
- `main` is at `0.45.0` (`ad07198`), pushed, and the main checkout stays on `main` (the user's web
  server and weekly runs import its code). Step-6 work lives in the integration worktree
  `~/scratch/step6` on `feat/step6-sos-port`; implementer branches merge into it. The other
  branch is `feat/web-ui`: leave it alone. `scripts/gate.sh --web` exits `0` on `main` (1580
  passed, coverage 93.29%), and the main checkout's environment is synced with the LightGBM CUDA
  flags.
- Landed on `feat/step6-sos-port` (each reviewed independently, gate green on the merged tree):
  `0.46.0` the nflverse player sources, `constants.MODELS_PATH` and the `compare` tie rule;
  `0.47.0` this repo's own Elo (not yet in the ETL); `0.48.0` the Pool page, the margin-only
  switch and the live-betting design; `0.49.0` the ported ratings (`src/nfl_predictor/ratings/`);
  `0.50.0` the play-by-play cache extension (139 columns) and the stadium table; `0.50.1` the
  phase-1 inventory and parity scripts (reports pinned to `main`'s code; a refresh is due after
  the integration lands). Gate on `0.50.1`: 2205 passed.
- In review or fixes at 06:55 on 2026-10-10 (branches of implementer worktrees under
  `.claude/worktrees/`): the ratings in the ETL (`worktree-agent-a0778ee61e6367a09`), the
  play-by-play corrections, one starter and the own Elo in the ETL
  (`worktree-agent-ab77a9529b8ed64b8`, which contains the former; merge the reviewed ratings
  branch first), the season statistics port (`worktree-agent-a5d833400b788544f`), the
  production feature-group setting (in progress), and the web pages for the ratings and season
  statistics (in progress). The stream table, the integration design and the user's list are in
  `.agents/step6_plan.md`; the phase-4 build and run scripts are `models/step6/build.sh` and
  `driver.sh` (dry-run tested), with the pre-registered rules in `models/step6/HYPOTHESIS.md`
  and `models/step6/rankings/HYPOTHESIS.md` (each with a `.sha256`).
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

1. Roadmap step 6, every phase, per the stream table in `.agents/step6_plan.md` (which records
   each stream's branch and state). Phase 1 (the scripted inventory of `nfl-sos-ratings` and
   the play-by-play parity check) follows the plan the user accepted on 2026-10-10 (step 6,
   item 1 in `TODO.md`); its differences now go to the decision log rather than waiting for the
   user. If this session ended mid-stream, check `git worktree list` and each stream's branch;
   a separate `reviewer` checks every chunk (and re-runs any script whose numbers enter a doc)
   before it merges into `feat/step6-sos-port`, and the delegating session writes
   `CHANGELOG.md` and the version at each merge.
2. `constants.MODELS_PATH` and the `compare` tie rule landed in `0.46.0` on the step-6 branch.
   Left from the 2026-10-09 review: `.agents/benchmarks.md` still quotes pool-point intervals
   without their resampling unit (around lines 459, 503 and 577); label each only after
   confirming the unit from the run that produced it.
3. Weekly runs and picks are the user's. Production-changing work lands between game weeks
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
