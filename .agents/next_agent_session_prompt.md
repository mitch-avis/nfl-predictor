# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-02, step 4 in progress)

- Roadmap step 4 runs on `feat/step4-feature-values` (local only, not pushed), checked out in the
  main checkout at `0.42.0`; `uv sync` has run and the tree is clean. `scripts/gate.sh` exits `0`
  on it (coverage 93.22%). `main` is at `0.39.3` (merged 2026-10-01, pushed 2026-10-02, both
  approved by the user). No agent worktrees remain.
- On `main` (reviewed, each with a `CHANGELOG.md` entry; details in `ARCHIVE.md`, "Roadmap step
  4 (in progress) - Resolved follow-ups"): `0.37.2`-`0.38.2` rebuild reproducibility (nflreadpy
  cache column checks, byte-identical rebuilds at one Polars thread count, typed `qb_elos.csv`,
  the opt-in `--incremental` ETL with the `.rechunk()` fix, the weekly run refreshing with
  `--incremental`, the `data_collection.py` split, the schedule cache check); `0.39.0` the web
  weekly job over `config/weekly_run.yaml` (task 58.7); `0.39.1`-`0.39.2` the user's dependency
  refresh, bare `pytest`, and `update_requirements.sh` repairs (LightGBM CUDA step non-fatal);
  `0.39.3` the `next_opponent_identity` group. Since `0.37.3` a step-4 build compares against a
  reference rebuilt with current code, not against today's `data/`.
- Only on the branch, Phase A of `.agents/step4_plan.md` chunks 1, 2 and 4 (reviewed): `0.40.0`
  the `qb_def_adj` group (on by default once built; production has no group switch, so it must
  not reach `main` unless adopted); `0.40.1` `--strength-prior-blend-games`; `0.41.0` the QB
  defense term in per-dropback units; `0.42.0` `--line-source {stored,pick_time}` (stored builds
  byte-identical across the merge, `~/scratch/step4_merge_check/`). Each arm's feature-group
  switches are in the plan.
- Background, 2026-10-01: a global Stop hook ran a formatter from the home folder and rewrote the
  uv cache and managed Pythons; the user cleared the cache and reinstalled them, and the hook is
  now guarded by a `.pre-commit-config.yaml` check. Scratch evidence directories under
  `~/scratch/` keep their scripts, logs and venvs; their `data/` copies were deleted 2026-10-02.

## Open questions for the user

1. 56.6 map window (blocks the PT build): keep the per-season fit on earlier seasons only, or one
   pooled 2007-2025 fit? The agent's in-depth answer on 2026-10-02 recommended the expanding fit:
   production is identical either way (2026 fits on all of 2006-2025); the map only fills derived
   moneylines (features and the market yardstick), not the submitted floor probability; the
   largest gap (0.022) is at extreme spreads, and near pick'em both maps sit at about 50%; the
   pooled fit would price 2007 with 2015-2025 conventions, against `AGENTS.md`'s
   information-available-before rule. The user leaned toward pooling ("more data, no outcomes")
   and asked for that take; wait for the decision. Testing both would be a run beyond the cap.

Decided 2026-10-02: the `compare` market-from option (yes); push `main` (done); the old scratch
`data/` copies deleted (venvs and logs kept). Decided 2026-10-01: the plan, the per-dropback QB
units, the partial merge to `main`, no incremental option on `etl_full`; earlier decisions are in
`ARCHIVE.md`. The user handles weekly runs and picks; do not raise Week 5 for several days.

## Next

1. Phase A chunk 5 (`.agents/step4_plan.md`): the `compare` market-from option, one implementer
   and one reviewer, its own patch version.
2. Phase B after the map-window answer: the six scratch builds (from scratch tree copies, full
   history 1999+, each arm's feature-group switches as the plan lists), the leakage audit per
   build, then one driver for the 14 runs with a PID watcher; a check-in after an independent
   rescore.
3. After Phase B: the user's adoption decisions, then `F` (the adopted build, two seeds; the new
   GPU reference and the floor's sigma pool), one backed-up rebuild into `data/` (must-ask), and
   before any merge to `main` either adopt `qb_def_adj` or turn it off/remove it.
4. Weekly runs are the user's. If asked to run one, run it from `main` (the step-4 branch adds the
   unmeasured `qb_def_adj` columns to production); after switching branches run `uv sync` with the
   LightGBM CUDA flags (`uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`).

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
- Re-run a plain `uv sync` after every version bump and after every merge or checkout that
  changes the version or the layout.
- The user runs `nfl-predictor web` from this checkout. Never stop or restart it yourself
  (rule 5).
