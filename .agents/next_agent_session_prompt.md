# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-01, step 4 in progress)

- Roadmap step 4 runs on `feat/step4-feature-values` (off `main` at `3bd457c`, not pushed), checked
  out in the main checkout at `0.37.4`; `uv sync` has run. `scripts/gate.sh` exits `0` on it (1261
  passed, coverage 92.75%). `main` is at `0.37.1` plus the step-3 close-out, pushed.
- Landed on the branch, each reviewed by an independent reviewer and merged with `--no-ff`:
  - `0.37.2`: the team-stat and play-by-play nflreadpy caches record their requested columns in
    Parquet metadata and refetch when stale (all 56 existing files pass, nothing refetched);
    five data-CSV reads infer types from the whole file (Week 4 outputs unchanged).
  - `0.37.3`: byte-identical ETL rebuilds on one machine at one Polars thread count (sorted
    `sos_*` reductions, week-ordered season-to-date means, `maintain_order=True` on every ETL
    `group_by`/`unique` with an AST guard test, same-date games sorted by `game_id`). Scratch
    evidence in `~/scratch/etl_repro/` (4.6 GB, deletable once no longer needed). Values move
    against the current `data/` build by at most about `2e-16` of scale, and same-date row order
    changes, so the next rebuild gets a new dataset fingerprint once; any step-4 build-to-build
    comparison rebuilds its reference with current code rather than comparing with `data/`.
  - `0.37.4`: both `qb_elos.csv` loaders read declared column types (outputs and two 2019-2026
    scratch builds unchanged, `~/scratch/qbelo_types/`); a text token in a numeric column now
    fails the read.
- In flight when this was written (an implementer subagent in `.claude/worktrees/`):
  - an opt-in incremental ETL (`--incremental` or similar) that reuses finished seasons keyed on
    a code fingerprint and input hashes, proven byte-identical to the full build
    (`~/scratch/etl_incremental/`). Making it the default is a question for the user.
  If a session restarts, check `git worktree list` and the worktree branches' logs; an unfinished
  worktree can be resumed or redone from the task text in `TODO.md`.
- Week 4 picks are in `models/weekly_2026_week_04/` (`0.35.3`). No web server is running.

## Open questions for the user

None pending. Answered 2026-10-01: same-machine byte-identity is enough (accepted); a text token
in a numeric `qb_elos.csv` column stops the ETL (kept, tentatively; none in any of the 233
`nfeloqb` versions since 2023-08-09).

## Next

1. Review and merge the in-flight chunk (independent reviewer) under its own patch version,
   then the gate.
2. Then the feature-value changes, sharing one rebuild cycle and one new two-seed GPU reference:
   55.3, 53.7, 56.6 (the pick-time market line; plan agreed 2026-10-01 in the task text) and the
   step-4 follow-ups. Every ETL rebuild into `data/` is must-ask (back up `data/*.csv` first);
   scratch copies are not.
3. Week 4 refresh before the Thursday game (picks due about 17:00 MDT on 2026-10-01): staged in
   `models/weekly_2026_week_04_refresh/launch.sh` (full weekly run with a fresh ETL). Run it from
   `main`: `git checkout main && uv sync`, then `nohup setsid` the launch script, then return to
   `feat/step4-feature-values` and `uv sync` once it exits. Ask the user first whether
   `data/qb_elos.csv` has been updated. Check the games, `floor_sigma` and ranks as for Week 4.
4. Week 5 picks after the Monday game and the user's `data/qb_elos.csv` update:
   `.venv/bin/nfl-predictor weekly --run-id weekly_2026_week_05`, through a `launch.sh` in the run
   directory with `nohup setsid`. The weekly run uses the code on the checked-out branch: run it
   from `main` (or ask the user whether the step-4 branch is acceptable), and after switching
   branches run `uv sync`. Every checkpoint fingerprint changed with `0.36.0` and `0.37.0`, so
   stage 1 retrains (about a minute on the GPU).

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
