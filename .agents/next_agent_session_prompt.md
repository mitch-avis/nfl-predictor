# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-01, step 4 in progress)

- Roadmap step 4 runs on `feat/step4-feature-values` (off `main` at `3bd457c`, not pushed), checked
  out in the main checkout at `0.38.0`; `uv sync` has run. `scripts/gate.sh` exits `0` on it (1261
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
  - `0.37.5`: opt-in `nfl-predictor data --incremental` (`utils/season_cache.py`), about 48 s warm
    against about 640 s full in scratch (`~/scratch/etl_incremental/`). Narrowed in `TODO.md`: a
    reused season matches a full rebuild only until a later season gains rows, because
    `calculate_league_means` (`utils/polars/teamrankings.py`) reduces a slice whose chunk layout
    moves; two strict-xfail tests pin it. `constants.py` changed, so every checkpoint retrains.
  - `0.37.6`: `data_collection.py` split (week builder to `utils/polars/week_rows.py`, strength
    table to `utils/polars/strength_table.py`), byte-identical; the first `--incremental` run
    afterwards rebuilds every season (code fingerprint changed).
  - `0.38.0`: `calculate_league_means` rechunks its season slice (features moved once by up to
    about `1.8e-15`), so `--incremental` equals a full rebuild after later seasons gain rows
    (`~/scratch/etl_rechunk/`); the weekly run refreshes with `--incremental`
    (`config/weekly_run.yaml`). The first incremental run rebuilds every season (code fingerprint
    changed in `0.37.6`/`0.38.0`), then about 48 s.
- The "4, first" reproducibility work is done apart from the low-priority schedule cache and two
  small `0.37.6` review follow-ups (`TODO.md`). No subagent is in flight.
- Week 4 picks are in `models/weekly_2026_week_04/` (`0.35.3`). No web server is running.

## Open questions for the user

None pending. Decided 2026-10-01: same-machine byte-identity is enough; a text token in a
numeric `qb_elos.csv` column stops the ETL (tentatively; none ever seen in 233 `nfeloqb`
versions); `.rechunk()` approved; the weekly run passes `--incremental`; `data_collection.py`
split (with the extra `strength_table.py` module, recommended). The web weekly job's full
rebuild is task 58.7. The user runs the Week 4 refresh themselves.

## Next

1. Optional small chunks: the two `0.37.6` split-review follow-ups and the schedule cache
   (`TODO.md`, "4, first" row).
2. Then the feature-value changes, sharing one rebuild cycle and one new two-seed GPU reference:
   55.3, 53.7, 56.6 (the pick-time market line; plan agreed 2026-10-01 in the task text) and the
   step-4 follow-ups. Every ETL rebuild into `data/` is must-ask (back up `data/*.csv` first);
   scratch copies are not.
3. Week 4 refresh: the user runs it themselves on 2026-10-01 (staged in
   `models/weekly_2026_week_04_refresh/launch.sh`, for `main`). Do not launch it.
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
