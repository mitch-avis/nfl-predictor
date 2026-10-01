# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-01)

- `main` is at `0.37.1` plus the step-3 close-out and pushed to `origin` (the user approved the
  merge and push on 2026-10-01). It is checked out in the main checkout, `uv sync` has run, and
  the working tree is clean. `scripts/gate.sh --web` exits `0` on it (see the check-in of
  2026-10-01: 1234 passed, coverage 92.69%, 26 frontend tests).
- What landed on 2026-09-29 to 2026-10-01, all merged with `--no-ff`:
  - `fix/weekly-dry-run` (`0.35.2`-`0.35.3`): `weekly --dry-run` skips the data refresh;
    `leakage-audit` creates its report directory; `validate` infers column types from every row.
    All 12 web job templates were launched live (accepted by the user).
  - `refactor/ruff-all` (`0.36.0`-`0.37.1`):
    - ruff `select = ["ALL"]` with no inline `noqa`;
    - the package in `src/nfl_predictor/`, built with `uv_build`; a scratch ETL rebuild matched
      the flat layout to `2.2e-16`, the ETL's own noise (`~/scratch/etl_src_check/`);
    - `license = "MIT"` (SPDX), and `reports/` gitignored.
  - `docs/step3-close-out`:
    - roadmap step 3 closed by the user (`ARCHIVE.md`, "Roadmap step 3");
    - task 56.6 rewritten as the step-4 pick-time line plan;
    - the feature-importance toggle became task 58.6;
    - the user's `uv.lock` refresh (charset-normalizer 3.5.2, fastapi 0.142.2).
- Every walk-forward checkpoint fingerprint changed with `0.36.0` and `0.37.0`, so the next
  walk-forward or weekly stage 1 retrains from scratch (about a minute on the GPU for stage 1).
- After any checkout that switches between the flat and the `src/` layout (an old branch), run
  `uv sync` at once and delete any leftover untracked `nfl_predictor/` directory of
  `__pycache__` files at the repository root.
- All agent worktrees are removed. These merged branches still exist locally and could be
  deleted with the user's agreement:
  - `fix/weekly-dry-run` and `fix/weekly-dry-run-impl`;
  - `build/uv-build-src` and the `worktree-agent-*` branches;
  - `refactor/ruff-all`, `docs/step3-close-out` and `feat/step3-parity`.
- Week 4 picks are in `models/weekly_2026_week_04/` (`0.35.3`; 16 games, `floor_sigma` `13.2467`
  with no fallback, ranks 1..16). It is the active web run, with no pin. No web server is
  running; if the user restarts `nfl-predictor web`, rebuild `web/dist` first (`npm run build`
  in `web/`).

## Open questions for the user

None pending.

## Next

1. Roadmap step 3 is closed (the user, 2026-10-01; `ARCHIVE.md`, "Roadmap step 3"). Step 4 on a
   new branch off `main`:
   - rebuild reproducibility first: the "4, first" row of the follow-up table under "Roadmap
     Status" in `TODO.md` (bit-reproducible schedule-strength columns, a schema version in the
     play-by-play cache key), plus the step-4 follow-up about Polars' 100-row type guessing and
     the incremental-ETL item from the Week 3 run. Known cause of non-determinism: `unique()`
     without `maintain_order` reorders rows within a date, and parallel float sums differ at
     about `1e-16` in the `sos_*` columns. Code changes need no asking; every ETL rebuild into
     `data/` is must-ask (scratch copies, as in `~/scratch/etl_src_check/run.sh`, are not).
   - then the feature-value changes: 55.3, 53.7, 56.6 (the pick-time market line: the
     `nfelomarket_data` getter, the per-game line order, the fitted spread-to-moneyline map; the
     plan agreed with the user on 2026-10-01 is in the task text) and the step-4 follow-ups,
     sharing one rebuild cycle and one new two-seed GPU reference, which also refreshes the
     floor's sigma pool (`floor_sigma_reference_runs`).
2. Week 5 picks after the Monday game and the user's `data/qb_elos.csv` update:
   `.venv/bin/nfl-predictor weekly --run-id weekly_2026_week_05`, through a `launch.sh` in the run
   directory with `nohup setsid`. Check the games, `floor_sigma` and the ranks as for Week 4.
   Every checkpoint fingerprint changed with `0.36.0` and `0.37.0`, so stage 1 retrains, which
   is about a minute on the GPU.

## How the last sessions worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer`, fixes
  back to the implementer, and a `--no-ff` merge into the feature branch. The delegating session
  writes:
  - `CHANGELOG.md` and the version (`pyproject.toml`, `uv lock`, `uv sync`);
  - `TODO.md`, `ARCHIVE.md`, `AGENTS.md`, `.agents/benchmarks.md` and this file.

  It lints the Markdown it edits with `markdownlint-cli2` before committing.
- Every measured number goes through an independent reviewer's rescore before it enters the docs
  (rule 3). Decision rules are written into the run directory before scoring.
- Run the gate in a separate worktree, because the `--web` gate rebuilds `web/dist`. Watch long
  runs by polling a PID (`while kill -0 <pid>`), never with `pgrep -f` on a pattern that matches
  the watcher's own command line.
- Lint work: fix the code rather than suppress. Run `.venv/bin/pre-commit run --all-files`
  yourself before ending a turn (the Stop hook runs it). The commit hook lints against the
  committed `pyproject.toml`, so commit ruff config changes before the code that depends on them.
- Stop at a natural point well before the context fills, and rewrite this file. The user prefers
  a fresh session to a compacted one.

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
