# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-09-29, late evening)

- `main` is at `0.37.1` and pushed to `origin` with the user's approval on 2026-09-29. It is
  checked out in the main checkout, `uv sync` has run, and the working tree is clean. Two
  `--no-ff` merges landed that day:
  - `fix/weekly-dry-run` (`fc659d0`, `0.35.2`-`0.35.3`): `weekly --dry-run` skips the data
    refresh; `leakage-audit` creates its report directory; `validate` infers column types from
    every row. Task 56.7 and the Milestone 60 leftovers are archived.
  - `refactor/ruff-all` (`0.36.0`-`0.37.1`):
    - ruff `select = ["ALL"]` with no inline `noqa`;
    - the package in `src/nfl_predictor/`, built with `uv_build` (the user chose the `src/`
      layout). It was reviewed by a separate reviewer, whose one finding (the release
      workflow's import) was fixed in `ec99df8`;
    - a scratch ETL rebuild for 2019-2025 of the flat layout (`a8e65b7`) against the `src/`
      layout (`dba5a57`) matches to `2.2e-16` on every output file (`~/scratch/etl_src_check/`),
      the ETL's own run-to-run noise;
    - `license = "MIT"` (SPDX), and `reports/` gitignored.
  - `scripts/gate.sh --web` exits `0` on `0.37.1` (1234 passed, coverage 92.69%, 26 frontend
    tests).
- After any checkout that switches between the flat and the `src/` layout, run `uv sync` at
  once. Until then the editable install points at the old location and `import nfl_predictor`
  fails. Delete any leftover untracked `nfl_predictor/` directory of `__pycache__` files at the
  repository root.
- All agent worktrees are removed. Their merged branches still exist locally and could be deleted
  with the user's agreement: `fix/weekly-dry-run`, `fix/weekly-dry-run-impl`,
  `build/uv-build-src`, `worktree-agent-*`, `refactor/ruff-all` and `feat/step3-parity`.
- Week 4 picks are in `models/weekly_2026_week_04/`. The run used `fix/weekly-dry-run`
  (`0.35.3`), launched through its `launch.sh`, and exited `0`. Checked: all 16 games are
  present, `floor_sigma` is `13.2467` with `floor_sigma_fallback` false, and the confidence
  ranks run 1..16. It is the active web run (no pin).
- After the picks, the week-4 `lines_refresh` check changed one moneyline row in each of
  `data/all_data.csv`, `data/all_data_ml.csv` and `data/predict/week_04_games_to_predict.csv`.
- No web server is running. The temporary `agent_check` account is deleted.

## Open questions for the user

- Merge `docs/step3-close-out` (the step-3 close-out, docs only) into `main` and push?

## Next

1. Roadmap step 3 is closed (the user, 2026-10-01; `ARCHIVE.md`, "Roadmap step 3"). Step 4 on a
   new branch off `main`:
   - rebuild reproducibility first. The ETL is not byte-deterministic: `unique()` without
     `maintain_order` reorders rows within a date, and parallel float sums differ at about
     `1e-16` in the `sos_*` columns. The step-4 follow-up in `TODO.md` about Polars' 100-row type
     guessing belongs here.
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
