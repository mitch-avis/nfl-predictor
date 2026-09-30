# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-09-29, late evening)

- `main` is at `0.35.3`, local only: `fix/weekly-dry-run` merged with `--no-ff` (merge commit
  `fc659d0`) with the user's approval on 2026-09-29. **Not pushed**; pushing is must-ask.
  `origin/main` is still `0.35.1`.
- `refactor/ruff-all` is at `0.37.0` and checked out in the main checkout. The working tree is
  clean. It holds:
  - `0.36.0`: ruff `select = ["ALL"]` with no inline `noqa`.
  - `main` merged in (`e486088`).
  - `0.37.0`: the package moved to `src/nfl_predictor/` and builds with `uv_build` (the user
    chose the `src/` layout on 2026-09-29). Implemented on `build/uv-build-src`, reviewed by a
    separate reviewer (one finding, the release workflow's import, fixed in `ec99df8`), and
    merged with `--no-ff` (`20e9449`). A scratch ETL rebuild for 2019-2025 of the flat layout
    (`a8e65b7`) against the `src/` layout (`dba5a57`) matches to `2.2e-16` on every output file
    (`~/scratch/etl_src_check/`, `run.sh` and both logs), the ETL's own run-to-run noise.
  - `scripts/gate.sh --web` exits `0` on the final tree (1234 passed, coverage 92.69%, 26
    frontend tests).
- **Merging `refactor/ruff-all` into `main` is must-ask** and was not asked yet. It carries
  `0.36.0` and `0.37.0`. After the merge (or any checkout that switches between the flat and the
  `src/` layout), run `uv sync` at once: the editable install points at the old location until
  then, and `import nfl_predictor` fails. Delete any leftover untracked `nfl_predictor/` directory
  of `__pycache__` files at the repository root.
- Week 4 picks: `models/weekly_2026_week_04/` (run on `fix/weekly-dry-run`, `0.35.3`, through its
  `launch.sh`), exit `0`. Checked: all 16 scheduled games present, `floor_sigma` `13.2467` with
  `floor_sigma_fallback` false, confidence ranks 1..16. It is the active web run (no pin).
- The Milestone 60 live web checks are done: all 12 templates succeeded on 2026-09-29 (logs in
  `models/m60_web_live_checks/`). They went through the API (`job.py`), not the browser, so the
  TODO item stays open with a `Narrowed:` note until the user says whether that counts. The
  temporary `agent_check` account is deleted, the throwaway pin is cleared, and no web server is
  running.
- `lines_refresh` (week 4) changed one moneyline row in each of `data/all_data.csv`,
  `data/all_data_ml.csv` and `data/predict/week_04_games_to_predict.csv`, after the picks were
  made.
- Task 56.7 and the dry-run item are archived. `wf_eval_last_n_seasons` was already decided on
  2026-09-28 (it stays `3`; it never reaches the submitted probabilities), so step 3 has no
  measurement left.

## Open questions for the user

1. Merge `refactor/ruff-all` (`0.36.0`, `0.37.0`) into `main`? Push `main`? Recommendation:
   merge now, between game weeks. It changes no output (the ETL check above; the weekly run's
   probabilities come from the same code). Push only if the user wants the remote current.
2. Do the API-launched web checks meet "one live launch per template from the web UI"?
   Recommendation: yes. The UI submits exactly the same `POST /api/jobs` request. If yes, archive
   the item and close the step-3 remainder.
3. `reports/`: gitignore it? Recommendation: yes. The leakage-audit job writes
   `reports/leakage_audit.json` there by default, a generated output like `models/`.
4. Remove the agent worktrees under `.claude/worktrees/`? All are merged:
   `agent-a8257452d53054ed6` and `agent-aee722a24afa97ad6` (into `fix/weekly-dry-run`), and
   `agent-a3df1bc6e41502179` (`build/uv-build-src`, into `refactor/ruff-all`).
5. uv_build warns that the `License :: OSI Approved :: MIT License` classifier is deprecated
   (PEP 639). Recommendation: a small change to `license = "MIT"`, with the classifier dropped.
   It changes wheel metadata only.

## Next

1. The questions above. Then roadmap step 4 on a new branch off `main`:
   - rebuild reproducibility first. The ETL is not byte-deterministic: `unique()` without
     `maintain_order` reorders rows within a date, and parallel float sums differ at about
     `1e-16` in the `sos_*` columns. The step-4 follow-up in `TODO.md` about Polars' 100-row type
     guessing belongs here.
   - then the feature-value changes (55.3, 53.7, the step-4 follow-ups) and 56.6(d).
   - Every model change in steps 4 and 5 is followed by a new two-seed GPU reference, which also
     refreshes the floor's sigma pool (`floor_sigma_reference_runs`).
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
