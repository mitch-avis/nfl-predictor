# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## Plan for the next session (written 2026-09-29 evening)

The user wants one session to finish both open branches, in this order. Ask the open questions
in step 1 at the start, then work through the rest.

### Branch state

- `main` is at `0.35.1` (pushed 2026-09-28).
- `fix/weekly-dry-run` sits 13 commits ahead of `main`:
  - `0.35.2`: `weekly --dry-run` skips the data refresh.
  - `0.35.3`: `leakage-audit` creates its report directory; `validate` infers
    `all_data.csv`'s column types from every row; local markdownlint skips `models/`.
  - Task 56.7 closed by the user on 2026-09-28: `market_transform` stays `auto` after a
    two-seed tie.
  - Each fix was reviewed by a separate reviewer, and the gate passed (1199 passed).
- `refactor/ruff-all` branches from `fix/weekly-dry-run` at `afe926a` and adds `0.36.0`:
  - The user set ruff `select = ["ALL"]`, and the code base passes it with no inline `noqa`.
  - Commits `ca2e440`..`242f6ca`. The working tree is clean.
  - `scripts/gate.sh --web` exits `0` (1229 passed, coverage 92.69%).
  - Neither branch is merged, and merging is must-ask.
- Is `fix/weekly-dry-run` ready to merge? The code is: reviewed and gate-green. But the
  branch's own work, the Milestone 60 live web checks, stopped on an unanswered question.
  So finish or explicitly defer those checks first (step 3 below).

### 1. Questions for the user at the start (none was answered before this handoff)

Unanswered from the "week 4 picks and step 3 remainder" session's last message:

- **Five web templates write into the active run:** `predict`, `power_rankings`,
  `shap_analysis`, and the `lines_refresh` and `predict_week` chains. The active run is
  `weekly_2026_week_04_step3_test`, the step-3 merge-test record, and nothing should
  overwrite it.
  - Option 1 (that session's recommendation): the user pins the throwaway
    `models/train_20260929_061431/` in the web UI, or allows the agent to pin it. The
    permission classifier denied the agent doing it unasked. The five run against it, and
    then it is unpinned.
  - Option 2: they write into the merge-test run.
  - Option 3: they run against the real Week 4 weekly run.
- **Merge:** whether to merge `fix/weekly-dry-run` into `main`.
- **`reports/`:** gitignore it or not. The leakage-audit job creates it by default. It is
  currently empty and untracked.

New:

- **Week 4 picks.** Thursday's game is 2026-10-01, and `models/` has no
  `weekly_2026_week_04` run yet. `data/qb_elos.csv` is dated 2026-09-24, so the user's
  `../nfeloqb` update is still pending. Ask:
  - whether the user runs Week 4 themselves or wants the agent to;
  - on which code. Recommendation: `fix/weekly-dry-run` (reviewed, `0.35.3`) or `main`, not
    the refactor branch, until the refactor's merge is approved.
  - The run launches through a `launch.sh` with `nohup setsid`. Check afterwards: every
    Week 4 game is present, `floor_sigma` is near 13.25 with `floor_sigma_fallback` false,
    and ranks run 1..N.
- **`src/` layout.** See step 4.

### 2. State on disk that the next session must know

- No web server is running. The live-check server ran from
  `models/m60_web_live_checks/serve.sh` without `--reload`, with `job.py` as the API
  submitter. Restarting it for the remaining checks is must-ask (rule 5, the running web
  API).
- The temporary admin account `agent_check` still exists in `data/web/app.db` (user-approved
  for the checks). Delete it when the checks end, or when the user defers them.
- Two prunable agent worktrees are under `.claude/worktrees/`, both already merged into
  `fix/weekly-dry-run`. Remove them with `git worktree remove` once the user agrees.
- An untracked `src/nfl_predictor/` tree holds only stale `__pycache__` files from an older
  layout (it even has `lightgbm_cuda_bootstrap`, which no longer exists). Delete it before
  any `src/` move.

### 3. Finish `fix/weekly-dry-run`, then merge (must-ask)

- Run the five remaining web templates, per the user's step-1 answer.
- Delete `agent_check`.
- Step-3 close-out in `.agents/TODO.md`: move task 56.7 and the dry-run item to `ARCHIVE.md`.
- Still open in step 3: 56.7(b)'s remaining setting, `wf_eval_last_n_seasons`. It needs a
  written hypothesis and decision rule before any run, and a default change is must-ask.
- Three reads of the data CSVs use Polars' default 100-row type guessing (found by
  `0.35.3`'s implementer, not fixed):
  - `reporting/power_rankings.py:905` is the one that could break. The `power_rankings`
    template exercises it live.
  - `week_builder.py:84` only tests the scores for null, so the reviewer judged it safe.
  - `data_collection.load_dataframe` is only called from tests.
  - A crash there is a fix under rule 2: failing test first, then the fix.
- Then rebase or merge `refactor/ruff-all` onto the result. The refactor already contains
  the whole branch.

### 4. Finish `refactor/ruff-all`: switch the build backend to `uv_build`

- `pyproject.toml` has a commented-out `[build-system]` for `uv-build>=0.12`. The user wants it
  to replace hatchling.
- The user believes `uv_build` needs the package under `src/`. It doesn't:
  `[tool.uv.build-backend] module-root = ""` keeps the flat layout. Verify this against the
  uv docs for the installed uv version before relying on it.
- Ask the user: flat layout with `module-root = ""`, or move to `src/nfl_predictor/`?
  - Recommendation: move to `src/`, since the user asked for it. It is uv's default and
    removes `[tool.hatch.build.targets.wheel]`.
  - Cost: every path in the config and docs changes, and one path computation breaks
    silently (below).
  - How sure: moderate. If the user only wanted `uv_build`, the flat layout is one config
    line.
- Facts for a `src/` move, checked 2026-09-29:
  - **Silent breakage:** `nfl_predictor/constants.py:10` sets
    `ROOT_DIR = Path(__file__).parent.parent`, and `DATA_PATH` and the models directory
    derive from it. Under `src/`, it would point at `src/` instead of the repo root. Pin
    `ROOT_DIR` (and `DATA_PATH`) with a characterization test before the move, then use
    `parents[2]`.
  - Grep `git grep -n "__file__"` for other repo-root computations. The tests use
    `parents[1]` of `tests/`, which is unaffected.
  - Walk-forward checkpoint fingerprints hash each modelling file's name and bytes, not its
    path (`nfl_predictor/ml/walk_forward.py`, `fold_checkpoint_fingerprint`), so a pure
    move keeps them. The reference-pool reader (`ml/floor_sigma.py`) reads checkpoints by
    run path.
  - Config to update in `pyproject.toml`:
    - ruff: `src`, `known-first-party`, and every `per-file-ignores` key;
    - `[tool.ty.src] include`;
    - pytest `pythonpath = ["."]`: drop it so tests import the installed package;
    - `--cov=nfl_predictor` and coverage `source`: check that they still resolve;
    - `[tool.hatch...]`: remove.
  - Other files: `scripts/gate.sh` (lines 82 and 108), `.pre-commit-config.yaml`, the
    `.github/workflows/*.yml` files, and `.markdownlintignore` and `.gitignore`, if they
    name paths.
  - The nested `nfl_predictor/api/AGENTS.md` and `nfl_predictor/ml/AGENTS.md` move with the
    package. Check that Claude Code and Copilot still load them.
  - About 29 tracked files mention `nfl_predictor/` as a path: `AGENTS.md`, `README.md`,
    the module READMEs, and `.agents/*.md`. Update the living docs only. Frozen records
    (`.agents/ARCHIVE.md`, past `CHANGELOG.md` entries, `.agents/m60/`, run directories)
    stay as written. Generate the list with a script (rule 10), not by hand.
  - After the move: run `uv lock` and `uv sync`, check that `nfl-predictor --help` and
    `python -m nfl_predictor` work, and run the gate with `--web`.
  - Then do an ETL check into a scratch directory, never `data/`: a copy of the repo with
    `data/` inputs rebuilds 2019-2025, and `DATA_PATH` must resolve to the copy's `data/`.
    Compare against the previous commit with a key-sorted numeric diff, to about `1e-15`,
    because of the ETL's own non-determinism (below).
- Bump the version (probably `0.37.0`: build backend and layout), write the changelog entry,
  and run `uv lock` and `uv sync`. Rewrite this file.

### 5. Carried from the ruff work

- The user's rule for lint work: fix the code rather than suppress. A genuine false positive
  goes in `[tool.ruff.lint.per-file-ignores]` with its reason, never inline. Run
  `.venv/bin/pre-commit run --all-files` yourself before ending a turn: the Stop hook runs it,
  and its `ruff --fix` rewrites the tree if you haven't.
- The commit hook stashes unstaged files and lints against the committed `pyproject.toml`. So
  commit ruff config changes before the code that depends on them.
- Finding for roadmap step 4 (rebuild reproducibility): the ETL is not byte-deterministic even
  on unchanged code. Two sources:
  - `unique()` without `maintain_order` reorders rows within a date;
  - parallel float sums differ at about `1e-16` in the `sos_*` columns.
- Every walk-forward checkpoint fingerprint changes with `0.36.0`, so the next walk-forward
  retrains from scratch.
- Stop at a natural point well before the context fills, and rewrite this file. The user
  prefers a fresh session to a compacted one.

## State (written 2026-09-28, after the 0.35.1 merge)

- `main` is at `0.35.1` and pushed to `origin` with the user's approval on 2026-09-28 (pushing
  stays must-ask each time). Roadmap step 3 (production/benchmark parity) merged from
  `feat/step3-parity` (merge commit `ec1879e`); `0.35.1` (merge commit `0a99b02`, from
  `fix/gate-hygiene`) closes the web database's leaked SQLite connection, swaps the test
  client's `httpx` for `httpx2`, and keeps the gitignored `models/` and `.claude/worktrees/` out
  of the local pyright and markdownlint steps. All agent worktrees were removed. The working tree
  is clean.
- `scripts/gate.sh --web` exits `0` on the merged tree (1194 passed, no pytest warnings).
- What step 3 changed (full list in `CHANGELOG.md`, `0.29.0`-`0.35.0`):
  - every run type submits the deterministic floor `Phi(margin / sigma)`, with sigma the
    root-mean-square out-of-fold margin error before the week (`nfl_predictor/ml/floor_sigma.py`);
    production pools the GPU reference runs (`models/step3_gpu_reference/l1_seed42`,
    `l2_seed7`, read-only) with stage 1's errors;
  - picks, confidence ranks and `confidence_strength` come from the unrounded probability;
    equal confidences break by `game_id` (`metrics.confidence_ranks`);
  - the weekly stage 1 is one walk-forward of the production configuration from week 1; the
    stage-1 candidate matrix, fitted/Elo calibrators, market blend and clamp, the `blend` model
    kind, `sweep` and the calibration hold-out are retired; the final fit trains on every game
    with stage 1's tree settings;
  - the GPU (`--xgb-device auto`) is the default device; the two-seed 2007-2025 GPU reference is
    the benchmark arm (`.agents/benchmarks.md`, "GPU reference");
  - SHAP (XGBoost TreeSHAP) is the headline feature importance; the stability view and the
    settings-versus-production section are in every backtest report.
- Closed: tasks 55.4, 55.6, 56.5; 56.6(c) done. Open in step 3: see "Next".
- The user runs `nfl-predictor web` (without `--reload`) from this checkout; its Python is older
  code still in memory and `web/dist` was not rebuilt. The web UI shows the new pages only after
  the user restarts it and rebuilds `web/dist` (`npm run build` in `web/`). Never stop or restart
  the server yourself (rule 5).

## Merge test (2026-09-28)

`models/weekly_2026_week_04_step3_test/` (launched through its `launch.sh`): the full weekly run on
`0.35.0` exited `0`. Because the Week 3 Monday game (PHI at CHI) had not been played, the week
detection correctly targeted Week 3 and predicted that one game. Stage 1 took 67 s on the GPU
(Brier `0.2076` against the market's `0.2077`, producer's log, no second key); the final fit's
floor sigma was `13.2512` from 4975 earlier games. The ETL refresh rewrote `data/` as part of the
run.

## Next (after the plan above)

1. **Week 4 picks (before Thursday's game).** After the Monday game and the user's
   `data/qb_elos.csv` update from `../nfeloqb`, the user (or you, if asked) runs
   `.venv/bin/nfl-predictor weekly --run-id weekly_2026_week_04` through a `launch.sh` in the run
   directory with `nohup setsid`. It is the first real week on the step-3 code: check that the
   predictions file has all Week 4 games, `floor_sigma` near 13.25 with `floor_sigma_fallback`
   false, and ranks 1..N.
2. **Step-3 remainder**, then the step-3 close-out in `.agents/TODO.md` (move closed items to
   `ARCHIVE.md`):
   - task 56.7(b), measurement part: the remaining output-changing settings
     (`wf_eval_last_n_seasons`; `market_transform` is closed), with a written hypothesis and decision
     rule first; default changes are must-ask;
   - the Milestone 60 per-template live web checks (must-ask launches, on a server without
     `--reload`).
3. **Then roadmap step 4** on a new branch off `main`: rebuild reproducibility first, then the
   feature-value changes (55.3, 53.7, the step-4 follow-ups) and 56.6(d) (anchoring to pick-time
   lines). Every model change in steps 4 and 5 is followed by a new two-seed GPU reference, which
   also refreshes the floor's sigma pool (`floor_sigma_reference_runs`), per the standing process
   item in `TODO.md`.

## How the last session worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer`, fixes
  back to the implementer, merge with `--no-ff` into the feature branch; the delegating session
  writes `CHANGELOG.md`, the version (`pyproject.toml`, `uv lock`, `uv sync`), `TODO.md`,
  `ARCHIVE.md`, `AGENTS.md`, `.agents/benchmarks.md` and this file, and lints the Markdown it
  edits with `markdownlint-cli2` before committing.
- Every measured number goes through an independent reviewer's own rescore before it enters the
  docs (rule 3); decision rules are written into the run directory before scoring.
- Run the gate in a separate worktree (the `--web` gate rebuilds `web/dist`, which the user's
  server serves). Watch long runs by polling a PID (`while kill -0 <pid>`), never with
  `pgrep -f` on a pattern that matches the watcher's own command line (that stalled a check-in
  for 7 hours on 2026-09-28).
- Old agent worktrees under `.claude/worktrees/` (all merged) can be removed with
  `git worktree remove` once the user agrees; they are not needed.

## Notes

- Never run two walk-forwards at once; check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- A backtest that judges probabilities must run from 2007 week 1 or pass
  `--floor-sigma-reference-runs` with the GPU reference, and pass `--wf-start-week 1` (rule 11).
- Re-run a plain `uv sync` after every version bump and after every merge that bumps the version.
