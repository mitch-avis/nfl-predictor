# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## Update (2026-09-29, branch `fix/weekly-dry-run`)

- `fix/weekly-dry-run` (off `main`, not merged: must-ask) carries `0.35.2` (`weekly --dry-run`
  skips the data refresh) and `0.35.3` (`leakage-audit` creates its report directory; `validate`
  infers `all_data.csv`'s types from every row; local markdownlint skips `models/`). Each fix was
  reviewed by a separate reviewer; `scripts/gate.sh` exit `0` (1199 passed).
- Task 56.7 closed (user, 2026-09-28): `market_transform` stays `auto` after a two-seed tie
  (`.agents/benchmarks.md`, "Market transform on against off"). Still to move to `ARCHIVE.md` at
  the step-3 close-out, with the dry-run item.
- Web live checks (`models/m60_web_live_checks/`, the job logs and `job.py`, the API submitter):
  the server runs from `serve.sh` without `--reload`, `web/dist` rebuilt; a temporary admin
  `agent_check` (user-approved) submits jobs and is deleted when the checks end. Passed:
  `etl_full` (rewrote `data/`, user-approved; Week 3 complete, `week_04_games_to_predict.csv` has
  16 games), `train` (`models/train_20260929_061431/`), `weekly_run` (dry run),
  `walk_forward_backtest` (two seasons, progress per fold), and after `0.35.3` `leakage_audit`,
  `validate_offline`, `validate_live`. Only walk-forward jobs report progress. Open: `predict`,
  `power_rankings`, `shap_analysis` and the `lines_refresh`/`predict_week` chains write into the
  active run (`weekly_2026_week_04_step3_test`); pinning the throwaway train run was denied by
  the permission classifier, so the user decides (pin it, write into the merge-test run, or wait
  for the real Week 4 run). `reports/` (the leakage report) is untracked, not gitignored.

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

## Next

1. **Week 4 picks (before Thursday's game).** After the Monday game and the user's
   `data/qb_elos.csv` update from `../nfeloqb`, the user (or you, if asked) runs
   `.venv/bin/nfl-predictor weekly --run-id weekly_2026_week_04` through a `launch.sh` in the run
   directory with `nohup setsid`. It is the first real week on the step-3 code: check that the
   predictions file has all Week 4 games, `floor_sigma` near 13.25 with `floor_sigma_fallback`
   false, and ranks 1..N.
2. **Step-3 remainder**, then the step-3 close-out in `.agents/TODO.md` (move closed items to
   `ARCHIVE.md`):
   - the `--dry-run` defect ("From the step-3 merge test");
   - task 56.7(b), measurement part: the remaining output-changing settings
     (`wf_eval_last_n_seasons`, `market_transform`), each with a written hypothesis and decision
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
