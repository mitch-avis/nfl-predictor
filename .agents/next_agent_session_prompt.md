# Next Agent Session Prompt

You are the orchestrating agent for a session in the `nfl-predictor` workspace
(`/home/mitch/workspace/nfl-predictor`). Your job is **Milestone 59, tasks 59.1 to 59.3** in
`.agents/TODO.md`: give the walk-forward a probability instrument that can rank feature work,
replace the 4-week Platt calibration with one that cannot blow up, and make production and
walk-forward fit the same model. Nothing else can be measured until this lands, which is why the
user put it first on 2026-09-18.

**Start on a new branch off `main`** (suggested name `feat/m59-benchmark-instrument`). `main` is
clean at the `0.12.0` commits from the 2026-09-18 audit session; confirm with
`git log --oneline -6` and `git status` before starting, and do not work on `main` directly.
Merge to `main` when a task is finished and gated, not at the end of the milestone.

## Calendar

- Today's date is in your environment. The 2026 season is in progress; Week 2 finishes with the
  Monday 2026-09-21 game and the Week 3 weekly run is due before the Thursday 2026-09-24 kickoff.
  The user runs `scripts/weekly_run.py` themselves unless they ask you to; if so, it needs about 30
  minutes cold (ETL about 10 minutes from the nflreadpy cache, walk-forward compare 3 to 18
  minutes, training and reports a few minutes) and must not overlap with any walk-forward you
  have running.
- A from-week-1 walk-forward over `--eval-last-n-seasons 3` takes about 60 minutes on this idle
  machine (54 folds, the last one measured 2026-09-18 took 60 minutes); six seasons roughly
  double that. One at a time, `uptime` first, default OpenMP policy when idle.

## 0. Read first, in this order

1. `AGENTS.md`: non-negotiables, the changelog rules (a new incremented version per landed
   change, never `[Unreleased]`, `pyproject.toml` in step, no tags or releases), the new
   "standing yardstick" bullet under the mission, the benchmark table **and the caveat under it**,
   and the walk-forward operating notes.
2. `.agents/TODO.md`: Milestone 59 in full (findings and tasks), then Milestone 54 (now
   PBP-first; its task 54.0 is the small schedule-skeleton fix you may be asked to take after 59.3).
3. `.agents/ARCHIVE.md`: the review note under Milestone 49 ("Review note (2026-09-18 audit)")
   for what the rescoring found, then Milestone 52 (the total head) and Milestone 56 (partial)
   (task 56.4, the calibration window that crosses the season boundary in production).
4. `CHANGELOG.md` entries `0.10.0` to `0.12.0`.
5. Code you will change: `nfl_predictor/ml/walk_forward.py` (`run_walk_forward_backtest`,
   `select_calibration_data`, the checkpoint store), `nfl_predictor/ml/ml_model_core.py`
   (`_fit_win_prob_calibrator`, `_predict_home_win_prob`, `_margin_to_home_win_prob`,
   `_fit_margin_total_models`, `resolve_win_prob_calibration_method`),
   `nfl_predictor/ml/ml_model_training.py` (the production fit and its calibration window from
   task 56.4), `nfl_predictor/ml/metrics.py`, `nfl_predictor/ml/wf_compare_utils.py`, and their
   tests.

## 1. Facts to trust unless your verification disproves them

- Version `0.12.0` in `pyproject.toml`, `uv.lock` aligned; no tag and no GitHub release exist or
  may be created. Gate on `main` at the end of the audit session: `818 passed`, coverage `92.93%`,
  ruff, pyright, ty, markdownlint, `uv lock --check` and `uv sync --check --active` clean. The
  frontend gate (`web/`: lint, typecheck, vitest, build) was last verified at the `0.8.0` merge;
  re-run it only if you touch `nfl_predictor/api/` or `web/`.
- **The instrument finding.** The walk-forward's `--calibration platt --wf-calibration-weeks 4`
  fits `LogisticRegression(solver="lbfgs")` with default `C` on the last 4 weeks of the eval
  season (about 60 games) and emits probabilities of exactly `0.0` and `1.0` (25% of weeks-3-18
  predictions fall outside `[0.05, 0.95]`). On the `0.11.0` arm the same `predicted_margin`
  scores weeks 3-18 Brier `0.2324` / log loss `0.7612` through Platt and `0.2106` / `0.6087`
  through `Phi(margin / SCORE_DIFF_STD_DEV)`; the market spread through the same map scores
  `0.2099` / `0.6076`. Every checkpointed arm rescored deterministically sits between `0.2082`
  and `0.2135`; every archived arm-versus-arm difference is within `0.0013` with intervals
  covering zero (`models/feature_audit_2026_09_18/rescored_arms.json`; the script that produced
  it is not in the repo, so re-derive from the checkpoints' `predictions` frames, which carry
  `predicted_margin`, `home_win_prob`, `home_spread`, both moneylines and both scores).
- **Weeks 1-2 are the model's edge.** With no calibration frame those folds use the deterministic
  map, and the model beats the market there (Brier `0.2153` against `0.2218` over 96 games).
- **Early stopping today.** Walk-forward folds with a calibration frame run to the `598`-tree cap
  (best iteration `595-597` in 15 of 18 sampled folds); the Week 2 production run stopped the
  margin head at iteration `0` and the total head at `1` on its 64-game window
  (`models/weekly_2026_week_02/metadata.json`, `early_stopping`), so production predicted the
  spread while the benchmark evaluated a 598-tree model. Production trains with
  `calibration = elo` (a fixed logistic map), walk-forward with `platt`.
- **Feature gain is flat.** Over 18 retrained folds the median feature's share of total gain
  equals the uniform `2 / 482`; 158 features carry half; the top of the ranking is rare-event
  counts (`models/feature_audit_2026_09_18/feature_ranking.json`). `0.12.0` pruned 20 dead-weight
  and duplicate columns at training time (`482` to `462`); the ablation was a tie everywhere
  (`models/wf_deadweight_2023_2025_pruned/`, table in `CHANGELOG.md`).
- **Data.** `data/completed_games_ml.csv` is the 2026-09-17 22:28 build (`7278` rows, `519`
  columns, `8bacad41...`). The walk-forward input for comparisons is
  `data/completed_games_ml.m49_through_2025.csv` (`7261` rows, seasons `<= 2025`) and the
  `0.12.0` arm's cut of it, `data/completed_games_ml.m49_through_2025.deadweight_cut.csv`
  (`499` columns; the same model inputs as the current prune list, verified column-for-column).
  Any new arm must run on a build cut to seasons `<= 2025` so the eval window does not slide onto
  2026.
- **Upstream gap, not yours to fix in this milestone.** nflverse team stats hold only
  Jacksonville's 8 road games for 2001 and 2002 (verified live 2026-09-18); every JAX season-to-
  date family for those seasons is road-only. That is task 54.0. Division context for 1999-2001
  uses today's map; that is task 59.4, after 59.3.
- `../nfeloqb` and `../nfl-sos-ratings` are the user's; never modify them.

## 2. Decided (do not relitigate; record deviations)

- The closing market line is the standing yardstick; every walk-forward report compares against it
  on the same games. "Not worse than the market, better early-season calibration" is the bar;
  beating the closing line is a stretch goal, never a claim.
- Milestone 59 before Milestone 54; 54.0 (schedule skeleton) before the rest of 54; 53.7 and the
  Milestone 55 sweep only on the 59.1 instrument.
- Elo stays a feature (`elo_pre`, `qb_elo_pre` and trends). Calibration is a post-processing
  layer fit on pooled out-of-fold predictions, never on the last 60 games; the deterministic
  one-parameter map is the floor it must beat.
- The quarterback per-dropback family stays (2026-09-11); the schedule lenses are out
  (2026-09-17); `games_played` is pruned (2026-09-17); the 20 `0.12.0` prunes stand.
- Totals stay diagnostic-only until a total model beats the closing line.
- New milestones take number 60 onward.
- Walk-forward runs one at a time; check `pgrep -af walk_forward` and the web UI's Jobs page
  (the API on port 8765 runs from `../nfl-predictor-web`; do not bounce it unasked).

## 3. Non-negotiables

- TDD, no leakage, time-aware evaluation, comparisons only within one build and code version.
- All Python tooling via `.venv/bin/...`; `uv` from PATH.
- Update `CHANGELOG.md` as each chunk lands, under a new incremented version, with
  `pyproject.toml` bumped and `uv lock` / `uv sync` run; never `[Unreleased]`, never a tag.
- No milestone numbers or TODO labels in code, comments, docstrings or test names.
- Keep walk-forward artifacts under `models/`; every number written into `.agents/` or
  `AGENTS.md` must be auditable from disk.
- Commits (when the user asks): one logical change per commit, Conventional Commits subjects, a
  body with what and why, the harness attribution line.

## Phase 1 - Task 59.1, the instrument

Add to every walk-forward fold and to the aggregated report: Brier, log loss and pick accuracy for
(a) the configured calibrator, (b) `Phi(margin / sigma)` with `sigma = SCORE_DIFF_STD_DEV`, and
(c) the market-implied probability (no-vig from the two moneylines; `Phi(spread / sigma)` when
moneylines are missing). Add the paired model-minus-market Brier and log loss with a bootstrap
interval per window (week 1, week 2, weeks 3-18, all) to the report, `summary_table` and
`wf_compare`, and add `market_brier` / `market_log_loss` columns to `wf_compare.csv`. Keep the
checkpoint payload readable (bump `FOLD_CHECKPOINT_VERSION` only if the layout changes). Then
rebuild the `AGENTS.md` benchmark table from the deterministic columns of the existing
`0.12.0` arm (`models/wf_deadweight_2023_2025_pruned/`, checkpoints `bae0e56db951d1a890d4`) and
put the market row beside it; the numbers must match the rescoring in the archive note.

## Phase 2 - Task 59.2, calibration that cannot blow up

Replace the 4-week Platt fit. For eval season `S`, the calibrator's training set is the pooled
walk-forward predictions for seasons `S-2` and `S-1` plus the completed weeks of `S` (about 540
games at week 1, growing), never the last 60 games alone. Implement in this order and measure each
on the 2023-2025 checkpoints:

1. the one-parameter map: estimate `sigma` from the pooled residuals `actual - predicted_margin`;
2. Platt on the pooled set with an L2 penalty (`C` chosen time-aware, not on the eval season);
3. isotonic only past the existing 200-game threshold.

Acceptance: weeks 3-18 log loss within `0.005` of the deterministic `0.607`; no probability outside
`[0.02, 0.98]` unless the spread exceeds 14 points; `auto` resolves to this path; production
(`weekly_run`, `golden_command`, `betting_pipeline`) and walk-forward call one calibration
function. The production calibration window from task 56.4 becomes this pooled set.

## Phase 3 - Task 59.3, fit parity and early stopping

Make production and walk-forward call the same fit function with the same stopping rule. Remove
early stopping from in-season fits and fix `n_estimators` from a time-aware tuning whose eval set
is a whole prior season (at least 250 games), or early-stop on that season-sized set; record
`best_iteration` for every head in `metadata.json` and in the walk-forward report, and warn when it
is below `10` or at the cap. Re-tuning of the other parameters stays in Milestone 55 but must use
the 59.1 instrument. Run one from-week-1 walk-forward on the `0.12.0` cut with the new fit and
report it against `models/wf_deadweight_2023_2025_pruned/` on the deterministic and market
columns.

## Phase 4 - If time remains

Task 54.0 (schedule skeleton and coverage check) is small and independent; task 59.4 (season-aware
divisions) is three seasons of known facts listed in the task. Ask before starting 59.5 (the
noise-family ablation), because it needs a six-season walk-forward.

## 5. The web UI lives on `main`

`nfl_predictor/api/` (FastAPI), `web/` (React), `tests/api/`, and the `web` CI job merged as
`0.8.0`. Design and phase status: `.agents/web_ui_plan.md` (phases 0-3 done; 4-6 open as
Milestone 58). The user runs the API from the worktree `../nfl-predictor-web` on port 8765 against
this checkout's `data/`, `models/` and `reports/`; ask before restarting it. Touching
`scripts/weekly_run.py` flags or `metadata.json` keys can affect the API's readers and job
catalog (`nfl_predictor/api/jobs/catalog.py`), so run the whole suite, `tests/api/` included.

## Final report to the user

1. What landed per task, with the walk-forward table on the deterministic and market columns and
   the run directories.
2. Whether production and walk-forward now fit and calibrate identically, and the recorded
   `best_iteration` values.
3. Anything the audit's facts above turned out to be wrong about.
4. What is left in Milestone 59 and the recommendation for the next session.
