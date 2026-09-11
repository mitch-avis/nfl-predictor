# Next Agent Session Prompt

You are the orchestrating agent (Claude Opus 5) for an implementation session in the `nfl-predictor`
workspace (`/home/mitch/workspace/nfl-predictor`). Your deliverable is **Milestone 52: the total
(over/under) head carries almost no signal**, tasks 52.2 and 52.3 in `.agents/TODO.md`. The
diagnosis (52.1) is done and verified: the root cause is a shared early-stopping callback, the
fix is small, and the cost is in the walk-forward arms that measure it.

Milestone numbers changed on 2026-09-10 (map at the top of `.agents/ARCHIVE.md`). Milestone 52 was
"50"; older commits and the crosswalk use the old numbers.

## Priority call (the user can flip it)

The 52.2 plan in `.agents/TODO.md` asks for four walk-forward arms (reference and fixed, anchored
and unanchored). This prompt trims that to **two mandatory arms plus one optional**:

1. **Fixed, anchored**, on the benchmark's exact input file. The fix cannot touch the margin head
   (it is fit first, with a fresh callback), so this arm must reproduce the benchmark's Brier,
   log loss, pick accuracy and margin MAE to four decimals. That reproduction is the reference-arm
   comparison, and it doubles as the proof the fix changed nothing else. Do not rerun the reference
   anchored arm unless the reproduction fails.
2. **Fixed, unanchored** (`--no-market-anchor`), the production configuration. This is the number
   the user cares about: does the total head beat the market line without being handed it?
3. **Reference, unanchored**: optional, last, only if time remains. Its outcome is already known
   in kind (a flat 44), and the production model's holdout `total_mae` of `10.9974` stands in.

Each arm takes about 75 minutes on an idle machine and they run **one at a time**. If the user's
opening message asks for all four arms, run all four in the order above.

## Calendar (decides what is time-sensitive)

- Today's date is in your environment. 2026 Week 1 is in progress: NE at SEA (2026-09-09) and SF at
  LAR (2026-09-10) are complete; the Sunday slate is 2026-09-13 and DEN at KC is Monday 2026-09-14.
- Week 2 opens Thursday 2026-09-17 (DET at BUF). The Week 2 weekly run has to happen after Monday
  night and before Thursday, and it should run on the fixed code so Week 2 totals are usable. Week
  1 totals were produced on 2026-09-08 by a one-tree head and are not actionable; spreads and
  moneylines from that run are fine.
- Nothing in this milestone needs an ETL rebuild.

## 0. Read first, in this order

1. `AGENTS.md`: non-negotiables, the current benchmark table (measured 2026-09-10, reproduced from
   `models/wf_shrink_2023_2025_on/metrics_report.json` on 2026-09-11), and the walk-forward
   operating notes (one run at a time, checkpoints, OpenMP wait policy).
2. `.agents/TODO.md`: Milestone 52 in full, including the 52.1 findings, then the open follow-ups.
3. `.agents/ARCHIVE.md`: the Milestone 51 entry (what the last session shipped and how it
   verified it) and Milestone 49 (how the benchmark was measured).
4. `nfl_predictor/ml/ml_model_xgb_utils.py::_with_xgb_early_stopping_params` (line 182) and
   `_build_xgb_fit_kwargs`: the helper that puts one `xgb.callback.EarlyStopping` instance into
   the params dict.
5. `nfl_predictor/ml/ml_model_core.py::_fit_margin_total_models` (lines 600-645): both
   `XGBRegressor`s are built from the same `active_params`, including the fallback path through
   `_coerce_tree_method_on_error`. Compare `_fit_quantile_models` (line 659), which builds fresh
   params per quantile and is healthy, and `_fit_models` (line 480, `ScoreModel`), which uses no
   early stopping and is unaffected. `_early_stopping_info` (line 1225) is what `metadata.json`
   records.
6. `scripts/walk_forward_backtest.py` (`--data-path`, `--eval-last-n-seasons`, `--wf-start-week`,
   `--market-anchor` / `--no-market-anchor`, `--resume`, `--checkpoint-dir`, `--out-json`) and
   `nfl_predictor/ml/walk_forward.py` for the fingerprint and checkpoint logic.
7. Tests: `tests/test_ml_model_xgb_utils.py` (the early-stopping helper),
   `tests/test_ml_model_training_margin_total.py` and `tests/test_ml_model_core_cv.py` (existing
   fixtures around `_fit_margin_total_models`), and `tests/test_wf_checkpointing.py`.

## 1. Facts to trust unless your verification disproves them

- Version `0.5.0` is committed, tagged, and **published as a GitHub release** (2026-09-11 00:02
  MDT). Everything since is unreleased.
- Milestone 51 is committed on `feat/pbp-per-snap-epa` (five commits after `0.5.0`, ending in
  the docs commit) and unreleased; the working tree should be clean apart from gitignored data,
  so check `git status` first. The gate was green on it: `631 passed`, coverage `91.16%`, all
  linters, `uv lock --check`, `uv sync --check --active`, markdownlint.
- **Root cause, verified twice on 2026-09-11.** With xgboost `3.4.1`, `fit()` no longer accepts
  `early_stopping_rounds`, so `_with_xgb_early_stopping_params` sets the init parameter
  `early_stopping_rounds` **and** adds one `EarlyStopping` callback instance to the params. Both
  estimators in `_fit_margin_total_models` are built from that dict and so share the instance,
  whose best score and patience counter survive across fits (`EarlyStopping.before_training` only
  records the starting round). The total fit starts against the margin head's best RMSE, which a
  total RMSE never reaches, and stops after one round. On disk:
  `models/week01_2026_refreshed`, `models/week01_2026_strength` and `models/weekly_2025_week_22`
  all have a 1-tree `total_model` with no `best_iteration` next to a 276-283 tree margin head.
  A synthetic reproduction with realistic noise (margin RMSE about 9, total RMSE about 13) gives a
  1-tree total head with prediction std `0.22`; the same fit with its own callback keeps 98 trees,
  std `3.75`, and lowers eval MAE from `11.36` to `10.82`.
- **The explicit callback is redundant.** `xgboost.training.train` builds its own fresh
  `EarlyStopping(rounds=early_stopping_rounds)` from the init parameter (`training.py` line 189),
  and the sklearn wrapper passes both that parameter and `self.callbacks`, so today every fit runs
  two early-stopping callbacks and the shared one wins. Verified: an `XGBRegressor` with only the
  init parameter and no explicit callback stops at the same round as one with a fresh callback.
- The benchmark in `AGENTS.md` (`models/wf_shrink_2023_2025_on/`) was measured with
  `market_anchor` **on**, `include_market` on, `early_stopping_rounds 50`, `--wf-start-week 1`,
  `--eval-last-n-seasons 3`, on `data/completed_games_ml.m49_on_through_2025.csv` (fingerprint
  `5d67ddff...`, still on disk, `7260` rows, seasons `<= 2025`). Its overall row is Brier `0.2272`,
  log loss `0.7273`, pick accuracy `0.6814`, margin MAE `9.8143`, total MAE `10.1000` over 816
  games. Under anchoring the crippled head predicts roughly the market line plus a constant, which
  is why total MAE looked normal there. The market line alone scores `10.1207` and the healthy
  `total_q0.5` quantile head `10.1378` on the same games (52.1 findings).
- The production weekly model (`models/week01_2026_refreshed/metadata.json`) has `market_anchor`
  **off** and `early_stopping_rounds 50`; its holdout `total_mae` is `10.9974`.
- The live dataset `data/completed_games_ml.csv` is the 2026-09-11 rebuild (`7262` rows, `498`
  columns, fingerprint `e388dc7a...`) and includes two 2026 games. Do not use it for the
  walk-forward arms: `--eval-last-n-seasons 3` would slide onto 2026. Use the `m49_on_through_2025`
  file for every arm so all arms share one build.
- Walk-forward operating rules (`AGENTS.md`): check `uptime` first; never run two XGBoost-heavy
  jobs at once; every finished week checkpoints under `models/wf_checkpoints/<fingerprint>/` and
  an identical command resumes; the fingerprint includes the modelling source, so the fixed arms
  get a new checkpoint directory by design; choose `OMP_WAIT_POLICY=PASSIVE` only under load; use
  `--out-json models/<name>/metrics_report.json` to name the run directory, and measure the first
  week before quoting an ETA.
- `.agents/skills/` is the user's separate clone of agent skills. It is gitignored and excluded
  from ruff and markdownlint; never edit it.

## 2. Decided (do not relitigate; record deviations)

- The fix is to stop adding the explicit `EarlyStopping` callback in
  `_with_xgb_early_stopping_params` when the `early_stopping_rounds` init parameter is supported,
  leaving xgboost to build a fresh callback per fit. If you instead keep an explicit callback, it
  must be a new instance per estimator, on both the normal and the tree-method fallback path.
- Tuning is out of scope. Note in `.agents/TODO.md` (under Milestone 55) that Optuna's
  `combined_mae` objective scored a crippled total head until this fix, so existing tuned params
  were chosen on the margin head alone.
- A separate feature set for the total head (the tail of 52.2) is only for the case where the fixed
  unanchored head still trails the market line. Diagnose and record; do not build it this session.
- 52.3 is a documentation decision: if the fixed head's unanchored total MAE is at or below the
  market line's, remove the "not actionable" caveat from Milestone 52 and say so in README and
  CHANGELOG; if it still trails, keep the totals labelled diagnostic-only in the betting workbook
  section of README and in the weekly report.
- New work that is not part of an existing milestone takes number 58 onward; never renumber
  existing milestones (numbering rules at the top of `.agents/TODO.md`).

## 3. Non-negotiables

- TDD: failing test first, then production code, small diffs.
- No leakage; time-aware evaluation; compare arms only within one dataset build and code version.
- Docstrings with formulas; type hints; no milestone numbers in code, comments, or tests; no new
  `noqa` / `type: ignore` / `pragma: no cover` without a real reason.
- All Python tooling via `.venv/bin/...`; `uv` from PATH; never bare `python` / `pytest` / `ruff`.
- Do not modify `../nfeloqb` or `../nfl-sos-ratings`.
- Commit only if the user asks. If asked: one logical change per commit, Conventional Commits
  subject (`type(scope): imperative summary`), a body explaining what and why, and the attribution
  line the harness provides. Do not push commits or tags unless asked; pushing a version tag
  publishes a GitHub release.

## Phase 1 - The fix, tests first (short)

1. Regression test on `_fit_margin_total_models` with an eval set and
   `early_stopping_rounds=50`: synthetic data where the margin target is easier than the total
   target (margin noise about 9, total noise about 13, a planted total signal). Assert the total
   head's `get_booster().num_boosted_rounds()` is greater than 1, equals the round count of a total
   head fit on its own, and that its predictions have a spread (std above 1). Make it fail on the
   current code before touching production code; keep it fast (a few thousand rows, depth 3).
2. Unit test on `_with_xgb_early_stopping_params`: two calls, or the params for two estimators,
   never share a callback object; on this xgboost version the result carries the init parameter
   and no `callbacks` entry.
3. Implement the decided fix. Run the full gate.
4. Retrain the production configuration once (the `golden_command.py` or `weekly_run.py` path the
   user uses, or `ml_model_core` directly on the live dataset) and confirm `metadata.json` now
   records `total_model.best_iteration` and the 2026 Week 1 predicted totals spread across the
   market's `40.5-47.5` range rather than `43.9-44.1`.

## Phase 2 - Measure (long; one arm at a time)

1. `uptime`, then the fixed anchored arm on `data/completed_games_ml.m49_on_through_2025.csv`
   with the benchmark flags. Confirm Brier, log loss, pick accuracy and margin MAE match the
   `AGENTS.md` table to four decimals in every window; if they do not, stop and find out why
   before running anything else. Record the new total MAE per window.
2. The fixed unanchored arm (`--no-market-anchor`, otherwise identical). Record total MAE per
   window and, from the fold outputs, the standard deviation of `predicted_total - total_line`
   (the 52.1 findings give the crippled head's median within-fold std as `0.51`).
3. Optional: the reference unanchored arm.
4. Write the walk-forward table into `.agents/TODO.md` under Milestone 52, then move the milestone
   to `.agents/ARCHIVE.md` with the table, the commands, run directories and fingerprints.

## Phase 3 - Docs and the gate

`CHANGELOG.md` `[Unreleased]` (a `Fixed` entry that names the shared callback and the versions
affected: every model trained with early stopping on xgboost `>= 2.0`, which is every model in
`models/`), `README.md` (the betting workbook totals caveat, per the 52.3 decision), `AGENTS.md`
(benchmark table: add the fixed arm's total MAE and point the benchmark at the new run directory
if the margin metrics reproduced), `.agents/TODO.md` and `.agents/ARCHIVE.md`. Then the full gate
(commands in `AGENTS.md`; the local markdownlint command excludes `#.agents/skills`).

## Final report to the user (structure)

1. Outcome first: whether the fix landed, whether the anchored arm reproduced the benchmark's
   margin metrics exactly, and the unanchored total MAE against the market line's `10.1207`.
2. The walk-forward table (windows as in `AGENTS.md`) for every arm you ran, with run directories.
3. What the retrained production model predicts for 2026 Week 1 totals.
4. What was left out or deferred (the optional arm, the separate-feature-set question), and why.
5. Recommendation for the next session: Milestone 53 (QB per-dropback EPA), then the Milestone 49
   `games_played` follow-up, with the Week 2 weekly run scheduled between Monday night and
   Thursday 2026-09-17.
