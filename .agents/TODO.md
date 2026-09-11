# TODO - Active Work for nfl-predictor

This file is the **authoritative worklist** for the repo and contains **active** work only.
Completed milestones live in `ARCHIVE.md` (same directory). Agent workflow and guardrails live in
`../AGENTS.md`. The cross-repo review that motivates the current ordering lives in
`feature_crosswalk.md`.

- Completed work moves to `ARCHIVE.md` (with dates and notes); a partly done milestone moves its
  finished tasks there and keeps the rest here.
- Active milestones are numbered in execution order. When the order changes, move the section rather
  than renumbering it; renumber only in a deliberate cleanup that records the old-to-new map in
  `ARCHIVE.md`. Archived numbers never change.
- Milestones up to 51 are archived or retired (the last cleanup, 2026-09-10, is recorded at the top
  of `ARCHIVE.md`). Active milestones run 52 to 57, so new work starts at 58.

---

## Execution loop (required)

For each task:

1. **Understand scope**: read the relevant modules/tests/docs.
2. **Plan**: outline the smallest set of changes needed.
3. **Test-Driven Development**: add/adjust tests for all planned changes. Aim to increase coverage.
   For executable code, confirm the exact behavior/lines you plan to touch are covered first; if
   not, add focused characterization or failing tests before editing production code.
4. **Implement**: make changes incrementally (small diffs, one logical change at a time).
5. **Run checks (venv only)**:
   - `.venv/bin/ruff format .`
   - `.venv/bin/ruff check .`
   - `.venv/bin/pyright .`
   - `.venv/bin/ty check .`
   - `.venv/bin/python -m pytest`
   - `markdownlint .` (CI installs `markdownlint-cli`; the local binary is `markdownlint-cli2`,
     so run `markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings" "#.agents/skills"` here)
   - `uv lock --check`
   - `uv sync --check --active`
6. **Update docs** where behavior changes (README/AGENTS/CHANGELOG), and update TODO/ARCHIVE.
7. **Commit** (when asked) with Conventional Commits subjects, `type(scope): summary`, per
   `AGENTS.md`.

### Always use the repo venv

Agents and humans should not rely on the shell activation state.

- Do **not** run `python`, `pip`, `pytest`, `ruff`, `pyright`, or `ty` without the venv prefix.
- Use `.venv/bin/python ...` or the tool-specific binary under `.venv/bin/`.
- Use `uv ...` from `PATH` for dependency management and environment sync.

### Current validated baseline (2026-09-11, version `0.5.0` plus unreleased work)

- `.venv/bin/ruff format .`, `.venv/bin/ruff check .`, `.venv/bin/ty check .`, and
  `.venv/bin/pyright .` pass cleanly.
- `.venv/bin/python -m pytest` passes (`631 passed`) with coverage `91.16%` against the enforced
  `90%` floor.
- `markdownlint-cli2`, `uv lock --check`, and `uv sync --check --active` pass.
- `.agents/skills/` is a separate git clone of agent skills. It is gitignored and excluded from
  ruff and markdownlint (`.markdownlintignore`); pyright already skips dot-directories and ty only
  checks `nfl_predictor`, `scripts`, and `tests`.
- `data/completed_games_ml.csv` is the 2026-09-11 00:34 rebuild: `7262` rows (`1999-2025` plus two
  2026 Week 1 games, NE at SEA and SF at LAR), `498` columns, fingerprint `e388dc7a...`. Its `7261`
  shared rows equal the stat-prior-blend build of 2026-09-10 (`e46f1be9...`, backed up with the
  other CSVs in `data/backup_pre_m51/`) apart from last-ULP drift in the `sos_*` columns. The same
  run wrote `data/strength_snapshots.csv` (`18818` rows, `ff5f4823...`).
  `data/predict/week_01_games_to_predict.csv` holds the `14` remaining 2026 Week 1 games. The
  pre-blend build is kept as `data/completed_games_ml.pre_m49.csv` (`5b6af6aa...`) and
  `data/backup_pre_m49/`.
- The leakage audit passed on the 2026-09-10 build (`463` features, `0` findings); the rebuild
  added one game and no feature.
- The walk-forward benchmark lives in `AGENTS.md` and was re-measured on this build
  (`models/wf_shrink_2023_2025_on/`). Compare new feature work only against a reference arm run on
  the same build and code version.

---

## Guardrails checklist (must stay true)

- [ ] Confirm all new features apply to all matchups (no end-of-season-only logic).
- [ ] Confirm no leakage: all splits, features, and labels are time-aware.
- [ ] Confirm PBP-derived and schedule-adjusted features for week `N` use only games strictly
      before week `N` in that season; playoff rows use the full regular season.
- [ ] Confirm ETL is Polars-first and pulls NFLverse via `nflreadpy`.
- [ ] Confirm every nflreadpy source (PBP included) is cached per season and degrades non-fatally
      for the current season before data is published.
- [ ] Confirm artifacts remain reproducible (run folder, metadata, metrics).
- [ ] Confirm new code includes unit tests and does not reduce coverage.
- [ ] Confirm historical data is cached (TeamRankings + nflreadpy), and re-scrapes are minimized.
- [ ] Confirm the readiness behaviors in `AGENTS.md` (Week 1 detection, non-fatal current-season
      404s, CSV-based prediction-file resolution) still hold.
- [ ] Confirm neighboring repos (`../nfeloqb`, `../nfl-sos-ratings`) are not modified.

---

## Roadmap Status

Done so far in the feature-engineering workstream (see `ARCHIVE.md`): Milestone 45 (play-by-play
EPA families), 46 (schedule-adjusted strength), 49 (continuous early-season shrinkage), the first
phase of 43 (current-season Bradley-Terry defaults), and 51 (power rankings on the adjusted
composite). Execution order:

0. Task 56.4 - Early-season training calibration crash (**first, urgent**: it blocks every weekly
   run for weeks 2-4, including the 2026 Week 2 run due between 2026-09-14 and 2026-09-17; do it
   at the start of the Milestone 52 session)
1. Milestone 52 - The total (over/under) head carries almost no signal (**next**; 52.1 found the
   cause, a shared early-stopping callback, so 52.2 is a small fix plus a walk-forward)
2. Milestone 53 - QB per-dropback EPA families for the expected starter
3. Milestone 54 - PBP situational stats replace the TeamRankings stat scrape
4. Milestone 55 - Off-season configuration sweep, after the feature work lands
5. Milestone 56 - Weekly orchestration residuals
6. Milestone 57 - Ensembles and alternative models (parked until the user reopens it)

The open follow-ups below are not milestones. Pick them up when their area is next touched, or
promote one to a milestone when it grows.

Rules for every feature milestone:

- Land each family as one ablatable feature group with a walk-forward switch (mirror
  `--disable-trend-features`).
- Report a `2023-2025` walk-forward comparison with the group on and off before marking done.
- Keep the invariant output schema: when a source is missing for a season, emit nulls.
- XGBoost margin/total remains the only model family in scope.
- The method borrowed from `nfl-sos-ratings` is its head-to-head-excluded opponent profiling
  (`feature_crosswalk.md` section 3.1): it landed as the weekly ridge snapshot (its all-hops form)
  plus a one-hop schedule-strength companion, and the QB milestone carries the same two lenses.
- Keep walk-forward comparison artifacts under `models/`; numbers reported in these files must be
  auditable from disk.

---

## Milestone 52 - The total (over/under) head carries almost no signal

Formerly Milestone 50 (found 2026-09-09). Predicted totals for the 2026 Week 1 slate all land
between `43.9` and `44.1` while market totals for the same games range `40.5` to `47.5`. The model
is effectively predicting the league mean for every game. Training holdout `total_mae` is `10.9974`
against a `margin_mae` of `9.8471`.

Consequence: the `total_value_side`, `total_edge_prob`, `total_confidence_1_10` and `total_ev`
columns in the betting workbook are computed from that flat prediction and are not actionable. The
spread and moneyline columns are unaffected. Do not present total-based betting recommendations as
usable until this is resolved.

Tasks:

- [x] 52.1 Diagnose: feature importance for the total head; whether the total target is being
      learned at all (early-stopping round, train vs holdout MAE); whether the pruning or feature
      selection step is dropping total-relevant columns. Done 2026-09-11; findings below. No code
      changed.
- [ ] 52.2 Fix the shared early-stopping callback: give each estimator in
      `_fit_margin_total_models` its own `EarlyStopping` instance (a fresh params copy per head),
      with a regression test that the total head's round count does not depend on the margin fit
      (the synthetic reproduction below makes a good fixture). Then run the walk-forward reference
      and fixed arms on one build and code version, anchored (the benchmark config) and unanchored
      (the production config), and record total MAE. Only if a healthy total head still trails
      the market line, test a separate feature set for it.
- [ ] 52.3 Record the walk-forward table; either fix the default or mark the total columns of the
      betting workbook as diagnostic-only in the report and README.

Findings from 52.1 (2026-09-11):

- **Root cause: the total head shares the margin head's early-stopping callback.** With xgboost
  `3.4.1`, `fit()` no longer takes `early_stopping_rounds`, so `_with_xgb_early_stopping_params`
  puts one `xgb.callback.EarlyStopping` instance into the params dict, and
  `_fit_margin_total_models` builds both `XGBRegressor`s from that dict. The callback keeps its
  best score and patience counter between fits. The total fit therefore starts against the margin
  head's best validation RMSE (about `9.5`, which a total RMSE never beats) with the patience
  counter already spent when the margin head stopped early, and it stops after one round. When the
  margin head runs to `n_estimators` without stopping, the total head gets at most the patience
  left over (up to 50 rounds). `_fit_quantile_models` builds fresh params for every quantile, so the
  quantile heads are healthy (`total_q0.5` stopped at iteration `346` in the 2026 model).
- **Evidence on disk.** `models/week01_2026_refreshed/model.joblib` and
  `models/week01_2026_strength/model.joblib`: margin heads of `283` and `276` trees
  (`best_iteration` `232`, `225`), total heads of **1 tree** with no `best_iteration`, and
  `metadata.json` records early stopping for every head except `total_model`.
  `models/weekly_2025_week_22` has the same 1-tree total head. Runs without early stopping
  (`models/review_*`) keep all `598` trees in both heads. Train versus holdout MAE cannot say more
  than "the total is flat": holdout `total_mae` is `10.9974`, and feature importance for a one-tree
  head is meaningless.
- **Reproduction.** Calling `_fit_margin_total_models` on synthetic data with a planted total
  signal and `early_stopping_rounds=50` gives a 1-tree total head whose predictions span
  `43.70-44.08` (std `0.07`), the 2026 symptom. The same total head fit with its own callback keeps
  `235` trees (std `3.85`) and cuts eval MAE from `8.60` to `8.08`.
- **Scope.** Every caller of `_fit_margin_total_models` that passes an eval set: final training,
  Optuna trials (whose `combined_mae` objective has been scoring a crippled total), walk-forward
  folds, the blended model and `model_compare.py`. The margin head is fit first with a fresh
  callback, so margin predictions, win probabilities, Brier, log loss and pick accuracy are not
  affected; only total predictions and total MAE are (and tuning, through the objective).
- **Why the benchmark hid it.** The walk-forward benchmark runs with `market_anchor` on, so the
  total head predicts a residual on `total_line` and a crippled head gives roughly the market line
  plus a constant. Over the benchmark's 816 games (fold checkpoints in
  `models/wf_checkpoints/5ea347bc3339f5d3a9e3/`, the on arm) total MAE is `10.1000`, against
  `10.1207` for the market line alone and `10.1378` for the p50 quantile head; the within-fold std
  of `predicted_total - total_line` has a median of `0.51`. The production model has
  `market_anchor` off, which is why it prints a flat 44.
- **Not the cause.** Feature pruning or selection (the head never gets past its first round), and
  the total target itself (the quantile heads learn it).

Acceptance:

- [ ] Weekly predicted totals span a range comparable to the market's, or the total outputs are
      explicitly labelled non-actionable.

---

## Milestone 53 - QB per-dropback EPA families for the expected starter

Formerly Milestone 47. Goal: give the model the expected starter's per-dropback production instead
of only the nfeloqb value/Elo pair.

Tasks:

- [ ] 53.1 Identity bridge: read a copied `data/qb_meta_data.csv` (from
      `../nfeloqb/Other Data/meta_data.csv`, read-only contract) to map `away_qb`/`home_qb` names
      to GSIS ids; fallback name match against PBP `passer_player_name`; log the unmatched rate;
      tests for aliases and misses.
- [ ] 53.2 QB-game aggregation from PBP by `passer_player_id`: dropbacks, attempts, completions,
      pass yards, TDs, INTs, sacks, sack yards, `qb_epa` sum, CPOE mean (2006+). Formulas in
      docstrings; fixture tests.
- [ ] 53.3 Career-to-date and season-to-date rates strictly before the game date, plus a rolling
      dropback window for recency; new-starter fallback to a regressed league mean with a
      `qb_history_dropbacks` column so the model can see sample size.
- [ ] 53.4 Join for away/home plus diffs; constants; finalization; null policy documented.
- [ ] 53.5 Leakage audit and walk-forward ablation; record results.
- [ ] 53.6 Schedule lenses (`feature_crosswalk.md` section 3.1 applied to QBs):
      `qb_faced_pass_def_adj`, the dropback-weighted mean of the faced defenses' pre-week ridge
      pass-defense coefficient from Milestone 46 (the sos `QSoS` construct), and the one-hop
      `qb_faced_pass_def_raw`, the faced defenses' EPA per dropback allowed from prior-week games
      excluding games against the QB's team.
- [ ] 53.7 Optional phase 2: opponent-adjusted QB EPA via a dropback-weighted ridge against faced
      defenses (design in `nfl-sos-ratings/simultaneous_adjustment.solve_qb_stat_ridge`).

Acceptance:

- [ ] Unmatched-QB rate is reported and below an agreed threshold for `2006+`.
- [ ] Walk-forward table recorded; all gates green.

---

## Milestone 54 - PBP situational stats replace the TeamRankings stat scrape

Formerly Milestone 48. Goal: compute third/fourth-down, red-zone, and two-point rates (offense and
allowed) from the Milestone 45 counts so those eight columns no longer depend on scraping and extend
to 1999.

Tasks:

- [ ] 54.1 Derive the rates from PBP counts in `_compute_derived_metrics`. Fix the `red_zone_tds`
      attribution first (see the Milestone 45 follow-ups below).
- [ ] 54.2 Keep the existing TR column names and switch the source, behind a transitional
      `--tr-stats-source pbp|scrape` option; the TeamRankings ratings scrape is unchanged.
- [ ] 54.3 Sanity-compare PBP values against scraped values for `2010-2025`; walk-forward check;
      update README data sources.

Acceptance:

- [ ] The eight situational columns are populated for `1999+` offline; walk-forward is not worse.

---

## Milestone 55 - Off-season configuration sweep + lock default settings

Formerly Milestone 39, with former Milestone 40 folded in. Deferred until the feature milestones
land, because new feature families would invalidate sweep results.

Goal: run an objective, repeatable sweep of modeling configurations under the canonical evaluation
protocol, then write the selected configuration as the default for weekly runs.

Tasks:

- [ ] 55.1 Define a sweep config schema (JSON or YAML) covering model kind (`margin_total`,
      `blended_margin_total`), calibration, market mode and probability source, blend method,
      weight and clamp grids, uncertainty, tuning, and XGBoost params including GPU preference.
- [ ] 55.2 Implement `scripts/config_sweep.py` (or `--mode sweep` in `scripts/weekly_run.py`) that
      runs walk-forward per config, writes `sweep_summary.csv` and `best_config.json`, supports
      resume by dataset hash plus config hash (per-week fold checkpoints already exist), and always
      includes a baseline row with market blending and clamping off, with deltas versus that
      baseline.
- [ ] 55.3 Include `PRIOR_BLEND_GAMES` (`K`, for example `2`, `4`, `8`) in the sweep; it was reused
      from the strength blend rather than chosen (see the Milestone 49 follow-ups).
- [ ] 55.4 Add `xgb_device=auto` (prefer `cuda`, else CPU) used identically in evaluation and final
      training.
- [ ] 55.5 Decide the `ScoreModel` fate: document as experimental or deprecate cleanly.
- [ ] 55.6 Add the stability view by season and week bucket, and a "recommended defaults" section.

Acceptance:

- [ ] One command plus one config file produce the sweep summary and `best_config.json`, and
      `scripts/weekly_run.py --defaults-path best_config.json` runs end to end.

---

## Milestone 56 - Weekly orchestration residuals

Formerly Milestone 41.

- [ ] 56.1 Add data-refresh pass-through (`--data-min-season` / `--data-max-season` or a generic
      `--data-collection-args`) to `scripts/weekly_run.py`; the same pass-through carries the
      `--stat-prior-blend*` flags, which it cannot set today.
- [ ] 56.2 Decide and document how postseason games enter evaluation and training; when the
      prediction week is postseason, default the power-rankings through-week to the last
      regular-season week.
- [ ] 56.3 Wire sweep-selected defaults once Milestone 55 lands; confirm resume behavior.
- [ ] 56.4 **Urgent, do first.** Roll the in-season calibration window back across the season
      boundary. Found 2026-09-11 in a weekly-run smoke test, which stopped at Stage 2 training
      with `ValueError: Not enough weeks in season 2026 for calibration.` Cause:
      `ml_model_core._split_train_calibration_holdout` takes in-season calibration weeks only from
      the newest season in the pool (`base_pool[-1]`) and raises when it has fewer than
      `calibration_weeks` (weekly default `4`, from `--wf-calibration-weeks`). So every weekly run
      for weeks 2-4 of a season fails. The code dates from January 2026; it surfaced now because
      the ETL rebuild added two completed 2026 Week 1 games. The Week 1 production model avoided
      it because it came from `golden_command.py` with whole-season calibration.
      Fix: take the most recent `calibration_weeks` completed `(season, week)` pairs across the
      pool in time order (for Week 2: 2026 week 1 plus 2025 weeks 16-18 on the regular-season
      frame) and exclude exactly those pairs from training. Raise only when the whole pool has
      fewer weeks than requested. When the newest season already has enough weeks, the result
      must be identical to today's. Whole-season calibration (`calibration_seasons`) must not
      also claim a season the rolling window touched. The metadata `calibration_inseason` block
      (written in `ml_model_training.py` at two places, around lines 691 and 1091) stays readable:
      keep `season` and `weeks` for the newest season and add the full list of pairs. Tests: the
      Week-2 case, the unchanged case, the too-few-weeks error, and training excluding the window;
      update `tests/test_ml_model_core_helpers.py` (the test around line 226 pins today's error).
      Workaround until then: `--train-calibration-weeks 0 --train-calibration-seasons 2`, which
      calibrates on 2025 plus the 2026 games. `--train-calibration-seasons 1` alone calibrates on
      the two 2026 games and must not be used; `0` and `0` is forced back to 4 weeks by a guard in
      `scripts/weekly_run.py`.

Acceptance:

- [ ] One command produces the complete weekly package from scratch and re-running skips work
      whose inputs and config did not change.

---

## Milestone 57 - Ensembles and alternative models (parked)

Formerly Milestone 42. Parked by user direction on 2026-09-09: XGBoost margin/total remains the
primary model and benchmark. Reopen only when the user asks. Original scope: direct win-probability
classifier, logit-space probability ensemble, optional LightGBM/CatBoost extras, season-phase
specialization.

---

## Open follow-ups from completed milestones

Each group names the archived milestone it came from; the milestone's full record is in
`ARCHIVE.md`. Resolved items have moved there.

### From Milestone 45 (play-by-play EPA families)

- [ ] `red_zone_tds` counts a touchdown by either team on a red-zone play, so a pick-six is credited
      to the offense. It is computed but not published today; fix before Milestone 54 publishes it.
- [ ] Half of `PBP_COUNT_COLUMNS` (third/fourth down, red zone, two-point, total plays) is joined,
      aggregated and regressed but never published. Publish it in Milestone 54 or stop carrying it.
- [ ] The play-by-play cache key has no schema version, so growing `constants.PBP_COLUMNS` will not
      invalidate existing per-season caches. Same latent issue as `load_team_stats`.

### From Milestone 46 (schedule-adjusted strength)

- [ ] `strength_games_played_diff` has a gain-based importance of exactly `0.0`: the two teams in a
      game have almost always played the same number of games, so the diff is a constant zero
      outside bye weeks. Drop the `_diff` companion (keep the per-team columns, which do rank) the
      next time the strength schema is touched.
- [ ] The raw `adj_off_*` / `adj_def_*` columns drift in scale across a season because the frozen
      ridge penalty shrinks harder when fewer games have been played (about 30% of true magnitude
      at 4 games, about 50% by 17). They rank far below the standardized composite. Consider either
      a games-aware penalty or publishing only the composite plus `adj_srs`, and measure it.
- [ ] `sos_played_raw` is null for every week-1 and week-2 row (11.7% of the dataset) because a
      week-2 opponent's only prior game is the one against the subject. That is the method being
      correct, but a documented fallback (prior-season profile, or the adjusted lens) would make
      the column usable in the two weeks where schedule strength is least knowable.
- [ ] Schedule-strength columns are not bit-reproducible across identical rebuilds: Polars parallel
      `group_by` summation order moves the last 1-2 ULP. The ridge and SRS columns are exactly
      stable. This is pre-existing (`aggregate_team_stats_to_week` has the same property) but it
      does mean the dataset fingerprint in the model artifact contract changes across identical
      runs. Worth a line in the artifact contract docs.

### From Milestone 49 (continuous early-season shrinkage)

- [ ] `games_played` publishes `17` for a week-1 fallback row (the prior's game count) and `1` for
      a blended week-2 row whose values are 80% prior, so it tells the model the opposite of how
      much evidence stands behind the numbers. Candidates: an effective-games column such as
      `games + K * (1 - WEEK1_REGRESSION_FACTOR)`, or a separate `stat_prior_weight`. Needs its own
      ETL rebuild and from-week-1 walk-forward. The strongest candidate for the next feature slot.
- [ ] Weeks 3-18 margin MAE got worse with the blend (`9.8952` to `9.9578`) and season Brier is
      slightly worse in 2024 and 2025. The `K` check is task 55.3.
- [ ] Choose the OpenMP wait policy automatically in the walk-forward entry points (default when
      idle, `PASSIVE` under load) instead of relying on the operator; see `AGENTS.md`. It changes
      scheduling only, never results.
- [ ] `models/wf_checkpoints/` grows by a few hundred KB per distinct run and is never pruned. Any
      edit under `nfl_predictor/ml/` changes the fingerprint by design, so stale directories pile
      up. Add a cleanup note or command once it matters.
