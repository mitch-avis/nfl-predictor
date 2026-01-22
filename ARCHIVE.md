# ARCHIVE - Completed Milestones

This file contains completed milestones and optional enhancements that were previously tracked in
`TODO.md`. Keep this as the audit trail. If future changes regress behavior, re-run the acceptance
checks from the relevant section.

---

## Completed core milestones (0-11)

> These items were completed and verified in prior work. They are archived here to keep `TODO.md`
> focused on active work.

### Milestone 0 - Repo scan & plan

- [x] Identify ML entrypoints (train, predict, backtest scripts).
- [x] Identify where margin/total and win prob calibration live.
- [x] List current artifact outputs and missing metadata.
- [x] Confirm where ML datasets and week prediction inputs are produced.

### Milestone 1 - Canonical Margin/Total pipeline

- [x] Margin/total modeling is the primary path.
- [x] Score derivation is stable and unit-tested.
- [x] Outputs include margin/total and derived scores.

### Milestone 2 - Preprocessing cleanup

- [x] XGBoost path avoids scaling and avoids accidental densification.
- [x] Missing values are handled intentionally.

### Milestone 3 - Training improvements

- [x] Early stopping is enabled.
- [x] `eval_metric` aligns to the optimization target.
- [x] Parallelism is configurable.
- [x] Training config is serialized into metadata.

### Milestone 4 - Walk-forward evaluation

- [x] Walk-forward backtest exists and is time-aware.
- [x] Per-week and per-season metrics are emitted.

### Milestone 5 - Probability calibration + diagnostics

- [x] Calibration options are implemented.
- [x] Brier/log loss and reliability summaries are reported.

### Milestone 6 - Quantile intervals

- [x] Margin and total include p10/p50/p90 outputs.
- [x] Interval columns exist and are validated.

### Milestone 7 - Market transforms + anchoring

- [x] Market transforms are explicit and configurable.
- [x] Market anchoring is supported.
- [x] Market probability blending/clamping exists as configured.

### Milestone 8 - Leakage audit

- [x] Leakage audit mode exists and emits a JSON report.
- [x] Tests verify detection of leaked columns.

### Milestone 9 - Artifact contract + metadata

- [x] Run directories include model + metadata + metrics report.
- [x] Artifacts are loadable without hidden state.

### Milestone 10 - Golden command entrypoint

- [x] One command runs train + backtest + weekly predictions and writes artifacts.

### Milestone 11 - Dependency pinning + documentation

- [x] ML dependencies are pinned.
- [x] CPU-only path works and is documented.

---

## Completed milestones (12-19)

### Milestone 12 - Documentation + repository cleanup (Polars-only narrative)

- [x] Remove documentation references to deprecated data collection and utility modules.
- [x] Ensure all docs describe `nfl_predictor/data_collection.py` as the authoritative ETL entrypoint.
- [x] Add a short "Data sources + missing data" section describing fallbacks and season coverage
  limits.

Acceptance:

- [x] Docs reference only the Polars+nflreadpy pipeline and current ML entrypoints.

### Milestone 13 - constants.py cleanup and organization

- [x] Audit `nfl_predictor/constants.py` for unused constants and remove them.
- [x] Group constants into clear sections (paths, season/week rules, team mappings, feature names,
  defaults).
- [x] Ensure schema/feature lists are centralized and used everywhere (no hard-coded columns).

Tests:

- [x] Team alias mapping resolves to canonical abbreviations.
- [x] Schema lists contain no duplicates.
- [x] Required output columns exist in the ML datasets.

Acceptance:

- [x] `constants.py` is organized, minimal, and referenced consistently across ETL/ML/docs.

### Milestone 14 - Missing data handling across seasons

- [x] Inventory sources with limited historical coverage (injuries, markets, etc.).
- [x] Define a per-feature-group missing-data policy: null, default, or carry-forward.
- [x] Implement ETL fallbacks so output schema is invariant across seasons.
- [x] Ensure ML preprocessing handles nulls explicitly and logs fallback usage counts.

Tests:

- [x] ETL produces the same columns for a season with missing sources and one without.
- [x] Model train/predict completes when market fields are null.
- [x] Fallback counters appear in metrics/report outputs.

Acceptance:

- [x] Pipeline and ML runs succeed across the full historical range with consistent schema.

### Milestone 15 - Season-to-date record features (W-L-T, division, conference)

- [x] Implement record features for away and home teams (prefix columns `away_` and `home_`).
- [x] Record columns are defined in `constants.py` and included in the ML feature range.

Tests:

- [x] Computed season-to-date records match known records for a small fixture season/week range.
- [x] Divisional records reconcile with overall when a team's prior games are divisional.
- [x] Week 1 records are zero for all teams.

Acceptance:

- [x] Datasets include record features and they are available for training and prediction.

### Milestone 16 - Divisional rivalry feature

- [x] Add a `is_divisional_matchup` feature for each game.
- [x] Implement using a division mapping table in `constants.py`.
- [x] Ensure this applies to all seasons and teams.

Tests:

- [x] Known divisional pairings are flagged correctly.
- [x] Cross-division pairings are not flagged.

Acceptance:

- [x] All game rows contain the divisional indicator and it is stable across seasons.

### Milestone 17 - Lookahead / trap indicators

- [x] Build next-week opponent features using the schedule.
- [x] Add per-team lookahead features and join to games for away/home teams.

Tests:

- [x] Next-week opponent lookup is correct for a fixed season/week range.
- [x] Missing next-week opponent (end of season) yields null/default.

Acceptance:

- [x] Lookahead features exist in the ML dataset for all games with defined fallbacks.

### Milestone 18 - Motivational asymmetry features

- [x] Create a playoff-incentive feature set computed from standings and tiebreak proxies.
- [x] Integrate into ETL as season-to-date features available prior to each game.

Tests:

- [x] Incentive state features do not use future games.
- [x] Motivation feature join is schema-invariant when schedule scores are missing.

Acceptance:

- [x] Motivation/standings proxy features are available for all games without leakage.

---

## Completed optional enhancements

- [x] Realistic score post-processing for display outputs.
- [x] Market-only model removed when anchoring sufficed.
- [x] Blending weights constrained where applicable.
- [x] Interval coverage diagnostics implemented.

---

### Milestone 19 - Blocked/time-series cross-validation for tuning

- [x] Implement blocked CV at the season-week level for hyperparameter tuning and model selection.
- [x] Ensure folds are strictly time-ordered (train < validation).
- [x] Integrate CV into Optuna objectives so tuning does not overfit a single season holdout.
- [x] Report CV mean/std metrics in tuning CV summary (stored under `metrics_report.json`).

Tests:

- [x] CV fold generation is strictly time-ordered.
- [x] CV fold generation is deterministic.

Acceptance:

- [x] Optuna tuning evaluates parameters using time-series CV over season-week timepoints.

Primary files:

- [x] `nfl_predictor/ml/ml_model_core.py`
- [x] `tests/test_time_series_cv.py`

### Milestone 20 - Unit tests and code coverage hardening

- [x] `pytest-cov` is configured and coverage is reported by default.
- [x] Coverage threshold is enforced (current floor: 80%).
- [x] Tests exist across ETL joins and feature derivations introduced in prior milestones.

Acceptance:

- [x] `pytest --cov=nfl_predictor --cov-report=term-missing --cov-fail-under=80` passes.

Primary files:

- [x] `setup.cfg`
- [x] `tests/`

### Milestone 21 - Remove deprecated modules from the import surface

- [x] No code or docs reference deprecated modules.
- [x] Compatibility facades import cleanly.

Acceptance:

- [x] The package imports cleanly and no deprecated modules are referenced.

Primary files:

- [x] `tests/test_imports.py`

---

## Completed milestones (22-30)

### Milestone 22 - Repo/tooling alignment (blocking)

Completion note: Setup and tooling workflow verified; README and config alignment complete.

- [x] Update `README.md` setup instructions to use the pinned requirements workflow:
  - install from `requirements.txt` + `requirements-dev.txt`
  - use `--no-deps` for editable installs to avoid unpinned dependency drift
  - document how to regenerate pins (`uv pip compile`)
- [x] Confirm dev requirements include:
  `black`, `ruff`, `pytest`, `pytest-cov` (and any other test plugins required by
  `pyproject.toml` addopts).
- [x] Fix minor config gotchas:
  - Ruff isort config should not treat `__main__` as a third-party package.
  - Coverage `exclude_lines` should match `if __name__ == "__main__":` exactly.
- [x] Confirm `.gitignore` covers run artifacts (models, optuna db, caches) and that `data/` being
  ignored is intentional and documented.

Acceptance:

- [x] `ruff check .` and `black --check .` pass in a clean environment.
- [x] `python -m pytest` passes (and `python -m pytest --no-cov` works as documented).
- [x] README instructions work end-to-end on a clean machine.

### Milestone 23 - Canonical training + validation methodology (the "source of truth")

Completion note: Walk-forward evaluation protocol and metrics schema standardized.

- [x] Define the **primary evaluation lens** for the project:
  - Walk-forward (rolling-origin) over multiple seasons is authoritative.
  - Season-blocked CV exists mainly for hyperparameter tuning.
- [x] Standardize the **outer evaluation window**:
  - Default: last N seasons (configurable), regular season only by default.
  - Explicit handling for incomplete current season and postseason evaluation.
- [x] Standardize the **inner calibration window**:
  - Time-aware calibration weeks (e.g., last K weeks before prediction week) and/or
    calibration seasons.
  - Minimum sample size rules (see Milestone 24).
- [x] Decide (and document) the **selection hierarchy** for "best model":
  - Primary: probability quality (Brier, log loss, reliability).
  - Secondary: confidence pool expected points (and stability across weeks).
  - Tertiary: margin/total MAE (and market-relative residual MAE if anchoring).
- [x] Produce a single **metrics report schema** that always includes:
  - per-week, per-season, and overall aggregates
  - mean + variance across folds
  - market-relative metrics when market is present

Acceptance:

- [x] There is one "blessed" evaluation command (or script) that reproduces the reported metrics.
- [x] A config sweep (Milestones 24/25) can run under this protocol without ad hoc code.

### Milestone 23.5 - Reporting pipeline correctness + schema safety (blocking)

Completion note: Reporting scripts now fail fast on schema issues and apply calibration correctly.

#### Tasks (Milestone 23.5)

- [x] **Fix REG-only consistency in record computation**
  - Update `_load_current_records()` to explicitly filter `game_type == "REG"` before computing
    wins/losses/ties.
  - Add or update a unit test that includes both REG and POST games (same season/week) and verifies
    that only REG games affect the computed record.

  Acceptance:
  - [x] Given mixed REG/POST inputs, computed records exactly match REG-only results.

- [x] **Fail fast when required ML feature columns are missing**
  - Replace silent column-dropping logic with explicit validation:
    - Compute `missing_required = set(spec.feature_columns) - set(available_cols)`
    - If non-empty, raise a `ValueError` listing missing columns (truncate list if long).
  - Add a unit test that constructs a minimal ML dataset missing at least one required feature and
    asserts that a clear, informative error is raised.

  Acceptance:
  - [x] The script refuses to run when required feature columns are missing.
  - [x] Error messages name missing columns and indicate how many are missing.

- [x] **Apply win-prob calibration consistently for `ScoreModel`**
  - Update the `ScoreModel` path in `scripts/power_rankings.py` so that:
    - If a calibrator is present, win probabilities are produced via the calibrated path
      (e.g., `predict_home_win_prob(margin, calibrator)`).
    - If no calibrator is intended, this behavior is explicit and documented in code.
  - Add a unit test that proves calibration is applied when a non-identity calibrator exists.

  Acceptance:
  - [x] `ScoreModel` probabilities change appropriately when a calibrator is attached.
  - [x] Behavior matches `margin_total` and `blended_margin_total` semantics.

- [x] **Add minimal runtime diagnostics**
  - Log (INFO-level, single-line):
    - number of past games used in ratings fit
    - number of future games used
    - effective `ratings_min_season` value
  - Ensure logs are stable and suitable for automation/CI logs.

  Acceptance:
  - [x] Running the script prints these diagnostics exactly once per invocation.

### Milestone 23.6 - Operational documentation: weekly pipeline + evaluation rule

Completion note: README documents the weekly workflow and authoritative evaluation rule.

#### Tasks (Milestone 23.6)

- [x] **Add an authoritative "Weekly pipeline" section to `README.md`**
  - Clearly document:
    - data refresh step
    - canonical training/validation step (from Milestone 23)
    - prediction + reporting steps (including power rankings and standings)
  - Specify:
    - where outputs land on disk
    - naming conventions for run folders and artifacts

  Acceptance:
  - [x] A new user can follow the README end-to-end and produce weekly outputs without guessing.

- [x] **Add a single canonical evaluation rule to `README.md`**
  - Explicitly state:
    > "Model selection is based on time-aware walk-forward evaluation; random CV is not
    > authoritative."
  - Reference Milestone 23 outputs as the source-of-truth evaluation.

  Acceptance:
  - [x] The evaluation rule is visible and unambiguous in the README.

### Milestone 24 - Win-prob calibration: choose (and/or auto-choose) the best method

Completion note: Calibration comparison harness added; platt chosen as default.

Options in code today: `none`, `platt`, `isotonic`, `elo`.

- [x] Add a **calibration comparison harness** that evaluates calibration choices under the
  canonical walk-forward protocol (Milestone 23).
  - At minimum: compare Brier, log loss, and reliability.
  - Include pool metrics as tie-breakers.
- [x] Implement **"auto" calibration** (optional but recommended):
  - Use isotonic only when calibration sample size is large enough.
  - Fall back to Platt when calibration data is small/noisy.
  - Always keep an explicit override.
- [x] Add CLI **compatibility alias**: accept `logistic` as a synonym for `platt`.
- [x] Validate that calibrators are trained only on time-appropriate rows.

Acceptance:

- [x] Walk-forward results clearly show which calibration choice is best (and how sensitive it is
  by season/week).
- [x] `--win-prob-calibration logistic` behaves identically to `--win-prob-calibration platt`.

### Milestone 25 - Market integration decisions + correct probability blending

Completion note: Market anchoring and blending validated under walk-forward.

Decide, then enforce, the objectively best usage of market inputs:

- Market as **features**
- Market as **anchoring** (residual modeling)
- Hybrid (anchor + selected transforms)

#### Tasks (Milestone 25)

- [x] Evaluate market as features vs anchoring under the canonical protocol.
- [x] Fix/confirm the market probability source used for blending/clamping:
  - Current: implied prob from moneyline (includes vig).
  - Add: **no-vig** implied probability (normalize home/away to sum to 1).
- [x] Implement/validate **market probability blending** "the right way":
  - Consider blending in **log-odds space** (more stable than linear prob blends).
  - Add clear configuration: source (`raw` vs `novig`), blend method (`prob` vs `logit`),
    weight, and clamp delta.
- [x] Add a small test suite around moneyline->prob and no-vig normalization.

Acceptance:

- [x] The selected market mode (features vs anchor vs hybrid) is chosen via walk-forward.
- [x] Market blending/clamping uses the intended probability definition (raw or no-vig) and is
  unit-tested.

### Milestone 26 - Continuous retraining + weekly orchestration (one command, resumable)

Completion note: Weekly orchestration script added with resumable artifacts.

Goal: a single script to run 1–2x per week that:

1) runs data refresh (`python -m nfl_predictor.data_collection`)
2) re-trains and time-validates the best-known model configuration
3) emits all weekly outputs in a consistent, predictable place

Outputs to include (as available):

- weekly predictions (`*_predictions.csv`)
- confidence pool picks (unique 1..N ranks)
- power rankings for the week
- betting report + optional Excel template
- (optional) projected standings / season win distributions (Milestone 28)

#### Tasks (Milestone 26)

- [x] Create `scripts/weekly_run.py` (or equivalent) that composes existing steps:
  - data collection
  - config selection (Milestones 23–25)
  - tuning (optional)
  - final train
  - prediction + reports
- [x] Make it resumable (like `scripts/betting_pipeline.py`): reuse prior artifacts when inputs
  match.
- [x] Add a config file option (YAML/JSON) to avoid 200-character CLI invocations.

Acceptance:

- [x] One command produces a complete weekly output package from scratch.
- [x] Re-running does not redo expensive work unless inputs or config changed.

### Milestone 27 - Use uncertainty estimates to improve probabilities + confidence ranking

Completion note: Uncertainty-aware win probabilities and ranking path implemented and evaluated.

The repo already produces quantile intervals for margin/total. Use them more directly.

- [x] Derive a per-game uncertainty estimate (e.g., infer σ from p10/p90 width).
- [x] Convert margin + σ into a win probability via a distributional mapping (e.g., normal CDF),
  then optionally calibrate.
- [x] Compare uncertainty-aware probabilities vs current approach via walk-forward.
- [x] Consider uncertainty-aware confidence ranks (e.g., prioritize higher expected points with
  lower upset risk).

Acceptance:

- [x] Walk-forward shows whether uncertainty-aware probabilities improve Brier/log loss and/or
  pool points.

### Milestone 28 - Metric strategy: decide what "better" means (and track it)

Completion note: Metrics hierarchy and diagnostics added to reports.

- [x] Decide which metrics are first-class for model iteration:
  - margin MAE, total MAE
  - Brier, log loss, reliability
  - confidence pool expected/actual points
  - market-relative residual metrics (when market is used)
- [x] Add optional season-level diagnostics:
  - predicted vs actual season win totals (requires projecting remaining games)
  - calibration drift by season/week

Acceptance:

- [x] Metrics are easy to compare across runs (stable JSON schema + summary table).

### Milestone 29 - Hyperparameter optimization (Optuna) hygiene

Completion note: Full Optuna sweep run; artifacts and guardrails captured.

- [x] Run a "full" Optuna sweep for the current best configuration (time-series CV objective).
- [x] Persist best params + study metadata into the run artifacts.
- [x] Add guardrails to prevent accidental tuning on holdout.

Acceptance:

- [x] Optuna results are reproducible and clearly tied to a dataset fingerprint + config.

### Milestone 30 - Feature importance + regularization

Completion note: Feature-importance reports and SHAP script added; pruning/regularization validated
via walk-forward with platt as the best calibration.

- [x] Add a feature-importance report (XGBoost gain/weight) for each trained run.
- [x] Add an optional SHAP analysis script for deeper inspection (keep it optional; do not require
  it for CI).
- [x] Use importance results to:
  - prune noisy/redundant features
  - tune regularization (L1/L2, depth, min_child_weight, etc.)

Acceptance:

- [x] Feature pruning decisions are validated via walk-forward (no "it looked right" commits).

### Milestone 31A - Data collection performance + caching hygiene (blocking)

Completion note: Added nflreadpy caching, profiling toggles, and cache visibility in logs.

Goal: shorten and stabilize data-collection runs while minimizing network calls.

#### Tasks (Milestone 31A)

- [x] Add opt-in timing/profiling logs for data collection (per major step) with a clear toggle.
- [x] Add targeted debug logs around schedule/TeamRankings/ELO/team-stats merges so slow steps are
  visible.
- [x] Audit TeamRankings caching behavior and document the cache hit/miss rules.
- [x] Implement caching for nflreadpy outputs (schedule + team stats) and a clear refresh toggle.
- [x] Document caching and expected run-time behavior in `README.md`.

Acceptance:

- [x] A debug/profiling run prints step timings and shows cache hits.
- [x] A second run reuses cached data without network calls (unless refresh is forced).

### Milestone 31 - Recency + trend features (non-linearity and drift)

Completion note: Trend features and recency weighting shipped with ablation tooling. Walk-forward
ablation shows trend features improve Brier/log loss and margin MAE, while recency weighting with
half-life seasons=2 worsens probability metrics despite a small total-MAE improvement.

Goal: add leakage-safe trend/recency signals plus optional time-weighted training.

#### Feature design + audit

- [x] Inventory existing recency signals (TR last_5/last_10 ratings, lookahead, motivation).
- [x] Finalize minimal trend feature set and confirm they are time-safe.

#### Trend features (time-safe)

- [x] Rating trend: `last_5_games_rating - last_10_games_rating` for away/home + diff.
- [x] Elo trend: `elo_pre - rolling_4wk_mean(elo_pre)` for away/home + diff.
- [x] QB Elo trend: `qb_elo_pre - rolling_4wk_mean(qb_elo_pre)` for away/home + diff.
- [x] Performance trend (select 1-2 stats): recent 4-week mean vs season-to-date mean
  (scoring margin and turnover margin) for away/home + diff.
- [x] Season-phase features: normalized `week_in_season` plus early/mid/late bucket flags.

#### ETL + schema

- [x] Implement rolling aggregates in Polars (per team, per season, prior weeks only).
- [x] Add derived columns to `constants.py` and enforce schema ordering.
- [x] Ensure missing-data policy is consistent for early weeks and short seasons.

#### Recency weighting (exponential half-life)

- [x] Add optional exponential half-life sample-weighting for training + calibration.
- [x] Add CLI/config flags for half-life (weeks or seasons) in training + walk-forward.
- [x] Keep default off and ensure weights are deterministic.

#### Tests

- [x] Unit tests verifying trend features only use prior weeks.
- [x] Unit tests for recency weights (monotonic decay, boundary cases).
- [x] Unit tests for season-phase buckets and normalization.

#### Evaluation

- [x] Walk-forward comparisons with/without trend features and with/without weights.
- [x] Track Brier/log loss first; pool points as tie-breakers; MAE third.

Acceptance:

- [x] New features are leakage-safe and schema-invariant.
- [x] Walk-forward results show a clear improvement or documented tradeoff.

### Milestone 32 - Weather + venue effects (consistent, non-leaky)

Completion note: Stadium metadata features were kept and expanded; weather fields were later
removed after confirming they update post-kickoff.

#### Tasks (Milestone 32)

- [x] Extend stadium metadata beyond city/state (type + altitude).
- [x] Pull historical weather fields from NFLverse schedule data (implemented, later removed).
- [x] Define missing-data policy and enforce invariant schema.
- [x] Add tests for missing-weather fallbacks and schema invariance.

Acceptance:

- [x] Stadium metadata features are maintained; weather features were removed due to leakage risk.

### Milestone 32B - Stadium metadata + venue features (non-leaky)

Completion note: Stadium metadata expanded and wired through ETL/tests with safe defaults.

#### Tasks (Milestone 32B)

- [x] Expand `STADIUMS` to include `name` and `elevation` (and keep city/state).
- [x] Update stadium feature derivation to use the new `STADIUMS` fields and drop any legacy
  altitude map if redundant.
- [x] Keep stadium type/surface features derived from NFLverse schedule fields.
- [x] Add/adjust tests for stadium metadata parsing and safe fallbacks.
- [x] Update README feature list to reflect stadium-only (no weather/ref).

Acceptance:

- [x] Stadium metadata features are present for all games with safe defaults.

### Milestone 33 - Head coach features (if data is robust)

Completion note: Added coach prior record features (career and team-specific) with time-safe
aggregation and leakage tests; walk-forward ablation completed and coach_on retained for
full-feature training.

#### Tasks (Milestone 33)

- [x] Confirm coach coverage via NFLverse schedule fields.
- [x] Add coach prior record features computed strictly to date.
- [x] Add tests that verify no leakage in coach-derived features.

Acceptance:

- [x] Coach features are leakage-safe; walk-forward ablation completed (coach_on retained).

### Milestone 34 - Referee features (if data is robust)

Completion note: Referee features were removed after confirming assignments update post-kickoff.

#### Tasks (Milestone 34)

- [x] Confirm referee coverage via NFLverse schedule fields.
- [x] Implemented referee features (later removed due to post-game updates).

Acceptance:

- [x] Referee features are removed; data is not reliable pre-kickoff.
- [x] Feature ordering/schema remains invariant after removal.

### Milestone 35 - Pandas to Polars audit/refactor

Completion note: Completed a pandas usage audit; ETL is Polars-first and pandas usage is confined
to ML, reporting, and orchestration layers. No safe non-ML/reporting refactors were identified.

#### Tasks (Milestone 35)

- [x] Inventory pandas usage across the repo and classify by module (ETL vs ML vs reporting).
- [x] Identify pandas usage that can move to Polars safely (none found outside ML/reporting).
- [x] Refactor candidate modules to Polars-first implementations (no safe candidates).
- [x] Document any pandas usage that must remain (e.g., sklearn pipelines, calibration).
- [x] Confirm existing tests remain sufficient since no refactor was required.

Acceptance:

- [x] ETL and feature engineering are fully Polars-first with minimal pandas use.
- [x] Remaining pandas usage is justified and documented.

### Milestone 36 - Data availability guards (nflreadpy + TeamRankings)

Completion note: Added guardrails for nflreadpy/TeamRankings availability with tests.

#### Tasks (Milestone 36)

- [x] Enforce nflreadpy availability (min season >= 1999) in data collection CLI.
- [x] Skip TeamRankings loads for seasons before 2003 and use week 2 as the earliest week in 2003.
- [x] Add unit tests for the guardrails.

Acceptance:

- [x] Data collection fails fast for pre-1999 seasons and skips TR pre-2003 without errors.
