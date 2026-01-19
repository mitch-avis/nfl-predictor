# TODO - Active Work for nfl-predictor

This file contains **active** work only. Completed milestones live in `ARCHIVE.md`.

Execution loop for each milestone:

- implement
- add/adjust tests (increase coverage)
- run `python -m pytest` (and coverage)
- run `ruff check .` and `black .` (or CI equivalents)
- update docs if behavior changes

---

## Guardrails checklist (must stay true)

- [ ] Confirm all new features apply to all matchups (no end-of-season-only logic).
- [ ] Confirm no leakage: all splits, features, and labels are time-aware.
- [ ] Confirm ETL is Polars-first and pulls NFLverse via `nflreadpy`.
- [ ] Confirm artifacts remain reproducible (run folder, metadata, metrics).
- [ ] Confirm new code includes unit tests and does not reduce coverage.

---

## Milestone 22 - Repo/tooling alignment (blocking)

Goal: ensure a fresh clone can set up a reproducible dev environment and run the standard checks
without surprises, before starting the new modeling milestones.

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
- [x] Confirm `.gitignore` covers run artifacts (models, optuna db, caches) and that `data/` being ignored
  is intentional and documented.

Acceptance:

- [x] `ruff check .` and `black --check .` pass in a clean environment.
- [x] `python -m pytest` passes (and `python -m pytest --no-cov` works as documented).
- [x] README instructions work end-to-end on a clean machine.

---

## Milestone 23 - Canonical training + validation methodology (the "source of truth")

Goal: define (and enforce) a single, defensible evaluation protocol for choosing *any* modeling
option (features, calibration, market integration, weights, algorithms) using historical data.

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
- [x] Decide (and document) the **selection hierarchy** for “best model”:
  - Primary: probability quality (Brier, log loss, reliability).
  - Secondary: confidence pool expected points (and stability across weeks).
  - Tertiary: margin/total MAE (and market-relative residual MAE if anchoring).
- [x] Produce a single **metrics report schema** that always includes:
  - per-week, per-season, and overall aggregates
  - mean + variance across folds
  - market-relative metrics when market is present

Acceptance:

- [x] There is one “blessed” evaluation command (or script) that reproduces the reported metrics.
- [x] A config sweep (Milestones 24/25) can run under this protocol without ad hoc code.

---

## Milestone 23.5 - Reporting pipeline correctness + schema safety (blocking)

Goal: ensure reporting scripts (especially `scripts/power_rankings.py`) produce correct,
consistent outputs and fail fast on schema/config issues before downstream modeling work continues.

### Tasks

- [x] **Fix REG-only consistency in record computation**
  - Update `_load_current_records()` to explicitly filter `game_type == "REG"` before computing
    wins/losses/ties.
  - Add or update a unit test that includes both REG and POST games (same season/week) and verifies
    that only REG games affect the computed record.

  Acceptance:
  - Given mixed REG/POST inputs, computed records exactly match REG-only results.

- [x] **Fail fast when required ML feature columns are missing**
  - Replace silent column-dropping logic with explicit validation:
    - Compute `missing_required = set(spec.feature_columns) - set(available_cols)`
    - If non-empty, raise a `ValueError` listing missing columns (truncate list if long).
  - Add a unit test that constructs a minimal ML dataset missing at least one required feature and
    asserts that a clear, informative error is raised.

  Acceptance:
  - The script refuses to run when required feature columns are missing.
  - Error messages name missing columns and indicate how many are missing.

- [x] **Apply win-prob calibration consistently for `ScoreModel`**
  - Update the `ScoreModel` path in `scripts/power_rankings.py` so that:
    - If a calibrator is present, win probabilities are produced via the calibrated path
      (e.g., `predict_home_win_prob(margin, calibrator)`).
    - If no calibrator is intended, this behavior is explicit and documented in code.
  - Add a unit test that proves calibration is applied when a non-identity calibrator exists.

  Acceptance:
  - `ScoreModel` probabilities change appropriately when a calibrator is attached.
  - Behavior matches `margin_total` and `blended_margin_total` semantics.

- [x] **Add minimal runtime diagnostics**
  - Log (INFO-level, single-line):
    - number of past games used in ratings fit
    - number of future games used
    - effective `ratings_min_season` value
  - Ensure logs are stable and suitable for automation/CI logs.

  Acceptance:
  - Running the script prints these diagnostics exactly once per invocation.

---

## Milestone 23.6 - Operational documentation: weekly pipeline + evaluation rule

Goal: prevent accidental misuse of evaluation methods and clarify the intended weekly workflow.

### Tasks

- [x] **Add an authoritative “Weekly pipeline” section to `README.md`**
  - Clearly document:
    - data refresh step
    - canonical training/validation step (from Milestone 23)
    - prediction + reporting steps (including power rankings and standings)
  - Specify:
    - where outputs land on disk
    - naming conventions for run folders and artifacts

  Acceptance:
  - A new user can follow the README end-to-end and produce weekly outputs without guessing.

- [x] **Add a single canonical evaluation rule to `README.md`**
  - Explicitly state:
    > “Model selection is based on time-aware walk-forward evaluation; random CV is not authoritative.”
  - Reference Milestone 23 outputs as the source-of-truth evaluation.

  Acceptance:
  - The evaluation rule is visible and unambiguous in the README.

---

## Milestone 24 - Win-prob calibration: choose (and/or auto-choose) the best method

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

---

## Milestone 25 - Market integration decisions + correct probability blending

Decide, then enforce, the objectively best usage of market inputs:

- Market as **features**
- Market as **anchoring** (residual modeling)
- Hybrid (anchor + selected transforms)

Key tasks:

- [x] Evaluate market as features vs anchoring under the canonical protocol.
- [x] Fix/confirm the market probability source used for blending/clamping:
  - Current: implied prob from moneyline (includes vig).
  - Add: **no-vig** implied probability (normalize home/away to sum to 1).
- [x] Implement/validate **market probability blending** “the right way”:
  - Consider blending in **log-odds space** (more stable than linear prob blends).
  - Add clear configuration: source (`raw` vs `novig`), blend method (`prob` vs `logit`),
    weight, and clamp delta.
- [x] Add a small test suite around moneyline->prob and no-vig normalization.

Acceptance:

- [x] The selected market mode (features vs anchor vs hybrid) is chosen via walk-forward.
- [x] Market blending/clamping uses the intended probability definition (raw or no-vig) and is
  unit-tested.

---

## Milestone 26 - Continuous retraining + weekly orchestration (one command, resumable)

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

Tasks:

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

---

## Milestone 27 - Use uncertainty estimates to improve probabilities + confidence ranking

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

---

## Milestone 28 - Metric strategy: decide what "better" means (and track it)

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

---

## Milestone 29 - Hyperparameter optimization (Optuna) hygiene

- [x] Run a “full” Optuna sweep for the current best configuration (time-series CV objective).
- [x] Persist best params + study metadata into the run artifacts.
- [x] Add guardrails to prevent accidental tuning on holdout.

Acceptance:

- [x] Optuna results are reproducible and clearly tied to a dataset fingerprint + config.

---

## Milestone 30 - Feature importance + regularization

- [x] Add a feature-importance report (XGBoost gain/weight) for each trained run.
- [x] Add an optional SHAP analysis script for deeper inspection (keep it optional; do not require
  it for CI).
- [x] Use importance results to:
  - prune noisy/redundant features
  - tune regularization (L1/L2, depth, min_child_weight, etc.)

Acceptance:

- [x] Feature pruning decisions are validated via walk-forward (no “it looked right” commits).

---

## Milestone 31 - Recency + trend features (non-linearity and drift)

- [ ] Add explicit trend direction features (e.g., change between last-10 and last-5 ratings).
- [ ] Evaluate time-weighted training (upweight more recent seasons/weeks) without leaking.
- [ ] Explore limited non-linear time interaction features (e.g., early-season vs late-season
  effects) while guarding against overfitting.

Acceptance:

- [ ] Any added time/recency features materially improve walk-forward metrics.

---

## Milestone 32 - Weather + venue effects (consistent, non-leaky)

- [ ] Extend stadium metadata beyond city/state:
  - stadium type (open / dome / retractable)
  - altitude (Denver, Mexico City, etc.)
- [ ] Pull historical weather where available (ideally from NFLverse via `nflreadpy`):
  - temperature, wind, precipitation flags
- [ ] Define strict missing-data policy and ensure invariant schema across seasons.

Acceptance:

- [ ] Weather/venue features exist for all games with safe fallbacks and improve metrics.

---

## Milestone 33 - Head coach features (if data is robust)

- [ ] Determine availability/coverage of head coach information (ideally via NFLverse).
- [ ] If reliable:
  - encode coach identity (categorical) and/or tenure/experience features
  - add coach prior record (career and with current team) computed strictly to date

Acceptance:

- [ ] Coach features do not leak and demonstrate value in walk-forward.

---

## Milestone 34 - Referee features (if data is robust)

- [ ] Determine availability/coverage of referee assignments historically.
- [ ] If reliable:
  - encode referee identity (categorical) and/or per-ref historical tendencies computed to date
    (penalties, home bias proxies, etc.)

Acceptance:

- [ ] Ref features do not leak and improve at least one primary metric.

---

## Explicitly out of scope (unless the world changes)

- Injury/practice participation features.
  - Prior work showed the data is not reliably updated in-season.
  - QB Elo + manual verified starters remain the preferred approach.
