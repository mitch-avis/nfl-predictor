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

- [ ] Define the **primary evaluation lens** for the project:
  - Walk-forward (rolling-origin) over multiple seasons is authoritative.
  - Season-blocked CV exists mainly for hyperparameter tuning.
- [ ] Standardize the **outer evaluation window**:
  - Default: last N seasons (configurable), regular season only by default.
  - Explicit handling for incomplete current season and postseason evaluation.
- [ ] Standardize the **inner calibration window**:
  - Time-aware calibration weeks (e.g., last K weeks before prediction week) and/or
    calibration seasons.
  - Minimum sample size rules (see Milestone 24).
- [ ] Decide (and document) the **selection hierarchy** for “best model”:
  - Primary: probability quality (Brier, log loss, reliability).
  - Secondary: confidence pool expected points (and stability across weeks).
  - Tertiary: margin/total MAE (and market-relative residual MAE if anchoring).
- [ ] Produce a single **metrics report schema** that always includes:
  - per-week, per-season, and overall aggregates
  - mean + variance across folds
  - market-relative metrics when market is present

Acceptance:

- [ ] There is one “blessed” evaluation command (or script) that reproduces the reported metrics.
- [ ] A config sweep (Milestones 24/25) can run under this protocol without ad hoc code.

---

## Milestone 24 - Win-prob calibration: choose (and/or auto-choose) the best method

Options in code today: `none`, `platt`, `isotonic`, `elo`.

- [ ] Add a **calibration comparison harness** that evaluates calibration choices under the
  canonical walk-forward protocol (Milestone 23).
  - At minimum: compare Brier, log loss, and reliability.
  - Include pool metrics as tie-breakers.
- [ ] Implement **"auto" calibration** (optional but recommended):
  - Use isotonic only when calibration sample size is large enough.
  - Fall back to Platt when calibration data is small/noisy.
  - Always keep an explicit override.
- [ ] Add CLI **compatibility alias**: accept `logistic` as a synonym for `platt`.
- [ ] Validate that calibrators are trained only on time-appropriate rows.

Acceptance:

- [ ] Walk-forward results clearly show which calibration choice is best (and how sensitive it is
  by season/week).
- [ ] `--win-prob-calibration logistic` behaves identically to `--win-prob-calibration platt`.

---

## Milestone 25 - Market integration decisions + correct probability blending

Decide, then enforce, the objectively best usage of market inputs:

- Market as **features**
- Market as **anchoring** (residual modeling)
- Hybrid (anchor + selected transforms)

Key tasks:

- [ ] Evaluate market as features vs anchoring under the canonical protocol.
- [ ] Fix/confirm the market probability source used for blending/clamping:
  - Current: implied prob from moneyline (includes vig).
  - Add: **no-vig** implied probability (normalize home/away to sum to 1).
- [ ] Implement/validate **market probability blending** “the right way”:
  - Consider blending in **log-odds space** (more stable than linear prob blends).
  - Add clear configuration: source (`raw` vs `novig`), blend method (`prob` vs `logit`),
    weight, and clamp delta.
- [ ] Add a small test suite around moneyline->prob and no-vig normalization.

Acceptance:

- [ ] The selected market mode (features vs anchor vs hybrid) is chosen via walk-forward.
- [ ] Market blending/clamping uses the intended probability definition (raw or no-vig) and is
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

- [ ] Create `scripts/weekly_run.py` (or equivalent) that composes existing steps:
  - data collection
  - config selection (Milestones 23–25)
  - tuning (optional)
  - final train
  - prediction + reports
- [ ] Make it resumable (like `scripts/betting_pipeline.py`): reuse prior artifacts when inputs
  match.
- [ ] Add a config file option (YAML/JSON) to avoid 200-character CLI invocations.

Acceptance:

- [ ] One command produces a complete weekly output package from scratch.
- [ ] Re-running does not redo expensive work unless inputs or config changed.

---

## Milestone 27 - Use uncertainty estimates to improve probabilities + confidence ranking

The repo already produces quantile intervals for margin/total. Use them more directly.

- [ ] Derive a per-game uncertainty estimate (e.g., infer σ from p10/p90 width).
- [ ] Convert margin + σ into a win probability via a distributional mapping (e.g., normal CDF),
  then optionally calibrate.
- [ ] Compare uncertainty-aware probabilities vs current approach via walk-forward.
- [ ] Consider uncertainty-aware confidence ranks (e.g., prioritize higher expected points with
  lower upset risk).

Acceptance:

- [ ] Walk-forward shows whether uncertainty-aware probabilities improve Brier/log loss and/or
  pool points.

---

## Milestone 28 - Metric strategy: decide what "better" means (and track it)

- [ ] Decide which metrics are first-class for model iteration:
  - margin MAE, total MAE
  - Brier, log loss, reliability
  - confidence pool expected/actual points
  - market-relative residual metrics (when market is used)
- [ ] Add optional season-level diagnostics:
  - predicted vs actual season win totals (requires projecting remaining games)
  - calibration drift by season/week

Acceptance:

- [ ] Metrics are easy to compare across runs (stable JSON schema + summary table).

---

## Milestone 29 - Hyperparameter optimization (Optuna) hygiene

- [ ] Run a “full” Optuna sweep for the current best configuration (time-series CV objective).
- [ ] Persist best params + study metadata into the run artifacts.
- [ ] Add guardrails to prevent accidental tuning on holdout.

Acceptance:

- [ ] Optuna results are reproducible and clearly tied to a dataset fingerprint + config.

---

## Milestone 30 - Feature importance + regularization

- [ ] Add a feature-importance report (XGBoost gain/weight) for each trained run.
- [ ] Add an optional SHAP analysis script for deeper inspection (keep it optional; do not require
  it for CI).
- [ ] Use importance results to:
  - prune noisy/redundant features
  - tune regularization (L1/L2, depth, min_child_weight, etc.)

Acceptance:

- [ ] Feature pruning decisions are validated via walk-forward (no “it looked right” commits).

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
