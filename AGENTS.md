# Copilot Instructions for nfl-predictor

## 0) Mission + Non-Negotiables

This repository predicts NFL outcomes and scores for **Pick 'Em** and **Confidence Pools**.

- **Primary objective:** calibrated win probabilities and weekly confidence rankings.
- **Secondary objective:** realistic score outputs for display and reporting.

Rules that are always enforced:

- **No data leakage.** Features, splits, calibration, blending, and evaluation only use information
  available before the predicted games ("known at prediction time" for the specific week).
- **Time-aware evaluation.** Hyperparameter tuning and model selection use blocked, time-ordered
  splits (season/week aware).
- **Polars-first ETL.** Dataset creation and feature engineering run in Polars; pandas/numpy are
  acceptable inside ML modules only as needed.
- **NFLverse via nflreadpy is the core data source.**
- **Reproducible artifacts.** Training and backtests write run directories with metadata and
  metrics.
- **Tests are required.** New functionality includes unit tests and improves or maintains coverage.
- **TDD is the default for executable code changes.** Before changing production code, verify that
  the exact behavior and lines you plan to touch are covered; if not, add focused characterization
  or failing tests first, then edit the production code.

## Current Preseason Focus

- The current preseason hardening plan lives in `docs/preseason_2026_readiness_plan.md`.
- `TODO.md` is the authoritative active checklist.
- Validated baseline on 2026-06-13:
  - `.venv/bin/ruff format --check .` passes.
  - `.venv/bin/python -m pytest` passes (`406 passed`).
  - `markdownlint .` passes.
  - `uv lock --check` passes.
  - `uv sync --check --active` passes.
  - `.venv/bin/ruff check .` passes cleanly.
  - `.venv/bin/pyright .` passes (0 errors; fixed via `pandas-stubs` + typed transform helpers).
  - `.venv/bin/ty check .` passes (0 diagnostics; now mandatory alongside Pyright).
  - Coverage is `90.01%`, clearing the enforced preseason target.

## Source of truth for work

- Current milestones and tasks live in `TODO.md` (authoritative active worklist).
- `CHANGELOG.md` is the authoritative release history.
- Before starting any task: read `TODO.md` and work only on the highest-priority blocking items.
- When a task is completed: move it from `TODO.md` to `ARCHIVE.md` with a short completion note.
- Milestone numbering is authoritative in `ARCHIVE.md`:
  - Completed milestones are archived there.
  - New milestones must continue numbering from the latest archived milestone.

## Engineering Standards (Logic, Docs, Lint, Coverage)

- Docstrings are required for every module, class, and function (including tests).
- Type hints are required for new/modified code.
- Fix linter findings introduced by your changes. Do not leave new warnings behind.
- Avoid adding new `noqa`, `type: ignore`, or `pragma: no cover` suppressions unless they are truly
  necessary, narrowly scoped, and justified by the code rather than convenience.
- Do not reference temporary planning artifacts in code: do not mention roadmap items, milestone
  numbers, or TODO goal labels in any code, comments, docstrings, or test descriptions.
- Prefer small, deterministic unit tests.
- Load and apply relevant skills before acting. Default to `python` for Python work; add
  `test-driven-development`, `clean-code`, `systematic-debugging`, `code-review`, `observability`,
  `task-orchestrator`, and the `python-*` skills when their domains apply.
- If a Python file grows beyond ~2000 lines, propose a refactor plan to split it into smaller,
  focused modules (helpers/utils) and implement the split if it reduces complexity.
- Keep `TODO.md` accurate: verify items before checking them off.
- Keep `docs/preseason_2026_readiness_plan.md`, `TODO.md`, `ARCHIVE.md`, and `CHANGELOG.md`
  synchronized with the actual repo state after meaningful progress, completed tasks, or validation
  changes.
- Keep `README.md` current: update it when behavior, CLI usage, features, or outputs change.
- Keep project documentation current after significant changes. When a subsystem outgrows the
  top-level `README.md`, add or update nested module README files and link them from the top-level
  README.

## Changelog and Commit Workflow

- `CHANGELOG.md` follows Common Changelog: latest release first, `## VERSION - YYYY-MM-DD`, then
  `Changed`, `Added`, `Removed`, and `Fixed` in that order.
- The historical baseline is `0.1.0` on `main`. Add the next release entry above it and reference
  the most relevant commits before tagging a release.
- Keep changelog entries focused on notable user-facing, tooling, or workflow changes; skip routine
  formatting-only noise.
- Keep git tags aligned with changelog versions so `.github/workflows/release.yml` can publish or
  update GitHub releases from `CHANGELOG.md`.
- When committing work, prefer one file per commit, including deletions, unless the user explicitly
  asks for different commit granularity.

## CI Direction

- GitHub Actions is the first CI target and currently stays validation-only.
- `.github/workflows/validation.yml` provisions `.venv` with `uv` and runs Ruff format/check,
  Pyright, Ty, pytest, markdownlint, `uv lock --check`, `uv sync --check --active`, and the existing
  editable-install plus primary CLI help smoke checks.
- `.github/workflows/release.yml` publishes or updates GitHub releases for `0.x.y` and `v0.x.y` tags
  by extracting the matching `CHANGELOG.md` entry.

## Command execution rules (non-negotiable)

This project uses a **local virtual environment located at `.venv/`**.

When running any commands, you MUST invoke tools from the virtual environment explicitly. Do NOT
rely on shell activation, PATH inference, or system-installed binaries.

### Required command forms

Use these forms **at all times**:

- Python:
  - `.venv/bin/python`
- uv:
  - `uv`
- pytest:
  - `.venv/bin/python -m pytest` **or** `.venv/bin/pytest` **or** `uv run pytest`
- ruff:
  - `.venv/bin/ruff`
- pyright:
  - `.venv/bin/pyright`
- ty:
  - `.venv/bin/ty`

### Explicitly forbidden

- `python`, `pip`, `pytest`, `ruff`, `pyright`, or `ty` **without a `.venv/` prefix**
- assuming an activated shell or implicit virtualenv
- using system Python, Conda, pyenv, or global tools

For Python-based tooling, assume the `.venv/bin/` prefix is required even if it is not written in a
doc example. `uv` is expected to come from `PATH` as an external project manager.

### Formatting, linting, and style

- **Ruff** formatting (line length 100).
- **Ruff** linting (including import sorting, docstrings, security, simplify, NumPy, and
  pygrep-hooks rules).
- **Pyright** and **Ty** are both mandatory local gates today.
- **Pyright** remains the more mature signal in pandas-heavy code, so keep both green rather than
  replacing one with the other.
- PEP 8 / PEP 257 conventions unless explicitly overridden by repo tooling.

Recommended local commands:

- `.venv/bin/ruff format .`
- `.venv/bin/ruff check .` (and optionally `.venv/bin/ruff check . --fix`)
- `.venv/bin/pyright .`
- `.venv/bin/ty check .`
- `.venv/bin/python -m pytest`
- `uv lock --check`
- `uv sync --check --active`

## Project Shape (Big Picture)

- **Primary Pipeline:** Polars for data processing + `nflreadpy` for NFLverse sources (schedule,
  team stats, etc.). The pipeline integrates schedule/results, team statistics, Elo/QB ratings,
  TeamRankings stats, and market odds to produce ML-ready datasets.
- **Orchestration Script (ETL):** `nfl_predictor/data_collection.py` (run as a module). This
  orchestrator fetches data, applies transformations, and writes output CSVs.
- **Core Data Transforms:** Polars ETL helpers live under `nfl_predictor/utils/polars/`.
  `nfl_predictor/utils/polars_utils.py` is a compatibility facade that forwards imports to the split
  modules.
- **Game-Specific Enrichments:** `nfl_predictor/utils/game_utils.py` contains domain-specific
  calculations and dataset enrichments.
- **External Data Scraping/Caching:** `nfl_predictor/utils/scraping_utils.py` fetches and caches
  external web data used by the pipeline.

### ML implementation layout

- `nfl_predictor/ml/` contains the split ML implementation modules.
- `nfl_predictor/ml_model.py` is a compatibility facade for legacy imports and a primary CLI
  entrypoint.
- XGBoost version/build compatibility helpers live in `nfl_predictor/ml/ml_model_xgb_utils.py`.

### Repo scripts (operational entrypoints)

- `scripts/weekly_run.py`: canonical weekly orchestration (data refresh -> compare -> tune/train ->
  predict -> reports).
- `scripts/golden_command.py`: convenience orchestration for walk-forward + training + prediction
  and artifact stamping.
- `scripts/betting_pipeline.py`: end-to-end orchestration for selecting calibration/probability
  post-processing, resumable Optuna tuning, final training, week predictions, and betting report
  outputs.
- `scripts/walk_forward_backtest.py`: walk-forward evaluation utility.
- `scripts/wf_compare.py`: sweep calibration + market-prob variants and summarize metrics.

## Modeling Philosophy (Important Context)

- Implementation is purely in Python.
- Team strength is represented by learned relationships between engineered features and outcomes.
- Feature interactions and weights are learned by the model; feature engineering provides signal,
  not fixed scoring formulas.

## Data Inputs/Outputs (Repo Conventions)

- **Data directory:** all datasets live under `data/` (see `constants.DATA_PATH` in
  `nfl_predictor/constants.py`).
- **Key output files (examples; do not hard-code filenames):**
  - `data/all_data_ml.csv` - master ML dataset (includes engineered features and targets where
    available)
  - `data/all_data.csv` - combined dataset without ML-only targets
  - `data/completed_games_ml.csv` and `data/completed_games.csv` - completed games subsets
  - `data/predict/week_XX_games_to_predict.csv` - upcoming week games with engineered features

### I/O rules

- Prefer the project’s Polars-based load/save helpers in `nfl_predictor/data_collection.py`.
- Any pandas-based CSV I/O utilities are legacy. Do not add new pandas-based I/O helpers; prefer the
  Polars ETL helpers when touching related code.

## Column & Schema Rules (Source of Truth)

- Column names and schema lists are defined in `nfl_predictor/constants.py`.
- Team identifiers are normalized using the canonical mapping in `constants.py`.
- Do not hard-code column lists; use constants to prevent schema drift.
- ETL does not silently drop columns.
- Always normalize team identifiers via `constants.ALIAS_TO_CANONICAL` and (when available)
  `normalize_team_column(df, col)`.

## Season/Week Logic & Edge Cases

- Use `constants.get_regular_season_weeks(season)` to determine regular-season length.
- Week 1 / early-season rows with missing history use a documented fallback consistent with tests
  (regression-to-mean and/or previous-season values + global mean).
- Future games have missing outcomes; the pipeline still outputs a structurally complete row
  suitable for prediction.
- Postseason rows may exist. Training/evaluation defaults should be explicit about whether
  postseason is included and (if included) how it is weighted.

## Prediction & Modeling Logic

### Canonical targets

The primary model predicts:

- `margin = home_score - away_score`
- `total  = home_score + away_score`

Derived scores are computed as:

- `home_score = (total + margin) / 2`
- `away_score = (total - margin) / 2`

Direct home/away score regressors are allowed only as secondary ensemble members.

### Win probability

- Win probability is derived from the margin prediction.
- Win probabilities are calibrated using time-aware calibration data.
- Calibration metrics (Brier, log loss, reliability table) are reported in evaluation.

Calibration methods (canonical names):

- `none`: deterministic margin->prob mapping (baseline)
- `platt`: logistic regression (Platt scaling)
- `isotonic`: isotonic regression
- `elo`: deterministic Elo-style logistic mapping

Notes:

- Prefer time-aware calibration (`platt` or `isotonic`) when enough calibration rows exist.
- If adding new CLI options, keep names stable and document them.

### Market integration

When market lines exist, the system produces market-derived features and supports market anchoring:

- Market transforms produce `market_home_margin`, `market_total_line`, `home_market_prob`,
  `away_market_prob`.
- Market anchoring trains residuals vs market baselines and adds the baseline back at prediction
  time.
- Market probability blending/clamping uses explicit CLI/config values and is validated in
  time-aware evaluation.

Market anchoring details:

- Prefer residual training: `target_resid = target - market_baseline` and `pred = market_baseline +
pred_resid`.

Market probability post-processing (blend/clamp):

- Blending must be explicit and bounded (weights in [0, 1]).
- Clamping must be explicit and bounded (delta in [0, 0.5]).
- If adding "no-vig" market probability options, implement them consistently (home/away normalize to
  sum to 1) and validate in walk-forward.

### Uncertainty

- Predictions include uncertainty intervals for margin and total (p10/p50/p90 or equivalent).
- Interval outputs are evaluated (coverage/width diagnostics) and are part of the run artifacts.

Minimum requirement:

- Output a median plus at least one interval for both margin and total (quantiles preferred).

### Realistic score outputs

- Realistic score outputs are produced as post-processing applied after margin/total predictions are
  generated.
- Realistic score adjustments are used for display and reporting.
- Realistic score adjustments do not alter win probabilities, confidence rankings, pool scoring, or
  tuning objectives.

If implementing score "realism":

- Apply post-processing only after core predictions; rounding/snapping policies must be
  configurable.
- Never change training targets to enforce an "NFL score lattice" unless explicitly designed and
  documented.

### Confidence pool deliverable

- Weekly outputs include a **1..N** unique confidence ranking across that week's games.
- Predicted winner is derived from calibrated win probability.
- Confidence strength is derived from calibrated win probability (default: `abs(p - 0.5)`).

Authoritative pool scoring rules:

- Each week assign unique confidence values `1..N` to the chosen winner in each matchup.
- Max weekly points = `N*(N+1)/2`.
- Realized points = `sum(conf_i * 1[pick_i_correct])`.
- Tie games: treat as incorrect for both sides.
- Picks are submitted before the first game of the week (single-shot; no in-week updates in
  backtests).

## Evaluation

Required evaluation modes:

- **Season-blocked CV** (acceptable baseline; primarily used for hyperparameter tuning).
- **Walk-forward evaluation (authoritative):** for each season and each week `w` (e.g., `3..end`),
  train on all games strictly before week `w` (plus prior seasons if configured), predict week `w`,
  and record metrics.

Required metrics:

- margin MAE
- total MAE
- win probability Brier score
- win probability log loss
- binned reliability summary
- confidence pool point summaries (expected + actual)

Market-relative metrics (when market anchoring is enabled):

- residual MAE vs market baseline for margin/total
- edge vs spread/total as diagnostics only (do not claim profitability)

### Model selection protocol (how to choose "best" settings)

When multiple options exist (calibration method, market integration mode, probability blend/clamp
rules, weighting choices):

- Prefer selecting settings via walk-forward over multiple seasons.
- Pick a primary selection metric (typically Brier/log loss for probability quality) and use
  secondary tie-breakers (confidence pool expected points, then margin/total MAE).
- Report mean and variance across folds; avoid choosing a setting that wins by a hair on one season
  but regresses elsewhere.
- Never use the holdout window to tune hyperparameters.

Required run artifacts:

- saved model artifact
- metadata JSON (see "Model artifact contract")
- metrics report JSON (walk-forward aggregated + per-season/per-week summaries)
- plots are optional and must not block CI

## ML Implementation Standards

Preprocessing:

- Use `ColumnTransformer` for categorical one-hot + numeric passthrough/impute.
- Avoid densifying large sparse matrices unintentionally.
- Tree-based models (XGBoost): do not use `StandardScaler` unless a non-tree model requires it.
- Missing values: XGBoost can handle them; impute only if required for consistency.

Training:

- Use early stopping and set `eval_metric` explicitly (aligned to objective).
- Tune hyperparameters consistently with the evaluation metric (Optuna supported).
- Use `random_state` everywhere applicable.
- Do not hard-code `n_jobs`; prefer `os.cpu_count()` or a config default.

Blending:

- Prefer explicit, interpretable blends (market anchoring often sufficient).
- If using a blender/regressor, avoid unstable unconstrained weights; prefer non-negative and/or
  sum-to-1 when appropriate.
- Validate blends using time-aware splits.

## Leakage Audit (Required)

Maintain a leakage audit tool/mode (see `scripts/leakage_audit.py`):

- checks for target/label columns in features
- flags suspiciously predictive columns (e.g., absurd correlations)
- validates season-to-date features exclude the current game row

## Reproducibility & Model Artifact Contract

Every saved model must include adjacent metadata JSON with:

- created timestamp
- git commit hash (if available)
- dataset fingerprint (hash of training CSV and/or stable row ids)
- library versions (xgboost, sklearn, numpy, pandas, polars, scipy)
- training config (CLI args / config object)
- season/week ranges used for train/calibration/holdout
- feature list used
- best params (if tuned) and early-stopping info

Artifacts must be loadable without hidden external state.

## Dependency & Environment Hygiene

- Pin key ML dependencies for reproducibility: xgboost, scikit-learn, numpy, pandas, polars, scipy,
  optuna (if used).
- Document supported Python version(s) and CPU/GPU constraints if applicable.
- Avoid optional GPU paths that break CPU-only execution unless explicitly guarded.

### Dependency management (uv + pinned requirements)

- Declare runtime and development dependencies in `pyproject.toml`.
- Treat `uv.lock` as the lockfile source of truth for reproducible environments.
- Preferred install/update flow is `uv lock` / `uv sync`.
- `update_requirements.sh` is the convenience wrapper for refreshing the lockfile and syncing the
  active project environment.

## Feature Development Rules

All engineered features apply to **every matchup**, not only end-of-season games.

Feature areas tracked in `TODO.md` include (examples):

- season-to-date record features (overall, division, conference W-L-T)
- divisional rivalry indicator
- lookahead/trap indicators (next-week opponent strength + rest/travel context)
- motivational asymmetry features (playoff leverage and clinch/elimination context)

## Missing Data Rules

Some sources do not exist for all seasons.

- ETL emits an invariant schema for every run.
- Missing sources become nulls or defined defaults.
- The model path handles nulls without crashing and reports how many rows use fallbacks for each
  feature group.

## constants.py Hygiene

`nfl_predictor/constants.py` remains the canonical reference for:

- file paths
- feature column names/lists
- team mapping tables
- season/week rules
- default parameters used in data collection

Unused constants are removed and the file remains organized into clear sections.

## Dev Workflows (How to Run Things)

- Refresh data:
  - `.venv/bin/python -m nfl_predictor.data_collection`
- Train/predict (CLI):
  - `.venv/bin/python -m nfl_predictor.ml_model --help`
- Walk-forward evaluation:
  - `.venv/bin/python scripts/walk_forward_backtest.py --help`
- Convenience orchestration:
  - `.venv/bin/python scripts/golden_command.py --help`
  - `.venv/bin/python scripts/betting_pipeline.py --help`
- Testing:
  - `.venv/bin/python -m pytest`

Training/prediction entrypoints may be updated/replaced, but must remain runnable and documented.

## Logging & Coding Style

- Logging uses the project logger (`from nfl_predictor.utils.logger import log`). No `print`.
- Prefer Polars expressions over Python loops in ETL.

## Safety, Scope, and Prohibited Behaviors

- Do not introduce offensive/unsafe content or harmful instructions.
- Do not add unrelated features (dashboards, new scrapers, unrelated pipelines).
- Do not remove/alter existing pipeline behavior without updating tests and documentation.
- Avoid new external services or network dependencies beyond existing scraping utilities.
- Do not claim betting profitability; report metrics and uncertainty honestly.

## Choosing APIs & Libraries

- Polars is used for ETL and feature engineering.
- `nflreadpy` is used for NFLverse data.
- scikit-learn and XGBoost are used for modeling.

## Assistant guidance

- Use existing project utilities and constants.
- Implement changes in small, testable increments.
- Keep outputs deterministic under fixed seeds.
- Do not change behavior without updating tests and documentation.
- Do not delete `models/<run_id>/wf_compare/` during active walk-forward runs; those artifacts power
  resume behavior.
- To resume a walk-forward comparison, re-run the same command with `--resume` and inspect
  `wf_compare/wf_summary.csv` for live progress.
