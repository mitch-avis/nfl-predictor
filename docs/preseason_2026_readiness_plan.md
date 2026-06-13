# 2026 Preseason Readiness Plan

## Purpose

Bring the repository back to a coherent, season-ready baseline on Python 3.14 before resuming larger
feature work for the 2026-2027 NFL season.

## Validated Baseline

The following checks were run on 2026-06-13 against the refreshed `.venv`:

| Check                             | Result      | Notes                                                                                            |
| --------------------------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| `.venv/bin/ruff format --check .` | Pass        | `104 files already formatted`                                                                    |
| `.venv/bin/ruff check .`          | Pass        | All 25 E501 findings in `betting_excel.py` scoped with justified suppressions                    |
| `.venv/bin/pyright .`             | **Pass**    | **0 errors** after adding `pandas-stubs` and `_fit_transform_matrix`/`_transform_matrix` helpers |
| `.venv/bin/ty check .`            | Pass        | `0 errors`; now mandatory alongside `pyright`                                                    |
| `.venv/bin/python -m pytest`      | Pass        | `391 passed`                                                                                     |
| Coverage from pytest              | Pass        | `90%`, meeting the preseason target of `90%` or higher                                           |
| `markdownlint .`                  | Pass        | `0 error(s)`                                                                                     |
| `uv lock --check`                 | Pass        | Lockfile is in sync with `pyproject.toml`                                                        |
| `uv sync --check --active`        | Pass        | Active project environment matches `uv.lock`                                                     |
| Editable install smoke check      | Pass        | `uv pip install --python .venv/bin/python -e . --no-deps` succeeds                              |
| Primary CLI help smoke checks     | Pass        | `nfl_predictor.ml_model`, `weekly_run.py`, and `power_rankings.py`                               |

## Tooling Recommendations

### Hatchling

Keep `hatchling`. It is actively used by `pip install -e . --no-deps`, which completed successfully
through the `pyproject.toml` build backend.

### Pyright vs Ty

Keep `pyright` and `ty` as mandatory local validation gates.

`pyright` remains the more mature and predictable checker for this pandas-heavy codebase, but the
repo now passes `ty` as well and should keep both green rather than treating `ty` as advisory.

No repo-specific `[tool.ty]` configuration is currently required for a clean pass.

### update_requirements.sh

The script should center on `uv lock` and `uv sync`, with explicit handling for missing or inactive
virtual environments.

### GitHub Actions

Prefer GitHub Actions if CI is added during this hardening pass. Start with a validation-only
workflow that creates `.venv`, runs `uv sync`, and then executes the same Ruff, Pyright, Ty, pytest,
and markdownlint gates used locally. After version tags and `CHANGELOG.md` entries are standardized,
a second workflow can publish GitHub releases from tagged changelog entries.

## Workstreams

### 1. Align metadata, docs, and instructions

- Update `AGENTS.md`, `README.md`, and `TODO.md` so they match the actual Python 3.14 and
  Ruff/Pyright/Ty workflow.
- Add `CHANGELOG.md`, record `0.1.0` on `main` as the historical baseline, and document how future
  release entries should be maintained and tagged.
- Remove stale Black references and `.venv/bin/pip` assumptions.
- Align `project.requires-python`, `tool.ruff.target-version`, and `tool.pyright.pythonVersion`.
- Remove stale comments copied from other repositories.
- Decide where nested module README files would add real value, likely starting with the `ml` and
  `reporting` packages.

Exit criteria:

- Every documented command matches a real command that exists in `.venv/bin` or is intentionally
  external.
- Project metadata points to the same Python baseline everywhere.
- Changelog maintenance rules are documented and consistent with future release tagging.

### 2. Burn down the remaining Ruff diagnostics

- Package/module docstrings, entrypoint docstring-style fixes, safe simplify rewrites,
  security-justified suppressions (S110, S607, S101), NumPy RNG suppressions, and domain-appropriate
  Excel formula line suppressions (E501) are complete. **Ruff now passes cleanly: 0 remaining
  diagnostics.**
- Prefer structural fixes over reintroducing broad `noqa`, `type: ignore`, or `pragma` comments.

Exit criteria:

- `.venv/bin/ruff check .` passes cleanly.

### 3. Establish a realistic type-checking baseline

**Complete as of 2026-06-12.**

Resolution approach:

- Added `pandas-stubs` to dev dependencies — reduced pyright errors from 444 to 58.
- Added `_fit_transform_matrix` and `_transform_matrix` helpers to `ml_model_xgb_utils.py` with 2
  narrowly-scoped `type: ignore[return-value]` comments (sklearn's `ColumnTransformer` returns a
  broad inferred union; the helpers narrow it to `np.ndarray | spmatrix`).
- Updated all call sites (23 across 6 files) to use the typed helpers.
- Used `cast()` in test files where `None` was passed to non-Optional dataclass fields for
  serialization testing.
- Used a dead-code type-narrowing guard in `validation_utils.py` (pyright cannot narrow through
  `try/except`).
- Replaced the optional `shap` import with `importlib.import_module("shap")`, removing the need for
  a direct import suppression.
- Cleared the remaining `ty` diagnostics by tightening walk-forward metrics typing, avoiding
  heterogeneous `WalkForwardConfig(**dict)` reconstruction in tests, and adding small helper-level
  type clarifications in Polars and feature-importance code.

Exit criteria:

- ~~`.venv/bin/pyright .` passes.~~ **Done: 0 errors.**
- ~~`ty` either passes or has a documented advisory scope with explicit next steps.~~ **Done: `ty`
  passes and is now mandatory alongside `pyright`.**

### 4. Fix high-value operational issues

- Correct validation script exit-code propagation. Complete as of 2026-06-13: both validation
  scripts now raise `SystemExit(main())` so shell automation sees the correct non-zero status.
- Replace validation-script `print` usage with the project logger when appropriate. Complete as of
  2026-06-13: `scripts/validate_offline.py` and `scripts/validate_live.py` now log missing-data,
  warning, success, and mismatch states through the project logger.
- Refresh stale season-specific defaults in `config/weekly_run.yaml` and
  `scripts/betting_pipeline.py`. Complete as of 2026-06-12: weekly orchestration defaults now rely
  on runtime path/week inference instead of checked-in 2025 postseason values.
- Re-check editable install and key CLI help flows after those updates. Complete as of 2026-06-13:
  editable install plus the `nfl_predictor.ml_model`, `weekly_run.py`, and `power_rankings.py`
  help entrypoints all pass on Python 3.14.

Exit criteria:

- Validation scripts fail correctly in shell automation. Complete as of 2026-06-13.
- Checked-in config defaults no longer point at the 2025 postseason.

### 5. Re-establish quality gates

- The coverage audit and targeted test-addition pass now meets the preseason `90%` target.
- The next hardening slice should move to the remaining validation-script cleanup and CI workflow
  decisions.
- Completed coverage slices now include:
  - `nfl_predictor/ml/artifacts.py` and `nfl_predictor/utils/fingerprints.py` at `100%`.
  - `nfl_predictor/ml/sample_weights.py`, `nfl_predictor/utils/validation_utils.py`, and
    `nfl_predictor/ml/ml_model_cli.py` at `100%`.
  - `nfl_predictor/ml/feature_spec.py` at `100%` and `nfl_predictor/ml/feature_importance.py` at
    `98%`.
  - `nfl_predictor/data_collection.py` raised to `89%` through deeper ETL control-flow,
    orchestration, and merge-helper tests.
  - `nfl_predictor/ml/ml_model_core.py` raised to `76%` through helper-focused tests covering
    target-column selection, missing-data summaries, season bounds, and time-aware split helpers.
  - `nfl_predictor/ml/ml_model_training.py` raised to `88%` through orchestration, calibration,
    and guard-rail coverage.
  - `nfl_predictor/ml/leakage_audit.py` raised to `93%` through helper, heuristic, and report I/O
    coverage.
  - `nfl_predictor/utils/game_utils.py` raised to `94%` through QB-fill and future-line fallback
    coverage.
  - `nfl_predictor/utils/scraping_utils.py` raised to `96%` through TeamRankings parser,
    SurvivorGrid, and fallback-branch coverage.
  - `nfl_predictor/utils/polars/teamrankings.py` raised to `90%` through aggregation,
    filtering/conversion, and cache-fallback coverage.
  - `nfl_predictor/ml/walk_forward.py` raised to `94%` through helper-edge-case and market-aware
    backtest tests.
- Remaining optional coverage tails now concentrate in `nfl_predictor/ml/ml_model_core.py`,
  `nfl_predictor/ml/ml_model_training.py`, `nfl_predictor/utils/polars/loaders.py`,
  `nfl_predictor/utils/polars/features.py`, and a few parser-edge branches in
  `nfl_predictor/utils/scraping_utils.py`.
- Keep the enforced coverage floor at the preseason target of `90%` or higher, with `100%` as the
  aspirational ceiling.
- Document the canonical local validation sequence.
- Decide whether to add GitHub Actions now or defer CI until the preseason hardening pass is
  complete.
- If CI lands, keep the first workflow focused on validation and defer deployment concerns.
- Decide whether to add a tag-driven GitHub release workflow that uses `CHANGELOG.md`.

Exit criteria:

- The repository has one clearly documented validation gate.
- Coverage expectations are explicit and the repo now meets the preseason `90%` baseline.
- The intended CI and release-automation path is documented, even if implementation is deferred.

### 6. Resume the feature roadmap

- Return to Milestones 39-43 only after the baseline hardening work is complete.
- Prioritize the defaults sweep and power-rankings recency work before new model experiments.

Exit criteria:

- Baseline hardening is complete.
- Active roadmap work resumes from a clean, validated starting point.

## Suggested Execution Order

1. Align metadata, docs, and instructions.
2. Fix the remaining Ruff diagnostics.
3. Stabilize `pyright`.
4. Fix the operational issues in scripts and checked-in config.
5. Audit and improve coverage, then restore the coverage gate and document the final validation
   flow.
6. Finish the remaining validation-script cleanup and CI decisions.
7. Resume Milestones 39-43.

## Definition of Ready for the 2026 Season

The repo is ready to resume weekly work when all of the following are true:

- Ruff format, Ruff check, Pyright, Ty, pytest, and markdownlint pass.
- Editable install works.
- Checked-in configs target the current season rather than the 2025 postseason.
- Agent instructions, TODOs, README guidance, and `CHANGELOG.md` rules match the actual toolchain.
- The next active milestone can focus on forecasting capability rather than repo repair.
