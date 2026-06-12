# 2026 Preseason Readiness Plan

## Purpose

Bring the repository back to a coherent, season-ready baseline on Python 3.14 before resuming larger
feature work for the 2026-2027 NFL season.

## Validated Baseline

The following checks were run on 2026-06-12 against the refreshed `.venv`:

| Check                             | Result      | Notes                                                                         |
| --------------------------------- | ----------- | ----------------------------------------------------------------------------- |
| `.venv/bin/ruff format --check .` | Pass        | `104 files already formatted`                                                 |
| `.venv/bin/ruff check .`          | Pass        | All 25 E501 findings in `betting_excel.py` scoped with justified suppressions |
| `.venv/bin/pyright .`             | Fail        | Broad pandas-heavy typing issues                                              |
| `.venv/bin/ty check .`            | Fail        | Broad typing issues; currently advisory                                       |
| `.venv/bin/python -m pytest`      | Pass        | `238 passed`                                                                  |
| Coverage from pytest              | Fail target | `78%`, below the new preseason target of `90%` or higher                      |
| `markdownlint .`                  | Pass        | `0 error(s)`                                                                  |
| `uv lock --check`                 | Pass        | Lockfile is in sync with `pyproject.toml`                                     |
| `uv sync --check --active`        | Pass        | Active project environment matches `uv.lock`                                  |
| Primary CLI help smoke checks     | Pass        | `nfl_predictor.ml_model`, `weekly_run.py`, and `power_rankings.py`            |

## Tooling Recommendations

### Hatchling

Keep `hatchling`. It is actively used by `pip install -e . --no-deps`, which completed successfully
through the `pyproject.toml` build backend.

### Pyright vs Ty

Keep `pyright` as the primary blocking type checker for now. The codebase is pandas-heavy, and
`pyright` remains the more mature and predictable gate for this style of project.

Keep `ty` installed and run it alongside `pyright`, but treat it as advisory until the repo reaches
parity on both checkers. Replacing `pyright` with `ty` today would lower confidence rather than
raise it.

If a starter `[tool.ty]` configuration is added, keep it minimal:

```toml
[tool.ty.environment]
python = ".venv"
python-version = "3.14"

[tool.ty.src]
include = ["nfl_predictor", "scripts", "tests"]

[[tool.ty.overrides]]
include = ["tests/**"]

[tool.ty.overrides.rules]
all = "warn"
```

That keeps `ty` useful for signal gathering without pretending the repo is ready to gate on it.

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

- Fix the current `pyright` failures in `feature_spec.py`, `leakage_audit.py`, `walk_forward.py`,
  `power_rankings.py`, and the pandas-heavy tests.
- Decide where typed helper wrappers, casts, or localized overrides are justified.
- Add a minimal `[tool.ty]` section only if it improves signal.
- Re-run `pyright` and `ty` after each cluster of fixes.

Exit criteria:

- `.venv/bin/pyright .` passes.
- `ty` either passes or has a documented advisory scope with explicit next steps.

### 4. Fix high-value operational issues

- Correct validation script exit-code propagation.
- Replace validation-script `print` usage with the project logger when appropriate.
- Refresh stale season-specific defaults in `config/weekly_run.yaml` and
  `scripts/betting_pipeline.py`.
- Re-check editable install and key CLI help flows after those updates.

Exit criteria:

- Validation scripts fail correctly in shell automation.
- Checked-in config defaults no longer point at the 2025 postseason.

### 5. Re-establish quality gates

- Raise the enforced coverage floor from the old `80%` target toward the new preseason target of
  `90%` or higher, with `100%` as the aspirational ceiling.
- Document the canonical local validation sequence.
- Decide whether to add GitHub Actions now or defer CI until the preseason hardening pass is
  complete.
- If CI lands, keep the first workflow focused on validation and defer deployment concerns.
- Decide whether to add a tag-driven GitHub release workflow that uses `CHANGELOG.md`.

Exit criteria:

- The repository has one clearly documented validation gate.
- Coverage expectations are explicit and enforced or intentionally deferred.
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
4. Decide whether to add a minimal `[tool.ty]` section and keep `ty` advisory.
5. Fix the operational issues in scripts and checked-in config.
6. Restore the coverage gate and document the final validation flow.
7. Resume Milestones 39-43.

## Definition of Ready for the 2026 Season

The repo is ready to resume weekly work when all of the following are true:

- Ruff format, Ruff check, Pyright, pytest, and markdownlint pass.
- Editable install works.
- Checked-in configs target the current season rather than the 2025 postseason.
- Agent instructions, TODOs, README guidance, and `CHANGELOG.md` rules match the actual toolchain.
- The next active milestone can focus on forecasting capability rather than repo repair.
