# Next Agent Session Prompt

Use the following prompt to start the next agent session on this repository.

```text
You are continuing preseason 2026 hardening work on the nfl-predictor repository.

Before changing code, read these files first:
- `AGENTS.md`
- `TODO.md`
- `docs/preseason_2026_readiness_plan.md`
- `README.md`
- `CHANGELOG.md`

Repository goals and constraints:
- Preserve the repo's no-leakage and time-aware evaluation guarantees.
- Keep ETL Polars-first and NFLverse/nflreadpy-based.
- Keep outputs reproducible and artifact-friendly.
- Do not reintroduce broad noqa, type: ignore, or pragma suppressions unless they are truly
  necessary, narrowly scoped, and justified.
- Keep documentation current as you go: `README.md`, `AGENTS.md`, `TODO.md`, `ARCHIVE.md`, `CHANGELOG.md`,
  and the preseason plan if the baseline or priorities change.
- Keep `CHANGELOG.md` in Common Changelog order: `Changed`, `Added`, `Removed`, `Fixed`.
- Public docs should stay user-friendly and assume an activated virtual environment.
- Agent-facing docs may continue using explicit `.venv/bin/python`-style commands for Python tools.
- If asked to commit, prefer one file per commit, including deletions, unless told otherwise.

Current validated baseline as of 2026-06-12:
- `.venv/bin/ruff format --check .` passes (104 files already formatted)
- `.venv/bin/ruff check .` passes cleanly (0 diagnostics; all 69 findings resolved)
- `.venv/bin/pyright .` passes (0 errors) after adding `pandas-stubs` and typed transform helpers
- `.venv/bin/ty check .` passes (0 errors) after cleaning the remaining helper, walk-forward, and
  test-surface diagnostics
- `.venv/bin/python -m pytest` passes (326 passed)
- Coverage is 86%, below the preseason target of 90% or higher
- `markdownlint` passes on the maintained Markdown docs
- `uv lock --check` passes
- `uv sync --check --active` passes

Active milestone:
- Milestone 44 in `TODO.md`: Preseason 2026 repo hardening and tooling alignment

Recommended first work slice:
- Pyright and Ty are both green. Stay on the remaining Milestone 44 items.
- Start with the next deliberate coverage-improvement pass before the other Milestone 44 slices.
- The highest-yield completed slices so far are now: artifact/repro helpers, validation utilities,
  sample weights, `ml_model_cli`, `feature_spec`, `feature_importance`, `data_collection`, and
  `walk_forward`.
- The next best targets are `nfl_predictor/ml/ml_model_core.py` first, then
  `nfl_predictor/ml/ml_model_training.py`, with small cleanup tails remaining in
  `nfl_predictor/ml/feature_importance.py`, `nfl_predictor/data_collection.py`, and
  `nfl_predictor/ml/walk_forward.py`.
- After the coverage slice, take the validation script exit-code/logger cleanup slice.
- After that, evaluate and implement the GitHub Actions validation workflow if it still belongs in
  Milestone 44.
- Treat TDD as mandatory for any code change: confirm the exact behavior/lines you plan to touch
  are covered first; if they are not, add focused characterization or failing tests before editing
  production code.
- Load relevant skills before acting. For most repo work, start with `python` plus whichever of
  `test-driven-development`, `clean-code`, `systematic-debugging`, `code-review`,
  `observability`, `task-orchestrator`, and the `python-*` skills fit the slice.
- Use localized type: ignore or cast() only when truly necessary and narrowly scoped.
- Avoid broad suppressions; prefer typed helper wrappers or explicit annotations.
- Treat both `.venv/bin/pyright .` and `.venv/bin/ty check .` as mandatory validation gates.

Suggested execution flow:
1. Start with a coverage audit: inspect the latest pytest coverage output and identify the highest-
  value low-coverage modules.
2. Pick the smallest coherent coverage slice that improves risk, not just percentage.
3. Read the owning modules/tests/docs before editing.
4. Verify direct coverage for the exact behavior/lines you plan to change; add focused tests first
  if coverage is missing.
5. Implement the smallest coherent change.
6. Run the narrowest relevant tests/checks first, then the broader repo validation commands.
7. Update planning docs if the baseline, scope, or priorities change materially.

Validation commands:
- `.venv/bin/ruff format --check .`
- `.venv/bin/ruff check .`
- `.venv/bin/pyright .`
- `.venv/bin/ty check .`
- `.venv/bin/python -m pytest`
- `markdownlint .`
- `uv lock --check`
- `uv sync --check --active`

What to update as you work:
- `TODO.md`: keep the active checklist accurate
- `ARCHIVE.md`: move completed items out of `TODO.md` when they are actually done
- `docs/preseason_2026_readiness_plan.md`: update baseline counts or workstream guidance if the
  situation changes
- `README.md`: keep user-facing workflow and guidance accurate
- `CHANGELOG.md`: add or revise the current release entry when notable behavior or workflow changes

Deliverables for this session:
- Real code or documentation changes, not just analysis
- A reduced risk or failure surface for the chosen slice
- A concrete coverage improvement plan or measurable coverage increase for the selected modules
- Updated planning/docs if the repo state changed
- A concise summary of what changed, what passed, and what remains
```
