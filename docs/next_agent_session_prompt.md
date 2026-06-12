# Next Agent Session Prompt

Use the following prompt to start the next agent session on this repository.

```text
You are continuing preseason 2026 hardening work on the nfl-predictor repository.

Before changing code, read these files first:
- AGENTS.md
- TODO.md
- docs/preseason_2026_readiness_plan.md
- README.md
- CHANGELOG.md

Repository goals and constraints:
- Preserve the repo's no-leakage and time-aware evaluation guarantees.
- Keep ETL Polars-first and NFLverse/nflreadpy-based.
- Keep outputs reproducible and artifact-friendly.
- Do not reintroduce broad noqa, type: ignore, or pragma suppressions unless they are truly
  necessary, narrowly scoped, and justified.
- Keep documentation current as you go: README.md, AGENTS.md, TODO.md, ARCHIVE.md, and the
- Keep documentation current as you go: README.md, AGENTS.md, TODO.md, ARCHIVE.md, CHANGELOG.md,
  and the preseason plan if the baseline or priorities change.
- Keep `CHANGELOG.md` in Common Changelog order: `Changed`, `Added`, `Removed`, `Fixed`.
- Public docs should stay user-friendly and assume an activated virtual environment.
- Agent-facing docs may continue using explicit .venv/bin/python-style commands for Python tools.
- If asked to commit, prefer one file per commit, including deletions, unless told otherwise.

Current validated baseline as of 2026-06-12:
- .venv/bin/ruff format --check . passes
- .venv/bin/python -m pytest passes (238 passed)
- Coverage is 78%, below the preseason target of 90% or higher
- markdownlint passes on the maintained Markdown docs
- uv lock --check passes
- uv sync --check --active passes
- .venv/bin/ruff check . fails with 69 diagnostics
- .venv/bin/pyright . fails broadly, especially in pandas-heavy modules
- .venv/bin/ty check . fails and remains advisory

Active milestone:
- Milestone 44 in TODO.md: Preseason 2026 repo hardening and tooling alignment

Recommended first work slice:
- Start with the remaining Ruff diagnostics, not the type-checking failures.
- Prefer the smallest coherent cluster that can be fixed and validated end to end.
- A strong first target is the low-risk documentation/lint cluster:
  - package/module docstrings
  - imperative docstring rewrites
  - validation/logger correctness cleanup
  - safe simplify fixes
- Avoid starting with the large pandas typing surface unless Ruff debt is materially reduced first.

Suggested execution flow:
1. Re-run the current baseline checks you need for the selected slice.
2. Pick one coherent lint cluster.
3. Add or update tests first when behavior changes.
4. Implement the smallest grounded fix set.
5. Re-run the narrowest relevant validation immediately.
6. Update README.md and planning docs if behavior, workflow, or baseline status changes.

Validation commands:
- .venv/bin/ruff format --check .
- .venv/bin/ruff check .
- .venv/bin/pyright .
- .venv/bin/ty check .
- .venv/bin/python -m pytest
- markdownlint .
- uv lock --check
- uv sync --check --active

What to update as you work:
- TODO.md: keep the active checklist accurate
- ARCHIVE.md: move completed items out of TODO.md when they are actually done
- docs/preseason_2026_readiness_plan.md: update baseline counts or workstream guidance if the
  situation changes
- README.md: keep user-facing workflow and guidance accurate
- CHANGELOG.md: add or revise the current release entry when notable behavior or workflow changes

Deliverables for this session:
- Real code or documentation changes, not just analysis
- A reduced failure surface for the chosen slice
- Updated planning/docs if the repo state changed
- A concise summary of what changed, what passed, and what remains
```
