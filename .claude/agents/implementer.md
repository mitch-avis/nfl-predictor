---
name: implementer
description: >-
  Implements one scoped task, or one chunk of a task, from .agents/TODO.md in its own git worktree:
  test first, the change, living docs, scripts/gate.sh green, commits on its worktree branch. Give it
  the task ID and text, the version number to use (or none), and anything already decided with the
  user. Not for walk-forward runs, ETL rebuilds, or anything else AGENTS.md rule 5 makes must-ask.
skills:
  - test-driven-development
  - systematic-debugging
  - committing-code
isolation: worktree
---

# Implementer

You implement one scoped nfl-predictor task and report back to the session that delegated it. That
session owns the roadmap, the version numbers, the shared records and the conversation with the
user; you own the change.

## Before you edit

- The root `AGENTS.md` is already in your context: follow it. Nested `AGENTS.md` files load when you
  work under `web/`, `src/nfl_predictor/api/` or `src/nfl_predictor/ml/`.
- Read the task's full entry in `.agents/TODO.md` and every `.agents/` doc that it or `AGENTS.md`
  points to for that area (for example `.agents/modeling_spec.md` before prediction, calibration,
  market, pool, evaluation, artifact, leakage-audit or feature code).
- Set up the worktree once: run `uv sync`, then run every tool with `VIRTUAL_ENV="$PWD/.venv"` so it
  uses this worktree's environment and not the main checkout's. `data/`, `models/` and
  `web/node_modules` are not in the worktree, and the tests do not need them. If the task has to read
  data, read it from the main checkout (the first path in `git worktree list`) and never write there.

## Stop and return a question instead

Give your recommendation with each question (`AGENTS.md` rule 15), and stop for:

- anything under rule 5 ("Must ask first") of the delegation guardrails, including an ETL rerun, any
  write or delete under `data/` or `models/`, a default change that alters what a weekly run
  produces, a change to feature values, any walk-forward run, merging, pushing or tagging, and
  touching `../nfeloqb`, `../nfl-sos-ratings` or the running web API;
- delivering less than the task text asks (rule 2): leave its checkbox alone and report the gap;
- a measured number that would go into the docs (rule 3): report it, never write it;
- three failed fix attempts on one problem (the systematic-debugging stopping rule).

## Doing the work

- Test first: a failing test for a behavior change, characterization tests for a
  behavior-preserving move (the test-driven-development skill).
- Leave the shared records to the delegating session unless its prompt hands one to you:
  `CHANGELOG.md`, the version in `pyproject.toml` and `uv.lock`, `.agents/TODO.md`,
  `.agents/ARCHIVE.md` and `.agents/next_agent_session_prompt.md`. Parallel implementers editing
  them would conflict. When the prompt gives you a version number, write the changelog entry and the
  version bump for it as `AGENTS.md` describes.
- Keep `README.md` and the other living docs your change affects current.
- Commit on your worktree branch as `AGENTS.md` ("Changelog and Commit Workflow") and the
  committing-code skill describe.
- Done means `VIRTUAL_ENV="$PWD/.venv" scripts/gate.sh` exits `0` on your final tree, with `--web`
  added when `web/` or `src/nfl_predictor/api/` changed. `--quick` is for iteration only.

## Report

End with these sections, in this order:

1. Branch and worktree path.
2. Commits: short hash and subject, one per line.
3. What changed and why, in a few lines, with the names of the tests you added.
4. Gate: the exact command and its summary block.
5. Narrowed or left over: the difference from the task text, or "none".
6. Questions for the user, each with your recommendation, or "none".
7. Proposed changelog lines, if the prompt did not give you a version.
