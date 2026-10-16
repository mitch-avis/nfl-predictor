# Agent Instructions for nfl-predictor

This repository predicts NFL outcomes and scores for Pick 'Em and Confidence Pools. The primary
objective is calibrated win probabilities and weekly confidence rankings, measured against the
closing market line on the same games: "not worse than the market with better early-season
calibration" is the bar, and beating the closing line is a stretch goal, never a claim. Realistic
score outputs are secondary and display-only.

## Non-negotiables

- No data leakage: features, splits, calibration, blending and evaluation use only information
  known before the predicted week.
- Time-aware evaluation: tuning and model selection use blocked, time-ordered splits. Walk-forward
  is the authoritative evaluation, and the holdout window is never used to tune.
- Polars-first ETL; pandas and numpy only inside ML modules as needed. NFLverse via nflreadpy is
  the core data source: every nflreadpy source caches per season under `data/cache/nflreadpy/`
  and degrades non-fatally when the current season is not published yet.
- Reproducible artifacts: training and backtests write run directories with metadata and
  metrics, and any number quoted in the docs is auditable from disk under `models/`, except
  from the runs listed in `.agents/models_cleanup/deleted_2026-10-16.txt`, whose directories were
  deleted on 2026-10-16.
  Outputs are deterministic under fixed seeds.
- Tests are required, test-first by default: before changing production code, make sure the
  lines you will touch are covered, adding characterization or failing tests first. Delegated
  work runs as planner, implementer and reviewer (the global rule); the task text carries the
  contract the tests must satisfy.
- XGBoost margin/total is the primary model and the benchmark. No alternative model families or
  large tuning campaigns unless the user asks.

## Where things live

- `.agents/TODO.md` is the active worklist and the roadmap order ("Roadmap Status"); work only on
  the highest-priority blocking items. `.agents/ARCHIVE.md` holds completed milestones, whose
  numbers never change; a new milestone takes the next number after the highest in either file.
- `.agents/next_agent_session_prompt.md` is the handoff (branch, version, uncommitted state,
  next task, open questions). Rewrite it at every landed chunk. It is injected at session start
  and after every compaction.
- `.agents/modeling_spec.md` specifies targets, the probability path, market anchoring, pool
  scoring, evaluation, the artifact contract, the leakage audit, feature rules and the season and
  week conventions. Read it before changing prediction, calibration, market, pool, evaluation,
  artifact or feature code.
- `.agents/benchmarks.md` holds the benchmark table, the reference arm per dataset build, the ETL
  rebuild records, the tree-budget ladder, the season-weighting arms and the fit-noise floors.
  Read it before any walk-forward run, comparison or decision. The floors in short: on one
  three-season arm, a Brier difference under about `0.002`, pick accuracy under `0.01` or margin
  MAE under `0.06` is re-seeding noise; on six seasons, `0.0013` Brier, `0.008` pick accuracy,
  `75` pool points or `0.03` margin MAE.
- `.agents/guardrails.md` is the long form of the delegation guardrails below, and
  `.agents/guardrail_history.md` the incidents behind them. `.agents/walk_forward_runbook.md`
  has run timings, machine load and how to launch a run. `.agents/feature_crosswalk.md` is the
  cross-repo feature review. `.agents/frontend_audit_notes.md` collects web-app problems noticed
  in passing (wrong data on a page, component and accessibility defects): add one when you see
  one. `CHANGELOG.md` is the release history.
- Nested instructions load on demand: `src/nfl_predictor/ml/AGENTS.md`,
  `src/nfl_predictor/api/AGENTS.md`, `web/AGENTS.md`, and the path-scoped rules under
  `.claude/rules/` for `constants.py`, the ETL helpers and the model code.

## Delegation guardrails

Binding in every session; definitions, limits and examples are in `.agents/guardrails.md`.

1. **One gate.** `scripts/gate.sh` exiting `0` on the final tree is the only definition of done;
   `--quick` never counts for the report.
2. **Narrowing is never a checkbox.** A delivery that differs from the task text stays `[ ]`
   with a `Narrowed:` note and goes on the user's question list. Small factual corrections
   outside the task (a stale name, a typo, a bug with one documented fix; no output, record or
   fingerprint change) may be fixed on sight and reported under "Fixed without asking".
3. **Two keys on every number.** The agent that produced a run never writes its numbers into
   the docs; an independent reviewer rescores from the fold checkpoints, checks provenance and
   writes `REVIEW.md` first.
4. **Compute budget.** Write the hypothesis and the decision rule (windows, columns, how they
   combine) before each walk-forward; after two runs on one task without a decision, stop and
   ask. An accepted ladder with a stated cap approves its rungs. One walk-forward at a time.
5. **Must ask first:** rebuilding, deleting or overwriting anything under `data/` or `models/`;
   changing a default that alters a weekly run or feature values; reopening, reordering or
   retiring (dropping undone) roadmap items; merging to `main` or pushing (tags and releases:
   never); touching `../nfeloqb`, `../nfl-sos-ratings` or the running web API; a third
   walk-forward on one task or any six-season run outside an accepted ladder; anything the task
   says to decide with the user.
6. **May proceed:** commits on the feature branch per versioned chunk; fixes with a failing
   test first; corrections under rule 2; doc updates restating verified numbers; one
   walk-forward per written hypothesis; creating the milestone branch off `main`; closing a
   finished roadmap item once its independent review passed and the gate is green.
7. **Check-ins** at every landed version and after every walk-forward: what landed, the run
   directory, the gate result, the governing-window numbers, the open questions, and how long
   runs are sequenced and watched. Never promise to "keep monitoring" with nothing watching.
8. **Handoff hygiene.** Rewrite the handoff file at every landed chunk.
9. **Read every interval.** List every paired interval that excludes zero in every window and
   where each arm ranks on every named column; a tie goes to the simpler setting and to the user.
10. **Inventories are generated** by a checked-in script; a failed search is reported, never
    filled by inference.
11. **The benchmark measures what production does.** Any difference between the reference
    walk-forward configuration and the production weekly run is a defect until the user approves
    it and it is recorded in `.agents/benchmarks.md`.
12. **A decision rule is applied as written;** a result between its branches goes to the user.
13. **Two seeds before any default changes,** combined per game and bootstrapped over games.
14. **Order of work:** restructure before changing behavior; close parity gaps before new
    measurements; change feature values before tuning; group checkpoint-invalidating changes;
    production-changing steps land between game weeks.
15. **Recommend when the answer isn't obvious:** one line on what the component does today
    (from the code, naming the file), the recommended option with its cost, and how sure you are.
    A recommendation is never consent.
16. **Name things in plain words:** arms, builds, runs, settings and steps get descriptive names,
    with a code at most once in parentheses.

## Commands and the gate

- Every Python tool runs from the repo's virtualenv: `.venv/bin/python`, `.venv/bin/pytest`,
  `.venv/bin/ruff`, `.venv/bin/pyright`, `.venv/bin/ty` and `.venv/bin/nfl-predictor <command>`.
  Never bare `python`, `pip`, `pytest`, `ruff`, `pyright` or `ty`, and never `uv run <tool>`,
  which syncs the venv first and can replace the CUDA LightGBM build. `uv` comes from `PATH`.
- `scripts/gate.sh` runs every check the way CI does (lock, sync, ruff, ty, pyright, pytest,
  the `src/` coverage bar, every console script's and front-door command's `--help`, shellcheck,
  markdownlint). It is shared with the other projects through the Copier template
  `~/workspace/python-template` (`.copier-answers.yml`); this repo's own steps live in
  `scripts/gate.hooks.sh`, and `copier update` brings in template changes. Add `--web` whenever
  `web/` or `src/nfl_predictor/api/` changed; `--quick` skips pytest while iterating.
- After every version bump, including a merged one, run
  `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`. A plain `uv sync` replaces the local
  CUDA build of LightGBM with the CPU wheel.
- Key ML dependencies stay pinned in `pyproject.toml` and `uv.lock`; optional GPU paths never
  break CPU-only execution. The full dependency refresh is `update_requirements.sh`: ask first.

## Versions, changelog and commits

- `CHANGELOG.md` follows Common Changelog (`## [VERSION] - YYYY-MM-DD`, then `Changed`, `Added`,
  `Removed`, `Fixed`). Add an entry as each chunk lands, each under its own incremented version
  and never `[Unreleased]`: patch for fixes and small additions, minor for a new feature family,
  a changed default or a schema change. Set `pyproject.toml` to the same version in the same
  change, then run `uv lock` and the sync above.
- The project is private: never create or push a git tag or a GitHub release.
- Conventional Commits, with the type list enforced by the commit hook and scopes naming the
  area (`etl`, `ml`, `rankings`, `reporting`, `agents`, `walk-forward`). One logical change per
  commit for multi-file schema work, otherwise one file per commit unless the user asks otherwise.

## Repository shape

- `src/nfl_predictor/`: `data_collection.py` and `utils/polars_utils.py` (the Polars ETL
  helpers; pandas CSV helpers are legacy), `constants.py` (paths, column lists, team mapping,
  season and week rules: the canonical schema reference, so never hard-code a column list),
  `ml/` (model code; any edit changes every walk-forward checkpoint fingerprint), `weekly_run/`,
  `cli/`, `reporting/`, `api/` (the FastAPI backend) and `web/` (the Vite + React app).
- Neighbors: `../nfeloqb` produces `qb_elos.csv` (538-style schema, `team1` is home), consumed
  as a manually copied `data/qb_elos.csv`; treat it as a downstream contract and never edit that
  repo. `../nfl-sos-ratings` is being absorbed into this repo (roadmap step 6) from the frozen
  commit in `.agents/sos_port_source.md`: read it there, read-only, never import it; the
  repo-root symlink to it exists only for reading. Leave untracked files in neighbors alone.
- Style: logging through `from nfl_predictor.utils.logger import log`, no `print`; Polars
  expressions over Python loops in ETL; type hints on all new or modified code; a file past about
  2,000 lines gets a split proposal; no roadmap labels, milestone or task numbers in code,
  comments, docstrings or tests.
- Lint and type findings get fixed, never silenced inline: no `noqa`, `type: ignore`,
  `pyright: ignore` or `pragma: no cover`. A genuine false positive, checked against Ruff's own
  settings first, goes in `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml` with a one-line
  reason beside the rule. `tests/test_no_inline_suppressions.py` fails on any of these, on
  `ty: ignore`, file-level `noqa` or a pyright rule switched off or downgraded, in every tracked
  `.py` and `.pyi`.
- Keep `README.md`, nested READMEs, `.agents/TODO.md`, `.agents/ARCHIVE.md`,
  `.agents/feature_crosswalk.md` and `CHANGELOG.md` in step with the repo after meaningful
  progress. No unrelated features, scrapers or external services, and no betting-profitability
  claims.

## Modeling rules never to miss

- The model predicts `margin = home_score - away_score` and `total = home_score + away_score`;
  scores derive from them. Win probability is the deterministic floor `Phi(margin / sigma)` in
  every run type, sigma from `src/nfl_predictor/ml/floor_sigma.py`; the pick and its confidence
  come from the unrounded probability.
- The market enters only as features and through anchoring: no blend or clamp on the
  probability. Edges against the market are diagnostics only.
- Confidence pools: unique `1..N` per week; a tie scores as incorrect for both sides; picks are
  single-shot before the week's first game.
- Realistic score adjustments are display-only and never touch probabilities, rankings, pool
  scoring, tuning objectives or training targets.
- Every saved model carries adjacent metadata JSON and loads without hidden state;
  `nfl-predictor leakage-audit` stays maintained.
- A schedule- or opponent-adjusted value for week `N` is solved from games strictly before
  week `N`. A change to feature values needs two dataset builds, not `--disable-feature-groups`.

## Runs and the running server

- One walk-forward at a time: check `uptime` and `pgrep -af "nfl-predictor (backtest|weekly)"`
  first. Launch every run through a `launch.sh` in its own run directory with `nohup setsid`,
  never from a harness-bound shell (10-minute limit); accepted rungs run back to back through
  one driver script. The runbook has the timings and the launch lessons.
- Runs checkpoint every finished week under `models/wf_checkpoints/<fingerprint>/`, and
  rerunning the identical command resumes. Never delete `models/<run_id>/wf_compare/` during a
  run, and keep comparison artifacts under `models/`.
- `nfl-predictor web` serves `web/dist/` on port 8000. `--reload` restarts on any Python file
  change (an agent's edits, checkouts and merges included) and fails running jobs: never launch
  a web job from a `--reload` server, and say so before editing code while one runs. Never stop
  or restart the user's server.
