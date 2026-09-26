# Agent Instructions for nfl-predictor

## 0) Mission + Non-Negotiables

This repository predicts NFL outcomes and scores for **Pick 'Em** and **Confidence Pools**.

- **Primary objective:** calibrated win probabilities and weekly confidence rankings.
- **Standing yardstick:** the closing market line. Every walk-forward report compares the model's
  win probability and margin against the market-implied probability and the spread on the same
  games; the model is measured by that difference, and "not worse than the market with better
  early-season calibration" is the current bar. Beating the closing line is a stretch goal, never a
  claim.
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

## Current Focus (2026 season start)

- The active workstream is feature engineering for true team strength. What has landed, version
  by version, with the user's decisions and the verification arms, is under "Workstream history"
  in `.agents/benchmarks.md`; completed milestones are in `.agents/ARCHIVE.md`.
- The order agreed with the user on 2026-09-24, each step on its own branch and merged before
  the next, with no deadline:
  (1) close 55.8 (done, `0.18.0`);
  (2) Milestone 60, CLI and entrypoint consolidation widened to every file under `scripts/`,
  behavior-preserving, with task 55.5 (done: closed and merged into `main` on 2026-09-25 as
  `0.28.1`, record in `.agents/ARCHIVE.md`, Milestone 60);
  (3) production/benchmark parity: GPU as the default device (55.4), how production
  probabilities are formed (56.5), pick-time lines (56.6), and the out-of-fold calibration pool;
  (4) rebuild reproducibility, then every feature-value change (55.3, 53.7 and the feature
  follow-ups);
  (5) the Optuna re-tune with wiring into production (55.9 with 56.3);
  (6) the web UI, Milestone 58 phases 4-6.
  Tasks 55.1 and 55.2 are retired. The reasoning and the follow-up assignments are under "Roadmap
  Status" in `.agents/TODO.md`. Step 3 is next.
- XGBoost margin/total stays the primary model and benchmark. Do not build alternative model
  families or run large tuning campaigns unless the user asks.
- Borrow proven methodology from `../nfl-sos-ratings` before inventing new metrics; treat that
  repo as read-only reference material. The method being ported is its head-to-head-excluded
  opponent profiling and the simultaneous ridge that generalizes it (see
  `.agents/feature_crosswalk.md` section 3.1).
- Validated baseline on 2026-09-25 (version `0.28.5`): `scripts/gate.sh --web` exits `0`
  (`1032 passed`, coverage `92.21%` against the enforced `90%` floor, 22 frontend tests). Run
  `scripts/gate.sh --web` whenever `web/` or `nfl_predictor/api/` changes.
  - Re-run a plain `uv sync` after every version bump (including after merging a branch that
    bumped the version), or `uv sync --check --active` fails on the stale installed package.
- **Benchmarks and data state live in `.agents/benchmarks.md`.** Read it before any walk-forward
  run, comparison or decision: it holds the current benchmark table, the reference arm for each
  dataset build (a new arm compares against the reference on its own build and code version),
  the ETL rebuild records, the tree-budget ladder, the season-weighting arms and the fit-noise
  floors. The noise floors in short: on one three-season arm, a Brier difference under about
  `0.002`, a pick-accuracy difference under about `0.01` and a margin MAE difference under about
  `0.06` are indistinguishable from re-seeding; on six seasons, a single-seed difference under
  about `0.0013` Brier, `0.008` pick accuracy, `75` pool points or `0.03` margin MAE is within
  re-seeding noise.

### Readiness behaviors that must not regress

- Pre-kickoff current-week detection resolves to the new season's Week 1, not the prior season's
  final playoff week.
- Current-season nflreadpy team-stat 404s are non-fatal so ETL still runs when the schedule exists
  before weekly stats are published. Any new nflreadpy source (PBP included) must follow the same
  cache-then-degrade pattern.
- Default weekly prediction-file resolution uses the CSV `season`/`week` values, not the filename
  week, so a stale `week_22` file never wins over a new `week_01` file.

## Delegation guardrails (every agent session)

Added 2026-09-19 after the Milestone 59 audit and amended on 2026-09-21, 2026-09-23, 2026-09-24
and 2026-09-25; what went wrong each time, and so why each rule exists, is in
`.agents/guardrail_history.md`. An autonomous session has no other supervision, so these rules
are not advisory.

1. **One gate.** `scripts/gate.sh` is the definition of "checks pass". No task, chunk or version
   is reported done, and no changelog entry is written as landed, until it exits `0` on the
   final tree. Reporting a subset of checks as the gate is a defect; `--quick` is for iteration,
   never for the report.
2. **Narrowing is never a checkbox.** If what landed differs from the task text (smaller scope, a
   substitute method, a skipped acceptance criterion), the task stays `[ ]` with a `Narrowed:`
   note giving the difference, the reason and where the remainder now lives, and the difference
   goes on the user's question list. Rewriting the acceptance text to fit the delivery is not
   allowed. **Correcting is not narrowing:** an error noticed outside the current task may be
   fixed without asking first when all of these hold; if any fails, it is a question.
   - It is a fact, not a choice: a stale path, command or option name, a broken pointer, a
     typo, or a bug whose correct behavior the repo already states (a docstring, test,
     documented contract or error message), with exactly one plausible fix.
   - Meaning is unchanged. Corrected task or acceptance text names the current equivalent of
     what was asked, keeps the same bar and every criterion, and no checkbox changes state. A
     code fix changes no output of a weekly run, walk-forward or ETL, and touches no
     fingerprinted file (`nfl_predictor/ml/*.py`, `constants.py`, `ml_model.py`).
   - Nothing under rule 5 is involved, no measured number changes (rule 3), and records stay as
     written: `.agents/ARCHIVE.md`, past `CHANGELOG.md` entries, run directories, benchmark
     provenance.
   - It is small and verified: one logical change of a few lines, a failing test first for
     code, `scripts/gate.sh` green.
   - The user is told: the next check-in lists it under "Fixed without asking" (before, after,
     why), and a code or tooling fix gets a changelog line.
3. **Two keys on every number.** The agent that produced a run never writes its numbers into
   `AGENTS.md`, `.agents/benchmarks.md`, `.agents/ARCHIVE.md`, `.agents/TODO.md`
   or `CHANGELOG.md`. A reviewer (a separate
   subagent, or the next session) rescores the artifact from disk, writes the run directory and
   the reproduction command beside the number, and only then may the docs change. A number in
   the docs without a run directory is a defect. The review is independent only if (a) its
   `REVIEW.md` names who reviewed and states that the reviewer did not produce the run; (b) it
   recomputes the metrics from the fold checkpoints (`models/wf_checkpoints/<fingerprint>/`), not
   from `metrics_report.json` or only through the producer's comparison script; and (c) it checks
   provenance: dataset hash, git commit, fold count, `best_iteration`/`early_stopped`, and that
   only the intended setting differs between arms. If a reviewer subagent fails or returns
   without numbers, the producing agent reports that and stops; it never writes the review
   itself.
4. **Compute budget.** Before each walk-forward run, write the hypothesis and the decision rule
   (which result changes what). The decision rule names the exact window or windows it reads
   (week 1, week 2, weeks 3-18, all weeks), the exact columns (deterministic Brier, log loss,
   pick accuracy, margin MAE, the paired deterministic-minus-market interval), and how the
   windows combine when they disagree, so a rung cannot read as a tie on one window and a win
   on another without the rule saying which one governs. After two runs on one task without a
   decision, stop and ask. A **ladder** (several runs of one hypothesis family, with a stated
   cap on the number of rungs and a stopping rule) written into the check-in and accepted by
   the user counts as approval for every rung up to that cap; only a rung beyond the cap, or
   outside the ladder as written, needs a fresh ask.
   One walk-forward at a time; `uptime` and `pgrep -af walk_forward` first; OpenMP policy by
   load. Any edit under `nfl_predictor/ml/` changes every checkpoint fingerprint, so a rerun
   after a code change retrains from scratch; plan runs after the code is stable.
5. **Must ask first** (stop and wait; never assume):
   - rebuilding anything under `data/` (an ETL rerun), or deleting or overwriting any file under
     `data/` or `models/`;
   - changing a default (CLI, config, constants) that alters what a weekly run produces, or any
     change that alters feature values at ETL time;
   - closing a milestone, reopening a parked one (Milestone 57), or reordering the roadmap;
   - merging to `main` or pushing (tags and releases: never, see above);
   - touching `../nfeloqb`, `../nfl-sos-ratings`, or the running web API (`nfl-predictor web`,
     port 8000 by default);
   - a third walk-forward run on one task, or any six-season run, unless it is a rung of a
     ladder the user has already accepted under rule 4 and is within that ladder's cap;
   - anything the task text says to decide with the user.
6. **May proceed without asking:**
   - commits on the working feature branch after each versioned chunk (Conventional Commits, one
     logical change per commit, the attribution line the harness provides);
   - fixes with a failing test first within the current task, corrections under rule 2, and doc
     updates that restate numbers already verified under rule 3;
   - one walk-forward run per written hypothesis, within rule 4;
   - creating the milestone's feature branch off `main` when none exists.
7. **Check-ins.** Report at every landed version and after every walk-forward run: what landed,
   the run directory, the gate result, the open questions. Stop for a question whenever rule 5
   triggers; a session that ends blocked on a question has done the right thing. The check-in
   after a walk-forward run carries its governing-window numbers (not only "it finished"), and it
   is written before the next rung is launched. A session that launches long runs says in the
   same check-in how they will be sequenced and supervised: either one driver script that runs
   the accepted rungs back to back, or a watcher that wakes the session when the run exits.
   Never end a turn while promising to "keep monitoring" with nothing actually watching.
8. **Handoff hygiene.** Rewrite `.agents/next_agent_session_prompt.md` at every landed chunk
   (branch, version, uncommitted state, the next task, open questions) so a restart after a
   closed terminal or a context summary resumes without re-deriving anything.
9. **Read every interval; the incumbent gets no benefit of the doubt.** A decision or close-out
   lists every paired interval that excludes zero, in every window reported and in either
   direction, plus where each arm ranks on every named column and tie-breaker (deterministic
   Brier, log loss, margin MAE, total MAE, pick accuracy, confidence-pool points). "No arm beat
   X" is incomplete without how X compares to the others. A shipped setting that was never
   measured is an assumption, not a baseline: when a measurement ties, the simpler setting (fewer
   knobs, fewer assumptions) is the recommendation, and the tie goes to the user as a question
   rather than closing in the incumbent's favor.
10. **Inventories and audits are generated, not written.** Any inventory of code (flags, files,
    columns, consumers, call sites) is produced by a script checked in beside it, and its
    headline counts are reproduced by that script. A search or tool call that fails, errors, or
    returns "No matches found" where matches must exist is reported as a failure. The gap is
    never filled by inference, and a subagent's summary is input to verify, not a source to
    copy.
11. **The benchmark measures what production does.** Any setting that differs between the
    reference walk-forward configuration (the benchmark arms in `.agents/benchmarks.md`) and the
    production weekly run (`config/weekly_run.yaml` and the weekly stage-1 selection) is a
    defect until the user approves it and it is recorded here. The season-weighting gap closed in
    `0.18.0` (both train unweighted, task 55.8). Open gaps, all scheduled:
    - the probability path: production submits the stage-1 winner, currently `elo` with a market
      blend and clamp, while the benchmark scores the deterministic map (task 56.5);
    - the device: the weekly run trains on the GPU, standalone walk-forwards on the CPU (task
      55.4; the user chose the GPU for everything, 2026-09-23);
    - line timing: backtests anchor to and score against the stored, probably closing, lines,
      while production anchors to mid-week lines and picks are made before Thursday (task 56.6).
12. **A decision rule favors no outcome after the fact.** The rule written under rule 4 is
    applied as written. If the result falls between its branches, or its first condition fails
    and a fallback branch rescues the preferred outcome, the result goes to the user as a
    question with the numbers.
13. **Two seeds before any default changes.** A walk-forward result that would change a default
    (a setting, a feature family kept or dropped, a tuned parameter) holds on two seeds. Combine
    them per game: the candidate-minus-reference loss difference averaged over the seeds (same
    seed paired with same seed), bootstrapped over games. Game-resampled intervals hold the fit
    fixed, so they leave out seed-to-seed variance: a single-seed six-season interval that
    excludes zero is not enough on its own, above all for pick accuracy and pool points. A new
    reference arm (a new build, device or code path) is measured on two seeds too, which also
    records that reference's own noise floor.
14. **Order of work.** Restructure before changing behavior (a behavior-preserving move is
    pinned by a characterization test and lands before any output-changing task that touches the
    same code). Close benchmark/production parity gaps before new measurements (rule 11). Change
    feature values before tuning. Group changes that invalidate walk-forward checkpoints so they
    share one new reference. Production-changing steps land between game weeks. The step order
    in `.agents/TODO.md` ("Roadmap Status") follows these principles; changing it is must-ask.
15. **Recommend when the answer isn't obvious.** A question or pending decision for the user
    carries the agent's brief recommendation unless the answer is plain from the question itself.
    The user may not remember the details of something built long ago, or may not know the area
    well, so write for a reader who cannot check it without digging: one line on what the
    component does today and why the decision comes up, from the code or docs (name the file),
    not from memory; the recommended option first, with its reason and its main cost or risk;
    how sure the agent is and what would change its mind. If the evidence is too thin to
    recommend, say so and name what would settle it. A recommendation is advice, never consent:
    the agent still waits for the answer on everything under rule 5, and it never bends a
    written decision rule toward its own preference (rules 9 and 12).

## Source of truth for work

- Active milestones and tasks live in `.agents/TODO.md` (authoritative active worklist).
- Completed milestones live in `.agents/ARCHIVE.md`; archived numbers never change. Active
  milestones in `.agents/TODO.md` are numbered in execution order (renumbered once on 2026-09-10;
  the old-to-new map is at the top of `ARCHIVE.md`), and new milestones take the next number after
  the highest one in either file. When priorities change, move a section instead of renumbering.
- The cross-repo feature review and prioritized shortlist live in `.agents/feature_crosswalk.md`.
- The handoff prompt for the next session lives in `.agents/next_agent_session_prompt.md`.
- `CHANGELOG.md` is the authoritative release history.
- Before starting any task: read `.agents/TODO.md` and work only on the highest-priority blocking
  items.
- When a task is completed: move it from `.agents/TODO.md` to `.agents/ARCHIVE.md` with a short
  completion note.

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
- Keep `.agents/TODO.md` accurate: verify items before checking them off.
- Keep `.agents/TODO.md`, `.agents/ARCHIVE.md`, `.agents/feature_crosswalk.md`, and
  `CHANGELOG.md` synchronized with the actual repo state after meaningful progress, completed
  tasks, or validation changes.
- Keep `README.md` current: update it when behavior, CLI usage, features, or outputs change.
- Keep project documentation current after significant changes. When a subsystem outgrows the
  top-level `README.md`, add or update nested module README files and link them from the top-level
  README.

## Changelog and Commit Workflow

- `CHANGELOG.md` follows Common Changelog: latest version first, `## [VERSION] - YYYY-MM-DD`,
  then `Changed`, `Added`, `Removed`, and `Fixed` in that order.
- **Update the changelog as you go**, not at the end of a session: add an entry each time a
  milestone, a task, or a sizable chunk of one lands (a fix, a feature group, a changed default).
- **Every entry gets its own incremented version. Never write `[Unreleased]`.** Bump the patch
  (`0.6.1` to `0.6.2`) for fixes and small additions, and the minor (`0.6.x` to `0.7.0`) for a new
  feature family, a changed default, or a schema change. Date the entry with the day the change
  lands.
- **Set `pyproject.toml` to the newest changelog version in the same change**, then run `uv lock`
  and `uv sync`, or `uv lock --check` and `uv sync --check --active` fail on the stale version.
- The project is private and not ready for releases: **never create or push a git tag or a GitHub
  release.** Versions live only in `CHANGELOG.md`, `pyproject.toml`, and `uv.lock`;
  `.github/workflows/release.yml` stays idle because no tag is ever pushed.
- The historical baseline is `0.1.0` on `main`.
- Keep changelog entries focused on notable user-facing, tooling, or workflow changes; skip routine
  formatting-only noise.
- When committing work, prefer one logical change per commit for multi-file schema work and one file
  per commit for all other changes, including deletions, unless the user explicitly asks for
  different commit granularity.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): a subject
  of the form `type(scope): imperative summary`, at most 72 characters, followed by a blank line
  and a body that explains what changed and why. Types: `feat`, `fix`, `docs`, `test`, `refactor`,
  `perf`, `build`, `ci`, `chore`. The scope is optional and names the area (`etl`, `ml`,
  `rankings`, `reporting`, `agents`, `walk-forward`). Mark breaking changes with `!` after the
  type or scope. Examples: `feat(etl): blend early-season stats toward the regressed prior`,
  `docs(agents): regenerate the handoff prompt`, `fix(reporting): balance the workbook formulas`.

## CI Direction

- GitHub Actions stays validation-only: `.github/workflows/validation.yml` runs Ruff, Pyright, Ty,
  pytest, markdownlint, the `uv` lock and sync checks and the CLI help smoke checks.
  `.github/workflows/release.yml` fires only on a pushed tag, and no tag is ever pushed.

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
- the project's own commands:
  - `.venv/bin/nfl-predictor <command>` (the same as `.venv/bin/python -m nfl_predictor <command>`)

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

`scripts/gate.sh` runs every check the way CI does and is the only form that counts as "the
gate" (add `--web` when `web/` changed, `--quick` to skip pytest while iterating); the individual
`.venv/bin/...` tools are for iteration.

## Project Shape (Big Picture)

- Polars ETL over `nflreadpy` NFLverse sources (schedule, team stats, play-by-play), Elo/QB
  ratings, TeamRankings stats and market odds produces the ML-ready datasets.
  `nfl_predictor/data_collection.py` (`nfl-predictor data`) orchestrates it; transforms live under
  `nfl_predictor/utils/polars/`, game enrichments in `nfl_predictor/utils/game_utils.py`, web
  scraping and caching in `nfl_predictor/utils/scraping_utils.py`.
- **Front door:** `nfl-predictor <command>` (`nfl_predictor/cli/main.py`, a `[project.scripts]`
  entry, also `python -m nfl_predictor`) runs every task; see "Commands" below. The weekly run
  lives in `nfl_predictor/weekly_run/`, the other command-line code in `nfl_predictor/cli/`,
  shared option builders in `nfl_predictor/cli/options.py`.
- Compatibility facades: `nfl_predictor/utils/polars_utils.py` forwards imports to the split
  Polars modules, and `nfl_predictor/ml_model.py` forwards to `nfl_predictor/ml/` and keeps the
  `python -m nfl_predictor.ml_model` form of `nfl-predictor train`/`predict`
  (`nfl_predictor/cli/train.py`, outside the checkpoint fingerprint). XGBoost version/build
  compatibility helpers live in `nfl_predictor/ml/ml_model_xgb_utils.py`.

### Commands (operational entrypoints)

`nfl-predictor --help` lists every command by group; `nfl-predictor <command> --help` shows its
options. `scripts/` holds only `gate.sh`.

- `nfl-predictor weekly` (`nfl_predictor/weekly_run/`): canonical weekly orchestration (data
  refresh -> compare -> tune/train -> predict -> reports). Reads `config/weekly_run.yaml` unless
  `--config` names another file.
- `nfl-predictor backtest` (`nfl_predictor/cli/backtest.py`): walk-forward evaluation, the
  benchmark.
- `nfl-predictor sweep` (`nfl_predictor/cli/sweep.py`): sweep calibration + market-prob variants
  and summarize metrics.
- `nfl-predictor compare` (`nfl_predictor/cli/compare.py`, definitions in
  `nfl_predictor/reporting/run_comparison.py`): paired comparison of walk-forward runs rescored
  from their fold checkpoints, per window, with bootstrap intervals and two seeds combined per
  game when given. It reproduces the task 55.8 independent rescore exactly
  (`.agents/m60/verify_compare.py`), so a reviewer can use it under rule 3(b) as long as the
  review also checks provenance and the reviewer did not produce the run.
- `nfl-predictor rankings` (`nfl_predictor/cli/rankings.py`): power rankings + projected
  standings. The default `--method composite` ranks on the ETL's schedule-adjusted composite from
  `data/strength_snapshots.csv`; `README.md` covers `--method bradley_terry` and
  `--legacy-franchise-fit`. `nfl-predictor weekly` calls the same `compute_power_rankings`
  (`nfl_predictor/reporting/power_rankings.py`).
- `nfl-predictor train` / `predict`, `data`, `validate` (`--live`), `leakage-audit`, `lines`,
  `build-week`, `explain`, `checkpoints`, `web`, `users`: see `README.md`, "Command line".
- Training/prediction entrypoints may be updated/replaced, but must remain runnable and
  documented.

### Web UI (FastAPI + React)

- The backend is `nfl_predictor/api/` and the app is `web/`. Their layouts, the web gate and the
  test locations are in `nfl_predictor/api/AGENTS.md` and `web/AGENTS.md`, which load when you
  work in those directories. Run `scripts/gate.sh --web` whenever either changes.
- `nfl-predictor web` serves the built app from `web/dist/` on port 8000. `--reload` restarts the
  server on any Python file change in the checkout (so an agent's edits, checkouts and merges
  restart it), which marks running jobs failed, and its watcher takes about half a core: never
  launch a web job from a `--reload` server, and say so before editing code while one runs.
  `web/README.md`, "Which mode to use", covers the Vite dev server on port 5173.

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
  - `data/strength_snapshots.csv` - pre-week adjusted strength per `(season, week, team)` for
    every scheduled team, bye teams included; the source of the default power rankings

### I/O rules

- Prefer the project’s Polars-based load/save helpers in `nfl_predictor/data_collection.py`.
- Any pandas-based CSV I/O utilities are legacy. Do not add new pandas-based I/O helpers; prefer the
  Polars ETL helpers when touching related code.

## Neighboring Repos and Cross-Repo Contracts

- `../nfeloqb` produces `qb_elos.csv` (538-style schema; `team1` is the home team).
  `nfl-predictor` consumes a manually copied `data/qb_elos.csv`. Treat that file as a downstream
  contract: do not propose schema changes there casually, and never edit `../nfeloqb` outputs from
  this repo. `../nfeloqb/Other Data/meta_data.csv` maps Elo QB names to GSIS ids and is the
  intended identity bridge for QB-level PBP features.
- `../nfl-sos-ratings` is the reference implementation for the strength-of-schedule method this
  repo is porting: for each subject (team or QB) and each opponent it faced, build that opponent's
  statistical profile from only its games against the rest of the league, excluding every
  head-to-head game with the subject, so subject and opponent profiles are independent for every
  matchup; then compare the subject to that adjusted schedule. Its simultaneous ridge solve
  (`simultaneous_adjustment.py`) is the all-hops generalization of that one-hop method and is its
  published backbone; the one-hop profiles remain for descriptive views. It is also the reference
  for PBP-derived per-snap EPA, success and explosive rates, and special-teams EPA. Read its
  `README.md`, `AGENTS.md`, and `docs/` (`methodology.md`, `validation-report.md`, both stats
  catalogs) before designing any adjusted feature, and note that its own walk-forward puts the
  within-season ridge at parity with SRS and raw EPA and behind prior-carrying Elo. Port ideas and
  formulas into this repo's Polars ETL; do not import it as a dependency and do not modify it from
  here. The repo-root symlink `nfl-sos-ratings -> ../nfl-sos-ratings/` is gitignored and exists
  only for convenient reading; always state which repo you are inspecting.
- Leave untracked local files in neighboring repos alone (for example `../nfeloqb/.bash_history`).
- Play-by-play comes from `nflreadpy.load_pbp`. nflreadpy caches only in memory, so cache selected
  columns per season as Parquet under `data/cache/nflreadpy/` with the same current-season refresh
  and non-fatal failure behavior as schedules and team stats. Filter to the regular season for
  feature inputs and normalize `posteam`, `defteam`, `home_team`, and `away_team`.

## Column & Schema Rules (Source of Truth)

- Column names and schema lists are defined in `nfl_predictor/constants.py`.
- Team identifiers are normalized using the canonical mapping in `constants.py`.
- Do not hard-code column lists; use constants to prevent schema drift.
- ETL does not silently drop columns.
- Always normalize team identifiers via `constants.ALIAS_TO_CANONICAL` and (when available)
  `normalize_team_column(df, col)`.

## Season/Week Logic & Edge Cases

- Use `constants.get_regular_season_weeks(season)` to determine regular-season length.
- Week 1 / early-season rows with missing history use the previous regular season regressed by
  `constants.WEEK1_REGRESSION_FACTOR`. From week 2 on, season-to-date stats blend toward that same
  prior with in-season weight `games / (games + constants.PRIOR_BLEND_GAMES)`, and rates are
  recomputed from the blended sums (`polars_utils.blend_with_prior_stats`). The adjusted-strength
  family blends its own previous-season snapshot the same way. New season-to-date families inherit
  the stat blend automatically if they flow through `team_stats_df`.
- Future games have missing outcomes; the pipeline still outputs a structurally complete row
  suitable for prediction.
- Postseason rows may exist. Training/evaluation defaults should be explicit about whether
  postseason is included and (if included) how it is weighted.

## Prediction, evaluation and feature rules

The full specification (targets, calibration methods, market anchoring and blend/clamp bounds,
uncertainty outputs, realistic scores, confidence-pool scoring, evaluation modes and metrics,
the model-selection protocol, the leakage audit, the model artifact contract and the feature
development rules) is in `.agents/modeling_spec.md`. Read it before changing prediction,
calibration, market, pool, evaluation, artifact, leakage-audit or feature code. The rules that
must never be missed:

- The model predicts `margin = home_score - away_score` and `total = home_score + away_score`,
  and scores derive from them. Win probability derives from the margin prediction; the pick and
  its confidence come from the calibrated probability.
- Confidence pools: unique `1..N` per week; a tie scores as incorrect for both sides; picks are
  single-shot before the week's first game (no in-week updates in backtests).
- Realistic score adjustments are display-only: they never alter win probabilities, confidence
  rankings, pool scoring, tuning objectives or training targets.
- Market blending and clamping are explicit and bounded (weights in `[0, 1]`, clamp delta in
  `[0, 0.5]`); edges against the market are diagnostics only, never a profitability claim.
- Walk-forward is the authoritative evaluation; never use the holdout window to tune.
- Every saved model carries adjacent metadata JSON (the artifact contract) and loads without
  hidden external state; `nfl-predictor leakage-audit` stays maintained.
- Features apply to every matchup. Carry counts and sums through season-to-date aggregation and
  compute rates afterward. Before adding a stat to `EXCLUDE_FROM_OPPONENT_STATS`, grep
  `_compute_derived_metrics` for its `opponent_` mirror: a derived metric that reads it needs the
  stat in `OPPONENT_MIRROR_INTERMEDIATES` too, or the derived columns go null at the next
  rebuild. A schedule- or opponent-adjusted value for week `N` is solved from games strictly
  before week `N`. A change to feature values cannot be ablated with `--disable-feature-groups`;
  it needs two dataset builds (the procedure is in the spec).

## ML Implementation Standards

The preprocessing, training and blending standards for the model code are in
`nfl_predictor/ml/AGENTS.md`, which loads when you work under `nfl_predictor/ml/`.

## Dependency & Environment Hygiene

- Key ML dependencies (xgboost, scikit-learn, numpy, pandas, polars, scipy, optuna) stay pinned;
  `pyproject.toml` declares them and `uv.lock` is the lockfile source of truth (`uv lock` /
  `uv sync`, or the `update_requirements.sh` wrapper).
- Document supported Python versions and CPU/GPU constraints; avoid optional GPU paths that break
  CPU-only execution unless explicitly guarded.

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

## Logging & Coding Style

- Logging uses the project logger (`from nfl_predictor.utils.logger import log`). No `print`.
- Prefer Polars expressions over Python loops in ETL.

## Safety, Scope, and Prohibited Behaviors

- Do not add unrelated features (new scrapers, unrelated pipelines). The web UI under
  `nfl_predictor/api/` and `web/` is in scope; extend it by phase per `.agents/web_ui_plan.md`.
- Do not remove/alter existing pipeline behavior without updating tests and documentation.
- Avoid new external services or network dependencies beyond existing scraping utilities.
- Do not claim betting profitability; report metrics and uncertainty honestly.

## Assistant guidance

- Use existing project utilities and constants.
- Implement changes in small, testable increments.
- Keep outputs deterministic under fixed seeds.
- Do not change behavior without updating tests and documentation.
- Do not delete `models/<run_id>/wf_compare/` during active walk-forward runs; those artifacts power
  resume behavior.
- Keep walk-forward comparison artifacts under `models/` (do not point `--out-json` at a temporary
  directory). Any number reported in `.agents/` or `AGENTS.md` must be auditable from disk.
- Walk-forward runs checkpoint every finished week (`models/wf_checkpoints/<fingerprint>/` by
  default; `wf_compare/wf_folds/` inside `weekly_run` runs). After a stop, re-run the
  identical command and it resumes at the next unfinished week; `--no-resume` retrains
  everything. Watch progress in the run's log: every finished week prints a
  `Walk-forward fold N/M done` line with elapsed and remaining time. For `weekly_run` comparisons,
  `wf_compare/wf_summary.csv` still shows per-candidate results.
- Run **one walk-forward at a time**: XGBoost uses every core, and two concurrent runs each cost
  more CPU than a solo run without finishing. Check `uptime` and
  `ps -eo pcpu,args --sort=-pcpu | head -4` before a launch; run timings and load measurements
  are in `.agents/walk_forward_runbook.md`.
- Launch every walk-forward through a small `launch.sh` in its own run directory (it `cd`s to the
  repo root, runs `.venv/bin/nfl-predictor <command>` with the run's options, and echoes the exit
  code) with `nohup setsid`, never through a harness-bound shell, which stops at 10 minutes. Never
  `pkill -f` a pattern that can match your own shell. Several accepted rungs run back to back
  through one driver script. Two lessons from 2026-09-23:
  - Any logic added to a `launch.sh` (a load probe, for example) is tested under the script's own
    strict-mode header, because `IFS=$'\n\t'` changes how `read` splits. A probe written without
    it failed and stopped the whole queue overnight.
  - Never execute a fragment cut from a `launch.sh`: a cut that includes the walk-forward line
    (`nfl-predictor backtest`) starts a second walk-forward
    with default settings.
  Check the driver's log after its first run-to-run transition, not only at the end.
- Since `0.24.0` the walk-forward commands (`nfl-predictor weekly`, `backtest` and `sweep`) set
  `OMP_WAIT_POLICY=PASSIVE` unless it is already set; on a machine known to stay idle, launch
  with `OMP_WAIT_POLICY=` (empty) to keep the library default. The setting changes scheduling
  only, so it neither alters results nor invalidates fold checkpoints; switching mid-run means
  stop, relaunch with the other policy, and resume. The measurements behind it are in
  `.agents/walk_forward_runbook.md`.
