# Web UI Session Prompt

You are the orchestrating agent (Claude Opus) for an implementation session in the
`nfl-predictor-web` workspace (`/home/mitch/workspace/nfl-predictor-web`). Your deliverable is
**Phase 2 of the web UI plan** in `.agents/web_ui_plan.md`: the subprocess job runner with
streamed logs and the lines-only ETL refresh. If it lands with time to spare, continue into
**Phase 3** (future-week predictions).

Say in one sentence at the top of your first response that you are working in the worktree and
will not touch the sibling checkout, then start. Do not ask permission to begin.

## 0. What this workspace is

- This folder is a **git worktree** of `/home/mitch/workspace/nfl-predictor` on branch
  `feat/web-ui`, branched from commit `91aaffc` (the committed tip of `feat/pbp-per-snap-epa`,
  version `0.4.0`). It is temporary: a second Claude session is working on Milestone 49 in the
  main checkout at the same time. **Never edit, run, or `cd` into `/home/mitch/workspace/nfl-predictor`**
  beyond reading its `data/` and `models/` trees, and never run `pkill` on patterns that could
  match that session's processes.
- The branch is pushed to `origin/feat/web-ui`. Commit only when the user asks. When asked, use
  logical commits with Conventional Commits subjects (`type(scope): imperative summary`), a body
  explaining what and why, and the attribution line the harness provides. Do not rebase onto
  `feat/pbp-per-snap-epa` or `main`; that happens after the other session's work lands.
- `.agents/TODO.md`, `AGENTS.md`, `README.md`, and `CHANGELOG.md` belong to the other session.
  Read them; do not edit them here. The web UI plan and its status live in
  `.agents/web_ui_plan.md`; keep its **Status** section current. A Milestone 51 entry in
  `TODO.md` and an `AGENTS.md` amendment (dropping the "no dashboards" rule, documenting the `web`
  extra and Node commands) are deferred until the worktree merges.

## 1. Read first, in this order

1. `AGENTS.md`: non-negotiables (venv-only commands, TDD, docstrings, coverage floor, Polars-first,
   no milestone numbers in code, no `print`), the readiness behaviors, and the note that
   concurrent walk-forward runs are forbidden.
2. `.agents/web_ui_plan.md`: the full plan, the decisions already made, and the Phase 2 section.
3. `nfl_predictor/api/__init__.py` (app factory), `settings.py`, `db.py` (the `jobs` and
   `job_logs` tables already exist), `deps.py`, `runs/indexer.py`, `runs/active.py`,
   `routers/resolve.py` (how a request resolves to a predictions file).
4. `tests/api/conftest.py` and `tests/api/factories.py`: the fixture root and the run-directory
   builder every API test uses.
5. `web/src/api/{client,queries,types}.ts`, `web/src/app/nav.ts`, `web/src/pages/Runs.tsx`
   (the pattern every page follows), `web/src/components/table/DataTable.tsx`.
6. `nfl_predictor/utils/logger.py` (single stderr handler with ANSI color),
   `nfl_predictor/ml/walk_forward.py` around the `Walk-forward fold %d/%d done` log line,
   `nfl_predictor/utils/polars/loaders.py::load_schedule`, and
   `nfl_predictor/utils/game_utils.py::fill_missing_moneylines` (inputs to the lines refresh).
7. `scripts/weekly_run.py` `_allowed_config_keys` / `_validate_config_keys` / `--config`
   (the cleanest way to launch a weekly run from a JSON file) and `--dry-run`.

## 2. Facts to trust unless your verification disproves them

- Environment: this worktree has its own `.venv` synced with `uv sync --extra web`. The inherited
  shell may carry `VIRTUAL_ENV` pointing at the **main** checkout; always use `.venv/bin/...`
  paths and export `VIRTUAL_ENV=$PWD/.venv` before any `uv sync --active` command, or you will
  modify the other session's environment. Node 26 is nvm-managed: `source ~/.nvm/nvm.sh` first.
- Baseline on this branch (2026-09-10): `.venv/bin/python -m pytest` gives `631 passed`, coverage
  `92.16%` against the `90%` floor; ruff, pyright, ty, markdownlint
  (`markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings" "#web/node_modules"`), `uv lock --check`,
  and `uv sync --check --active --extra web` are clean. Frontend: `npm run lint`, `npm run
  typecheck`, `npx vitest run` (11 tests), `npm run build` pass; lint warnings only in
  `web/src/components/ui/*` (shadcn-generated) are acceptable.
- `data/` and `models/` are gitignored, so this worktree has none. Run the API against the main
  checkout's trees read-only:

  ```bash
  NFLP_MODELS_DIR=/home/mitch/workspace/nfl-predictor/models \
  NFLP_DATA_DIR=/home/mitch/workspace/nfl-predictor/data \
  NFLP_REPORTS_DIR=/home/mitch/workspace/nfl-predictor/reports \
  NFLP_STATE_DIR=$PWD/data/web NFLP_PORT=8765 \
  .venv/bin/python -m nfl_predictor.api
  ```

  A smoke admin exists in `data/web/app.db` (`mitch` / `smoke-test-pass`); `python -m
  nfl_predictor.api.auth.cli` manages users. Stop a server you started by PID (find it with
  `pgrep -f "^[^ ]*python -m nfl_predictor\.api$"`), never with a broad `pkill -f`.
- **Jobs that write must not run against the main checkout's `data/` or `models/` during this
  session**: the other session is rebuilding datasets and running walk-forwards there. Test the
  runner with `tests/api/fake_script.py` and, for a live check, with read-only templates
  (`validate_offline`, `leakage_audit --out-json` pointed at `NFLP_STATE_DIR`) or `--dry-run`.
- The newest complete weekly run on disk is `weekly_2025_week_22` (a single Super Bowl game), so
  it is the default active run. Week 1 2026 predictions exist only as an unattached
  `data/predict/week_01_predictions.csv`; the week selector already exposes it.
- The betting table is **derived from the predictions file** in `readers/betting.py` (workbook
  formulas), not read from `*_betting_report.csv`. Total columns carry `actionable=False`
  (Milestone 50) and a test enforces that.
- Scripts are subprocess-only: their `main()` functions read `sys.argv` and mutate global logging.
  Only `data_collection.main(argv)`, `validate_offline`, `validate_live`, and
  `betting_report_excel` are cleanly callable, and you should still launch them as subprocesses
  for isolation.
- Screenshots: `google-chrome --headless=new` is installed, and `NPM_CONFIG_USERCONFIG=/dev/null
  npx --yes agent-browser ...` works for logged-in captures (the user-level `~/.npmrc`
  `allow-scripts` line breaks npm 12 inside project installs; the same env override is needed
  for `npx shadcn@latest add`, after which fix generated `from "cn"` imports to `@/utils/cn`).

## 3. Design decisions already made (do not relitigate; record deviations in the plan)

- **Runner**: `subprocess.Popen` of `settings.python_path` with `cwd=root_dir`,
  `PYTHONUNBUFFERED=1`, `stderr=STDOUT`, `start_new_session=True`; a reader thread strips ANSI
  (`\x1b\[[0-9;]*m`), parses `[ts][LEVEL][file:func:line] msg`, batches rows into `job_logs`, and
  updates `progress` from the walk-forward fold regex. One worker for `exclusive_group=
  "walk_forward"` (weekly_run, walk_forward_backtest, golden_command), a two-slot pool for the
  rest. On startup mark `running` rows `failed (server restarted)`. Cancel = SIGTERM to the
  process group, SIGKILL after 10 s. SSE (`sse-starlette` is installed) polls the store every
  300 ms, replays from `?after=`, ends on a terminal status.
- **Templates** (`jobs/catalog.py`, each with a typed params schema the frontend renders as a
  form): `etl_full`, `lines_refresh` (chains a `predict` on the active model), `weekly_run`
  (write params JSON to `data/web/job_configs/<job_id>.json`, run `scripts/weekly_run.py
  --config`), `train`, `predict`, `power_rankings`, `betting_xlsx`, `leakage_audit`,
  `validate_offline`, `validate_live`, `walk_forward_backtest`, `shap_analysis`.
- **Lines refresh**: new module `nfl_predictor/lines_refresh.py` with `refresh_lines(season, week,
  *, data_dir, cache_dir)` and `main(argv)`. Call `load_schedule([season], force_refresh=True,
  current_season=season)`, keep `game_id` plus the five line columns, apply
  `fill_missing_moneylines`, then update `data/predict/week_WW_games_to_predict.csv`,
  `data/all_data_ml.csv`, and `data/all_data.csv` by `game_id` only (that season's rows), writing
  atomically and preserving column order. Leave `completed_games*` alone. Return and log
  per-file changed-row counts. Do not edit `data_collection.py`.
- **API shape**: `GET /api/jobs/catalog`, `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs/{id}`,
  `GET /api/jobs/{id}/logs?after=`, `GET /api/jobs/{id}/stream`, `POST /api/jobs/{id}/cancel`;
  409 when an exclusive group is busy, 422 on bad params, admin-only for POST and cancel.
  Register routers in `nfl_predictor/api/__init__.py`; invalidate `app.state.run_index` when a
  job finishes.
- **Frontend**: `/jobs` (catalog cards grouped by category, auto-generated form, history table)
  and `/jobs/:id` (virtualized log console, level filter, progress bar, cancel). Add admin
  buttons on the Data page (Run ETL, Refresh lines) and the Betting page (Generate workbook).
  Set `phase` on the Jobs nav item to undefined once it ships. Follow the existing patterns:
  hooks in `api/queries.ts`, types in `api/types.ts`, pages in `pages/`, shared pieces in
  `components/common/`.
- **Phase 3 (only if time remains)**: template `predict_week {season, week}` that builds the
  `week_WW_games_to_predict.csv` from `all_data_ml.csv` when missing (mirror the availability
  check in `scripts/power_rankings.py::_predict_future_games`) and predicts into the active run;
  the week selector already lists run-attached and unattached weeks.

## 4. Non-negotiables

- TDD: tests first under `tests/api/`, using the `project_root` / `settings` / `admin_client`
  fixtures and `factories.make_run_dir`. Keep `nfl_predictor/api/**` near 95% coverage so the
  repo-wide floor holds. Frontend components that carry logic get a vitest test.
- Docstrings and type hints on everything; no milestone numbers in code or tests; no new
  `noqa` / `type: ignore` / `pragma: no cover` without a real reason (tests already ignore
  `S101`, `S105`, `S106`; the registry ignores `E501`).
- All Python tooling via `.venv/bin/...`; `uv` from PATH; never bare `python` / `pytest`.
- Do not modify `../nfeloqb`, `../nfl-sos-ratings`, or the main checkout.
- Full gate before reporting: ruff format/check, pyright, ty, pytest, markdownlint, `uv lock
  --check`, `uv sync --check --active --extra web`, and the four `web/` commands.

## 5. Final report to the user (structure)

1. Outcome first: what landed, gate status, and which templates were exercised live.
2. Screenshots or a description of the Jobs pages on desktop and at 400 px width.
3. What was left out or deferred, and why.
4. The exact commands to run a lines refresh from the UI once the other session is idle.
5. Your recommendation for the next session (Phase 3 if it did not fit, then Phase 4).
