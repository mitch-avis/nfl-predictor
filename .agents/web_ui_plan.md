# Web UI for nfl-predictor: FastAPI backend + React frontend

Planned 2026-09-10. This is the implementation plan for the web UI milestone (Milestone 51 once
`.agents/TODO.md` is updated). Status of each phase is tracked at the bottom.

## Context

Today every output of this project (weekly predictions, confidence picks, betting report, power
rankings, model metrics, ETL state) lives as CSV/JSON/xlsx files scattered across `data/predict/`,
`reports/`, and `models/<run_id>/`. The user wants one browsable, mobile-friendly web app that
displays all of it cleanly and, over time, controls the project (run ETL, refresh lines, train,
predict, run scripts) behind an auth layer so the link can be shared safely. Later phases fold in
the `nfl-sos-ratings` team/QB reports, pool helpers (confidence, tiebreakers, survivor), and
future-week predictions.

Exploration findings that shape the design:

- `../nfl-sos-ratings` already has a working React 19 + Vite + TanStack Table frontend
  (`ui/web/`) over a FastAPI backend (`nfl_sos_ratings/ui_api.py`, `ui_data.py`) that reads
  Parquet with Polars and ships a `TablePayload = {rows, visible_columns, column_groups,
  column_metadata}` contract with a metric registry driving labels/tooltips/heatmaps. We port
  that contract and the table ideas (sticky identity columns, heatmap cells, compare workflow,
  glossary) onto shadcn primitives rather than forking its 923-line `App.tsx`.
- There is no "latest model" concept. `nfl_predictor/ml/artifacts.py` only builds paths
  (`resolve_run_paths` L117). `scripts/weekly_run.py` writes a self-contained run dir
  (`models/weekly_2025_week_22/` is the canonical example) with stage markers
  `{wf_compare,train,predictions,reports}_state.json`; a run is complete iff
  `reports_state.json` exists. Power rankings in a run are stamped `week = predictions_week - 1`.
- Scripts are subprocess-only: every `main()` except `data_collection.main(argv)` reads
  `sys.argv`, and the logger is a single stderr `StreamHandler` with `coloredlogs` ANSI
  (`nfl_predictor/utils/logger.py`). Walk-forward logs `Walk-forward fold N/M done ... about Ns
  remaining` (`nfl_predictor/ml/walk_forward.py` ~L911) and AGENTS.md forbids concurrent WF runs.
- Market lines come from the nflverse schedule via `polars_utils.load_schedule(...,
  force_refresh=True)` (`nfl_predictor/utils/polars/loaders.py` L288); `_prepare_schedule` derives
  `away_spread`/`home_spread`, and `game_utils.fill_missing_moneylines` (L61) fills gaps.
  `data/nfl_lines.csv` is unused legacy. Market features feed the model
  (`nfl_predictor/ml/feature_spec.py` L88-93), so a lines refresh must be followed by a predict job.
- `.agents/TODO.md` Milestone 50: the total/over-under head has no signal. `total_*` betting
  columns must never be presented as actionable.
- Another session is editing `data_collection.py`, `constants.py`, `walk_forward.py`,
  `strength_snapshot.py`, `teamrankings.py`, `weekly_run.py`, `betting_pipeline.py`,
  `golden_command.py`, `.agents/TODO.md`, `AGENTS.md`, `README.md`, `CHANGELOG.md`. This plan
  touches none of those files except tiny additive edits to `pyproject.toml`, `.gitignore`, and
  CI, and it lives on its own branch/worktree.
- Repo constraints: Python >= 3.14, `.venv/bin/` prefixes, ruff line-length 100, pyright + ty,
  pytest coverage floor 90% over `nfl_predictor` (baseline 90.83%, so new `api/` code needs ~95%),
  docstrings everywhere, TDD, no `print`. Coverage omits any module named `config.py` (use
  `settings.py`). `.gitignore` ignores `lib/`, `dist/`, `*.db`; no Node entries; CI has no Node
  step. Node v26.8.1 via nvm, npm only.
- `../survivor-ratings` is a CBS *Survivor* TV-show scraper, unrelated. Nothing to absorb.

## Decisions (made with the user)

1. Code lives in this repo: `nfl_predictor/api/` (FastAPI) + `web/` (React). Work on branch
   `feat/web-ui` in a separate worktree `../nfl-predictor-web` created from `main`.
2. Frontend: Vite + React 19 + TypeScript + Tailwind v4 + shadcn/ui + TanStack Table + TanStack
   Query + react-router 7 + recharts.
3. Hosting: uvicorn on this WSL box serving API + built SPA over LAN/Tailscale. Auth:
   username/password, argon2 hashes, signed JWT in an httpOnly cookie, roles `viewer` / `admin`.
   Admin-only for jobs, run activation, and user management. Bootstrap admin via CLI.
4. Canonical data: index `models/*/metadata.json` (sorted by `created_at`), admin marks one run
   **active**; predictions/picks/betting/power all read from that run dir. `data/predict/` and
   `reports/` ad-hoc files are surfaced as "unattached".
5. Market lines: a new lines-only ETL mode in a **new module** `nfl_predictor/lines_refresh.py`,
   exposed as an on-demand job that chains a predict job on the active model.
6. Jobs: subprocess runner over existing CLIs, SQLite job table, per-job persisted logs streamed
   over SSE, walk-forward jobs serialized.
7. Plan document goes to `.agents/web_ui_plan.md` on the branch. The Milestone 51 entry in
   `.agents/TODO.md` and the `AGENTS.md` amendment (drop the "no dashboards" rule, add the `web`
   extra and Node commands) land in a small follow-up once the other session finishes.
8. Later phases scoped now: future-week predictions, survivor optimizer, tiebreakers, team/QB
   pages over `nfl-sos-ratings` Parquet.

## Layout

### Backend `nfl_predictor/api/`

```text
api/
  __init__.py   create_app(settings) factory (mirrors nfl-sos-ratings ui_api.create_app)
  __main__.py   python -m nfl_predictor.api  -> uvicorn (bootstrap under __main__ guard)
  settings.py   pydantic-settings: NFLP_ROOT_DIR, NFLP_DB_PATH (default data/web/app.db),
                NFLP_JWT_SECRET (auto-generated to data/web/secret.key, 0600), NFLP_COOKIE_SECURE,
                NFLP_SOS_DATA_DIR, NFLP_WEB_DIST, NFLP_PYTHON (default ROOT/.venv/bin/python)
  deps.py       get_settings, get_db, current_user, require_admin
  errors.py     ApiError -> {"error": {"code", "message"}}
  db.py         sqlite3 schema + migrations: users, jobs, job_logs, kv
  auth/         passwords.py (argon2-cffi), tokens.py (PyJWT HS256), router.py, cli.py
  registry/     columns.py (ColumnMeta, REGISTRY, project()), predictions.py, betting.py,
                power.py, model.py
  runs/         indexer.py (scan + kind/stage/season/week detection + TTL cache),
                active.py (kv-backed active run + fallback), files.py (RunFiles extends
                artifacts.resolve_run_paths with every artifact name)
  readers/      cache.py (mtime-keyed bounded cache), predictions.py, picks.py, betting.py,
                power.py, model.py, data_status.py, unattached.py, market.py (novig helpers)
  jobs/         catalog.py (JobTemplate: id, label, description, params schema, build_cmd,
                exclusive_group, chain_after), runner.py, store.py, stream.py, router.py
  routers/      runs.py predictions.py betting.py power.py model.py data.py registry.py
                users.py static.py (SPA mount + fallback, 503 page when web/dist missing)
  schemas/      common.py (TablePayload, ColumnMeta), runs.py, predictions.py, betting.py,
                power.py, model.py, data.py, jobs.py, auth.py
```

`ColumnMeta`: `key, label, description, group, kind (text|int|float|pct|prob|money|spread|
datetime|team|action), polarity (higher|lower|neutral), heatmap, decimals, actionable, sticky`.
`project(df, keys)` selects registered keys present in the frame, converts NaN to null and dates
to ISO, and attaches metadata. `GET /api/registry` returns the whole registry once.

### Frontend `web/`

```text
web/  package.json  vite.config.ts (proxy /api -> 127.0.0.1:8000)  components.json  eslint  vitest
  src/
    main.tsx  App.tsx  router.tsx
    api/       client.ts (credentials: include, 401 -> /login)  schema.d.ts (generated)
               queries.ts (TanStack Query hooks)  sse.ts
    app/       AppShell.tsx  Sidebar.tsx (collapsible="icon" desktop, Sheet on mobile, Ctrl+B)
               ThemeProvider.tsx  RunContext.tsx
    pages/     Dashboard Predictions PowerRankings Betting DataStatus Model Jobs JobDetail
               Runs Users Login Glossary  (Phase 4: Pool; Phase 5: Teams, TeamDetail, QBs, QBDetail)
    components/table/{DataTable,HeatCell,ColumnPicker,ColumnHeader}.tsx
               predictions/{MatchupCard,WinProbBar,QuantileBand}.tsx
               betting/ActionBadge.tsx  power/{RatingBar,MovementChip}.tsx
               model/{FeatureImportanceChart,CalibrationChart}.tsx
               jobs/{JobForm,LogConsole,JobStatusBadge}.tsx
               common/{WeekSelector,RunSelector,InfoTooltip,StatTile}.tsx  ui/ (shadcn)
    hooks/     useQueryParam useMediaQuery useColumnMeta
    registry/  columnMeta.ts (hydrates /api/registry; port of sos metricMetadata.ts)
    utils/     cn.ts format.ts   (NOT src/lib: gitignored)
  tests/       vitest + testing-library
```

### Config edits (small, additive; outside the other session's dirty set)

- `pyproject.toml`: `[project.optional-dependencies] web = [fastapi, uvicorn[standard],
  pydantic-settings, pyjwt, argon2-cffi, sse-starlette, python-multipart]`; add `httpx` to dev
  group for TestClient; add `"web"` to ruff `extend-exclude` and pyright `exclude`.
- `.gitignore`: `web/node_modules/`, `web/dist/`, `web/coverage/`, `web/*.tsbuildinfo`,
  `data/web/`.
- `.github/workflows/validation.yml`: `uv sync --extra web` in the Python job; new `web` job
  (`actions/setup-node@v6`, node 26, npm cache on `web/package-lock.json`, `npm ci`, `npm run
  lint`, `npm run typecheck`, `npm test -- --run`, `npm run build`).
- `web/README.md` documents `source ~/.nvm/nvm.sh`, dev/build commands, and the backend command.

## API surface

All under `/api`; cookie `nflp_session` (httpOnly, SameSite=Lax, Secure when configured);
mutating routes require header `X-Requested-With: nflp`.

| Route | Notes |
| --- | --- |
| `POST /auth/login`, `POST /auth/logout`, `GET /auth/me` | login rate-limited in memory |
| `GET/POST /users`, `PATCH/DELETE /users/{id}` | admin; cannot delete self or last admin |
| `GET /registry` | full ColumnMeta registry |
| `GET /runs?kind=`, `GET /runs/{id}`, `POST /runs/{id}/activate`, `GET /runs/{id}/files/{name}` | RunSummary: run_id, created_at, kind (weekly/training/walk_forward), season, week, stages, complete, git_commit_hash, dataset_hash, model_kind, holdout metrics, file presence. Files endpoint is allow-listed. |
| `GET /predictions?run=&week=` | ~40 projected columns (identity, model, market, edge, quantiles, context) + derived `market_home_prob_novig`, `market_home_margin`, `edge_home_prob`; summary block |
| `GET /predictions/picks`, `GET /predictions/weeks` | picks CSV as-is; weeks lists run-attached + unattached files |
| `GET /betting`, `GET /betting/xlsx` | 25-col betting CSV, or derived from predictions when the CSV is older than the predictions file; ladder thresholds 0.02/0.04/0.07/0.10; `total_*` columns `actionable=false` |
| `GET /power` | rankings + movement vs the prior run's `through_week - 1` file + standings + division standings |
| `GET /model`, `GET /runs/{id}/model` | metadata, holdout metrics, pool summary, missing-data groups, top-40 gain importance from `base_features.combined.gain`, calibration bins when a WF report exists, `wf_compare` + `wf_best`, `metric_strategy` |
| `GET /data/status`, `GET /data/unattached` | current season/week via `data_collection._determine_nfl_week` (imported, not edited), file inventory with sizes/mtimes/row counts (`pl.scan_csv().select(pl.len())`), lazy cached sha256 via `fingerprints.dataset_fingerprint`, cache parquet coverage, latest leakage audit, last ETL job |
| `GET /jobs/catalog`, `POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/logs?after=`, `GET /jobs/{id}/stream` (SSE), `POST /jobs/{id}/cancel` | 409 when an exclusive group is busy; 422 on bad params |
| `GET /{path}` fallback | serves `web/dist/index.html` |

### Job runner

- `build_cmd` always starts with `settings.python` (`.venv/bin/python`); env `PYTHONUNBUFFERED=1`;
  `Popen(cwd=ROOT, stdout=PIPE, stderr=STDOUT, start_new_session=True)`. Reader thread strips
  ANSI (`\x1b\[[0-9;]*m`), parses `[ts][LEVEL][file:func:line] msg`, batches rows into `job_logs`,
  and updates `progress` from the WF fold regex.
- One worker for `exclusive_group="walk_forward"` (weekly_run, walk_forward_backtest,
  golden_command); a 2-slot pool for everything else. On startup, `running` rows become
  `failed (server restarted)`. Cancel sends SIGTERM to the group, SIGKILL after 10s.
- SSE polls the store every 300ms for new `seq`, replays from `after`, ends on terminal status.
- Templates: `etl_full`, `lines_refresh` (chains `predict`), `weekly_run` (writes params JSON to
  `data/web/job_configs/{job_id}.json`, runs `scripts/weekly_run.py --config`, reusing the
  script's own key validation), `train` (`python -m nfl_predictor.ml_model`), `predict`,
  `power_rankings`, `betting_xlsx` (`scripts/betting_report_excel.py`), `leakage_audit`,
  `validate_offline`, `validate_live`, `walk_forward_backtest`, `shap_analysis`.

### `nfl_predictor/lines_refresh.py`

`refresh_lines(season, week, *, data_dir, cache_dir) -> LinesRefreshResult` and `main(argv)`:

1. `load_schedule([season], force_refresh=True, current_season=season)` (rewrites the season
   cache parquet, which is desired).
2. Select `game_id` + the five line columns; apply `fill_missing_moneylines`.
3. For `data/predict/week_{WW}_games_to_predict.csv`, `data/all_data_ml.csv`, `data/all_data.csv`:
   read, `DataFrame.update(lines, on="game_id")` restricted to that season's rows, write
   atomically (tmp + replace) preserving column order. `completed_games*` untouched.
4. Log and return per-file changed-row counts and old/new diffs. Join on `game_id` only.

## Frontend behavior

- Shell: header shows active run chip (run_id, season/week), running-jobs indicator, theme
  toggle, user menu. Content area `min-w-0`; wide tables scroll inside their own container with
  sticky identity columns. Every column header has an info tooltip from the registry.
- DataTable: TanStack Table; polarity-aware heatmap cells (port `buildColumnStats` /
  `getHeatCellStyle` from sos `tableState.ts`); column-group picker persisted in localStorage;
  density toggle.
- Predictions: week selector; desktop table grouped identity | model | market | edge | quantiles
  | context; mobile matchup cards with two-sided win-prob bar, predicted score vs line, p10-p90
  band, confidence rank badge, model-vs-market disagreement highlight. Confidence picks tab with
  copy-to-clipboard.
- Power Rankings: rating bar, movement chip, record; tabs Rankings / Conference / Division
  standings.
- Betting: moneyline and spread recommendations with ActionBadge colors (PASS muted, LEAN slate,
  SMALL/MEDIUM/STRONG greens), EV and edge; totals in a collapsed "informational only" section
  with a tooltip citing the audit; xlsx download or "Generate" (admin) when missing.
- Data/ETL: stat tiles (season/week, rows, seasons, dataset hash, last ETL), file table, cache
  coverage strip, leakage audit badge, Run ETL / Refresh lines buttons (admin).
- Model: metadata cards, holdout metric tiles with direction arrows from `metric_strategy`,
  feature importance bar chart, calibration reliability chart, wf_compare table with the best row
  highlighted, collapsible config JSON.
- Jobs: catalog cards, auto-generated form from the params schema (react-hook-form + zod),
  history table, detail page with virtualized log console, level filter, progress bar, cancel.
- Runs: list with stage chips and holdout Brier/accuracy; Activate with confirm dialog.
- Charts follow the `dataviz` skill; UI follows the `frontend-design` skill in `.agents/skills`.

## Phases

### Phase 0: scaffolding, auth, run index

Backend core (`__init__`, `__main__`, `settings`, `deps`, `errors`, `db`), `auth/*`, `runs/*`,
`routers/{runs,static,users}`, schemas. Frontend scaffold (Vite, Tailwind, shadcn with
`src/utils` alias, router, AppShell, Login, Runs, ThemeProvider). Config edits, CI web job,
`web/README.md`, `.agents/web_ui_plan.md`.
Tests: `tests/api/conftest.py` with `make_run_dir(tmp_path, run_id, kind, season, week)`
writing real-shaped metadata/metrics/feature-importance JSON and small CSVs with real headers;
`test_auth.py`, `test_db.py`, `test_runs_indexer.py` (weekly/training/wf/empty dirs, sort,
stage and week detection), `test_active_run.py`, `test_static.py`.
Done when: `.venv/bin/python -m nfl_predictor.api` serves login and the Runs list from the real
`models/`, admin can activate a run, all gates pass with coverage >= 90.

### Phase 1: the five read-only pages

`registry/*`, `readers/*`, `routers/{predictions,betting,power,model,data,registry}`; pages
Dashboard, Predictions, PowerRankings, Betting, DataStatus, Model, Glossary; table and chart
components.
Tests: registry completeness (every key has label + description), projection drops unknown
columns and nulls NaN, readers over fixtures, totals non-actionable assertion, power movement,
calibration present/absent, data status fingerprint cache, route tests; vitest for DataTable,
ActionBadge, WeekSelector, format.
Done when: all five pages render the active run on desktop Edge and iPhone Edge with tooltips on
every column and the xlsx download works.

### Phase 2: jobs

`jobs/*`, `nfl_predictor/lines_refresh.py`, Jobs/JobDetail pages, LogConsole, admin buttons on
Data and Betting pages.
Tests: catalog (every template's argv starts with the venv python; param validation), runner
against `tests/api/fake_script.py` (ANSI lines, a WF progress line, sleeps, honors SIGTERM; run
with `sys.executable`) covering persistence, ANSI stripping, progress, cancel, exclusive-group
serialization, restart recovery; SSE route via streaming client; `test_lines_refresh.py` with
`load_schedule` monkeypatched and tmp CSVs asserting only line columns change and row order holds.
Done when: every template runs from the UI with live logs, lines refresh chains a predict on the
active model, and two WF jobs queue rather than overlap.

### Phase 3: future-week predictions

Template `predict_week {season, week}`: build `data/predict/week_{WW}_games_to_predict.csv` from
`all_data_ml.csv` when missing (replicating the feature-availability check in
`scripts/power_rankings.py::_predict_future_games` L328), then predict into the active run.
`GET /predictions?week=` resolves run-attached files first, then unattached. Week selector shows
predicted / not-yet status with a Generate action for admins, and a caveat that future weeks lack
current lines, QB, and rest updates until an ETL runs.
Done when: in week 1 the user selects week 3 and sees predictions from the active model.

### Phase 4: pool helpers

- Picks export (`/predictions/picks?format=csv`).
- `GET /pool/tiebreakers?week=`: MNF and SNF games from `game_datetime` weekday and latest
  kickoff, with predicted scores and p10-p90 band; highest and lowest projected team score.
- `POST /pool/survivor {season, start_week, used_teams, horizon_weeks, beam_width}`: pure module
  `nfl_predictor/api/pool/survivor.py`, beam search over (used-set, log survival) states with a
  future-value lookahead penalty so strong teams are saved; returns ranked plans with weekly picks
  and cumulative survival probability. Depends on Phase 3 per-week files.
- `/pool` page with Confidence / Tiebreakers / Survivor tabs.
Done when: the tiebreaker card answers Yahoo's two tiebreakers and a survivor plan renders.

### Phase 5: team and QB pages over nfl-sos-ratings Parquet

`readers/sos.py` copies the minimal readers from `nfl_sos_ratings/ui_data.py` (season discovery,
team/QB payloads, game logs) reading `{season}_combined.parquet`, `_qb_combined.parquet`, and
game-log files from `NFLP_SOS_DATA_DIR`. Routes `/sos/seasons`, `/sos/{season}/teams`, `/qbs`,
game logs. Pages `/teams`, `/teams/:id`, `/qbs`, `/qbs/:id` with the compare workflow. Fixtures
are tiny Parquet files written by Polars.
Done when: the current-season team page shows rating heatmaps and a team detail with game logs.

### Phase 6 (deferred, design only): live betting and live odds

Odds-provider adapter interface plus the `Live` blend from `betting_excel.py`
(`w_time = min(1, minutes_remaining / 60)`, sigma shrink) as an endpoint. No implementation now.

## Verification

- Backend gates from the worktree: `.venv/bin/ruff format .`, `.venv/bin/ruff check .`,
  `.venv/bin/pyright .`, `.venv/bin/ty check .`, `.venv/bin/python -m pytest` (coverage >= 90),
  `markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings" "#web/node_modules"`,
  `uv lock --check`, `uv sync --check --active`.
- Frontend: `npm run lint`, `npm run typecheck`, `npm test -- --run`, `npm run build`; OpenAPI
  type drift check (`npm run gen:api` regenerates `src/api/schema.d.ts`; CI diffs it).
- End to end: build `web/`, start `.venv/bin/python -m nfl_predictor.api`, log in as the
  bootstrap admin, activate `models/weekly_2025_week_22`, and check every page on desktop and at
  400px width; run `validate_offline` and `lines_refresh` from the Jobs page and watch logs stream.
- Use the `run` skill for launching and screenshotting the app during implementation.

## Risks

- Coverage floor: budget tests per module before writing code; keep threads and SSE thin and unit
  test parsers separately.
- Collisions: rebase `feat/web-ui` only after the other session's branch lands; shared-file edits
  limited to `pyproject.toml`, `.gitignore`, CI.
- Node not on PATH for uvicorn: the backend never needs node at runtime.
- Large CSVs: column projection at read, mtime cache, lazy sha256 in a background thread.
- Lines refresh early in the week may find no lines yet: report unchanged counts and warn.
- Security: JWT secret from env or generated file, argon2, login rate limit, Tailscale-only
  exposure recommended, cookie Secure flag documented.

## Status

- Phase 0 (2026-09-10): done. `nfl_predictor/api/` serves auth, users, runs, and the built SPA;
  `web/` is a Vite + React 19 + Tailwind v4 + shadcn app with the shell, login, overview, runs,
  and users pages. `tests/api/` covers the backend; `web/src/**/*.test.ts` the frontend.
- Phase 1 (2026-09-10): done. Column registry (`nfl_predictor/api/registry/`), readers for
  predictions, betting (derived from predictions with the workbook formulas), power rankings with
  week-over-week movement, model metadata/metrics/importance/calibration, and data status; routes
  under `/api/{registry,predictions,betting,power,model,data}`. Frontend pages Predictions (table
  and mobile cards, week selector, confidence picks), Power Rankings (rankings, conference and
  division standings), Betting (action ladder, totals hidden by default, workbook download), Data
  & ETL, Model (tiles, feature importance, reliability diagram, walk-forward candidates), and
  Glossary.
- Phase 2 (2026-09-10): done. `nfl_predictor/api/jobs/` (catalog, SQLite store, subprocess runner,
  SSE stream, routes) and `nfl_predictor/lines_refresh.py`; routes under `/api/jobs`; frontend
  `/jobs` and `/jobs/:jobId` with the generated form, streamed log console, progress bar and
  cancel, plus admin Run ETL / Refresh lines buttons on Data & ETL and Generate workbook on
  Betting. Verified live: `lines_refresh` chained into `predict` against copies of the real
  datasets, writing a predictions CSV.

  Deviations from the plan above, all deliberate:

  - **Progress regex.** The plan cited a `Walk-forward fold N/M done` line in
    `nfl_predictor/ml/walk_forward.py`; the line that actually exists is `WF candidate %d/%d` in
    `scripts/weekly_run.py`. The runner matches both (`(?:fold|candidate) N/M`).
  - **Queue vs 409.** Both behaviors are implemented: the runner gives each exclusive group a
    single worker, so queued jobs in a group never overlap (this is the path chained jobs take),
    while `POST /api/jobs` still answers 409 `group_busy` when the group is already occupied.
  - **Log flushing.** Batching only on new output would hide the last line of a job that logs and
    then works silently, so a reader thread fills a buffer and the worker flushes it every 250 ms.
  - **Join key for the lines refresh.** `game_id` remains the preferred key, but the real
    `data/predict/week_NN_games_to_predict.csv` files have no `game_id` column, so those fall back
    to `(season, week, away_abbr, home_abbr)`. Files with neither key are skipped with a warning.
  - **Unchanged files are not rewritten.** A refresh that finds no changed cell leaves the file
    byte-identical instead of rewriting it through Polars.
  - **Log console.** Rendering is capped at the newest 2000 filtered lines with a "… N earlier
    lines not shown" note rather than a virtualized list; no virtualization library is installed.
  - **`etl_full`, `validate_offline` and `validate_live`** read `nfl_predictor.constants.DATA_PATH`
    (the checkout's own `data/`) and ignore `NFLP_DATA_DIR`; every other template is given explicit
    paths from the settings. This only matters when the API is pointed at another checkout's data.

  Fixed along the way: `Settings.python_executable` was resolved to its target, and because
  `.venv/bin/python` is a symlink to the base interpreter, jobs launched outside the virtual
  environment and failed on `import polars`. The path is now made absolute but never resolved.

- Phase 3 (2026-09-10): done. `nfl_predictor/week_builder.py` extracts one week of upcoming games
  from `all_data_ml.csv` with the ETL's own `filter_upcoming_games` rule; the `predict_week`
  template runs it and chains `predict`, so a future week goes from nothing to predictions in one
  click. `GET /api/predictions/weeks` now also lists the current season's unplayed weeks with
  source `available`, and the Predictions page answers one of those with a "not predicted yet"
  panel carrying the Generate action and the caveat that lines, rest, and quarterbacks are as of
  the last ETL. Verified live: week 3 of 2026 generated and predicted (16 games) from the active
  run's model.

  Deviations: the availability check is the ETL's upcoming-game rule (season, week, and a missing
  score) rather than a copy of `scripts/power_rankings.py::_predict_future_games`, whose feature
  check needs a loaded model; the week file is written with every column of `all_data_ml.csv`,
  which the prediction CLI narrows through the model's feature spec.

- Phase 4: not started.
