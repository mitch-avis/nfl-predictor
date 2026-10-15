# Web UI for nfl-predictor: FastAPI backend + React frontend

Planned 2026-09-10. This is the implementation plan for the web UI milestone, Milestone 58 in
`.agents/TODO.md` (51 was taken by the power-rankings work in the 2026-09-10 renumbering).
Phases 0-3 merged into `main` on 2026-09-11 as version `0.8.0`. Status of each phase is
tracked at the bottom.

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
- There is no "latest model" concept. `src/nfl_predictor/ml/artifacts.py` only builds paths
  (`resolve_run_paths` L117). The weekly run writes a self-contained run dir
  (`models/weekly_2025_week_22/` is the canonical example) with stage markers
  `{wf_compare,train,predictions,reports}_state.json`; a run is complete iff
  `reports_state.json` exists. Power rankings in a run are stamped `week = predictions_week - 1`.
- Scripts are subprocess-only: every `main()` except `data_collection.main(argv)` reads
  `sys.argv`, and the logger is a single stderr `StreamHandler` with `coloredlogs` ANSI
  (`src/nfl_predictor/utils/logger.py`). Walk-forward logs `Walk-forward fold N/M done ... about Ns
  remaining` (`src/nfl_predictor/ml/walk_forward.py` ~L911) and AGENTS.md forbids concurrent WF runs.
- Market lines come from the nflverse schedule via `polars_utils.load_schedule(...,
  force_refresh=True)` (`src/nfl_predictor/utils/polars/loaders.py` L288); `_prepare_schedule` derives
  `away_spread`/`home_spread`, and `game_utils.fill_missing_moneylines` (L61) fills gaps.
  `data/nfl_lines.csv` is unused legacy. Market features feed the model
  (`src/nfl_predictor/ml/feature_spec.py` L88-93), so a lines refresh must be followed by a
  predict job.
- `.agents/TODO.md` Milestone 50: the total/over-under head has no signal. `total_*` betting
  columns must never be presented as actionable.
- Repo constraints: Python >= 3.14, `.venv/bin/` prefixes, ruff line-length 100, pyright + ty,
  pytest coverage floor 90% over `nfl_predictor` (baseline 90.83%, so new `api/` code needs ~95%),
  docstrings everywhere, TDD, no `print`. Coverage omits any module named `config.py` (use
  `settings.py`). `.gitignore` ignores `lib/`, `dist/`, `*.db`; no Node entries; CI has no Node
  step. Node v26.8.1 via nvm, npm only.
- `../survivor-ratings` is a CBS *Survivor* TV-show scraper, unrelated. Nothing to absorb.

## Decisions (made with the user)

1. Code lives in this repo: `src/nfl_predictor/api/` (FastAPI) + `web/` (React). Each phase is
   built on a feature branch off `main` and merged back when it lands.
2. Frontend: Vite + React 19 + TypeScript + Tailwind v4 + shadcn/ui + TanStack Table + TanStack
   Query + react-router 7 + recharts.
3. Hosting: uvicorn on this WSL box serving API + built SPA over LAN/Tailscale. Auth:
   username/password, argon2 hashes, signed JWT in an httpOnly cookie, roles `viewer` / `admin`.
   Admin-only for jobs, run activation, and user management. Bootstrap admin via CLI.
4. Canonical data: index `models/*/metadata.json` (sorted by `created_at`), admin marks one run
   **active**; predictions/picks/betting/power all read from that run dir. `data/predict/` and
   `reports/` ad-hoc files are surfaced as "unattached".
5. Market lines: a new lines-only ETL mode in a **new module** `src/nfl_predictor/lines_refresh.py`,
   exposed as an on-demand job that chains a predict job on the active model.
6. Jobs: subprocess runner over existing CLIs, SQLite job table, per-job persisted logs streamed
   over SSE, walk-forward jobs serialized.
7. Plan document goes to `.agents/web_ui_plan.md` on the branch. The Milestone 51 entry in
   `.agents/TODO.md` and the `AGENTS.md` amendment (drop the "no dashboards" rule, add the `web`
   extra and Node commands) land in a small follow-up once the other session finishes.
8. Later phases scoped now: future-week predictions, survivor optimizer, tiebreakers, team/QB
   pages over `nfl-sos-ratings` Parquet.

## Layout

### Backend `src/nfl_predictor/api/`

```text
api/
  __init__.py   create_app(settings) factory (mirrors nfl-sos-ratings ui_api.create_app); gzips
                responses (GZipMiddleware at level 6, which leaves text/event-stream alone)
  __main__.py   python -m nfl_predictor.api  -> uvicorn (bootstrap under __main__ guard)
  settings.py   pydantic-settings: NFLP_ROOT_DIR, NFLP_DB_PATH (default data/web/app.db),
                NFLP_JWT_SECRET (auto-generated to data/web/secret.key, 0600), NFLP_COOKIE_SECURE,
                NFLP_WEB_DIST, NFLP_PYTHON_EXECUTABLE (default ROOT/.venv/bin/python); the
                ratings tables are read from NFLP_DATA_DIR/ratings (ratings_path)
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
  ratings/      the team and quarterback pages' readers over data/ratings/{season}_{kind}.parquet:
                files.py (the file contract, season discovery, reads), columns.py (metric
                registry -> MetricColumnOut), tables.py (index tables, categories, team names),
                sections.py (game logs, rating and rank history, rank ranges, head to head),
                schedules.py (the schedule-strength leaderboard), glossary.py (registry payload)
  jobs/         catalog.py (JobTemplate: id, label, description, params schema, build_cmd,
                exclusive_group, chain_after), runner.py, store.py, stream.py, router.py
  routers/      runs.py predictions.py betting.py power.py model.py data.py registry.py
                ratings.py (/ratings, the team and quarterback pages)
                users.py static.py (SPA mount + fallback, 503 page when web/dist missing)
  schemas/      common.py (TablePayload, ColumnMeta), runs.py, predictions.py, betting.py,
                power.py, model.py, data.py, jobs.py, auth.py, ratings.py (MetricColumnOut,
                the ratings tables and the registry payload)
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
               Runs Users Login Glossary  (Phase 4: Pool; Phase 5: Ratings (teams and QBs),
               RatingsDetail, Schedules, RankRanges, HeadToHead)
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

- `pyproject.toml`: the API dependencies (fastapi, uvicorn[standard], pydantic-settings, pyjwt,
  argon2-cffi, sse-starlette, python-multipart) are core dependencies; add `httpx` to the dev
  group for TestClient; add `"web"` to ruff `extend-exclude` and pyright `exclude`.
- `.gitignore`: `web/node_modules/`, `web/dist/`, `web/coverage/`, `web/*.tsbuildinfo`,
  `data/web/`.
- `.github/workflows/validation.yml`: `uv sync` in the Python job; new `web` job
  (`pnpm/action-setup`, `actions/setup-node@v6`, node 26, pnpm cache on `web/pnpm-lock.yaml`,
  `pnpm install --frozen-lockfile`, `pnpm run lint`, `pnpm run typecheck`, `pnpm exec vitest run`,
  `pnpm run build`).
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
| `GET /predictions/picks`, `GET /predictions/weeks` | picks: the run's picks file when it is for the week and no older than its predictions, else derived from the predictions; `format=csv` downloads them. Weeks: the active run's own week, the later weeks predicted into it, other runs, unattached files, then unplayed weeks |
| `GET /pool/confidence`, `GET /pool/tiebreakers`, `POST /pool/survivor` | the Pool page (Phase 4 below): the confidence sheet, the tiebreaker card and ranked survivor plans; read-only |
| `GET /betting`, `GET /betting/xlsx` | 25-col betting CSV, or derived from predictions when the CSV is older than the predictions file; ladder thresholds 0.02/0.04/0.07/0.10; `total_*` columns `actionable=false` |
| `GET /power` | rankings + movement vs the prior run's `through_week - 1` file + standings + division standings |
| `GET /ratings/seasons`, `GET /ratings/{season}/{kind}`, `GET /ratings/{season}/{kind}/rank-ranges`, `GET /ratings/{season}/{kind}/{id}/{game-logs,rating-history,rank-history,head-to-head}`, `GET /ratings/schedules`, `GET /ratings/metrics` | the team and quarterback pages (Phase 5 below), `kind` `teams` or `qbs`, over `data/ratings/{season}_{kind}.parquet`; columns described by the metric registry; read-only; a missing season or table is a 404 `no_ratings_table`, an entity without rows a 404 `no_ratings_entity`, a table the route needs that cannot be read a 503 `ratings_table_unreadable`; an unreadable table that only adds to a page, or one season of the schedule leaderboard, is left out with a warning |
| `GET /model`, `GET /runs/{id}/model` | metadata, holdout metrics, pool summary, missing-data groups, top-40 base features by mean absolute SHAP in points (`base_features.combined.mean_abs_shap`, returned as `{measure, rows, margin_rows}` with per-head `margin_value`/`total_value`; `margin_rows` ranks every feature by the margin head alone; runs without SHAP fall back to `combined.total_gain`, and runs written before total gain was recorded to `combined.gain`, labelled `summed_average_gain`), calibration bins when a WF report exists, `wf_compare` + `wf_best`, `metric_strategy` |
| `GET /data/status`, `GET /data/unattached` | current season/week via `clock.nfl_season` and `clock.nfl_week` (the ETL's calendar rules), file inventory with sizes/mtimes/row counts (`pl.scan_csv().select(pl.len())`), lazy cached sha256 via `fingerprints.dataset_fingerprint`, cache parquet coverage, latest leakage audit, last ETL job |
| `GET /jobs/catalog`, `POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/logs?after=`, `GET /jobs/{id}/stream` (SSE), `POST /jobs/{id}/cancel` | 409 when an exclusive group is busy; 422 on bad params |
| `GET /{path}` fallback | serves `web/dist/index.html` for paths outside `/api`; an unmatched `/api/...` path gets the JSON 404 (`not_found`) |

### Job runner

- `build_cmd` always starts with `settings.python` (`.venv/bin/python`); env `PYTHONUNBUFFERED=1`;
  `Popen(cwd=ROOT, stdout=PIPE, stderr=STDOUT, start_new_session=True)`. Reader thread strips
  ANSI (`\x1b\[[0-9;]*m`), parses `[ts][LEVEL][file:func:line] msg`, batches rows into `job_logs`,
  and updates `progress` from the WF fold regex.
- One worker for `exclusive_group="walk_forward"` (weekly_run, train, walk_forward_backtest);
  a 2-slot pool for everything else. On startup, `running` rows become
  `failed (server restarted)`. Cancel sends SIGTERM to the group, SIGKILL after 10s.
- SSE polls the store every 300ms for new `seq`, replays from `after`, ends on terminal status.
- Templates (since `0.27.0` every one runs `<python> -m nfl_predictor <command>`): `etl_full`
  (`data`), `lines_refresh` (`lines`, chains `predict`), `weekly_run` (`weekly`; passes the form's
  values as options and no `--config`, so the shipped `config/weekly_run.yaml` supplies every
  field left blank, `data_collection_args: "--incremental"` included), `train` (`train`),
  `predict` (`predict`), `predict_week` (`build-week`, chains `predict`), `power_rankings`
  (`rankings`), `leakage_audit` (`leakage-audit`), `validate_offline` (`validate`),
  `validate_live` (`validate --live`), `walk_forward_backtest` (`backtest`), `shap_analysis`
  (`explain`).

### `src/nfl_predictor/lines_refresh.py`

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
  feature importance bar chart (says which measure it shows; both heads added by default, a
  "Margin head only" switch for the margin head's own ranking), calibration reliability chart,
  walk-forward evaluation table (`wf_compare`; one row for the production configuration, or an
  older run's candidates with the trained row highlighted), collapsible config JSON.
- Jobs: catalog cards, auto-generated form from the params schema (react-hook-form + zod),
  history table, detail page with virtualized log console, level filter, progress bar, cancel.
- Runs: list with stage chips and holdout Brier/accuracy; Activate with confirm dialog.
- Charts follow the `dataviz` skill; UI follows the `frontend-design` skill.

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

`jobs/*`, `src/nfl_predictor/lines_refresh.py`, Jobs/JobDetail pages, LogConsole, admin buttons on
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
`src/nfl_predictor/reporting/power_rankings.py::_predict_future_games`), then predict into the active
run. `GET /predictions?week=` resolves run-attached files first, then unattached. Week selector
shows predicted / not-yet status with a Generate action for admins, and a caveat that future weeks
lack current lines, QB, and rest updates until an ETL runs.
Done when: in week 1 the user selects week 3 and sees predictions from the active model.

### Phase 4: pool helpers

- Picks export (`/predictions/picks?format=csv`).
- `GET /pool/tiebreakers?week=`: MNF and SNF games from `game_datetime` weekday and latest
  kickoff, with predicted scores and p10-p90 band; highest and lowest projected team score.
- `POST /pool/survivor {season, start_week, used_teams, horizon_weeks, beam_width}`: pure module
  `src/nfl_predictor/api/pool/survivor.py`, beam search over (used-set, log survival) states with a
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

### Phase 6 (design only): live betting and live odds

Designed 2026-10-10; nothing is built. During a game, show each game's score and clock, an
in-game win probability that starts from the model's pregame prediction, and the live market's
price beside it as a diagnostic. It never changes a stored prediction, pick or confidence rank
(pool picks lock before the week's first game), never feeds training, calibration or the
walk-forward, and makes no profitability claim.

**In-game probability** (a pure function, `src/nfl_predictor/api/live/probability.py`). Stern's
random-walk model of the margin (Stern, "A Brownian Motion Model for the Progress of Sports
Scores", 1994): the final home margin is the current lead plus the margin still to be scored,
normal with mean `r * mu` and standard deviation `sigma * sqrt(r)`, where
`r = min(1, max(0, minutes_remaining / 60))` is the share of regulation left (the `w_time` of the
first sketch), `mu` is the unrounded pregame predicted margin and `sigma` the floor sigma recorded
for that week (the "sigma shrink"):

```text
p_live = Phi((lead + r * mu) / (sigma * sqrt(r)))    for r > 0
```

- At kickoff (`r = 1`, `lead = 0`) this is `Phi(mu / sigma)`, the published pregame probability,
  so the live number starts exactly where the prediction did.
- At the end of regulation it is 1 or 0 by the lead. A tie goes to overtime with `r` for the
  overtime period (10 of 60 minutes in the regular season, 15 in the playoffs), ignoring the
  regular season's chance of a tie and the overtime possession rules.
- It reads only the score and the clock: possession, field position, down and distance and
  timeouts are ignored, so it is least accurate late in close games. Using them would need live
  play-by-play and a model of its own (nflfastR's `wp` is one).
- The market is never blended into it, the same rule as for the pregame probability.

**Market side.** Moneylines (spread and total optional) from the odds adapter, made no-vig with
`readers/market.py`, shown beside the live probability with the difference labelled a
diagnostic. No action ladder for live markets: the live probability has not been measured against
them.

**Adapters** (`src/nfl_predictor/api/live/providers.py`): two `Protocol`s, so tests use fakes and
providers can be swapped without touching the routes.

- `GameStateProvider.fetch(season, week) -> list[GameState]`: game id, scores, quarter, clock,
  minutes remaining, status (scheduled, live, final) and `fetched_at`.
- `OddsProvider.fetch(season, week) -> list[LiveOdds]`: game id, bookmaker, moneylines, spread,
  total and `quoted_at`.
- A `none` provider for each is the default; the live view then says live data is off. Games
  match on `(season, week, home, away)` after team normalization, like the lines refresh.

**Polling and serving.** Background tasks in the app's lifespan, each started only when its
provider is configured and polling only inside the week's game windows (30 minutes before the
first kickoff to the last final), keep the latest snapshot in memory with its time and back off
on errors:

- Scores and clock every `NFLP_LIVE_POLL_SECONDS` (default 60).
- Odds on their own schedule, set by a monthly credit budget, because odds feeds charge per
  request. The Odds API charges one credit per market and region per request and reports the
  credits left in its `x-requests-remaining` response header. A 60-second odds poll through one
  Sunday's games (about 10.5 hours) would make about 630 requests and spend the free plan's 500
  credits in that one Sunday. The poller instead spreads `NFLP_ODDS_CREDITS_PER_MONTH` (default
  500, one credit per poll for moneylines in one region) over the game windows left in the month:
  with Thursday, Sunday and Monday games (about 17.5 hours a week) that is one odds poll about
  every 9 minutes. It re-reads the credits left after each call, stretches the interval when they
  run low, never polls faster than `NFLP_LIVE_ODDS_MIN_SECONDS` (default 120), and stops at zero
  (the live view then shows the last odds with their age).

`GET /api/live?season=&week=` returns per game the state, the pregame and live probabilities,
the live no-vig market probability, their difference and the age of each part (each marked stale
after three missed polls of its own schedule). The frontend refetches every 30 seconds while a
game is live (a strip on the Predictions page during game windows, and a `/live` page).

**Configuration.** `NFLP_LIVE_PROVIDER` and `NFLP_LIVE_ODDS_PROVIDER` (both `none` by default),
`NFLP_LIVE_POLL_SECONDS`, `NFLP_ODDS_CREDITS_PER_MONTH`, `NFLP_LIVE_ODDS_MIN_SECONDS`, and
`NFLP_ODDS_API_KEY` read from the environment only, never logged or written to disk.

**Validation before anything is shown** (no network needed). Score the in-game probability on
past seasons' nflverse play-by-play: at every snap, the lead and `game_seconds_remaining`, `mu`
from the adopted reference's walk-forward out-of-fold predictions (known before kickoff) and that
week's sigma; Brier and log loss by time bucket (each quarter, the last five minutes), beside
nflfastR's `wp` (no market) and `vegas_wp` (market-informed) on the same plays. The hypothesis and
decision rule are written first and the numbers reviewed independently (`AGENTS.md` rules 3 and
4). If the random walk is clearly worse than `wp` late in games, the live view stops at the end of
the third quarter or moves to a play-by-play model.

**Questions for the user before any build** (each adds an external service, which `AGENTS.md`
asks about first):

1. The live score source. Recommendation: ESPN's public scoreboard JSON behind the adapter (free,
   no key, unofficial, so it can change without notice), because the feature is a display aid; a
   paid official feed (SportsDataIO, Sportradar) only if it proves worth keeping.
2. The odds source. Recommendation: none at first, showing the live probability alone; add The
   Odds API (the-odds-api.com) only if the probability passes its validation. Its free plan was
   500 credits a month at one credit per market and region per request (check before choosing),
   which the budgeted poller above spreads to one moneyline poll about every 9 minutes through
   the Thursday, Sunday and Monday games; a paid plan buys a shorter interval.
3. Whether to keep odds snapshots under `data/live/` for research, such as the pick-time line
   yardstick (a write under `data/`).

**Build order once approved:** the pure probability function with its tests and the historical
validation; the adapters with fakes; the poller and `/api/live`; the live views.

## Verification

- Backend gates from the worktree: `.venv/bin/ruff format .`, `.venv/bin/ruff check .`,
  `.venv/bin/pyright .`, `.venv/bin/ty check .`, `.venv/bin/python -m pytest` (coverage >= 90),
  `markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings" "#web/node_modules"`,
  `uv lock --check`, `uv sync --check --active`.
- Frontend: `pnpm run lint`, `pnpm run typecheck`, `pnpm exec vitest run`, `pnpm run build`;
  OpenAPI type drift check (`pnpm run gen:api` regenerates `src/api/schema.d.ts`; CI diffs it).
- End to end: build `web/`, start `.venv/bin/python -m nfl_predictor.api`, log in as the
  bootstrap admin, activate `models/weekly_2025_week_22`, and check every page on desktop and at
  400px width; run `validate_offline` and `lines_refresh` from the Jobs page and watch logs stream.
- Use the `run` skill for launching and screenshotting the app during implementation.

## Risks

- Coverage floor: budget tests per module before writing code; keep threads and SSE thin and unit
  test parsers separately.
- Node not on PATH for uvicorn: the backend never needs node at runtime.
- Large CSVs: column projection at read, mtime cache, lazy sha256 in a background thread.
- Lines refresh early in the week may find no lines yet: report unchanged counts and warn.
- Security: JWT secret from env or generated file, argon2, login rate limit, Tailscale-only
  exposure recommended, cookie Secure flag documented.

## Status

- Phase 0 (2026-09-10): done. `src/nfl_predictor/api/` serves auth, users, runs, and the built SPA;
  `web/` is a Vite + React 19 + Tailwind v4 + shadcn app with the shell, login, overview, runs,
  and users pages. `tests/api/` covers the backend; `web/src/**/*.test.ts` the frontend.
- Phase 1 (2026-09-10): done. Column registry (`src/nfl_predictor/api/registry/`), readers for
  predictions, betting (derived from predictions with the workbook formulas), power rankings with
  week-over-week movement, model metadata/metrics/importance/calibration, and data status; routes
  under `/api/{registry,predictions,betting,power,model,data}`. Frontend pages Predictions (table
  and mobile cards, week selector, confidence picks), Power Rankings (rankings, conference and
  division standings), Betting (action ladder, totals hidden by default, workbook download), Data
  & ETL, Model (tiles, feature importance, reliability diagram, walk-forward candidates), and
  Glossary.
- Phase 2 (2026-09-10): done. `src/nfl_predictor/api/jobs/` (catalog, SQLite store, subprocess runner,
  SSE stream, routes) and `src/nfl_predictor/lines_refresh.py`; routes under `/api/jobs`; frontend
  `/jobs` and `/jobs/:jobId` with the generated form, streamed log console, progress bar and
  cancel, plus admin Run ETL / Refresh lines buttons on Data & ETL and Generate workbook on
  Betting. Verified live: `lines_refresh` chained into `predict` against copies of the real
  datasets, writing a predictions CSV.

  Deviations from the plan above, all deliberate:

  - **Progress regex.** Walk-forwards log `Walk-forward fold N/M done`
    (`src/nfl_predictor/ml/walk_forward.py`), the weekly run's stage 1 included; older weekly runs
    logged `WF candidate N/M`. The runner matches both (`(?:fold|candidate) N/M`).
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
  - **`etl_full`, `validate_offline` and `validate_live`** originally read
    `nfl_predictor.constants.DATA_PATH` (the checkout's own `data/`) and ignored `NFLP_DATA_DIR`;
    every other template is given explicit paths from the settings. The three CLIs now take
    `--data-dir` and the templates pass `NFLP_DATA_DIR`, so the datasets they read and write follow
    the configured tree. The ETL's upstream inputs (the copied `qb_elos.csv`, the quarterback
    identity file, the TeamRankings and nflreadpy caches) still resolve from
    `constants.DATA_PATH`.

  Fixed along the way: `Settings.python_executable` was resolved to its target, and because
  `.venv/bin/python` is a symlink to the base interpreter, jobs launched outside the virtual
  environment and failed on `import polars`. The path is now made absolute but never resolved.

- Phase 3 (2026-09-10): done. `src/nfl_predictor/week_builder.py` extracts one week of upcoming games
  from `all_data_ml.csv` with the ETL's own `filter_upcoming_games` rule; the `predict_week`
  template runs it and chains `predict`, so a future week goes from nothing to predictions in one
  click. `GET /api/predictions/weeks` now also lists the current season's unplayed weeks with
  source `available`, and the Predictions page answers one of those with a "not predicted yet"
  panel carrying the Generate action and the caveat that lines, rest, and quarterbacks are as of
  the last ETL. Verified live: week 3 of 2026 generated and predicted (16 games) from the active
  run's model.

  Deviations: the availability check is the ETL's upcoming-game rule (season, week, and a missing
  score) rather than a copy of `src/nfl_predictor/reporting/power_rankings.py::_predict_future_games`,
  whose feature check needs a loaded model; the week file is written with every column of
  `all_data_ml.csv`, which the prediction CLI narrows through the model's feature spec.

- Merged (2026-09-11): `feat/web-ui` merged into `main` as version `0.8.0` after merging
  `main` (the calibration-window, total-head and quarterback work, `0.6.1`-`0.7.1`) into it;
  no conflicts. The web server libraries became core dependencies, and `README.md` /
  `AGENTS.md` now describe the app. Open phases continue as Milestone 58 in `.agents/TODO.md`.
- Phase 4 (2026-10-10, branch of roadmap step 6, not merged): built. `src/nfl_predictor/api/pool/`
  (`sheet.py`, `tiebreakers.py`, `survivor.py`, `results.py`, `kickoff.py`) and
  `routers/pool.py` serve `GET /api/pool/confidence`, `GET /api/pool/tiebreakers` and
  `POST /api/pool/survivor`; `GET /api/predictions/picks?format=csv` downloads the picks. The
  `/pool` page has the Confidence sheet, Tiebreakers and Survivor tabs. Every helper reads the
  predictions file the Predictions page shows for the week and changes no probability, pick or
  rank. Covered by the route and module tests in `tests/api/test_pool_*.py` and the component
  tests under `web/src/components/pool/`; the page was not checked in a browser. An independent
  review asked for the fixes recorded below. The judgment calls are logged in
  `.agents/sos_port/decisions/web.md`.

  Deviations from the Phase 4 text above, all deliberate:

  - **A confidence sheet route.** The plan named the Confidence tab only; `GET
    /api/pool/confidence` reads the week's existing ranks (reporting, not repairing, ranks that
    are not the unique `1..N`), scores final games by the pool rules (a tie scores for neither
    side; final scores from the predictions file or `all_data.csv`), names the lock (the first
    kickoff) and flags predictions written after it. When the predictions were rewritten after
    the lock and the weekly run's picks file was written before it, the sheet and its points come
    from that picks file (picks are single-shot), the newer ranks only as notes; the picks route
    and its CSV download serve the same file.
  - **Tiebreak games are chosen, not fixed.** Yahoo designates its two tiebreak games each week,
    so the card lists every game with its entry and preselects the latest Sunday and Monday
    kickoffs (filled with the latest other kickoffs in weeks without them). Entries round the
    total and split it by the margin, so they keep the rounded total.
  - **Highest and lowest team score.** The tiebreaker asks for the week's actual top and bottom
    scores, which run well past the best and worst projections, so the card gives the median of
    each extreme of normal team scores (spreads from the quantile bands) and names the projected
    top and bottom teams beside it.
  - **Survivor plans are exact: a substitute method, on the user's approval list.** The task
    text above asks for a beam search over (used-set, log survival) states with a future-value
    lookahead penalty. The first build did that; the independent review found its plans below
    the exact optimum, and the problem is an assignment of weeks to teams at cost `-log p` that
    `scipy.optimize.linear_sum_assignment` solves exactly in milliseconds. So each possible pick
    this week gets the exact best plan over every week with predictions, `beam_width` is gone, and
    `horizon_weeks` only sets how many weeks show as near-term picks (decision log, entry 9).
    Under `AGENTS.md` rule 2 this substitute for the task text's method needs the user's approval
    before task 58.1 is checked off. Plans come back one per first pick; a team's future value is
    what the weeks after this one lose without it. Weeks without predictions are listed rather
    than guessed, games already final are not offered, and a request lists at most 32 used
    teams.

  Fixed along the way: predicting a later week into the active run (Phase 3's Generate) moved the
  run to that week and hid its own week (the selector dropped it and asking for it returned 404);
  a file predicted ahead into a pinned run then outranked a newer weekly run's own week (a week
  now goes to the newest run whose predictions stage wrote it, before any file predicted ahead);
  the picks route served the run's own week's picks file for any week; the week selector set its
  four URL parameters separately, so only the last took effect; and the navigation, the Overview
  cards and the catch-all page carried plan labels ("phase N"), now removed.
- Margin-only feature importance (2026-10-10, task 58.6, not merged): `GET /model` adds
  `feature_importance.margin_rows`, every feature ranked by the margin head alone in the run's
  measure, and the Model page's chart gains a "Margin head only" switch (off by default, hidden
  for runs without per-head numbers).
- Phase 5 (2026-10-10, branch of roadmap step 6, not merged): built. `src/nfl_predictor/api/ratings/`
  (`files.py`, `columns.py`, `tables.py`, `sections.py`, `schedules.py`, `glossary.py`) and
  `routers/ratings.py` serve the routes under `/api/ratings` over the season tables in
  `data/ratings/{season}_{kind}.parquet` (the kinds and columns of `nfl-sos-ratings`, Washington
  as `WSH`). Every column is described from the ported metric registry
  (`nfl_predictor.season_stats.metrics`) in the app's column format, so the existing table
  component labels, sorts and shades it; context columns (schedule strength, the opponents faced)
  shade in a muted slate hue. Pages: `/teams` and `/qbs` (views, registry categories, search, the
  qualifier, a compare workflow of up to four rows), `/teams/:id` and `/qbs/:id` (ratings with
  ranks, rank range, head to head, rating and rank by week, season stats by view, game log with
  each opponent's season ratings, opponents faced), `/schedules`, `/rank-ranges`, `/head-to-head`,
  and a team-and-quarterback tab in the Glossary. The sidebar and the overview group the pages in
  sections, and Power Rankings and Teams link to each other. Covered by `tests/api/test_ratings_*.py`
  and the tests beside `web/src/components/ratings/` and `web/src/pages/`. Checked against the
  published `nfl-sos-ratings` tables of 1999-2026, copied to a scratch folder with canonical team
  codes and served from the worktree on a spare port: every route answered (the 492-column team
  table in about 0.12 s, about 0.9 MB of JSON), and the frontend's view model was run over those
  payloads; the pages were not checked in a browser. The judgment calls are logged in
  `.agents/sos_port/decisions/web_pages.md`. After review, responses are gzipped app-wide (level
  6), a table a route needs that cannot be read answers 503 while one that only adds to a page is
  left out with a warning, and each opponent faced carries a league tier.

  The ETL writes every table these pages read to `data/ratings/`: the ratings tables
  (`utils/polars/ratings_build.py`) and, built beside each season's ratings, the season-statistics
  tables the index and game-log pages need (`combined`, `qb_combined`, the game logs;
  `utils/polars/season_stats_build.py`). Built through the ETL's own path on the snapshot those
  files came from, they equal the published files cell for cell
  (`.agents/sos_port/SEASON_STATS_ETL_PARITY.md`), so they are the tables the pages were checked
  against. While the folder holds none, the pages show an empty state that names the folder and
  its files and says to run the ETL. The done-criterion above (the current-season team page shows
  rating heatmaps and a team detail with game logs) now waits on an ETL run that writes
  `data/ratings/` (held with step 6's approvals) and a look at the pages in a browser.

  Deviations from the Phase 5 text above, all deliberate:

  - **The data folder.** The tables come from `data/ratings/` under the configured data directory
    (`Settings.ratings_path`), where the ETL writes them, not from `NFLP_SOS_DATA_DIR` in the
    sibling checkout; that setting is gone.
  - **Routes and modules.** The routes are under `/api/ratings/{season}/{kind}/...` (one handler for
    teams and quarterbacks) rather than `/sos/...`, and the readers are the `api/ratings/` package
    rather than one `readers/sos.py`, because they port every reader the pages need, not the
    minimal set.
  - **More pages than named.** Schedules, rank ranges, head to head and the glossary tab join the
    team and quarterback pages, as the step-6 task asks.
  - **Comparison.** The compare workflow compares up to four rows on the index pages, and the
    head-to-head page compares two with their head-to-head chance.
- Phase 6: designed 2026-10-10 (section above); nothing built, and the build waits on the user's
  answers to its three questions.
- Fixed in `0.26.1` (task 58.5): an unknown `/api/...` GET now returns the JSON 404 instead of
  the frontend's `index.html`.
