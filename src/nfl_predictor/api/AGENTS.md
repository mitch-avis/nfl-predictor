# Web API backend (`src/nfl_predictor/api/`)

Scoped instructions for the FastAPI backend; project-wide rules are in the root `AGENTS.md`.
Run `scripts/gate.sh --web` whenever this directory or `web/` changes.

- `src/nfl_predictor/api/` is the FastAPI backend (`nfl-predictor web`): auth (argon2,
  JWT cookie, `viewer`/`admin`), a run index over `models/*/metadata.json` with one **active**
  run, readers and a column registry for predictions, betting, power rankings, model and data
  status, and a job runner (`src/nfl_predictor/api/jobs/`) that launches `python -m nfl_predictor
  <command>` as subprocesses with streamed logs; walk-forward jobs share one worker so two never overlap.
  `src/nfl_predictor/lines_refresh.py` and `src/nfl_predictor/week_builder.py` are the CLIs it added.
- `api/ratings/` reads the team and quarterback pages' season tables
  (`data/ratings/{season}_{kind}.parquet`, `Settings.ratings_path`) and describes every column
  from the ported metric registry (`nfl_predictor.season_stats.metrics`); `routers/ratings.py`
  serves them under `/api/ratings`, with the response models in `schemas/ratings.py`. A missing
  table is a 404 (`no_ratings_table`), and one Polars cannot read is a 503
  (`ratings_table_unreadable`, `errors.ServiceUnavailableError`).
- `create_app` gzips every response of 500 bytes or more for clients that accept it
  (`GZipMiddleware`, at level 6); Starlette leaves `text/event-stream`, the job log stream,
  uncompressed.
- Tests for the backend live in `tests/api/`; the design, decisions and phase status live in
  `.agents/web_ui_plan.md`, and `web/README.md` documents the runtime configuration.
