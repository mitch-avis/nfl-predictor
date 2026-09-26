# Web API backend (`nfl_predictor/api/`)

Scoped instructions for the FastAPI backend; project-wide rules are in the root `AGENTS.md`.
Run `scripts/gate.sh --web` whenever this directory or `web/` changes.

- `nfl_predictor/api/` is the FastAPI backend (`nfl-predictor web`): auth (argon2,
  JWT cookie, `viewer`/`admin`), a run index over `models/*/metadata.json` with one **active**
  run, readers and a column registry for predictions, betting, power rankings, model and data
  status, and a job runner (`nfl_predictor/api/jobs/`) that launches `python -m nfl_predictor
  <command>` as subprocesses with streamed logs; walk-forward jobs share one worker so two never overlap.
  `nfl_predictor/lines_refresh.py` and `nfl_predictor/week_builder.py` are the CLIs it added.
- Tests for the backend live in `tests/api/`; the design, decisions and phase status live in
  `.agents/web_ui_plan.md`, and `web/README.md` documents the runtime configuration.
