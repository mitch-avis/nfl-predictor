# NFL Predictor web UI

A React single-page app served by the FastAPI backend in `nfl_predictor/api/`. It shows the
project's outputs (predictions, power rankings, betting edges, data and model status) and, for
admins, runs the project's jobs.

## Prerequisites

- Node 26 via nvm: `source ~/.nvm/nvm.sh` (nvm-managed Node is not on `PATH` in fresh shells).
- The Python environment with the web extra: `uv sync --extra web` from the repository root.

## Run it

```bash
# 1. Create the first admin (once)
.venv/bin/python -m nfl_predictor.api.auth.cli create-user <name> --role admin

# 2. Backend (serves /api and, once built, the SPA)
.venv/bin/python -m nfl_predictor.api            # http://127.0.0.1:8000
.venv/bin/python -m nfl_predictor.api --reload   # development

# 3. Frontend
cd web
npm ci
npm run dev      # Vite dev server on http://localhost:5173, proxies /api to :8000
npm run build    # writes web/dist, which the backend serves
```

Set `NFLP_HOST=0.0.0.0` (or `--host 0.0.0.0`) to reach the app from another device on the LAN or
over Tailscale. Set `NFLP_COOKIE_SECURE=1` when the app is behind HTTPS.

## Configuration

Every setting is an `NFLP_`-prefixed environment variable (see `nfl_predictor/api/settings.py`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `NFLP_ROOT_DIR` | repository root | Base for every other path |
| `NFLP_DATA_DIR` | `data/` | ETL outputs |
| `NFLP_MODELS_DIR` | `models/` | Run directories |
| `NFLP_STATE_DIR` | `data/web/` | SQLite database and signing key |
| `NFLP_JWT_SECRET` | generated into the state dir | Session signing secret |
| `NFLP_SESSION_HOURS` | `168` | Session lifetime |
| `NFLP_WEB_DIST` | `web/dist/` | Built frontend |
| `NFLP_PYTHON` | `.venv/bin/python` | Interpreter for jobs |

## Checks

```bash
npm run lint        # oxlint
npm run typecheck   # tsc -b
npx vitest run      # unit tests
npm run build
```

## Layout

```text
web/src/
  api/          fetch client, payload types, TanStack Query hooks
  app/          shell, sidebar, theme, auth gate, navigation
  pages/        one component per route
  components/   ui/ (shadcn primitives), common/ (shared pieces)
  utils/        cn() and formatting helpers  (not src/lib: that path is gitignored)
```

Add shadcn components with `NPM_CONFIG_USERCONFIG=/dev/null npx shadcn@latest add <name>` (the
user-level npmrc sets `allow-scripts`, which npm 12 rejects inside project installs), then fix the
generated `from "cn"` imports to `from "@/utils/cn"`.
