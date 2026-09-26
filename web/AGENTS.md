# Web app (`web/`)

Scoped instructions for the React app; project-wide rules are in the root `AGENTS.md`, and the
backend's are in `nfl_predictor/api/AGENTS.md`.

- `web/` is the Vite + React 19 + Tailwind app; it is excluded from ruff, pyright and ty.
  Its gate (`source ~/.nvm/nvm.sh`, then in `web/`: `npm run lint`, `npm run typecheck`,
  `npx vitest run`, `npm run build`) runs as the `web` job in CI; run it whenever `web/` changes.
- Tests for the backend live in `tests/api/`; the design, decisions and phase status live in
  `.agents/web_ui_plan.md`, and `web/README.md` documents the runtime configuration.
