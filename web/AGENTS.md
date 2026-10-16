# Web app (`web/`)

Scoped instructions for the React app; project-wide rules are in the root `AGENTS.md`, and the
backend's are in `src/nfl_predictor/api/AGENTS.md`.

- `web/` is the Vite + React 19 + Tailwind app; it is excluded from ruff, pyright and ty.
  Its gate (`source ~/.nvm/nvm.sh`, then in `web/`: `pnpm install --frozen-lockfile`,
  `pnpm run lint`, `pnpm run typecheck`, `pnpm exec vitest run`, `pnpm run build`) runs as the
  `web` job in CI; run it whenever `web/` changes.
- The package manager is pnpm, locked by `web/pnpm-lock.yaml`. Add or remove packages with
  `pnpm add` / `pnpm remove`; never run `npm install`, which would write a second lockfile
  (`package-lock.json`) that nothing reads.
- Tests for the backend live in `tests/api/`; the design, decisions and phase status live in
  `.agents/web_ui_plan.md`, and `web/README.md` documents the runtime configuration.
- `.agents/frontend_audit_notes.md` collects problems noticed in the app while doing other work;
  read it before changing a page, add an entry (with `file:line` on `main`) when you see one, and
  move an entry to its "Fixed since noted" section when your change fixes it.
