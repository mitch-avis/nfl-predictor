"""Serve the built single-page frontend with a history-API fallback."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

MISSING_BUILD_HTML = """<!doctype html><title>nfl-predictor</title>
<h1>Frontend not built</h1>
<p>The API is running, but <code>web/dist</code> does not exist. Run <code>npm run build</code>
inside <code>web/</code> (or <code>npm run dev</code> for the dev server).</p>
"""


def _safe_child(base: Path, relative: str) -> Path | None:
    """Return ``base/relative`` when it stays inside ``base``, else ``None``."""
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def mount_frontend(app: FastAPI, dist_dir: Path) -> None:
    """Mount ``dist_dir/assets`` and add the SPA fallback route.

    Any path outside ``/api`` returns the matching file from ``dist_dir`` when it exists and
    ``index.html`` otherwise, so client-side routes deep-link correctly. When ``dist_dir`` has no
    ``index.html`` a short explanatory page is served with status 503.
    """
    assets = dist_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    router = APIRouter()

    @router.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:
        """Serve a built file or fall back to ``index.html``."""
        index = dist_dir / "index.html"
        if not index.is_file():
            return HTMLResponse(MISSING_BUILD_HTML, status_code=503)
        candidate = _safe_child(dist_dir, path) if path else None
        if candidate is not None and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)

    app.include_router(router)
