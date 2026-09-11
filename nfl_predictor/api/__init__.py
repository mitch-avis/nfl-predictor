"""FastAPI application serving project outputs and controlling jobs.

Build the app with :func:`create_app`; run it with ``python -m nfl_predictor.api``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from nfl_predictor.api.auth.ratelimit import LoginRateLimiter
from nfl_predictor.api.auth.router import router as auth_router
from nfl_predictor.api.db import Database
from nfl_predictor.api.errors import install_error_handlers
from nfl_predictor.api.jobs.router import router as jobs_router
from nfl_predictor.api.jobs.runner import JobRunner
from nfl_predictor.api.readers.data_status import FingerprintCache
from nfl_predictor.api.routers.betting import router as betting_router
from nfl_predictor.api.routers.data import router as data_router
from nfl_predictor.api.routers.model import router as model_router
from nfl_predictor.api.routers.power import router as power_router
from nfl_predictor.api.routers.predictions import router as predictions_router
from nfl_predictor.api.routers.registry import router as registry_router
from nfl_predictor.api.routers.runs import router as runs_router
from nfl_predictor.api.routers.static import mount_frontend
from nfl_predictor.api.routers.users import router as users_router
from nfl_predictor.api.runs.indexer import RunIndex
from nfl_predictor.api.settings import CSRF_HEADER_NAME, CSRF_HEADER_VALUE, Settings

API_PREFIX = "/api"
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _csrf_rejection() -> JSONResponse:
    """Return the 403 response for a mutating request without the CSRF header."""
    return JSONResponse(
        status_code=403,
        content={
            "error": {
                "code": "csrf_header_missing",
                "message": f"Mutating requests must send {CSRF_HEADER_NAME}: {CSRF_HEADER_VALUE}",
            }
        },
    )


def create_app(settings: Settings | None = None, *, serve_frontend: bool = True) -> FastAPI:
    """Build the application.

    Args:
        settings: Runtime settings; read from the environment when omitted.
        serve_frontend: Whether to mount the built SPA and its fallback route.

    """
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(running_app: FastAPI) -> AsyncIterator[None]:
        """Start the job runner with the server and stop it on shutdown."""
        runner: JobRunner = running_app.state.job_runner
        runner.start()
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(
        lifespan=lifespan,
        title="nfl-predictor",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.db = Database(settings.database_path)
    app.state.login_limiter = LoginRateLimiter()
    app.state.run_index = RunIndex(settings.models_path)
    app.state.fingerprints = FingerprintCache(app.state.db)
    app.state.job_runner = JobRunner(app.state.db, settings, app.state.run_index)
    install_error_handlers(app)

    @app.middleware("http")
    async def require_csrf_header(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Reject cookie-authenticated mutations that lack the custom header."""
        if (
            request.method in MUTATING_METHODS
            and request.url.path.startswith(API_PREFIX)
            and request.headers.get(CSRF_HEADER_NAME, "").lower() != CSRF_HEADER_VALUE
        ):
            return _csrf_rejection()
        return await call_next(request)

    app.include_router(auth_router, prefix=API_PREFIX)
    app.include_router(users_router, prefix=API_PREFIX)
    app.include_router(runs_router, prefix=API_PREFIX)
    app.include_router(registry_router, prefix=API_PREFIX)
    app.include_router(predictions_router, prefix=API_PREFIX)
    app.include_router(betting_router, prefix=API_PREFIX)
    app.include_router(power_router, prefix=API_PREFIX)
    app.include_router(model_router, prefix=API_PREFIX)
    app.include_router(data_router, prefix=API_PREFIX)
    app.include_router(jobs_router, prefix=API_PREFIX)

    @app.get(f"{API_PREFIX}/health", tags=["meta"])
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    if serve_frontend:
        mount_frontend(app, settings.web_dist_path)
    return app
