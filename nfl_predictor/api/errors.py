"""Structured API errors and their JSON rendering."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """An error with an HTTP status, a stable machine-readable code, and a message."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        """Store the status, code, and message."""
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class NotFoundError(ApiError):
    """A 404 error."""

    def __init__(self, message: str, code: str = "not_found") -> None:
        """Build a 404 error."""
        super().__init__(404, code, message)


class ConflictError(ApiError):
    """A 409 error."""

    def __init__(self, message: str, code: str = "conflict") -> None:
        """Build a 409 error."""
        super().__init__(409, code, message)


class UnauthorizedError(ApiError):
    """A 401 error."""

    def __init__(
        self, message: str = "Authentication required", code: str = "unauthorized"
    ) -> None:
        """Build a 401 error."""
        super().__init__(401, code, message)


class ForbiddenError(ApiError):
    """A 403 error."""

    def __init__(self, message: str = "Admin role required", code: str = "forbidden") -> None:
        """Build a 403 error."""
        super().__init__(403, code, message)


class UnprocessableEntityError(ApiError):
    """A 422 error, matching FastAPI's own validation status."""

    def __init__(self, message: str, code: str = "invalid_params") -> None:
        """Build a 422 error."""
        super().__init__(422, code, message)


class BadRequestError(ApiError):
    """A 400 error."""

    def __init__(self, message: str, code: str = "bad_request") -> None:
        """Build a 400 error."""
        super().__init__(400, code, message)


def render_api_error(_request: Request, exc: Exception) -> JSONResponse:
    """Render an ``ApiError`` as ``{"error": {"code", "message"}}``."""
    assert isinstance(exc, ApiError)  # noqa: S101 - registered only for ApiError
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register the ``ApiError`` handler on ``app``."""
    app.add_exception_handler(ApiError, render_api_error)
