"""Tests for the API error rendering."""

from __future__ import annotations

import json

import pytest
from fastapi import Request

from nfl_predictor.api.errors import BadRequestError, render_api_error

REQUEST = Request(scope={"type": "http"})


def test_an_api_error_renders_as_its_status_and_code() -> None:
    """An ``ApiError`` becomes ``{"error": {"code", "message"}}`` with its status."""
    response = render_api_error(REQUEST, BadRequestError("nope", code="bad_thing"))

    assert response.status_code == 400
    assert json.loads(bytes(response.body)) == {"error": {"code": "bad_thing", "message": "nope"}}


def test_any_other_exception_passes_through() -> None:
    """The handler renders only API errors; anything else is re-raised untouched."""
    with pytest.raises(ValueError, match="boom"):
        render_api_error(REQUEST, ValueError("boom"))
