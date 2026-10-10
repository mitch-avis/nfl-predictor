"""Response compression: JSON goes out gzipped, the job log stream does not."""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

from nfl_predictor.api.jobs.store import JobStore, LogLine

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

GZIP = {"Accept-Encoding": "gzip"}
IDENTITY = {"Accept-Encoding": "identity"}
GZIP_CONTAINER = 16 + zlib.MAX_WBITS
"""The ``wbits`` value that wraps a deflate stream in a gzip header, as the middleware does."""


def test_a_large_json_response_is_gzipped(viewer_client: TestClient) -> None:
    """The glossary's registry, well over a few hundred bytes, travels compressed."""
    # Act
    response = viewer_client.get("/api/ratings/metrics", headers=GZIP)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert response.json()["metrics"]["team_rating"]["label"] == "Team Rating"


def test_json_is_sent_plain_to_a_client_that_does_not_accept_gzip(
    viewer_client: TestClient,
) -> None:
    """Compression follows the request's Accept-Encoding."""
    # Act
    response = viewer_client.get("/api/ratings/metrics", headers=IDENTITY)

    # Assert
    assert "content-encoding" not in response.headers


def test_json_is_compressed_at_gzip_level_6(viewer_client: TestClient) -> None:
    """Gzip's default level: level 9 saves a few percent more bytes for noticeably more CPU."""
    # Arrange
    plain = viewer_client.get("/api/ratings/metrics", headers=IDENTITY).content

    # Act
    with viewer_client.stream("GET", "/api/ratings/metrics", headers=GZIP) as response:
        sent = b"".join(response.iter_raw())

    # Assert
    assert sent == zlib.compress(plain, 6, wbits=GZIP_CONTAINER)


def test_the_job_log_stream_is_not_compressed(admin_client: TestClient, app: FastAPI) -> None:
    """Server-sent events stay uncompressed, so each line reaches the browser as it is written."""
    # Arrange
    store: JobStore = app.state.job_runner.store
    record = store.create("validate_offline", {})
    store.append_logs(record.id, [LogLine("INFO", "x" * 2000)])
    store.finish(record.id, "succeeded", exit_code=0)

    # Act
    with admin_client.stream("GET", f"/api/jobs/{record.id}/stream", headers=GZIP) as response:
        body = "".join(response.iter_text())

    # Assert
    assert "content-encoding" not in response.headers
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: end" in body
