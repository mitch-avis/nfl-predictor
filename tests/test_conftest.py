"""Tests for the suite-wide setup: thread pools, data paths and network access."""

from __future__ import annotations

import os
import socket

import polars as pl
import threadpoolctl
import xgboost
from tests.conftest import THREAD_VARIABLES, DataIsolation, limit_threads

from nfl_predictor import constants, lines_refresh, week_builder

POOL_VARIABLES = {
    "openmp": "OMP_NUM_THREADS",
    "openblas": "OPENBLAS_NUM_THREADS",
    "mkl": "MKL_NUM_THREADS",
}


def test_limit_threads_defaults_every_pool_to_one_thread() -> None:
    # Arrange
    environ: dict[str, str] = {}

    # Act
    limit_threads(environ)

    # Assert
    assert environ == dict.fromkeys(THREAD_VARIABLES, "1")


def test_limit_threads_keeps_an_explicit_setting() -> None:
    # Arrange
    environ = {"POLARS_MAX_THREADS": "8"}

    # Act
    limit_threads(environ)

    # Assert
    assert environ["POLARS_MAX_THREADS"] == "8"
    assert environ["OMP_NUM_THREADS"] == "1"


def test_polars_starts_its_pool_at_the_configured_size() -> None:
    # Act
    size = pl.thread_pool_size()

    # Assert
    assert size == int(os.environ["POLARS_MAX_THREADS"])


def test_openmp_and_blas_start_their_pools_at_the_configured_size() -> None:
    # Arrange
    xgboost.build_info()  # loads XGBoost's native library, so its OpenMP runtime is reported too

    # Act
    pools = threadpoolctl.threadpool_info()

    # Assert
    sizes = {
        (pool["internal_api"], pool["num_threads"])
        for pool in pools
        if pool["internal_api"] in POOL_VARIABLES
    }
    expected = {(api, int(os.environ[POOL_VARIABLES[api]])) for api, _size in sizes}
    assert sizes
    assert sizes == expected


def test_every_data_path_points_at_the_tests_own_temporary_directory(
    isolated_data: DataIsolation,
) -> None:
    # Act
    paths = {
        "DATA_PATH": constants.DATA_PATH,
        "NFLREADPY_CACHE_DIR": constants.NFLREADPY_CACHE_DIR,
        "NFELO_CACHE_DIR": constants.NFELO_CACHE_DIR,
        "lines_refresh": lines_refresh.DEFAULT_DATA_DIR,
        "week_builder": week_builder.DEFAULT_DATA_DIR,
    }

    # Assert
    data_dir = isolated_data.data_dir
    assert paths == {
        "DATA_PATH": data_dir,
        "NFLREADPY_CACHE_DIR": data_dir / "cache" / "nflreadpy",
        "NFELO_CACHE_DIR": data_dir / "cache" / "nfelo",
        "lines_refresh": data_dir,
        "week_builder": data_dir,
    }
    assert data_dir.resolve() != (constants.ROOT_DIR / "data").resolve()


def _refused_connection(isolated_data: DataIsolation) -> tuple[str, list[str]]:
    """Try to open a TCP connection and return the refusal and the recorded attempts."""
    message = ""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            # 192.0.2.0/24 is reserved for documentation, so nothing real is ever reached.
            sock.connect(("192.0.2.1", 80))
        except OSError as error:
            message = str(error)
    return message, isolated_data.take_network_attempts()


def test_a_network_connection_is_refused_and_recorded(isolated_data: DataIsolation) -> None:
    # Act
    message, attempts = _refused_connection(isolated_data)

    # Assert
    assert "network access is blocked in tests" in message
    assert attempts == ["('192.0.2.1', 80)"]
