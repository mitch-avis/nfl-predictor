"""Tests for the suite-wide setup: one thread for every native thread pool."""

from __future__ import annotations

import os

import polars as pl
import threadpoolctl
import xgboost
from tests.conftest import THREAD_VARIABLES, limit_threads

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
