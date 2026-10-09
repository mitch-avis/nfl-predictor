"""Tests for the suite-wide setup: thread pools, data paths and network access."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
import pkgutil
import socket
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import polars as pl
import threadpoolctl
import xgboost
from tests import data_isolation
from tests.conftest import THREAD_VARIABLES, DataIsolation, limit_threads

import nfl_predictor
from nfl_predictor import constants, lines_refresh, week_builder
from nfl_predictor.reporting import power_rankings

if TYPE_CHECKING:
    from types import ModuleType

# The checkout's data directory, worked out from this file rather than from the package, whose
# paths the suite redirects.
PACKAGED_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

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
    assert attempts == ["connect ('192.0.2.1', 80)"]


def _refused_lookup(isolated_data: DataIsolation) -> tuple[str, list[str]]:
    """Look a host name up and return the refusal and the recorded attempts."""
    message = ""
    try:
        # .invalid is reserved, so no name server could ever answer for it.
        socket.getaddrinfo("nfl-predictor-tests.invalid", 443)
    except socket.gaierror as error:
        message = str(error)
    return message, isolated_data.take_network_attempts()


def test_a_host_name_lookup_is_refused_and_recorded(isolated_data: DataIsolation) -> None:
    # Act
    message, attempts = _refused_lookup(isolated_data)

    # Assert
    assert "name resolution is blocked in tests" in message
    assert attempts == ["resolve 'nfl-predictor-tests.invalid'"]


def test_a_local_or_numeric_address_still_resolves() -> None:
    # Act
    resolved = [socket.getaddrinfo(host, 80) for host in ("localhost", "127.0.0.1", "192.0.2.1")]

    # Assert
    assert all(resolved)
    assert data_isolation.NETWORK_ATTEMPTS == []


def test_an_import_time_copy_of_a_data_path_starts_in_the_session_directory() -> None:
    # Act
    default = power_rankings.DEFAULT_STRENGTH_SNAPSHOTS

    # Assert
    assert default.is_relative_to(data_isolation.SESSION_DATA_DIR)


def _package_modules() -> list[ModuleType]:
    """Import and return every module of the package except the ``__main__`` entry points."""
    names = sorted(
        info.name
        for info in pkgutil.walk_packages(nfl_predictor.__path__, "nfl_predictor.")
        if not info.name.endswith("__main__")
    )
    return [importlib.import_module(name) for name in names]


def _packaged_data_path(value: object) -> bool:
    """Return whether a value is a path inside the checkout's data directory."""
    if not isinstance(value, PurePath):
        return False
    path = Path(value)
    if not path.is_absolute():
        path = PACKAGED_DATA_DIR.parent / path
    return path.resolve().is_relative_to(PACKAGED_DATA_DIR)


def _function_defaults(label: str, function: object) -> list[tuple[str, object]]:
    """Return a function's positional and keyword-only defaults, labelled."""
    defaults = list(getattr(function, "__defaults__", None) or ())
    keyword = getattr(function, "__kwdefaults__", None) or {}
    return [(f"{label}()", value) for value in [*defaults, *keyword.values()]]


def _class_defaults(label: str, cls: type) -> list[tuple[str, object]]:
    """Return a class's attributes, method defaults and dataclass or pydantic field defaults."""
    found: list[tuple[str, object]] = []
    for name, member in vars(cls).items():
        found.append((f"{label}.{name}", member))
        if inspect.isfunction(member):
            found.extend(_function_defaults(f"{label}.{name}", member))
    if dataclasses.is_dataclass(cls):
        found.extend((f"{label}.{field.name}", field.default) for field in dataclasses.fields(cls))
    model_fields = getattr(cls, "model_fields", None)
    if isinstance(model_fields, dict):
        found.extend(
            (f"{label}.{name}", getattr(info, "default", None))
            for name, info in model_fields.items()
        )
    return found


def _module_values(module: ModuleType) -> list[tuple[str, object]]:
    """Return a module's globals and the defaults of the functions and classes it defines."""
    found: list[tuple[str, object]] = []
    for name, value in vars(module).items():
        label = f"{module.__name__}.{name}"
        found.append((label, value))
        if getattr(value, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(value):
            found.extend(_function_defaults(label, value))
        elif inspect.isclass(value):
            found.extend(_class_defaults(label, value))
    return found


def test_no_module_captures_a_path_inside_the_checkouts_data_directory() -> None:
    # Arrange
    modules = _package_modules()

    # Act
    captured = sorted(
        {
            f"{label} = {value}"
            for module in modules
            for label, value in _module_values(module)
            if _packaged_data_path(value)
        }
    )

    # Assert
    assert captured == []
