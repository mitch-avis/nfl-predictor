"""Suite-wide setup: thread pools, no usable GPU, a private data directory and no network.

Polars, OpenMP (XGBoost) and the BLAS libraries each size a thread pool to every core. The suite
builds thousands of tiny frames and models, where those pools spend their time waking and spinning
threads instead of working, so every variable below defaults to one thread; an explicit value in
the environment is kept. Each library reads its variable once, when it loads, so this module sets
them before importing anything that loads one; the test modules import those only after it runs.

Every test also gets its own empty data directory in place of the checkout's ``data/`` (every
data path the package resolves at import time points into it), and any TCP connection is refused
and recorded, failing the test at teardown. So no test reads the checkout's datasets, writes its
caches or line record, or downloads anything; a test that needs the packaged paths themselves
asks for ``packaged_data_paths``.
"""

from __future__ import annotations

import importlib
import os
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator, MutableMapping
    from pathlib import Path

THREAD_VARIABLES = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def limit_threads(environ: MutableMapping[str, str]) -> None:
    """Default every thread-pool variable to one thread in ``environ``, keeping explicit values."""
    for name in THREAD_VARIABLES:
        environ.setdefault(name, "1")


limit_threads(os.environ)


@pytest.fixture(autouse=True, scope="session")
def _no_usable_gpu() -> Generator[None]:
    """Report no usable GPU, so the default ``auto`` XGBoost device resolves to the CPU.

    Results and timings then match on every machine, with or without a GPU. The fixture is
    session-scoped so it also covers module-scoped fixtures that train models. A test that
    needs the GPU branch patches ``xgb_cuda_usable`` again with its own ``monkeypatch``. The
    target is named by its import path, so XGBoost loads only after the thread limits are set.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("nfl_predictor.ml.ml_model_xgb_utils.xgb_cuda_usable", lambda: False)
        yield


# Every data path the package resolves at import time, as an import path and the part of the
# data directory it names.
DATA_PATH_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nfl_predictor.constants.DATA_PATH", ()),
    ("nfl_predictor.constants.NFLREADPY_CACHE_DIR", ("cache", "nflreadpy")),
    ("nfl_predictor.constants.NFELO_CACHE_DIR", ("cache", "nfelo")),
    ("nfl_predictor.lines_refresh.DEFAULT_DATA_DIR", ()),
    ("nfl_predictor.week_builder.DEFAULT_DATA_DIR", ()),
)
_NETWORK_FAMILIES = (socket.AF_INET, socket.AF_INET6)


@dataclass
class DataIsolation:
    """A test's private data directory and the network connections it tried to open."""

    data_dir: Path
    patch: pytest.MonkeyPatch
    packaged_paths: dict[str, Path]
    network_attempts: list[str] = field(default_factory=list)

    def take_network_attempts(self) -> list[str]:
        """Return the recorded connection attempts and forget them, so teardown passes."""
        attempts = list(self.network_attempts)
        self.network_attempts.clear()
        return attempts


@pytest.fixture(autouse=True)
def isolated_data(tmp_path_factory: pytest.TempPathFactory) -> Generator[DataIsolation]:
    """Point every data path at a fresh directory and refuse network connections.

    A refused connection raises ``OSError``, as an offline machine would, so code that degrades
    on a network failure keeps running; the attempt is recorded and fails the test here.
    """
    data_dir = tmp_path_factory.mktemp("data")
    packaged_paths = {target: _resolve(target) for target, _parts in DATA_PATH_TARGETS}
    real_connect = socket.socket.connect
    with pytest.MonkeyPatch.context() as patch:
        isolation = DataIsolation(data_dir=data_dir, patch=patch, packaged_paths=packaged_paths)

        def refuse(sock: socket.socket, address: Any) -> None:
            if sock.family in _NETWORK_FAMILIES:
                isolation.network_attempts.append(repr(address))
                msg = f"network access is blocked in tests: {address!r}"
                raise OSError(msg)
            real_connect(sock, address)

        for target, parts in DATA_PATH_TARGETS:
            patch.setattr(target, data_dir.joinpath(*parts))
        patch.setattr(socket.socket, "connect", refuse)
        yield isolation
    assert not isolation.network_attempts, (
        f"the test tried to reach the network: {isolation.network_attempts}"
    )


@pytest.fixture
def packaged_data_paths(isolated_data: DataIsolation) -> None:
    """Restore the data paths the package resolves at import time, for tests that check them."""
    for target, value in isolated_data.packaged_paths.items():
        isolated_data.patch.setattr(target, value)


def _resolve(target: str) -> Path:
    """Return the value an ``module.attribute`` import path names, importing the module."""
    module, _, name = target.rpartition(".")
    return getattr(importlib.import_module(module), name)
