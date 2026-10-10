"""Suite-wide setup: thread pools, no usable GPU, a private data directory and no network.

Polars, OpenMP (XGBoost) and the BLAS libraries each size a thread pool to every core. The suite
builds thousands of tiny frames and models, where those pools spend their time waking and spinning
threads instead of working, so every variable below defaults to one thread; an explicit value in
the environment is kept. Each library reads its variable once, when it loads, so this module sets
them before importing anything that loads one; the test modules import those only after it runs.

Then ``tests.data_isolation.install`` points the package's data paths and models directory at
session directories and refuses the network for the whole session, before any test module imports
the package (see that module). On top of it every test gets its own empty data and models
directories (paths a module copied at import time keep the shared session directory), and a
refused connection or lookup fails the next test to finish, or the session when none does. So no
test reads the checkout's data through ``constants.DATA_PATH``, writes its caches or line record,
writes or reads a run under ``constants.MODELS_PATH``, or downloads anything; a test that checks
the packaged paths themselves asks for ``packaged_data_paths`` or ``packaged_models_path``. Not
redirected (see ``tests/data_isolation.py``): ``api.settings.Settings()`` built without a root
(its data, state and database paths, the checkout's ``data/web/app.db``), the floor-sigma
reference runs (resolved against ``constants.ROOT_DIR``, by default
``models/step4_reference/adopted_seed*``), and the relative ``data/`` defaults of
``cli/rankings.py``; tests that touch them pass their own paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

# Light: it loads only the package's constants, which load nothing with a thread pool.
from tests import data_isolation

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
data_isolation.install()


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


# The data paths each test points at its own directory: the constants and the module-level copies
# the package's tools use as their default data directory, as an import path and the part of the
# data directory each one names.
DATA_PATH_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nfl_predictor.constants.DATA_PATH", ()),
    ("nfl_predictor.constants.NFLREADPY_CACHE_DIR", ("cache", "nflreadpy")),
    ("nfl_predictor.constants.NFELO_CACHE_DIR", ("cache", "nfelo")),
    ("nfl_predictor.lines_refresh.DEFAULT_DATA_DIR", ()),
    ("nfl_predictor.week_builder.DEFAULT_DATA_DIR", ()),
)


@dataclass
class DataIsolation:
    """A test's private data and models directories and the patch that put them in place."""

    data_dir: Path
    models_dir: Path
    patch: pytest.MonkeyPatch

    @staticmethod
    def take_network_attempts() -> list[str]:
        """Return the refused connections and lookups so far and forget them."""
        return data_isolation.take_network_attempts()


def _fail_on_network_attempts(where: str) -> None:
    """Fail with every refused connection or lookup not yet reported."""
    attempts = data_isolation.take_network_attempts()
    assert not attempts, f"{where} tried to reach the network: {attempts}"


@pytest.fixture(autouse=True, scope="session")
def _no_network_outside_tests() -> Generator[None]:
    """Fail the session for a refused attempt no test reported (a fixture's teardown, say)."""
    yield
    _fail_on_network_attempts("a session or module fixture")


@pytest.fixture(autouse=True)
def isolated_data(tmp_path_factory: pytest.TempPathFactory) -> Generator[DataIsolation]:
    """Point every data path and the models directory at fresh directories, and fail the test if
    it touched the network.

    The refusals themselves are installed for the whole session (``tests.data_isolation``); an
    attempt made by a module- or session-scoped fixture is reported by the next test that ends.
    """
    data_dir = tmp_path_factory.mktemp("data")
    models_dir = tmp_path_factory.mktemp("models")
    with pytest.MonkeyPatch.context() as patch:
        for target, parts in DATA_PATH_TARGETS:
            patch.setattr(target, data_dir.joinpath(*parts))
        patch.setattr("nfl_predictor.constants.MODELS_PATH", models_dir)
        yield DataIsolation(data_dir=data_dir, models_dir=models_dir, patch=patch)
        _fail_on_network_attempts("the test")


@pytest.fixture
def packaged_data_paths(isolated_data: DataIsolation) -> None:
    """Restore the checkout's data paths in every loaded package module, for path checks."""
    data_isolation.restore_packaged_paths(
        isolated_data.patch, (isolated_data.data_dir, data_isolation.SESSION_DATA_DIR)
    )


@pytest.fixture
def packaged_models_path(isolated_data: DataIsolation) -> None:
    """Restore the checkout's models directory in every loaded package module, for path checks.

    Only for tests that compute paths: a test that writes a run uses its own models directory.
    """
    data_isolation.restore_packaged_paths(
        isolated_data.patch,
        (isolated_data.models_dir, data_isolation.SESSION_MODELS_DIR),
        data_isolation.PACKAGED_MODELS_DIR,
    )
