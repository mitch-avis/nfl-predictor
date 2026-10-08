"""Suite-wide setup: one thread for every native thread pool, and no usable GPU.

Polars, OpenMP (XGBoost) and the BLAS libraries each size a thread pool to every core. The suite
builds thousands of tiny frames and models, where those pools spend their time waking and spinning
threads instead of working, so every variable below defaults to one thread; an explicit value in
the environment is kept. Each library reads its variable once, when it loads, so this module sets
them before importing anything that loads one; the test modules import those only after it runs.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator, MutableMapping

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
