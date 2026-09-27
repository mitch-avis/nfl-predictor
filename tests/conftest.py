"""Suite-wide fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from nfl_predictor.ml import ml_model_xgb_utils


@pytest.fixture(autouse=True, scope="session")
def _no_usable_gpu() -> Iterator[None]:
    """Report no usable GPU, so the default ``auto`` XGBoost device resolves to the CPU.

    Results and timings then match on every machine, with or without a GPU. The fixture is
    session-scoped so it also covers module-scoped fixtures that train models. A test that
    needs the GPU branch patches ``xgb_cuda_usable`` again with its own ``monkeypatch``.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: False)
        yield
