"""A CUDA failure while fitting retrains on the CPU with a warning instead of failing the run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from nfl_predictor.ml import ml_model_core as core
from nfl_predictor.ml import ml_model_xgb_utils as xgb_utils
from nfl_predictor.ml import walk_forward
from tests.test_walk_forward import _base_config, _fixture_df

xgb.set_config(verbosity=0)


class _CudaFailingRegressor(xgb.XGBRegressor):
    """Fails every fit on the GPU, the way a CUDA error at fit time surfaces from XGBoost."""

    def fit(self, *args: Any, **kwargs: Any) -> _CudaFailingRegressor:
        """Raise on CUDA, fit normally on the CPU."""
        if self.get_params().get("device") == "cuda":
            raise xgb.core.XGBoostError("CUDA error: out of memory")
        super().fit(*args, **kwargs)
        return self


@pytest.fixture
def warnings_logged(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Reset the one-time fallback state and capture warnings."""
    monkeypatch.setattr(xgb_utils, "_RUNTIME_STATE", xgb_utils._RuntimeState())
    messages: list[str] = []

    def _capture(msg: str, *args: object) -> None:
        messages.append(msg % args if args else msg)

    monkeypatch.setattr(xgb_utils.log, "warning", _capture)
    monkeypatch.setattr(core.xgb, "XGBRegressor", _CudaFailingRegressor)
    return messages


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a small regression problem."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 3))
    return x, 3.0 * x[:, 0] + rng.normal(size=60), 44.0 + x[:, 1] + rng.normal(size=60)


def _cuda_params() -> dict[str, Any]:
    """Return small params that ask for the GPU."""
    return {"n_estimators": 5, "max_depth": 2, "n_jobs": 1, "device": "cuda", "tree_method": "hist"}


def test_margin_total_fit_falls_back_to_the_cpu_with_a_warning(
    warnings_logged: list[str],
) -> None:
    """Both heads train on the CPU after the CUDA fit fails, and the fallback is a warning."""
    x, margin, total = _data()

    margin_model, total_model = core._fit_margin_total_models(x, margin, total, _cuda_params())

    assert margin_model.get_params()["device"] == "cpu"
    assert total_model.get_params()["device"] == "cpu"
    assert len(warnings_logged) == 1


def test_quantile_fit_falls_back_to_the_cpu_with_a_warning(warnings_logged: list[str]) -> None:
    """The quantile heads fall back the same way."""
    x, margin, _total = _data()

    models = core._fit_quantile_models(x, margin, _cuda_params(), quantiles=(0.1, 0.9))

    assert {model.get_params()["device"] for model in models.values()} == {"cpu"}
    assert len(warnings_logged) == 1


def test_walk_forward_stops_instead_of_checkpointing_a_week_trained_on_the_wrong_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warnings_logged: list[str]
) -> None:
    """A week that fell back to the CPU in a CUDA run fails the run and saves nothing.

    Its checkpoint would sit under the CUDA fingerprint, so resuming would mix devices.
    """
    monkeypatch.setattr(xgb_utils, "xgb_cuda_usable", lambda: True)

    with pytest.raises(RuntimeError, match="cuda"):
        walk_forward.run_walk_forward_backtest(
            _fixture_df(), _base_config(), checkpoint_dir=tmp_path
        )

    assert not list(tmp_path.rglob("fold_*.joblib"))
    assert warnings_logged


def test_walk_forward_weeks_record_the_device_they_trained_on() -> None:
    """Each week's metrics name the device its models were fitted on."""
    result = walk_forward.run_walk_forward_backtest(_fixture_df(), _base_config())

    assert {week["xgb_device"] for week in result["per_week"]} == {"cpu"}
