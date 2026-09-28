"""The in-season calibration hold-out is gone from every command.

The final fit trains on every eligible completed game, as each walk-forward fold does, and no
fit hands XGBoost an eval frame. The options that held weeks or seasons out, or picked that
frame, no longer parse; a weekly config that still sets one is refused with the reason, and
run directories written before the change still read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from nfl_predictor.api.readers import model as model_reader
from nfl_predictor.api.runs.files import resolve_run_files
from nfl_predictor.cli import backtest, train
from nfl_predictor.ml import walk_forward
from nfl_predictor.weekly_run import config as run_config
from tests.api import factories
from tests.test_walk_forward import _base_config, _fixture_df

RETIRED_WEEKLY_KEYS = [
    "wf_calibration_weeks",
    "train_calibration_weeks",
    "train_calibration_seasons",
]


@pytest.mark.parametrize("flag", ["--calibration-weeks", "--calibration-seasons"])
def test_train_no_longer_parses_a_hold_out_option(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``train`` fits every non-holdout game, so it has nothing to hold out."""
    monkeypatch.setattr(sys, "argv", ["prog", flag, "1"])
    with pytest.raises(SystemExit):
        train._parse_args()


@pytest.mark.parametrize("flag", ["--wf-calibration-weeks", "--calibration-weeks"])
def test_backtest_no_longer_parses_the_calibration_frame_option(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``backtest`` hands XGBoost no eval frame, so there is no frame to size."""
    monkeypatch.setattr(sys, "argv", ["prog", flag, "4"])
    with pytest.raises(SystemExit):
        backtest._parse_args()


@pytest.mark.parametrize(
    "flag", ["--wf-calibration-weeks", "--train-calibration-weeks", "--train-calibration-seasons"]
)
def test_weekly_no_longer_parses_a_hold_out_option(flag: str) -> None:
    """The weekly run's final fit and stage 1 hold nothing out."""
    with pytest.raises(SystemExit):
        run_config._build_parser().parse_args([flag, "4"])


@pytest.mark.parametrize("key", RETIRED_WEEKLY_KEYS)
def test_a_config_with_a_retired_hold_out_key_is_rejected(tmp_path: Path, key: str) -> None:
    """An old config that sets a hold-out key fails with the reason, before any data refresh."""
    config_path = tmp_path / "weekly.json"
    config_path.write_text(json.dumps({key: 4}), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{key}.*every eligible completed game"):
        run_config._parse_args(["--config", str(config_path)])


def test_the_walk_forward_config_records_no_calibration_weeks() -> None:
    """The recorded config, and so the checkpoint fingerprint, carries no retired key."""
    config = walk_forward.WalkForwardConfig()

    assert not hasattr(config, "calibration_weeks")
    assert "calibration_weeks" not in config.to_dict()


def test_walk_forward_folds_hand_xgboost_no_eval_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every fold fits its heads without an eval frame, as the final fit does."""
    calls: list[dict[str, Any]] = []
    fit_heads = walk_forward.ml_model._fit_margin_total_models
    fit_quantiles = walk_forward.ml_model._fit_quantile_models

    def spy_heads(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return fit_heads(*args, **kwargs)

    def spy_quantiles(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return fit_quantiles(*args, **kwargs)

    monkeypatch.setattr(walk_forward.ml_model, "_fit_margin_total_models", spy_heads)
    monkeypatch.setattr(walk_forward.ml_model, "_fit_quantile_models", spy_quantiles)

    walk_forward.run_walk_forward_backtest(_fixture_df(), _base_config())

    assert calls
    assert all(kwargs.get("x_eval") is None for kwargs in calls)


def test_a_run_written_with_the_hold_out_still_reads(tmp_path: Path) -> None:
    """Metadata that records the old calibration split and config keys reads unchanged."""
    run_dir = factories.make_run_dir(tmp_path, "weekly")
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    old_splits = {
        "train_seasons": [2020, 2021],
        "calibration_seasons": [],
        "holdout_seasons": [],
        "calibration_inseason": {"season": 2026, "weeks": [1], "pairs": [[2026, 1]]},
    }
    metadata["splits"] = old_splits
    metadata["config"]["wf_calibration_weeks"] = 4
    metadata["config"]["train_config"] = {"calibration_seasons": 0, "calibration_weeks": 4}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    payload = model_reader.model_payload(resolve_run_files(run_dir))

    assert payload["metadata"]["splits"] == old_splits
    assert payload["metadata"]["config"]["train_config"]["calibration_weeks"] == 4
