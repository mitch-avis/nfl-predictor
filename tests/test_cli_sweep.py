"""Tests for the walk-forward comparison script's checkpoint wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_predictor.cli import sweep
from nfl_predictor.ml import ml_model_xgb_utils, walk_forward


def test_parse_args_resumes_from_the_shared_checkpoint_dir_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every candidate resumes from saved weeks unless the run asks otherwise."""
    monkeypatch.setattr(sys, "argv", ["wf_compare.py"])
    defaults = sweep._parse_args()
    assert defaults.resume is True
    assert defaults.checkpoint_dir == walk_forward.DEFAULT_CHECKPOINT_DIR

    monkeypatch.setattr(sys, "argv", ["wf_compare.py", "--no-resume"])
    assert sweep._parse_args().resume is False


def test_run_one_forwards_checkpoint_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each candidate run hands the checkpoint settings to the engine."""
    captured: dict[str, object] = {}

    def fake_run(
        _df: pd.DataFrame, _config: walk_forward.WalkForwardConfig, **kwargs: object
    ) -> dict[str, object]:
        """Record the keyword arguments and return a minimal result."""
        captured.update(kwargs)
        return {
            "overall": {
                "deterministic_brier": 0.2,
                "deterministic_log_loss": 0.6,
                "market_brier": 0.21,
                "market_log_loss": 0.61,
                "deterministic_brier_vs_market": -0.01,
                "deterministic_log_loss_vs_market": -0.01,
            },
            "reliability": [],
        }

    monkeypatch.setattr(sweep.walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(sweep.metrics_utils, "reliability_ece", lambda _bins: 0.0)

    row = sweep._run_one(
        pd.DataFrame(),
        label="candidate",
        eval_last_n_seasons=1,
        wf_start_week=3,
        calibration="none",
        calibration_weeks=1,
        exclude_incomplete_seasons=False,
        include_market=False,
        market_anchor=False,
        market_mode="features",
        market_prob_source="raw",
        market_prob_blend_method="prob",
        win_prob_use_uncertainty=False,
        market_prob_weight=0.0,
        market_prob_clamp=0.0,
        xgb_params_overrides={},
        include_quantiles=False,
        checkpoint_dir=tmp_path,
        resume=False,
    )

    assert captured == {"checkpoint_dir": tmp_path, "resume": False}
    assert row["label"] == "candidate"
    assert row["deterministic_brier"] == pytest.approx(0.2)
    assert row["deterministic_log_loss"] == pytest.approx(0.6)
    assert row["market_brier"] == pytest.approx(0.21)
    assert row["market_log_loss"] == pytest.approx(0.61)
    assert row["deterministic_brier_vs_market"] == pytest.approx(-0.01)
    assert row["deterministic_log_loss_vs_market"] == pytest.approx(-0.01)


def test_printed_summary_shows_each_probability_view_with_its_pick_accuracy() -> None:
    """The console table pairs the configured and deterministic views with their pick accuracy."""
    columns = sweep.SUMMARY_DISPLAY_COLUMNS

    assert {"brier", "log_loss", "pick_accuracy"} <= set(columns)
    assert {"deterministic_brier", "deterministic_pick_accuracy"} <= set(columns)
    assert columns.index("pick_accuracy") == columns.index("log_loss") + 1
    assert columns.index("deterministic_pick_accuracy") == (
        columns.index("deterministic_log_loss") + 1
    )


def test_xgb_device_defaults_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sweep picks the GPU when one is usable unless told otherwise."""
    monkeypatch.setattr(sys, "argv", ["wf_compare.py"])
    assert sweep._parse_args().xgb_device == "auto"

    monkeypatch.setattr(sys, "argv", ["wf_compare.py", "--xgb-device", "cpu"])
    assert sweep._parse_args().xgb_device == "cpu"


@pytest.mark.parametrize(
    ("overrides", "usable", "expected"),
    [({}, True, "cuda"), ({}, False, "cpu"), ({"device": "cpu"}, True, "cpu")],
)
def test_run_one_trains_on_and_records_the_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    usable: bool,
    expected: str,
) -> None:
    """Each candidate row names the concrete device its walk-forward trained on."""
    captured: list[walk_forward.WalkForwardConfig] = []

    def fake_run(
        _df: pd.DataFrame, config: walk_forward.WalkForwardConfig, **_kwargs: object
    ) -> dict[str, object]:
        """Record the config and return a minimal result."""
        captured.append(config)
        return {"overall": {}, "reliability": []}

    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: usable)
    monkeypatch.setattr(sweep.walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(sweep.metrics_utils, "reliability_ece", lambda _bins: 0.0)

    row = sweep._run_one(
        pd.DataFrame(),
        label="candidate",
        eval_last_n_seasons=1,
        wf_start_week=3,
        calibration="none",
        calibration_weeks=1,
        exclude_incomplete_seasons=False,
        include_market=False,
        market_anchor=False,
        market_mode="features",
        market_prob_source="raw",
        market_prob_blend_method="prob",
        win_prob_use_uncertainty=False,
        market_prob_weight=0.0,
        market_prob_clamp=0.0,
        xgb_params_overrides=overrides,
        include_quantiles=False,
    )

    assert (captured[0].xgb_params_overrides or {})["device"] == expected
    assert row["xgb_device"] == expected


def test_main_passes_the_requested_device_to_every_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--xgb-device`` reaches each candidate's XGBoost overrides."""
    data_path = tmp_path / "games.csv"
    data_path.write_text("season,week\n2025,1\n", encoding="utf-8")
    devices: set[object] = set()

    def fake_run_one(_df: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        """Record the candidate's device and return a displayable row."""
        overrides = kwargs["xgb_params_overrides"]
        assert isinstance(overrides, dict)
        devices.add(overrides.get("device"))
        return dict.fromkeys(sweep.SUMMARY_DISPLAY_COLUMNS, 0.0)

    monkeypatch.setattr(sweep, "_run_one", fake_run_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wf_compare.py",
            "--data-path",
            str(data_path),
            "--out",
            str(tmp_path / "out.csv"),
            "--xgb-device",
            "cpu",
        ],
    )

    assert sweep.main() == 0
    assert devices == {"cpu"}
