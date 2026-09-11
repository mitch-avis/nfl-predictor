"""Tests for the walk-forward comparison script's checkpoint wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_predictor.ml import walk_forward
from scripts import wf_compare


def test_parse_args_resumes_from_the_shared_checkpoint_dir_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every candidate resumes from saved weeks unless the run asks otherwise."""
    monkeypatch.setattr(sys, "argv", ["wf_compare.py"])
    defaults = wf_compare._parse_args()
    assert defaults.resume is True
    assert defaults.checkpoint_dir == walk_forward.DEFAULT_CHECKPOINT_DIR

    monkeypatch.setattr(sys, "argv", ["wf_compare.py", "--no-resume"])
    assert wf_compare._parse_args().resume is False


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
        return {"overall": {}, "reliability": []}

    monkeypatch.setattr(wf_compare.walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(wf_compare.metrics_utils, "reliability_ece", lambda _bins: 0.0)

    row = wf_compare._run_one(
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
        early_stopping_rounds=1,
        include_quantiles=False,
        checkpoint_dir=tmp_path,
        resume=False,
    )

    assert captured == {"checkpoint_dir": tmp_path, "resume": False}
    assert row["label"] == "candidate"
