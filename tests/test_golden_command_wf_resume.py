"""Tests for the golden command's walk-forward checkpoint wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_predictor.ml import walk_forward
from scripts import golden_command


class _StageReachedError(Exception):
    """Raised by a stub to stop the command once the walk-forward stage is reached."""


def test_parse_args_resumes_walk_forward_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk-forward stage resumes from saved weeks unless told not to."""
    monkeypatch.setattr(sys, "argv", ["golden_command.py"])
    defaults = golden_command._parse_args()
    assert defaults.wf_resume is True
    assert defaults.wf_checkpoint_dir == walk_forward.DEFAULT_CHECKPOINT_DIR

    monkeypatch.setattr(sys, "argv", ["golden_command.py", "--no-wf-resume"])
    assert golden_command._parse_args().wf_resume is False


def test_walk_forward_stage_passes_checkpoint_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walk-forward stage hands its checkpoint settings to the engine."""
    data_path = tmp_path / "data.csv"
    data_path.write_text("season,week\n2024,1\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(
        _df: pd.DataFrame, _config: walk_forward.WalkForwardConfig, **kwargs: object
    ) -> dict[str, object]:
        """Record the keyword arguments, then stop the command."""
        captured.update(kwargs)
        raise _StageReachedError

    monkeypatch.setattr(golden_command.constants, "ROOT_DIR", str(tmp_path))
    monkeypatch.setattr(golden_command.walk_forward, "load_games", lambda _path: pd.DataFrame())
    monkeypatch.setattr(golden_command.walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "golden_command.py",
            "--data-path",
            str(data_path),
            "--run-id",
            "golden_resume_test",
            "--no-wf-resume",
            "--wf-checkpoint-dir",
            str(tmp_path / "checkpoints"),
        ],
    )

    with pytest.raises(_StageReachedError):
        golden_command.main()

    assert captured == {"checkpoint_dir": tmp_path / "checkpoints", "resume": False}
