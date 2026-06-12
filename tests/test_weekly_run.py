"""Unit tests for the weekly_run orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import weekly_run


def test_load_config_json(tmp_path: Path) -> None:
    """JSON configs should load into a dict."""
    config_path = tmp_path / "weekly.json"
    config_path.write_text(json.dumps({"wf_eval_last_n_seasons": 4}), encoding="utf-8")
    payload = weekly_run._load_config(config_path)
    assert payload["wf_eval_last_n_seasons"] == 4


def test_load_config_yaml_optional(tmp_path: Path) -> None:
    """YAML configs should parse when PyYAML is available or raise a clear error."""
    config_path = tmp_path / "weekly.yaml"
    config_path.write_text("wf_eval_last_n_seasons: 3\n", encoding="utf-8")
    try:
        payload = weekly_run._load_config(config_path)
    except RuntimeError as exc:
        assert "PyYAML" in str(exc)
    else:
        assert payload["wf_eval_last_n_seasons"] == 3


def test_resolve_predict_path_prefers_latest_week(tmp_path: Path) -> None:
    """When predict_path is None, the newest week file should be chosen."""
    predict_dir = tmp_path / "predict"
    predict_dir.mkdir()
    week_01 = predict_dir / "week_01_games_to_predict.csv"
    week_10 = predict_dir / "week_10_games_to_predict.csv"
    week_01.write_text("season,week\n2025,1\n", encoding="utf-8")
    week_10.write_text("season,week\n2025,10\n", encoding="utf-8")

    resolved = weekly_run._resolve_predict_path(None, tmp_path)
    assert resolved == week_10


def test_build_confidence_picks_adds_winner_and_rank() -> None:
    """Confidence picks should include predicted winners and ranks."""
    df = pd.DataFrame(
        {
            "away_abbr": ["A", "B"],
            "home_abbr": ["C", "D"],
            "home_win_prob": [0.65, 0.45],
            "away_win_prob": [0.35, 0.55],
        }
    )
    picks = weekly_run._build_confidence_picks(df)
    assert "predicted_winner" in picks.columns
    assert "confidence_rank" in picks.columns
    assert picks["confidence_rank"].is_unique


def test_stage_marker_reuse(tmp_path: Path) -> None:
    """Stage markers should only reuse when hashes match and outputs exist."""
    marker = tmp_path / "stage_state.json"
    output_path = tmp_path / "output.csv"
    output_path.write_text("ok", encoding="utf-8")

    weekly_run._write_stage_marker(
        marker,
        dataset_hash="abc123",
        config_hash="def456",
        stage="wf_compare",
        extra={"outputs": [str(output_path)]},
    )
    assert weekly_run._stage_can_reuse(marker, "abc123", "def456", [output_path])
    assert not weekly_run._stage_can_reuse(marker, "abc123", "wrong", [output_path])
