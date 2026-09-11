"""Tests for walk_forward_backtest script helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_predictor.ml import walk_forward
from scripts import walk_forward_backtest


def test_trend_feature_columns_collects_trend_and_phase_fields() -> None:
    """Trend ablation drops trend and season-phase columns only."""
    df = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2023],
            "week": [1],
            "week_in_season_norm": [0.1],
            "season_phase_early": [1],
            "season_phase_mid": [0],
            "season_phase_late": [0],
            "away_elo_4wk_trend": [0.0],
            "home_qb_value_4wk_trend": [0.0],
            "last_5_games_rating_trend_diff": [0.0],
            "away_total_yards": [300.0],
            "home_scoring_margin": [7.0],
        }
    )

    dropped = walk_forward_backtest._trend_feature_columns(df)

    assert set(dropped) == {
        "week_in_season_norm",
        "season_phase_early",
        "season_phase_mid",
        "season_phase_late",
        "away_elo_4wk_trend",
        "home_qb_value_4wk_trend",
        "last_5_games_rating_trend_diff",
    }


def test_resume_is_on_by_default_and_uses_the_shared_checkpoint_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running an identical command picks up saved weeks unless told not to."""
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])
    defaults = walk_forward_backtest._parse_args()
    assert defaults.resume is True
    assert defaults.checkpoint_dir == walk_forward.DEFAULT_CHECKPOINT_DIR

    monkeypatch.setattr(
        sys,
        "argv",
        ["walk_forward_backtest.py", "--no-resume", "--checkpoint-dir", "models/elsewhere"],
    )
    overridden = walk_forward_backtest._parse_args()
    assert overridden.resume is False
    assert overridden.checkpoint_dir == Path("models/elsewhere")


def test_main_passes_checkpoint_settings_and_records_the_restore_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI hands its checkpoint settings to the engine and reports what was restored."""
    captured: dict[str, object] = {}
    checkpoint = {"dir": "somewhere", "restored_folds": 3, "computed_folds": 1}

    def fake_run(
        _df: pd.DataFrame, _config: walk_forward.WalkForwardConfig, **kwargs: object
    ) -> dict[str, object]:
        """Record the keyword arguments and return a minimal result."""
        captured.update(kwargs)
        return {"checkpoint": checkpoint, "per_week": []}

    def fake_report(
        _run_id: str, _created_at: str, payload: dict[str, object], _results: object
    ) -> dict[str, object]:
        """Return the config payload so the test can read it back."""
        return {"config": payload}

    def fake_metadata(
        _created_at: str, _hash: str, payload: dict[str, object]
    ) -> dict[str, object]:
        """Return a minimal metadata payload."""
        return {"config": payload}

    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame({"season": [2023]}))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", lambda _path: "hash")
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(walk_forward, "build_metrics_report", fake_report)
    monkeypatch.setattr(walk_forward, "build_metadata", fake_metadata)
    out_json = tmp_path / "metrics_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "walk_forward_backtest.py",
            "--no-resume",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--out-json",
            str(out_json),
        ],
    )

    walk_forward_backtest.main()

    assert captured == {"checkpoint_dir": tmp_path / "checkpoints", "resume": False}
    assert json.loads(out_json.read_text())["config"]["checkpoint"] == checkpoint


def test_disable_feature_groups_arg_parses_comma_separated_list() -> None:
    """`--disable-feature-groups pbp,other` parses into the expected stripped tuple."""
    old_argv = sys.argv
    try:
        sys.argv = [
            "walk_forward_backtest.py",
            "--disable-feature-groups",
            "pbp,other",
        ]
        args = walk_forward_backtest._parse_args()
    finally:
        sys.argv = old_argv

    assert args.disable_feature_groups == "pbp,other"
    assert walk_forward_backtest._parse_feature_groups(args.disable_feature_groups) == (
        "pbp",
        "other",
    )


def test_parse_feature_groups_strips_whitespace_and_drops_empty_entries() -> None:
    """Parsing tolerates surrounding whitespace, empty segments, and a missing/empty value."""
    assert walk_forward_backtest._parse_feature_groups(" pbp , other ,") == ("pbp", "other")
    assert walk_forward_backtest._parse_feature_groups(None) == ()
    assert walk_forward_backtest._parse_feature_groups("") == ()


def test_disable_feature_groups_drops_resolved_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved feature-group columns are the ones dropped from the loaded dataframe."""
    monkeypatch.setattr(
        walk_forward.constants,
        "FEATURE_GROUP_COLUMN_MARKERS",
        {"pbp": ("epa_per_play",)},
    )
    df = pd.DataFrame(
        {
            "game_id": [1],
            "season": [2023],
            "week": [1],
            "away_epa_per_play": [0.1],
            "home_epa_per_play": [0.2],
            "epa_per_play_diff": [-0.1],
            "away_total_yards": [300.0],
        }
    )
    groups = walk_forward_backtest._parse_feature_groups("pbp")

    dropped = walk_forward.resolve_feature_group_columns(list(df.columns), groups)
    remaining = df.drop(columns=dropped)

    assert dropped == [
        "away_epa_per_play",
        "epa_per_play_diff",
        "home_epa_per_play",
    ]
    assert set(remaining.columns) == {"game_id", "season", "week", "away_total_yards"}


def test_disable_feature_groups_unknown_group_raises_from_cli_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown feature group name raises a clear error, not an obscure library traceback."""
    monkeypatch.setattr(walk_forward.constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ("epa",)})
    groups = walk_forward_backtest._parse_feature_groups("not_a_real_group")

    with pytest.raises(ValueError, match="not_a_real_group"):
        walk_forward.resolve_feature_group_columns(["away_epa"], groups)
