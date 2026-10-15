"""The shared resolver of a feature group's columns warns when a known group matches none.

A group that matches no column (production asking for ``own_elo`` while ``data/`` is a build
without own-Elo columns, say) leaves out nothing, so the run trains differently from an arm that
did leave those columns out. ``feature_spec.resolve_feature_group_columns`` is the one place every
fit resolves groups, so its warning reaches the backtest, ``train`` and both weekly stages.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import pytest
from tests.feature_group_fixture import build_group_fixture, write_group_weekly_config
from tests.weekly_fixture import build_fixture

from nfl_predictor import constants
from nfl_predictor.cli import backtest
from nfl_predictor.ml import feature_spec, walk_forward
from nfl_predictor.weekly_run import pipeline

if TYPE_CHECKING:
    from pathlib import Path

COLUMNS = ["away_rest", "away_elo_pre", "home_elo_pre", "elo_pre_diff", "home_moneyline"]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the warning messages logged so far."""
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


def test_a_known_group_that_matches_no_column_logs_a_warning_naming_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)

    # Act
    resolved = feature_spec.resolve_feature_group_columns(COLUMNS, ["elo", "own_elo"])

    # Assert
    assert resolved == ["away_elo_pre", "elo_pre_diff", "home_elo_pre"]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "own_elo" in warnings[0]
    assert "'elo'" not in warnings[0]


def test_groups_that_all_match_columns_log_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    # Arrange
    caplog.set_level(logging.INFO)

    # Act
    feature_spec.resolve_feature_group_columns(COLUMNS, ["elo"])

    # Assert
    assert _warnings(caplog) == []


def test_a_backtest_whose_groups_match_columns_logs_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    paths = build_group_fixture(tmp_path)
    monkeypatch.setattr(walk_forward, "BOOTSTRAP_SAMPLES", 50)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nfl-predictor backtest",
            "--data-path",
            str(paths["completed"]),
            "--eval-last-n-seasons",
            "1",
            "--wf-start-week",
            "4",
            "--n-estimators",
            "5",
            "--xgb-n-jobs",
            "1",
            "--xgb-device",
            "cpu",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--out-json",
            str(tmp_path / "wf" / "metrics_report.json"),
            "--disable-feature-groups",
            "elo,own_elo_wt",
        ],
    )

    # Act
    backtest.main()

    # Assert
    assert [message for message in _warnings(caplog) if "match no column" in message] == []


class _StopAfterStage1Error(Exception):
    """Stop the weekly run when the final fit starts."""


def _stop_at_the_final_fit(_options: object) -> None:
    raise _StopAfterStage1Error


def test_stage1_warns_about_a_group_the_data_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.INFO)
    paths = build_fixture(tmp_path)
    config_path = write_group_weekly_config(
        tmp_path, paths, {"disable_feature_groups": ["own_elo"]}
    )
    monkeypatch.setattr(walk_forward, "BOOTSTRAP_SAMPLES", 50)
    monkeypatch.setattr(pipeline, "train_margin_total_model_with_report", _stop_at_the_final_fit)
    monkeypatch.setattr(sys, "argv", ["nfl-predictor weekly", "--config", str(config_path)])

    # Act & Assert
    with pytest.raises(_StopAfterStage1Error):
        pipeline.main()
    unmatched = [message for message in _warnings(caplog) if "match no column" in message]
    assert len(unmatched) == 1
    assert "own_elo" in unmatched[0]


# Columns no feature group may match. The final fit resolves its market settings on the full
# header and stage 1 on the frame after the drop, so a group that matched a line column would
# split the two weekly stages; a target, a feature-range boundary or a key column every fit reads
# by name would break every run that disables the group.
PROTECTED_COLUMNS = (
    *constants.LINES_COLUMNS,
    *feature_spec.MARKET_DERIVED_COLUMNS,
    *constants.RESULT_COLUMNS,
    feature_spec.DEFAULT_FEATURE_START_COLUMN,
    feature_spec.DEFAULT_FEATURE_END_COLUMN,
    "season",
    "week",
    "game_type",
    "game_id",
)


@pytest.mark.parametrize("group", sorted(constants.FEATURE_GROUP_COLUMN_MARKERS))
def test_no_group_matches_a_market_target_boundary_or_key_column(group: str) -> None:
    # Act
    matched = feature_spec.resolve_feature_group_columns(
        PROTECTED_COLUMNS, [group], warn_unmatched=False
    )

    # Assert
    assert matched == []
