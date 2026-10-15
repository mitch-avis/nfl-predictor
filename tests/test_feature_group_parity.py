"""Parity: a backtest and production leave out exactly the same columns for the same groups.

A walk-forward arm that disables feature groups can be adopted only if production then trains
on exactly the columns the arm trained on. Each test runs ``nfl-predictor backtest
--disable-feature-groups`` on the weekly fixture with four groups' columns and the ported
ratings added (``tests/feature_group_fixture.py``; every set leaves out the strength snapshot,
which production allows only beside the ported ratings) as the reference, then one production
entry point with the
same groups: the weekly run (``disable_feature_groups`` in its config; stage 1's walk-forward and
the final fit) or ``nfl-predictor train``. They must drop the same columns, the hand-written
columns of those groups, and train on the same feature list. The production runs use the
market settings the backtest resolves to (features transformed, the model anchored), so the
feature lists can be compared whole.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import joblib
import pytest
from tests.feature_group_fixture import (
    GROUP_COLUMNS,
    build_group_fixture,
    write_group_weekly_config,
)
from tests.weekly_fixture import CURRENT_SEASON

from nfl_predictor.cli import backtest, train
from nfl_predictor.ml import walk_forward
from nfl_predictor.reporting import production_settings
from nfl_predictor.weekly_run import config as weekly_config
from nfl_predictor.weekly_run import pipeline

if TYPE_CHECKING:
    from pathlib import Path

TEST_BOOTSTRAP_SAMPLES = 50
# The four combinations of groups the feature-group walk-forward arms disable, in the order the
# arms' launch commands name them.
GROUP_SETS = pytest.mark.parametrize(
    "groups",
    [
        ("strength", "own_elo", "own_elo_wt"),
        ("strength", "elo", "own_elo", "own_elo_wt"),
        ("strength", "elo", "own_elo"),
        ("strength", "elo", "own_elo_wt"),
    ],
    ids=[
        "keeps-elo",
        "keeps-no-elo",
        "keeps-win-total-elo",
        "keeps-market-free-own-elo",
    ],
)


def _expected_dropped(groups: tuple[str, ...]) -> list[str]:
    """Return the fixture's columns of ``groups``, sorted, as every run records them."""
    return sorted(column for group in groups for column in GROUP_COLUMNS[group])


def _run_backtest(
    root: Path, paths: dict[str, Path], groups: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Run the backtest over the current season without ``groups``; return its metadata."""
    out_json = root / "wf" / "metrics_report.json"
    monkeypatch.setattr(walk_forward, "BOOTSTRAP_SAMPLES", TEST_BOOTSTRAP_SAMPLES)
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
            "1",
            "--n-estimators",
            "5",
            "--xgb-n-jobs",
            "1",
            "--xgb-device",
            "cpu",
            "--checkpoint-dir",
            str(root / "checkpoints"),
            "--out-json",
            str(out_json),
            "--disable-feature-groups",
            ",".join(groups),
        ],
    )
    backtest.main()
    return json.loads((out_json.parent / "metadata.json").read_text(encoding="utf-8"))


def _metadata(run_dir: Path) -> dict[str, Any]:
    """Return a training run's ``metadata.json``."""
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def _first_stage1_fold_columns(run_dir: Path) -> list[str]:
    """Return the feature columns stage 1's first week trained on, from its checkpoint."""
    (path,) = sorted(run_dir.glob(f"wf_compare/wf_folds/*/fold_{CURRENT_SEASON}_w01.joblib"))
    return list(joblib.load(path)["feature_columns"])


@GROUP_SETS
def test_the_weekly_run_drops_the_columns_the_backtest_drops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, groups: tuple[str, ...]
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path, ratings=True)
    reference = _run_backtest(tmp_path, paths, groups, monkeypatch)
    config_path = write_group_weekly_config(
        tmp_path, paths, {"disable_feature_groups": list(groups)}
    )
    monkeypatch.setattr(walk_forward, "BOOTSTRAP_SAMPLES", TEST_BOOTSTRAP_SAMPLES)
    monkeypatch.setattr(sys, "argv", ["nfl-predictor weekly", "--config", str(config_path)])

    # Act
    status = pipeline.main()

    # Assert
    assert status == 0
    final_fit = _metadata(tmp_path / "run")
    assert reference["config"]["dropped_feature_group_columns"] == _expected_dropped(groups)
    assert final_fit["dropped_feature_group_columns"] == _expected_dropped(groups)
    assert final_fit["disabled_feature_groups"] == list(groups)
    assert final_fit["feature_list"] == reference["feature_list"]
    assert _first_stage1_fold_columns(tmp_path / "run") == reference["feature_list"]
    (candidate,) = sorted((tmp_path / "run" / "wf_compare").glob("wf_candidate_*.json"))
    stage1 = json.loads(candidate.read_text(encoding="utf-8"))
    assert stage1["config"]["disabled_feature_groups"] == list(groups)


@GROUP_SETS
def test_train_drops_the_columns_the_backtest_drops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, groups: tuple[str, ...]
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path, ratings=True)
    reference = _run_backtest(tmp_path, paths, groups, monkeypatch)
    run_dir = tmp_path / "train_run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nfl-predictor train",
            "--data-path",
            str(paths["completed"]),
            "--holdout-seasons",
            "0",
            "--market-transform",
            "--market-anchor",
            "--floor-sigma-reference-runs",
            "--xgb-device",
            "cpu",
            "--xgb-n-jobs",
            "1",
            "--run-dir",
            str(run_dir),
            "--disable-feature-groups",
            ",".join(groups),
        ],
    )

    # Act
    train.main()

    # Assert
    trained = _metadata(run_dir)
    assert (
        trained["dropped_feature_group_columns"]
        == reference["config"]["dropped_feature_group_columns"]
    )
    assert trained["feature_list"] == reference["feature_list"]


def test_a_backtest_and_production_with_the_same_groups_show_no_group_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    paths = build_group_fixture(tmp_path)
    config_path = write_group_weekly_config(
        tmp_path, paths, {"disable_feature_groups": ["own_elo_wt", "elo"]}
    )
    monkeypatch.setattr(weekly_config, "DEFAULT_CONFIG_PATH", config_path)

    # Act
    metadata = _run_backtest(tmp_path, paths, ("elo", "own_elo_wt"), monkeypatch)

    # Assert
    report = json.loads((tmp_path / "wf" / "metrics_report.json").read_text(encoding="utf-8"))
    section = report[production_settings.SECTION_KEY]
    assert section["production_config"] == str(config_path)
    for stage in ("stage1", "final_fit"):
        differing = [row["setting"] for row in section[stage]["differences"]]
        assert "disabled_feature_groups" not in differing
    assert metadata["config"]["disabled_feature_groups"] == ["elo", "own_elo_wt"]
