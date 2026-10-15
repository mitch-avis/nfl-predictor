"""The feature groups a run leaves out by default, by whether it uses game lines.

A run without game lines (the weekly run's ``--wf-market-mode none``; ``backtest`` or ``train``
with no market feature, no anchor and no market transform) leaves out the reviewed list of the
step-6 runs without game lines by default; every other run leaves out production's groups. An
explicit list always wins. Production's own resolution, with the shipped config and the default
market mode, is pinned exactly.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import pandas as pd

from nfl_predictor.cli import backtest, train
from nfl_predictor.ml import artifacts
from nfl_predictor.ml.floor_sigma import ErrorPool
from nfl_predictor.weekly_run import config, pipeline

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

PRODUCTION_GROUPS = ("strength", "own_elo", "own_elo_wt", "qb_adj")
NO_GAME_LINES = ("--exclude-market", "--no-market-anchor", "--no-market-transform")
# Fixed thread count, so the stage settings do not depend on the machine's cores.
WEEKLY_PINNED = ("--xgb-device", "cpu", "--xgb-n-jobs", "4")


def _backtest_groups(monkeypatch: pytest.MonkeyPatch, *argv: str) -> tuple[str, ...]:
    """Return the groups a backtest with ``argv`` leaves out, as its ablation resolves them."""
    monkeypatch.setattr(sys, "argv", ["nfl-predictor backtest", *argv])
    args = backtest._parse_args()
    return backtest._apply_ablations(pd.DataFrame({"season": [2024]}), args)[1].disabled_groups


def _train_groups(monkeypatch: pytest.MonkeyPatch, *argv: str) -> tuple[str, ...]:
    """Return the groups ``nfl-predictor train`` with ``argv`` leaves out."""
    monkeypatch.setattr(sys, "argv", ["nfl-predictor train", *argv])
    return tuple(train._parse_args().disable_feature_groups)


def _stage1_settings(args: Any) -> dict[str, Any]:
    """Return stage 1's settings as the weekly run hashes them for its reuse marker."""
    return {
        **asdict(config.stage1_options(args)),
        "checkpoint_per_fold": bool(args.wf_checkpoint_per_fold),
        "floor_sigma_history": ErrorPool().digest(),
    }


# Production is unchanged: the shipped config, the default market mode and no list given.


def test_production_weekly_stage1_options_are_unchanged() -> None:
    # Arrange
    args = config.parse_args(WEEKLY_PINNED)

    # Act
    options = asdict(config.stage1_options(args))

    # Assert
    assert options == {
        "eval_last_n_seasons": 3,
        "wf_start_week": 1,
        "include_postseason": False,
        "exclude_incomplete_seasons": False,
        "recency_half_life_seasons": None,
        "market_mode": "hybrid",
        "xgb_params_overrides": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.0165,
            "n_jobs": 4,
            "verbosity": 0,
            "device": "cpu",
        },
        "include_quantiles": False,
        "market_transform": None,
        "max_cardinality_ratio": 0.5,
        "disabled_feature_groups": PRODUCTION_GROUPS,
    }


def test_production_weekly_stage1_hash_is_unchanged() -> None:
    # Arrange
    args = config.parse_args(WEEKLY_PINNED)

    # Act
    stage_hash = artifacts.stable_short_hash(pipeline._marker_settings(_stage1_settings(args)))

    # Assert
    assert stage_hash == "ef442e1e"


def test_production_weekly_run_fingerprint_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(artifacts, "git_commit_hash", lambda: "commit")
    args = config.parse_args(WEEKLY_PINNED)
    context = pipeline._RunContext(
        args=args,
        run_id="run",
        run_dir=tmp_path,
        output_dir=tmp_path,
        created_at="now",
        dataset_hash="hash",
        dataset_fingerprint={"sha256": "dataset"},
        config_payload={},
    )

    # Act
    fingerprint = pipeline._wf_run_fingerprint(context, config.stage1_options(args), ErrorPool())

    # Assert
    assert fingerprint == ("d91893454a70a6d65f0ebbf4f4bc9f188f4835d4abd54ddd1ff93870f9a1d452")


def test_production_weekly_final_fit_records_productions_groups() -> None:
    # Arrange
    args = config.parse_args(WEEKLY_PINNED)
    frame = pd.DataFrame({"season": [2025], "home_spread": [-3.0], "total_line": [44.5]})

    # Act
    _, train_config = config.final_fit_options(args, frame, run_dir=None)

    # Assert
    assert train_config["disabled_feature_groups"] == list(PRODUCTION_GROUPS)


def test_a_backtest_with_no_flags_leaves_out_productions_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Act
    groups = _backtest_groups(monkeypatch)

    # Assert
    assert groups == PRODUCTION_GROUPS


def test_train_with_no_flags_leaves_out_productions_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Act
    groups = _train_groups(monkeypatch)

    # Assert
    assert groups == PRODUCTION_GROUPS
