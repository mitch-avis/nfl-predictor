"""The feature groups a run leaves out by default, by whether it uses game lines.

A run without game lines (the weekly run's ``--wf-market-mode none``; ``backtest`` or ``train``
with no market feature, no anchor and no market transform) leaves out the reviewed list of the
walk-forwards without game lines by default; every other run leaves out production's groups. An
explicit list always wins. Production's own resolution, with the shipped config and the default
market mode, is pinned exactly.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from nfl_predictor import constants
from nfl_predictor.cli import backtest, options, train
from nfl_predictor.ml import artifacts, feature_spec, floor_sigma, walk_forward
from nfl_predictor.ml.floor_sigma import ErrorPool
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars.finalize import build_final_column_order
from nfl_predictor.weekly_run import config, pipeline

if TYPE_CHECKING:
    from pathlib import Path

PRODUCTION_GROUPS = ("strength", "own_elo", "own_elo_wt", "qb_adj")
# The reviewed list of the walk-forwards without game lines, plus the unmeasured qb_adj.
NO_GAME_LINES_GROUPS = ("strength", "elo", "own_elo_wt", "qb_adj")
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


# The constant and the shared resolution.


def test_the_no_game_lines_list_is_the_reviewed_list_plus_qb_adj() -> None:
    # Act
    groups = constants.NO_GAME_LINES_DISABLED_FEATURE_GROUPS

    # Assert
    assert groups == NO_GAME_LINES_GROUPS


@pytest.mark.parametrize(
    ("include_market", "market_anchor", "market_transform", "expected"),
    [
        (False, False, False, True),
        (False, False, None, False),
        (False, False, True, False),
        (False, True, False, False),
        (True, False, False, False),
        (True, True, None, False),
    ],
    ids=["none", "transform-unset", "transform-on", "anchor-on", "features-on", "production"],
)
def test_a_run_uses_no_game_lines_only_with_all_three_off(
    *, include_market: bool, market_anchor: bool, market_transform: bool | None, expected: bool
) -> None:
    # Act
    without_game_lines = options.is_run_without_game_lines(
        include_market=include_market,
        market_anchor=market_anchor,
        market_transform=market_transform,
    )

    # Assert
    assert without_game_lines is expected


@pytest.mark.parametrize(
    ("explicit", "without_game_lines", "expected"),
    [
        (None, True, NO_GAME_LINES_GROUPS),
        (None, False, PRODUCTION_GROUPS),
        ((), True, ()),
        (("elo",), True, ("elo",)),
        (("elo",), False, ("elo",)),
    ],
    ids=["no-lines-default", "production-default", "explicit-empty", "explicit", "explicit-prod"],
)
def test_an_explicit_list_wins_over_the_mode_default(
    *, explicit: tuple[str, ...] | None, without_game_lines: bool, expected: tuple[str, ...]
) -> None:
    # Act
    groups = options.resolve_disabled_feature_groups(
        explicit, without_game_lines=without_game_lines
    )

    # Assert
    assert groups == expected


# backtest and train.


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (NO_GAME_LINES, NO_GAME_LINES_GROUPS),
        ((*NO_GAME_LINES, "--disable-feature-groups", "elo,own_elo"), ("elo", "own_elo")),
        ((*NO_GAME_LINES, "--disable-feature-groups", ""), ()),
        (("--exclude-market",), PRODUCTION_GROUPS),
        (("--exclude-market", "--no-market-anchor"), PRODUCTION_GROUPS),
    ],
    ids=["no-lines", "explicit", "explicit-empty", "anchor-on", "transform-unset"],
)
def test_a_backtest_resolves_its_groups_by_mode(
    monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    # Act
    groups = _backtest_groups(monkeypatch, *argv)

    # Assert
    assert groups == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (NO_GAME_LINES, NO_GAME_LINES_GROUPS),
        ((*NO_GAME_LINES, "--disable-feature-groups", "elo,own_elo"), ("elo", "own_elo")),
        ((*NO_GAME_LINES, "--disable-feature-groups", ""), ()),
        (("--exclude-market", "--market-anchor"), PRODUCTION_GROUPS),
        (("--exclude-market",), NO_GAME_LINES_GROUPS),
    ],
    ids=["no-lines", "explicit", "explicit-empty", "anchor-on", "anchor-and-transform-off"],
)
def test_train_resolves_its_groups_by_mode(
    monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    # Act
    groups = _train_groups(monkeypatch, *argv)

    # Assert
    assert groups == expected


def test_a_backtest_without_game_lines_records_the_resolved_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["nfl-predictor backtest", *NO_GAME_LINES, *WEEKLY_PINNED])
    args = backtest._parse_args()
    _, ablation = backtest._apply_ablations(pd.DataFrame({"season": [2024]}), args)
    wf_config = backtest._walk_forward_config(args, ablation.disabled_groups)

    # Act
    payload = backtest._config_payload(args, wf_config, {}, ablation, {})

    # Assert
    assert payload["disabled_feature_groups"] == list(NO_GAME_LINES_GROUPS)
    assert wf_config.disabled_feature_groups == NO_GAME_LINES_GROUPS


def test_train_without_game_lines_trains_and_records_the_resolved_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["nfl-predictor train", *NO_GAME_LINES])
    monkeypatch.setattr(floor_sigma, "load_reference_pool", lambda _runs: None)
    args = train._parse_args()

    # Act
    training = train._training_options(args, train._optuna_config(args))

    # Assert
    assert training.disabled_feature_groups == NO_GAME_LINES_GROUPS
    assert (training.include_market, training.market_anchor, training.market_transform) == (
        False,
        False,
        False,
    )


def _ignored_list_notes(monkeypatch: pytest.MonkeyPatch, *argv: str) -> list[str]:
    """Predict from a stub checkpoint with ``argv``; return the "ignoring" notes logged."""
    monkeypatch.setattr(sys, "argv", ["nfl-predictor train", "--model-in", "model.joblib", *argv])
    monkeypatch.setattr(train, "load_model_checkpoint", lambda _path, _kind: object())
    infos: list[str] = []
    monkeypatch.setattr(log, "info", lambda message, *args: infos.append(message % args))
    train._predict_from_checkpoint(train._parse_args(), None)
    return [message for message in infos if "ignoring --disable-feature-groups" in message]


@pytest.mark.parametrize(
    ("argv", "noted"),
    [
        ((), False),
        (NO_GAME_LINES, False),
        ((*NO_GAME_LINES, "--disable-feature-groups", "elo"), True),
        (("--disable-feature-groups", ",".join(NO_GAME_LINES_GROUPS)), True),
    ],
    ids=["production-default", "no-lines-default", "no-lines-explicit", "production-explicit"],
)
def test_train_notes_an_ignored_list_only_when_it_is_not_the_runs_default(
    monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...], *, noted: bool
) -> None:
    # Act
    notes = _ignored_list_notes(monkeypatch, *argv)

    # Assert
    assert bool(notes) is noted


def _win_total_notes(monkeypatch: pytest.MonkeyPatch, *argv: str) -> list[str]:
    """Return the "No game lines" notes of a backtest feature spec built with ``argv``."""
    order = build_final_column_order()
    games = pd.DataFrame(np.random.default_rng(0).normal(size=(40, len(order))), columns=order)
    monkeypatch.setattr(sys, "argv", ["nfl-predictor backtest", *argv])
    kept, _ = backtest._apply_ablations(games, backtest._parse_args())
    monkeypatch.setattr(feature_spec, "_WARNED_MARKET_INFORMED", set())
    infos: list[str] = []
    monkeypatch.setattr(log, "info", lambda message, *args: infos.append(message % args))
    feature_spec.build_feature_spec(
        kept,
        feature_spec.FeatureSelection(
            include_market=False,
            max_cardinality_ratio=walk_forward.WalkForwardConfig().max_cardinality_ratio,
        ),
    )
    return [message for message in infos if message.startswith("No game lines:")]


def test_the_win_total_note_fires_when_an_explicit_list_keeps_the_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Act
    notes = _win_total_notes(monkeypatch, *NO_GAME_LINES, "--disable-feature-groups", "strength")

    # Assert
    assert len(notes) == 1
    assert "['elo', 'own_elo_wt']" in notes[0]


def test_the_win_total_note_stays_quiet_with_the_no_game_lines_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Act
    notes = _win_total_notes(monkeypatch, *NO_GAME_LINES)

    # Assert
    assert notes == []


# The weekly run.


def _write_config(tmp_path: Path, settings: dict[str, Any]) -> str:
    """Write a weekly config file holding only ``settings``; return its path."""
    path = tmp_path / "weekly_run.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return str(path)


def test_the_shipped_config_pins_no_group_list() -> None:
    # Act
    shipped = config._load_config(config.DEFAULT_CONFIG_PATH)

    # Assert
    assert "disable_feature_groups" not in shipped


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("--wf-market-mode", "none"), NO_GAME_LINES_GROUPS),
        (("--wf-market-mode", "none", "--disable-feature-groups", "elo"), ("elo",)),
        (("--wf-market-mode", "none", "--disable-feature-groups", ""), ()),
        (("--wf-market-mode", "anchor"), PRODUCTION_GROUPS),
        (("--wf-market-mode", "features"), PRODUCTION_GROUPS),
    ],
    ids=["no-lines", "explicit", "explicit-empty", "anchor", "features"],
)
def test_the_weekly_run_with_the_shipped_config_resolves_its_groups_by_mode(
    argv: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    # Act
    args = config.parse_args(argv)

    # Assert
    assert tuple(args.disable_feature_groups) == expected


@pytest.mark.parametrize(
    ("settings", "argv", "expected"),
    [
        ({"disable_feature_groups": ["elo"]}, ("--wf-market-mode", "none"), ("elo",)),
        ({"disable_feature_groups": []}, ("--wf-market-mode", "none"), ()),
        ({"wf_market_mode": "none"}, (), NO_GAME_LINES_GROUPS),
        ({"wf_market_mode": "none", "disable_feature_groups": ["elo"]}, (), ("elo",)),
        ({}, (), PRODUCTION_GROUPS),
    ],
    ids=["file-list", "file-empty", "file-mode", "file-mode-and-list", "empty-file"],
)
def test_a_weekly_config_file_list_wins_over_the_mode_default(
    tmp_path: Path, settings: dict[str, Any], argv: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    # Arrange
    path = _write_config(tmp_path, settings)

    # Act
    args = config.parse_args(["--config", path, *argv])

    # Assert
    assert tuple(args.disable_feature_groups) == expected


def test_the_weekly_run_without_game_lines_walks_forward_and_fits_without_the_list() -> None:
    # Arrange
    args = config.parse_args(["--wf-market-mode", "none", *WEEKLY_PINNED])
    frame = pd.DataFrame({"season": [2025], "home_spread": [-3.0], "total_line": [44.5]})

    # Act
    stage1_options = config.stage1_options(args)
    _, train_config = config.final_fit_options(args, frame, run_dir=None)

    # Assert
    assert stage1_options.disabled_feature_groups == NO_GAME_LINES_GROUPS
    assert train_config["disabled_feature_groups"] == list(NO_GAME_LINES_GROUPS)
