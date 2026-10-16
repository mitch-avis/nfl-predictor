"""Tests for runs that train with no market column at all, the market kept only as a yardstick."""

from __future__ import annotations

import itertools
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from nfl_predictor import constants
from nfl_predictor.cli import backtest, train
from nfl_predictor.ml import feature_spec, floor_sigma, ml_model_xgb_utils, walk_forward
from nfl_predictor.ml.feature_spec import (
    MARKET_DERIVED_COLUMNS,
    FeatureSelection,
    apply_feature_spec,
    build_feature_spec,
    build_preprocessor,
)
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.narrowing import as_list, as_str_dict
from nfl_predictor.utils.polars.finalize import build_final_column_order
from nfl_predictor.weekly_run import config as weekly_config
from nfl_predictor.weekly_run import stage1
from tests.fakes import returning

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

ROWS = 128
TEAMS = sorted(set(constants.ALIAS_TO_CANONICAL.values()))
STRING_FEATURES = {
    "stadium_surface": ("grass", "fieldturf", "a_turf"),
    "stadium_type": ("outdoors", "dome", "retractable"),
}
# Name tokens of a betting-line column: spreads, moneylines, totals, their odds and anything the
# market implies. A published column carrying one of these is treated as derived from lines.
BETTING_TOKENS = frozenset(
    {"spread", "spreads", "moneyline", "line", "odds", "market", "implied", "vig", "over", "under"}
)


def _line_derived(columns: list[str]) -> set[str]:
    """Return the columns derived from betting lines: the lines, their transforms, any named so."""
    named = {column for column in columns if BETTING_TOKENS & set(column.split("_"))}
    return {*constants.LINES_COLUMNS, *MARKET_DERIVED_COLUMNS} | named


def _schema_frame(*, derived_in_range: bool) -> pd.DataFrame:
    """Return a frame with every published column, typed the way the real dataset loads.

    With ``derived_in_range`` the four market transforms also sit inside the feature range, as
    they would in a dataset saved after the transforms ran.
    """
    rng = np.random.default_rng(11)
    order = build_final_column_order()
    if derived_in_range:
        start = order.index(walk_forward.WalkForwardConfig().feature_start) + 1
        order = [*order[:start], *MARKET_DERIVED_COLUMNS, *order[start:]]
    data: dict[str, object] = {}
    for column in order:
        if column.endswith("next_opponent_abbr"):
            data[column] = [TEAMS[int(i) % len(TEAMS)] for i in rng.integers(0, 10**6, ROWS)]
        elif column in STRING_FEATURES:
            choices = STRING_FEATURES[column]
            data[column] = [choices[int(i)] for i in rng.integers(0, len(choices), ROWS)]
        else:
            data[column] = rng.normal(size=ROWS)
    frame = pd.DataFrame(data)
    frame["season"] = 2024
    frame["week"] = 3
    frame["game_type"] = "REG"
    return frame


def _matrix_inputs(frame: pd.DataFrame, selection: FeatureSelection) -> list[str]:
    """Return the input columns behind every column of the encoded matrix a week fits on."""
    spec = build_feature_spec(frame, selection)
    preprocessor = build_preprocessor(spec, for_tree=True)
    ml_model_xgb_utils.fit_transform_matrix(preprocessor, apply_feature_spec(frame, spec))
    inputs: list[str] = []
    for _name, _transformer, columns in preprocessor.transformers_:
        column_list = as_list(columns)
        if column_list is not None:
            inputs.extend(str(column) for column in column_list)
    return inputs


@pytest.mark.parametrize("market_transform", [False, True], ids=["raw", "transformed"])
@pytest.mark.parametrize("derived_in_range", [False, True], ids=["build", "derived-in-range"])
def test_excluding_market_features_leaves_no_line_derived_column_in_the_matrix(
    *, market_transform: bool, derived_in_range: bool
) -> None:
    """With the market excluded, no column derived from betting lines reaches the model."""
    # Arrange
    frame = _schema_frame(derived_in_range=derived_in_range)
    selection = FeatureSelection(
        include_market=False,
        max_cardinality_ratio=walk_forward.WalkForwardConfig().max_cardinality_ratio,
        market_transform=market_transform,
    )

    # Act
    inputs = _matrix_inputs(frame, selection)

    # Assert
    assert inputs
    assert set(inputs) & _line_derived(list(frame.columns)) == set()


def test_including_market_features_keeps_the_transformed_lines() -> None:
    """The default selection still feeds the four market transforms, so the check above bites."""
    # Arrange
    frame = _schema_frame(derived_in_range=False)
    selection = FeatureSelection(
        include_market=True,
        max_cardinality_ratio=walk_forward.WalkForwardConfig().max_cardinality_ratio,
        market_transform=True,
    )

    # Act
    inputs = _matrix_inputs(frame, selection)

    # Assert
    assert set(inputs) & _line_derived(list(frame.columns)) == set(MARKET_DERIVED_COLUMNS)


def test_the_published_schema_names_no_line_column_beyond_the_five_lines() -> None:
    """Every published column named for the betting market is one of the five line columns."""
    # Arrange
    order = build_final_column_order()

    # Act
    named = {column for column in order if BETTING_TOKENS & set(column.split("_"))}

    # Assert
    assert named == set(constants.LINES_COLUMNS)


def _weekly_frame() -> pd.DataFrame:
    """Return two seasons of games over the default feature range, lines last, then scores."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(5)
    for season, week, game in itertools.product((2022, 2023), (1, 2, 3), range(4)):
        spread = float(rng.normal(0, 6))
        margin = -spread + float(rng.normal(0, 10))
        rows.append(
            {
                "game_id": f"{season}_{week:02d}_{game}",
                "season": season,
                "week": week,
                "game_type": "REG",
                "away_rest": float(rng.integers(6, 8)),
                "home_rest": float(rng.integers(6, 8)),
                "elo_pre_diff": float(rng.normal(0, 50)),
                "total_line": 44.0 + float(rng.normal(0, 3)),
                "away_spread": -spread,
                "home_spread": spread,
                "away_moneyline": 150 if spread < 0 else -170,
                "home_moneyline": -170 if spread < 0 else 150,
                "away_score": 20.0,
                "home_score": 20.0 + round(margin),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def market_free_results() -> dict[str, object]:
    """Run a small walk-forward with no market feature, no anchor and no transform."""
    config = walk_forward.WalkForwardConfig(
        eval_seasons=[2023],
        wf_start_week=2,
        random_seed=7,
        include_market=False,
        market_anchor=False,
        market_transform=False,
        include_quantiles=False,
        xgb_params_overrides={
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "device": "cpu",
        },
    )
    return walk_forward.run_walk_forward_backtest(_weekly_frame(), config)


def test_a_market_free_walk_forward_trains_on_no_line_column(
    market_free_results: dict[str, object],
) -> None:
    """The fitted feature list holds the non-market columns only."""
    # Act
    features = as_list(market_free_results["feature_list"])

    # Assert
    assert features is not None
    assert set(features) == {"away_rest", "home_rest", "elo_pre_diff"}


def test_a_market_free_walk_forward_still_scores_the_market_yardstick(
    market_free_results: dict[str, object],
) -> None:
    """The market-implied probability is scored beside the model, never fed to it."""
    # Act
    predictions = market_free_results["predictions"]

    # Assert
    assert isinstance(predictions, pd.DataFrame)
    assert predictions["market_home_win_prob"].notna().all()
    assert "market_baseline_margin" not in predictions.columns
    settings = market_free_results["resolved_settings"]
    assert isinstance(settings, dict)
    assert (settings["include_market"], settings["market_anchor"]) == (False, False)
    assert settings["market_transform"] is False
    overall = as_str_dict(market_free_results["overall"])
    assert overall is not None
    assert np.isfinite(overall["market_brier"])


def _backtest_args(monkeypatch: pytest.MonkeyPatch, *extra: str) -> argparse.Namespace:
    """Parse a backtest command line with the real parser."""
    monkeypatch.setattr(sys, "argv", ["nfl-predictor backtest", *extra])
    return backtest._parse_args()


def test_backtest_includes_market_features_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the option a backtest trains on the market features, as production does."""
    # Arrange
    args = _backtest_args(monkeypatch, "--xgb-device", "cpu")

    # Act
    config = backtest._walk_forward_config(args, ())

    # Assert
    assert config.include_market is True


def test_backtest_exclude_market_builds_a_market_free_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The market-free command line turns off the features, the anchor and the transform."""
    # Arrange
    args = _backtest_args(
        monkeypatch,
        "--exclude-market",
        "--no-market-anchor",
        "--no-market-transform",
        "--xgb-device",
        "cpu",
    )

    # Act
    config = backtest._walk_forward_config(args, ())

    # Assert
    assert (config.include_market, config.market_anchor, config.market_transform) == (
        False,
        False,
        False,
    )
    assert config.to_dict()["include_market"] is False


def test_weekly_market_mode_none_turns_off_the_features_and_the_anchor() -> None:
    """The weekly run's market-free mode trains with neither market features nor anchoring."""
    # Act
    flags = stage1.market_mode_flags("none")

    # Assert
    assert flags == (False, False)


def test_weekly_parser_accepts_market_mode_none(tmp_path: Path) -> None:
    """``--wf-market-mode none`` is a valid weekly option."""
    # Arrange
    empty_config = tmp_path / "weekly_run.yaml"
    empty_config.write_text("{}\n", encoding="utf-8")

    # Act
    args = weekly_config.parse_args(["--config", str(empty_config), "--wf-market-mode", "none"])

    # Assert
    assert args.wf_market_mode == "none"


def test_train_exclude_market_builds_market_free_training_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``nfl-predictor train --exclude-market`` trains market-free: no anchor or transform."""
    # Arrange
    monkeypatch.setattr(sys, "argv", ["nfl-predictor train", "--exclude-market"])
    monkeypatch.setattr(floor_sigma, "load_reference_pool", returning(None))
    args = train._parse_args()

    # Act
    options = train._training_options(args, train._optuna_config(args))

    # Assert
    assert (options.include_market, options.market_anchor, options.market_transform) == (
        False,
        False,
        False,
    )


def _weekly_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    """Parse a weekly command line over an empty config file."""
    empty_config = tmp_path / "weekly_run.yaml"
    empty_config.write_text("{}\n", encoding="utf-8")
    return weekly_config.parse_args(["--config", str(empty_config), *extra])


def test_weekly_market_mode_none_runs_stage_one_without_the_transform() -> None:
    """The market-free weekly walk-forward records the transform off, not resolved from lines."""
    # Arrange
    options = stage1.ProductionOptions(
        eval_last_n_seasons=3,
        wf_start_week=1,
        include_postseason=False,
        exclude_incomplete_seasons=False,
        recency_half_life_seasons=None,
        market_mode="none",
        xgb_params_overrides={"device": "cpu"},
        include_quantiles=False,
    )

    # Act
    config = stage1.production_walk_forward_config(options)

    # Assert
    assert (config.include_market, config.market_anchor, config.market_transform) == (
        False,
        False,
        False,
    )


def test_weekly_market_mode_none_trains_the_final_fit_without_the_transform(
    tmp_path: Path,
) -> None:
    """The market-free weekly final fit records the transform off on a dataset with lines."""
    # Arrange
    args = _weekly_args(tmp_path, "--wf-market-mode", "none")
    frame = _weekly_frame()

    # Act
    _optuna, train_config = weekly_config.final_fit_options(args, frame, run_dir=None)

    # Assert
    settings = (
        train_config["include_market"],
        train_config["market_anchor"],
        train_config["market_transform"],
    )
    assert settings == (False, False, False)


def test_weekly_market_mode_hybrid_keeps_the_transform_on(tmp_path: Path) -> None:
    """The default mode still trains on the transformed market features."""
    # Arrange
    args = _weekly_args(tmp_path, "--wf-market-mode", "hybrid")

    # Act
    _optuna, train_config = weekly_config.final_fit_options(args, _weekly_frame(), run_dir=None)

    # Assert
    assert train_config["market_transform"] is True


def _recorded_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the project logger's warnings."""
    warnings: list[str] = []

    def record(message: str, *args: object) -> None:
        warnings.append(message % args if args else message)

    monkeypatch.setattr(log, "warning", record)
    return warnings


@pytest.mark.parametrize(
    ("argv", "warned"),
    [
        (["--wf-market-mode", "none", "--market-transform"], True),
        (["--wf-market-mode", "none"], False),
        (["--wf-market-mode", "none", "--no-market-transform"], False),
        (["--wf-market-mode", "hybrid", "--market-transform"], False),
    ],
    ids=["none-with-transform", "none", "none-without-transform", "hybrid-with-transform"],
)
def test_weekly_warns_only_when_the_market_free_mode_overrides_a_requested_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], *, warned: bool
) -> None:
    """Asking for the transform in the market-free mode is overridden, and the run says so."""
    # Arrange
    warnings = _recorded_warnings(monkeypatch)

    # Act
    _weekly_args(tmp_path, *argv)

    # Assert
    overrides = [message for message in warnings if "--market-transform" in message]
    assert len(overrides) == (1 if warned else 0)


def test_the_overridden_transform_warning_says_no_game_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    warnings = _recorded_warnings(monkeypatch)
    expected = (
        "--wf-market-mode none overrides --market-transform (or market_transform: true): "
        "a run with no game lines trains with no market transform."
    )

    # Act
    _weekly_args(tmp_path, "--wf-market-mode", "none", "--market-transform")

    # Assert
    assert warnings == [expected]


def _recorded_infos(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the project logger's info messages."""
    infos: list[str] = []

    def record(message: str, *args: object) -> None:
        infos.append(message % args if args else message)

    monkeypatch.setattr(log, "info", record)
    return infos


def _spec_notes(
    monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, *, include_market: bool, builds: int = 1
) -> list[str]:
    """Build the feature spec ``builds`` times and return its "No game lines" notes.

    The notes are info messages; a warning mentioning the win-total priors is returned too, so a
    test can tell the levels apart.
    """
    warnings = _recorded_warnings(monkeypatch)
    infos = _recorded_infos(monkeypatch)
    monkeypatch.setattr(feature_spec, "_WARNED_MARKET_INFORMED", set[tuple[str, ...]]())
    selection = FeatureSelection(
        include_market=include_market,
        max_cardinality_ratio=walk_forward.WalkForwardConfig().max_cardinality_ratio,
    )
    for _ in range(builds):
        build_feature_spec(frame, selection)
    return [message for message in [*infos, *warnings] if "win-total" in message]


def test_a_run_without_game_lines_notes_the_win_total_groups_it_keeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    frame = _schema_frame(derived_in_range=False)

    # Act
    notes = _spec_notes(monkeypatch, frame, include_market=False)

    # Assert
    expected = (
        "No game lines: the model trains without betting-line columns but keeps the preseason "
        "win-total priors in the feature group(s) ['elo', 'own_elo_wt']; for a run with no "
        "market information at all, add elo,own_elo_wt to the groups left out: "
        "`--disable-feature-groups` (backtest, train; it replaces the default list) or "
        "`disable_feature_groups` in the weekly config."
    )
    assert notes == [expected]


def test_a_run_without_game_lines_logs_its_note_at_info_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    warnings = _recorded_warnings(monkeypatch)
    infos = _recorded_infos(monkeypatch)
    monkeypatch.setattr(feature_spec, "_WARNED_MARKET_INFORMED", set[tuple[str, ...]]())
    selection = FeatureSelection(
        include_market=False,
        max_cardinality_ratio=walk_forward.WalkForwardConfig().max_cardinality_ratio,
    )

    # Act
    build_feature_spec(_schema_frame(derived_in_range=False), selection)

    # Assert
    assert len([message for message in infos if message.startswith("No game lines:")]) == 1
    assert not [message for message in warnings if "own_elo_wt" in message]


def test_a_run_without_game_lines_notes_once_for_every_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    frame = _schema_frame(derived_in_range=False)

    # Act
    notes = _spec_notes(monkeypatch, frame, include_market=False, builds=3)

    # Assert
    assert len(notes) == 1


def test_a_run_that_leaves_one_group_out_names_only_the_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    frame = _schema_frame(derived_in_range=False)
    frame = frame.drop(
        columns=feature_spec.resolve_feature_group_columns(list(frame.columns), ["elo"])
    )

    # Act
    notes = _spec_notes(monkeypatch, frame, include_market=False)

    # Assert
    assert len(notes) == 1
    assert "['own_elo_wt']" in notes[0]


@pytest.mark.parametrize(
    ("emptied", "kept"),
    [(("own_elo_wt",), ["elo"]), (("elo", "own_elo_wt"), [])],
    ids=["own-elo-wt-empty", "both-empty"],
)
def test_a_market_informed_group_the_spec_drops_as_empty_goes_unnamed(
    monkeypatch: pytest.MonkeyPatch, emptied: tuple[str, ...], kept: list[str]
) -> None:
    """A group whose columns are all null leaves the features, so the note leaves it out."""
    # Arrange
    frame = _schema_frame(derived_in_range=False)
    empty = feature_spec.resolve_feature_group_columns(list(frame.columns), emptied)
    frame.loc[:, empty] = np.nan

    # Act
    notes = _spec_notes(monkeypatch, frame, include_market=False)

    # Assert
    assert len(notes) == (1 if kept else 0)
    assert all(f"feature group(s) {kept};" in message for message in notes)


@pytest.mark.parametrize(
    ("include_market", "dropped"),
    [(True, ()), (False, ("elo", "own_elo_wt"))],
    ids=["with-market", "groups-left-out"],
)
def test_no_warning_when_the_market_columns_stay_or_the_groups_are_left_out(
    monkeypatch: pytest.MonkeyPatch, *, include_market: bool, dropped: tuple[str, ...]
) -> None:
    # Arrange
    frame = _schema_frame(derived_in_range=False)
    if dropped:
        frame = frame.drop(
            columns=feature_spec.resolve_feature_group_columns(list(frame.columns), dropped)
        )

    # Act
    notes = _spec_notes(monkeypatch, frame, include_market=include_market)

    # Assert
    assert notes == []
