"""Tests for backtest script helpers."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from nfl_predictor import constants
from nfl_predictor.cli import backtest, options
from nfl_predictor.ml import floor_sigma, ml_model_xgb_utils, walk_forward
from nfl_predictor.ml.feature_spec import resolve_feature_group_columns
from nfl_predictor.reporting import run_comparison
from tests.fakes import returning

if TYPE_CHECKING:
    from collections.abc import Callable


def _loading(games: pd.DataFrame) -> Callable[[Path], pd.DataFrame]:
    """Return a ``load_games`` stand-in that hands out a fresh copy of ``games`` per call."""

    def load(_path: Path) -> pd.DataFrame:
        return games.copy()

    return load


def test_trend_feature_columns_collects_trend_and_phase_fields() -> None:
    """Trend ablation drops trend and season-phase columns only."""
    # Arrange
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

    # Act
    dropped = backtest._trend_feature_columns(df)

    # Assert
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
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    # Act
    defaults = backtest._parse_args()

    # Assert
    assert defaults.resume is True
    assert defaults.checkpoint_dir == walk_forward.default_checkpoint_dir()


def test_the_default_leaves_out_the_groups_production_leaves_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A walk-forward measures production: the strength snapshot and both own-Elo presets."""
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    # Act
    defaults = backtest._parse_args()

    # Assert
    assert options.parse_feature_groups(defaults.disable_feature_groups) == (
        "strength",
        "own_elo",
        "own_elo_wt",
    )


def test_an_empty_group_list_leaves_no_group_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--disable-feature-groups ''`` keeps every feature group."""
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py", "--disable-feature-groups", ""])

    # Act
    args = backtest._parse_args()

    # Assert
    assert options.parse_feature_groups(args.disable_feature_groups) == ()


def test_no_resume_and_another_checkpoint_dir_can_be_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--no-resume`` retrains every week and ``--checkpoint-dir`` moves the checkpoints."""
    # Arrange
    monkeypatch.setattr(
        sys,
        "argv",
        ["walk_forward_backtest.py", "--no-resume", "--checkpoint-dir", "models/elsewhere"],
    )

    # Act
    overridden = backtest._parse_args()

    # Assert
    assert overridden.resume is False
    assert overridden.checkpoint_dir == Path("models/elsewhere")


def test_main_passes_checkpoint_settings_and_records_the_restore_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI hands its checkpoint settings to the engine and reports what was restored."""
    # Arrange
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

    monkeypatch.setattr(walk_forward, "load_games", _loading(pd.DataFrame({"season": [2023]})))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", returning("hash"))
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

    # Act
    backtest.main()

    # Assert
    assert captured == {
        "checkpoints": walk_forward.FoldCheckpoints(tmp_path / "checkpoints", resume=False),
        "floor_sigma_history": None,
    }
    assert json.loads(out_json.read_text())["config"]["checkpoint"] == checkpoint


def test_xgb_device_defaults_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk-forward picks the GPU when one is usable unless told otherwise."""
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    # Act
    args = backtest._parse_args()

    # Assert
    assert args.xgb_device == "auto"


def test_xgb_device_typo_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelled device stops at the parser instead of reaching XGBoost."""
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py", "--xgb-device", "cdua"])

    # Act & Assert
    with pytest.raises(SystemExit):
        backtest._parse_args()


@pytest.mark.parametrize(
    ("extra_argv", "usable", "expected"),
    [([], True, "cuda"), ([], False, "cpu"), (["--xgb-device", "cpu"], True, "cpu")],
)
def test_main_runs_and_records_the_resolved_xgb_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_argv: list[str],
    *,
    usable: bool,
    expected: str,
) -> None:
    """The engine gets the concrete device, and the run's metadata records it."""
    # Arrange
    captured: list[walk_forward.WalkForwardConfig] = []

    def fake_run(
        _df: pd.DataFrame, config: walk_forward.WalkForwardConfig, **_kwargs: object
    ) -> dict[str, object]:
        """Record the config the CLI built and return a minimal result."""
        captured.append(config)
        return {"checkpoint": {}, "per_week": []}

    def fake_report(
        _run_id: str, _created_at: str, payload: dict[str, object], _results: object
    ) -> dict[str, object]:
        """Return the config payload unchanged."""
        return {"config": payload}

    def fake_metadata(
        _created_at: str, _hash: str, payload: dict[str, object]
    ) -> dict[str, object]:
        """Return the config payload unchanged."""
        return {"config": payload}

    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: usable)
    monkeypatch.setattr(walk_forward, "load_games", _loading(pd.DataFrame({"season": [2023]})))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", returning("hash"))
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(walk_forward, "build_metrics_report", fake_report)
    monkeypatch.setattr(walk_forward, "build_metadata", fake_metadata)
    out_json = tmp_path / "metrics_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "walk_forward_backtest.py",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--out-json",
            str(out_json),
            *extra_argv,
        ],
    )

    # Act
    backtest.main()

    # Assert
    assert (captured[0].xgb_params_overrides or {})["device"] == expected
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["config"]["xgb_device"] == expected
    assert metadata["config"]["xgb_params_overrides"]["device"] == expected


def test_disable_feature_groups_arg_parses_comma_separated_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--disable-feature-groups pbp,other` parses into the expected stripped tuple."""
    # Arrange
    monkeypatch.setattr(
        sys, "argv", ["walk_forward_backtest.py", "--disable-feature-groups", "pbp,other"]
    )

    # Act
    args = backtest._parse_args()

    # Assert
    assert args.disable_feature_groups == "pbp,other"
    assert options.parse_feature_groups(args.disable_feature_groups) == ("pbp", "other")


def test_parse_args_accepts_regularization_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should expose the regularization knobs used by the noise-family arm."""
    # Arrange
    monkeypatch.setattr(
        sys, "argv", ["walk_forward_backtest.py", "--min-child-weight", "4.5", "--gamma", "2.0"]
    )

    # Act
    args = backtest._parse_args()

    # Assert
    assert args.min_child_weight == pytest.approx(4.5)
    assert args.gamma == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" pbp , other ,", ("pbp", "other")), (None, ()), ("", ())],
    ids=["whitespace-and-empty-segment", "missing", "empty"],
)
def test_parse_feature_groups_strips_whitespace_and_drops_empty_entries(
    value: str | None, expected: tuple[str, ...]
) -> None:
    """Parsing tolerates surrounding whitespace, empty segments, and a missing/empty value."""
    # Act
    groups = options.parse_feature_groups(value)

    # Assert
    assert groups == expected


def test_disable_feature_groups_drops_resolved_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved feature-group columns are the ones dropped from the loaded dataframe."""
    # Arrange
    monkeypatch.setattr(
        constants,
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
    groups = options.parse_feature_groups("pbp")

    # Act
    dropped = resolve_feature_group_columns(list(df.columns), groups)

    # Assert
    assert dropped == [
        "away_epa_per_play",
        "epa_per_play_diff",
        "home_epa_per_play",
    ]
    remaining = df.drop(columns=dropped)
    assert set(remaining.columns) == {"game_id", "season", "week", "away_total_yards"}


def test_disable_feature_groups_unknown_group_raises_from_cli_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown feature group name raises a clear error, not an obscure library traceback."""
    # Arrange
    monkeypatch.setattr(constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ("epa",)})
    groups = options.parse_feature_groups("not_a_real_group")

    # Act & Assert
    with pytest.raises(ValueError, match="not_a_real_group"):
        resolve_feature_group_columns(["away_epa"], groups)


def test_parse_args_accepts_n_estimators_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--n-estimators` parses as an integer."""
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py", "--n-estimators", "900"])

    # Act
    args = backtest._parse_args()

    # Assert
    assert args.n_estimators == 900


def test_parse_args_leaves_n_estimators_unset_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--n-estimators` defaults to None when omitted."""
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    # Act
    defaults = backtest._parse_args()

    # Assert
    assert defaults.n_estimators is None


def _record_engine_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[list[walk_forward.WalkForwardConfig], list[str]]:
    """Replace the engine with one that records its config; return the configs and base argv."""
    captured: list[walk_forward.WalkForwardConfig] = []

    def fake_run(
        _df: pd.DataFrame, config: walk_forward.WalkForwardConfig, **_kwargs: object
    ) -> dict[str, object]:
        """Record the config the CLI built and return a minimal result."""
        captured.append(config)
        return {"checkpoint": {}, "per_week": []}

    def fake_report(
        _run_id: str, _created_at: str, payload: dict[str, object], _results: object
    ) -> dict[str, object]:
        """Return the config payload unchanged."""
        return {"config": payload}

    def fake_metadata(
        _created_at: str, _hash: str, payload: dict[str, object]
    ) -> dict[str, object]:
        """Return a minimal metadata payload."""
        return {"config": payload}

    monkeypatch.setattr(walk_forward, "load_games", _loading(pd.DataFrame({"season": [2023]})))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", returning("hash"))
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(walk_forward, "build_metrics_report", fake_report)
    monkeypatch.setattr(walk_forward, "build_metadata", fake_metadata)
    base_argv = [
        "walk_forward_backtest.py",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--out-json",
        str(tmp_path / "metrics_report.json"),
    ]
    return captured, base_argv


def test_main_forwards_n_estimators_into_the_xgb_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tree-budget flag reaches the engine config as an int."""
    # Arrange
    captured, base_argv = _record_engine_configs(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [*base_argv, "--n-estimators", "900"])

    # Act
    backtest.main()

    # Assert
    overrides = captured[0].xgb_params_overrides or {}
    assert overrides["n_estimators"] == 900
    assert isinstance(overrides["n_estimators"], int)


def test_main_leaves_n_estimators_out_of_the_xgb_overrides_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the tree-budget flag the engine config carries no tree budget override."""
    # Arrange
    captured, base_argv = _record_engine_configs(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", base_argv)

    # Act
    backtest.main()

    # Assert
    assert "n_estimators" not in (captured[0].xgb_params_overrides or {})


# A dataset built before the ported ratings: the strength snapshot and no `ratings` column.
GAMES_WITHOUT_RATINGS = pd.DataFrame({"season": [2023], "home_adj_srs": [1.0]})


def _ratings_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the warnings that name the ported ratings."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "ported ratings" in record.getMessage()
    ]


def test_a_backtest_on_data_without_the_ported_ratings_warns_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    caplog.set_level(logging.WARNING)
    captured, base_argv = _record_engine_configs(monkeypatch, tmp_path)
    monkeypatch.setattr(walk_forward, "load_games", _loading(GAMES_WITHOUT_RATINGS))
    monkeypatch.setattr(sys, "argv", base_argv)

    # Act
    backtest.main()

    # Assert
    assert len(captured) == 1
    (message,) = _ratings_warnings(caplog)
    assert "nfl-predictor data" in message
    assert "--disable-feature-groups" in message


@pytest.mark.parametrize(
    ("games", "extra"),
    [
        (GAMES_WITHOUT_RATINGS.assign(ratings_team_diff=[0.5]), []),
        (GAMES_WITHOUT_RATINGS, ["--disable-feature-groups", ""]),
    ],
    ids=["ratings present", "strength kept"],
)
def test_a_backtest_says_nothing_when_the_ratings_are_there_or_strength_is_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    games: pd.DataFrame,
    extra: list[str],
) -> None:
    # Arrange
    caplog.set_level(logging.WARNING)
    captured, base_argv = _record_engine_configs(monkeypatch, tmp_path)
    monkeypatch.setattr(walk_forward, "load_games", _loading(games))
    monkeypatch.setattr(sys, "argv", [*base_argv, *extra])

    # Act
    backtest.main()

    # Assert
    assert len(captured) == 1
    assert _ratings_warnings(caplog) == []


def _prediction_rows() -> pd.DataFrame:
    """Return a small walk-forward prediction frame over two seasons."""
    games = [
        ("2023_01_a", 2023, 1, 0.70, 7),
        ("2023_01_b", 2023, 1, 0.40, 3),
        ("2023_03_a", 2023, 3, 0.60, -4),
        ("2024_01_a", 2024, 1, 0.55, 6),
        ("2024_02_a", 2024, 2, 0.35, -2),
    ]
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "deterministic_home_win_prob": p,
                "market_home_win_prob": 0.5,
                "actual_home_win": int(margin > 0),
                "actual_margin": float(margin),
                "actual_total": 41.0,
                "predicted_margin": 2.0,
                "predicted_total": 44.0,
            }
            for game_id, season, week, p, margin in games
        ]
    )


def test_main_adds_the_season_stability_view_to_the_report_and_the_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The report carries the per-season view ``compare`` would compute, and the log prints it."""
    # Arrange
    predictions = _prediction_rows()

    def fake_run(
        _df: pd.DataFrame, _config: walk_forward.WalkForwardConfig, **_kwargs: object
    ) -> dict[str, object]:
        """Return a result carrying the run's prediction frame."""
        return {"checkpoint": {}, "per_week": [], "predictions": predictions}

    def fake_report(
        _run_id: str, _created_at: str, payload: dict[str, object], _results: object
    ) -> dict[str, object]:
        """Return a report with an empty metrics block, as the engine's report has one."""
        return {"config": payload, "metrics": {}}

    monkeypatch.setattr(walk_forward, "load_games", _loading(pd.DataFrame({"season": [2023]})))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", returning("hash"))
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(walk_forward, "build_metrics_report", fake_report)

    def empty_metadata(*_args: object) -> dict[str, object]:
        """Return an empty metadata payload."""
        return {}

    monkeypatch.setattr(walk_forward, "build_metadata", empty_metadata)
    out_json = tmp_path / "metrics_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "walk_forward_backtest.py",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--out-json",
            str(out_json),
        ],
    )
    caplog.set_level(logging.INFO)

    # Act
    backtest.main()

    # Assert
    stability = json.loads(out_json.read_text())["metrics"]["stability"]
    expected = json.loads(json.dumps(run_comparison.stability_report(predictions)))
    assert stability == expected
    assert stability["resamples"] == run_comparison.DEFAULT_RESAMPLES
    assert set(stability["windows"]["week 1"]) == {"all seasons", "2023", "2024"}
    messages = [record.getMessage() for record in caplog.records]
    assert run_comparison.STABILITY_HEADING in messages
    assert any(message.startswith("| 2024 | 1 | ") for message in messages)


def test_a_standalone_backtest_supplies_no_floor_sigma_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """By default each week's sigma comes from the run's own earlier weeks only."""
    # Arrange
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    # Act
    args = backtest._parse_args()

    # Assert
    assert args.floor_sigma_reference_runs is None


def test_named_reference_runs_become_the_walk_forwards_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference runs are read once, handed to the engine and named in the run's config."""
    # Arrange
    captured: dict[str, object] = {}
    history = floor_sigma.ErrorPool(sources=("ref_a",))
    loaded: list[object] = []

    def fake_run(
        _df: pd.DataFrame, _config: walk_forward.WalkForwardConfig, **kwargs: object
    ) -> dict[str, object]:
        """Record the history and return a minimal result naming it."""
        captured.update(kwargs)
        return {"checkpoint": {}, "per_week": [], "floor_sigma": {"history_sources": ["ref_a"]}}

    def fake_load(paths: object) -> floor_sigma.ErrorPool:
        """Record the requested runs."""
        loaded.append(paths)
        return history

    monkeypatch.setattr(floor_sigma, "load_reference_pool", fake_load)
    monkeypatch.setattr(walk_forward, "load_games", _loading(pd.DataFrame({"season": [2023]})))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", returning("hash"))
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", fake_run)

    def fake_report(
        _run_id: str, _created_at: str, payload: dict[str, object], _results: object
    ) -> dict[str, object]:
        """Return the config payload as the report."""
        return {"config": payload}

    def fake_metadata(
        _created_at: str, _hash: str, payload: dict[str, object]
    ) -> dict[str, object]:
        """Return the config payload as the metadata."""
        return {"config": payload}

    monkeypatch.setattr(walk_forward, "build_metrics_report", fake_report)
    monkeypatch.setattr(walk_forward, "build_metadata", fake_metadata)
    out_json = tmp_path / "metrics_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "walk_forward_backtest.py",
            "--out-json",
            str(out_json),
            "--floor-sigma-reference-runs",
            "ref_a",
        ],
    )

    # Act
    backtest.main()

    # Assert
    assert loaded == [[Path("ref_a")]]
    assert captured["floor_sigma_history"] is history
    config = json.loads(out_json.read_text())["config"]
    assert config["floor_sigma"] == {"history_sources": ["ref_a"]}
