"""Tests for the weekly run's settings that rank on the ported, model or blend rating."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from nfl_predictor.reporting import power_rankings, rating_sources
from nfl_predictor.weekly_run import config as run_config
from nfl_predictor.weekly_run import pipeline


def _args(argv: list[str]):
    return run_config._build_parser().parse_args(argv)


def test_weekly_rating_source_flags_reach_the_ranking_options() -> None:
    # Arrange
    args = _args(
        [
            "--power-rankings-method",
            "blend",
            "--power-rankings-ratings-dir",
            "ratings",
            "--power-rankings-model-predictions",
            "runs/a",
            "runs/b",
            "--power-rankings-blend-weight",
            "0.4",
        ]
    )

    # Act
    options = run_config.power_ranking_options(args)

    # Assert
    assert options.method == "blend"
    assert options.ratings_dir == Path("ratings")
    assert options.model_predictions == (Path("runs/a"), Path("runs/b"))
    assert options.blend_weight == pytest.approx(0.4)


def test_weekly_rating_source_defaults_leave_the_composite_in_place() -> None:
    # Act
    options = run_config.power_ranking_options(_args([]))

    # Assert
    assert options.method == "composite"
    assert options.ratings_dir == power_rankings.DEFAULT_RATINGS_DIR
    assert options.model_predictions == ()
    assert options.blend_weight is None


def test_weekly_rating_source_settings_are_valid_config_keys() -> None:
    # Act
    allowed = run_config._allowed_config_keys(run_config._build_parser())

    # Assert
    assert {
        "power_rankings_ratings_dir",
        "power_rankings_model_predictions",
        "power_rankings_blend_weight",
    } <= allowed


def test_default_report_config_carries_no_rating_source_setting() -> None:
    # Act
    report = run_config.power_rankings_report_config(_args([]))

    # Assert
    assert not {key for key in report if key.startswith("power_rankings_ratings")}
    assert "power_rankings_model_predictions" not in report


def test_report_config_of_a_points_method_names_its_sources() -> None:
    # Arrange
    args = _args(["--power-rankings-method", "ported", "--power-rankings-ratings-dir", "r"])

    # Act
    report = run_config.power_rankings_report_config(args)

    # Assert
    assert report["power_rankings_ratings_dir"] == "r"
    assert report["power_rankings_method"] == "ported"


def test_weekly_rankings_skip_an_unavailable_rating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    data = tmp_path / "all_data_ml.csv"
    data.write_text("season,week\n", encoding="utf-8")
    args = _args(
        [
            "--power-rankings-method",
            "ported",
            "--power-rankings-data-ml",
            str(data),
            "--power-rankings-data-schedule",
            str(data),
            "--power-rankings-season",
            "2024",
            "--power-rankings-through-week",
            "3",
            "--power-rankings-out-dir",
            str(tmp_path / "out"),
        ]
    )
    context = pipeline._RunContext(
        args=args,
        run_id="run",
        run_dir=tmp_path,
        output_dir=tmp_path,
        created_at="now",
        dataset_hash="hash",
        dataset_fingerprint={},
        config_payload={},
    )

    def unavailable(*_args: object, **_kwargs: object) -> None:
        msg = "no ported team rating for 2024 week 4"
        raise rating_sources.RatingSourceUnavailableError(msg)

    monkeypatch.setattr(pipeline.ml_model_core, "load_model_checkpoint", lambda *_a: None)
    monkeypatch.setattr(pipeline.power_rankings, "compute_power_rankings", unavailable)

    # Act
    with caplog.at_level(logging.WARNING):
        written = pipeline._write_power_rankings(context, 2024, 4, tmp_path / "model.joblib")

    # Assert
    assert written == []
    assert "no ported team rating" in caplog.text


def test_weekly_rankings_skip_unreadable_model_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    data = tmp_path / "all_data_ml.csv"
    data.write_text("season,week\n", encoding="utf-8")
    args = _args(
        [
            "--power-rankings-method",
            "model",
            "--power-rankings-model-predictions",
            str(tmp_path / "no_such_run"),
            "--power-rankings-data-ml",
            str(data),
            "--power-rankings-data-schedule",
            str(data),
            "--power-rankings-season",
            "2024",
            "--power-rankings-through-week",
            "3",
            "--power-rankings-out-dir",
            str(tmp_path / "out"),
        ]
    )
    context = pipeline._RunContext(
        args=args,
        run_id="run",
        run_dir=tmp_path,
        output_dir=tmp_path,
        created_at="now",
        dataset_hash="hash",
        dataset_fingerprint={},
        config_payload={},
    )
    monkeypatch.setattr(pipeline.ml_model_core, "load_model_checkpoint", lambda *_a: None)

    # Act
    with caplog.at_level(logging.WARNING):
        written = pipeline._write_power_rankings(context, 2024, 4, tmp_path / "model.joblib")

    # Assert
    assert written == []
    assert "no_such_run" in caplog.text
