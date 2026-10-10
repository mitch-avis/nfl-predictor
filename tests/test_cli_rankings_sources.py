"""Tests for the rankings command's ported, model and blend methods."""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predictor.cli import rankings
from nfl_predictor.reporting import rating_sources
from nfl_predictor.reporting.power_rankings import (
    DEFAULT_RATINGS_DIR,
    RankingInputs,
    RankingOptions,
)


def _base_args(tmp_path: Path) -> list[str]:
    """Return arguments naming a model, both datasets and an output directory under tmp_path."""
    for name in ("model.joblib", "ml.csv", "schedule.csv"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    return [
        "--model-in",
        str(tmp_path / "model.joblib"),
        "--season",
        "2024",
        "--through-week",
        "3",
        "--data-ml",
        str(tmp_path / "ml.csv"),
        "--data-schedule",
        str(tmp_path / "schedule.csv"),
        "--out-dir",
        str(tmp_path / "out"),
    ]


def _captured_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]):
    """Run the command with ``extra`` and return the options it ranked with."""
    computed: list[RankingOptions] = []
    monkeypatch.setattr(rankings.ml_model_core, "load_model_checkpoint", lambda *_args: None)

    def fake_compute(_model: object, _inputs: RankingInputs, options: RankingOptions) -> str:
        computed.append(options)
        return "result"

    monkeypatch.setattr(rankings, "compute_power_rankings", fake_compute)
    monkeypatch.setattr(rankings, "write_ranking_outputs", lambda *_args, **_kwargs: None)
    rankings.main([*_base_args(tmp_path), *extra])
    return computed[0]


def test_ported_method_reads_the_ratings_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Act
    options = _captured_options(
        tmp_path, monkeypatch, ["--method", "ported", "--ratings-dir", "published"]
    )

    # Assert
    assert options.method == "ported"
    assert options.ratings_dir == Path("published")


def test_blend_method_takes_its_predictions_and_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    extra = [
        "--method",
        "blend",
        "--model-predictions",
        "runs/a",
        "runs/b",
        "--blend-weight",
        "0.6",
    ]

    # Act
    options = _captured_options(tmp_path, monkeypatch, extra)

    # Assert
    assert options.model_predictions == (Path("runs/a"), Path("runs/b"))
    assert options.blend_weight == pytest.approx(0.6)
    assert options.ratings_dir == DEFAULT_RATINGS_DIR


def test_model_method_without_predictions_is_a_usage_error(tmp_path: Path) -> None:
    # Arrange
    args = [*_base_args(tmp_path), "--method", "model"]

    # Act & Assert
    with pytest.raises(SystemExit, match="model-predictions"):
        rankings.main(args)


def test_an_unavailable_rating_source_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(rankings.ml_model_core, "load_model_checkpoint", lambda *_args: None)

    def unavailable(*_args: object) -> None:
        msg = "no ported team rating for 2024 week 4"
        raise rating_sources.RatingSourceUnavailableError(msg)

    monkeypatch.setattr(rankings, "compute_power_rankings", unavailable)
    args = [*_base_args(tmp_path), "--method", "ported"]

    # Act & Assert
    with pytest.raises(SystemExit, match="no ported team rating"):
        rankings.main(args)
