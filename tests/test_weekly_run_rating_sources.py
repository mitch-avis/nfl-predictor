"""Tests for the weekly run's settings that rank on the ported, model or blend rating."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core, walk_forward
from nfl_predictor.reporting import power_rankings, rating_sources
from nfl_predictor.weekly_run import config as run_config
from nfl_predictor.weekly_run import pipeline, stage1
from tests import snapshots
from tests.fakes import returning
from tests.rating_source_fixtures import fold_frame, write_fold_checkpoints
from tests.test_weekly_run_characterization import (
    SNAPSHOT_DIR,
    _run_weekly,
    _snapshot_outputs,
    _stable_frame,
)
from tests.weekly_fixture import build_fixture, write_weekly_config


def _args(argv: list[str]):
    return run_config._build_parser().parse_args(argv)


def _from_root(*parts: str) -> Path:
    """Return the repository-root path the rankings read for a relative entry."""
    return Path(constants.ROOT_DIR).joinpath(*parts)


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
    assert options.model_predictions == (_from_root("runs/a"), _from_root("runs/b"))
    assert options.blend_weight == pytest.approx(0.4)


def test_weekly_rating_source_defaults_rank_on_the_even_blend() -> None:
    # Act
    options = run_config.power_ranking_options(_args([]), run_dir=Path("run"))

    # Assert
    assert options.method == "blend"
    assert options.ratings_dir == power_rankings.DEFAULT_RATINGS_DIR
    assert options.model_predictions == (Path("run/wf_compare"),)
    assert options.blend_weight == pytest.approx(0.5)


def test_weekly_rating_source_settings_are_valid_config_keys() -> None:
    # Act
    allowed = run_config._allowed_config_keys(run_config._build_parser())

    # Assert
    assert {
        "power_rankings_ratings_dir",
        "power_rankings_model_predictions",
        "power_rankings_blend_weight",
    } <= allowed


def test_composite_report_config_carries_no_rating_source_setting() -> None:
    # Arrange
    args = _args(["--power-rankings-method", "composite"])

    # Act
    report = run_config.power_rankings_report_config(args, run_dir=Path("run"))

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


def _no_standings(*_args: object, **_kwargs: object) -> list[Path]:
    """Stand in for the projected standings, which these stub schedules cannot build."""
    return []


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

    monkeypatch.setattr(ml_model_core, "load_model_checkpoint", returning(None))
    monkeypatch.setattr(power_rankings, "compute_power_rankings", unavailable)
    monkeypatch.setattr(pipeline, "_write_projected_standings", _no_standings)

    # Act
    with caplog.at_level(logging.WARNING):
        written = pipeline._write_power_rankings(context, 2024, 4, tmp_path / "model.joblib")

    # Assert
    assert written.paths == []
    assert written.skipped is not None
    assert "no ported team rating" in written.skipped
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
    monkeypatch.setattr(ml_model_core, "load_model_checkpoint", returning(None))
    monkeypatch.setattr(pipeline, "_write_projected_standings", _no_standings)

    # Act
    with caplog.at_level(logging.WARNING):
        written = pipeline._write_power_rankings(context, 2024, 4, tmp_path / "model.joblib")

    # Assert
    assert written.paths == []
    assert "no_such_run" in caplog.text


@pytest.mark.parametrize(
    ("season", "missing"), [(None, None), (2024, "all_data_ml.csv"), (2024, "schedule.csv")]
)
def test_weekly_rankings_skipped_before_ranking_log_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    season: int | None,
    missing: str | None,
) -> None:
    # Arrange
    data_ml = tmp_path / "all_data_ml.csv"
    schedule = tmp_path / "schedule.csv"
    for path in (data_ml, schedule):
        if path.name != missing:
            path.write_text("season,week\n", encoding="utf-8")
    argv = [
        "--power-rankings-data-ml",
        str(data_ml),
        "--power-rankings-data-schedule",
        str(schedule),
        "--power-rankings-out-dir",
        str(tmp_path / "out"),
    ]
    if season is not None:
        argv += ["--power-rankings-season", str(season), "--power-rankings-through-week", "3"]
    context = pipeline._RunContext(
        args=_args(argv),
        run_id="run",
        run_dir=tmp_path,
        output_dir=tmp_path,
        created_at="now",
        dataset_hash="hash",
        dataset_fingerprint={},
        config_payload={},
    )

    # Act
    with caplog.at_level(logging.INFO):
        written = pipeline._write_power_rankings(context, None, None, tmp_path / "model.joblib")

    # Assert
    assert written.paths == []
    skips = [r for r in caplog.records if r.getMessage().startswith("Power rankings skipped")]
    reason = "unable to infer season/week." if missing is None else f"missing {tmp_path / missing}"
    assert [(r.levelno, r.getMessage()) for r in skips] == [
        (logging.WARNING, f"Power rankings skipped: {reason}")
    ]


DEFAULT_SOURCE_FLAGS = [
    "--power-rankings-method",
    "blend",
    "--power-rankings-blend-weight",
    "0.5",
    "--floor-sigma-reference-runs",
    "ref/seed42",
    "ref/seed7",
]


def test_the_rankings_read_relative_prediction_sources_from_the_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run started outside the repo root still finds the predictions a config names."""
    # Arrange
    absolute = tmp_path / "abs_run"
    args = _args(["--power-rankings-model-predictions", "runs/a", str(absolute)])
    monkeypatch.chdir(tmp_path)

    # Act
    options = run_config.power_ranking_options(args, run_dir=Path("run"))

    # Assert
    assert options.model_predictions == (_from_root("runs/a"), absolute)


def test_weekly_points_method_defaults_to_stage_one_folds_and_reference_runs() -> None:
    # Arrange
    args = _args(DEFAULT_SOURCE_FLAGS)

    # Act
    options = run_config.power_ranking_options(args, run_dir=Path("run"))

    # Assert
    assert options.model_predictions == (Path("run/wf_compare"),)
    assert options.reference_predictions == (
        _from_root("ref/seed42"),
        _from_root("ref/seed7"),
    )


def test_explicit_model_predictions_override_the_weekly_default() -> None:
    # Arrange
    args = _args([*DEFAULT_SOURCE_FLAGS, "--power-rankings-model-predictions", "runs/a"])

    # Act
    options = run_config.power_ranking_options(args, run_dir=Path("run"))

    # Assert
    assert options.model_predictions == (_from_root("runs/a"),)
    assert options.reference_predictions == ()


def test_report_config_of_the_default_blend_names_its_resolved_sources() -> None:
    # Arrange
    args = _args(DEFAULT_SOURCE_FLAGS)

    # Act
    report = run_config.power_rankings_report_config(args, run_dir=Path("run"))

    # Assert
    assert report["power_rankings_model_predictions"] == [str(Path("run/wf_compare"))]
    assert report["power_rankings_reference_predictions"] == [
        str(_from_root("ref/seed42")),
        str(_from_root("ref/seed7")),
    ]
    assert report["power_rankings_blend_weight"] == pytest.approx(0.5)


def test_shipped_weekly_config_ranks_on_the_even_blend() -> None:
    # Arrange
    args = run_config.parse_args([])

    # Act
    options = run_config.power_ranking_options(args, run_dir=Path("run"))

    # Assert
    assert options.method == "blend"
    assert options.blend_weight == pytest.approx(0.5)
    assert args.power_rankings_model_predictions == []


def test_weekly_rankings_skip_without_stage_one_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    data = tmp_path / "all_data_ml.csv"
    data.write_text("season,week\n", encoding="utf-8")
    args = _args(
        [
            "--power-rankings-method",
            "model",
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
        run_dir=tmp_path / "run",
        output_dir=tmp_path,
        created_at="now",
        dataset_hash="hash",
        dataset_fingerprint={},
        config_payload={},
    )
    monkeypatch.setattr(ml_model_core, "load_model_checkpoint", returning(None))
    monkeypatch.setattr(pipeline, "_write_projected_standings", _no_standings)

    # Act
    with caplog.at_level(logging.WARNING):
        written = pipeline._write_power_rankings(context, 2024, 4, tmp_path / "model.joblib")

    # Assert
    assert written.paths == []
    assert "Power rankings skipped" in caplog.text
    assert str(Path("run") / "wf_compare") in caplog.text


def test_stage_one_records_its_fold_checkpoints_as_a_prediction_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    folds = write_fold_checkpoints(
        tmp_path / "run" / "wf_compare" / "wf_folds" / "abc",
        [fold_frame(2024, 1, [("BBB", "AAA", 3.0)])],
    )
    result = {
        "overall": {"brier": 0.21, "deterministic_brier": 0.21, "games": 1, "weeks": 1},
        "resolved_settings": {},
        "predictions": pd.DataFrame(
            columns=["game_id", "season", "week", "actual_margin", "predicted_margin"]
        ),
        "checkpoint": {"dir": str(folds), "restored_folds": 0, "computed_folds": 1},
    }
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", returning(result))
    options = stage1.ProductionOptions(
        eval_last_n_seasons=1,
        wf_start_week=1,
        include_postseason=False,
        exclude_incomplete_seasons=False,
        recency_half_life_seasons=None,
        market_mode="hybrid",
        xgb_params_overrides={},
        include_quantiles=False,
    )
    run = stage1.Stage1Run(
        run_dir=tmp_path / "run",
        resume=True,
        dataset_fingerprint={"sha256": "fp"},
        wf_run_fingerprint="wf123",
        checkpoint_per_fold=False,
    )

    # Act
    stage1.evaluate_production(pd.DataFrame(), options, run)

    # Assert
    read = rating_sources.read_fold_predictions(stage1.predictions_source(tmp_path / "run"))
    assert read.checkpoint_dir == folds
    assert read.predictions.get_column("game_id").to_list() == ["2024_01_BBB_AAA"]


def _weekly_run_without_ported_ratings(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Write the weekly fixture with its ported ratings moved away, so the blend is unavailable.

    Returns:
        The config, the output directory, the ratings directory and where it was moved.

    """
    paths = build_fixture(tmp_path)
    output_dir = tmp_path / "out"
    config_path = write_weekly_config(tmp_path, paths, output_dir)
    moved = paths["ratings"].rename(tmp_path / "ratings_moved")
    return config_path, output_dir, paths["ratings"], moved


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_skipped_weekly_ranking_logs_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    config_path, _, _, _ = _weekly_run_without_ported_ratings(tmp_path)

    # Act
    with caplog.at_level(logging.INFO):
        _run_weekly(config_path, monkeypatch)

    # Assert
    skips = [record for record in caplog.records if "Power rankings skipped" in record.message]
    assert [record.levelno for record in skips] == [logging.ERROR]
    assert "no ported team rating" in skips[0].message


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_skipped_weekly_ranking_still_writes_the_projected_standings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    config_path, output_dir, _, _ = _weekly_run_without_ported_ratings(tmp_path)

    # Act
    _run_weekly(config_path, monkeypatch)

    # Assert
    outputs = _snapshot_outputs(tmp_path, output_dir)
    for name in ("projected_standings.csv", "projected_division_standings.csv"):
        snapshots.assert_frames_match(
            name, _stable_frame(outputs[name]), _stable_frame(SNAPSHOT_DIR / name)
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_skipped_weekly_ranking_writes_no_other_method_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    config_path, output_dir, _, _ = _weekly_run_without_ported_ratings(tmp_path)

    # Act
    with caplog.at_level(logging.INFO):
        _run_weekly(config_path, monkeypatch)

    # Assert
    assert not list(output_dir.glob("power_rankings_*.csv"))
    assert "Ranking on the" not in caplog.text


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_skipped_weekly_ranking_leaves_the_reports_marker_not_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    config_path, _, _, _ = _weekly_run_without_ported_ratings(tmp_path)
    marker = tmp_path / "run" / "reports_state.json"

    # Act
    _run_weekly(config_path, monkeypatch)

    # Assert
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert "no ported team rating" in payload["power_rankings_skipped"]
    assert not pipeline._stage_can_reuse(
        marker, payload["dataset_hash"], payload["config_hash"], []
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_resume_after_a_skipped_ranking_writes_the_rankings_once_the_source_is_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    config_path, output_dir, ratings_dir, moved = _weekly_run_without_ported_ratings(tmp_path)
    _run_weekly(config_path, monkeypatch)
    moved.rename(ratings_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["resume"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    # Act
    _run_weekly(config_path, monkeypatch)

    # Assert
    written = _snapshot_outputs(tmp_path, output_dir)["power_rankings.csv"]
    snapshots.assert_frames_match(
        "power_rankings.csv",
        _stable_frame(written),
        _stable_frame(SNAPSHOT_DIR / "power_rankings.csv"),
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_resume_with_a_reused_walk_forward_writes_its_predictions_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    paths = build_fixture(tmp_path)
    config_path = write_weekly_config(tmp_path, paths, tmp_path / "out")
    _run_weekly(config_path, monkeypatch)
    (pointer,) = tmp_path.rglob("wf_compare/metadata.json")
    written_by_the_walk = json.loads(pointer.read_text(encoding="utf-8"))
    pointer.unlink()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["resume"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    # Act
    with caplog.at_level(logging.INFO):
        _run_weekly(config_path, monkeypatch)

    # Assert
    assert any("Stage 1: reuse" in record.message for record in caplog.records)
    assert json.loads(pointer.read_text(encoding="utf-8")) == written_by_the_walk


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_resume_with_a_reused_walk_forward_and_no_fold_checkpoints_writes_no_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    paths = build_fixture(tmp_path)
    config_path = write_weekly_config(tmp_path, paths, tmp_path / "out")
    _run_weekly(config_path, monkeypatch)
    (pointer,) = tmp_path.rglob("wf_compare/metadata.json")
    pointer.unlink()
    (folds,) = tmp_path.rglob("wf_compare/wf_folds")
    folds.rename(tmp_path / "wf_folds_moved")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["resume"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    # Act
    with caplog.at_level(logging.INFO):
        _run_weekly(config_path, monkeypatch)

    # Assert
    assert not pointer.exists()
    assert _rerun_stage_one_warnings(caplog) == [logging.WARNING]


def _finished_run_without_its_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Run the weekly fixture, remove stage 1's pointer, and set the config to resume.

    Returns:
        The config and where the pointer was.

    """
    paths = build_fixture(tmp_path)
    config_path = write_weekly_config(tmp_path, paths, tmp_path / "out")
    _run_weekly(config_path, monkeypatch)
    (pointer,) = tmp_path.rglob("wf_compare/metadata.json")
    pointer.unlink()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["resume"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config_path, pointer


def _upgrade_a_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Change the library versions the fold checkpoint fingerprint covers, not the reuse check."""
    original = walk_forward._library_versions
    monkeypatch.setattr(
        walk_forward, "_library_versions", lambda: {**original(), "xgboost": "upgraded"}
    )


def _rerun_stage_one_warnings(caplog: pytest.LogCaptureFixture) -> list[int]:
    """Return the levels of the records that tell the user to rerun stage 1."""
    return [record.levelno for record in caplog.records if "--no-resume" in record.message]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reused_stage_one_after_a_fingerprint_change_still_finds_its_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    config_path, pointer = _finished_run_without_its_pointer(tmp_path, monkeypatch)
    (folds,) = (tmp_path / "run" / "wf_compare" / "wf_folds").iterdir()
    _upgrade_a_library(monkeypatch)

    # Act
    _run_weekly(config_path, monkeypatch)

    # Assert
    written = json.loads(pointer.read_text(encoding="utf-8"))
    assert written == {"config": {"checkpoint": {"dir": str(folds)}}}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reused_stage_one_with_two_matching_fold_folders_writes_no_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    config_path, pointer = _finished_run_without_its_pointer(tmp_path, monkeypatch)
    (folds,) = (tmp_path / "run" / "wf_compare" / "wf_folds").iterdir()
    copy = folds.with_name("0" * len(folds.name))
    copy.mkdir()
    (copy / "manifest.json").write_bytes((folds / "manifest.json").read_bytes())
    _upgrade_a_library(monkeypatch)

    # Act
    with caplog.at_level(logging.INFO):
        _run_weekly(config_path, monkeypatch)

    # Assert
    assert not pointer.exists()
    assert _rerun_stage_one_warnings(caplog) == [logging.WARNING]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reused_stage_one_with_no_matching_fold_folder_writes_no_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    config_path, pointer = _finished_run_without_its_pointer(tmp_path, monkeypatch)
    (manifest_path,) = tmp_path.rglob("wf_compare/wf_folds/*/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _upgrade_a_library(monkeypatch)

    # Act
    with caplog.at_level(logging.INFO):
        _run_weekly(config_path, monkeypatch)

    # Assert
    assert not pointer.exists()
    assert _rerun_stage_one_warnings(caplog) == [logging.WARNING]
