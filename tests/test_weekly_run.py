"""Unit tests for the weekly_run orchestration helpers."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
import xgboost as xgb

from nfl_predictor import data_collection
from nfl_predictor.api.readers import cache as api_cache
from nfl_predictor.api.readers import model as api_model
from nfl_predictor.api.runs.files import resolve_run_files
from nfl_predictor.ml import (
    artifacts,
    floor_sigma,
    ml_model_core,
    ml_model_training,
    ml_model_xgb_utils,
    walk_forward,
)
from nfl_predictor.utils import fingerprints
from nfl_predictor.weekly_run import config as run_config
from nfl_predictor.weekly_run import inputs, pipeline, stage1
from tests.api import factories
from tests.fakes import returning

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

    from nfl_predictor.ml.ml_model_training import TrainingOptions


@pytest.fixture(autouse=True)
def _no_floor_sigma_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the floor sigma's reference runs and the final fit's pool.

    These tests stub the stages around them; the pools are tested in
    ``tests/test_weekly_run_floor_sigma.py``.
    """

    def empty_reference(_paths: Sequence[Path]) -> floor_sigma.ErrorPool:
        return floor_sigma.ErrorPool()

    def empty_production(
        _args: argparse.Namespace, _run_dir: Path, _reference: floor_sigma.ErrorPool
    ) -> tuple[floor_sigma.ErrorPool, tuple[int, int]]:
        return floor_sigma.ErrorPool(), (2025, 2)

    monkeypatch.setattr(floor_sigma, "load_reference_pool", empty_reference)
    monkeypatch.setattr(pipeline, "_production_floor_sigma", empty_production)


def _empty_games(_path: Path) -> pd.DataFrame:
    """Return an empty dataset in place of reading one."""
    return pd.DataFrame()


def _fixed_fingerprint(_path: Path) -> dict[str, Any]:
    """Return a constant dataset fingerprint in place of hashing a file."""
    return {"sha256": "fp"}


def test_load_config_json(tmp_path: Path) -> None:
    """JSON configs should load into a dict."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text(json.dumps({"wf_eval_last_n_seasons": 4}), encoding="utf-8")

    # Act
    payload = run_config._load_config(config_path)

    # Assert
    assert payload["wf_eval_last_n_seasons"] == 4


def test_load_config_parses_yaml(tmp_path: Path) -> None:
    """YAML configs parse into a mapping."""
    # Arrange
    config_path = tmp_path / "weekly.yaml"
    config_path.write_text("wf_eval_last_n_seasons: 3\n", encoding="utf-8")

    # Act
    payload = run_config._load_config(config_path)

    # Assert
    assert payload["wf_eval_last_n_seasons"] == 3


def test_shipped_weekly_run_config_matches_approved_defaults() -> None:
    """The shipped weekly config should reflect the approved defaults.

    That is 200 trees, no postseason (in evaluation, training or the power rankings), no tuning,
    and no season recency weighting in either the walk-forward stage or the final training fit.
    """
    # Arrange
    config_path = Path(__file__).resolve().parents[1] / "config" / "weekly_run.yaml"
    config = run_config._load_config(config_path)

    # Act
    args = run_config._build_parser(run_config._normalize_config_defaults(config)).parse_args([])

    # Assert
    assert args.wf_n_estimators == 200
    assert args.wf_include_postseason is False
    assert args.include_postseason is False
    assert args.power_rankings_include_postseason is False
    assert args.tune is False
    assert args.wf_recency_half_life_seasons is None
    assert args.train_recency_half_life_seasons is None


def test_weekly_run_parser_defaults_follow_shared_xgb_defaults() -> None:
    """Bare weekly-run defaults should match the shared production XGBoost defaults."""
    # Arrange
    defaults = ml_model_core.DEFAULT_XGB_PARAMS

    # Act
    args = run_config._build_parser().parse_args([])

    # Assert
    assert args.wf_n_estimators == defaults["n_estimators"]
    assert args.wf_max_depth == defaults["max_depth"]
    assert args.wf_learning_rate == pytest.approx(defaults["learning_rate"])


def test_resolve_predict_path_prefers_latest_week(tmp_path: Path) -> None:
    """When predict_path is None, the newest week file should be chosen."""
    # Arrange
    predict_dir = tmp_path / "predict"
    predict_dir.mkdir()
    week_01 = predict_dir / "week_01_games_to_predict.csv"
    week_10 = predict_dir / "week_10_games_to_predict.csv"
    week_01.write_text("season,week\n2025,1\n", encoding="utf-8")
    week_10.write_text("season,week\n2025,10\n", encoding="utf-8")

    # Act
    resolved = inputs.resolve_predict_path(None, tmp_path)

    # Assert
    assert resolved == week_10


def test_resolve_predict_path_prefers_latest_season(tmp_path: Path) -> None:
    """When seasons differ, the newest season should win even if its week number is lower."""
    # Arrange
    predict_dir = tmp_path / "predict"
    predict_dir.mkdir()
    week_22 = predict_dir / "week_22_games_to_predict.csv"
    week_01 = predict_dir / "week_01_games_to_predict.csv"
    week_22.write_text("season,week\n2025,22\n", encoding="utf-8")
    week_01.write_text("season,week\n2026,1\n", encoding="utf-8")

    # Act
    resolved = inputs.resolve_predict_path(None, tmp_path)

    # Assert
    assert resolved == week_01


def test_build_confidence_picks_adds_winner_and_rank() -> None:
    """Confidence picks should include predicted winners and ranks."""
    # Arrange
    df = pd.DataFrame(
        {
            "away_abbr": ["A", "B"],
            "home_abbr": ["C", "D"],
            "home_win_prob": [0.65, 0.45],
            "away_win_prob": [0.35, 0.55],
        }
    )

    # Act
    picks = pipeline._build_confidence_picks(df)

    # Assert
    assert "predicted_winner" in picks.columns
    assert "confidence_rank" in picks.columns
    assert picks["confidence_rank"].is_unique


def _write_wf_compare_marker(tmp_path: Path) -> tuple[Path, Path]:
    """Write a wf_compare stage marker over one existing output and return both paths."""
    marker = tmp_path / "stage_state.json"
    output_path = tmp_path / "output.csv"
    output_path.write_text("ok", encoding="utf-8")
    pipeline._write_stage_marker(
        marker,
        dataset_hash="abc123",
        config_hash="def456",
        stage="wf_compare",
        extra={"outputs": [str(output_path)]},
    )
    return marker, output_path


def test_stage_marker_reuse(tmp_path: Path) -> None:
    """Stage markers should reuse when hashes match and outputs exist."""
    # Arrange
    marker, output_path = _write_wf_compare_marker(tmp_path)

    # Act
    result = pipeline._stage_can_reuse(marker, "abc123", "def456", [output_path])

    # Assert
    assert result


def test_stage_marker_reuse_rejects_a_changed_config_hash(tmp_path: Path) -> None:
    """Stage markers should not reuse when the config hash differs."""
    # Arrange
    marker, output_path = _write_wf_compare_marker(tmp_path)

    # Act
    result = pipeline._stage_can_reuse(marker, "abc123", "wrong", [output_path])

    # Assert
    assert not result


def _stub_walk_forward_results() -> dict[str, object]:
    """Return a minimal walk-forward result for the production evaluation."""
    return {
        "overall": {"brier": 0.21, "deterministic_brier": 0.21, "games": 2, "weeks": 1},
        "per_week": [{"season": 2024, "week": 3, "games": 2}],
        "per_season": [{"season": 2024, "games": 2}],
        "reliability": [{"bin_lower": 0.0, "bin_upper": 0.1, "count": 2}],
        "resolved_settings": {"market_anchor": True},
        "predictions": pd.DataFrame(
            columns=["game_id", "season", "week", "actual_margin", "predicted_margin"]
        ),
    }


def test_stage1_walks_forward_once_on_the_production_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 1 is one walk-forward of what production submits: the deterministic floor."""
    # Arrange
    seen: list[tuple[walk_forward.WalkForwardConfig, dict[str, object]]] = []

    def _fake_run(
        _df: pd.DataFrame, config: walk_forward.WalkForwardConfig, **kwargs: object
    ) -> dict[str, object]:
        seen.append((config, kwargs))
        return _stub_walk_forward_results()

    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", _fake_run)
    run_dir = tmp_path / "run"

    # Act
    summary = stage1.evaluate_production(
        pd.DataFrame(),
        stage1.ProductionOptions(
            eval_last_n_seasons=3,
            wf_start_week=3,
            include_postseason=False,
            exclude_incomplete_seasons=False,
            recency_half_life_seasons=None,
            market_mode="hybrid",
            xgb_params_overrides={"n_estimators": 20},
            include_quantiles=False,
        ),
        stage1.Stage1Run(
            run_dir=run_dir,
            resume=True,
            dataset_fingerprint={"sha256": "fp"},
            wf_run_fingerprint="wf123",
            checkpoint_per_fold=False,
        ),
    )

    # Assert
    assert len(seen) == 1
    config, kwargs = seen[0]
    assert config.calibration == "auto"
    assert (config.include_market, config.market_anchor) == (True, True)
    assert kwargs["checkpoints"] == walk_forward.FoldCheckpoints(
        run_dir / "wf_compare" / "wf_folds", resume=True
    )
    assert summary["market_mode"] == "hybrid"
    assert summary["brier"] == pytest.approx(0.21)
    for retired in ("calibration", "market_prob_weight", "market_prob_clamp"):
        assert retired not in summary
    artifact = json.loads(
        (run_dir / "wf_compare" / f"wf_candidate_{summary['candidate_key']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["metrics"]["reliability"] == [{"bin_lower": 0.0, "bin_upper": 0.1, "count": 2}]
    assert artifact["summary"]["candidate_key"] == summary["candidate_key"]


def test_the_model_page_reads_the_production_walk_forward_of_a_weekly_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web Model page shows stage 1's one row and its reliability bins."""
    # Arrange

    def fake_walk_forward(*_a: object, **_k: object) -> dict[str, object]:
        return _stub_walk_forward_results()

    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", fake_walk_forward)
    run_dir = factories.make_run_dir(tmp_path, "weekly_floor")
    shutil.rmtree(run_dir / "wf_compare", ignore_errors=True)
    summary = stage1.evaluate_production(
        pd.DataFrame(),
        stage1.ProductionOptions(
            eval_last_n_seasons=3,
            wf_start_week=3,
            include_postseason=False,
            exclude_incomplete_seasons=False,
            recency_half_life_seasons=None,
            market_mode="hybrid",
            xgb_params_overrides={},
            include_quantiles=False,
        ),
        stage1.Stage1Run(
            run_dir=run_dir,
            resume=True,
            dataset_fingerprint={"sha256": "fp"},
            wf_run_fingerprint="wf123",
            checkpoint_per_fold=False,
        ),
    )
    stage1.write_summary(run_dir, summary)
    api_cache.clear()

    # Act
    payload = api_model.model_payload(resolve_run_files(run_dir))

    # Assert
    table = payload["wf_compare"]
    assert table is not None
    assert [row["label"] for row in table.rows] == ["production"]
    assert table.rows[0]["wf_rank"] == 1
    assert "calibration" not in table.visible_columns
    assert payload["wf_best"]["candidate_key"] == summary["candidate_key"]
    assert payload["calibration"]["bin_count"] == 1


def test_the_final_fit_uses_the_configured_market_mode_and_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final fit trains with ``--wf-market-mode`` and submits the floor, with no blend."""

    # Arrange
    class _StopAtFinalFitError(Exception):
        """Stop weekly_run once the final fit's settings have been captured."""

    captured: dict[str, object] = {}
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week,home_spread,total_line\n2025,1,-3.0,44.5\n", encoding="utf-8")

    def _fake_train(options: TrainingOptions) -> None:
        captured.update(vars(options))
        raise _StopAtFinalFitError

    monkeypatch.setattr(walk_forward, "load_games", _empty_games)
    monkeypatch.setattr(artifacts, "sha256_file", returning("hash"))
    monkeypatch.setattr(fingerprints, "dataset_fingerprint", _fixed_fingerprint)
    monkeypatch.setattr(pipeline, "_stage_can_reuse", returning(value=False))

    def fake_evaluate(
        _df: pd.DataFrame, options: stage1.ProductionOptions, _run: stage1.Stage1Run
    ) -> dict[str, object]:
        return {"market_mode": options.market_mode}

    monkeypatch.setattr(stage1, "evaluate_production", fake_evaluate)
    monkeypatch.setattr(pipeline, "train_margin_total_model_with_report", _fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_run.py",
            "--skip-data-refresh",
            "--data-path",
            str(data_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--output-dir",
            str(tmp_path / "out"),
            "--wf-market-mode",
            "anchor",
        ],
    )

    # Act & Assert
    with pytest.raises(_StopAtFinalFitError):
        pipeline.main()

    assert (captured["include_market"], captured["market_anchor"]) == (False, True)
    assert "win_prob_calibration" not in captured
    assert "market_prob_config" not in captured
    assert "win_prob_use_uncertainty" not in captured
    # The final fit trains on every completed game: no weeks or seasons are held out.
    assert captured["holdout_seasons"] == 0
    assert "calibration_weeks" not in captured
    assert "calibration_seasons" not in captured


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("features", (True, False)), ("anchor", (False, True)), ("hybrid", (True, True))],
)
def test_market_mode_flags_name_market_features_and_anchoring(
    mode: str, expected: tuple[bool, bool]
) -> None:
    """Each market mode says whether market features and market anchoring are on."""
    # Act
    result = stage1.market_mode_flags(mode)

    # Assert
    assert result == expected


def test_an_unknown_market_mode_is_an_error() -> None:
    """A market mode outside the three is refused rather than guessed."""
    # Act & Assert
    with pytest.raises(ValueError, match="all"):
        stage1.market_mode_flags("all")


@pytest.mark.parametrize(
    ("extra_argv", "transform", "ratio"),
    [([], None, 0.5), (["--no-market-transform", "--max-cardinality-ratio", "0.3"], False, 0.3)],
)
def test_stage1_walks_forward_with_the_final_fits_market_transform_and_cardinality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_argv: list[str],
    *,
    transform: bool | None,
    ratio: float,
) -> None:
    """The stage-1 walk-forward reads the same two training options the final fit reads."""

    # Arrange
    class _StopAfterStage1Error(Exception):
        """Stop weekly_run once the stage-1 walk-forward config has been captured."""

    captured: list[walk_forward.WalkForwardConfig] = []
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week\n2025,1\n", encoding="utf-8")

    def _fake_run(
        _df: pd.DataFrame, config: walk_forward.WalkForwardConfig, **_kwargs: object
    ) -> dict[str, object]:
        captured.append(config)
        raise _StopAfterStage1Error

    monkeypatch.setattr(walk_forward, "load_games", _empty_games)
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", _fake_run)
    monkeypatch.setattr(artifacts, "sha256_file", returning("hash"))
    monkeypatch.setattr(fingerprints, "dataset_fingerprint", _fixed_fingerprint)
    monkeypatch.setattr(pipeline, "_stage_can_reuse", returning(value=False))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_run.py",
            "--skip-data-refresh",
            "--data-path",
            str(data_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--output-dir",
            str(tmp_path / "out"),
            *extra_argv,
        ],
    )

    # Act & Assert
    with pytest.raises(_StopAfterStage1Error):
        pipeline.main()

    assert captured[0].market_transform is transform
    assert captured[0].max_cardinality_ratio == pytest.approx(ratio)


def test_a_config_with_the_retired_all_market_mode_is_rejected(tmp_path: Path) -> None:
    """``wf_market_mode: all`` in a config file fails at parse time, before any ETL refresh."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"wf_market_mode": "all"}', encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match=r"wf_market_mode.*all.*features, anchor, hybrid"):
        run_config._parse_args(["--config", str(config_path)])


def test_the_retired_all_market_mode_no_longer_parses() -> None:
    """One production configuration means one market mode; ``all`` compared three."""
    # Act & Assert
    with pytest.raises(SystemExit):
        run_config._build_parser().parse_args(["--wf-market-mode", "all"])


@pytest.mark.parametrize(
    "key", ["wf_market_prob_source", "wf_market_prob_blend_method", "wf_win_prob_uncertainty"]
)
def test_a_config_with_a_retired_probability_key_is_rejected(tmp_path: Path, key: str) -> None:
    """Old configs that set a retired stage-1 probability option fail with a clear message."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text(json.dumps({key: "raw"}), encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match=rf"{key}.*deterministic floor"):
        run_config._parse_args(["--config", str(config_path)])


@pytest.mark.parametrize("spec", [None, "   "])
def test_data_refresh_without_arguments_calls_data_collection_bare(
    monkeypatch: pytest.MonkeyPatch, spec: str | None
) -> None:
    """With no pass-through string the refresh calls data collection with no argv."""
    # Arrange
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_main(*args: object, **kwargs: object) -> None:
        """Record how the data-collection entrypoint was invoked."""
        calls.append((args, kwargs))

    monkeypatch.setattr(data_collection, "main", _fake_main)

    # Act
    pipeline._refresh_data(spec)

    # Assert
    assert calls == [((), {})]


def test_data_refresh_forwards_split_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pass-through string is split shell-style and forwarded as the argv list."""
    # Arrange
    captured: list[list[str]] = []

    def _fake_main(argv: list[str] | None = None) -> None:
        """Record the argv list handed to the data-collection entrypoint."""
        assert argv is not None
        captured.append(argv)

    monkeypatch.setattr(data_collection, "main", _fake_main)

    # Act
    pipeline._refresh_data("--min-season 2010 --stat-prior-blend-games 4")

    # Assert
    assert captured == [["--min-season", "2010", "--stat-prior-blend-games", "4"]]


def _weekly_argv(tmp_path: Path, *extra: str) -> list[str]:
    """Build a weekly_run argv whose dataset and outputs all live under ``tmp_path``."""
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week,home_spread,total_line\n2025,1,-3.0,44.5\n", encoding="utf-8")
    return [
        "weekly_run.py",
        "--data-path",
        str(data_path),
        "--run-dir",
        str(tmp_path / "run"),
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


def test_a_dry_run_does_not_refresh_the_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--dry-run`` prints the plan without running any stage, the data refresh included."""

    # Arrange
    def _fail_refresh(_spec: str | None) -> None:
        msg = "a dry run must not refresh the data"
        raise AssertionError(msg)

    monkeypatch.setattr(pipeline, "_refresh_data", _fail_refresh)
    monkeypatch.setattr(sys, "argv", _weekly_argv(tmp_path, "--dry-run"))

    # Act
    result = pipeline.main()

    # Assert
    assert result == 0


def test_a_run_without_dry_run_refreshes_the_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--dry-run`` or ``--skip-data-refresh`` the refresh runs before the stages."""

    # Arrange
    class _StopAfterRefreshError(Exception):
        """Stop weekly_run once the refresh has been called."""

    calls: list[str | None] = []

    def _record_refresh(spec: str | None) -> None:
        calls.append(spec)
        raise _StopAfterRefreshError

    monkeypatch.setattr(pipeline, "_refresh_data", _record_refresh)
    monkeypatch.setattr(sys, "argv", _weekly_argv(tmp_path))

    # Act & Assert
    with pytest.raises(_StopAfterRefreshError):
        pipeline.main()

    assert calls == ["--incremental"]


def test_data_collection_args_is_a_valid_config_key() -> None:
    """A config file can set the data-collection pass-through arguments."""
    # Act
    allowed = run_config._allowed_config_keys(run_config._build_parser())

    # Assert
    assert "data_collection_args" in allowed


def test_data_collection_args_parses_from_the_command_line() -> None:
    """The command-line flag stores the raw pass-through string."""
    # Act
    args = run_config._build_parser().parse_args(["--data-collection-args", "--min-season 2010"])

    # Assert
    assert args.data_collection_args == "--min-season 2010"


def _shipped_weekly_args(argv: list[str]) -> argparse.Namespace:
    """Parse ``argv`` over the shipped ``config/weekly_run.yaml``."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "weekly_run.yaml"
    config = run_config._load_config(config_path)
    return run_config._build_parser(run_config._normalize_config_defaults(config)).parse_args(argv)


def test_the_shipped_weekly_run_refreshes_data_incrementally() -> None:
    """The shipped refresh reuses finished seasons and otherwise runs the ETL's own defaults."""
    # Arrange
    argv = pipeline._data_collection_argv(_shipped_weekly_args([]).data_collection_args)
    full_rebuild = data_collection._resolve_config(None)

    # Act
    refresh = data_collection._resolve_config(argv)

    # Assert
    assert refresh.incremental is True
    assert dataclasses.replace(refresh, incremental=False) == full_rebuild


def test_command_line_data_collection_args_replace_the_shipped_string() -> None:
    """A command-line string replaces the configured one rather than adding to it."""
    # Act
    args = _shipped_weekly_args(["--data-collection-args", "--min-season 2010"])

    # Assert
    assert args.data_collection_args == "--min-season 2010"


def test_a_config_file_without_data_collection_args_refreshes_with_a_full_rebuild(
    tmp_path: Path,
) -> None:
    """A ``--config`` file replaces the shipped one, so leaving the key out drops --incremental."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"wf_eval_last_n_seasons": 5}', encoding="utf-8")

    # Act
    args = run_config._parse_args(["--config", str(config_path)])

    # Assert
    assert args.data_collection_args is None


def test_a_config_file_without_data_collection_args_passes_no_refresh_argv(
    tmp_path: Path,
) -> None:
    """Leaving the key out of a ``--config`` file hands data collection no argv."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"wf_eval_last_n_seasons": 5}', encoding="utf-8")
    args = run_config._parse_args(["--config", str(config_path)])

    # Act
    result = pipeline._data_collection_argv(args.data_collection_args)

    # Assert
    assert result is None


def test_data_collection_without_argv_is_a_full_rebuild() -> None:
    """Data collection with no argv does not run incrementally."""
    # Act
    result = data_collection._resolve_config(None)

    # Assert
    assert result.incremental is False


def test_an_empty_command_line_string_refreshes_with_a_full_rebuild() -> None:
    """``--data-collection-args ""`` replaces the shipped string with nothing: a full rebuild."""
    # Act
    args = _shipped_weekly_args(["--data-collection-args", ""])

    # Assert
    assert args.data_collection_args == ""


def test_an_empty_command_line_string_passes_no_refresh_argv() -> None:
    """An empty pass-through string hands data collection no argv."""
    # Arrange
    args = _shipped_weekly_args(["--data-collection-args", ""])

    # Act
    result = pipeline._data_collection_argv(args.data_collection_args)

    # Assert
    assert result is None


def test_weekly_run_stage1_uses_shared_xgb_defaults_when_not_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 1 should evaluate the same XGBoost defaults the final fit uses."""

    # Arrange
    class _StopAfterStage1Error(Exception):
        """Stop weekly_run once the Stage 1 config has been captured."""

    captured: dict[str, object] = {}
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week\n2025,1\n", encoding="utf-8")

    def _fake_run_wf_compare(
        _df: pd.DataFrame, options: stage1.ProductionOptions, _run: stage1.Stage1Run
    ) -> dict[str, object]:
        """Capture the Stage 1 overrides and stop before later stages run."""
        captured.update(options.xgb_params_overrides)
        raise _StopAfterStage1Error

    monkeypatch.setattr(walk_forward, "load_games", _empty_games)
    monkeypatch.setattr(artifacts, "sha256_file", returning("hash"))
    monkeypatch.setattr(fingerprints, "dataset_fingerprint", _fixed_fingerprint)
    monkeypatch.setattr(pipeline, "_stage_can_reuse", returning(value=False))
    monkeypatch.setattr(stage1, "evaluate_production", _fake_run_wf_compare)

    old_argv = sys.argv

    # Act & Assert
    try:
        sys.argv = [
            "weekly_run.py",
            "--skip-data-refresh",
            "--data-path",
            str(data_path),
            "--run-id",
            "weekly_test",
            "--run-dir",
            str(tmp_path / "run"),
            "--output-dir",
            str(tmp_path / "out"),
        ]

        with pytest.raises(_StopAfterStage1Error):
            pipeline.main()
    finally:
        sys.argv = old_argv

    resolved = ml_model_xgb_utils.resolve_xgb_params(
        ml_model_core.DEFAULT_XGB_PARAMS,
        overrides=captured,
    )
    defaults = ml_model_core.DEFAULT_XGB_PARAMS

    assert resolved["n_estimators"] == defaults["n_estimators"]
    assert resolved["max_depth"] == defaults["max_depth"]
    assert resolved["learning_rate"] == pytest.approx(defaults["learning_rate"])
    assert resolved["subsample"] == pytest.approx(defaults["subsample"])
    assert resolved["colsample_bytree"] == pytest.approx(defaults["colsample_bytree"])


def test_renamed_config_keys_load_under_their_new_names(tmp_path: Path) -> None:
    """A config file with the old tuning key names still sets the renamed options."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text(
        '{"tune_n_trials": 7, "train_early_stopping_rounds": 30}', encoding="utf-8"
    )

    # Act
    args = run_config._parse_args(["--config", str(config_path)])

    # Assert
    assert (args.tune_trials, args.tune_early_stopping_rounds) == (7, 30)


def test_a_config_with_both_names_of_a_key_is_rejected(tmp_path: Path) -> None:
    """Setting a key under both its old and new names is ambiguous and fails."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"tune_n_trials": 7, "tune_trials": 8}', encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match="tune_n_trials"):
        run_config._parse_args(["--config", str(config_path)])


@pytest.mark.parametrize(
    ("argv", "dest", "value"),
    [
        (["--tune-trials", "5"], "tune_trials", 5),
        (["--tune-n-trials", "5"], "tune_trials", 5),
        (["--tune-early-stopping-rounds", "9"], "tune_early_stopping_rounds", 9),
        (["--train-early-stopping-rounds", "9"], "tune_early_stopping_rounds", 9),
    ],
)
def test_weekly_tuning_options_accept_both_spellings(
    argv: list[str], dest: str, value: int
) -> None:
    """The renamed tuning options parse under their new and their old spelling."""
    # Act
    args = run_config._build_parser().parse_args(argv)

    # Assert
    assert getattr(args, dest) == value


def test_the_shipped_config_is_read_by_default_and_matches_the_code_defaults() -> None:
    """Without --config the shipped YAML is read, and it changes nothing a run produces.

    The only differences from the bare code defaults are the recorded config path,
    ``postseason_weight``, which applies only when postseason games are in training (off), and
    ``data_collection_args``, whose ``--incremental`` refresh writes what a full rebuild writes.
    """
    # Arrange
    bare = vars(run_config._build_parser().parse_args([]))

    # Act
    loaded = vars(run_config._parse_args([]))

    # Assert
    differences = {key for key in bare if loaded[key] != bare[key]}
    assert differences == {"config", "postseason_weight", "data_collection_args"}
    assert loaded["data_collection_args"] == "--incremental"
    assert loaded["config"] == run_config.DEFAULT_CONFIG_PATH
    assert loaded["include_postseason"] is False


def test_an_explicit_config_replaces_the_shipped_one(tmp_path: Path) -> None:
    """``--config`` reads only the given file; keys it omits keep their code defaults."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"wf_eval_last_n_seasons": 5}', encoding="utf-8")

    # Act
    args = run_config._parse_args(["--config", str(config_path)])

    # Assert
    assert args.config == config_path
    assert args.wf_eval_last_n_seasons == 5
    assert args.postseason_weight == 1.0


@pytest.mark.parametrize("option", ["skip_data_refresh", "dry_run"])
def test_a_config_file_turns_a_switch_on(tmp_path: Path, option: str) -> None:
    """A config that sets a switch turns it on."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text(json.dumps({option: True}), encoding="utf-8")

    # Act
    configured = run_config._parse_args(["--config", str(config_path)])

    # Assert
    assert getattr(configured, option) is True


@pytest.mark.parametrize("option", ["skip_data_refresh", "dry_run"])
def test_a_command_line_switch_turns_off_what_the_config_turns_on(
    tmp_path: Path, option: str
) -> None:
    """``--no-<switch>`` overrides a config that sets the switch, as for every other option."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text(json.dumps({option: True}), encoding="utf-8")
    flag = option.replace("_", "-")

    # Act
    overridden = run_config._parse_args(["--config", str(config_path), f"--no-{flag}"])

    # Assert
    assert getattr(overridden, option) is False


def test_xgb_n_jobs_defaults_to_every_cpu_core() -> None:
    """One thread option sets XGBoost's CPU threads, and it defaults to every core."""
    # Arrange
    args = run_config._build_parser().parse_args([])

    # Act
    thread_count = run_config.xgb_thread_count(args)

    # Assert
    assert thread_count == ml_model_core.DEFAULT_XGB_PARAMS["n_jobs"]
    assert thread_count == (os.cpu_count() or 1)
    assert not hasattr(args, "wf_n_jobs")


def test_the_removed_stage1_thread_option_no_longer_parses() -> None:
    """``--wf-n-jobs`` is gone; ``--xgb-n-jobs`` covers stage 1 too."""
    # Act & Assert
    with pytest.raises(SystemExit):
        run_config._build_parser().parse_args(["--wf-n-jobs", "2"])


def test_a_config_with_the_removed_stage1_thread_key_is_rejected(tmp_path: Path) -> None:
    """An old config that sets ``wf_n_jobs`` fails with a message naming its replacement."""
    # Arrange
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"wf_n_jobs": 12}', encoding="utf-8")

    # Act & Assert
    with pytest.raises(ValueError, match=r"wf_n_jobs.*xgb_n_jobs"):
        run_config._parse_args(["--config", str(config_path)])


def _thread_count_argv(tmp_path: Path, extra_argv: list[str]) -> list[str]:
    """Build the weekly_run arguments (without the program name) for the thread-count tests."""
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week\n2025,1\n", encoding="utf-8")
    return [
        "--skip-data-refresh",
        "--data-path",
        str(data_path),
        "--run-id",
        "weekly_test",
        "--run-dir",
        str(tmp_path / "run"),
        "--output-dir",
        str(tmp_path / "out"),
        *extra_argv,
    ]


@pytest.mark.parametrize(
    ("extra_argv", "expected"),
    [([], os.cpu_count() or 1), (["--xgb-n-jobs", "3"], 3)],
)
def test_one_thread_count_reaches_stage1_and_the_final_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_argv: list[str],
    expected: int,
) -> None:
    """Stage 1's walk-forward and the final training fit use the same thread count."""

    # Arrange
    class _StopAfterTrainingConfigError(Exception):
        """Stop weekly_run once the final training config has been captured."""

    captured: dict[str, object] = {}

    def _fake_run_wf_compare(
        _df: pd.DataFrame, options: stage1.ProductionOptions, _run: stage1.Stage1Run
    ) -> dict[str, object]:
        """Capture stage 1's thread count and return one usable candidate row."""
        captured["stage1"] = options.xgb_params_overrides["n_jobs"]
        raise _StopAfterTrainingConfigError

    monkeypatch.setattr(walk_forward, "load_games", _empty_games)
    monkeypatch.setattr(artifacts, "sha256_file", returning("hash"))
    monkeypatch.setattr(fingerprints, "dataset_fingerprint", _fixed_fingerprint)
    monkeypatch.setattr(pipeline, "_stage_can_reuse", returning(value=False))
    monkeypatch.setattr(stage1, "evaluate_production", _fake_run_wf_compare)

    argv = _thread_count_argv(tmp_path, extra_argv)
    monkeypatch.setattr(sys, "argv", ["weekly_run.py", *argv])

    # Act & Assert
    with pytest.raises(_StopAfterTrainingConfigError):
        pipeline.main()

    assert captured["stage1"] == expected


@pytest.mark.parametrize(
    ("extra_argv", "expected"),
    [([], os.cpu_count() or 1), (["--xgb-n-jobs", "3"], 3)],
)
def test_the_final_fit_thread_count_follows_the_parsed_arguments(
    tmp_path: Path,
    extra_argv: list[str],
    expected: int,
) -> None:
    """The final training fit reads the same thread count stage 1 receives."""
    # Arrange
    argv = _thread_count_argv(tmp_path, extra_argv)

    # Act
    result = run_config.xgb_thread_count(run_config._parse_args(argv))

    # Assert
    assert result == expected


def test_xgb_device_defaults_to_auto_in_the_parser() -> None:
    """Both stages pick the GPU when one is usable unless told otherwise."""
    # Act
    args = run_config._build_parser().parse_args([])

    # Assert
    assert args.xgb_device == "auto"


def test_xgb_device_defaults_to_auto_in_the_shipped_config() -> None:
    """The shipped config leaves the device on ``auto``."""
    # Act
    args = run_config._parse_args([])

    # Assert
    assert args.xgb_device == "auto"


@pytest.mark.parametrize(
    ("extra_argv", "usable", "expected"),
    [([], True, "cuda"), ([], False, "cpu"), (["--xgb-device", "cpu"], True, "cpu")],
)
def test_one_resolved_device_reaches_stage1_the_final_fit_and_the_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_argv: list[str],
    *,
    usable: bool,
    expected: str,
) -> None:
    """``auto`` resolves once; stage 1, the final fit and the run config all get that device."""

    # Arrange
    class _StopAtFinalFitError(Exception):
        """Stop weekly_run once the final fit's settings have been captured."""

    captured: dict[str, object] = {}
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week\n2025,1\n", encoding="utf-8")
    best_row: dict[str, object] = {"market_mode": "hybrid"}

    def _fake_run_wf_compare(
        _df: pd.DataFrame, options: stage1.ProductionOptions, _run: stage1.Stage1Run
    ) -> dict[str, object]:
        """Capture stage 1's device and return its summary row."""
        captured["stage1"] = options.xgb_params_overrides["device"]
        return best_row

    def _fake_train(options: TrainingOptions) -> None:
        """Capture the final fit's device and stop."""
        optuna_config = options.optuna_config
        assert isinstance(optuna_config, ml_model_core.OptunaConfig)
        captured["final_fit"] = optuna_config.device
        raise _StopAtFinalFitError

    def _fake_run_id(_prefix: str, _hash: str, config: dict[str, object]) -> str:
        """Capture the recorded run config."""
        captured["run_config"] = config["xgb_device"]
        return "weekly_test"

    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: usable)
    monkeypatch.setattr(walk_forward, "load_games", _empty_games)
    monkeypatch.setattr(artifacts, "sha256_file", returning("hash"))
    monkeypatch.setattr(artifacts, "generate_run_id", _fake_run_id)
    monkeypatch.setattr(fingerprints, "dataset_fingerprint", _fixed_fingerprint)
    monkeypatch.setattr(pipeline, "_stage_can_reuse", returning(value=False))
    monkeypatch.setattr(stage1, "evaluate_production", _fake_run_wf_compare)
    monkeypatch.setattr(pipeline, "train_margin_total_model_with_report", _fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weekly_run.py",
            "--skip-data-refresh",
            "--data-path",
            str(data_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--output-dir",
            str(tmp_path / "out"),
            *extra_argv,
        ],
    )

    # Act & Assert
    with pytest.raises(_StopAtFinalFitError):
        pipeline.main()

    assert captured == {"run_config": expected, "stage1": expected, "final_fit": expected}


def test_training_artifacts_record_the_device_the_model_trained_on(tmp_path: Path) -> None:
    """The saved model's metadata names the device its estimator was fitted on.

    The requested device says ``cuda`` here, but the fit fell back to the CPU.
    """
    # Arrange
    result = ml_model_core.TrainingResult(
        model=SimpleNamespace(margin_model=xgb.XGBRegressor(device="cpu")),
        metrics_report={},
        splits={},
        params={"device": "cuda"},
        tuned_params=None,
        feature_list=["feat1"],
        early_stopping={},
    )

    record = ml_model_training.TrainingRecord(
        run_id="weekly_test",
        created_at="2026-09-27T00:00:00+00:00",
        dataset_hash="hash",
        config_payload={"xgb_device": "cuda"},
    )

    # Act
    paths = ml_model_training.write_training_artifacts(result, record, tmp_path)

    # Assert
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["xgb_device"] == "cpu"


def test_the_weekly_parser_records_every_option_argparse_defines() -> None:
    """The parser's own option list matches argparse's internal one, the help option included."""
    # Act
    parser = run_config._build_parser()

    # Assert
    assert [action.dest for action in parser.defined_actions] == [
        action.dest for action in parser._actions
    ]
