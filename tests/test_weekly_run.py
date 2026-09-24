"""Unit tests for the weekly_run orchestration helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import weekly_run


def test_load_config_json(tmp_path: Path) -> None:
    """JSON configs should load into a dict."""
    config_path = tmp_path / "weekly.json"
    config_path.write_text(json.dumps({"wf_eval_last_n_seasons": 4}), encoding="utf-8")
    payload = weekly_run._load_config(config_path)
    assert payload["wf_eval_last_n_seasons"] == 4


def test_load_config_yaml_optional(tmp_path: Path) -> None:
    """YAML configs should parse when PyYAML is available or raise a clear error."""
    config_path = tmp_path / "weekly.yaml"
    config_path.write_text("wf_eval_last_n_seasons: 3\n", encoding="utf-8")
    try:
        payload = weekly_run._load_config(config_path)
    except RuntimeError as exc:
        assert "PyYAML" in str(exc)
    else:
        assert payload["wf_eval_last_n_seasons"] == 3


def test_shipped_weekly_run_config_matches_approved_defaults() -> None:
    """The shipped weekly config should reflect the approved defaults.

    That is 200 trees, no postseason (in evaluation, training or the power rankings), no tuning,
    and no season recency weighting in either the walk-forward stage or the final training fit.
    """
    config_path = Path(__file__).resolve().parents[1] / "config" / "weekly_run.yaml"

    config = weekly_run._load_config(config_path)
    args = weekly_run._build_parser(weekly_run._normalize_config_defaults(config)).parse_args([])

    assert args.wf_n_estimators == 200
    assert args.wf_include_postseason is False
    assert args.include_postseason is False
    assert args.power_rankings_include_postseason is False
    assert args.tune is False
    assert args.wf_recency_half_life_seasons is None
    assert args.wf_recency_half_life_weeks is None
    assert args.train_recency_half_life_seasons is None
    assert args.train_recency_half_life_weeks is None


def test_weekly_run_parser_defaults_follow_shared_xgb_defaults() -> None:
    """Bare weekly-run defaults should match the shared production XGBoost defaults."""
    args = weekly_run._build_parser().parse_args([])
    defaults = weekly_run.ml_model_core.DEFAULT_XGB_PARAMS

    assert args.wf_n_estimators == defaults["n_estimators"]
    assert args.wf_max_depth == defaults["max_depth"]
    assert args.wf_learning_rate == pytest.approx(defaults["learning_rate"])


def test_resolve_predict_path_prefers_latest_week(tmp_path: Path) -> None:
    """When predict_path is None, the newest week file should be chosen."""
    predict_dir = tmp_path / "predict"
    predict_dir.mkdir()
    week_01 = predict_dir / "week_01_games_to_predict.csv"
    week_10 = predict_dir / "week_10_games_to_predict.csv"
    week_01.write_text("season,week\n2025,1\n", encoding="utf-8")
    week_10.write_text("season,week\n2025,10\n", encoding="utf-8")

    resolved = weekly_run._resolve_predict_path(None, tmp_path)
    assert resolved == week_10


def test_resolve_predict_path_prefers_latest_season(tmp_path: Path) -> None:
    """When seasons differ, the newest season should win even if its week number is lower."""
    predict_dir = tmp_path / "predict"
    predict_dir.mkdir()
    week_22 = predict_dir / "week_22_games_to_predict.csv"
    week_01 = predict_dir / "week_01_games_to_predict.csv"
    week_22.write_text("season,week\n2025,22\n", encoding="utf-8")
    week_01.write_text("season,week\n2026,1\n", encoding="utf-8")

    resolved = weekly_run._resolve_predict_path(None, tmp_path)
    assert resolved == week_01


def test_build_confidence_picks_adds_winner_and_rank() -> None:
    """Confidence picks should include predicted winners and ranks."""
    df = pd.DataFrame(
        {
            "away_abbr": ["A", "B"],
            "home_abbr": ["C", "D"],
            "home_win_prob": [0.65, 0.45],
            "away_win_prob": [0.35, 0.55],
        }
    )
    picks = weekly_run._build_confidence_picks(df)
    assert "predicted_winner" in picks.columns
    assert "confidence_rank" in picks.columns
    assert picks["confidence_rank"].is_unique


def test_stage_marker_reuse(tmp_path: Path) -> None:
    """Stage markers should only reuse when hashes match and outputs exist."""
    marker = tmp_path / "stage_state.json"
    output_path = tmp_path / "output.csv"
    output_path.write_text("ok", encoding="utf-8")

    weekly_run._write_stage_marker(
        marker,
        dataset_hash="abc123",
        config_hash="def456",
        stage="wf_compare",
        extra={"outputs": [str(output_path)]},
    )
    assert weekly_run._stage_can_reuse(marker, "abc123", "def456", [output_path])
    assert not weekly_run._stage_can_reuse(marker, "abc123", "wrong", [output_path])


def test_pick_best_row_prefers_deterministic_metrics() -> None:
    """WF candidate selection should use the deterministic probability instrument."""
    rows = [
        {
            "label": "configured-better",
            "brier": 0.20,
            "log_loss": 0.60,
            "deterministic_brier": 0.22,
            "deterministic_log_loss": 0.62,
        },
        {
            "label": "deterministic-better",
            "brier": 0.23,
            "log_loss": 0.63,
            "deterministic_brier": 0.19,
            "deterministic_log_loss": 0.59,
        },
    ]

    assert weekly_run._pick_best_row(rows)["label"] == "deterministic-better"


def test_rank_summary_prefers_deterministic_metrics() -> None:
    """WF comparison ranking should sort by deterministic Brier then deterministic log loss."""
    frame = pd.DataFrame(
        [
            {
                "label": "configured-better",
                "brier": 0.20,
                "log_loss": 0.60,
                "deterministic_brier": 0.22,
                "deterministic_log_loss": 0.62,
            },
            {
                "label": "deterministic-better",
                "brier": 0.23,
                "log_loss": 0.63,
                "deterministic_brier": 0.19,
                "deterministic_log_loss": 0.59,
            },
        ]
    )

    ranked = weekly_run._rank_summary(frame)
    assert ranked.iloc[0]["label"] == "deterministic-better"
    assert ranked.iloc[0]["rank"] == 1


def test_data_refresh_without_arguments_calls_data_collection_bare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no pass-through string the refresh calls data collection with no argv."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_main(*args: object, **kwargs: object) -> None:
        """Record how the data-collection entrypoint was invoked."""
        calls.append((args, kwargs))

    monkeypatch.setattr(weekly_run.data_collection, "main", _fake_main)

    weekly_run._refresh_data(None)
    weekly_run._refresh_data("   ")

    assert calls == [((), {}), ((), {})]


def test_data_refresh_forwards_split_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pass-through string is split shell-style and forwarded as the argv list."""
    captured: list[list[str]] = []

    def _fake_main(argv: list[str] | None = None) -> None:
        """Record the argv list handed to the data-collection entrypoint."""
        assert argv is not None
        captured.append(argv)

    monkeypatch.setattr(weekly_run.data_collection, "main", _fake_main)

    weekly_run._refresh_data("--min-season 2010 --stat-prior-blend-games 4")

    assert captured == [["--min-season", "2010", "--stat-prior-blend-games", "4"]]


def test_data_collection_args_is_a_valid_config_key() -> None:
    """A config file can set the data-collection pass-through arguments."""
    allowed = weekly_run._allowed_config_keys(weekly_run._build_parser())

    assert "data_collection_args" in allowed


def test_data_collection_args_parses_from_the_command_line() -> None:
    """The command-line flag stores the raw pass-through string."""
    args = weekly_run._build_parser().parse_args(["--data-collection-args", "--min-season 2010"])

    assert args.data_collection_args == "--min-season 2010"


def test_weekly_run_stage1_uses_shared_xgb_defaults_when_not_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 1 should evaluate the same XGBoost defaults the final fit uses."""

    class _StopAfterStage1Error(Exception):
        """Stop weekly_run once the Stage 1 config has been captured."""

    captured: dict[str, object] = {}
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week\n2025,1\n", encoding="utf-8")

    def _fake_run_wf_compare(_df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Capture the Stage 1 overrides and stop before later stages run."""
        xgb_params_overrides = kwargs.get("xgb_params_overrides")
        assert isinstance(xgb_params_overrides, dict)
        captured.update(xgb_params_overrides)
        raise _StopAfterStage1Error()

    monkeypatch.setattr(weekly_run.walk_forward, "load_games", lambda _path: pd.DataFrame())
    monkeypatch.setattr(weekly_run.artifacts, "sha256_file", lambda _path: "hash")
    monkeypatch.setattr(
        weekly_run.fingerprints,
        "dataset_fingerprint",
        lambda _path: {"sha256": "fp"},
    )
    monkeypatch.setattr(weekly_run, "_stage_can_reuse", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(weekly_run, "_run_wf_compare", _fake_run_wf_compare)

    old_argv = sys.argv
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
            weekly_run.main()
    finally:
        sys.argv = old_argv

    resolved = weekly_run.ml_model_core._resolve_xgb_params(
        weekly_run.ml_model_core.DEFAULT_XGB_PARAMS,
        overrides=captured,
    )
    defaults = weekly_run.ml_model_core.DEFAULT_XGB_PARAMS

    assert resolved["n_estimators"] == defaults["n_estimators"]
    assert resolved["max_depth"] == defaults["max_depth"]
    assert resolved["learning_rate"] == pytest.approx(defaults["learning_rate"])
    assert resolved["subsample"] == pytest.approx(defaults["subsample"])
    assert resolved["colsample_bytree"] == pytest.approx(defaults["colsample_bytree"])
