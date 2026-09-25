"""Unit tests for the weekly_run orchestration helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_predictor import data_collection
from nfl_predictor.ml import artifacts, ml_model_core, walk_forward
from nfl_predictor.utils import fingerprints
from nfl_predictor.weekly_run import config as run_config
from nfl_predictor.weekly_run import inputs, pipeline, stage1


def test_load_config_json(tmp_path: Path) -> None:
    """JSON configs should load into a dict."""
    config_path = tmp_path / "weekly.json"
    config_path.write_text(json.dumps({"wf_eval_last_n_seasons": 4}), encoding="utf-8")
    payload = run_config._load_config(config_path)
    assert payload["wf_eval_last_n_seasons"] == 4


def test_load_config_yaml_optional(tmp_path: Path) -> None:
    """YAML configs should parse when PyYAML is available or raise a clear error."""
    config_path = tmp_path / "weekly.yaml"
    config_path.write_text("wf_eval_last_n_seasons: 3\n", encoding="utf-8")
    try:
        payload = run_config._load_config(config_path)
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

    config = run_config._load_config(config_path)
    args = run_config._build_parser(run_config._normalize_config_defaults(config)).parse_args([])

    assert args.wf_n_estimators == 200
    assert args.wf_include_postseason is False
    assert args.include_postseason is False
    assert args.power_rankings_include_postseason is False
    assert args.tune is False
    assert args.wf_recency_half_life_seasons is None
    assert args.train_recency_half_life_seasons is None


def test_weekly_run_parser_defaults_follow_shared_xgb_defaults() -> None:
    """Bare weekly-run defaults should match the shared production XGBoost defaults."""
    args = run_config._build_parser().parse_args([])
    defaults = ml_model_core.DEFAULT_XGB_PARAMS

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

    resolved = inputs._resolve_predict_path(None, tmp_path)
    assert resolved == week_10


def test_resolve_predict_path_prefers_latest_season(tmp_path: Path) -> None:
    """When seasons differ, the newest season should win even if its week number is lower."""
    predict_dir = tmp_path / "predict"
    predict_dir.mkdir()
    week_22 = predict_dir / "week_22_games_to_predict.csv"
    week_01 = predict_dir / "week_01_games_to_predict.csv"
    week_22.write_text("season,week\n2025,22\n", encoding="utf-8")
    week_01.write_text("season,week\n2026,1\n", encoding="utf-8")

    resolved = inputs._resolve_predict_path(None, tmp_path)
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
    picks = pipeline._build_confidence_picks(df)
    assert "predicted_winner" in picks.columns
    assert "confidence_rank" in picks.columns
    assert picks["confidence_rank"].is_unique


def test_stage_marker_reuse(tmp_path: Path) -> None:
    """Stage markers should only reuse when hashes match and outputs exist."""
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
    assert pipeline._stage_can_reuse(marker, "abc123", "def456", [output_path])
    assert not pipeline._stage_can_reuse(marker, "abc123", "wrong", [output_path])


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

    assert stage1._pick_best_row(rows)["label"] == "deterministic-better"


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

    ranked = stage1._rank_summary(frame)
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

    monkeypatch.setattr(data_collection, "main", _fake_main)

    pipeline._refresh_data(None)
    pipeline._refresh_data("   ")

    assert calls == [((), {}), ((), {})]


def test_data_refresh_forwards_split_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pass-through string is split shell-style and forwarded as the argv list."""
    captured: list[list[str]] = []

    def _fake_main(argv: list[str] | None = None) -> None:
        """Record the argv list handed to the data-collection entrypoint."""
        assert argv is not None
        captured.append(argv)

    monkeypatch.setattr(data_collection, "main", _fake_main)

    pipeline._refresh_data("--min-season 2010 --stat-prior-blend-games 4")

    assert captured == [["--min-season", "2010", "--stat-prior-blend-games", "4"]]


def test_data_collection_args_is_a_valid_config_key() -> None:
    """A config file can set the data-collection pass-through arguments."""
    allowed = run_config._allowed_config_keys(run_config._build_parser())

    assert "data_collection_args" in allowed


def test_data_collection_args_parses_from_the_command_line() -> None:
    """The command-line flag stores the raw pass-through string."""
    args = run_config._build_parser().parse_args(["--data-collection-args", "--min-season 2010"])

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

    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame())
    monkeypatch.setattr(artifacts, "sha256_file", lambda _path: "hash")
    monkeypatch.setattr(
        fingerprints,
        "dataset_fingerprint",
        lambda _path: {"sha256": "fp"},
    )
    monkeypatch.setattr(pipeline, "_stage_can_reuse", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(stage1, "_run_wf_compare", _fake_run_wf_compare)

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
            pipeline.main()
    finally:
        sys.argv = old_argv

    resolved = ml_model_core._resolve_xgb_params(
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
    config_path = tmp_path / "weekly.json"
    config_path.write_text(
        '{"tune_n_trials": 7, "train_early_stopping_rounds": 30}', encoding="utf-8"
    )

    args = run_config._parse_args(["--config", str(config_path)])

    assert (args.tune_trials, args.tune_early_stopping_rounds) == (7, 30)


def test_a_config_with_both_names_of_a_key_is_rejected(tmp_path: Path) -> None:
    """Setting a key under both its old and new names is ambiguous and fails."""
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"tune_n_trials": 7, "tune_trials": 8}', encoding="utf-8")

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
    assert getattr(run_config._build_parser().parse_args(argv), dest) == value


def test_the_shipped_config_is_read_by_default_and_matches_the_code_defaults() -> None:
    """Without --config the shipped YAML is read, and it changes nothing a run produces.

    The only differences from the bare code defaults are the recorded config path and
    ``postseason_weight``, which applies only when postseason games are in training (off).
    """
    loaded = vars(run_config._parse_args([]))
    bare = vars(run_config._build_parser().parse_args([]))

    differences = {key for key in bare if loaded[key] != bare[key]}
    assert differences == {"config", "postseason_weight"}
    assert loaded["config"] == run_config.DEFAULT_CONFIG_PATH
    assert loaded["include_postseason"] is False
    assert loaded["wf_win_prob_uncertainty"] == "off"


def test_an_explicit_config_replaces_the_shipped_one(tmp_path: Path) -> None:
    """``--config`` reads only the given file; keys it omits keep their code defaults."""
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"wf_eval_last_n_seasons": 5}', encoding="utf-8")

    args = run_config._parse_args(["--config", str(config_path)])

    assert args.config == config_path
    assert args.wf_eval_last_n_seasons == 5
    assert args.postseason_weight == 1.0


def test_xgb_n_jobs_defaults_to_every_cpu_core() -> None:
    """One thread option sets XGBoost's CPU threads, and it defaults to every core."""
    args = run_config._build_parser().parse_args([])

    assert run_config.xgb_thread_count(args) == ml_model_core.DEFAULT_XGB_PARAMS["n_jobs"]
    assert run_config.xgb_thread_count(args) == (os.cpu_count() or 1)
    assert not hasattr(args, "wf_n_jobs")


def test_the_removed_stage1_thread_option_no_longer_parses() -> None:
    """``--wf-n-jobs`` is gone; ``--xgb-n-jobs`` covers stage 1 too."""
    with pytest.raises(SystemExit):
        run_config._build_parser().parse_args(["--wf-n-jobs", "2"])


def test_a_config_with_the_removed_stage1_thread_key_is_rejected(tmp_path: Path) -> None:
    """An old config that sets ``wf_n_jobs`` fails with a message naming its replacement."""
    config_path = tmp_path / "weekly.json"
    config_path.write_text('{"wf_n_jobs": 12}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"wf_n_jobs.*xgb_n_jobs"):
        run_config._parse_args(["--config", str(config_path)])


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

    class _StopAfterTrainingConfigError(Exception):
        """Stop weekly_run once the final training config has been captured."""

    captured: dict[str, object] = {}
    data_path = tmp_path / "completed_games_ml.csv"
    data_path.write_text("season,week\n2025,1\n", encoding="utf-8")

    def _fake_run_wf_compare(_df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Capture stage 1's thread count and return one usable candidate row."""
        overrides = kwargs.get("xgb_params_overrides")
        assert isinstance(overrides, dict)
        captured["stage1"] = overrides["n_jobs"]
        raise _StopAfterTrainingConfigError()

    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame())
    monkeypatch.setattr(artifacts, "sha256_file", lambda _path: "hash")
    monkeypatch.setattr(fingerprints, "dataset_fingerprint", lambda _path: {"sha256": "fp"})
    monkeypatch.setattr(pipeline, "_stage_can_reuse", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(stage1, "_run_wf_compare", _fake_run_wf_compare)

    argv = [
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
    monkeypatch.setattr(sys, "argv", ["weekly_run.py", *argv])
    with pytest.raises(_StopAfterTrainingConfigError):
        pipeline.main()

    assert captured["stage1"] == expected
    assert run_config.xgb_thread_count(run_config._parse_args(argv)) == expected
