"""Tests for backtest script helpers."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

from nfl_predictor.cli import backtest, options
from nfl_predictor.ml import ml_model_xgb_utils, walk_forward
from nfl_predictor.reporting import run_comparison


def test_trend_feature_columns_collects_trend_and_phase_fields() -> None:
    """Trend ablation drops trend and season-phase columns only."""
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

    dropped = backtest._trend_feature_columns(df)

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
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])
    defaults = backtest._parse_args()
    assert defaults.resume is True
    assert defaults.checkpoint_dir == walk_forward.DEFAULT_CHECKPOINT_DIR

    monkeypatch.setattr(
        sys,
        "argv",
        ["walk_forward_backtest.py", "--no-resume", "--checkpoint-dir", "models/elsewhere"],
    )
    overridden = backtest._parse_args()
    assert overridden.resume is False
    assert overridden.checkpoint_dir == Path("models/elsewhere")


def test_main_passes_checkpoint_settings_and_records_the_restore_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI hands its checkpoint settings to the engine and reports what was restored."""
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

    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame({"season": [2023]}))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", lambda _path: "hash")
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

    backtest.main()

    assert captured == {"checkpoint_dir": tmp_path / "checkpoints", "resume": False}
    assert json.loads(out_json.read_text())["config"]["checkpoint"] == checkpoint


def test_xgb_device_defaults_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk-forward picks the GPU when one is usable unless told otherwise."""
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py"])

    assert backtest._parse_args().xgb_device == "auto"


def test_xgb_device_typo_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelled device stops at the parser instead of reaching XGBoost."""
    monkeypatch.setattr(sys, "argv", ["walk_forward_backtest.py", "--xgb-device", "cdua"])

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
    usable: bool,
    expected: str,
) -> None:
    """The engine gets the concrete device, and the run's metadata records it."""
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
    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame({"season": [2023]}))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", lambda _path: "hash")
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

    backtest.main()

    assert (captured[0].xgb_params_overrides or {})["device"] == expected
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["config"]["xgb_device"] == expected
    assert metadata["config"]["xgb_params_overrides"]["device"] == expected


def test_disable_feature_groups_arg_parses_comma_separated_list() -> None:
    """`--disable-feature-groups pbp,other` parses into the expected stripped tuple."""
    old_argv = sys.argv
    try:
        sys.argv = [
            "walk_forward_backtest.py",
            "--disable-feature-groups",
            "pbp,other",
        ]
        args = backtest._parse_args()
    finally:
        sys.argv = old_argv

    assert args.disable_feature_groups == "pbp,other"
    assert options.parse_feature_groups(args.disable_feature_groups) == (
        "pbp",
        "other",
    )


def test_parse_args_accepts_regularization_overrides() -> None:
    """The CLI should expose the regularization knobs used by the noise-family arm."""
    old_argv = sys.argv
    try:
        sys.argv = [
            "walk_forward_backtest.py",
            "--min-child-weight",
            "4.5",
            "--gamma",
            "2.0",
        ]
        args = backtest._parse_args()
    finally:
        sys.argv = old_argv

    assert args.min_child_weight == pytest.approx(4.5)
    assert args.gamma == pytest.approx(2.0)


def test_parse_args_accepts_sigma_calibration() -> None:
    """The residual-sigma calibrator is selectable from the CLI like the other methods."""
    old_argv = sys.argv
    try:
        sys.argv = ["walk_forward_backtest.py", "--calibration", "sigma"]
        args = backtest._parse_args()
    finally:
        sys.argv = old_argv

    assert args.calibration == "sigma"


def test_parse_feature_groups_strips_whitespace_and_drops_empty_entries() -> None:
    """Parsing tolerates surrounding whitespace, empty segments, and a missing/empty value."""
    assert options.parse_feature_groups(" pbp , other ,") == ("pbp", "other")
    assert options.parse_feature_groups(None) == ()
    assert options.parse_feature_groups("") == ()


def test_disable_feature_groups_drops_resolved_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved feature-group columns are the ones dropped from the loaded dataframe."""
    monkeypatch.setattr(
        walk_forward.constants,
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

    dropped = walk_forward.resolve_feature_group_columns(list(df.columns), groups)
    remaining = df.drop(columns=dropped)

    assert dropped == [
        "away_epa_per_play",
        "epa_per_play_diff",
        "home_epa_per_play",
    ]
    assert set(remaining.columns) == {"game_id", "season", "week", "away_total_yards"}


def test_disable_feature_groups_unknown_group_raises_from_cli_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown feature group name raises a clear error, not an obscure library traceback."""
    monkeypatch.setattr(walk_forward.constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ("epa",)})
    groups = options.parse_feature_groups("not_a_real_group")

    with pytest.raises(ValueError, match="not_a_real_group"):
        walk_forward.resolve_feature_group_columns(["away_epa"], groups)


def test_parse_args_accepts_n_estimators_override() -> None:
    """`--n-estimators` parses as an integer and defaults to None when omitted."""
    old_argv = sys.argv
    try:
        sys.argv = ["walk_forward_backtest.py", "--n-estimators", "900"]
        args = backtest._parse_args()
        sys.argv = ["walk_forward_backtest.py"]
        defaults = backtest._parse_args()
    finally:
        sys.argv = old_argv

    assert args.n_estimators == 900
    assert defaults.n_estimators is None


def test_main_forwards_n_estimators_into_the_xgb_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tree-budget flag reaches the engine config as an int, and is absent when omitted."""
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

    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame({"season": [2023]}))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", lambda _path: "hash")
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(walk_forward, "build_metrics_report", fake_report)
    monkeypatch.setattr(walk_forward, "build_metadata", fake_metadata)
    out_json = tmp_path / "metrics_report.json"
    base_argv = [
        "walk_forward_backtest.py",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--out-json",
        str(out_json),
    ]

    monkeypatch.setattr(sys, "argv", [*base_argv, "--n-estimators", "900"])
    backtest.main()

    monkeypatch.setattr(sys, "argv", base_argv)
    backtest.main()

    overrides = captured[0].xgb_params_overrides or {}
    assert overrides["n_estimators"] == 900
    assert isinstance(overrides["n_estimators"], int)
    assert "n_estimators" not in (captured[1].xgb_params_overrides or {})


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

    monkeypatch.setattr(walk_forward, "load_games", lambda _path: pd.DataFrame({"season": [2023]}))
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", lambda _path: "hash")
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(walk_forward, "build_metrics_report", fake_report)
    monkeypatch.setattr(walk_forward, "build_metadata", lambda *_args: {})
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

    backtest.main()

    stability = json.loads(out_json.read_text())["metrics"]["stability"]
    expected = json.loads(json.dumps(run_comparison.stability_report(predictions)))
    assert stability == expected
    assert stability["resamples"] == run_comparison.DEFAULT_RESAMPLES
    assert set(stability["windows"]["week 1"]) == {"all seasons", "2023", "2024"}
    messages = [record.getMessage() for record in caplog.records]
    assert run_comparison.STABILITY_HEADING in messages
    assert any(message.startswith("| 2024 | 1 | ") for message in messages)
