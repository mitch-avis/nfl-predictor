"""Tests for walk-forward split correctness, determinism, and calibration time-awareness."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from nfl_predictor import constants
from nfl_predictor.ml import walk_forward


def _fixture_df() -> pd.DataFrame:
    """Create a tiny deterministic dataset spanning multiple seasons/weeks."""
    rows = []
    for season in (2022, 2023):
        for week in (1, 2, 3):
            for game_idx in (0, 1):
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "game_id": f"{season}_{week}_{game_idx}",
                        "feat1": float(season % 2000) + week + game_idx,
                        "feat2": float(season % 2000) - week + game_idx,
                        "away_score": 17 + week + (game_idx * 3),
                        "home_score": 24 + week - (game_idx * 5),
                        "home_moneyline": -110,
                    }
                )
    df = pd.DataFrame(rows)
    return df[
        [
            "season",
            "week",
            "game_type",
            "game_id",
            "feat1",
            "feat2",
            "home_moneyline",
            "away_score",
            "home_score",
        ]
    ]


def _base_config() -> walk_forward.WalkForwardConfig:
    """Return a walk-forward config suitable for unit tests."""
    return walk_forward.WalkForwardConfig(
        eval_seasons=[2023],
        eval_last_n_seasons=1,
        wf_start_week=2,
        calibration="none",
        calibration_weeks=1,
        random_seed=7,
        include_market=False,
        market_anchor=False,
        market_prob_weight=0.0,
        market_prob_clamp=0.0,
        feature_start="feat1",
        feature_end="feat2",
        xgb_params_overrides={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "verbosity": 0,
        },
    )


def test_walk_forward_split_excludes_eval_week() -> None:
    """Train set must exclude any games from the predicted eval week."""
    df = _fixture_df()
    folds = walk_forward.build_walk_forward_folds(df, [2023], start_week=2)

    assert folds
    for fold in folds:
        same_season = fold.train_df[fold.train_df["season"] == fold.season]
        assert (same_season["week"] < fold.week).all()


def test_summarize_eval_window_flags_incomplete_regular_season() -> None:
    """Summarize eval window reports incomplete regular seasons when data is partial."""
    df = _fixture_df()
    summary = walk_forward.summarize_eval_window(
        df,
        eval_seasons=[2023],
        start_week=2,
        include_postseason=False,
    )

    assert summary["include_postseason"] is False
    assert 2023 in summary["incomplete_seasons"]
    season_summary = summary["seasons"]["2023"]
    assert season_summary["eval_start_week"] == 2
    assert season_summary["incomplete_regular_season"] is True


def test_walk_forward_deterministic_outputs() -> None:
    """Fixed seeds yield identical per-fold outputs."""
    df = _fixture_df()
    config = _base_config()

    result_a = walk_forward.run_walk_forward_backtest(df, config)
    result_b = walk_forward.run_walk_forward_backtest(df, config)

    pdt.assert_frame_equal(result_a["predictions"], result_b["predictions"])
    assert result_a["per_week"] == result_b["per_week"]


def test_walk_forward_probabilities_in_bounds() -> None:
    """Home win probabilities are always in [0, 1]."""
    df = _fixture_df()
    config = _base_config()

    result = walk_forward.run_walk_forward_backtest(df, config)
    probs = result["predictions"]["home_win_prob"]

    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_calibration_data_is_time_aware() -> None:
    """Calibration data must come from weeks strictly before the eval week."""
    df = _fixture_df()
    folds = walk_forward.build_walk_forward_folds(df, [2023], start_week=2)

    for fold in folds:
        calibration_df = walk_forward.select_calibration_data(
            fold.train_df, fold.season, fold.week, calibration_weeks=1
        )
        if not calibration_df.empty:
            assert calibration_df["week"].max() < fold.week


def test_walk_forward_quantile_intervals_monotonic() -> None:
    """Walk-forward outputs include monotonic quantile intervals for margin/total."""
    df = _fixture_df()
    config = _base_config()

    result = walk_forward.run_walk_forward_backtest(df, config)
    preds = result["predictions"]

    required = {
        "predicted_margin_p10",
        "predicted_margin_p50",
        "predicted_margin_p90",
        "predicted_total_p10",
        "predicted_total_p50",
        "predicted_total_p90",
    }
    assert required.issubset(preds.columns)

    assert (preds["predicted_margin_p10"] <= preds["predicted_margin_p50"]).all()
    assert (preds["predicted_margin_p50"] <= preds["predicted_margin_p90"]).all()
    assert (preds["predicted_total_p10"] <= preds["predicted_total_p50"]).all()
    assert (preds["predicted_total_p50"] <= preds["predicted_total_p90"]).all()


def test_walk_forward_can_disable_quantiles() -> None:
    """Walk-forward can skip quantile model training for faster comparisons."""
    df = _fixture_df()
    config = _base_config()
    config = replace(
        config,
        eval_seasons=[2023],
        include_quantiles=False,
    )

    result = walk_forward.run_walk_forward_backtest(df, config)
    preds = result["predictions"]

    assert "predicted_margin" in preds.columns
    assert "predicted_total" in preds.columns
    assert "predicted_margin_p10" not in preds.columns
    assert "predicted_total_p90" not in preds.columns


def test_wf_market_prob_weight_overrides_probs() -> None:
    """When market_prob_weight=1, home_win_prob should match implied market prob."""
    df = _fixture_df()
    config = _base_config()
    config = replace(
        config,
        eval_seasons=[2023],
        market_prob_weight=1.0,
        market_prob_clamp=0.0,
    )

    result = walk_forward.run_walk_forward_backtest(df, config)
    probs = result["predictions"]["home_win_prob"].to_numpy(dtype=float)

    market_prob = 110.0 / (110.0 + 100.0)
    assert np.allclose(probs, market_prob)


def test_dataset_fingerprint_matches_sha256(tmp_path: Path) -> None:
    """Computes SHA-256 fingerprint of file contents."""
    path = Path(tmp_path) / "data.bin"
    payload = b"abc\x00def"
    path.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert walk_forward.dataset_fingerprint(path) == expected


def test_generate_run_id_is_deterministic_under_fixed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builds a stable run_id when datetime is fixed."""

    class _FixedDatetime:
        @staticmethod
        def now(_tz: object) -> _FixedDatetime:
            """Return a fixed datetime for testing."""
            return _FixedDatetime()

        def strftime(self, _fmt: str) -> str:
            """Return a fixed timestamp string for testing."""
            return "20260110_000000"

    monkeypatch.setattr(walk_forward, "datetime", _FixedDatetime)

    cfg = walk_forward.WalkForwardConfig(
        eval_seasons=[2024],
        eval_last_n_seasons=1,
        wf_start_week=3,
        calibration="none",
        include_market=False,
        market_anchor=False,
        include_quantiles=False,
    )
    run_id_a = walk_forward.generate_run_id("deadbeef", cfg)
    run_id_b = walk_forward.generate_run_id("deadbeef", cfg)

    assert run_id_a == run_id_b
    assert run_id_a.startswith("wf_20260110_000000_")
    assert len(run_id_a.split("_")[-1]) == 8


def test_build_metrics_report_shape() -> None:
    """Builds a JSON-serializable metrics report envelope."""
    report = walk_forward.build_metrics_report(
        run_id="wf_test",
        created_at="2026-01-10T00:00:00Z",
        config_payload={"foo": "bar"},
        results={
            "per_week": [{"week": 3, "games": 1}],
            "per_season": [{"season": 2024, "games": 1}],
            "overall": {"games": 1},
            "reliability": [{"bin_lower": 0.0, "bin_upper": 0.1, "count": 1}],
            "eval_window": {"include_postseason": False, "seasons": {}},
        },
    )

    assert report["run_id"] == "wf_test"
    assert report["metrics"]["overall"]["games"] == 1
    assert report["metrics"]["fold_summary"]["folds"] == 1
    assert isinstance(report["metrics"]["summary_table"], list)
    assert report["metric_strategy"]["primary"][0]["metric"] == "brier"
    assert report["calibration"]["bin_count"] == walk_forward.RELIABILITY_BINS
    assert report["splits"]["eval_window"]["include_postseason"] is False
    assert report["splits"]["excluded_incomplete_seasons"] == []


def test_filter_incomplete_eval_seasons_tracks_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filtering incomplete seasons returns kept + dropped lists."""
    df = _fixture_df()
    eval_seasons = [2022, 2023]

    monkeypatch.setattr(walk_forward.constants, "get_regular_season_weeks", lambda _season: 4)
    kept, dropped = walk_forward.filter_incomplete_eval_seasons(df, eval_seasons)
    assert kept == []
    assert dropped == eval_seasons

    monkeypatch.setattr(walk_forward.constants, "get_regular_season_weeks", lambda _season: 3)
    kept, dropped = walk_forward.filter_incomplete_eval_seasons(df, eval_seasons)
    assert kept == eval_seasons
    assert dropped == []


def test_aggregate_metrics_includes_market_residuals_and_interval_coverage() -> None:
    """Computes optional market residual MAE and interval coverage diagnostics."""
    frame = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 3],
            "actual_margin": [3.0, -7.0],
            "predicted_margin": [2.0, -6.0],
            "actual_total": [41.0, 38.0],
            "predicted_total": [40.0, 39.0],
            "actual_home_win": [1, 0],
            "home_win_prob": [0.7, 0.3],
            "expected_points": [1.5, 2.0],
            "actual_points": [1.0, 2.0],
            "pick_correct": [True, True],
            "market_baseline_margin": [1.0, -5.0],
            "market_baseline_total": [39.0, 37.0],
            "predicted_margin_p10": [0.0, -9.0],
            "predicted_margin_p90": [5.0, -3.0],
            "predicted_total_p10": [35.0, 33.0],
            "predicted_total_p90": [47.0, 45.0],
        }
    )

    metrics = walk_forward._aggregate_metrics(frame, market_anchor=True)
    assert metrics["season"] == 2024
    assert metrics["games"] == 2
    assert "market_margin_resid_mae" in metrics
    assert "market_total_resid_mae" in metrics
    assert "margin_p10_p90_coverage" in metrics
    assert "total_p10_p90_coverage" in metrics
    assert "reliability_ece" in metrics


def test_season_win_totals_summary() -> None:
    """Summarize expected vs actual season win totals."""
    predictions = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 4],
            "away_abbr": ["AAA", "BBB"],
            "home_abbr": ["BBB", "AAA"],
            "home_win_prob": [0.7, 0.4],
            "actual_margin": [3.0, -7.0],
        }
    )

    totals = walk_forward._season_win_totals(predictions)
    per_team = {row["team"]: row for row in totals["per_team"]}
    assert totals["overall"]["teams"] == 2
    assert per_team["AAA"]["expected_wins"] == pytest.approx(0.7)
    assert per_team["AAA"]["actual_wins"] == pytest.approx(0.0)
    assert per_team["BBB"]["expected_wins"] == pytest.approx(1.3)
    assert per_team["BBB"]["actual_wins"] == pytest.approx(2.0)
    assert totals["overall"]["mean_abs_error"] == pytest.approx(0.7)


def test_calibration_drift_summary() -> None:
    """Summarize calibration drift by season/week."""
    predictions = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 3],
            "home_win_prob": [0.8, 0.2],
            "actual_home_win": [1, 0],
        }
    )

    drift = walk_forward._calibration_drift(predictions)
    assert len(drift["per_week"]) == 1
    row = drift["per_week"][0]
    assert row["season"] == 2024
    assert row["week"] == 3
    assert row["games"] == 2
    assert row["avg_pred"] == pytest.approx(0.5)
    assert row["avg_actual"] == pytest.approx(0.5)
    assert row["bias"] == pytest.approx(0.0)
    assert row["brier"] == pytest.approx(0.04)


def test_git_commit_hash_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns None when git command fails or returns non-zero."""

    class _Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(walk_forward.subprocess, "run", lambda *_args, **_kwargs: _Result())
    assert walk_forward._git_commit_hash() is None


def test_library_versions_handles_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Records None version when a dependency import fails."""

    def _fake_import(name: str):
        if name == "optuna":
            raise ImportError("missing")
        module = type("M", (), {"__version__": "1.0.0"})
        return module

    monkeypatch.setattr(walk_forward.importlib, "import_module", _fake_import)
    versions = walk_forward._library_versions()
    assert versions["optuna"] is None
    assert versions["numpy"] == "1.0.0"


def test_load_games_and_filter_regular_season_cover_helper_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Game loading and season filtering should handle regular and passthrough paths."""
    path = tmp_path / "games.csv"
    path.write_text("season,week,game_type\n2024,1,REG\n2024,2,WC\n", encoding="utf-8")

    messages: list[str] = []
    monkeypatch.setattr(
        walk_forward.log,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )

    loaded = walk_forward.load_games(path)
    assert len(loaded) == 2
    assert any(str(path) in message for message in messages)

    filtered = walk_forward.filter_regular_season(loaded)
    assert len(filtered) == 1
    assert filtered["game_type"].iloc[0] == "REG"
    assert any("Filtered to regular-season games" in message for message in messages)

    assert walk_forward.filter_regular_season(loaded, include_postseason=True) is loaded
    no_game_type = loaded.drop(columns=["game_type"])
    assert walk_forward.filter_regular_season(no_game_type) is no_game_type


def test_resolve_eval_seasons_and_fold_building_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation-season and fold helpers should cover missing-data and skip branches."""
    empty_df = pd.DataFrame(columns=["season", "week"])
    with pytest.raises(ValueError, match="No seasons available"):
        walk_forward.resolve_eval_seasons(empty_df, None, 1)

    df = _fixture_df()
    messages: list[str] = []
    monkeypatch.setattr(
        walk_forward.log,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )

    assert walk_forward.resolve_eval_seasons(df, [2021, 2023], 1) == [2023]
    assert any("Dropping missing eval seasons" in message for message in messages)

    with pytest.raises(ValueError, match="None of the requested eval seasons"):
        walk_forward.resolve_eval_seasons(df, [1999], 1)

    with pytest.raises(ValueError, match="eval_last_n_seasons must be positive"):
        walk_forward.resolve_eval_seasons(df, None, 0)

    assert walk_forward.resolve_eval_seasons(df, None, 5) == [2022, 2023]

    with pytest.raises(ValueError, match="season and week columns are required"):
        walk_forward.build_walk_forward_folds(pd.DataFrame({"season": [2024]}), [2024], 1)

    single_week_df = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "game_type": ["REG"],
            "away_score": [17],
            "home_score": [24],
        }
    )
    assert walk_forward.build_walk_forward_folds(single_week_df, [2024], 1) == []
    assert any("no training data" in message for message in messages)

    postseason_df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [18, 20],
            "game_type": ["REG", "SB"],
            "away_score": [17, 20],
            "home_score": [24, 27],
        }
    )
    postseason_folds = walk_forward.build_walk_forward_folds(
        postseason_df,
        [2024],
        20,
        include_postseason=True,
    )
    assert len(postseason_folds) == 1
    assert postseason_folds[0].week == 20


def test_calibration_market_and_xgb_helper_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration, market, and XGBoost helper branches should resolve edge cases cleanly."""
    df = _fixture_df()

    assert walk_forward.select_calibration_data(df, 2023, 2, calibration_weeks=0).empty
    assert walk_forward.select_calibration_data(df, 2030, 2, calibration_weeks=1).empty
    assert walk_forward.select_calibration_data(df, 2023, 2, calibration_weeks=5).empty

    postseason_summary = walk_forward.summarize_eval_window(
        pd.DataFrame({"season": [2024], "week": [20]}),
        eval_seasons=[2024],
        start_week=19,
        include_postseason=True,
    )
    assert postseason_summary["seasons"]["2024"]["eval_end_week"] == 20

    messages: list[str] = []
    monkeypatch.setattr(
        walk_forward.log,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )

    no_market_df = pd.DataFrame({"season": [2024], "week": [1]})
    assert walk_forward.resolve_market_settings(no_market_df, True, None, True) == (
        False,
        False,
        False,
    )
    assert any("Market columns missing" in message for message in messages)
    assert any("Market anchor requested" in message for message in messages)

    market_df = pd.DataFrame({"home_spread": [-3.0], "total_line": [44.5]})
    assert walk_forward.resolve_market_settings(market_df, False, None, True) == (
        False,
        True,
        True,
    )

    assert walk_forward._fit_calibrator(np.array([1.0]), np.array([1]), "none") is None
    assert walk_forward._fit_calibrator(np.array([1.0, 2.0]), np.array([1, 1]), "platt") is None

    sentinel = object()
    monkeypatch.setattr(
        walk_forward.ml_model,
        "_fit_win_prob_calibrator",
        lambda *args, **kwargs: sentinel,
    )
    assert (
        walk_forward._fit_calibrator(
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            "platt",
            sample_weight=np.array([1.0, 0.5]),
        )
        is sentinel
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        walk_forward.ml_model,
        "_resolve_xgb_params",
        lambda _defaults, overrides: (
            captured.setdefault("overrides", dict(overrides)) or dict(overrides)
        ),
    )
    config = replace(_base_config(), xgb_params_overrides={"max_depth": 4}, random_seed=99)
    walk_forward._resolve_xgb_params(config)
    assert captured["overrides"] == {"max_depth": 4, "random_state": 99}


def test_walk_forward_backtest_covers_market_anchor_uncertainty_and_callback() -> None:
    """A market-aware walk-forward run should exercise calibration, quantiles, and callbacks."""
    df = _fixture_df().copy()
    df["home_spread"] = -3.0
    df["total_line"] = 44.5
    df["away_moneyline"] = 100

    config = replace(
        _base_config(),
        calibration="platt",
        include_market=True,
        market_anchor=True,
        market_prob_weight=0.25,
        market_prob_clamp=0.05,
        win_prob_use_uncertainty=True,
        include_quantiles=True,
    )

    folds: list[tuple[int, int, str]] = []

    def _capture(metrics: dict[str, object], fold: walk_forward.WalkForwardFold) -> None:
        """Record the fold callback payload for assertions."""
        folds.append((fold.season, fold.week, str(metrics["calibration_method"])))

    result = walk_forward.run_walk_forward_backtest(df, config, fold_callback=_capture)

    assert len(folds) == 2
    assert result["resolved_settings"]["market_anchor"] is True
    assert result["resolved_settings"]["win_prob_use_uncertainty"] is True
    assert "market_baseline_margin" in result["predictions"].columns
    assert "market_baseline_total" in result["predictions"].columns
    assert "predicted_margin_p10" in result["predictions"].columns
    assert result["excluded_incomplete_seasons"] == []
    assert result["eval_window"]["include_postseason"] is False
    assert set(result["predictions"]["calibration_method"].unique()) == {"platt"}


def test_walk_forward_backtest_handles_elo_uncertainty_and_incomplete_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walk-forward should downgrade unsupported elo uncertainty.

    It should also fail cleanly when incomplete-season filtering removes every eval season.
    """
    df = _fixture_df()
    info_messages: list[str] = []
    monkeypatch.setattr(
        walk_forward.log,
        "info",
        lambda message, *args: info_messages.append(message % args if args else message),
    )

    elo_config = replace(
        _base_config(),
        calibration="elo",
        win_prob_use_uncertainty=True,
        include_quantiles=True,
    )
    elo_result = walk_forward.run_walk_forward_backtest(df, elo_config)
    assert set(elo_result["predictions"]["calibration_method"].unique()) == {"none"}
    assert any("Elo calibration ignored" in message for message in info_messages)

    monkeypatch.setattr(walk_forward.constants, "get_regular_season_weeks", lambda _season: 99)
    incomplete_config = replace(_base_config(), exclude_incomplete_seasons=True)
    with pytest.raises(ValueError, match="No complete seasons available"):
        walk_forward.run_walk_forward_backtest(df, incomplete_config)


def test_resolve_feature_group_columns_matches_configured_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group with non-empty markers resolves to the columns whose names contain a marker."""
    monkeypatch.setattr(
        walk_forward.constants,
        "FEATURE_GROUP_COLUMN_MARKERS",
        {"pbp": ("epa",), "other": ("success_rate",)},
    )
    columns = ["away_epa_per_play", "home_success_rate", "week", "season"]

    resolved = walk_forward.resolve_feature_group_columns(columns, ["pbp"])

    assert resolved == ["away_epa_per_play"]


def test_resolve_feature_group_columns_empty_markers_matches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty marker tuple for a group resolves to zero columns, dropping nothing."""
    monkeypatch.setattr(walk_forward.constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ()})
    columns = ["away_epa_per_play", "home_epa_per_play", "epa_per_play_diff"]

    resolved = walk_forward.resolve_feature_group_columns(columns, ["pbp"])

    assert resolved == []


def test_resolve_feature_group_columns_substring_matches_prefix_and_suffix_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single marker catches away_/home_/opponent_ prefixed and _diff suffixed columns."""
    monkeypatch.setattr(
        walk_forward.constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ("epa_per_play",)}
    )
    columns = [
        "away_epa_per_play",
        "home_epa_per_play",
        "opponent_epa_per_play",
        "epa_per_play_diff",
        "unrelated_column",
    ]

    resolved = walk_forward.resolve_feature_group_columns(columns, ["pbp"])

    assert resolved == sorted(
        [
            "away_epa_per_play",
            "home_epa_per_play",
            "opponent_epa_per_play",
            "epa_per_play_diff",
        ]
    )


def test_resolve_feature_group_columns_unknown_group_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown group name raises a ValueError naming the bad group."""
    monkeypatch.setattr(walk_forward.constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ("epa",)})

    with pytest.raises(ValueError, match="not_a_real_group"):
        walk_forward.resolve_feature_group_columns(["away_epa"], ["not_a_real_group"])


def test_walk_forward_config_disabled_feature_groups_serializes_to_list() -> None:
    """`disabled_feature_groups` round-trips through `to_dict` as a list, defaulting to empty."""
    default_config = _base_config()
    assert default_config.to_dict()["disabled_feature_groups"] == []

    configured = replace(_base_config(), disabled_feature_groups=("pbp",))
    assert configured.to_dict()["disabled_feature_groups"] == ["pbp"]


def test_run_walk_forward_backtest_drops_disabled_feature_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config alone is enough to ablate a group, without the caller pre-filtering.

    The CLI scripts drop the columns before building the config, so this guards a direct
    caller of `run_walk_forward_backtest` against silently getting no ablation at all.
    """
    monkeypatch.setattr(
        constants,
        "FEATURE_GROUP_COLUMN_MARKERS",
        {"demo": ("widget",)},
    )

    captured: dict[str, list[str]] = {}

    def fake_filter(df: pd.DataFrame, include_postseason: bool = False) -> pd.DataFrame:
        captured["columns"] = list(df.columns)
        raise RuntimeError("stop after the drop")

    monkeypatch.setattr(walk_forward, "filter_regular_season", fake_filter)

    df = pd.DataFrame(
        {
            "season": [2023],
            "week": [3],
            "away_widget_rate": [1.0],
            "home_widget_rate": [2.0],
            "widget_rate_diff": [-1.0],
            "away_rest": [7],
        }
    )
    config = walk_forward.WalkForwardConfig(disabled_feature_groups=("demo",))

    with pytest.raises(RuntimeError, match="stop after the drop"):
        walk_forward.run_walk_forward_backtest(df, config)

    assert captured["columns"] == ["season", "week", "away_rest"]
