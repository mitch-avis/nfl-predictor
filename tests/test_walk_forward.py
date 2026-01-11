"""Tests for walk-forward split correctness, determinism, and calibration time-awareness."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

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
    config = walk_forward.WalkForwardConfig(
        **{
            **config.to_dict(),
            "eval_seasons": [2023],
            "include_quantiles": False,
        }
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
    config = walk_forward.WalkForwardConfig(
        **{
            **config.to_dict(),
            "eval_seasons": [2023],
            "market_prob_weight": 1.0,
            "market_prob_clamp": 0.0,
        }
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
        def now(_tz: object) -> "_FixedDatetime":
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
        },
    )

    assert report["run_id"] == "wf_test"
    assert report["metrics"]["overall"]["games"] == 1
    assert report["calibration"]["bin_count"] == walk_forward.RELIABILITY_BINS


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
