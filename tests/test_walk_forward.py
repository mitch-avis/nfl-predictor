"""Tests for walk-forward split correctness, determinism, and calibration time-awareness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

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
