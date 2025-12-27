from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from nfl_predictor.ml import walk_forward


def _fixture_df() -> pd.DataFrame:
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
            "away_score",
            "home_score",
        ]
    ]


def _base_config() -> walk_forward.WalkForwardConfig:
    return walk_forward.WalkForwardConfig(
        eval_seasons=[2023],
        eval_last_n_seasons=1,
        wf_start_week=2,
        calibration="none",
        calibration_weeks=1,
        random_seed=7,
        include_market=False,
        market_anchor=False,
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
    df = _fixture_df()
    folds = walk_forward.build_walk_forward_folds(df, [2023], start_week=2)

    assert folds
    for fold in folds:
        same_season = fold.train_df[fold.train_df["season"] == fold.season]
        assert (same_season["week"] < fold.week).all()


def test_walk_forward_deterministic_outputs() -> None:
    df = _fixture_df()
    config = _base_config()

    result_a = walk_forward.run_walk_forward_backtest(df, config)
    result_b = walk_forward.run_walk_forward_backtest(df, config)

    pdt.assert_frame_equal(result_a["predictions"], result_b["predictions"])
    assert result_a["per_week"] == result_b["per_week"]


def test_walk_forward_probabilities_in_bounds() -> None:
    df = _fixture_df()
    config = _base_config()

    result = walk_forward.run_walk_forward_backtest(df, config)
    probs = result["predictions"]["home_win_prob"]

    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_calibration_data_is_time_aware() -> None:
    df = _fixture_df()
    folds = walk_forward.build_walk_forward_folds(df, [2023], start_week=2)

    for fold in folds:
        calibration_df = walk_forward.select_calibration_data(
            fold.train_df, fold.season, fold.week, calibration_weeks=1
        )
        if not calibration_df.empty:
            assert calibration_df["week"].max() < fold.week
