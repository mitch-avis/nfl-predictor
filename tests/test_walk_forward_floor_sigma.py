"""The walk-forward's probability floor uses the spread of its own earlier out-of-fold errors."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from scipy.stats import norm

from nfl_predictor import constants
from nfl_predictor.ml import floor_sigma, walk_forward

if TYPE_CHECKING:
    from pathlib import Path

SEASONS = (2018, 2019, 2020, 2021, 2022, 2023)
EVAL_SEASONS = SEASONS[1:]
WEEKS = (1, 2, 3)


def _fixture_df() -> pd.DataFrame:
    """Four games a week over six short seasons, with seeded noise in the scores."""
    rng = np.random.default_rng(11)
    rows = []
    for season in SEASONS:
        for week in WEEKS:
            for game in range(4):
                edge = float(rng.normal(0, 4))
                margin = round(edge + float(rng.normal(0, 10)))
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "game_id": f"{season}_{week:02d}_{game}",
                        "feat1": edge + float(rng.normal(0, 1)),
                        "feat2": float(rng.normal(0, 1)),
                        "away_score": 20,
                        "home_score": 20 + margin,
                    }
                )
    return pd.DataFrame(rows)


def _config() -> walk_forward.WalkForwardConfig:
    """Every week of every season after the first, as the standalone reference ran."""
    return walk_forward.WalkForwardConfig(
        eval_seasons=list(EVAL_SEASONS),
        wf_start_week=1,
        random_seed=7,
        include_market=False,
        market_anchor=False,
        include_quantiles=False,
        feature_start="feat1",
        feature_end="feat2",
        xgb_params_overrides={
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.3,
            "n_jobs": 1,
            "verbosity": 0,
            "device": "cpu",
        },
    )


def _measured_expanding_sigma(frame: pd.DataFrame) -> np.ndarray:
    """Return the ``expanding`` candidate as the reference measurement computed it, per game.

    For the fold of season ``s``, week ``w``: the root-mean-square error of every earlier fold
    of the same run (earlier seasons, and season ``s`` weeks before ``w``). ``frame`` is sorted
    by ``game_id``.
    """
    err2 = (frame["actual_margin"] - frame["predicted_margin"]).to_numpy(float) ** 2
    season = frame["season"].to_numpy(int)
    week = frame["week"].to_numpy(int)
    out = np.full(len(frame), np.nan)
    for s, w in sorted(set(zip(season, week, strict=True))):
        fold = (season == s) & (week == w)
        earlier = (season < s) | ((season == s) & (week < w))
        out[fold] = np.sqrt(err2[earlier].mean()) if earlier.any() else np.nan
    return out


@pytest.fixture(scope="module")
def standalone() -> dict[str, object]:
    """One standalone walk-forward over the fixture, with no supplied history."""
    return walk_forward.run_walk_forward_backtest(_fixture_df(), _config())


def test_a_standalone_run_reproduces_the_measured_expanding_sigma(
    standalone: dict[str, object],
) -> None:
    """From the fourth season on, each week's sigma is the measured candidate, bit for bit."""
    predictions = standalone["predictions"]
    assert isinstance(predictions, pd.DataFrame)
    frame = predictions.sort_values("game_id").reset_index(drop=True)
    scored = (
        frame["season"] >= EVAL_SEASONS[0] + constants.FLOOR_SIGMA_MIN_POOL_SEASONS
    ).to_numpy()

    expected = _measured_expanding_sigma(frame)

    np.testing.assert_array_equal(frame.loc[scored, "floor_sigma"].to_numpy(), expected[scored])
    assert not frame.loc[scored, "floor_sigma_fallback"].any()
    for _, week in frame[scored].groupby(["season", "week"]):
        np.testing.assert_array_equal(
            week["deterministic_home_win_prob"].to_numpy(),
            norm.cdf(week["predicted_margin"].to_numpy() / float(week["floor_sigma"].iloc[0])),
        )


def test_the_first_three_seasons_use_the_constant(standalone: dict[str, object]) -> None:
    """Before the pool spans three earlier seasons (any weeks) the floor falls back."""
    predictions = standalone["predictions"]
    assert isinstance(predictions, pd.DataFrame)
    early = predictions[predictions["season"] < EVAL_SEASONS[0] + 3]

    assert early["floor_sigma_fallback"].all()
    assert (early["floor_sigma"] == constants.SCORE_DIFF_STD_DEV).all()
    np.testing.assert_array_equal(
        early["deterministic_home_win_prob"].to_numpy(),
        norm.cdf(early["predicted_margin"].to_numpy() / constants.SCORE_DIFF_STD_DEV),
    )


def test_every_week_records_its_sigma(standalone: dict[str, object]) -> None:
    """Each week's metrics carry the sigma, whether it fell back and the pool it came from."""
    per_week = standalone["per_week"]
    assert isinstance(per_week, list)
    first, last = per_week[0], per_week[-1]

    assert first["floor_sigma_fallback"] is True
    assert first["floor_sigma_pool_games"] == 0
    assert last["floor_sigma_fallback"] is False
    assert last["floor_sigma_pool_games"] == 4 * (len(EVAL_SEASONS) * len(WEEKS) - 1)
    assert standalone["floor_sigma"] == {
        "min_pool_seasons": constants.FLOOR_SIGMA_MIN_POOL_SEASONS,
        "fallback_sigma": constants.SCORE_DIFF_STD_DEV,
        "history_sources": [],
        "history_games": 0,
    }


def test_no_week_sigma_reads_its_own_or_a_later_week() -> None:
    """Changing the results of the last week changes no week's sigma or probability."""
    df = _fixture_df()
    last = (df["season"] == SEASONS[-1]) & (df["week"] == WEEKS[-1])
    changed = df.copy()
    changed.loc[last, "home_score"] = changed.loc[last, "home_score"] + 30

    base = walk_forward.run_walk_forward_backtest(df, _config())["predictions"]
    moved = walk_forward.run_walk_forward_backtest(changed, _config())["predictions"]

    pdt.assert_series_equal(base["floor_sigma"], moved["floor_sigma"])
    pdt.assert_series_equal(
        base["deterministic_home_win_prob"], moved["deterministic_home_win_prob"]
    )


def _history_before_the_run() -> floor_sigma.ErrorPool:
    """Errors of three seasons before the fixture, and of one week the run predicts itself."""
    rows = [
        {"game_id": f"h{season}_{week}", "season": season, "week": week, "squared_error": 100.0}
        for season in (2015, 2016, 2017)
        for week in WEEKS
    ]
    rows.append({"game_id": "h2019_01", "season": 2019, "week": 1, "squared_error": 1e6})
    return floor_sigma.ErrorPool(pd.DataFrame(rows), ("reference",))


def test_a_supplied_history_joins_the_pool_and_the_runs_weeks_replace_it() -> None:
    """History before the run counts from the first week; history at the run's weeks does not."""
    history = _history_before_the_run()

    result = walk_forward.run_walk_forward_backtest(
        _fixture_df(), _config(), floor_sigma_history=history
    )
    predictions = result["predictions"]
    per_week = {(row["season"], row["week"]): row for row in result["per_week"]}

    first = predictions[(predictions["season"] == 2019) & (predictions["week"] == 1)]
    assert (first["floor_sigma"] == 10.0).all()
    assert not first["floor_sigma_fallback"].any()
    # 2019 week 2: nine history games and the run's own four from week 1, not the 1e6 row.
    own = predictions[(predictions["season"] == 2019) & (predictions["week"] == 1)]
    own_err2 = ((own["actual_margin"] - own["predicted_margin"]) ** 2).to_numpy()
    expected = np.sqrt(np.concatenate([np.full(9, 100.0), own_err2]).mean())
    assert per_week[(2019, 2)]["floor_sigma"] == pytest.approx(expected, rel=1e-12)
    assert result["floor_sigma"]["history_sources"] == ["reference"]
    assert result["floor_sigma"]["history_games"] == 10


def test_sigma_changes_probabilities_but_never_margins_picks_or_ranks(
    standalone: dict[str, object],
) -> None:
    """With and without a history the margins, picks and confidence ranks are identical."""
    with_history = walk_forward.run_walk_forward_backtest(
        _fixture_df(), _config(), floor_sigma_history=_history_before_the_run()
    )["predictions"]
    without = standalone["predictions"]
    assert isinstance(without, pd.DataFrame)

    for column in ("predicted_margin", "predicted_total", "confidence_rank", "pick_correct"):
        pdt.assert_series_equal(with_history[column], without[column])
    assert ((with_history["home_win_prob"] > 0.5) == (without["home_win_prob"] > 0.5)).all()
    assert not np.allclose(with_history["home_win_prob"], without["home_win_prob"])


def test_the_history_is_part_of_the_checkpoint_fingerprint() -> None:
    """A run with another history never restores this one's weeks."""
    df, config = _fixture_df(), _config()
    history = _history_before_the_run()

    plain = walk_forward.fold_checkpoint_fingerprint(df, config)
    with_history = walk_forward.fold_checkpoint_fingerprint(df, config, history)

    assert plain != with_history
    assert with_history == walk_forward.fold_checkpoint_fingerprint(df, config, history)


def test_restored_weeks_feed_the_pool_like_trained_ones(tmp_path: Path) -> None:
    """A resumed run returns the sigmas of an uninterrupted one."""
    df = _fixture_df()
    config = replace(_config(), eval_seasons=list(EVAL_SEASONS[:4]))
    first = walk_forward.run_walk_forward_backtest(
        df, config, checkpoints=walk_forward.FoldCheckpoints(tmp_path)
    )
    for path in sorted(tmp_path.rglob("fold_2021_*.joblib")):
        path.unlink()

    resumed = walk_forward.run_walk_forward_backtest(
        df, config, checkpoints=walk_forward.FoldCheckpoints(tmp_path)
    )

    pdt.assert_frame_equal(first["predictions"], resumed["predictions"])
