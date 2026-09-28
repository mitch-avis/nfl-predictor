"""Every confidence-ranking site breaks mathematically equal confidences by ``game_id``.

A home favorite and a home underdog by the same number of points have equal confidence
``|p - 0.5|`` in exact arithmetic, but ``Phi(m / s) - 0.5`` and ``0.5 - Phi(-m / s)`` differ in
the last bits, so without quantizing the confidence the float noise, not ``game_id``, decides
their order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor.ml import metrics, ml_utils
from nfl_predictor.ml.ml_model_core import (
    _build_prediction_output,
    _margin_to_home_win_prob,
    _summarize_confidence_pool,
)
from nfl_predictor.reporting import run_comparison
from nfl_predictor.weekly_run import pipeline

# game "a" is the home underdog by 3, game "b" the home favorite by 3.
TIED_MARGINS = np.array([-3.0, 3.0])
TIED_GAME_IDS = np.array(["2026_01_AAA_BBB", "2026_01_CCC_DDD"], dtype=object)


def _tied_probs() -> np.ndarray:
    probs = _margin_to_home_win_prob(TIED_MARGINS)
    raw = np.abs(probs - 0.5)
    # Precondition: float noise makes the second game look less confident than the first.
    assert raw[1] < raw[0]
    return probs


def test_float_noise_pair_is_one_mathematical_tie() -> None:
    """The home -3 / home +3 pair shares one quantized confidence."""
    strength = metrics.confidence_strength(_tied_probs())

    assert strength[0] == strength[1]


def test_confidence_ranks_break_the_tie_by_game_id() -> None:
    """The shared ranking rule orders the pair by game_id."""
    ranks = metrics.confidence_ranks(_tied_probs(), tiebreaker=TIED_GAME_IDS)

    assert ranks.tolist() == [1, 2]


def test_confidence_ranks_without_tiebreaker_keep_input_order() -> None:
    """With no tiebreaker, tied games keep their input order."""
    ranks = metrics.confidence_ranks(_tied_probs())

    assert ranks.tolist() == [1, 2]


def test_confidence_ranks_restart_within_each_group() -> None:
    """Grouped ranking numbers each (season, week) from 1, ties by game_id."""
    probs = np.concatenate([_tied_probs(), [0.9], _tied_probs()])
    game_ids = np.array(["w1_b", "w1_c", "w2_a", "w2_b", "w2_c"], dtype=object)
    seasons = np.array([2026] * 5)
    weeks = np.array([1, 1, 2, 2, 2])

    ranks = metrics.confidence_ranks(probs, tiebreaker=game_ids, groups=(seasons, weeks))

    assert ranks.tolist() == [1, 2, 3, 1, 2]


def test_confidence_pool_columns_break_the_tie_by_game_id() -> None:
    """Walk-forward pool columns rank the tied pair in game_id order."""
    cols = metrics.confidence_pool_columns(
        _tied_probs(),
        home_score=np.array([20.0, 20.0]),
        away_score=np.array([17.0, 17.0]),
        tiebreaker=TIED_GAME_IDS,
    )

    assert cols["confidence_rank"].tolist() == [1, 2]


def test_prediction_output_ranks_break_the_tie_by_game_id() -> None:
    """Production pick ranks (on the 4-decimal probabilities) follow game_id order."""
    # At 2 points the 4-decimal probabilities carry the float noise in the other direction.
    probs = _margin_to_home_win_prob(np.array([2.0, -2.0]))
    rounded = np.round(probs, 4)
    assert abs(rounded[0] - 0.5) < abs(rounded[1] - 0.5)
    games = pd.DataFrame(
        {
            "game_id": ["2026_01_ZZZ_YYY", "2026_01_AAA_BBB"],
            "away_abbr": ["ZZZ", "AAA"],
            "home_abbr": ["YYY", "BBB"],
        }
    )

    output = _build_prediction_output(games, np.array([20.0, 22.0]), np.array([22.0, 20.0]), probs)

    assert output["confidence_rank"].tolist() == [2, 1]
    assert output["confidence_strength"].iloc[0] == output["confidence_strength"].iloc[1]


def test_prediction_output_picks_the_side_from_the_unrounded_probability() -> None:
    """A home probability of 0.49996 publishes as 0.5 but picks the away side, as walk-forward does.

    At exactly 0.5 the home side is picked (``p >= 0.5``), the walk-forward's rule.
    """
    probs = np.array([0.49996, 0.5])
    games = pd.DataFrame(
        {
            "game_id": ["2026_01_AAA_BBB", "2026_01_CCC_DDD"],
            "away_abbr": ["AAA", "CCC"],
            "home_abbr": ["BBB", "DDD"],
        }
    )

    output = _build_prediction_output(games, np.array([20.0, 20.0]), np.array([20.0, 20.0]), probs)

    assert output["home_win_prob"].tolist() == [0.5, 0.5]
    assert output["predicted_winner"].tolist() == ["AAA", "DDD"]
    walk_forward_home = metrics.confidence_pool_columns(
        probs, home_score=np.array([20.0, 20.0]), away_score=np.array([17.0, 17.0])
    )["pick_correct"]
    assert walk_forward_home.tolist() == [False, True]


def test_prediction_output_ranks_on_the_unrounded_probability() -> None:
    """Two games that share a 4-decimal probability rank by their unrounded values.

    Game "a" sorts first by game_id, so ranking on the published (rounded) probabilities would
    call it the less confident of a tie; unrounded, it is the more confident one.
    """
    probs = np.array([0.612364, 0.612356])
    assert np.round(probs[0], 4) == np.round(probs[1], 4)
    games = pd.DataFrame(
        {
            "game_id": ["2026_01_AAA_BBB", "2026_01_CCC_DDD"],
            "away_abbr": ["AAA", "CCC"],
            "home_abbr": ["BBB", "DDD"],
        }
    )

    output = _build_prediction_output(games, np.array([20.0, 20.0]), np.array([22.0, 22.0]), probs)

    assert output["confidence_rank"].tolist() == [2, 1]
    assert output["home_win_prob"].tolist() == [0.6124, 0.6124]
    assert output["confidence_strength"].tolist() == metrics.confidence_strength(probs).tolist()


def test_summarize_confidence_pool_breaks_the_tie_by_game_id() -> None:
    """The training pool summary ranks the tied pair by game_id: the correct pick scores 2."""
    df = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 1],
            "game_id": TIED_GAME_IDS,
            "away_score": [17.0, 17.0],
            "home_score": [20.0, 20.0],
        }
    )

    summary = _summarize_confidence_pool(df, _tied_probs(), ("away_score", "home_score"))

    # Game "a" picks the away side and loses (rank 1); game "b" picks home and wins (rank 2).
    assert summary["weekly_actual_points_avg"] == 2.0


def _comparison_rows(probs: np.ndarray, game_ids: np.ndarray) -> pd.DataFrame:
    n = len(probs)
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "season": [2026] * n,
            "week": [1] * n,
            "deterministic_home_win_prob": probs,
            "actual_home_win": [1] * n,
            "actual_margin": [3.0] * n,
            "predicted_margin": [0.0] * n,
            "predicted_total": [44.0] * n,
            "actual_total": [44.0] * n,
            "market_home_win_prob": probs,
        }
    )


def test_per_game_scores_break_the_tie_by_game_id() -> None:
    """The compare pool ranks follow game_id order for the tied pair."""
    games = run_comparison.per_game_scores(_comparison_rows(_tied_probs(), TIED_GAME_IDS))

    assert games["rank"].tolist() == [1, 2]
    assert games["pool"].tolist() == [0.0, 2.0]


def test_confidence_picks_fallback_breaks_the_tie_by_game_id() -> None:
    """The weekly picks fallback ranks the tied pair by game_id."""
    probs = _tied_probs()
    picks = pipeline._build_confidence_picks(
        pd.DataFrame(
            {
                "game_id": TIED_GAME_IDS,
                "away_abbr": ["AAA", "CCC"],
                "home_abbr": ["BBB", "DDD"],
                "home_win_prob": probs,
                "away_win_prob": 1 - probs,
            }
        )
    )

    ranks = dict(zip(picks["game_id"], picks["confidence_rank"], strict=True))
    assert ranks == {TIED_GAME_IDS[0]: 1, TIED_GAME_IDS[1]: 2}


def test_untied_games_rank_exactly_as_before() -> None:
    """Games whose confidences differ rank by raw |p - 0.5|, as they always did."""
    rng = np.random.default_rng(7)
    probs = rng.uniform(0.02, 0.98, size=200)
    game_ids = np.array([f"g{i:03d}" for i in range(200)], dtype=object)
    raw_order = np.argsort(np.abs(probs - 0.5), kind="mergesort")
    expected = np.empty(200, dtype=int)
    expected[raw_order] = np.arange(1, 201)

    shuffled_ids = rng.permutation(game_ids)
    assert metrics.confidence_ranks(probs, tiebreaker=shuffled_ids).tolist() == expected.tolist()
    cols = metrics.confidence_pool_columns(
        probs, np.full(200, 20.0), np.full(200, 17.0), tiebreaker=shuffled_ids
    )
    assert cols["confidence_rank"].tolist() == expected.tolist()
    games = run_comparison.per_game_scores(_comparison_rows(probs, game_ids))
    assert games["rank"].tolist() == expected.tolist()


def test_display_weekly_predictions_orders_the_tie_by_game_id(monkeypatch) -> None:
    """Without a rank column, the log display lists games by the shared ranking rule."""
    messages: list[str] = []
    monkeypatch.setattr(ml_utils.log, "info", lambda msg, *args: messages.append(msg % args))
    probs = _tied_probs()
    ml_utils.display_weekly_predictions(
        pd.DataFrame(
            {
                "game_id": TIED_GAME_IDS,
                "away_abbr": ["AAA", "CCC"],
                "home_abbr": ["BBB", "DDD"],
                "home_win_prob": probs,
                "away_win_prob": 1 - probs,
                "confidence_strength": np.abs(probs - 0.5),
            }
        )
    )

    games = [msg for msg in messages if msg.startswith("#")]
    # Most confident first: the later game_id holds rank 2 of the tied pair.
    assert "CCC" in games[0]
    assert "AAA" in games[1]
