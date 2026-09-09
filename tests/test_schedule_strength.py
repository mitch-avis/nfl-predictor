"""Tests for opponent-averaged schedule-strength helpers.

Every expectation in this module is a hand-computed arithmetic expression over the
fixture values, never a re-derivation of the implementation.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor.utils.polars import schedule_strength

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

# Six teams across weeks 1-5 of 2024. BUF/MIA meet in weeks 1 and 3 (twice before the
# analysis week) and MIA/NE meet in weeks 4 and 5 (twice on or after it), so the
# "faced twice counts once" rule is exercised on both sides. NYJ and DEN are idle in
# week 4, which supplies the bye-week case.
_SCHEDULE_GAMES: list[tuple[int, str, str]] = [
    (1, "BUF", "MIA"),
    (1, "NYJ", "NE"),
    (1, "KC", "DEN"),
    (2, "BUF", "NYJ"),
    (2, "MIA", "KC"),
    (2, "NE", "DEN"),
    (3, "MIA", "BUF"),
    (3, "NE", "KC"),
    (3, "NYJ", "DEN"),
    (4, "MIA", "NE"),
    (4, "BUF", "KC"),
    (5, "NE", "MIA"),
    (5, "NYJ", "BUF"),
    (5, "KC", "DEN"),
]

_RATING_VALUES: dict[str, float] = {
    "BUF": 1.0,
    "MIA": 2.0,
    "NYJ": 3.0,
    "NE": 4.0,
    "KC": 5.0,
    "DEN": 6.0,
}

# One row per team-game for weeks 1-3 of 2024, mirroring the schedule above. The margin
# of the visiting team is listed here; the home team's mirror row carries its negation.
_TEAM_GAME_MARGINS: list[tuple[int, str, str, float]] = [
    (1, "BUF", "MIA", 0.10),
    (1, "NYJ", "NE", 0.20),
    (1, "KC", "DEN", 0.30),
    (2, "BUF", "NYJ", 0.40),
    (2, "MIA", "KC", 0.50),
    (2, "NE", "DEN", 0.60),
    (3, "MIA", "BUF", 0.70),
    (3, "NE", "KC", 0.80),
    (3, "NYJ", "DEN", 0.90),
]


def _schedule_frame(
    games: list[tuple[int, str, str]] | None = None,
    *,
    season: int = 2024,
) -> pl.DataFrame:
    """Build a schedule frame using the repo's away/home abbreviation column names."""
    rows = _SCHEDULE_GAMES if games is None else games
    return pl.DataFrame(
        {
            "season": pl.Series([season] * len(rows), dtype=pl.Int64),
            "week": pl.Series([week for week, _, _ in rows], dtype=pl.Int64),
            "away_abbr": pl.Series([away for _, away, _ in rows], dtype=pl.Utf8),
            "home_abbr": pl.Series([home for _, _, home in rows], dtype=pl.Utf8),
        }
    )


def _ratings_frame(values: dict[str, float] | None = None) -> pl.DataFrame:
    """Build a pre-week ratings snapshot keyed by team abbreviation."""
    ratings = _RATING_VALUES if values is None else values
    return pl.DataFrame(
        {
            "team_abbr": pl.Series(list(ratings), dtype=pl.Utf8),
            "rating": pl.Series(list(ratings.values()), dtype=pl.Float64),
        }
    )


def _team_games_frame(
    margins: list[tuple[int, str, str, float]] | None = None,
    *,
    season: int = 2024,
) -> pl.DataFrame:
    """Expand per-game margins into one row per team-game with both mirror rows."""
    rows = _TEAM_GAME_MARGINS if margins is None else margins
    seasons: list[int] = []
    weeks: list[int] = []
    teams: list[str] = []
    opponents: list[str] = []
    values: list[float] = []
    for week, away, home, margin in rows:
        seasons.extend([season, season])
        weeks.extend([week, week])
        teams.extend([away, home])
        opponents.extend([home, away])
        values.extend([margin, -margin])
    return pl.DataFrame(
        {
            "season": pl.Series(seasons, dtype=pl.Int64),
            "week": pl.Series(weeks, dtype=pl.Int64),
            "team_abbr": pl.Series(teams, dtype=pl.Utf8),
            "opponent_abbr": pl.Series(opponents, dtype=pl.Utf8),
            "epa_margin_per_play": pl.Series(values, dtype=pl.Float64),
        }
    )


def _played_adj(result: pl.DataFrame, team: str) -> float | None:
    """Return one team's played adjusted schedule strength from a result frame."""
    return result.filter(pl.col("team_abbr") == team).item(0, "sos_played_adj")


def _remaining_adj(result: pl.DataFrame, team: str) -> float | None:
    """Return one team's remaining adjusted schedule strength from a result frame."""
    return result.filter(pl.col("team_abbr") == team).item(0, "sos_remaining_adj")


def _played_raw(result: pl.DataFrame, team: str) -> float | None:
    """Return one team's raw played schedule strength from a result frame."""
    return result.filter(pl.col("team_abbr") == team).item(0, "sos_played_raw")


# --------------------------------------------------------------------------------------
# Adjusted schedule strength
# --------------------------------------------------------------------------------------


def test_played_adjusted_matches_hand_computed_opponent_rating_mean() -> None:
    """Played strength averages the ratings of the unique opponents already faced."""
    result = schedule_strength.compute_schedule_strength_adjusted(
        _schedule_frame(), _ratings_frame(), season=2024, week=4
    )

    # BUF faced MIA (w1, w3) and NYJ (w2): mean(2.0, 3.0).
    assert _played_adj(result, "BUF") == pytest.approx((2.0 + 3.0) / 2)
    # MIA faced BUF (w1, w3) and KC (w2): mean(1.0, 5.0).
    assert _played_adj(result, "MIA") == pytest.approx((1.0 + 5.0) / 2)
    # NYJ faced NE, BUF, DEN: mean(4.0, 1.0, 6.0).
    assert _played_adj(result, "NYJ") == pytest.approx((4.0 + 1.0 + 6.0) / 3)
    # NE faced NYJ, DEN, KC: mean(3.0, 6.0, 5.0).
    assert _played_adj(result, "NE") == pytest.approx((3.0 + 6.0 + 5.0) / 3)
    # KC faced DEN, MIA, NE: mean(6.0, 2.0, 4.0).
    assert _played_adj(result, "KC") == pytest.approx((6.0 + 2.0 + 4.0) / 3)
    # DEN faced KC, NE, NYJ: mean(5.0, 4.0, 3.0).
    assert _played_adj(result, "DEN") == pytest.approx((5.0 + 4.0 + 3.0) / 3)


def test_remaining_adjusted_matches_hand_computed_opponent_rating_mean() -> None:
    """Remaining strength averages the ratings of the unique opponents still to face."""
    result = schedule_strength.compute_schedule_strength_adjusted(
        _schedule_frame(), _ratings_frame(), season=2024, week=4
    )

    # BUF still faces KC (w4) and NYJ (w5): mean(5.0, 3.0).
    assert _remaining_adj(result, "BUF") == pytest.approx((5.0 + 3.0) / 2)
    # MIA still faces NE twice: 4.0.
    assert _remaining_adj(result, "MIA") == pytest.approx(4.0)
    # NYJ still faces BUF only: 1.0.
    assert _remaining_adj(result, "NYJ") == pytest.approx(1.0)
    # NE still faces MIA twice: 2.0.
    assert _remaining_adj(result, "NE") == pytest.approx(2.0)
    # KC still faces BUF (w4) and DEN (w5): mean(1.0, 6.0).
    assert _remaining_adj(result, "KC") == pytest.approx((1.0 + 6.0) / 2)
    # DEN still faces KC only: 5.0.
    assert _remaining_adj(result, "DEN") == pytest.approx(5.0)


def test_week_one_has_null_played_strength_and_full_remaining_schedule() -> None:
    """Before any game is played the played side is null and everything remains."""
    result = schedule_strength.compute_schedule_strength_adjusted(
        _schedule_frame(), _ratings_frame(), season=2024, week=1
    )

    assert result["sos_played_adj"].null_count() == result.height
    # BUF's full slate is MIA, NYJ, KC: mean(2.0, 3.0, 5.0).
    assert _remaining_adj(result, "BUF") == pytest.approx((2.0 + 3.0 + 5.0) / 3)
    # NE's full slate is NYJ, DEN, KC, MIA: mean(3.0, 6.0, 5.0, 2.0).
    assert _remaining_adj(result, "NE") == pytest.approx((3.0 + 6.0 + 5.0 + 2.0) / 4)
    # KC's full slate is DEN, MIA, NE, BUF: mean(6.0, 2.0, 4.0, 1.0).
    assert _remaining_adj(result, "KC") == pytest.approx((6.0 + 2.0 + 4.0 + 1.0) / 4)


def test_bye_week_team_gets_correct_means_and_no_phantom_opponent() -> None:
    """A team idle in a week keeps correct means and gains no invented opponent."""
    result = schedule_strength.compute_schedule_strength_adjusted(
        _schedule_frame(), _ratings_frame(), season=2024, week=4
    )

    # NYJ and DEN are idle in week 4, so only their week 5 game remains.
    assert _remaining_adj(result, "NYJ") == pytest.approx(1.0)
    assert _remaining_adj(result, "DEN") == pytest.approx(5.0)
    # Their played sides are unaffected by the gap.
    assert _played_adj(result, "NYJ") == pytest.approx((4.0 + 1.0 + 6.0) / 3)
    assert _played_adj(result, "DEN") == pytest.approx((5.0 + 4.0 + 3.0) / 3)
    assert result.height == 6


def test_opponent_faced_twice_counts_once_on_both_sides() -> None:
    """A repeat opponent carries equal weight with a single-meeting opponent."""
    result = schedule_strength.compute_schedule_strength_adjusted(
        _schedule_frame(), _ratings_frame(), season=2024, week=4
    )

    # BUF met MIA in weeks 1 and 3; weighting per game would give (2 + 3 + 2) / 3.
    assert _played_adj(result, "BUF") == pytest.approx(2.5)
    assert _played_adj(result, "BUF") != pytest.approx((2.0 + 3.0 + 2.0) / 3)
    # NE meets MIA in weeks 4 and 5; a single unique opponent means the rating itself.
    assert _remaining_adj(result, "NE") == pytest.approx(2.0)


def test_unrated_opponents_are_excluded_and_all_unrated_yields_null() -> None:
    """Opponents without a rating drop out of the mean instead of biasing it."""
    result = schedule_strength.compute_schedule_strength_adjusted(
        _schedule_frame(), _ratings_frame({"NYJ": 3.0}), season=2024, week=4
    )

    # BUF faced MIA and NYJ; only NYJ is rated, so the mean is NYJ's rating.
    assert _played_adj(result, "BUF") == pytest.approx(3.0)
    # MIA faced BUF and KC, neither rated.
    assert _played_adj(result, "MIA") is None


def test_adjusted_ignores_other_seasons() -> None:
    """A decoy row from another season never reaches the requested season's means."""
    decoy = pl.DataFrame(
        {
            "season": pl.Series([2023], dtype=pl.Int64),
            "week": pl.Series([1], dtype=pl.Int64),
            "away_abbr": pl.Series(["BUF"], dtype=pl.Utf8),
            "home_abbr": pl.Series(["DEN"], dtype=pl.Utf8),
        }
    )
    schedule = pl.concat([_schedule_frame(), decoy], how="vertical")

    result = schedule_strength.compute_schedule_strength_adjusted(
        schedule, _ratings_frame(), season=2024, week=4
    )
    baseline = schedule_strength.compute_schedule_strength_adjusted(
        _schedule_frame(), _ratings_frame(), season=2024, week=4
    )

    assert_frame_equal(result, baseline)


def test_adjusted_played_side_excludes_current_and_postseason_weeks() -> None:
    """Games at or after the requested week, postseason included, stay off the played side."""
    postseason = pl.DataFrame(
        {
            "season": pl.Series([2024], dtype=pl.Int64),
            "week": pl.Series([20], dtype=pl.Int64),
            "away_abbr": pl.Series(["BUF"], dtype=pl.Utf8),
            "home_abbr": pl.Series(["DEN"], dtype=pl.Utf8),
        }
    )
    schedule = pl.concat([_schedule_frame(), postseason], how="vertical")

    result = schedule_strength.compute_schedule_strength_adjusted(
        schedule, _ratings_frame(), season=2024, week=4
    )

    # BUF's played mean stays mean(MIA, NYJ) even though weeks 4, 5 and 20 exist.
    assert _played_adj(result, "BUF") == pytest.approx((2.0 + 3.0) / 2)
    # The postseason meeting only shows up on the remaining side: KC, NYJ, DEN.
    assert _remaining_adj(result, "BUF") == pytest.approx((5.0 + 3.0 + 6.0) / 3)


# --------------------------------------------------------------------------------------
# Raw, head-to-head-excluded schedule strength
# --------------------------------------------------------------------------------------


def test_raw_played_matches_hand_computed_head_to_head_excluded_mean() -> None:
    """Each opponent profile drops its games against the subject before averaging."""
    result = schedule_strength.compute_schedule_strength_raw(
        _team_games_frame(), season=2024, week=4
    )

    # BUF faced MIA and NYJ.
    #   MIA without BUF: {+0.50}                -> 0.50
    #   NYJ without BUF: {+0.20, +0.90}         -> 0.55
    assert _played_raw(result, "BUF") == pytest.approx((0.50 + 0.55) / 2)
    # MIA faced BUF and KC.
    #   BUF without MIA: {+0.40}                -> 0.40
    #   KC without MIA:  {+0.30, -0.80}         -> -0.25
    assert _played_raw(result, "MIA") == pytest.approx((0.40 - 0.25) / 2)
    # NYJ faced NE, BUF and DEN.
    #   NE without NYJ:  {+0.60, +0.80}         -> 0.70
    #   BUF without NYJ: {+0.10, -0.70}         -> -0.30
    #   DEN without NYJ: {-0.30, -0.60}         -> -0.45
    assert _played_raw(result, "NYJ") == pytest.approx((0.70 - 0.30 - 0.45) / 3)
    # NE faced NYJ, DEN and KC.
    #   NYJ without NE:  {-0.40, +0.90}         -> 0.25
    #   DEN without NE:  {-0.30, -0.90}         -> -0.60
    #   KC without NE:   {+0.30, -0.50}         -> -0.10
    assert _played_raw(result, "NE") == pytest.approx((0.25 - 0.60 - 0.10) / 3)
    # KC faced DEN, MIA and NE.
    #   DEN without KC:  {-0.60, -0.90}         -> -0.75
    #   MIA without KC:  {-0.10, +0.70}         -> 0.30
    #   NE without KC:   {-0.20, +0.60}         -> 0.20
    assert _played_raw(result, "KC") == pytest.approx((-0.75 + 0.30 + 0.20) / 3)
    # DEN faced KC, NE and NYJ.
    #   KC without DEN:  {-0.50, -0.80}         -> -0.65
    #   NE without DEN:  {-0.20, +0.80}         -> 0.30
    #   NYJ without DEN: {+0.20, -0.40}         -> -0.10
    assert _played_raw(result, "DEN") == pytest.approx((-0.65 + 0.30 - 0.10) / 3)


def test_raw_played_ignores_head_to_head_games_but_tracks_third_party_games() -> None:
    """Head-to-head games cannot move the subject's value while third-party games can."""
    baseline = schedule_strength.compute_schedule_strength_raw(
        _team_games_frame(), season=2024, week=4
    )
    baseline_buf = _played_raw(baseline, "BUF")

    # Perturb BUF's own week 3 meeting with MIA on both mirror rows.
    head_to_head = [
        (week, away, home, -5.0 if (week, away, home) == (3, "MIA", "BUF") else margin)
        for week, away, home, margin in _TEAM_GAME_MARGINS
    ]
    unchanged = schedule_strength.compute_schedule_strength_raw(
        _team_games_frame(head_to_head), season=2024, week=4
    )
    assert _played_raw(unchanged, "BUF") == pytest.approx(baseline_buf)

    # Perturb MIA's week 2 game against KC, a third team.
    third_party = [
        (week, away, home, 2.0 if (week, away, home) == (2, "MIA", "KC") else margin)
        for week, away, home, margin in _TEAM_GAME_MARGINS
    ]
    changed = schedule_strength.compute_schedule_strength_raw(
        _team_games_frame(third_party), season=2024, week=4
    )
    # MIA without BUF is now {+2.00}; NYJ without BUF is unchanged at 0.55.
    assert _played_raw(changed, "BUF") == pytest.approx((2.00 + 0.55) / 2)
    assert _played_raw(changed, "BUF") != pytest.approx(baseline_buf)


def test_raw_drops_opponents_left_with_no_games_after_the_exclusion() -> None:
    """An opponent whose only prior game was the head-to-head contributes nothing."""
    margins = [
        (1, "BUF", "MIA", 0.10),
        (1, "NYJ", "NE", 0.20),
        (2, "BUF", "NYJ", 0.40),
    ]
    result = schedule_strength.compute_schedule_strength_raw(
        _team_games_frame(margins), season=2024, week=3
    )

    # BUF faced MIA and NYJ; MIA has played only BUF, so only NYJ survives: {+0.20}.
    assert _played_raw(result, "BUF") == pytest.approx(0.20)

    lone_game = schedule_strength.compute_schedule_strength_raw(
        _team_games_frame([(1, "BUF", "MIA", 0.10)]), season=2024, week=2
    )
    assert _played_raw(lone_game, "BUF") is None
    assert _played_raw(lone_game, "MIA") is None


def test_raw_is_null_before_any_games_are_played() -> None:
    """Week 1 leaves nothing to profile, so every team's value is null."""
    result = schedule_strength.compute_schedule_strength_raw(
        _team_games_frame(), season=2024, week=1
    )

    assert result.height == 6
    assert result["sos_played_raw"].null_count() == 6


def test_raw_ignores_other_seasons() -> None:
    """A decoy team-game from another season never reaches the requested season."""
    decoy = _team_games_frame([(1, "BUF", "DEN", 9.0)], season=2023)
    team_games = pl.concat([_team_games_frame(), decoy], how="vertical")

    result = schedule_strength.compute_schedule_strength_raw(team_games, season=2024, week=4)
    baseline = schedule_strength.compute_schedule_strength_raw(
        _team_games_frame(), season=2024, week=4
    )

    assert_frame_equal(result, baseline)


def test_raw_played_side_excludes_current_and_later_weeks() -> None:
    """Team-games at or after the requested week never leak into the played side."""
    future = _team_games_frame([(4, "BUF", "DEN", 9.0), (20, "BUF", "KC", -9.0)])
    team_games = pl.concat([_team_games_frame(), future], how="vertical")

    result = schedule_strength.compute_schedule_strength_raw(team_games, season=2024, week=4)
    baseline = schedule_strength.compute_schedule_strength_raw(
        _team_games_frame(), season=2024, week=4
    )

    assert_frame_equal(result, baseline)


def test_raw_uses_ratio_of_sums_when_numerator_and_denominator_are_supplied() -> None:
    """Summed numerator over summed denominator differs from the mean of per-game rates."""
    team_games = pl.DataFrame(
        {
            "season": pl.Series([2024] * 10, dtype=pl.Int64),
            "week": pl.Series([1, 1, 1, 1, 2, 2, 2, 2, 3, 3], dtype=pl.Int64),
            "team_abbr": pl.Series(
                ["BUF", "MIA", "NYJ", "NE", "MIA", "NYJ", "BUF", "NE", "MIA", "NE"],
                dtype=pl.Utf8,
            ),
            "opponent_abbr": pl.Series(
                ["MIA", "BUF", "NE", "NYJ", "NYJ", "MIA", "NE", "BUF", "NE", "MIA"],
                dtype=pl.Utf8,
            ),
            "epa_margin_sum": pl.Series(
                [10.0, -10.0, 4.0, -4.0, 30.0, -30.0, 6.0, -6.0, 10.0, -10.0],
                dtype=pl.Float64,
            ),
            "total_play_count": pl.Series(
                [100, 100, 50, 50, 50, 50, 60, 60, 150, 150], dtype=pl.Int64
            ),
        }
    ).with_columns(
        (pl.col("epa_margin_sum") / pl.col("total_play_count")).alias("epa_margin_per_play")
    )

    sums = schedule_strength.compute_schedule_strength_raw(
        team_games,
        season=2024,
        week=4,
        numerator_col="epa_margin_sum",
        denominator_col="total_play_count",
    )
    rates = schedule_strength.compute_schedule_strength_raw(team_games, season=2024, week=4)

    # BUF faced MIA (w1) and NE (w2).
    #   MIA without BUF: (30 + 10) / (50 + 150) = 0.20 by sums,
    #                    mean(30 / 50, 10 / 150) = 1 / 3 by rates.
    #   NE without BUF:  (-4 + -10) / (50 + 150) = -0.07 by sums,
    #                    mean(-4 / 50, -10 / 150) by rates.
    assert _played_raw(sums, "BUF") == pytest.approx((0.20 - 0.07) / 2)
    assert _played_raw(rates, "BUF") == pytest.approx(
        ((30.0 / 50.0 + 10.0 / 150.0) / 2 + (-4.0 / 50.0 + -10.0 / 150.0) / 2) / 2
    )
    assert _played_raw(sums, "BUF") != pytest.approx(_played_raw(rates, "BUF"))


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------


def test_empty_inputs_return_typed_empty_frames() -> None:
    """Empty inputs produce empty frames carrying the full documented schema."""
    empty_schedule = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "away_abbr": pl.Utf8,
            "home_abbr": pl.Utf8,
        }
    )
    empty_ratings = pl.DataFrame(schema={"team_abbr": pl.Utf8, "rating": pl.Float64})
    adjusted = schedule_strength.compute_schedule_strength_adjusted(
        empty_schedule, empty_ratings, season=2024, week=4
    )
    assert adjusted.height == 0
    assert adjusted.schema == pl.Schema(
        {
            "team_abbr": pl.Utf8,
            "sos_played_adj": pl.Float64,
            "sos_remaining_adj": pl.Float64,
        }
    )

    empty_team_games = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "team_abbr": pl.Utf8,
            "opponent_abbr": pl.Utf8,
            "epa_margin_per_play": pl.Float64,
        }
    )
    raw = schedule_strength.compute_schedule_strength_raw(empty_team_games, season=2024, week=4)
    assert raw.height == 0
    assert raw.schema == pl.Schema({"team_abbr": pl.Utf8, "sos_played_raw": pl.Float64})


def test_populated_results_are_sorted_and_float64() -> None:
    """Both helpers return team-sorted frames with Float64 value columns."""
    adjusted = schedule_strength.compute_schedule_strength_adjusted(
        _schedule_frame(), _ratings_frame(), season=2024, week=4
    )
    assert adjusted["team_abbr"].to_list() == sorted(_RATING_VALUES)
    assert adjusted.schema["sos_played_adj"] == pl.Float64
    assert adjusted.schema["sos_remaining_adj"] == pl.Float64

    raw = schedule_strength.compute_schedule_strength_raw(_team_games_frame(), season=2024, week=4)
    assert raw["team_abbr"].to_list() == sorted(_RATING_VALUES)
    assert raw.schema["sos_played_raw"] == pl.Float64


def test_invalid_inputs_raise_value_errors() -> None:
    """Missing columns, duplicate ratings, and half-supplied sums are rejected."""
    with pytest.raises(ValueError, match="missing required columns"):
        schedule_strength.compute_schedule_strength_adjusted(
            _schedule_frame().drop("home_abbr"), _ratings_frame(), season=2024, week=4
        )

    duplicated = pl.concat([_ratings_frame(), _ratings_frame().head(1)], how="vertical")
    with pytest.raises(ValueError, match="one rating per team"):
        schedule_strength.compute_schedule_strength_adjusted(
            _schedule_frame(), duplicated, season=2024, week=4
        )

    with pytest.raises(ValueError, match="numerator_col and denominator_col"):
        schedule_strength.compute_schedule_strength_raw(
            _team_games_frame(), season=2024, week=4, numerator_col="epa_margin_sum"
        )
