"""Tests for the own Elo's quarterback adjustment, FiveThirtyEight's VALUE model."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_predictor.utils.polars import elo_quarterbacks as quarterbacks

PLAY_SCHEMA = {
    "game_id": pl.String,
    "season": pl.Int64,
    "season_type": pl.String,
    "week": pl.Int64,
    "posteam": pl.String,
    "defteam": pl.String,
    "qb_dropback": pl.Float64,
    "qb_kneel": pl.Float64,
    "qb_spike": pl.Float64,
    "rush": pl.Float64,
    "rush_attempt": pl.Float64,
    "pass_attempt": pl.Float64,
    "two_point_attempt": pl.Float64,
    "sack": pl.Float64,
    "complete_pass": pl.Float64,
    "pass_touchdown": pl.Float64,
    "interception": pl.Float64,
    "rush_touchdown": pl.Float64,
    "yards_gained": pl.Float64,
    "passer_player_id": pl.String,
    "passer_player_name": pl.String,
}
GAME_SCHEMA = {
    "game_id": pl.String,
    "season": pl.Int64,
    "week": pl.Int64,
    "game_type": pl.String,
    "home_abbr": pl.String,
    "away_abbr": pl.String,
    "home_score": pl.Int64,
    "away_score": pl.Int64,
    "neutral": pl.Int64,
}
BOX_SCHEMA = {
    "season": pl.Int64,
    "week": pl.Int64,
    "team_abbr": pl.String,
    "qb_id": pl.String,
    "starter": pl.Boolean,
    "value": pl.Float64,
}


def _play(
    team: str, opponent: str, passer: str | None = None, **flags: object
) -> dict[str, object]:
    """Return one regular-season week-1 play of 2020 with every flag 0 unless given."""
    row: dict[str, object] = dict.fromkeys(PLAY_SCHEMA, 0.0)
    row.update(
        {
            "game_id": "2020_01_X",
            "season": 2020,
            "season_type": "REG",
            "week": 1,
            "posteam": team,
            "defteam": opponent,
            "passer_player_id": passer,
            "passer_player_name": passer,
        }
    )
    row.update(flags)
    return row


def _pass(team: str, passer: str, yards: float = 0.0, **flags: object) -> dict[str, object]:
    return _play(
        team, "OPP", passer, qb_dropback=1.0, pass_attempt=1.0, yards_gained=yards, **flags
    )


def _plays(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=PLAY_SCHEMA)


def _games(*rows: tuple[str, int, int, str, str]) -> pl.DataFrame:
    """Return completed regular-season games from (id, season, week, home, away) tuples."""
    return pl.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "game_type": "REG" if week <= 18 else "DIV",
                "home_abbr": home,
                "away_abbr": away,
                "home_score": 20,
                "away_score": 17,
                "neutral": 0,
            }
            for game_id, season, week, home, away in rows
        ],
        schema=GAME_SCHEMA,
    )


def _box(*rows: tuple[int, int, str, str, float]) -> pl.DataFrame:
    """Return starters' box-score values from (season, week, team, qb, value) tuples."""
    return pl.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "team_abbr": team,
                "qb_id": qb,
                "starter": True,
                "value": value,
            }
            for season, week, team, qb, value in rows
        ],
        schema=BOX_SCHEMA,
    )


def _row(frame: pl.DataFrame, game_id: str) -> dict[str, object]:
    return frame.filter(pl.col("game_id") == game_id).row(0, named=True)


def test_quarterback_value_is_fivethirtyeights_box_score_formula() -> None:
    # Arrange
    line = pl.DataFrame(
        {
            "attempts": [30.0],
            "completions": [20.0],
            "pass_yards": [250.0],
            "pass_tds": [2.0],
            "interceptions": [1.0],
            "sacks": [3.0],
            "rush_attempts": [4.0],
            "rush_yards": [20.0],
            "rush_tds": [1.0],
        }
    )
    expected = (
        -2.2 * 30 + 3.7 * 20 + 250 / 5 + 11.3 * 2 - 14.1 * 1 - 8 * 3 - 1.1 * 4 + 0.6 * 20 + 15.9
    )

    # Act
    value = line.select(quarterbacks.quarterback_value()).item()

    # Assert
    assert value == pytest.approx(expected)


def test_box_scores_count_passing_sacks_scrambles_and_kneels() -> None:
    # Arrange
    plays = _plays(
        _pass("AAA", "A", 10.0, complete_pass=1.0),
        _pass("AAA", "A", 20.0, complete_pass=1.0, pass_touchdown=1.0),
        _pass("AAA", "A"),
        _pass("AAA", "A", interception=1.0),
        _play("AAA", "OPP", "A", qb_dropback=1.0, sack=1.0, yards_gained=-7.0),
        _play("AAA", "OPP", None, qb_dropback=1.0, rush_attempt=1.0, yards_gained=8.0),
        _play("AAA", "OPP", None, qb_kneel=1.0, rush_attempt=1.0, yards_gained=-1.0),
        _play("AAA", "OPP", None, rush=1.0, rush_attempt=1.0, yards_gained=40.0),
    )

    # Act
    box = quarterbacks.quarterback_box_scores(plays)

    # Assert
    row = box.row(0, named=True)
    assert box.height == 1
    assert (row["attempts"], row["completions"], row["pass_yards"]) == (4.0, 2.0, 30.0)
    assert (row["pass_tds"], row["interceptions"], row["sacks"]) == (1.0, 1.0, 1.0)
    assert (row["rush_attempts"], row["rush_yards"], row["rush_tds"]) == (2.0, 7.0, 0.0)
    assert row["value"] == pytest.approx(-4.2)


def test_box_scores_mark_the_first_passer_as_the_starter() -> None:
    # Arrange
    plays = _plays(
        _pass("BBB", "B2", 5.0, complete_pass=1.0),
        *[_pass("BBB", "B1", 8.0, complete_pass=1.0) for _ in range(5)],
    )

    # Act
    box = quarterbacks.quarterback_box_scores(plays)

    # Assert
    starters = box.filter(pl.col("starter"))["qb_id"].to_list()
    assert starters == ["B2"]
    assert sorted(box["qb_id"].to_list()) == ["B1", "B2"]


def test_box_scores_credit_scrambles_to_the_passer_with_the_most_dropbacks() -> None:
    # Arrange
    plays = _plays(
        _pass("BBB", "B2"),
        _pass("BBB", "B1"),
        _pass("BBB", "B1"),
        _play("BBB", "OPP", None, qb_dropback=1.0, rush_attempt=1.0, yards_gained=12.0),
    )

    # Act
    box = quarterbacks.quarterback_box_scores(plays)

    # Assert
    rushing = dict(zip(box["qb_id"].to_list(), box["rush_yards"].to_list(), strict=True))
    assert rushing == {"B1": 12.0, "B2": 0.0}


def test_box_scores_leave_out_two_point_tries_and_the_postseason() -> None:
    # Arrange
    plays = _plays(
        _pass("AAA", "A", 10.0, complete_pass=1.0),
        _pass("AAA", "A", 2.0, complete_pass=1.0, two_point_attempt=1.0),
        _pass("AAA", "A", 30.0, complete_pass=1.0, season_type="POST"),
    )

    # Act
    box = quarterbacks.quarterback_box_scores(plays)

    # Assert
    assert box["pass_yards"].to_list() == [10.0]


def test_first_game_of_a_team_has_no_quarterback_adjustment() -> None:
    # Arrange
    games = _games(("g1", 2000, 1, "AAA", "BBB"))
    box = _box((2000, 1, "AAA", "A", 50.0), (2000, 1, "BBB", "B", 10.0))

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, box)

    # Assert
    row = _row(adjustments, "g1")
    assert (row["home_qb_adj"], row["away_qb_adj"]) == (0.0, 0.0)
    assert (row["home_qb_id"], row["away_qb_id"]) == ("A", "B")


def test_ratings_roll_ten_games_for_a_quarterback_and_twenty_for_a_team() -> None:
    # Arrange
    games = _games(("g1", 2000, 1, "AAA", "BBB"), ("g2", 2000, 2, "AAA", "BBB"))
    box = _box(
        (2000, 1, "AAA", "A", 50.0),
        (2000, 1, "BBB", "B", 50.0),
        (2000, 2, "AAA", "A", 50.0),
        (2000, 2, "BBB", "B", 50.0),
    )

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, box)

    # Assert
    row = _row(adjustments, "g2")
    assert row["home_qb_value"] == pytest.approx(0.1 * 50.0)
    assert row["home_team_qb_value"] == pytest.approx(0.05 * 50.0)
    assert row["home_qb_adj"] == pytest.approx(3.3 * (5.0 - 2.5))


def test_a_debut_starter_gets_the_debut_rating_against_the_teams_rating() -> None:
    # Arrange
    games = _games(("g1", 2000, 1, "AAA", "BBB"), ("g2", 2000, 2, "AAA", "BBB"))
    box = _box(
        (2000, 1, "AAA", "A", 50.0),
        (2000, 1, "BBB", "B", 50.0),
        (2000, 2, "AAA", "BACKUP", 50.0),
        (2000, 2, "BBB", "B", 50.0),
    )

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, box)

    # Assert
    row = _row(adjustments, "g2")
    assert row["home_qb_id"] == "BACKUP"
    assert row["home_qb_value"] == 0.0
    assert row["home_qb_adj"] == pytest.approx(3.3 * (0.0 - 2.5))


def test_value_is_adjusted_for_the_opposing_defense_before_it_rolls_in() -> None:
    # Arrange
    games = _games(
        ("g1", 2000, 1, "CCC", "DDD"),
        ("g2", 2000, 2, "CCC", "AAA"),
        ("g3", 2000, 3, "AAA", "EEE"),
    )
    box = _box(
        (2000, 1, "CCC", "C", 0.0),
        (2000, 1, "DDD", "D", 80.0),
        (2000, 2, "CCC", "C", 50.0),
        (2000, 2, "AAA", "A", 50.0),
        (2000, 3, "AAA", "A", 50.0),
        (2000, 3, "EEE", "E", 50.0),
    )
    # CCC allowed 80 in a week whose team-games averaged 40: its defense rating is 0.05 * 40.
    adjusted = 50.0 - 0.05 * (80.0 - 40.0)

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, box)

    # Assert
    row = _row(adjustments, "g3")
    assert row["home_qb_value"] == pytest.approx(0.1 * adjusted)
    assert row["home_team_qb_value"] == pytest.approx(0.05 * adjusted)


def _ten_week_season(season: int, value: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return AAA (quarterback A) against BBB (quarterback B) for ten weeks, equal values."""
    games = _games(*[(f"{season}_{week}", season, week, "AAA", "BBB") for week in range(1, 11)])
    box = _box(
        *[
            (season, week, team, qb, value)
            for week in range(1, 11)
            for team, qb in (("AAA", "A"), ("BBB", "B"))
        ]
    )
    return games, box


def test_a_veteran_with_ten_starts_reverts_a_quarter_toward_the_average_start() -> None:
    # Arrange
    first_games, first_box = _ten_week_season(2000, 40.0)
    games = pl.concat([first_games, _games(("next", 2001, 1, "AAA", "BBB"))])
    end_of_season = 40.0 * (1.0 - 0.9**10)

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, first_box)

    # Assert
    row = _row(adjustments, "next")
    assert row["home_qb_value"] == pytest.approx(end_of_season + 0.25 * (40.0 - end_of_season))


def test_a_quarterback_with_fewer_than_ten_starts_keeps_his_rating() -> None:
    # Arrange
    first_games, first_box = _ten_week_season(2000, 40.0)
    games = pl.concat(
        [first_games.filter(pl.col("week") <= 9), _games(("next", 2001, 1, "AAA", "BBB"))]
    )
    box = first_box.filter(pl.col("week") <= 9)

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, box)

    # Assert
    row = _row(adjustments, "next")
    assert row["home_qb_value"] == pytest.approx(40.0 * (1.0 - 0.9**9))


def test_a_teams_preseason_rating_is_a_third_last_season_and_two_thirds_its_starter() -> None:
    # Arrange
    first_games, first_box = _ten_week_season(2000, 40.0)
    games = pl.concat([first_games, _games(("next", 2001, 1, "AAA", "BBB"))])
    quarterback_end = 40.0 * (1.0 - 0.9**10)
    quarterback_preseason = quarterback_end + 0.25 * (40.0 - quarterback_end)
    team_end = 40.0 * (1.0 - 0.95**10)

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, first_box)

    # Assert
    row = _row(adjustments, "next")
    expected = team_end / 3.0 + quarterback_preseason * 2.0 / 3.0
    assert row["home_team_qb_value"] == pytest.approx(expected)


def test_a_debut_after_the_first_season_starts_at_the_mean_of_earlier_debut_starts() -> None:
    # Arrange
    games = _games(
        ("g2000", 2000, 1, "AAA", "BBB"),
        ("g2001", 2001, 1, "AAA", "BBB"),
        ("g2002", 2002, 1, "AAA", "BBB"),
    )
    box = _box(
        (2000, 1, "AAA", "A", 30.0),
        (2000, 1, "BBB", "B", 30.0),
        (2001, 1, "AAA", "ROOKIE1", 30.0),
        (2001, 1, "BBB", "ROOKIE2", 30.0),
        (2002, 1, "AAA", "ROOKIE3", 30.0),
        (2002, 1, "BBB", "B", 30.0),
    )

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, box)

    # Assert
    assert _row(adjustments, "g2001")["home_qb_value"] == 0.0
    assert _row(adjustments, "g2002")["home_qb_value"] == pytest.approx(30.0)


def test_an_unplayed_game_uses_the_last_starter_unless_one_is_projected() -> None:
    # Arrange
    games = _games(
        ("g1", 2000, 1, "AAA", "BBB"),
        ("g2", 2000, 2, "AAA", "BBB"),
        ("g3", 2000, 3, "AAA", "BBB"),
    ).with_columns(
        pl.when(pl.col("week") == 3).then(None).otherwise(pl.col("home_score")).alias("home_score"),
        pl.when(pl.col("week") == 3).then(None).otherwise(pl.col("away_score")).alias("away_score"),
    )
    box = _box(
        (2000, 1, "AAA", "A", 50.0),
        (2000, 1, "BBB", "B", 50.0),
        (2000, 2, "AAA", "BACKUP", 50.0),
        (2000, 2, "BBB", "B", 50.0),
    )
    projected = pl.DataFrame({"game_id": ["g3"], "team_abbr": ["AAA"], "qb_id": ["A"]})

    # Act
    last_starter = quarterbacks.value_qb_adjustments(games, box)
    projection = quarterbacks.value_qb_adjustments(games, box, projected_starters=projected)

    # Assert
    assert _row(last_starter, "g3")["home_qb_id"] == "BACKUP"
    assert _row(projection, "g3")["home_qb_id"] == "A"


def test_a_playoff_game_uses_the_last_starter_and_updates_nothing() -> None:
    # Arrange
    games = _games(
        ("g1", 2000, 1, "AAA", "BBB"),
        ("divisional", 2000, 19, "AAA", "BBB"),
        ("conference", 2000, 20, "AAA", "BBB"),
    )
    box = _box((2000, 1, "AAA", "A", 50.0), (2000, 1, "BBB", "B", 50.0))

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, box)

    # Assert
    divisional = _row(adjustments, "divisional")
    conference = _row(adjustments, "conference")
    assert divisional["home_qb_id"] == "A"
    assert divisional["home_qb_value"] == conference["home_qb_value"] == pytest.approx(5.0)


def test_box_scores_of_plays_without_a_passer_are_an_empty_typed_frame() -> None:
    # Arrange
    plays = _plays(_play("AAA", "OPP", None, rush=1.0, rush_attempt=1.0, yards_gained=4.0))

    # Act
    box = quarterbacks.quarterback_box_scores(plays)

    # Assert
    assert box.height == 0
    assert box.columns == list(quarterbacks.BOX_SCORE_SCHEMA)


def test_a_team_without_any_known_starter_gets_no_adjustment() -> None:
    # Arrange
    games = _games(("g2000", 2000, 1, "AAA", "BBB"), ("g2001", 2001, 1, "AAA", "BBB"))
    box = _box((2000, 1, "BBB", "B", 50.0), (2001, 1, "BBB", "B", 50.0))

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, box)

    # Assert
    for game_id in ("g2000", "g2001"):
        row = _row(adjustments, game_id)
        assert row["home_qb_id"] is None
        assert row["home_qb_value"] is None
        assert row["home_qb_adj"] == 0.0


def test_value_qb_adjustments_of_an_empty_schedule_is_an_empty_typed_frame() -> None:
    # Arrange
    games = _games()

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, _box())

    # Assert
    assert adjustments.height == 0
    assert adjustments.columns == list(quarterbacks.ADJUSTMENT_SCHEMA)


def test_a_weeks_box_scores_never_reach_that_weeks_pre_game_values() -> None:
    # Arrange
    games = _games(
        ("w1", 2000, 1, "AAA", "BBB"),
        ("w2a", 2000, 2, "AAA", "CCC"),
        ("w2b", 2000, 2, "BBB", "DDD"),
        ("w3", 2000, 3, "AAA", "BBB"),
    )
    earlier = [
        (2000, 1, "AAA", "A", 40.0),
        (2000, 1, "BBB", "B", 20.0),
        (2000, 2, "AAA", "A", 40.0),
        (2000, 2, "CCC", "C", 30.0),
    ]
    box = _box(*earlier, (2000, 2, "BBB", "B", 20.0), (2000, 2, "DDD", "D", 25.0))
    changed = _box(*earlier, (2000, 2, "BBB", "B", 90.0), (2000, 2, "DDD", "D", -60.0))

    # Act
    adjustments = quarterbacks.value_qb_adjustments(games, box)

    # Assert
    with_change = quarterbacks.value_qb_adjustments(games, changed)
    pre_game = ["home_qb_value", "away_qb_value", "home_team_qb_value", "away_team_qb_value"]
    for game_id in ("w1", "w2a", "w2b"):
        before = _row(adjustments, game_id)
        after = _row(with_change, game_id)
        assert [before[c] for c in pre_game] == [after[c] for c in pre_game]
    assert _row(adjustments, "w3")["away_qb_value"] != _row(with_change, "w3")["away_qb_value"]
