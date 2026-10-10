"""Tests for the possession-side special-teams play counts and EPA per team-game."""

import polars as pl
import pytest

from nfl_predictor.season_stats.team_stats_expanded import compute_expanded_team_game_stats


def _pbp() -> pl.DataFrame:
    """Return one game of plays: two AAA punts, one BBB kickoff return, and two scrimmage snaps."""
    return pl.DataFrame(
        {
            "game_id": ["g1"] * 5,
            "week": [1] * 5,
            "posteam": ["AAA", "AAA", "BBB", "AAA", "BBB"],
            "defteam": ["BBB", "BBB", "AAA", "BBB", "AAA"],
            "special": [1, 1, 1, 0, 0],
            "rush": [0, 0, 0, 1, 1],
            "epa": [0.5, -1.25, 2.0, 0.3, -0.4],
        }
    )


def _row(frame: pl.DataFrame, team: str) -> dict[str, object]:
    """Return one team's team-game row as a dict."""
    return frame.filter(pl.col("team") == team).row(0, named=True)


def test_compute_expanded_team_game_stats_counts_possession_special_teams_plays() -> None:
    # Arrange
    pbp = _pbp()

    # Act
    stats = compute_expanded_team_game_stats(pbp)

    # Assert
    assert (_row(stats, "AAA")["st_plays"], _row(stats, "BBB")["st_plays"]) == (2, 1)


def test_compute_expanded_team_game_stats_sums_possession_special_teams_epa() -> None:
    # Arrange
    pbp = _pbp()

    # Act
    stats = compute_expanded_team_game_stats(pbp)

    # Assert
    assert (_row(stats, "AAA")["st_epa"], _row(stats, "BBB")["st_epa"]) == pytest.approx(
        (-0.75, 2.0)
    )


def test_compute_expanded_team_game_stats_keeps_special_plays_out_of_offensive_epa() -> None:
    # Arrange
    pbp = _pbp()

    # Act
    stats = compute_expanded_team_game_stats(pbp)

    # Assert
    assert _row(stats, "AAA")["offensive_epa"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "pbp",
    [
        pl.DataFrame(),
        pl.DataFrame({"game_id": ["g1"], "epa": [0.1]}),
        pl.DataFrame(
            {"game_id": ["g1"], "week": [1], "posteam": [None], "defteam": [None], "epa": [0.1]},
            schema_overrides={"posteam": pl.String, "defteam": pl.String},
        ),
    ],
)
def test_compute_expanded_team_game_stats_without_possession_plays_is_empty(
    pbp: pl.DataFrame,
) -> None:
    # Act
    stats = compute_expanded_team_game_stats(pbp)

    # Assert
    assert stats.is_empty()
    assert stats.columns == ["team", "opponent_team"]
