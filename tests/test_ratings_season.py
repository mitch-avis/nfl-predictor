"""Tests for one season's rating tables, built as ``nfl-sos-ratings`` writes them."""

import itertools

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor.ratings.qb_rating import fit_qb_ratings
from nfl_predictor.ratings.season import (
    QB_RATINGS_ORDER,
    TEAM_RATINGS_ORDER,
    BootstrapSettings,
    build_qb_rank_ranges_by_week,
    build_qb_season,
    build_team_season,
)
from nfl_predictor.ratings.team_prior import (
    PriorHistory,
    prior_without_team,
    snapshot_prior,
    team_prior_table,
)
from nfl_predictor.ratings.team_rating import (
    TEAM_RATING_COLUMNS,
    compute_team_schedule_strength,
    fit_team_ratings,
    fit_team_ratings_with_previous_penalties,
)

_TEAMS = ("BUF", "MIA", "NE", "NYJ")
_STRENGTH = {"BUF": 0.08, "MIA": -0.02, "NE": 0.04, "NYJ": -0.10}
_BOOTSTRAP = BootstrapSettings(resamples=30)


def _games() -> list[tuple[int, str, str, str]]:
    """Return (week, game_id, home, away) for a double round-robin, one game a week."""
    return [
        (week, f"2025_{week:02d}_{away}_{home}", home, away)
        for week, (home, away) in enumerate(itertools.permutations(_TEAMS, 2), start=1)
    ]


def _snaps(week: int, team: str) -> int:
    """Return a team's offensive snaps in one week's game: 56 to 68, different game to game."""
    return 56 + (week * (1 + _TEAMS.index(team))) % 13


def _team_games() -> pl.DataFrame:
    """Return a season of team-game rows built from known strengths."""
    rows: list[dict[str, object]] = []
    for week, game_id, home, away in _games():
        margin = round((_STRENGTH[home] - _STRENGTH[away]) * 100) + 2
        for team, opponent, is_home, team_margin in (
            (home, away, True, margin),
            (away, home, False, -margin),
        ):
            epa_per_play = 0.01 + _STRENGTH[team] - _STRENGTH[opponent] / 2 + (week % 3) * 0.01
            rows.append(
                {
                    "game_id": game_id,
                    "week": week,
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": is_home,
                    "offensive_snaps": _snaps(week, team),
                    "offensive_epa": epa_per_play * _snaps(week, team),
                    "st_plays": 13,
                    "st_epa": 0.05 if is_home else -0.05,
                    "point_margin": team_margin,
                }
            )
    return pl.DataFrame(rows).sort("team", "week", "game_id")


def _qb_games(*, backup: bool = False) -> pl.DataFrame:
    """Return one starting passer per team-game, and optionally a backup with four dropbacks."""
    rows: list[dict[str, object]] = []
    for row in _team_games().iter_rows(named=True):
        epa = float(row["offensive_epa"]) / float(row["offensive_snaps"]) + 0.02
        rows.append(
            {
                "game_id": row["game_id"],
                "week": row["week"],
                "team": row["team"],
                "opponent_team": row["opponent_team"],
                "is_home": row["is_home"],
                "qb_id": f"qb-{row['team']}",
                "qb_name": f"{row['team']} Passer",
                "qb_dropbacks": 38,
                "qb_dropback_epa": epa * 38,
                "qb_attempts": 35,
            }
        )
    starters = pl.DataFrame(rows)
    if not backup:
        return starters
    relief = starters.filter(pl.col("team") == "BUF").head(1)
    return pl.concat(
        [
            starters,
            relief.with_columns(
                pl.lit("qb-BUF-2").alias("qb_id"),
                pl.lit("BUF Backup").alias("qb_name"),
                pl.lit(4, dtype=pl.Int64).alias("qb_dropbacks"),
                pl.lit(0.4).alias("qb_dropback_epa"),
                pl.lit(4, dtype=pl.Int64).alias("qb_attempts"),
            ),
        ]
    )


def _history() -> PriorHistory:
    """Return a prior history in which every earlier season is the same league."""
    return PriorHistory(lambda _season: _team_games())


def _sorted(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``frame`` sorted by its identity columns, for comparisons free of row order."""
    keys = [key for key in ("week", "team", "other_team", "excluded_team") if key in frame.columns]
    return frame.sort(keys, nulls_last=True)


def _team_season(season: int = 2025, *, weekly_ranges: bool = False):
    """Return the synthetic league's team tables for ``season``."""
    return build_team_season(
        season, _team_games(), _history(), weekly_ranges=weekly_ranges, bootstrap=_BOOTSTRAP
    )


def test_build_team_season_ratings_publish_columns_in_order() -> None:
    # Act
    ratings = _team_season().ratings

    # Assert
    assert ratings.columns == list(TEAM_RATINGS_ORDER)
    assert ratings.columns == ["team", "games_played", *TEAM_RATING_COLUMNS, "sos", "SRS"]


def test_build_team_season_ratings_rank_the_strongest_team_first() -> None:
    # Act
    ratings = _team_season().ratings

    # Assert
    assert ratings.get_column("team").to_list()[0] == "BUF"
    assert ratings.get_column("team_rating").is_sorted(descending=True)


def test_build_team_season_fits_with_the_prior_and_the_previous_penalties() -> None:
    """Six games each, under the nine that fade the prior out, so the prior counts."""
    # Arrange
    history = _history()
    means = snapshot_prior(history.season_prior(2025), _team_games())
    expected = fit_team_ratings_with_previous_penalties(
        _team_games(), history.penalties(2025), scrimmage_prior=means
    ).ratings

    # Act
    ratings = _team_season().ratings

    # Assert
    assert means is not None
    assert_frame_equal(_sorted(ratings.select(expected.columns)), _sorted(expected))


def test_build_team_season_first_season_cross_validates_its_own_penalties() -> None:
    # Act
    ratings = _team_season(1999).ratings

    # Assert
    expected = fit_team_ratings(_team_games()).ratings
    assert_frame_equal(_sorted(ratings.select(expected.columns)), _sorted(expected))


def test_build_team_season_refits_schedule_strength_with_the_left_out_teams_priors() -> None:
    # Arrange
    history = _history()
    means = snapshot_prior(history.season_prior(2025), _team_games())
    fit = fit_team_ratings_with_previous_penalties(
        _team_games(), history.penalties(2025), scrimmage_prior=means
    )
    expected = compute_team_schedule_strength(
        _team_games(), fit, prior_without=prior_without_team(history, 2025, _team_games())
    )
    without_prior = compute_team_schedule_strength(_team_games(), fit)

    # Act
    sos = _team_season().ratings.select("team", "sos")

    # Assert
    assert_frame_equal(_sorted(sos), _sorted(expected))
    assert not _sorted(sos).equals(_sorted(without_prior))


def test_build_team_season_writes_the_prior_means_its_fits_used() -> None:
    # Arrange
    history = _history()
    means = snapshot_prior(history.season_prior(2025), _team_games())
    expected = team_prior_table(
        means, prior_without_team(history, 2025, _team_games()), sorted(_TEAMS)
    )

    # Act
    table = _team_season().team_prior

    # Assert
    assert_frame_equal(_sorted(table), _sorted(expected))
    assert table.get_column("excluded_team").is_null().to_list()[-len(_TEAMS) :] == [True] * 4


def test_build_team_season_without_a_prior_has_no_prior_means() -> None:
    """The carryover slopes need three earlier season pairs, so 2002 has no prior."""
    # Act
    table = _team_season(2002).team_prior

    # Assert
    assert table.is_empty()


def test_build_team_season_history_ends_at_the_season_rating() -> None:
    # Arrange
    season = _team_season()

    # Act
    history = season.ratings_by_week

    # Assert
    last_week = history.filter(pl.col("week") == len(_games())).sort("team")
    assert last_week.get_column("team_rating").to_list() == pytest.approx(
        season.ratings.sort("team").get_column("team_rating").to_list()
    )
    assert history.select("team", "week").rows() == sorted(history.select("team", "week").rows())


def test_build_team_season_rank_ranges_follow_the_published_order() -> None:
    # Arrange
    season = _team_season()

    # Act
    ranges = season.rating_ranges

    # Assert
    assert ranges.get_column("team").to_list() == season.ratings.get_column("team").to_list()
    assert ranges.get_column("team_rank").to_list() == [1, 2, 3, 4]
    assert {"offense_rank", "defense_rank", "special_teams_rank"} <= set(ranges.columns)


def test_build_team_season_pairs_cover_every_ordered_pair_in_order() -> None:
    # Act
    pairs = _team_season().rating_pairs

    # Assert
    expected = [(team, other) for team in _TEAMS for other in _TEAMS if team != other]
    assert pairs.select("team", "other_team").rows() == expected


def test_build_team_season_weekly_rank_ranges_only_when_asked() -> None:
    # Act
    weekly = _team_season().rating_ranges_by_week

    # Assert
    assert weekly is None


def test_build_team_season_weekly_rank_ranges_end_at_the_season_ranks() -> None:
    # Arrange
    season = _team_season(weekly_ranges=True)

    # Act
    weekly = season.rating_ranges_by_week

    # Assert
    assert weekly is not None
    assert weekly.get_column("week").unique().sort().to_list() == list(range(1, 13))
    last = weekly.filter(pl.col("week") == 12).join(
        season.rating_ranges.select("team", pl.col("team_rank").alias("season_rank")), on="team"
    )
    assert last.get_column("team_rank").to_list() == last.get_column("season_rank").to_list()


def test_build_qb_season_rates_only_qualified_passers_in_the_ratings() -> None:
    # Act
    qb = build_qb_season(_qb_games(backup=True), _team_games(), bootstrap=_BOOTSTRAP)

    # Assert
    assert qb.qb_ratings.columns == list(QB_RATINGS_ORDER)
    assert sorted(qb.qb_ratings.get_column("qb_id").to_list()) == [f"qb-{t}" for t in _TEAMS]
    assert qb.qb_ratings.get_column("adj_qb_epa_per_dropback").is_sorted(descending=True)


def test_build_qb_season_keeps_every_passer_in_the_season_table() -> None:
    # Act
    qb = build_qb_season(_qb_games(backup=True), _team_games(), bootstrap=_BOOTSTRAP)

    # Assert
    backup = qb.qb_season.filter(pl.col("qb_id") == "qb-BUF-2").row(0, named=True)
    assert backup["qb_is_eligible"] is False
    assert backup["adj_qb_epa_per_dropback"] is not None


def test_build_qb_season_ranks_and_pairs_only_qualified_passers() -> None:
    # Act
    qb = build_qb_season(_qb_games(backup=True), _team_games(), bootstrap=_BOOTSTRAP)

    # Assert
    passers = {f"qb-{team}" for team in _TEAMS}
    assert set(qb.qb_rating_ranges.get_column("qb_id").to_list()) == passers
    assert set(qb.qb_rating_pairs.get_column("qb_id").to_list()) == passers
    assert qb.qb_rating_pairs.height == len(passers) * (len(passers) - 1)


def test_build_qb_season_rank_ranges_carry_the_passer_identity() -> None:
    # Act
    qb = build_qb_season(_qb_games(), _team_games(), bootstrap=_BOOTSTRAP)

    # Assert
    first = qb.qb_rating_ranges.row(0, named=True)
    assert (first["qb_rank"], first["qb_name"], first["team"]) == (1, "BUF Passer", "BUF")
    assert qb.qb_rating_ranges.columns[:4] == ["qb_id", "qb_name", "team", "qb_rank"]


def test_build_qb_season_history_ends_with_every_passer() -> None:
    # Act
    qb = build_qb_season(_qb_games(), _team_games(), bootstrap=_BOOTSTRAP)

    # Assert
    last_week = qb.qb_ratings_by_week.filter(pl.col("week") == len(_games()))
    assert sorted(last_week.get_column("qb_id").to_list()) == [f"qb-{team}" for team in _TEAMS]


def test_build_qb_season_weekly_rank_ranges_end_at_the_season_ranks() -> None:
    # Arrange
    qb = build_qb_season(_qb_games(), _team_games(), weekly_ranges=True, bootstrap=_BOOTSTRAP)

    # Act
    weekly = qb.qb_rating_ranges_by_week

    # Assert
    assert weekly is not None
    last = weekly.filter(pl.col("week") == weekly.get_column("week").max()).join(
        qb.qb_rating_ranges.select("qb_id", pl.col("qb_rank").alias("season_rank")), on="qb_id"
    )
    assert last.height == len(_TEAMS)
    assert last.get_column("qb_rank").to_list() == last.get_column("season_rank").to_list()


def test_weekly_qb_rank_ranges_start_when_a_qualified_passer_has_played() -> None:
    # Arrange
    qb_games = _qb_games()
    qb_season = build_qb_season(qb_games, _team_games(), bootstrap=_BOOTSTRAP).qb_season
    only_ne = qb_season.with_columns((pl.col("team") == "NE").alias("qb_is_eligible"))

    # Act
    weekly = build_qb_rank_ranges_by_week(
        qb_games, fit_qb_ratings(qb_games), only_ne, bootstrap=_BOOTSTRAP
    )

    # Assert
    first_ne_week = min(week for week, _, home, away in _games() if "NE" in (home, away))
    assert weekly.get_column("week").min() == first_ne_week


def test_weekly_qb_rank_ranges_without_qualified_passers_are_empty() -> None:
    # Arrange
    qb_games = _qb_games()
    qb_season = build_qb_season(qb_games, _team_games(), bootstrap=_BOOTSTRAP).qb_season
    nobody = qb_season.with_columns(pl.lit(value=False).alias("qb_is_eligible"))

    # Act
    weekly = build_qb_rank_ranges_by_week(
        qb_games, fit_qb_ratings(qb_games), nobody, bootstrap=_BOOTSTRAP
    )

    # Assert
    assert weekly.is_empty()
    assert weekly.columns[:3] == ["week", "qb_id", "qb_rank"]
