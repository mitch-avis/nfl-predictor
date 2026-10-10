"""Tests for the ETL's ratings build: the published tables and the pre-week rating features."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from tests.ratings_league import (
    BACKUP_FROM_WEEK,
    BACKUP_ID,
    POSTSEASON_WEEK,
    SEASONS,
    TEAMS,
    WEEKS,
    League,
    league,
    starter_id,
    with_backup_starting,
)
from threadpoolctl import threadpool_info, threadpool_limits

from nfl_predictor import constants
from nfl_predictor.ratings.pbp_inputs import quarterback_rows
from nfl_predictor.ratings.pre_week import RatingsHistory
from nfl_predictor.ratings.season import BootstrapSettings, TeamSeasonRatings
from nfl_predictor.utils.polars import qb_starters, ratings_build

_FAST = BootstrapSettings(resamples=20, seed=0)
_LAST = SEASONS[-1]
_FEATURES = [
    f"{prefix}{stat}" for stat in constants.RATINGS_FEATURE_STATS for prefix in ("away_", "home_")
] + [f"{stat}_diff" for stat in constants.RATINGS_FEATURE_STATS]


def _inputs(data: League) -> ratings_build.RatingsInputs:
    """Return the ratings inputs of a synthetic league."""
    return ratings_build.ratings_inputs(
        ratings_build.LoadedSources(
            plays=data.plays,
            schedule=data.schedule,
            player_stats=data.player_stats,
            snap_counts=data.snap_counts,
            players=data.players,
            weekly_rosters=data.weekly_rosters,
            starter_identity=data.starter_identity,
        )
    )


def _build(
    data: League | None = None, current_season: int = _LAST + 1
) -> ratings_build.RatingsBuild:
    """Return a ratings build of the league whose every season is built."""
    build = ratings_build.RatingsBuild(
        _inputs(data or league()), current_season=current_season, bootstrap=_FAST
    )
    for season in SEASONS:
        build.build(season)
    return build


def _game(games: pl.DataFrame, season: int, week: int, team: str) -> dict[str, Any]:
    """Return the game row of ``team`` in ``season`` and ``week``."""
    rows = games.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & ((pl.col("away_abbr") == team) | (pl.col("home_abbr") == team))
    )
    assert rows.height == 1
    return rows.row(0, named=True)


def _history(data: League) -> RatingsHistory:
    """Return the ratings package's history over the adapter's rows of the league."""
    inputs = _inputs(data)
    team = inputs.team_games
    passers = inputs.passer_games
    return RatingsHistory(
        lambda season: team.filter(pl.col("season") == season).drop("season"),
        lambda season: quarterback_rows(passers.filter(pl.col("season") == season)),
        load_qb_games_before=lambda season, week: quarterback_rows(
            passers.filter(pl.col("season") == season), before_week=week
        ),
    )


@pytest.fixture(scope="module")
def built() -> ratings_build.RatingsBuild:
    """Return the league's build, every season built (shared: building takes a few seconds)."""
    return _build()


def test_build_returns_the_published_tables_and_the_pre_week_tables() -> None:
    # Arrange
    build = ratings_build.RatingsBuild(_inputs(league()), current_season=2005, bootstrap=_FAST)

    # Act
    frames = build.build(2000)

    # Assert
    assert set(frames) == {
        *ratings_build.PUBLISHED_KINDS,
        ratings_build.PRE_WEEK_TEAMS,
        ratings_build.PRE_WEEK_QUARTERBACKS,
    }


def test_build_of_the_current_season_adds_its_weekly_rank_ranges() -> None:
    # Arrange
    build = ratings_build.RatingsBuild(_inputs(league()), current_season=_LAST, bootstrap=_FAST)

    # Act
    frames = build.build(_LAST)

    # Assert
    assert set(ratings_build.IN_PROGRESS_KINDS) <= set(frames)


def test_build_of_a_season_without_games_publishes_no_table() -> None:
    # Arrange
    data = league()
    pre_kickoff = data.plays.filter(pl.col("season") != _LAST)
    build = ratings_build.RatingsBuild(
        _inputs(replace(data, plays=pre_kickoff)), current_season=_LAST, bootstrap=_FAST
    )

    # Act
    frames = build.build(_LAST)

    # Assert
    assert set(frames) == {ratings_build.PRE_WEEK_TEAMS, ratings_build.PRE_WEEK_QUARTERBACKS}


def test_build_of_a_season_without_a_qualified_passer_publishes_only_team_tables() -> None:
    # Arrange
    data = league()
    first_attempts = pl.int_range(pl.len()).over("game_id", "posteam") < 3
    sparse = data.plays.filter(
        (pl.col("season") != 2000) | (pl.col("qb_dropback") == 0.0) | first_attempts
    )
    build = ratings_build.RatingsBuild(
        _inputs(replace(data, plays=sparse)), current_season=_LAST + 1, bootstrap=_FAST
    )

    # Act
    frames = build.build(2000)

    # Assert
    assert set(frames) == {
        "ratings",
        "team_prior",
        "ratings_by_week",
        "rating_ranges",
        "rating_pairs",
        ratings_build.PRE_WEEK_TEAMS,
        ratings_build.PRE_WEEK_QUARTERBACKS,
    }


def test_published_tables_are_named_by_season_and_kind(built: ratings_build.RatingsBuild) -> None:
    # Act
    tables = built.published_tables()

    # Assert
    assert sorted(tables) == sorted(
        f"{season}_{kind}" for season in SEASONS for kind in ratings_build.PUBLISHED_KINDS
    )


def test_published_ratings_rate_every_team(built: ratings_build.RatingsBuild) -> None:
    # Act
    ratings = built.published_tables()[f"{_LAST}_ratings"]

    # Assert
    assert sorted(ratings.get_column("team").to_list()) == sorted(TEAMS)


def test_pre_week_teams_cover_every_team_in_every_scheduled_week(
    built: ratings_build.RatingsBuild,
) -> None:
    # Act
    teams = built.build_frames(_LAST)[ratings_build.PRE_WEEK_TEAMS]

    # Assert
    weeks = [*range(1, WEEKS + 1), POSTSEASON_WEEK]
    assert teams.select("week", "team").sort("week", "team").rows() == [
        (week, team) for week in weeks for team in TEAMS
    ]


def test_pre_week_teams_equal_the_ratings_packages_pre_week_entry_point(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    expected = _history(league()).pre_week_teams(_LAST, 4, TEAMS)

    # Act
    teams = built.build_frames(_LAST)[ratings_build.PRE_WEEK_TEAMS]

    # Assert
    assert_frame_equal(
        teams.filter(pl.col("week") == 4).drop("season", "week"), expected, check_exact=True
    )


def test_pre_week_quarterbacks_equal_the_ratings_packages_pre_week_entry_point(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    expected = _history(league()).pre_week_quarterbacks(_LAST, 5)

    # Act
    quarterbacks = built.build_frames(_LAST)[ratings_build.PRE_WEEK_QUARTERBACKS]

    # Assert
    assert_frame_equal(
        quarterbacks.filter(pl.col("week") == 5).drop("season", "week"),
        expected,
        check_exact=True,
    )


def test_a_postseason_week_takes_the_whole_regular_season(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    teams = built.build_frames(_LAST)[ratings_build.PRE_WEEK_TEAMS]
    after_last_week = _history(league()).pre_week_teams(_LAST, WEEKS + 1, TEAMS)

    # Act
    postseason = teams.filter(pl.col("week") == POSTSEASON_WEEK).drop("season", "week")

    # Assert
    assert_frame_equal(postseason, after_last_week, check_exact=True)


def test_attach_adds_both_teams_pre_week_ratings(built: ratings_build.RatingsBuild) -> None:
    # Arrange
    games = league().games
    teams = built.build_frames(_LAST)[ratings_build.PRE_WEEK_TEAMS].filter(pl.col("week") == 3)
    game = _game(games, _LAST, 3, "AAA")

    # Act
    row = _game(built.attach(games), _LAST, 3, "AAA")

    # Assert
    for side in ("away", "home"):
        team = teams.filter(pl.col("team") == game[f"{side}_abbr"]).row(0, named=True)
        assert row[f"{side}_ratings_offense"] == team["offense_rating"]
        assert row[f"{side}_ratings_defense"] == team["defense_rating"]
        assert row[f"{side}_ratings_special_teams"] == team["special_teams_rating"]
        assert row[f"{side}_ratings_team"] == team["team_rating"]
        assert row[f"{side}_ratings_sos"] == team["sos"]


def test_attach_adds_each_difference_away_minus_home(built: ratings_build.RatingsBuild) -> None:
    # Arrange
    games = league().games

    # Act
    row = _game(built.attach(games), _LAST, 5, "BBB")

    # Assert
    for stat in constants.RATINGS_FEATURE_STATS:
        assert row[f"{stat}_diff"] == row[f"away_{stat}"] - row[f"home_{stat}"]


def test_attach_adds_the_expected_starters_pre_week_rating_and_dropbacks(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    games = league().games
    quarterbacks = built.build_frames(_LAST)[ratings_build.PRE_WEEK_QUARTERBACKS]
    game = _game(games, _LAST, 5, "CCC")
    side = "away" if game["away_abbr"] == "CCC" else "home"
    starter = quarterbacks.filter(
        (pl.col("week") == 5) & (pl.col("qb_id") == starter_id("CCC"))
    ).row(0, named=True)

    # Act
    row = _game(built.attach(games), _LAST, 5, "CCC")

    # Assert
    assert row[f"{side}_ratings_qb"] == starter["adj_qb_epa_per_dropback"]
    assert row[f"{side}_ratings_qb_dropbacks"] == starter["qb_dropbacks"]


def test_attach_gives_a_starter_without_a_dropback_yet_no_rating_and_zero_dropbacks(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    games = with_backup_starting(league().games)
    game = _game(games, _LAST, BACKUP_FROM_WEEK, "AAA")
    side = "away" if game["away_abbr"] == "AAA" else "home"

    # Act
    row = _game(built.attach(games), _LAST, BACKUP_FROM_WEEK, "AAA")

    # Assert
    assert (row[f"{side}_ratings_qb"], row[f"{side}_ratings_qb_dropbacks"]) == (None, 0)


def test_attach_rates_a_games_play_by_play_starter_over_the_named_one(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    games = with_backup_starting(league().games)
    game = _game(games, _LAST, BACKUP_FROM_WEEK, "AAA")
    side = "away" if game["away_abbr"] == "AAA" else "home"
    starters = pl.DataFrame(
        [
            {
                "game_id": game["game_id"],
                "season": _LAST,
                "week": BACKUP_FROM_WEEK,
                "team_abbr": "AAA",
                "qb_id": starter_id("AAA"),
            }
        ],
        schema=qb_starters.STARTERS_SCHEMA,
    )
    quarterbacks = built.build_frames(_LAST)[ratings_build.PRE_WEEK_QUARTERBACKS]
    starter = quarterbacks.filter(
        (pl.col("week") == BACKUP_FROM_WEEK) & (pl.col("qb_id") == starter_id("AAA"))
    ).row(0, named=True)

    # Act
    row = _game(built.attach(games, starters=starters), _LAST, BACKUP_FROM_WEEK, "AAA")

    # Assert
    assert row[f"{side}_ratings_qb"] == starter["adj_qb_epa_per_dropback"]
    assert row[f"{side}_ratings_qb_dropbacks"] == starter["qb_dropbacks"]


def test_attach_rates_the_backup_from_his_own_dropbacks_once_he_has_some(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    games = with_backup_starting(league().games)
    week = BACKUP_FROM_WEEK + 1
    quarterbacks = built.build_frames(_LAST)[ratings_build.PRE_WEEK_QUARTERBACKS]
    backup = quarterbacks.filter((pl.col("week") == week) & (pl.col("qb_id") == BACKUP_ID))
    game = _game(games, _LAST, week, "AAA")
    side = "away" if game["away_abbr"] == "AAA" else "home"

    # Act
    row = _game(built.attach(games), _LAST, week, "AAA")

    # Assert
    assert row[f"{side}_ratings_qb"] == backup.item(0, "adj_qb_epa_per_dropback")


def test_attach_leaves_an_unknown_starters_quarterback_columns_null(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    games = league().games.with_columns(pl.lit("Nobody Known").alias("home_qb"))

    # Act
    row = _game(built.attach(games), _LAST, 3, "AAA")

    # Assert
    assert (row["home_ratings_qb"], row["home_ratings_qb_dropbacks"]) == (None, None)


def test_attach_gives_week_one_starters_no_rating_and_zero_dropbacks(
    built: ratings_build.RatingsBuild,
) -> None:
    # Act
    row = _game(built.attach(league().games), _LAST, 1, "AAA")

    # Assert
    assert (row["away_ratings_qb"], row["away_ratings_qb_dropbacks"]) == (None, 0)
    assert row["away_ratings_sos"] is None


def test_attach_leaves_the_first_seasons_ratings_null(built: ratings_build.RatingsBuild) -> None:
    # Act
    row = _game(built.attach(league().games), SEASONS[0], 4, "AAA")

    # Assert
    assert {column: row[column] for column in _FEATURES} == dict.fromkeys(_FEATURES)


def test_attach_keeps_the_game_rows_and_replaces_earlier_rating_columns(
    built: ratings_build.RatingsBuild,
) -> None:
    # Arrange
    games = league().games.reverse().with_columns(pl.lit(9.0).alias("away_ratings_team"))

    # Act
    attached = built.attach(games)

    # Assert
    assert attached.get_column("game_id").to_list() == games.get_column("game_id").to_list()
    assert attached.columns == [*games.drop("away_ratings_team").columns, *_FEATURES]
    assert (attached.get_column("away_ratings_team") != 9.0).all()


def test_reused_season_frames_attach_the_same_features(built: ratings_build.RatingsBuild) -> None:
    # Arrange
    reused = ratings_build.RatingsBuild(
        _inputs(league()), current_season=_LAST + 1, bootstrap=_FAST
    )
    for season in SEASONS:
        reused.reuse(season, built.build_frames(season))

    # Act
    attached = reused.attach(league().games)

    # Assert
    assert_frame_equal(attached, built.attach(league().games), check_exact=True)


def test_reused_season_frames_publish_the_same_tables(built: ratings_build.RatingsBuild) -> None:
    # Arrange
    reused = ratings_build.RatingsBuild(
        _inputs(league()), current_season=_LAST + 1, bootstrap=_FAST
    )
    for season in SEASONS:
        reused.reuse(season, built.build_frames(season))

    # Act
    tables = reused.published_tables()

    # Assert
    assert tables.keys() == built.published_tables().keys()
    for name, table in tables.items():
        assert_frame_equal(table, built.published_tables()[name], check_exact=True)


def test_reuse_without_the_pre_week_tables_raises() -> None:
    # Arrange
    build = ratings_build.RatingsBuild(_inputs(league()), current_season=_LAST, bootstrap=_FAST)

    # Act & Assert
    with pytest.raises(ValueError, match=ratings_build.PRE_WEEK_TEAMS):
        build.reuse(2000, {})


def test_key_frames_hold_every_seasons_rows_with_their_season() -> None:
    # Arrange
    build = ratings_build.RatingsBuild(_inputs(league()), current_season=_LAST, bootstrap=_FAST)

    # Act
    frames = build.key_frames()

    # Assert
    seasons = {
        name: tuple(sorted(frame.get_column("season").unique().to_list()))
        for name, frame in frames.items()
    }
    assert seasons == dict.fromkeys(frames, SEASONS)


def test_build_fits_with_one_blas_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    seen: list[int] = []
    real = ratings_build.build_team_season

    def recording(*args: Any, **kwargs: Any) -> TeamSeasonRatings:
        seen.extend(info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas")
        return real(*args, **kwargs)

    monkeypatch.setattr(ratings_build, "build_team_season", recording)
    build = ratings_build.RatingsBuild(_inputs(league()), current_season=_LAST, bootstrap=_FAST)

    # Act
    with threadpool_limits(limits=2, user_api="blas"):
        build.build(2000)

    # Assert
    assert seen
    assert set(seen) == {1}


def test_write_published_tables_writes_one_parquet_file_per_table(
    built: ratings_build.RatingsBuild, tmp_path: Path
) -> None:
    # Arrange
    tables = built.published_tables()

    # Act
    ratings_build.write_published_tables(tables, tmp_path / "ratings")

    # Assert
    assert sorted(path.name for path in (tmp_path / "ratings").iterdir()) == sorted(
        f"{name}.parquet" for name in tables
    )
    assert_frame_equal(
        pl.read_parquet(tmp_path / "ratings" / f"{_LAST}_ratings.parquet"),
        tables[f"{_LAST}_ratings"],
    )


# Leakage: every input from a week on (and every later season) is rewritten, the week's
# features must not move.
def _rewritten(data: League, season: int, week: int) -> League:
    """Return the league with every play, roster week and score from a week on rewritten."""
    later = (pl.col("season") > season) | ((pl.col("season") == season) & (pl.col("week") >= week))
    plays = data.plays.with_columns(
        pl.when(later).then(pl.col("epa") * -3.0 + 1.25).otherwise(pl.col("epa")).alias("epa"),
        pl.when(later)
        .then(pl.col("qb_epa") * 2.0 - 0.75)
        .otherwise(pl.col("qb_epa"))
        .alias("qb_epa"),
        pl.when(later & (pl.col("play_type") == "run"))
        .then(pl.lit(0.0))
        .otherwise(pl.col("rush"))
        .alias("rush"),
        pl.when(later & (pl.col("play_type") == "run"))
        .then(pl.lit("no_play"))
        .otherwise(pl.col("play_type"))
        .alias("play_type"),
    )
    rosters = data.weekly_rosters.with_columns(
        pl.when(later).then(pl.lit("WR")).otherwise(pl.col("position")).alias("position"),
        pl.when(later).then(pl.lit("Renamed")).otherwise(pl.col("full_name")).alias("full_name"),
    )
    schedule = data.schedule.with_columns(
        pl.when(later)
        .then(pl.col("home_score") + 21)
        .otherwise(pl.col("home_score"))
        .alias("home_score")
    )
    return replace(data, plays=plays, weekly_rosters=rosters, schedule=schedule)


def _week_features(build: ratings_build.RatingsBuild, season: int, week: int) -> pl.DataFrame:
    """Return the rating features of every game row of ``season``'s ``week``."""
    games = with_backup_starting(league().games)
    return (
        build.attach(games)
        .filter((pl.col("season") == season) & (pl.col("week") == week))
        .select("game_id", *_FEATURES)
        .sort("game_id")
    )


@pytest.mark.parametrize(("season", "week"), [(2000, 1), (2000, 4), (_LAST, 3), (_LAST, WEEKS)])
def test_features_of_a_week_never_see_that_week_or_later(season: int, week: int) -> None:
    # Arrange
    expected = _week_features(_build(), season, week)
    rewritten = _build(_rewritten(league(), season, week))

    # Act
    features = _week_features(rewritten, season, week)

    # Assert
    assert_frame_equal(features, expected, check_exact=True)


def test_rewriting_the_week_before_does_move_the_features() -> None:
    """The leakage guard above is not vacuous: an earlier rewrite moves the week's features."""
    # Arrange
    expected = _week_features(_build(), _LAST, 4)
    rewritten = _build(_rewritten(league(), _LAST, 3))

    # Act
    features = _week_features(rewritten, _LAST, 4)

    # Assert
    assert not features.equals(expected)


# Determinism: two runs, the second fed its inputs in another order, write the same bytes.
def _reordered(data: League, seed: int) -> League:
    """Return the league with games, schedule, rosters and players in another row order.

    Each game's plays keep their order: the EPA sums follow nflverse's play order, which the
    cache keeps, so only whole games move.
    """
    rng = np.random.default_rng(seed)
    games = data.plays.get_column("game_id").unique(maintain_order=True).to_list()
    order = {game: position for position, game in enumerate(rng.permutation(games).tolist())}
    plays = (
        data.plays.with_row_index("_play")
        .with_columns(pl.col("game_id").replace_strict(order, return_dtype=pl.Int64).alias("_game"))
        .sort("_game", "_play")
        .drop("_game", "_play")
    )

    def shuffled(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.sample(fraction=1.0, shuffle=True, seed=seed)

    return replace(
        data,
        plays=plays,
        schedule=shuffled(data.schedule),
        weekly_rosters=shuffled(data.weekly_rosters),
        players=shuffled(data.players),
        starter_identity=shuffled(data.starter_identity),
    )


def test_two_runs_write_byte_identical_tables(tmp_path: Path) -> None:
    # Arrange
    first, second = _build(league()), _build(_reordered(league(), seed=3))
    ratings_build.write_published_tables(first.published_tables(), tmp_path / "first")

    # Act
    ratings_build.write_published_tables(second.published_tables(), tmp_path / "second")

    # Assert
    names = sorted(path.name for path in (tmp_path / "first").iterdir())
    assert names == sorted(path.name for path in (tmp_path / "second").iterdir())
    assert all(
        (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
        for name in names
    )


def test_two_runs_attach_identical_features() -> None:
    # Arrange
    games = with_backup_starting(league().games)
    expected = _build(league()).attach(games)
    second = _build(_reordered(league(), seed=5))

    # Act
    attached = second.attach(games)

    # Assert
    assert_frame_equal(attached, expected, check_exact=True)


def _passer_games(dropbacks: list[int | None]) -> pl.DataFrame:
    """Return one game each for a qualified passer (30 attempts) and a backup (5 attempts)."""
    return pl.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "week": [1, 1],
            "team": ["AAA", "BBB"],
            "opponent_team": ["BBB", "AAA"],
            "qb_id": ["QUALIFIED", "BACKUP"],
            "qb_name": ["Qualified Passer", "Backup Passer"],
            "qb_attempts": [30, 5],
            "qb_dropbacks": dropbacks,
            "qb_dropback_epa": [None if count is None else 2.0 for count in dropbacks],
        },
        schema_overrides={"qb_dropbacks": pl.Int64, "qb_dropback_epa": pl.Float64},
    )


@pytest.mark.parametrize(
    ("dropbacks", "qualified"),
    [([35, 10], True), ([None, 10], False)],
    ids=["qualified-and-rated", "qualified-but-unrated"],
)
def test_quarterback_tables_need_a_qualified_passer_the_fit_rates(
    dropbacks: list[int | None], *, qualified: bool
) -> None:
    # Arrange
    team_games = pl.DataFrame({"team": ["AAA", "BBB"]})

    # Act
    found = ratings_build._has_qualified_passer(_passer_games(dropbacks), team_games)

    # Assert
    assert found is qualified


def test_published_tables_replace_each_file_whole(tmp_path: Path) -> None:
    # Arrange
    directory = tmp_path / "ratings"
    ratings_build.write_published_tables({"2024_ratings": pl.DataFrame({"a": [1]})}, directory)

    # Act
    ratings_build.write_published_tables({"2024_ratings": pl.DataFrame({"a": [2]})}, directory)

    # Assert
    assert pl.read_parquet(directory / "2024_ratings.parquet").get_column("a").to_list() == [2]
    assert sorted(path.name for path in directory.iterdir()) == ["2024_ratings.parquet"]


def test_a_failed_table_write_leaves_the_published_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    directory = tmp_path / "ratings"
    ratings_build.write_published_tables({"2024_ratings": pl.DataFrame({"a": [1]})}, directory)

    def interrupted(_frame: pl.DataFrame, file: str | Path, *_args: Any, **_kwargs: Any) -> None:
        Path(file).write_bytes(b"half a parquet file")
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", interrupted)

    # Act
    with pytest.raises(OSError, match="disk full"):
        ratings_build.write_published_tables({"2024_ratings": pl.DataFrame({"a": [2]})}, directory)

    # Assert
    monkeypatch.undo()
    assert pl.read_parquet(directory / "2024_ratings.parquet").get_column("a").to_list() == [1]
    assert sorted(path.name for path in directory.iterdir()) == ["2024_ratings.parquet"]
