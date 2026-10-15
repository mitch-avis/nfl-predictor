"""Tests for attaching the quarterback family during data collection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn, cast

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.ml import feature_spec
from nfl_predictor.utils import game_utils, polars_utils
from nfl_predictor.utils.polars import nflverse_sources, player_identity, qb_starters
from nfl_predictor.utils.polars.finalize import build_final_column_order
from tests.fakes import returning

if TYPE_CHECKING:
    from pathlib import Path

    from nfl_predictor.utils.polars import ratings_build


def _same_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return ``df`` unchanged."""
    return df


def _same_frame_of_two(df: pl.DataFrame, _other: object) -> pl.DataFrame:
    """Return ``df`` unchanged, ignoring the second argument."""
    return df


def _same_games(games: pl.DataFrame, *_args: object, **_kwargs: object) -> pl.DataFrame:
    """Return ``games`` unchanged, ignoring every other argument."""
    return games


def _dropbacks(
    season: int, week: int, team: str, passer: str | None, count: int
) -> list[dict[str, Any]]:
    """Return ``count`` completed-pass dropbacks by ``passer`` for ``team``."""
    return [
        {
            "season": season,
            "week": week,
            "season_type": "REG",
            "posteam": team,
            "defteam": "OPP",
            "qb_dropback": 1,
            "sack": 0,
            "complete_pass": 1,
            "yards_gained": 5.0,
            "qb_epa": 0.2,
            "passer_player_id": passer,
            "passer_player_name": None,
        }
        for _ in range(count)
    ]


def _season_identity(positions: dict[str, str]) -> Any:
    """Return a season identity loader listing ``positions`` (id -> position) every season."""

    def load(season: int, **_kwargs: Any) -> pl.DataFrame:
        del season
        return pl.DataFrame(
            {
                "qb_id": list(positions),
                "snap_player_id": [None] * len(positions),
                "qb_name": list(positions),
                "qb_position": list(positions.values()),
            },
            schema=player_identity.QB_IDENTITY_SCHEMA,
        )

    return load


def _first_quarter_game(
    season: int, week: int, team: str, passers: list[str]
) -> list[dict[str, Any]]:
    """Return one team's first-quarter dropbacks in a game, in order, one per passer listed."""
    return [
        {
            **_dropbacks(season, week, team, passer, 1)[0],
            "game_id": f"{season}_{week:02d}_{team}_OPP",
            "qtr": 1.0,
            "play_id": float(index + 1),
        }
        for index, passer in enumerate(passers)
    ]


def _no_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """List nobody in any season's quarterback identity, so every passer may start."""
    monkeypatch.setattr(nflverse_sources, "load_season_qb_identity", _season_identity({}))


def _identity_file(tmp_path: Path) -> Path:
    """Write a two-quarterback identity file and return its path."""
    path = tmp_path / "meta_data.csv"
    pl.DataFrame({"name_id": ["Tom Brady", "Josh Allen"], "gsis_id": ["TB", "JA"]}).write_csv(path)
    return path


def _two_games(*week_and_ids: tuple[int, str]) -> pl.DataFrame:
    """Return Buffalo's game rows at OPP for each ``(week, game_id)``, named for Josh Allen."""
    return pl.DataFrame(
        {
            "game_id": [game_id for _week, game_id in week_and_ids],
            "season": [2001] * len(week_and_ids),
            "week": [week for week, _game_id in week_and_ids],
            "away_abbr": ["BUF"] * len(week_and_ids),
            "home_abbr": ["OPP"] * len(week_and_ids),
            "away_qb": ["Josh Allen"] * len(week_and_ids),
            "home_qb": [None] * len(week_and_ids),
        }
    )


def test_attach_qb_features_keys_a_completed_game_on_its_play_by_play_starter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A game the named starter did not start is keyed on the quarterback who did."""
    # Arrange
    in_memory = pl.DataFrame(
        _first_quarter_game(2001, 1, "BUF", ["JA"] * 12)
        + _first_quarter_game(2001, 2, "BUF", ["BK", "JA"])
    )
    monkeypatch.setattr(polars_utils, "load_pbp", returning(pl.DataFrame()))
    monkeypatch.setattr(
        nflverse_sources, "load_season_qb_identity", _season_identity({"JA": "QB", "BK": "QB"})
    )
    history = data_collection._quarterback_history(in_memory, max_season=2001, current_season=2026)
    games = _two_games((3, "2001_03_BUF_OPP"), (2, "2001_02_BUF_OPP"))

    # Act
    out = data_collection._attach_qb_features(
        games, history, identity_path=_identity_file(tmp_path)
    )

    # Assert
    dropbacks = dict(zip(out["game_id"], out["away_qb_history_dropbacks"], strict=True))
    assert dropbacks == {"2001_03_BUF_OPP": 13, "2001_02_BUF_OPP": 0}


def test_quarterback_history_loads_the_missing_seasons_from_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seasons missing from memory are loaded, never refreshed, so careers span the history."""
    # Arrange
    in_memory = pl.DataFrame(_dropbacks(2001, 1, "NE", "TB", 20))
    earlier = pl.DataFrame(_dropbacks(2000, 5, "NE", "TB", 30))
    requested: list[list[int]] = []
    refreshed: list[bool] = []

    def fake_load_pbp(seasons: list[int], **kwargs: Any) -> pl.DataFrame:
        """Record the requested seasons and refresh flag, return the earlier history."""
        requested.append(list(seasons))
        refreshed.append(bool(kwargs.get("force_refresh")))
        return earlier

    monkeypatch.setattr(polars_utils, "load_pbp", fake_load_pbp)
    _no_identity(monkeypatch)

    # Act
    history = data_collection._quarterback_history(in_memory, max_season=2001, current_season=2026)

    # Assert
    assert requested == [list(range(constants.NFLREADPY_MIN_SEASON, 2001))]
    assert refreshed == [False], "history seasons always come from the cache"
    assert history.qb_games.get_column("dropbacks").sum() == 50


def test_quarterback_history_never_requests_a_season_the_run_already_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A season the team-stat load asked for and got no plays for is not asked for again."""
    # Arrange
    in_memory = pl.DataFrame(_dropbacks(2000, 1, "NE", "TB", 5))
    requested: list[list[int]] = []

    def fake_load_pbp(seasons: list[int], **_kwargs: Any) -> pl.DataFrame:
        """Record the requested seasons."""
        requested.append(list(seasons))
        return pl.DataFrame(_dropbacks(1999, 1, "NE", "TB", 3))

    monkeypatch.setattr(polars_utils, "load_pbp", fake_load_pbp)
    _no_identity(monkeypatch)

    # Act
    data_collection._quarterback_history(
        in_memory, max_season=2001, current_season=2001, requested_seasons=(2000, 2001)
    )

    # Assert
    assert requested == [[1999]]


def _identity_sources(season: int, gsis_id: str, position: str) -> Any:
    """Return shared identity sources whose weekly roster lists ``gsis_id`` in ``season``."""
    players = pl.DataFrame(schema=dict.fromkeys(constants.NFLREADPY_PLAYERS_COLUMNS, pl.String))
    weekly_rosters = pl.DataFrame(
        {
            "season": [season],
            "week": [1],
            "game_type": ["REG"],
            "team": ["NE"],
            "position": [position],
            "full_name": ["Tom Brady"],
            "gsis_id": [gsis_id],
            "pfr_id": [None],
        },
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "game_type": pl.String,
            "team": pl.String,
            "position": pl.String,
            "full_name": pl.String,
            "gsis_id": pl.String,
            "pfr_id": pl.String,
        },
    )
    return nflverse_sources.QbIdentitySources(players=players, weekly_rosters=weekly_rosters)


def test_quarterback_history_reads_the_shared_identity_sources_without_loading_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run's one load of the players file and weekly rosters gives every season's identity."""
    # Arrange
    monkeypatch.setattr(constants, "NFLREADPY_MIN_SEASON", 2001)
    in_memory = pl.DataFrame(_dropbacks(2001, 1, "NE", "TB", 5))

    def load_again(*_a: object, **_k: object) -> NoReturn:
        pytest.fail("the identity was loaded again")

    monkeypatch.setattr(nflverse_sources, "load_season_qb_identity", load_again)

    # Act
    history = data_collection._quarterback_history(
        in_memory,
        max_season=2001,
        current_season=2026,
        identity_sources=_identity_sources(2001, "TB", "WR"),
    )

    # Assert
    assert history.positions.rows() == [(2001, "TB", "WR")]


def test_finish_games_hands_the_quarterback_history_the_shared_identity_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    shared = _identity_sources(2001, "TB", "QB")
    handed: list[Any] = []

    def history(*_args: Any, **kwargs: Any) -> None:
        handed.append(kwargs.get("identity_sources"))

    monkeypatch.setattr(data_collection, "_quarterback_history", history)
    monkeypatch.setattr(data_collection, "_attach_qb_features", _same_games)
    monkeypatch.setattr(data_collection, "_attach_own_elo", _same_games)
    monkeypatch.setattr(game_utils, "fill_future_qb_data", _same_frame_of_two)
    monkeypatch.setattr(game_utils, "fill_future_game_lines", _same_frame)
    monkeypatch.setattr(game_utils, "fill_missing_moneylines", _same_frame)
    monkeypatch.setattr(polars_utils, "select_final_columns", _same_frame)
    sources = data_collection._EtlSources(
        current_season=2025,
        current_week=1,
        schedule_df=pl.DataFrame(),
        team_stats_df=pl.DataFrame(),
        pbp_df=pl.DataFrame(),
        elo_df=pl.DataFrame(),
        raw_elo_df=pl.DataFrame(),
        qb_identity_sources=shared,
    )
    games = pl.DataFrame({"season": [2024], "week": [2], "away_qb": ["A"], "home_qb": ["B"]})

    # Act
    data_collection._finish_games(games, sources, 2024, enable_timing=False)

    # Assert
    assert handed == [shared]


def test_attach_qb_features_joins_both_sides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each side gets its own quarterback's career, and the diff is away minus home."""
    # Arrange
    in_memory = pl.DataFrame(
        _dropbacks(2001, 1, "NE", "TB", 20) + _dropbacks(2001, 1, "BUF", "JA", 12)
    )
    monkeypatch.setattr(
        polars_utils,
        "load_pbp",
        returning(pl.DataFrame(_dropbacks(2000, 5, "NE", "TB", 30))),
    )
    _no_identity(monkeypatch)
    history = data_collection._quarterback_history(in_memory, max_season=2001, current_season=2026)
    games = pl.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2001],
            "week": [2],
            "away_qb": ["Tom Brady"],
            "home_qb": ["Josh Allen"],
        }
    )

    # Act
    out = data_collection._attach_qb_features(
        games, history, identity_path=_identity_file(tmp_path)
    )

    # Assert
    row = out.row(0, named=True)
    assert row["away_qb_history_dropbacks"] == 50
    assert row["home_qb_history_dropbacks"] == 12
    assert row["qb_history_dropbacks_diff"] == 38
    assert out["game_id"].to_list() == ["g1"]


def test_attach_qb_features_reads_the_nfeloqb_identity_file_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without an explicit path, the identity map comes from ``DATA_PATH/meta_data.csv``."""
    # Arrange
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)
    monkeypatch.setattr(polars_utils, "load_pbp", returning(pl.DataFrame()))
    _no_identity(monkeypatch)
    _identity_file(tmp_path)
    in_memory = pl.DataFrame(
        _dropbacks(2001, 1, "NE", "TB", 20) + _dropbacks(2001, 1, "BUF", "JA", 12)
    )
    history = data_collection._quarterback_history(in_memory, max_season=2001, current_season=2026)
    games = pl.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2001],
            "week": [2],
            "away_qb": ["Tom Brady"],
            "home_qb": ["Josh Allen"],
        }
    )

    # Act
    out = data_collection._attach_qb_features(games, history)

    # Assert
    row = out.row(0, named=True)
    assert row["away_qb_history_dropbacks"] == 20
    assert row["home_qb_history_dropbacks"] == 12


@pytest.mark.parametrize(
    ("columns", "with_history"),
    [(["season", "week", "away_qb", "home_qb"], False), (["season", "week"], True)],
    ids=["no_history", "no_names"],
)
def test_attach_qb_features_without_a_history_or_quarterback_names_changes_nothing(
    columns: list[str], *, with_history: bool
) -> None:
    """A run without a quarterback history, or rows without names, come back unchanged."""
    # Arrange
    games = pl.DataFrame({column: [None] for column in columns})
    history = (
        qb_starters.quarterback_history(pl.DataFrame(), qb_starters.season_positions({}))
        if with_history
        else None
    )

    # Act
    out = data_collection._attach_qb_features(games, history)

    # Assert
    assert out.equals(games)


class _RecordingRatings:
    """A ratings build that records the play-by-play starters each attach is handed."""

    def __init__(self) -> None:
        self.starters: list[pl.DataFrame | None] = []

    def attach(self, games: pl.DataFrame, *, starters: pl.DataFrame | None = None) -> Any:
        self.starters.append(starters)
        return games


def test_finish_games_hands_the_ratings_the_runs_play_by_play_starters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    starters = pl.DataFrame(
        [{"game_id": "g", "season": 2024, "week": 2, "team_abbr": "NE", "qb_id": "QB1"}],
        schema=qb_starters.STARTERS_SCHEMA,
    )
    history = qb_starters.QuarterbackHistory(
        pl.DataFrame(), pl.DataFrame(), qb_starters.season_positions({}), starters
    )
    monkeypatch.setattr(data_collection, "_quarterback_history", returning(history))
    monkeypatch.setattr(data_collection, "_attach_qb_features", _same_games)
    monkeypatch.setattr(data_collection, "_attach_own_elo", _same_games)
    monkeypatch.setattr(game_utils, "fill_future_qb_data", _same_frame_of_two)
    monkeypatch.setattr(game_utils, "fill_future_game_lines", _same_frame)
    monkeypatch.setattr(game_utils, "fill_missing_moneylines", _same_frame)
    monkeypatch.setattr(polars_utils, "select_final_columns", _same_frame)
    ratings = _RecordingRatings()
    sources = data_collection._EtlSources(
        current_season=2025,
        current_week=1,
        schedule_df=pl.DataFrame(),
        team_stats_df=pl.DataFrame(),
        pbp_df=pl.DataFrame(),
        elo_df=pl.DataFrame(),
        raw_elo_df=pl.DataFrame(),
        ratings=cast("ratings_build.RatingsBuild", ratings),
    )
    games = pl.DataFrame({"season": [2024], "week": [2], "away_qb": ["A"], "home_qb": ["B"]})

    # Act
    data_collection._finish_games(games, sources, 2024, enable_timing=False)

    # Assert
    assert ratings.starters == [starters]


def test_final_schema_carries_no_defense_adjusted_quarterback_columns() -> None:
    """A build publishes the per-dropback quarterback family and no defense-adjusted rate."""
    # Act
    order = build_final_column_order()

    # Assert
    assert "away_qb_dropback_epa" in order
    assert [column for column in order if "def_adj" in column] == []


def test_defense_adjusted_quarterback_group_is_not_a_feature_group() -> None:
    """The retired defense-adjusted quarterback rate is no longer an ablation group."""
    # Arrange
    order = build_final_column_order()

    # Act & Assert
    with pytest.raises(ValueError, match="Unknown feature group"):
        feature_spec.resolve_feature_group_columns(order, ["qb_def_adj"])
