"""Tests for the team and quarterback game logs built from play-by-play and official stats."""

import polars as pl
import pytest

from nfl_predictor.season_stats import game_logs, qb_stats
from nfl_predictor.season_stats.metrics import get_registry
from nfl_predictor.season_stats.pooled_rates import (
    denominator_column,
    is_rate_part,
    numerator_column,
    rate_parts,
)
from nfl_predictor.utils.polars.player_identity import QB_IDENTITY_SCHEMA, qb_identity_crosswalk

_GAME = "2025_01_DEN_KC"


def _team_pbp() -> pl.DataFrame:
    """Return one DEN-KC game: a DEN completion, a DEN run, and a KC pass intercepted."""
    return pl.DataFrame(
        {
            "game_id": [_GAME] * 3,
            "season": [2025] * 3,
            "season_type": ["REG"] * 3,
            "week": [1] * 3,
            "posteam": ["DEN", "DEN", "KC"],
            "defteam": ["KC", "KC", "DEN"],
            "pass": [1, 0, 1],
            "rush": [0, 1, 0],
            "qb_dropback": [1, 0, 1],
            "qb_kneel": [0, 0, 0],
            "qb_spike": [0, 0, 0],
            "passing_yards": [20.0, 0.0, 15.0],
            "rushing_yards": [0.0, 10.0, 0.0],
            "epa": [1.0, 0.5, -0.5],
            "pass_touchdown": [1, 0, 0],
            "rush_touchdown": [0, 0, 0],
            "first_down": [1, 1, 1],
            "cpoe": [4.0, None, -2.0],
            "sack": [0, 0, 0],
            "interception": [0, 0, 1],
            "fumble_lost": [0, 0, 0],
        }
    )


def _defense_player_stats() -> pl.DataFrame:
    """Return DEN's defensive box score of the game: two sacks and four passes defended."""
    return pl.DataFrame(
        {
            "season": [2025],
            "season_type": ["REG"],
            "week": [1],
            "team": ["DEN"],
            "opponent_team": ["KC"],
            "def_sacks": [2.0],
            "def_pass_defended": [4],
        }
    )


def _schedule() -> pl.DataFrame:
    """Return the game's final score: DEN 24, KC 17."""
    return pl.DataFrame(
        {
            "game_id": [_GAME],
            "game_type": ["REG"],
            "home_team": ["DEN"],
            "away_team": ["KC"],
            "week": [1],
            "home_score": [24],
            "away_score": [17],
        }
    )


def _official_team_stats() -> pl.DataFrame:
    """Return the game's official weekly team stats, which differ from its play-by-play."""
    return pl.DataFrame(
        {
            "season": [2025, 2025],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "game_id": [_GAME, _GAME],
            "team": ["DEN", "KC"],
            "opponent_team": ["KC", "DEN"],
            "passing_yards": [250.0, 180.0],
            "rushing_yards": [110.0, 90.0],
            "passing_tds": [2, 1],
            "rushing_tds": [1, 0],
            "passing_first_downs": [12, 9],
            "rushing_first_downs": [6, 4],
            "passing_epa": [5.5, 1.7],
            "rushing_epa": [1.2, 0.4],
            "passing_cpoe": [3.5, -1.5],
            "sacks_suffered": [2, 3],
            "passing_interceptions": [1, 2],
            "sack_fumbles_lost": [1, 0],
            "rushing_fumbles_lost": [0, 1],
        }
    )


def test_build_weekly_team_stats_adds_scores_and_box_score_defense() -> None:
    # Act
    result = game_logs.build_weekly_team_stats(
        _team_pbp(), _defense_player_stats(), _schedule(), pl.DataFrame()
    )

    # Assert
    den = result.filter(pl.col("team") == "DEN").row(0, named=True)
    assert result.height == 2
    assert (den["total_yards"], den["points_for"], den["points_allowed"]) == (30.0, 24, 17)
    assert den["def_sacks"] == 2.0


def test_build_weekly_team_stats_prefers_the_official_team_stats() -> None:
    # Act
    result = game_logs.build_weekly_team_stats(
        _team_pbp(), _defense_player_stats(), _schedule(), _official_team_stats()
    )

    # Assert
    den = result.filter(pl.col("team") == "DEN").row(0, named=True)
    assert (den["passing_yards"], den["rushing_yards"], den["total_yards"]) == (250, 110, 360)
    assert (den["passing_tds"], den["rushing_tds"]) == (2, 1)
    assert (den["passing_first_downs"], den["rushing_first_downs"]) == (12, 6)
    assert (den["passing_epa"], den["rushing_epa"], den["passing_cpoe"]) == (5.5, 1.2, 3.5)
    assert (den["sacks_suffered"], den["passing_interceptions"]) == (2, 1)
    assert (den["sack_fumbles_lost"], den["rushing_fumbles_lost"]) == (1, 0)
    assert (den["passing_epa_allowed"], den["rushing_epa_allowed"]) == (1.7, 0.4)
    assert (den["passing_first_downs_allowed"], den["rushing_first_downs_allowed"]) == (9, 4)
    assert den["passing_cpoe_allowed"] == -1.5
    assert den["passing_epa_per_offensive_snap"] == pytest.approx(5.5 / 2.0)
    assert den["passing_epa_allowed_per_defensive_snap"] == pytest.approx(1.7)


def test_build_weekly_team_stats_pools_per_snap_rates_over_the_official_totals() -> None:
    # Act
    result = game_logs.build_weekly_team_stats(
        _team_pbp(), _defense_player_stats(), _schedule(), _official_team_stats()
    )

    # Assert
    den = result.filter(pl.col("team") == "DEN").row(0, named=True)
    rate = "passing_yards_per_offensive_snap"
    assert den[numerator_column(rate)] == 250.0
    assert den[rate] == pytest.approx(den[numerator_column(rate)] / den[denominator_column(rate)])
    for cpoe, official in (("passing_cpoe", 3.5), ("passing_cpoe_allowed", -1.5)):
        assert den[cpoe] == official
        assert den[numerator_column(cpoe)] == pytest.approx(
            official * den[denominator_column(cpoe)]
        )


def test_build_weekly_team_stats_keeps_play_by_play_where_one_team_has_official_rows() -> None:
    """A game nflverse credits wholly to one team keeps its play-by-play totals."""
    # Arrange
    official = _official_team_stats().with_columns(pl.lit("DEN").alias("team"))

    # Act
    result = game_logs.build_weekly_team_stats(
        _team_pbp(), _defense_player_stats(), _schedule(), official
    )

    # Assert
    den = result.filter(pl.col("team") == "DEN").row(0, named=True)
    assert (den["passing_yards"], den["rushing_yards"]) == (20.0, 10.0)


def test_two_team_games_only_drops_a_game_with_one_teams_rows() -> None:
    # Arrange
    stats = pl.DataFrame(
        {
            "game_id": ["2002_01_IND_JAX", "2002_01_IND_JAX", "2002_02_JAX_KC", "2002_02_JAX_KC"],
            "team": ["IND", "IND", "JAX", "KC"],
        }
    )

    # Act
    kept = game_logs.two_team_games_only(stats)

    # Assert
    assert kept.get_column("game_id").to_list() == ["2002_02_JAX_KC", "2002_02_JAX_KC"]


def test_official_weekly_team_surface_without_rows_is_typed_empty() -> None:
    # Act
    surface = game_logs.official_weekly_team_surface(pl.DataFrame())

    # Assert
    assert surface.is_empty()
    assert "official_passing_yards" in surface.columns


def test_override_team_game_stats_without_official_rows_keeps_the_input() -> None:
    # Arrange
    team_df = pl.DataFrame({"team": ["DEN"], "week": [1], "passing_yards": [250.0]})

    # Act
    result = game_logs.override_team_game_stats_with_official_weekly(team_df, pl.DataFrame())

    # Assert
    assert result.equals(team_df)


def _players(position: str = "QB") -> pl.DataFrame:
    """Return the players file row of John Doe (GSIS 00-0031234, PFR DoeJo00)."""
    return pl.DataFrame(
        {
            "gsis_id": ["00-0031234"],
            "display_name": ["John Doe"],
            "position": [position],
            "pfr_id": ["DoeJo00"],
        }
    )


def _identity(players: pl.DataFrame | None = None) -> pl.DataFrame:
    """Return the season's quarterback identity from ``players`` and no weekly roster."""
    return qb_identity_crosswalk(
        _players() if players is None else players,
        pl.DataFrame(schema={"gsis_id": pl.String, "full_name": pl.String}),
    )


def _passer_pbp() -> pl.DataFrame:
    """Return John Doe's game: an 18-yard completion and a sack, tagged by his short name."""
    return pl.DataFrame(
        {
            "game_id": [_GAME, _GAME],
            "season_type": ["REG", "REG"],
            "week": [1, 1],
            "posteam": ["DEN", "DEN"],
            "passer_player_id": ["00-0031234", "00-0031234"],
            "passer_player_name": ["J.Doe", "J.Doe"],
            "qb_dropback": [1, 1],
            "pass": [1, 0],
            "complete_pass": [1, 0],
            "passing_yards": [18.0, 0.0],
            "pass_touchdown": [0, 0],
            "interception": [0, 0],
            "sack": [0, 1],
            "fumble_lost": [0, 0],
            "qb_epa": [1.2, -0.4],
            "cpoe": [4.0, None],
        }
    )


def _snap_counts() -> pl.DataFrame:
    """Return John Doe's 60 offensive snaps, keyed by his PFR id."""
    return pl.DataFrame(
        {
            "game_id": [_GAME],
            "week": [1],
            "team": ["DEN"],
            "player": ["John Doe"],
            "pfr_player_id": ["DoeJo00"],
            "position": ["QB"],
            "offense_snaps": [60.0],
        }
    )


def test_build_qb_game_stats_merges_play_by_play_and_snap_counts_by_identity() -> None:
    # Act
    result = game_logs.build_qb_game_stats(
        _passer_pbp(), _snap_counts(), _identity(), pl.DataFrame()
    )

    # Assert
    row = result.row(0, named=True)
    assert result.height == 1
    assert (row["qb_id"], row["qb_name"], row["snap_player_id"]) == (
        "00-0031234",
        "John Doe",
        "DoeJo00",
    )
    assert (row["qb_dropbacks"], row["qb_offense_snaps"], row["qb_attempts"]) == (2, 60, 1)
    assert row["qb_passing_epa"] == pytest.approx(0.8)
    assert row["qb_epa_per_dropback"] == pytest.approx(0.4)
    assert (row["qb_sack_rate"], row["qb_any_a"], row["qb_passer_rating"]) == (0.5, 9.0, 118.8)


def test_build_qb_game_stats_leaves_out_a_passer_listed_at_another_position() -> None:
    # Arrange
    pbp = pl.concat(
        [
            _passer_pbp().head(1),
            _passer_pbp()
            .head(1)
            .with_columns(
                pl.lit("00-0099999").alias("passer_player_id"),
                pl.lit("G.Guy").alias("passer_player_name"),
            ),
        ]
    )
    players = pl.concat(
        [
            _players(),
            pl.DataFrame(
                {
                    "gsis_id": ["00-0099999"],
                    "display_name": ["Gadget Guy"],
                    "position": ["WR"],
                    "pfr_id": ["GadgGu00"],
                }
            ),
        ]
    )

    # Act
    result = game_logs.build_qb_game_stats(pbp, _snap_counts(), _identity(players), pl.DataFrame())

    # Assert
    assert result.get_column("qb_id").to_list() == ["00-0031234"]


def _official_passing() -> pl.DataFrame:
    """Return John Doe's official weekly passing line, which differs from the play-by-play."""
    return pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "season_type": ["REG"],
            "game_id": [_GAME],
            "team": ["DEN"],
            "opponent_team": ["KC"],
            "player_id": ["00-0031234"],
            "player_display_name": ["John Doe"],
            "position": ["QB"],
            "attempts": [5],
            "completions": [4],
            "passing_yards": [50.0],
            "passing_tds": [1],
            "passing_interceptions": [1],
            "sacks_suffered": [1],
            # nflverse stores the yards lost on sacks as a negative number.
            "sack_yards_lost": [-8.0],
            "passing_epa": [3.0],
            "passing_cpoe": [2.5],
        }
    )


def test_build_qb_game_stats_prefers_the_official_passing_line() -> None:
    # Act
    result = game_logs.build_qb_game_stats(
        _passer_pbp(), _snap_counts(), _identity(), _official_passing()
    )

    # Assert
    row = result.row(0, named=True)
    cpoe = "qb_completion_percentage_above_expectation"
    assert (row["qb_dropbacks"], row["qb_attempts"], row["qb_completions"]) == (2, 5, 4)
    assert (row["qb_pass_yards"], row["qb_pass_touchdowns"], row["qb_interceptions"]) == (
        50.0,
        1,
        1,
    )
    assert (row["qb_sacks"], row["qb_sack_yards_lost"], row["qb_passing_epa"]) == (1, 8.0, 3.0)
    assert (row[cpoe], row[numerator_column(cpoe)], row[denominator_column(cpoe)]) == (
        2.5,
        2.5,
        1,
    )
    assert (row["qb_epa_per_dropback"], row["qb_pass_yards_per_dropback"]) == (1.5, 25.0)
    assert row["qb_any_a"] == pytest.approx(17.0 / 6.0)
    assert row["qb_passer_rating"] == 108.3


def test_build_qb_game_stats_keeps_every_input_of_the_rates_it_pools() -> None:
    """Season and opponent rows rebuild each pooled rate from its totals (``QB_RATES``)."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": [_GAME] * 3,
            "season_type": ["REG"] * 3,
            "week": [1, 1, 1],
            "posteam": ["DEN"] * 3,
            "defteam": ["KC"] * 3,
            "passer_player_id": ["00-0031234"] * 3,
            "passer_player_name": ["John Doe"] * 3,
            "play_type": ["pass", "pass", "qb_spike"],
            "qb_dropback": [1, 1, 0],
            "qb_spike": [0, 0, 1],
            "pass": [1, 0, 1],
            "complete_pass": [1, 0, 0],
            "passing_yards": [18.0, 0.0, 0.0],
            "pass_touchdown": [0, 0, 0],
            "interception": [0, 0, 0],
            "sack": [0, 1, 0],
            "fumble_lost": [0, 0, 0],
            "qb_epa": [1.2, -0.4, -0.2],
        }
    )

    # Act
    result = game_logs.build_qb_game_stats(pbp, pl.DataFrame(), _identity(), pl.DataFrame())

    # Assert
    missing = {
        rate: sorted(set(qb_stats.QB_RATES[rate].inputs) - set(result.columns))
        for rate in qb_stats.QB_RATES
        if rate in result.columns
    }
    assert {rate: inputs for rate, inputs in missing.items() if inputs} == {}
    assert result.get_column("qb_spike_epa").to_list() == [pytest.approx(-0.2)]


def test_build_qb_game_stats_leaves_snaps_unknown_without_snap_counts() -> None:
    # Act
    result = game_logs.build_qb_game_stats(
        _passer_pbp(), pl.DataFrame(), _identity(), pl.DataFrame()
    )

    # Assert
    assert result.get_column("qb_offense_snaps").to_list() == [None]


def test_official_weekly_qb_stats_without_player_ids_is_empty() -> None:
    # Act
    official = game_logs.official_weekly_qb_stats(pl.DataFrame({"week": [1]}), pl.DataFrame())

    # Assert
    assert official.is_empty()


def test_official_weekly_qb_stats_without_an_identity_crosswalk_keeps_player_ids() -> None:
    # Arrange
    player_stats = pl.DataFrame(
        {
            "game_id": ["g1"],
            "week": [1],
            "team": ["DEN"],
            "player_id": ["00-0039732"],
            "player_display_name": ["Bo Nix"],
            "position": ["QB"],
            "attempts": [30],
        }
    )

    # Act
    official = game_logs.official_weekly_qb_stats(
        player_stats, pl.DataFrame(schema=QB_IDENTITY_SCHEMA)
    )

    # Assert
    assert official.get_column("qb_id").to_list() == ["00-0039732"]
    assert "qb_name" not in official.columns


def test_official_weekly_qb_stats_keeps_a_quarterback_listed_elsewhere() -> None:
    """The crosswalk, not the stats' latest position, decides who is a quarterback that season."""
    # Arrange
    player_stats = pl.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "week": [1, 1],
            "team": ["NO", "NO"],
            "player_id": ["00-0033357", "00-0020001"],
            "player_display_name": ["Taysom Hill", "A. Receiver"],
            "position": ["TE", "WR"],
            "attempts": [20, 1],
        }
    )
    crosswalk = pl.DataFrame(
        {
            "qb_id": ["00-0033357", "00-0020001"],
            "snap_player_id": ["HillTa00", None],
            "qb_name": ["Taysom Hill", "A. Receiver"],
            "qb_position": ["QB", "WR"],
        }
    )

    # Act
    official = game_logs.official_weekly_qb_stats(player_stats, crosswalk)

    # Assert
    assert official.get_column("qb_id").to_list() == ["00-0033357"]


def test_override_qb_game_stats_without_official_rows_keeps_the_input() -> None:
    # Arrange
    qb_df = pl.DataFrame({"qb_id": ["q1"], "week": [1], "qb_attempts": [30]})

    # Act
    result = game_logs.override_qb_game_stats_with_official_weekly(qb_df, pl.DataFrame())

    # Assert
    assert result.equals(qb_df)


def test_override_qb_game_stats_leaves_spikes_out_of_epa_per_dropback() -> None:
    """Official passing EPA counts the spike (-0.2): EPA per dropback is (0.6 + 0.2) / 2."""
    # Arrange
    pbp = pl.DataFrame(
        {
            "game_id": [_GAME] * 3,
            "week": [1, 1, 1],
            "posteam": ["DEN"] * 3,
            "defteam": ["KC"] * 3,
            "passer_player_id": ["GSIS_A"] * 3,
            "passer_player_name": ["Starter QB"] * 3,
            "play_type": ["pass", "pass", "qb_spike"],
            "qb_dropback": [1, 1, 0],
            "qb_spike": [0, 0, 1],
            "pass": [1, 0, 1],
            "complete_pass": [1, 0, 0],
            "passing_yards": [18.0, 0.0, 0.0],
            "pass_touchdown": [0, 0, 0],
            "interception": [0, 0, 0],
            "sack": [0, 1, 0],
            "yards_gained": [18.0, -6.0, 0.0],
            "fumble_lost": [0, 0, 0],
            "qb_epa": [1.2, -0.4, -0.2],
        }
    )
    player_stats = pl.DataFrame(
        {
            "game_id": [_GAME],
            "week": [1],
            "team": ["DEN"],
            "player_id": ["GSIS_A"],
            "player_display_name": ["Starter QB"],
            "position": ["QB"],
            "attempts": [2],
            "passing_epa": [0.6],
        }
    )
    qb_df = qb_stats.compute_qb_game_stats_from_pbp(pbp)
    official = game_logs.official_weekly_qb_stats(
        player_stats, pl.DataFrame(schema=QB_IDENTITY_SCHEMA)
    )

    # Act
    result = game_logs.override_qb_game_stats_with_official_weekly(qb_df, official)

    # Assert
    row = result.row(0, named=True)
    assert (row["qb_passing_epa"], row["qb_spike_epa"]) == pytest.approx((0.6, -0.2))
    assert row["qb_epa_per_dropback"] == pytest.approx(0.4)


def test_build_team_game_logs_leads_with_the_game_and_drops_rate_parts() -> None:
    # Arrange
    weekly = game_logs.build_weekly_team_stats(
        _team_pbp(), _defense_player_stats(), _schedule(), _official_team_stats()
    )

    # Act
    logs = game_logs.build_team_game_logs(weekly)

    # Assert
    assert logs.columns[:4] == ["game_id", "week", "team", "opponent_team"]
    assert not [column for column in logs.columns if is_rate_part(column)]
    assert {"season", "season_type"}.isdisjoint(logs.columns)
    assert logs.get_column("team").to_list() == ["DEN", "KC"]


def test_build_qb_game_logs_carries_the_opponent_venue_and_result() -> None:
    # Arrange
    weekly = game_logs.build_weekly_team_stats(
        _team_pbp(), _defense_player_stats(), _schedule(), pl.DataFrame()
    )
    qb_df = game_logs.build_qb_game_stats(
        _passer_pbp(), _snap_counts(), _identity(), pl.DataFrame()
    )

    # Act
    logs = game_logs.build_qb_game_logs(qb_df, weekly)

    # Assert
    row = logs.row(0, named=True)
    assert logs.columns[:6] == ["game_id", "week", "team", "opponent_team", "qb_id", "qb_name"]
    assert (row["opponent_team"], row["is_home"], row["point_margin"], row["win_value"]) == (
        "KC",
        True,
        7,
        1.0,
    )
    assert "snap_player_id" not in logs.columns
    assert not [column for column in logs.columns if is_rate_part(column)]


def _qb_game_rows() -> pl.DataFrame:
    """Return one passer's game rows as the loader builds them, official weekly stats included.

    Fixture: a completion, an incompletion, a sack, a scramble, a designed run, and a kneel.
    """
    play = {
        "game_id": _GAME,
        "season_type": "REG",
        "week": 1,
        "posteam": "DEN",
        "defteam": "KC",
        "passer_player_id": "00-0031234",
        "passer_player_name": "J.Doe",
        "rusher_player_id": None,
        "rusher_player_name": None,
        "qb_dropback": 1,
        "pass": 1,
        "rush": 0,
        "qb_scramble": 0,
        "qb_kneel": 0,
        "two_point_attempt": 0,
        "complete_pass": 0,
        "passing_yards": 0.0,
        "rushing_yards": 0.0,
        "yards_gained": 0.0,
        "pass_touchdown": 0,
        "interception": 0,
        "sack": 0,
        "fumble_lost": 0,
        "epa": 0.0,
        "qb_epa": 0.0,
        "cpoe": None,
    }
    runner = {
        "passer_player_id": None,
        "passer_player_name": None,
        "rusher_player_id": "00-0031234",
        "rusher_player_name": "John Doe",
    }
    plays = [
        {**play, "complete_pass": 1, "passing_yards": 12.0, "qb_epa": 0.9, "cpoe": 20.0},
        {**play, "qb_epa": -0.5, "cpoe": -30.0},
        {**play, "sack": 1, "yards_gained": -6.0, "qb_epa": -1.5},
        {**play, **runner, "qb_scramble": 1, "rushing_yards": 9.0, "epa": 0.8},
        {**play, **runner, "qb_dropback": 0, "pass": 0, "rush": 1, "rushing_yards": 4.0},
        {**play, **runner, "qb_dropback": 0, "pass": 0, "qb_kneel": 1, "rushing_yards": -1.0},
    ]
    player_stats = pl.DataFrame(
        {
            "season_type": ["REG"],
            "week": [1],
            "game_id": [_GAME],
            "team": ["DEN"],
            "player_id": ["00-0031234"],
            "position": ["QB"],
            "attempts": [2],
            "completions": [1],
            "passing_yards": [12.0],
            "passing_tds": [0],
            "passing_interceptions": [0],
            "sacks_suffered": [1],
            "sack_yards_lost": [-6.0],
            "passing_epa": [-1.1],
            "passing_cpoe": [-5.0],
            "carries": [3],
            "rushing_yards": [12.0],
            "rushing_epa": [0.6],
        }
    )
    return game_logs.build_qb_game_stats(
        pl.DataFrame(plays), pl.DataFrame(), pl.DataFrame(), player_stats
    )


def _qb_rate_columns(columns: list[str]) -> list[str]:
    """Return the QB game columns the registry calls a rate or a mean."""
    registry = get_registry()
    return [
        column
        for column in columns
        if (resolved := registry.resolve_column(column)) is not None
        and not is_rate_part(column)
        and resolved.base.shape in {"rate", "avg"}
    ]


def test_every_qb_game_rate_is_rebuilt_over_games() -> None:
    # Arrange
    games = _qb_game_rows()

    # Act
    with_parts = set(rate_parts(games.columns))

    # Assert
    rates = _qb_rate_columns(games.columns)
    assert rates
    assert [
        rate for rate in rates if rate not in qb_stats.QB_RATES and rate not in with_parts
    ] == []


def test_build_weekly_team_stats_returns_rows_in_team_week_game_order() -> None:
    # Arrange
    pbp = _team_pbp().reverse()

    # Act
    result = game_logs.build_weekly_team_stats(
        pbp, _defense_player_stats(), _schedule(), _official_team_stats()
    )

    # Assert
    assert result.select(game_logs.TEAM_GAME_ORDER).rows() == [
        ("DEN", 1, _GAME),
        ("KC", 1, _GAME),
    ]


def test_build_qb_game_stats_breaks_a_name_tie_by_passer_id() -> None:
    """Two passers the play-by-play names alike in one game are ordered by their ids."""
    # Arrange
    pbp = pl.concat(
        [
            _passer_pbp().head(1).with_columns(pl.lit("00-0000002").alias("passer_player_id")),
            _passer_pbp().head(1).with_columns(pl.lit("00-0000001").alias("passer_player_id")),
        ]
    ).with_columns(pl.lit("J.Smith").alias("passer_player_name"))
    identity = pl.DataFrame(
        {
            "qb_id": ["00-0000001", "00-0000002"],
            "snap_player_id": [None, None],
            "qb_name": ["J.Smith", "J.Smith"],
            "qb_position": ["QB", "QB"],
        },
        schema=QB_IDENTITY_SCHEMA,
    )

    # Act
    result = game_logs.build_qb_game_stats(pbp, pl.DataFrame(), identity, pl.DataFrame())

    # Assert
    assert result.select("qb_name", "qb_id").rows() == [
        ("J.Smith", "00-0000001"),
        ("J.Smith", "00-0000002"),
    ]
