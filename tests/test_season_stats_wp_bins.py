"""Tests for plays and EPA split into win-probability bins."""

from typing import TYPE_CHECKING

import polars as pl
import pytest

from nfl_predictor.season_stats.metrics import get_registry
from nfl_predictor.season_stats.qb_stats import compute_qb_game_volumes_from_pbp
from nfl_predictor.season_stats.team_stats import compute_team_snap_counts_from_pbp
from nfl_predictor.season_stats.team_stats_expanded import compute_expanded_team_game_stats
from nfl_predictor.season_stats.wp_bins import (
    QB_WP_BIN_SCHEMA,
    TEAM_WP_BIN_SCHEMA,
    compute_qb_wp_bins,
    compute_team_wp_bins,
    wp_bin_expr,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_GAME = {"game_id": "2025_01_DEN_KC", "season": 2025, "season_type": "REG", "week": 1}


def _play(**overrides: object) -> dict[str, object]:
    """Return one synthetic DEN-offense play with quiet defaults (not a scrimmage snap)."""
    play: dict[str, object] = {
        **_GAME,
        "posteam": "DEN",
        "defteam": "KC",
        "qb_dropback": 0,
        "rush": 0,
        "qb_kneel": 0,
        "qb_spike": 0,
        "special": 0,
        "epa": 0.0,
        "qb_epa": 0.0,
        "wp": 0.5,
        "passer_player_id": None,
        "passer_player_name": None,
    }
    play.update(overrides)
    return play


def _dropback(passer_id: str | None, name: str, epa: float, wp: float | None) -> dict[str, object]:
    """Return one DEN dropback by the given passer."""
    return _play(
        qb_dropback=1,
        epa=epa,
        qb_epa=epa,
        wp=wp,
        passer_player_id=passer_id,
        passer_player_name=name,
    )


def _pbp(plays: list[dict[str, object]]) -> pl.DataFrame:
    """Build a play-by-play frame with stable types from synthetic plays."""
    return pl.DataFrame(
        plays,
        schema_overrides={
            "wp": pl.Float64,
            "epa": pl.Float64,
            "qb_epa": pl.Float64,
            "passer_player_id": pl.String,
            "passer_player_name": pl.String,
            "posteam": pl.String,
            "defteam": pl.String,
        },
    )


def _mixed_game() -> pl.DataFrame:
    """Return a game mixing every kind of row the team totals filter on."""
    return _pbp(
        [
            _dropback("00-1", "B.Nix", 0.8, 0.62),
            _dropback("00-1", "B.Nix", -0.4, 0.97),
            _play(rush=1, epa=0.3, wp=0.995),
            _play(qb_kneel=1, epa=-0.2, wp=None),
            _play(rush=1, epa=None, wp=0.41),
            _play(special=1, epa=0.5, wp=0.55),
            # A fake punt run: both a scrimmage snap and a special-teams play.
            _play(special=1, rush=1, epa=1.2, wp=0.48),
            _play(posteam="KC", defteam="DEN", rush=1, epa=-0.1, wp=0.38),
            _play(posteam="KC", defteam="DEN", special=1, epa=0.2, wp=0.38),
            # A timeout row: no possession team, never counted.
            _play(posteam=None, defteam=None, epa=None, wp=0.5),
        ]
    )


def test_wp_bin_expr_rounds_down_to_whole_percentage_points() -> None:
    # Arrange
    frame = pl.DataFrame({"wp": [0.004, 0.29, 0.71, 0.5, 0.995, None, float("nan")]})

    # Act
    bins = frame.select(wp_bin_expr()).to_series().to_list()

    # Assert
    assert bins == [0, 29, 29, 50, 0, None, None]


def test_compute_team_wp_bins_counts_each_play_in_its_bin() -> None:
    # Arrange
    pbp = _pbp([_play(rush=1, epa=0.5, wp=0.62), _play(rush=1, epa=0.25, wp=0.9)])

    # Act
    bins = compute_team_wp_bins(pbp)

    # Assert
    assert bins.sort("wp_bin").select("wp_unit", "wp_bin", "wp_bin_plays", "wp_bin_epa").rows() == [
        ("scrimmage", 10, 1, 0.25),
        ("scrimmage", 38, 1, 0.5),
    ]


def test_compute_team_wp_bins_merges_plays_that_share_a_bin() -> None:
    # Arrange
    pbp = _pbp([_play(rush=1, epa=0.5, wp=0.62), _play(rush=1, epa=0.25, wp=0.385)])

    # Act
    bins = compute_team_wp_bins(pbp)

    # Assert
    assert bins.select("team", "opponent_team", "wp_unit", "wp_bin", "wp_bin_plays").rows() == [
        ("DEN", "KC", "scrimmage", 38, 2)
    ]
    assert bins.get_column("wp_bin_epa").to_list() == [pytest.approx(0.75)]


def test_compute_team_wp_bins_keeps_plays_without_wp_in_a_null_bin() -> None:
    # Arrange
    pbp = _pbp([_play(qb_kneel=1, epa=-0.2, wp=None)])

    # Act
    bins = compute_team_wp_bins(pbp)

    # Assert
    assert bins.select("wp_bin", "wp_bin_plays").rows() == [(None, 1)]


def test_team_bins_sum_to_the_team_game_totals_the_ratings_use() -> None:
    # Arrange
    pbp = _mixed_game()
    keys = ["game_id", "team"]
    expanded = compute_expanded_team_game_stats(pbp).select(
        *keys, "offensive_epa", "st_plays", "st_epa"
    )
    snaps = compute_team_snap_counts_from_pbp(pbp).select(*keys, "offensive_snaps")
    totals = expanded.join(snaps, on=keys).sort(keys)

    # Act
    bins = compute_team_wp_bins(pbp)

    # Assert
    summed = (
        bins.group_by(keys)
        .agg(
            pl.col("wp_bin_plays").filter(pl.col("wp_unit") == "scrimmage").sum().alias("plays"),
            pl.col("wp_bin_epa").filter(pl.col("wp_unit") == "scrimmage").sum().alias("epa"),
            pl.col("wp_bin_plays")
            .filter(pl.col("wp_unit") == "special_teams")
            .sum()
            .alias("st_plays"),
            pl.col("wp_bin_epa").filter(pl.col("wp_unit") == "special_teams").sum().alias("st_epa"),
        )
        .sort(keys)
    )
    assert summed.get_column("team").to_list() == totals.get_column("team").to_list()
    assert summed.get_column("plays").to_list() == totals.get_column("offensive_snaps").to_list()
    assert summed.get_column("epa").to_list() == pytest.approx(
        totals.get_column("offensive_epa").to_list()
    )
    assert summed.get_column("st_plays").to_list() == totals.get_column("st_plays").to_list()
    assert summed.get_column("st_epa").to_list() == pytest.approx(
        totals.get_column("st_epa").to_list()
    )


def test_qb_bins_sum_to_the_passer_game_dropbacks() -> None:
    # Arrange
    pbp = _pbp(
        [
            _dropback("00-1", "B.Nix", 0.8, 0.62),
            _dropback("00-1", "B.Nix", -0.4, 0.97),
            _dropback("00-1", "B.Nix", 0.1, None),
            _dropback("00-2", "J.Stidham", 0.3, 0.01),
            _play(rush=1, epa=0.3, wp=0.5),
        ]
    )
    volumes = compute_qb_game_volumes_from_pbp(pbp).select("game_id", "qb_id", "qb_dropbacks")

    # Act
    bins = compute_qb_wp_bins(pbp)

    # Assert
    summed = (
        bins.group_by("game_id", "qb_id")
        .agg(pl.col("qb_wp_bin_dropbacks").sum().alias("qb_dropbacks"))
        .sort("qb_id")
    )
    assert summed.rows() == volumes.sort("qb_id").rows()


def test_qb_bins_sum_play_level_passing_epa() -> None:
    # Arrange
    pbp = _pbp([_dropback("00-1", "B.Nix", 0.8, 0.62), _dropback("00-1", "B.Nix", -0.5, 0.615)])

    # Act
    bins = compute_qb_wp_bins(pbp)

    # Assert
    assert bins.select("qb_id", "wp_bin", "qb_wp_bin_dropbacks").rows() == [("00-1", 38, 2)]
    assert bins.get_column("qb_wp_bin_epa").to_list() == [pytest.approx(0.3)]


def test_qb_bins_group_a_passer_tagged_two_ways_by_id() -> None:
    # Arrange
    pbp = _pbp(
        [
            _dropback("00-1", "T.Pike", 0.2, 0.6),
            _dropback("00-1", "T.Pike (3rd QB)", 0.2, 0.6),
        ]
    )

    # Act
    bins = compute_qb_wp_bins(pbp)

    # Assert
    assert bins.select("qb_id", "qb_wp_bin_dropbacks").rows() == [("00-1", 2)]


def test_qb_bins_leave_out_passers_without_an_id() -> None:
    # Arrange
    pbp = _pbp([_dropback(None, "Unknown", 0.2, 0.6), _dropback("00-1", "B.Nix", 0.1, 0.6)])

    # Act
    bins = compute_qb_wp_bins(pbp)

    # Assert
    assert bins.get_column("qb_id").to_list() == ["00-1"]


@pytest.mark.parametrize(
    ("compute", "schema"),
    [(compute_team_wp_bins, TEAM_WP_BIN_SCHEMA), (compute_qb_wp_bins, QB_WP_BIN_SCHEMA)],
)
def test_bins_are_an_empty_typed_frame_without_win_probability(
    compute: Callable[[pl.DataFrame], pl.DataFrame], schema: pl.Schema
) -> None:
    # Arrange
    pbp = _mixed_game().drop("wp")

    # Act
    bins = compute(pbp)

    # Assert
    assert bins.is_empty()
    assert bins.schema == schema


def test_every_bin_column_resolves_in_the_metric_registry() -> None:
    # Arrange
    columns = [*TEAM_WP_BIN_SCHEMA, *QB_WP_BIN_SCHEMA]

    # Act
    unknown = get_registry().validate_columns(columns)

    # Assert
    assert unknown == []
