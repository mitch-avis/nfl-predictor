"""Every season statistics table of one season, as ``nfl-sos-ratings`` writes its data files.

The tables of that project's single-season build (``main.run_season`` there) that are not
ratings, built from one season's inputs (:mod:`nfl_predictor.season_stats.sources`) and returned
under the names of its files (``{season}_<name>.parquet`` there), each validated against the
metric registry and in its fixed row order (:mod:`nfl_predictor.season_stats.row_order`):

- ``team_game_logs`` and ``qb_game_logs``: one row per team-game and per passer-game.
- ``team_per_game_stats``: one row per team, counts averaged per game and rates pooled over the
  games, with the season's record.
- ``qb_per_game_stats``: one row per passer, with his primary team's record.
- ``qb_opponent_profiles`` and ``opponent_profiles``: the head-to-head-excluded profiles of the
  defenses a passer faced and of a team's opponents (absent when no opponent game supports one).
- ``combined``: ``team_per_game_stats``, the opponent profile as ``opp_`` columns, and the
  team's ratings.
- ``qb_combined``: ``qb_per_game_stats``, the faced-defense profile, and the passer's rating and
  faced pass defense.
- ``team_wp_bins`` and ``qb_wp_bins``: plays and EPA by win-probability bin.

The ratings come from the caller, as the ratings package builds them (``nfl_predictor.ratings``):
``combined`` takes ``TEAM_RATING_INPUT_COLUMNS`` and ``qb_combined`` ``QB_RATING_INPUT_COLUMNS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.season_stats.game_logs import (
    build_qb_game_logs,
    build_qb_game_stats,
    build_team_game_logs,
    build_weekly_team_stats,
)
from nfl_predictor.season_stats.metrics import get_registry
from nfl_predictor.season_stats.opponent_stats import compute_all_opponent_profiles
from nfl_predictor.season_stats.qb_opponent_stats import compute_qb_opponent_profiles
from nfl_predictor.season_stats.qb_stats import compute_qb_season_stats
from nfl_predictor.season_stats.row_order import data_file_row_order
from nfl_predictor.season_stats.team_stats import compute_all_teams_per_game, compute_win_totals
from nfl_predictor.season_stats.wp_bins import compute_qb_wp_bins, compute_team_wp_bins

if TYPE_CHECKING:
    from nfl_predictor.season_stats.sources import SeasonInputs

# The columns ``combined`` takes from the team ratings, in their published order.
TEAM_RATING_INPUT_COLUMNS = (
    "team",
    "offense_rating",
    "defense_rating",
    "special_teams_rating",
    "team_rating",
    "sos",
    "SRS",
)
# The columns ``qb_combined`` takes from the quarterback ratings, in their published order.
QB_RATING_INPUT_COLUMNS = ("qb_id", "adj_qb_epa_per_dropback", "qb_faced_pass_defense")
_QB_IDENTITY_KEYS = ("qb_id", "qb_name", "team")
_RECORD_COLUMNS = ("team", "wins", "losses", "ties", "win_pct")


def played_schedule(schedule_df: pl.DataFrame) -> pl.DataFrame:
    """Return only the games that have final scores, so in-progress seasons skip future games.

    A schedule without score columns is returned unchanged.
    """
    if not {"home_score", "away_score"} <= set(schedule_df.columns):
        return schedule_df
    return schedule_df.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )


def finalize_table(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    """Return ``frame`` in its file's fixed row order once every column is in the registry.

    Args:
        frame: One season table.
        name: The table's file name after the season, such as ``team_game_logs``.

    Returns:
        ``frame`` sorted by ``row_order.data_file_row_order`` (nulls last)

    Raises:
        ValueError: If a column does not resolve in the metric registry, or the table has no
            row order.

    """
    unknown = get_registry().validate_columns(frame.columns)
    if unknown:
        msg = f"Table {name} contains columns missing from the metric registry: " + ", ".join(
            unknown
        )
        raise ValueError(msg)
    order = data_file_row_order(name, frame.columns)
    return frame.sort(
        [column for column, _ in order],
        descending=[descending for _, descending in order],
        nulls_last=True,
    )


@dataclass(frozen=True, slots=True)
class SeasonGames:
    """One season's game rows, the inputs of every season table.

    Attributes:
        team_games: One row per team-game with every rate's hidden parts
            (``game_logs.build_weekly_team_stats``).
        qb_games: One row per passer-game with every rate's hidden parts
            (``game_logs.build_qb_game_stats``).
        schedule: The season's played regular-season games.

    """

    team_games: pl.DataFrame
    qb_games: pl.DataFrame
    schedule: pl.DataFrame


def build_season_games(inputs: SeasonInputs) -> SeasonGames:
    """Return one season's team-game and passer-game rows and its played schedule."""
    return SeasonGames(
        team_games=build_weekly_team_stats(
            inputs.pbp, inputs.player_stats, inputs.schedule, inputs.official_team_stats
        ),
        qb_games=build_qb_game_stats(
            inputs.pbp, inputs.snap_counts, inputs.qb_identity, inputs.player_stats
        ),
        schedule=played_schedule(inputs.schedule),
    )


@dataclass(frozen=True, slots=True)
class SeasonTables:
    """One season's statistics tables, each in its published row order."""

    team_game_logs: pl.DataFrame
    qb_game_logs: pl.DataFrame
    team_per_game_stats: pl.DataFrame
    qb_per_game_stats: pl.DataFrame
    qb_opponent_profiles: pl.DataFrame | None
    opponent_profiles: pl.DataFrame | None
    combined: pl.DataFrame
    qb_combined: pl.DataFrame
    team_wp_bins: pl.DataFrame
    qb_wp_bins: pl.DataFrame

    def tables(self) -> dict[str, pl.DataFrame]:
        """Return every table this season has, by file name, in the order they are built."""
        named: dict[str, pl.DataFrame | None] = {
            "team_game_logs": self.team_game_logs,
            "qb_game_logs": self.qb_game_logs,
            "team_per_game_stats": self.team_per_game_stats,
            "qb_per_game_stats": self.qb_per_game_stats,
            "qb_opponent_profiles": self.qb_opponent_profiles,
            "opponent_profiles": self.opponent_profiles,
            "combined": self.combined,
            "qb_combined": self.qb_combined,
            "team_wp_bins": self.team_wp_bins,
            "qb_wp_bins": self.qb_wp_bins,
        }
        return {name: frame for name, frame in named.items() if frame is not None}


def _rating_columns(ratings: pl.DataFrame, columns: tuple[str, ...], label: str) -> pl.DataFrame:
    """Return ``columns`` of ``ratings``, or raise naming the ones it lacks."""
    missing = [column for column in columns if column not in ratings.columns]
    if missing:
        msg = f"{label} lack the columns the combined table takes: " + ", ".join(missing)
        raise ValueError(msg)
    return ratings.select(columns)


def _matching_qb_join_keys(left: pl.DataFrame, right: pl.DataFrame) -> list[str]:
    """Return the QB identity keys both frames share."""
    return [key for key in _QB_IDENTITY_KEYS if key in left.columns and key in right.columns]


def build_season_tables(
    games: SeasonGames,
    pbp: pl.DataFrame,
    *,
    team_ratings: pl.DataFrame,
    qb_ratings: pl.DataFrame,
) -> SeasonTables:
    """Return one season's statistics tables.

    Args:
        games: The season's game rows and played schedule (:func:`build_season_games`).
        pbp: The season's prepared play-by-play, binned by win probability.
        team_ratings: One row per team with ``TEAM_RATING_INPUT_COLUMNS``.
        qb_ratings: One row per passer with ``QB_RATING_INPUT_COLUMNS``.

    Returns:
        The season's tables, each validated against the metric registry and in its row order

    Raises:
        ValueError: If a ratings frame lacks a column the combined tables take.

    """
    team_rating_columns = _rating_columns(team_ratings, TEAM_RATING_INPUT_COLUMNS, "Team ratings")
    qb_rating_columns = _rating_columns(qb_ratings, QB_RATING_INPUT_COLUMNS, "QB ratings")
    weekly_df = games.team_games
    qb_df = games.qb_games

    win_totals = compute_win_totals(weekly_df)
    team_combined = compute_all_teams_per_game(weekly_df).join(win_totals, on="team", how="left")
    qb_season_stats = compute_qb_season_stats(qb_df, weekly_df=weekly_df).join(
        win_totals.select(_RECORD_COLUMNS), on="team", how="left"
    )
    qb_opp_profiles, _ = compute_qb_opponent_profiles(weekly_df, qb_df, qb_season_stats)
    opp_profiles, _ = compute_all_opponent_profiles(weekly_df, games.schedule)
    combined = team_combined
    if opp_profiles is not None:
        combined = combined.join(
            opp_profiles.rename({c: f"opp_{c}" for c in opp_profiles.columns if c != "team"}),
            on="team",
            how="left",
        )
    qb_combined = qb_season_stats
    if qb_opp_profiles is not None:
        qb_combined = qb_combined.join(
            qb_opp_profiles, on=_matching_qb_join_keys(qb_combined, qb_opp_profiles), how="left"
        )

    return SeasonTables(
        team_game_logs=finalize_table(build_team_game_logs(weekly_df), "team_game_logs"),
        qb_game_logs=finalize_table(build_qb_game_logs(qb_df, weekly_df), "qb_game_logs"),
        team_per_game_stats=finalize_table(team_combined, "team_per_game_stats"),
        qb_per_game_stats=finalize_table(qb_season_stats, "qb_per_game_stats"),
        qb_opponent_profiles=(
            None
            if qb_opp_profiles is None
            else finalize_table(qb_opp_profiles, "qb_opponent_profiles")
        ),
        opponent_profiles=(
            None if opp_profiles is None else finalize_table(opp_profiles, "opponent_profiles")
        ),
        combined=finalize_table(
            combined.join(team_rating_columns, on="team", how="left"), "combined"
        ),
        qb_combined=finalize_table(
            qb_combined.join(qb_rating_columns, on="qb_id", how="left"), "qb_combined"
        ),
        team_wp_bins=finalize_table(compute_team_wp_bins(pbp), "team_wp_bins"),
        qb_wp_bins=finalize_table(compute_qb_wp_bins(pbp), "qb_wp_bins"),
    )


__all__ = [
    "QB_RATING_INPUT_COLUMNS",
    "TEAM_RATING_INPUT_COLUMNS",
    "SeasonGames",
    "SeasonTables",
    "build_season_games",
    "build_season_tables",
    "finalize_table",
    "played_schedule",
]
