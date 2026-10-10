"""Repair player team codes in the games nflverse credits wholly to one team.

In Jacksonville's 2001-2002 home games nflverse names the visiting team for every player
(nflverse-pbp issue 92): in the play-by-play team columns other than ``posteam`` and
``defteam`` (who scored, fumbled, recovered, tackled or drew a penalty) and in the weekly player
stats. These functions find such games and rebuild each player's team from that season's roster,
restricted to the two teams in the game. The method is the one ``nfl-sos-ratings`` uses
(``player_team_repair.py`` there), with two differences:

- The two teams are read from the game id (``2002_10_WAS_JAX``) through
  ``constants.ALIAS_TO_CANONICAL``, so the ``WAS`` written there matches a roster's ``WSH``; a
  team code read from a penalty's play text is normalized the same way.
- A roster frame may span several seasons; each game reads its own season's roster.

The team-stat side of the same quirk (one team-stat row holding both teams' totals) is handled
where the team stats join the schedule (``loaders.attach_team_stats_to_schedule``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants

if TYPE_CHECKING:
    from collections.abc import Sequence

# A game id reads `<season>_<week>_<away>_<home>`.
_GAME_ID_PARTS = 4
# A penalty on a team as a whole names no player; its play text names the team.
_PENALTY_TEAM_PATTERN = r"PENALTY on ([A-Z]{2,3})\b"
_SIDES_SCHEMA = pl.Schema(
    {"game_id": pl.String(), "season": pl.Int64(), "away": pl.String(), "home": pl.String()}
)


def one_team_games(frame: pl.DataFrame, team_columns: Sequence[str]) -> list[str]:
    """Return the games whose player team values, across ``team_columns``, name a single team.

    A game needs at least ``constants.ONE_TEAM_GAME_MIN_VALUES`` values before one team marks
    it as broken. Columns ``frame`` lacks are skipped; without ``game_id`` nothing is found.

    Returns:
        The affected game ids, sorted

    """
    present = [column for column in team_columns if column in frame.columns]
    if not present or "game_id" not in frame.columns:
        return []
    values = (
        frame.select("game_id", *(pl.col(column).cast(pl.String) for column in present))
        .unpivot(index="game_id", value_name="_team_value")
        .drop_nulls("_team_value")
    )
    return (
        values.group_by("game_id", maintain_order=True)
        .agg(pl.col("_team_value").n_unique().alias("teams"), pl.len().alias("values"))
        .filter((pl.col("teams") == 1) & (pl.col("values") >= constants.ONE_TEAM_GAME_MIN_VALUES))
        .get_column("game_id")
        .sort()
        .to_list()
    )


def game_sides(games: Sequence[str]) -> pl.DataFrame:
    """Return each game's season and its canonical visiting and home teams, read from its id.

    Raises:
        ValueError: If a game id is not ``<season>_<week>_<away>_<home>``.

    """
    rows: list[tuple[str, int, str, str]] = []
    for game in games:
        parts = game.split("_")
        if len(parts) != _GAME_ID_PARTS or not parts[0].isdigit():
            msg = f"game id {game!r} is not <season>_<week>_<away>_<home>"
            raise ValueError(msg)
        rows.append(
            (
                game,
                int(parts[0]),
                constants.normalize_team_abbr(parts[2]),
                constants.normalize_team_abbr(parts[3]),
            )
        )
    return pl.DataFrame(rows, schema=_SIDES_SCHEMA, orient="row")


def game_player_teams(games: Sequence[str], rosters: pl.DataFrame) -> pl.DataFrame:
    """Return ``game_id``, ``player_id`` and ``team`` for every rostered player in ``games``.

    ``rosters`` holds ``season``, ``player_id`` and a canonical ``team``, one row per team a player
    was on in a season. A player on the roster of exactly one of the game's two teams that season
    gets that team; a player on both or neither is left out, so his team stays unknown.

    Returns:
        One row per placed player and game, sorted by game and player

    """
    candidates = (
        game_sides(games)
        .join(rosters.select("season", "player_id", "team"), on="season")
        .filter((pl.col("team") == pl.col("away")) | (pl.col("team") == pl.col("home")))
    )
    return (
        candidates.group_by("game_id", "player_id", maintain_order=True)
        .agg(pl.col("team").unique(maintain_order=True))
        .filter(pl.col("team").list.len() == 1)
        .select("game_id", "player_id", pl.col("team").list.first())
        .sort("game_id", "player_id")
    )


def _sides_by_game(games: Sequence[str]) -> pl.DataFrame:
    """Return each game's two teams under private names, keyed for a join on ``game_id``."""
    return game_sides(games).select(
        "game_id", pl.col("away").alias("_away"), pl.col("home").alias("_home")
    )


def _plays_for_one_side(team: pl.Expr) -> pl.Expr:
    """Return ``team`` when it is one of the game's two teams, else null."""
    return pl.when((team == pl.col("_away")) | (team == pl.col("_home"))).then(team)


def repair_pbp_player_teams(
    pbp: pl.DataFrame, games: Sequence[str], player_teams: pl.DataFrame
) -> pl.DataFrame:
    """Return ``pbp`` with every player team column in ``games`` rebuilt from ``player_teams``.

    A column takes the team of the player in its id column
    (``constants.PBP_PLAYER_TEAM_ID_COLUMNS``), or null when the player is unknown; a penalty
    with no player takes the team its play text names, when that team plays in the game. Columns
    with no player behind them (``constants.PBP_TEAM_ONLY_COLUMNS``) become null. Other games, the
    row order and the columns are unchanged.

    Returns:
        The repaired play-by-play frame

    """
    broken = pl.col("game_id").is_in(list(games))
    repaired = pbp.with_row_index("_row").join(_sides_by_game(games), on="game_id", how="left")
    for team_column, id_column in constants.PBP_PLAYER_TEAM_ID_COLUMNS.items():
        if team_column not in pbp.columns or id_column not in pbp.columns:
            continue
        mapped = f"_{team_column}"
        repaired = repaired.join(
            player_teams.select(
                "game_id", pl.col("player_id").alias(id_column), pl.col("team").alias(mapped)
            ),
            on=["game_id", id_column],
            how="left",
        )
        replacement = pl.col(mapped)
        if team_column == "penalty_team" and "desc" in pbp.columns:
            text_team = (
                pl.col("desc")
                .str.extract(_PENALTY_TEAM_PATTERN, 1)
                .replace(constants.ALIAS_TO_CANONICAL)
            )
            replacement = pl.coalesce(replacement, _plays_for_one_side(text_team))
        repaired = repaired.with_columns(
            pl.when(broken).then(replacement).otherwise(pl.col(team_column)).alias(team_column)
        )
    repaired = repaired.with_columns(
        pl.when(broken).then(None).otherwise(pl.col(column)).alias(column)
        for column in constants.PBP_TEAM_ONLY_COLUMNS
        if column in pbp.columns
    )
    return repaired.sort("_row").select(pbp.columns)


def repair_player_stats_teams(
    stats: pl.DataFrame, games: Sequence[str], player_teams: pl.DataFrame
) -> pl.DataFrame:
    """Return weekly player stats with ``team`` and ``opponent_team`` rebuilt in ``games``.

    A player ``player_teams`` places in the game gets his team and the other team as his
    opponent; an unknown player gets null for both. Other games, the row order and the columns
    are unchanged.

    Returns:
        The repaired player stats

    """
    broken = pl.col("game_id").is_in(list(games))
    placed = pl.col("_team")
    repaired = (
        stats.with_row_index("_row")
        .join(
            player_teams.select("game_id", "player_id", pl.col("team").alias("_team")),
            on=["game_id", "player_id"],
            how="left",
        )
        .join(_sides_by_game(games), on="game_id", how="left")
        .with_columns(
            pl.when(broken).then(placed).otherwise(pl.col("team")).alias("team"),
            pl.when(broken)
            .then(
                pl.when(placed == pl.col("_away"))
                .then(pl.col("_home"))
                .when(placed == pl.col("_home"))
                .then(pl.col("_away"))
            )
            .otherwise(pl.col("opponent_team"))
            .alias("opponent_team"),
        )
    )
    return repaired.sort("_row").select(stats.columns)
