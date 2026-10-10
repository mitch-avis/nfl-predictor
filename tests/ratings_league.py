"""A small synthetic league in the shapes this repository's loaders return, for the ratings ETL.

Four teams play a double round robin (six weeks) in each of three seasons from 1999, the first
play-by-play season, plus one postseason game in the last season. Every team-game has a starter's
dropbacks (enough attempts to qualify him for the quarterback rankings), runs and a punt with
seeded, uneven EPA, so a summation in another order would show in the last bits. AAA's starter
hands over to a backup from week 4 of the last season, BBB's receiver throws in its first two
weeks, and CCC's convert throws in weeks 2 and 5 of the middle season. The players file lists
every starter and the backup at quarterback; the weekly rosters list the starters each week, the
backup from week 4 of the last season, the receiver at wide receiver through week 4 of the last
season and at quarterback from week 5, and the convert at wide receiver through the middle
season and at quarterback through the last. Game rows carry the starters' names, which the
identity file maps to their ids, as the ETL's game rows carry ``away_qb`` / ``home_qb``.
``with_season_stats_columns`` shapes the league as the ETL holds it for the season statistics,
and ``with_season_stats_play_columns`` does the same for any plays.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from nfl_predictor.season_stats.sources import SEASON_STATS_PBP_COLUMNS
from nfl_predictor.utils.polars import loaders, nflverse_sources

TEAMS = ("AAA", "BBB", "CCC", "DDD")
SEASONS = (1999, 2000, 2001)
WEEKS = 6
POSTSEASON_WEEK = 18
BACKUP_FROM_WEEK = 4
BACKUP_ID = "00-0000099"
BACKUP_NAME = "Backup Passer"
# BBB's receiver throws once in weeks 1 and 2 of the last season; no players file row places
# him, and the weekly roster lists him at wide receiver through week 4 and at quarterback from
# week 5. Only the season's identity, which looks ahead, knows him as a quarterback in those
# weeks, and only a pre-week identity that read week 5's own roster would at week 5.
TRICK_ID = "00-0000088"
TRICK_ROSTER_WEEK = 5
_TRICK_WEEKS = (1, 2)
# CCC's convert throws once in weeks 2 and 5 of the middle season, whose weekly roster lists him
# at wide receiver every week; the next season's roster lists him at quarterback. Only an
# identity that reads a later season's roster would count his passes as a quarterback's.
CONVERT_ID = "00-0000077"
CONVERT_NAME = "Convert Passer"
CONVERT_SEASON = SEASONS[1]
_CONVERT_WEEKS = (2, 5)
_DROPBACKS = 16

_PLAY_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "season": pl.Int32(),
    "season_type": pl.String(),
    "week": pl.Int32(),
    "posteam": pl.String(),
    "defteam": pl.String(),
    "play_type": pl.String(),
    "qb_dropback": pl.Float64(),
    "qb_scramble": pl.Float64(),
    "rush": pl.Float64(),
    "pass": pl.Float64(),
    "qb_kneel": pl.Float64(),
    "qb_spike": pl.Float64(),
    "special": pl.Float64(),
    "sack": pl.Float64(),
    "complete_pass": pl.Float64(),
    "yards_gained": pl.Float64(),
    "epa": pl.Float64(),
    "qb_epa": pl.Float64(),
    "passer_player_id": pl.String(),
    "passer_player_name": pl.String(),
}


def starter_id(team: str) -> str:
    """Return the GSIS id of ``team``'s starter."""
    return f"00-QB-{team}"


def starter_name(team: str) -> str:
    """Return ``team``'s starter's full name, as the identity file and game rows write it."""
    return f"{team.title()} Starter"


def _passer(team: str, season: int, week: int) -> tuple[str, str]:
    """Return the id and play-by-play name of the passer of one team-game."""
    if team == "AAA" and season == SEASONS[-1] and week >= BACKUP_FROM_WEEK:
        return BACKUP_ID, "B.Passer"
    return starter_id(team), f"{team[0]}.Starter"


def _slate(season: int) -> list[tuple[int, str, str]]:
    """Return ``(week, away, home)`` for a double round robin, the second half reversed."""
    rounds = [
        [("AAA", "BBB"), ("CCC", "DDD")],
        [("AAA", "CCC"), ("BBB", "DDD")],
        [("AAA", "DDD"), ("BBB", "CCC")],
    ]
    shift = season % 3
    rounds = rounds[shift:] + rounds[:shift]
    games = [(index + 1, away, home) for index, pairs in enumerate(rounds) for away, home in pairs]
    return games + [(week + 3, home, away) for week, away, home in games]


@dataclass(frozen=True)
class League:
    """Every input of the ratings build for the synthetic league."""

    plays: pl.DataFrame
    schedule: pl.DataFrame
    player_stats: pl.DataFrame
    snap_counts: pl.DataFrame
    players: pl.DataFrame
    weekly_rosters: pl.DataFrame
    starter_identity: pl.DataFrame
    games: pl.DataFrame


def _team_plays(
    rng: np.random.Generator, game: tuple[str, int, int, str, str, str]
) -> list[dict[str, object]]:
    """Return one team's plays in one game: sixteen dropbacks (one a sack), four runs, a punt.

    Fifteen pass attempts a game qualify a starter for the quarterback rankings (14 a game).
    """
    game_id, season, week, team, opponent, season_type = game
    passer_id, passer_name = _passer(team, season, week)
    base = {
        "game_id": game_id,
        "season": season,
        "season_type": season_type,
        "week": week,
        "posteam": team,
        "defteam": opponent,
        "qb_scramble": 0.0,
        "qb_kneel": 0.0,
        "qb_spike": 0.0,
        "sack": 0.0,
        "complete_pass": 0.0,
        "yards_gained": 0.0,
    }
    strength = 0.04 * (sum(map(ord, team)) % 5 - 2)
    plays: list[dict[str, object]] = []
    for index in range(_DROPBACKS):
        epa = float(rng.normal(strength, 1.1))
        plays.append(
            base
            | {
                "play_type": "pass",
                "qb_dropback": 1.0,
                "rush": 0.0,
                "pass": 0.0 if index == _DROPBACKS - 1 else 1.0,
                "sack": 1.0 if index == _DROPBACKS - 1 else 0.0,
                "complete_pass": 1.0 if index % 2 == 0 else 0.0,
                "yards_gained": float(rng.integers(-5, 25)),
                "special": 0.0,
                "epa": epa,
                "qb_epa": epa,
                "passer_player_id": passer_id,
                "passer_player_name": passer_name,
            }
        )
    for _ in range(4):
        epa = float(rng.normal(strength / 2, 0.8))
        plays.append(
            base
            | {
                "play_type": "run",
                "qb_dropback": 0.0,
                "rush": 1.0,
                "pass": 0.0,
                "special": 0.0,
                "epa": epa,
                "qb_epa": epa,
                "passer_player_id": None,
                "passer_player_name": None,
            }
        )
    if team == "BBB" and season == SEASONS[-1] and week in _TRICK_WEEKS:
        epa = float(rng.normal(0.5, 1.0))
        plays.append(
            base
            | {
                "play_type": "pass",
                "qb_dropback": 1.0,
                "rush": 0.0,
                "pass": 1.0,
                "special": 0.0,
                "epa": epa,
                "qb_epa": epa,
                "passer_player_id": TRICK_ID,
                "passer_player_name": "T.Receiver",
            }
        )
    if team == "CCC" and season == CONVERT_SEASON and week in _CONVERT_WEEKS:
        # A fixed EPA, so the seeded draws of every other play stay as they were.
        plays.append(
            base
            | {
                "play_type": "pass",
                "qb_dropback": 1.0,
                "rush": 0.0,
                "pass": 1.0,
                "special": 0.0,
                "epa": 0.4,
                "qb_epa": 0.4,
                "passer_player_id": CONVERT_ID,
                "passer_player_name": "C.Convert",
            }
        )
    punt = float(rng.normal(0.0, 0.5))
    plays.append(
        base
        | {
            "play_type": "punt",
            "qb_dropback": 0.0,
            "rush": 0.0,
            "pass": 0.0,
            "special": 1.0,
            "epa": punt,
            "qb_epa": punt,
            "passer_player_id": None,
            "passer_player_name": None,
        }
    )
    return plays


def rewritten_plays(plays: pl.DataFrame, later: pl.Expr) -> pl.DataFrame:
    """Return ``plays`` with every value the ratings read rewritten where ``later`` holds.

    Every later play's EPA changes; every other later dropback goes to another passer of the
    same team (a new id and name); a fifth
    of the later plays have their dropback, pass, run and special-teams flags flipped; some
    later runs become kneel-downs and some incomplete passes spikes; and the other later runs
    become plays a penalty wiped out. A feature that reads any of them from a later play moves,
    and every season still has the rows each fit needs.
    """
    row = pl.int_range(pl.len())
    flip = later & (row % 5 == 0)

    def flipped(column: str) -> pl.Expr:
        return pl.when(flip).then(1.0 - pl.col(column)).otherwise(pl.col(column)).alias(column)

    is_run = pl.col("play_type") == "run"
    incomplete = (pl.col("play_type") == "pass") & (pl.col("complete_pass") == 0)
    renamed = later & (row % 2 == 0)
    kneel = later & is_run & (row % 4 == 0)
    return plays.with_columns(
        pl.when(later).then(pl.col("epa") * -3.0 + 1.25).otherwise(pl.col("epa")).alias("epa"),
        pl.when(later)
        .then(pl.col("qb_epa") * 2.0 - 0.75)
        .otherwise(pl.col("qb_epa"))
        .alias("qb_epa"),
        pl.when(renamed & pl.col("passer_player_id").is_not_null())
        .then("00-RW-" + pl.col("posteam"))
        .otherwise(pl.col("passer_player_id"))
        .alias("passer_player_id"),
        pl.when(renamed & pl.col("passer_player_name").is_not_null())
        .then("R." + pl.col("posteam"))
        .otherwise(pl.col("passer_player_name"))
        .alias("passer_player_name"),
        flipped("qb_dropback"),
        flipped("pass"),
        flipped("rush"),
        flipped("special"),
        pl.when(kneel).then(1.0).otherwise(pl.col("qb_kneel")).alias("qb_kneel"),
        pl.when(later & incomplete & (row % 3 == 0))
        .then(1.0)
        .otherwise(pl.col("qb_spike"))
        .alias("qb_spike"),
        pl.when(later & is_run & ~(row % 4 == 0))
        .then(pl.lit("no_play"))
        .otherwise(pl.col("play_type"))
        .alias("play_type"),
    )


def with_player_sources(data: League) -> League:
    """Return the league with weekly player stats and snap counts for every passer-game.

    Each regular-season passer-game gets a player-stats row (its pass attempts and passing
    EPA) and a snap-count row (60 offensive snaps), linked to the players file through a
    snap-count id, and AAA's backup gets 5 snaps in every regular-season AAA game, so the games
    he does not pass in become quarterback-games without a dropback. Both sources are
    regular-season only, as the loaders keep them.
    """
    regular = data.plays.filter(pl.col("season_type") == "REG")
    passer_games = (
        regular.filter(pl.col("passer_player_id").is_not_null())
        .group_by("game_id", "season", "week", "posteam", "passer_player_id", maintain_order=True)
        .agg(
            pl.col("pass").sum().cast(pl.Int64).alias("attempts"),
            pl.col("qb_epa").sum().alias("passing_epa"),
            pl.col("passer_player_name").first(),
        )
    )
    player_stats = passer_games.select(
        pl.col("passer_player_id").alias("player_id"),
        pl.lit("QB").alias("position"),
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.lit("REG").alias("season_type"),
        "game_id",
        pl.col("posteam").alias("team"),
        "attempts",
        "passing_epa",
    )
    snap_counts = passer_games.select(
        "game_id",
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("posteam").alias("team"),
        pl.col("passer_player_name").alias("player"),
        ("pfr-" + pl.col("passer_player_id")).alias("pfr_player_id"),
        pl.lit("QB").alias("position"),
        pl.lit(60.0).alias("offense_snaps"),
    )
    backup_snaps = (
        regular.filter(pl.col("posteam") == "AAA")
        .select("game_id", "season", "week")
        .unique(maintain_order=True)
        .select(
            "game_id",
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.lit("AAA").alias("team"),
            pl.lit(BACKUP_NAME).alias("player"),
            pl.lit(f"pfr-{BACKUP_ID}").alias("pfr_player_id"),
            pl.lit("QB").alias("position"),
            pl.lit(5.0).alias("offense_snaps"),
        )
    )
    players = data.players.with_columns(("pfr-" + pl.col("gsis_id")).alias("pfr_id"))
    return replace(
        data,
        player_stats=player_stats,
        snap_counts=pl.concat([snap_counts, backup_snaps]),
        players=players,
    )


def plays_for(schedule: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Return play-by-play for every game of ``schedule``, in its row order.

    ``schedule`` needs ``game_id``, ``season``, ``week``, ``game_type``, ``away_abbr`` and
    ``home_abbr``; each team's passer is its starter (``starter_id``).
    """
    rng = np.random.default_rng(seed)
    plays: list[dict[str, object]] = []
    for game in schedule.iter_rows(named=True):
        season_type = "REG" if game["game_type"] == "REG" else "POST"
        for team, opponent in (
            (game["away_abbr"], game["home_abbr"]),
            (game["home_abbr"], game["away_abbr"]),
        ):
            plays += _team_plays(
                rng, (game["game_id"], game["season"], game["week"], team, opponent, season_type)
            )
    return pl.DataFrame(plays, schema=_PLAY_SCHEMA)


def league(seed: int = 0) -> League:
    """Return the synthetic league; ``seed`` changes every EPA and score."""
    rng = np.random.default_rng(seed)
    schedule: list[dict[str, object]] = []
    for season in SEASONS:
        slate = [(week, away, home, "REG") for week, away, home in _slate(season)]
        if season == SEASONS[-1]:
            slate.append((POSTSEASON_WEEK, "AAA", "BBB", "WC"))
        for week, away, home, game_type in slate:
            game_id = f"{season}_{week:02d}_{away}_{home}"
            schedule.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "game_type": game_type,
                    "week": week,
                    "away_abbr": away,
                    "home_abbr": home,
                    "away_score": int(rng.integers(3, 35)),
                    "home_score": int(rng.integers(3, 35)),
                    "away_qb": starter_name(away),
                    "home_qb": starter_name(home),
                }
            )
    schedule_frame = pl.DataFrame(schedule)
    passers = [(starter_id(team), starter_name(team)) for team in TEAMS]
    passers.append((BACKUP_ID, BACKUP_NAME))
    rosters = [
        {
            "season": season,
            "week": week,
            "game_type": "REG",
            "team": team,
            "position": "QB",
            "full_name": starter_name(team),
            "gsis_id": starter_id(team),
            "pfr_id": None,
        }
        for season in SEASONS
        for week in range(1, WEEKS + 1)
        for team in TEAMS
    ]
    rosters += [
        {
            "season": SEASONS[-1],
            "week": week,
            "game_type": "REG",
            "team": "AAA",
            "position": "QB",
            "full_name": BACKUP_NAME,
            "gsis_id": BACKUP_ID,
            "pfr_id": None,
        }
        for week in range(BACKUP_FROM_WEEK, WEEKS + 1)
    ]
    rosters += [
        {
            "season": SEASONS[-1],
            "week": week,
            "game_type": "REG",
            "team": "BBB",
            "position": "QB" if week >= TRICK_ROSTER_WEEK else "WR",
            "full_name": "Trick Receiver",
            "gsis_id": TRICK_ID,
            "pfr_id": None,
        }
        for week in range(1, WEEKS + 1)
    ]
    rosters += [
        {
            "season": season,
            "week": week,
            "game_type": "REG",
            "team": "CCC",
            "position": "WR" if season == CONVERT_SEASON else "QB",
            "full_name": CONVERT_NAME,
            "gsis_id": CONVERT_ID,
            "pfr_id": None,
        }
        for season in (CONVERT_SEASON, SEASONS[-1])
        for week in range(1, WEEKS + 1)
    ]
    return League(
        plays=plays_for(schedule_frame, seed),
        schedule=schedule_frame.drop("away_qb", "home_qb"),
        player_stats=pl.DataFrame(
            schema={
                "player_id": pl.String(),
                "position": pl.String(),
                "season": pl.Int64(),
                "week": pl.Int64(),
                "season_type": pl.String(),
                "game_id": pl.String(),
                "team": pl.String(),
                "attempts": pl.Int64(),
                "passing_epa": pl.Float64(),
            }
        ),
        snap_counts=pl.DataFrame(
            schema={
                "game_id": pl.String(),
                "season": pl.Int64(),
                "week": pl.Int64(),
                "team": pl.String(),
                "player": pl.String(),
                "pfr_player_id": pl.String(),
                "position": pl.String(),
                "offense_snaps": pl.Float64(),
            }
        ),
        players=pl.DataFrame(
            {
                "gsis_id": [passer[0] for passer in passers],
                "display_name": [passer[1] for passer in passers],
                "position": ["QB"] * len(passers),
                "pfr_id": [None] * len(passers),
                "espn_id": [None] * len(passers),
            },
            schema=dict.fromkeys(
                ("gsis_id", "display_name", "position", "pfr_id", "espn_id"), pl.String()
            ),
        ),
        weekly_rosters=pl.DataFrame(rosters),
        starter_identity=pl.DataFrame(
            {"qb_name": [p[1] for p in passers], "qb_id": [p[0] for p in passers]},
            schema={"qb_name": pl.String(), "qb_id": pl.String()},
        ),
        games=schedule_frame.select(
            "game_id", "season", "week", "game_type", "away_abbr", "home_abbr", "away_qb", "home_qb"
        ),
    )


def with_season_stats_play_columns(plays: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Return the regular-season plays with every column the season statistics read.

    The plays keep the regular season only, as the play-by-play loader does, and gain each column
    of ``SEASON_STATS_PBP_COLUMNS`` they lack: null, with the play-by-play cache's type, apart
    from a seeded win probability (``wp``) that spreads the plays over the win-probability bins.
    """
    rng = np.random.default_rng(seed)
    regular = plays.filter(pl.col("season_type") == "REG")
    missing = [column for column in SEASON_STATS_PBP_COLUMNS if column not in regular.columns]
    return regular.with_columns(
        pl.lit(None, dtype=loaders._PBP_COLUMN_DTYPES[column]).alias(column) for column in missing
    ).with_columns(pl.Series("wp", rng.uniform(0.02, 0.98, regular.height)))


def with_season_stats_columns(data: League, seed: int = 0) -> League:
    """Return the league as the ETL holds it for the season statistics.

    The plays are ``with_season_stats_play_columns``'. The player stats, snap counts, players
    file and weekly rosters take their loaders' columns (``nflverse_sources``), as the ratings'
    sources hold them.
    """
    return replace(
        data,
        plays=with_season_stats_play_columns(data.plays, seed),
        player_stats=nflverse_sources.conform(
            data.player_stats, nflverse_sources.PLAYER_STATS.schema
        ),
        snap_counts=nflverse_sources.conform(data.snap_counts, nflverse_sources.SNAP_COUNTS.schema),
        players=nflverse_sources.conform(data.players, nflverse_sources._PLAYERS_SCHEMA),
        weekly_rosters=nflverse_sources.conform(
            data.weekly_rosters, nflverse_sources.WEEKLY_ROSTERS.schema
        ),
    )


def with_backup_starting(games: pl.DataFrame) -> pl.DataFrame:
    """Return ``games`` with AAA's backup as its expected starter from the backup's first week."""
    takes_over = (pl.col("season") == SEASONS[-1]) & (pl.col("week") >= BACKUP_FROM_WEEK)
    return games.with_columns(
        pl.when(takes_over & (pl.col(f"{side}_abbr") == "AAA"))
        .then(pl.lit(BACKUP_NAME))
        .otherwise(pl.col(f"{side}_qb"))
        .alias(f"{side}_qb")
        for side in ("away", "home")
    )


__all__ = [
    "BACKUP_FROM_WEEK",
    "BACKUP_ID",
    "BACKUP_NAME",
    "CONVERT_ID",
    "CONVERT_NAME",
    "CONVERT_SEASON",
    "POSTSEASON_WEEK",
    "SEASONS",
    "TEAMS",
    "TRICK_ID",
    "TRICK_ROSTER_WEEK",
    "WEEKS",
    "League",
    "league",
    "plays_for",
    "rewritten_plays",
    "starter_id",
    "starter_name",
    "with_backup_starting",
    "with_player_sources",
    "with_season_stats_columns",
    "with_season_stats_play_columns",
]
