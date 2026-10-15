"""Builders for synthetic per-season ratings and statistics tables.

The pages read each season's tables from ``data/ratings/{season}_{kind}.parquet``, with the kinds
and columns of the ``nfl-sos-ratings`` files the ratings and season statistics packages were ported
from. ``data/`` is gitignored, so the route tests write small tables of their own: four teams and
four quarterbacks, a subset of the real columns of each kind, and Washington as ``WSH``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

TEAMS = ("BUF", "KC", "NE", "WSH")
# Best first: team_rating 6.5, 4.0, -2.0, -8.5.
TEAM_RATINGS = {"BUF": 6.5, "KC": 4.0, "NE": -2.0, "WSH": -8.5}
QBS = (
    ("00-0034857", "J.Allen", "BUF"),
    ("00-0033873", "P.Mahomes", "KC"),
    ("00-0039913", "D.Maye", "NE"),
    ("00-0039910", "J.Daniels", "WSH"),
)
# A backup below the qualifier, in the index table only.
BACKUP_QB = ("00-0035000", "M.Backup", "NE")
QUANTILE_SUFFIXES = ("_q025", "_q100", "_q250", "_q500", "_q750", "_q900", "_q975")
PAIR_SUFFIXES = ("_q025", "_q500", "_q975")
UNITS = ("offense", "defense", "special_teams")
# Two rounds of games between the four teams: weeks 1-3, then weeks 4-6 with home and away swapped.
SCHEDULE = (
    (1, "BUF", "KC"),
    (1, "NE", "WSH"),
    (2, "BUF", "NE"),
    (2, "KC", "WSH"),
    (3, "BUF", "WSH"),
    (3, "KC", "NE"),
    (4, "KC", "BUF"),
    (4, "WSH", "NE"),
    (5, "NE", "BUF"),
    (5, "WSH", "KC"),
    (6, "WSH", "BUF"),
    (6, "NE", "KC"),
)
INDEX_KINDS = ("combined", "qb_combined")
ALL_KINDS = (
    "combined",
    "qb_combined",
    "ratings",
    "qb_ratings",
    "team_game_logs",
    "qb_game_logs",
    "ratings_by_week",
    "qb_ratings_by_week",
    "rating_ranges",
    "qb_rating_ranges",
    "rating_pairs",
    "qb_rating_pairs",
)
IN_PROGRESS_KINDS = ("rating_ranges_by_week", "qb_rating_ranges_by_week")


def _rank(ids: Iterable[str]) -> dict[str, int]:
    """Return 1-based ranks in the given (best-first) order."""
    return {entity: index + 1 for index, entity in enumerate(ids)}


def combined(games_played: int) -> pl.DataFrame:
    """Return a team season table: record, per-game stats, opponent profiles and ratings."""
    rows: list[dict[str, object]] = []
    for index, team in enumerate(TEAMS):
        rating = TEAM_RATINGS[team]
        wins = 3 - index
        rows.append(
            {
                "team": team,
                "passing_yards": 250.5 - 10 * index,
                "points_for": 27.25 - 2 * index,
                "points_allowed": 18.5 + 2 * index,
                "point_margin": 8.75 - 4 * index,
                "win_value": 0.75 - 0.25 * index,
                "passing_yards_per_offensive_snap": 3.9 - 0.2 * index,
                "completion_pct": 0.681 - 0.02 * index,
                "epa_per_dropback": 0.21 - 0.1 * index,
                "epa_margin_per_play": 0.12 - 0.06 * index,
                "games_played": games_played,
                "wins": wins,
                "losses": 3 - wins,
                "ties": 0,
                "win_pct": wins / 3,
                "opp_points_for": 21.0 + index,
                "opp_completion_pct": 0.64 + 0.01 * index,
                "offense_rating": rating / 2 + 1.0,
                "defense_rating": rating / 2 - 1.0,
                "special_teams_rating": 0.0,
                "team_rating": rating,
                "sos": 0.5 - 0.25 * index,
                "SRS": rating + 0.5,
            }
        )
    return pl.DataFrame(rows)


def _qb_row(qb_id: str, name: str, team: str, index: int, *, eligible: bool) -> dict[str, object]:
    """Return one quarterback's season row."""
    return {
        "qb_id": qb_id,
        "qb_name": name,
        "qb_games_played": 6 if eligible else 1,
        "qb_epa_per_dropback": 0.25 - 0.1 * index,
        "qb_completion_pct": 0.68 - 0.02 * index,
        "qb_attempts_total": 200 - 10 * index if eligible else 12,
        "qb_dropbacks_total": 220 - 10 * index if eligible else 14,
        "qb_pass_yards_total": 1500 - 100 * index if eligible else 80,
        "team": team,
        "qb_pass_yards_per_game": 250.0 - 15 * index if eligible else 80.0,
        "qb_attempt_qualifier": 84,
        "qb_is_eligible": eligible,
        "wins": 3 - min(index, 3),
        "losses": min(index, 3),
        "ties": 0,
        "win_pct": (3 - min(index, 3)) / 3,
        "qopp_points_allowed": 21.5 + index,
        "qopp_qb_epa_per_dropback": 0.05 + 0.01 * index,
        "adj_qb_epa_per_dropback": 0.22 - 0.1 * index if eligible else None,
        "qb_faced_pass_defense": 0.01 * index if eligible else None,
    }


def qb_combined() -> pl.DataFrame:
    """Return a quarterback season table: three-plus qualifiers and one backup."""
    rows = [_qb_row(*qb, index, eligible=True) for index, qb in enumerate(QBS)]
    rows.append(_qb_row(*BACKUP_QB, 4, eligible=False))
    return pl.DataFrame(rows)


def ratings(games_played: int) -> pl.DataFrame:
    """Return the team ratings table, best team first."""
    frame = combined(games_played)
    return frame.select(
        "team",
        pl.col("games_played").cast(pl.UInt32),
        "offense_rating",
        "defense_rating",
        "special_teams_rating",
        "team_rating",
        "sos",
        "SRS",
    ).sort("team_rating", descending=True)


def qb_ratings() -> pl.DataFrame:
    """Return the qualifying quarterbacks' ratings, best first."""
    rows = [
        {
            "qb_id": qb_id,
            "qb_name": name,
            "team": team,
            "qb_games_played": 6,
            "qb_dropbacks_total": 220 - 10 * index,
            "qb_attempts_total": 200 - 10 * index,
            "qb_epa_per_dropback": 0.25 - 0.1 * index,
            "qb_faced_pass_defense": 0.01 * index,
            "adj_qb_epa_per_dropback": 0.22 - 0.1 * index,
        }
        for index, (qb_id, name, team) in enumerate(QBS)
    ]
    return pl.DataFrame(rows).with_columns(pl.col("qb_games_played").cast(pl.UInt32))


def _game_id(season: int, week: int, away: str, home: str) -> str:
    """Return an nflverse game id."""
    return f"{season}_{week:02d}_{away}_{home}"


def team_game_logs(season: int) -> pl.DataFrame:
    """Return every team-game of the schedule: the home team wins by the rating gap."""
    rows: list[dict[str, object]] = []
    for week, away, home in SCHEDULE:
        margin = round(TEAM_RATINGS[home] - TEAM_RATINGS[away] + 2)
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            sign = 1 if is_home else -1
            points_for = 20 + max(sign * margin, 0)
            points_allowed = 20 + max(-sign * margin, 0)
            rows.append(
                {
                    "game_id": _game_id(season, week, away, home),
                    "week": week,
                    "team": team,
                    "opponent_team": opponent,
                    "passing_yards": 200 + 5 * week,
                    "is_home": is_home,
                    "points_for": points_for,
                    "points_allowed": points_allowed,
                    "games": 1,
                    "point_margin": points_for - points_allowed,
                    "win_value": 1.0 if points_for > points_allowed else 0.0,
                    "passing_yards_per_offensive_snap": 3.5 + 0.1 * week,
                    "completion_pct": 0.6 + 0.01 * week,
                    "turnover_margin": sign,
                    "epa_margin_per_play": 0.01 * sign * margin,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("week").cast(pl.Int32)).sort("team", "week")


def qb_game_logs(season: int) -> pl.DataFrame:
    """Return every starter's passer-games, plus the backup's one game."""
    teams = team_game_logs(season)
    starters = {team: (qb_id, name) for qb_id, name, team in QBS}
    rows = [
        {
            "game_id": game["game_id"],
            "week": game["week"],
            "team": game["team"],
            "opponent_team": game["opponent_team"],
            "qb_id": starters[game["team"]][0],
            "qb_name": starters[game["team"]][1],
            "qb_dropbacks": 36,
            "qb_attempts": 33,
            "qb_pass_yards": game["passing_yards"],
            "qb_epa_per_dropback": 0.1 * game["turnover_margin"],
            "qb_fourth_quarter_comeback": 0,
            "qb_game_winning_drive": 0,
            "qb_completion_pct": game["completion_pct"],
            "is_home": game["is_home"],
            "points_for": game["points_for"],
            "points_allowed": game["points_allowed"],
            "point_margin": game["point_margin"],
            "win_value": game["win_value"],
            "turnover_margin": game["turnover_margin"],
        }
        for game in teams.iter_rows(named=True)
    ]
    backup_id, backup_name, backup_team = BACKUP_QB
    rows.append(
        {
            **rows[0],
            "game_id": _game_id(season, 6, "NE", "KC"),
            "week": 6,
            "team": backup_team,
            "opponent_team": "KC",
            "qb_id": backup_id,
            "qb_name": backup_name,
            "qb_dropbacks": 14,
            "qb_attempts": 12,
        }
    )
    return pl.DataFrame(rows).with_columns(pl.col("week").cast(pl.Int32))


def ratings_by_week(weeks: int = 6) -> pl.DataFrame:
    """Return every team's rating after each week; ratings drift toward the season's."""
    rows = [
        {
            "week": week,
            "team": team,
            "games_played": week,
            "offense_rating": TEAM_RATINGS[team] * week / weeks / 2 + 1.0,
            "defense_rating": TEAM_RATINGS[team] * week / weeks / 2 - 1.0,
            "special_teams_rating": 0.0,
            "team_rating": TEAM_RATINGS[team] * week / weeks,
        }
        for team in TEAMS
        for week in range(1, weeks + 1)
    ]
    return pl.DataFrame(rows)


def qb_ratings_by_week(weeks: int = 6) -> pl.DataFrame:
    """Return every qualifier's rating after each week."""
    rows = [
        {
            "week": week,
            "qb_id": qb_id,
            "qb_games_played": week,
            "qb_dropbacks": 36 * week,
            "adj_qb_epa_per_dropback": (0.22 - 0.1 * index) * week / weeks,
        }
        for index, (qb_id, _name, _team) in enumerate(QBS)
        for week in range(1, weeks + 1)
    ]
    return pl.DataFrame(rows)


def _quantiles(base: str, center: float, spread: float) -> dict[str, float]:
    """Return a rating's quantile columns around ``center``."""
    offsets = (-2.0, -1.3, -0.7, 0.0, 0.7, 1.3, 2.0)
    return {
        f"{base}{suffix}": center + spread * offset
        for suffix, offset in zip(QUANTILE_SUFFIXES, offsets, strict=True)
    }


def _rank_quantiles(base: str, rank: int, count: int) -> dict[str, int]:
    """Return a rank's quantile columns: one place either side for the middle, two for 95%."""
    offsets = (-2, -1, -1, 0, 1, 1, 2)
    return {
        f"{base}{suffix}": min(max(rank + offset, 1), count)
        for suffix, offset in zip(QUANTILE_SUFFIXES, offsets, strict=True)
    }


def _probabilities(rank: int, count: int) -> list[float]:
    """Return P(rank = k) for k = 1..count, peaked at ``rank``."""
    weights = [1.0 / (1 + abs(k - rank)) for k in range(1, count + 1)]
    total = sum(weights)
    return [weight / total for weight in weights]


def rating_ranges() -> pl.DataFrame:
    """Return every team's rank range with its units', by published rank."""
    ranks = _rank(TEAMS)
    count = len(TEAMS)
    rows: list[dict[str, object]] = []
    for team in TEAMS:
        rank = ranks[team]
        row: dict[str, object] = {"team": team, "team_rank": rank}
        row |= _quantiles("team_rating", TEAM_RATINGS[team], 1.5)
        row |= _rank_quantiles("team_rank", rank, count)
        row |= {
            "team_rank_missing_share": 0.0,
            "team_rank_top5_probability": 1.0,
            "team_rank_top10_probability": 1.0,
            "team_rank_probabilities": _probabilities(rank, count),
        }
        for unit in UNITS:
            row[f"{unit}_rank"] = rank
            row |= _quantiles(f"{unit}_rating", TEAM_RATINGS[team] / 2, 1.0)
            row |= _rank_quantiles(f"{unit}_rank", rank, count)
        rows.append(row)
    return pl.DataFrame(rows)


def qb_rating_ranges() -> pl.DataFrame:
    """Return every qualifier's rank range, by published rank."""
    count = len(QBS)
    rows: list[dict[str, object]] = []
    for index, (qb_id, name, team) in enumerate(QBS):
        rank = index + 1
        row: dict[str, object] = {"qb_id": qb_id, "qb_name": name, "team": team, "qb_rank": rank}
        row |= _quantiles("adj_qb_epa_per_dropback", 0.22 - 0.1 * index, 0.05)
        row |= _rank_quantiles("qb_rank", rank, count)
        row |= {
            "qb_rank_missing_share": 0.0,
            "qb_rank_top5_probability": 1.0,
            "qb_rank_top10_probability": 1.0,
            "qb_rank_probabilities": _probabilities(rank, count),
        }
        rows.append(row)
    return pl.DataFrame(rows)


def rating_pairs() -> pl.DataFrame:
    """Return every ordered pair of teams: the better-rated one is above in 80% of redraws."""
    rows: list[dict[str, object]] = []
    for team in TEAMS:
        for other in TEAMS:
            if other == team:
                continue
            gap = TEAM_RATINGS[team] - TEAM_RATINGS[other]
            rows.append(
                {
                    "team": team,
                    "other_team": other,
                    "team_rated_above_probability": 0.8 if gap > 0 else 0.2,
                    "team_rating_gap_q025": gap - 3.0,
                    "team_rating_gap_q500": gap,
                    "team_rating_gap_q975": gap + 3.0,
                    "team_pair_share": 1.0,
                }
            )
    return pl.DataFrame(rows)


def qb_rating_pairs() -> pl.DataFrame:
    """Return every ordered pair of qualifiers; the later pair appears in 90% of redraws."""
    rows: list[dict[str, object]] = []
    for index, (qb_id, _name, _team) in enumerate(QBS):
        for other_index, (other_id, _other_name, _other_team) in enumerate(QBS):
            if other_id == qb_id:
                continue
            gap = 0.1 * (other_index - index)
            rows.append(
                {
                    "qb_id": qb_id,
                    "other_qb_id": other_id,
                    "qb_rated_above_probability": 0.75 if gap > 0 else 0.25,
                    "qb_rating_gap_q025": gap - 0.08,
                    "qb_rating_gap_q500": gap,
                    "qb_rating_gap_q975": gap + 0.08,
                    "qb_pair_share": 0.9,
                }
            )
    return pl.DataFrame(rows)


def rating_ranges_by_week(weeks: int = 6) -> pl.DataFrame:
    """Return every team's rank range after each week of a season in progress."""
    ranks = _rank(TEAMS)
    rows: list[dict[str, object]] = []
    for team in TEAMS:
        for week in range(1, weeks + 1):
            row: dict[str, object] = {"week": week, "team": team, "team_rank": ranks[team]}
            row |= _rank_quantiles("team_rank", ranks[team], len(TEAMS))
            row |= {
                "team_rank_top5_probability": 1.0,
                "team_rank_top10_probability": 1.0,
                "team_rank_missing_share": 0.0,
            }
            rows.append(row)
    return pl.DataFrame(rows)


def qb_rating_ranges_by_week(weeks: int = 6) -> pl.DataFrame:
    """Return every qualifier's rank range after each week of a season in progress."""
    rows: list[dict[str, object]] = []
    for index, (qb_id, _name, _team) in enumerate(QBS):
        for week in range(1, weeks + 1):
            row: dict[str, object] = {"week": week, "qb_id": qb_id, "qb_rank": index + 1}
            row |= _rank_quantiles("qb_rank", index + 1, len(QBS))
            row |= {
                "qb_rank_top5_probability": 1.0,
                "qb_rank_top10_probability": 1.0,
                "qb_rank_missing_share": 0.0,
            }
            rows.append(row)
    return pl.DataFrame(rows)


def season_tables(season: int, games_played: int) -> dict[str, pl.DataFrame]:
    """Return every table of one season by kind, the in-progress kinds included."""
    return {
        "combined": combined(games_played),
        "qb_combined": qb_combined(),
        "ratings": ratings(games_played),
        "qb_ratings": qb_ratings(),
        "team_game_logs": team_game_logs(season),
        "qb_game_logs": qb_game_logs(season),
        "ratings_by_week": ratings_by_week(),
        "qb_ratings_by_week": qb_ratings_by_week(),
        "rating_ranges": rating_ranges(),
        "qb_rating_ranges": qb_rating_ranges(),
        "rating_pairs": rating_pairs(),
        "qb_rating_pairs": qb_rating_pairs(),
        "rating_ranges_by_week": rating_ranges_by_week(),
        "qb_rating_ranges_by_week": qb_rating_ranges_by_week(),
    }


def write_season(
    ratings_dir: Path,
    season: int,
    *,
    games_played: int = 17,
    kinds: Iterable[str] = ALL_KINDS,
) -> Path:
    """Write ``kinds`` of one season to ``ratings_dir`` as ``{season}_{kind}.parquet``.

    Returns:
        ``ratings_dir``, created when missing

    """
    ratings_dir.mkdir(parents=True, exist_ok=True)
    tables = season_tables(season, games_played)
    for kind in kinds:
        tables[kind].write_parquet(ratings_dir / f"{season}_{kind}.parquet")
    return ratings_dir
