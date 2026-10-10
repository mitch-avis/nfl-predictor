"""One team's or quarterback's season sections: game logs, histories, rank ranges and pairs.

Each builder takes the rows a route already narrowed to one season (and, for the per-entity
sections, one team or quarterback) and returns them grouped the way the detail page reads them,
as the ``nfl-sos-ratings`` app these pages come from grouped them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor.api.ratings.columns import ColumnOverride, metric_table
from nfl_predictor.api.ratings.tables import QB_RATING_COLUMNS, TEAM_RATING_COLUMNS, existing
from nfl_predictor.ratings.rating_ranges import (
    PAIR_QUANTILES,
    QB_PAIR_COLUMNS,
    QB_RANGE_COLUMNS,
    RANGE_QUANTILES,
    TEAM_PAIR_COLUMNS,
    TEAM_RANGE_COLUMNS,
    TOP_RANKS,
    UNIT_RANGE_COLUMNS,
    PairColumns,
    RangeColumns,
    quantile_suffix,
)
from nfl_predictor.season_stats.metrics import get_registry

if TYPE_CHECKING:
    from nfl_predictor.api.schemas.ratings import EntityKind, MetricTableOut

MIN_GAMES_FOR_RANK_HISTORY = 3
"""The weekly rank chart starts once every team has played this many games.

A game bootstrap of fewer games can only repeat or drop a team's games, never change their
results, so its ranges would look more certain than they are.
"""
TEAM_LOG_IDENTITY = ("game_id", "week", "team", "opponent_team")
QB_LOG_IDENTITY = (*TEAM_LOG_IDENTITY, "qb_id", "qb_name")
TEAM_LOG_RESULTS = ("points_for", "points_allowed", "point_margin", "win_value", "turnover_margin")
QB_LOG_RESULTS = (
    "points_for",
    "points_allowed",
    "qb_fourth_quarter_comeback",
    "qb_game_winning_drive",
)
QB_PER_DROPBACK_RATES = (
    "qb_epa_per_dropback",
    "qb_pass_yards_per_dropback",
    "qb_td_int_margin_rate",
    "qb_sack_rate",
)
OPPONENT_RATINGS = (
    "team_rating",
    "offense_rating",
    "defense_rating",
    "special_teams_rating",
    "SRS",
)
"""The season ratings attached to every game for its opponent, as ``opp_<rating>``."""
OPPONENT_RANK = "opp_team_rank"
"""The opponent's place in the league by season Team Rating, 1 the best (ties share it)."""
OPPONENT_COLUMNS = (
    "opp_team_rating",
    OPPONENT_RANK,
    "opp_offense_rating",
    "opp_defense_rating",
    "opp_special_teams_rating",
    "opp_SRS",
)
_LOG_EXCLUDED = ("season", "season_type")
_OPPONENT_NOTE = (
    "Shown for this game's opponent, rated over its whole season: it describes the opponent, "
    "not this game."
)
TEAM_HISTORY = (("week", "team"), ("games_played",), TEAM_RATING_COLUMNS)
QB_HISTORY = (("week", "qb_id"), ("qb_games_played", "qb_dropbacks"), QB_RATING_COLUMNS)
_RANGES: dict[EntityKind, tuple[tuple[str, ...], RangeColumns]] = {
    "teams": (("team",), TEAM_RANGE_COLUMNS),
    "qbs": (("qb_id", "qb_name", "team"), QB_RANGE_COLUMNS),
}
_PAIRS: dict[EntityKind, PairColumns] = {"teams": TEAM_PAIR_COLUMNS, "qbs": QB_PAIR_COLUMNS}


def _opponent_overrides(columns: list[str]) -> dict[str, ColumnOverride]:
    """Return the text of each attached opponent rating: one opponent's season, not an average."""
    metrics = get_registry().metrics
    overrides: dict[str, ColumnOverride] = {}
    for column in columns:
        base = metrics[column.removeprefix("opp_")]
        overrides[column] = {
            "full_name": f"Opponent's {base.full_name}",
            "description": f"{base.description} {_OPPONENT_NOTE}",
        }
    return overrides


def with_opponent_ratings(logs: pl.DataFrame, ratings: pl.DataFrame | None) -> pl.DataFrame:
    """Attach each game's opponent's season ratings as ``opp_<rating>`` columns.

    Args:
        logs: Game-log rows with ``opponent_team``.
        ratings: The season's team ratings table, or ``None`` when the season has none.

    Returns:
        ``logs`` in their own order, with the ratings ``ratings`` holds and the opponent's league
        rank on ``team_rating`` (``opp_team_rank``); unchanged without ``ratings``.

    """
    if ratings is None or "opponent_team" not in logs.columns or "team" not in ratings.columns:
        return logs
    rated = existing(ratings.columns, OPPONENT_RATINGS)
    rank = (
        [
            pl.col("team_rating")
            .rank(method="min", descending=True)
            .cast(pl.Int64)
            .alias(OPPONENT_RANK)
        ]
        if "team_rating" in ratings.columns
        else []
    )
    opponents = ratings.select(
        pl.col("team").alias("opponent_team"),
        *(pl.col(column).alias(f"opp_{column}") for column in rated),
        *rank,
    )
    return logs.join(opponents, on="opponent_team", how="left", maintain_order="left")


def game_log_table(
    kind: EntityKind, logs: pl.DataFrame, ratings: pl.DataFrame | None
) -> MetricTableOut:
    """Return one team's or quarterback's games: identity, results, totals, rates, opponents.

    Args:
        kind: Teams or quarterbacks.
        logs: The entity's game-log rows, in week order.
        ratings: The season's team ratings, attached for each game's opponent when given.

    """
    rows = with_opponent_ratings(logs, ratings)
    identity = existing(rows.columns, TEAM_LOG_IDENTITY if kind == "teams" else QB_LOG_IDENTITY)
    results = existing(rows.columns, TEAM_LOG_RESULTS if kind == "teams" else QB_LOG_RESULTS)
    opponents = existing(rows.columns, OPPONENT_COLUMNS)
    if kind == "teams":
        rate_group = "per_snap_rates"
        rates = [
            column
            for column in rows.columns
            if column.endswith(("_per_offensive_snap", "_per_defensive_snap"))
            and not column.startswith("opp_")
        ]
    else:
        rate_group = "per_dropback_rates"
        rates = [
            column
            for column in rows.columns
            if column in QB_PER_DROPBACK_RATES or column.endswith("_per_dropback")
        ]
    placed = {*identity, *results, *rates, *opponents, *_LOG_EXCLUDED}
    totals = [column for column in rows.columns if column not in placed]
    return metric_table(
        rows,
        {
            "identity": identity,
            "results": results,
            "raw_totals": totals,
            rate_group: rates,
            "opponent_ratings": opponents,
        },
        overrides=_opponent_overrides(opponents),
    )


def rating_history_table(kind: EntityKind, rows: pl.DataFrame) -> MetricTableOut:
    """Return one team's or quarterback's rating after each week: the sample, then the ratings."""
    identity, sample, ratings = TEAM_HISTORY if kind == "teams" else QB_HISTORY
    return metric_table(
        rows,
        {
            "identity": existing(rows.columns, identity),
            "sample": existing(rows.columns, sample),
            "ratings": existing(rows.columns, ratings),
        },
    )


def rank_range_table(kind: EntityKind, frame: pl.DataFrame) -> MetricTableOut:
    """Return every team's or qualifying quarterback's rank range, by published rank.

    Each row carries the published rank, the rating and rank percentiles over the season's game
    bootstrap, the top-5 and top-10 chances, the share of redraws without the entity, the chance
    of each rank, and for teams each unit's rank range.
    """
    identity, columns = _RANGES[kind]
    suffixes = [quantile_suffix(level) for level in RANGE_QUANTILES]
    groups = {
        "identity": existing(frame.columns, identity),
        "published": existing(frame.columns, (columns.rank,)),
        "rating_range": existing(frame.columns, [f"{columns.rating}{s}" for s in suffixes]),
        "rank_range": existing(frame.columns, [f"{columns.rank}{s}" for s in suffixes]),
        "rank_chances": existing(
            frame.columns,
            (
                *(f"{columns.rank}_top{top}_probability" for top in TOP_RANKS),
                f"{columns.rank}_missing_share",
                f"{columns.rank}_probabilities",
            ),
        ),
    }
    for unit in UNIT_RANGE_COLUMNS:
        unit_columns = existing(
            frame.columns,
            (
                unit.rank,
                *(f"{unit.rating}{suffix}" for suffix in suffixes),
                *(f"{unit.rank}{suffix}" for suffix in suffixes),
            ),
        )
        if unit_columns:
            groups[f"{unit.rank.removesuffix('_rank')}_range"] = unit_columns
    ordered = frame.sort(columns.rank) if columns.rank in frame.columns else frame
    return metric_table(ordered, groups)


def first_rank_history_week(history: pl.DataFrame | None) -> int | None:
    """Return the first week in which every team has played ``MIN_GAMES_FOR_RANK_HISTORY`` games.

    Args:
        history: The season's weekly team rating history (``week``, ``games_played``), or
            ``None`` when the season has none.

    Returns:
        ``None`` without a history (every week is then shown), and the week after the last one
        when no week qualifies yet.

    """
    if history is None:
        return None
    weeks = history.group_by("week").agg(pl.col("games_played").min()).sort("week")
    first = (
        weeks.filter(pl.col("games_played") >= MIN_GAMES_FOR_RANK_HISTORY).get_column("week").min()
    )
    if isinstance(first, int):
        return first
    last = weeks.get_column("week").max()
    return (last if isinstance(last, int) else 0) + 1


def rank_history_table(
    kind: EntityKind, rows: pl.DataFrame, *, first_week: int | None
) -> MetricTableOut:
    """Return one team's or quarterback's rank range after each week, from ``first_week`` on."""
    _, columns = _RANGES[kind]
    kept = rows if first_week is None else rows.filter(pl.col("week") >= first_week)
    return metric_table(
        kept,
        {
            "identity": existing(kept.columns, ("week", columns.id)),
            "rank_range": existing(
                kept.columns,
                (
                    columns.rank,
                    *(f"{columns.rank}{quantile_suffix(level)}" for level in RANGE_QUANTILES),
                ),
            ),
            "rank_chances": existing(
                kept.columns,
                (
                    *(f"{columns.rank}_top{top}_probability" for top in TOP_RANKS),
                    f"{columns.rank}_missing_share",
                ),
            ),
        },
    )


def pair_table(kind: EntityKind, rows: pl.DataFrame) -> MetricTableOut:
    """Return one team's or quarterback's head-to-head chances against every other one.

    Each row is how often the entity was rated above the other across the season's redraws (among
    the redraws with both), the share of redraws with both, and the rating gap's percentiles.
    """
    columns = _PAIRS[kind]
    ordered = rows.sort(columns.other) if columns.other in rows.columns else rows
    return metric_table(
        ordered,
        {
            "identity": existing(rows.columns, (columns.id, columns.other)),
            "chance": existing(rows.columns, (columns.above, columns.share)),
            "rating_gap": existing(
                rows.columns,
                [f"{columns.gap}{quantile_suffix(level)}" for level in PAIR_QUANTILES],
            ),
        },
    )
