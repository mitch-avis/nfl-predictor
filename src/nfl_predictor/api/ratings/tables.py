"""The team and quarterback index tables, their metric categories, and the season's progress.

The grouping follows the ``nfl-sos-ratings`` app these pages come from (its ``ui_data.py``): the
ratings lead, the other stats follow in the registry's category order, and the opponent profiles
(``opp_`` for teams, ``qopp_`` and ``opp_`` for quarterbacks) close the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nfl_predictor import constants
from nfl_predictor.api.ratings.columns import metric_table
from nfl_predictor.api.schemas.ratings import CategoryOut
from nfl_predictor.season_stats.metrics import get_registry

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from nfl_predictor.api.schemas.ratings import EntityKind, MetricTableOut
    from nfl_predictor.season_stats.metrics.schema import Entity

TEAM_RATING_COLUMNS = (
    "team_rating",
    "offense_rating",
    "defense_rating",
    "special_teams_rating",
    "sos",
    "SRS",
)
QB_RATING_COLUMNS = ("adj_qb_epa_per_dropback", "qb_faced_pass_defense")
QB_RATING_COMPANION_COLUMNS = ("qb_epa_per_dropback", "qb_dropbacks_total")
"""Shown beside the quarterback rating: the unadjusted rate it adjusts and the sample behind it."""
QB_PER_DROPBACK_RATE_COLUMNS = (
    "qb_epa_per_dropback",
    "qb_pass_yards_per_dropback",
    "qb_td_int_margin_rate",
    "qb_sack_rate",
)
TEAM_IDENTITY = ("team",)
QB_IDENTITY = ("qb_id", "qb_name", "team")
_PER_SNAP_SUFFIXES = ("_per_offensive_snap", "_per_defensive_snap")
_REGISTRY_ENTITY: dict[EntityKind, Entity] = {"teams": "team", "qbs": "qb"}
# Teams play every regular-season week but one, the bye.
_BYE_WEEKS = 1


def existing(columns: Iterable[str], preferred: Sequence[str]) -> list[str]:
    """Return the ``preferred`` columns that ``columns`` holds, in the preferred order."""
    available = set(columns)
    return [column for column in preferred if column in available]


def order_by_category(columns: Sequence[str], entity: Entity) -> list[str]:
    """Order ``columns`` by the registry's category and subcategory display order.

    Columns keep their own order within a subcategory; one the registry cannot resolve goes last.
    """
    registry = get_registry()
    categories = registry.categories(entity)
    category_rank = {category.name: index for index, category in enumerate(categories)}
    subcategory_rank = {
        (category.name, subcategory): index
        for category in categories
        for index, subcategory in enumerate(category.subcategories)
    }

    def sort_key(item: tuple[int, str]) -> tuple[int, int, int]:
        """Return a column's category, subcategory and original position."""
        position, column = item
        resolved = registry.resolve_column(column)
        if resolved is None:
            return (len(category_rank), 0, position)
        return (
            category_rank.get(resolved.category, len(category_rank)),
            subcategory_rank.get((resolved.category, resolved.subcategory or ""), -1),
            position,
        )

    return [column for _, column in sorted(enumerate(columns), key=sort_key)]


def _is_per_snap(column: str) -> bool:
    """Return whether a team column is divided by offensive or defensive snaps."""
    return column.endswith(_PER_SNAP_SUFFIXES)


def team_index(frame: pl.DataFrame) -> MetricTableOut:
    """Return the team table: ratings, per-game stats, per-snap stats, then opponent profiles."""
    identity = existing(frame.columns, TEAM_IDENTITY)
    ratings = existing(frame.columns, TEAM_RATING_COLUMNS)
    opponents = [
        column
        for column in frame.columns
        if column.startswith("opp_") and not column.startswith("opp_qb_")
    ]
    per_snap = [
        column for column in frame.columns if _is_per_snap(column) and not column.startswith("opp_")
    ]
    placed = {*identity, *ratings, *opponents, *per_snap}
    per_game = [column for column in frame.columns if column not in placed]
    return metric_table(
        frame,
        {
            "identity": identity,
            "ratings": ratings,
            "per_game_rates": order_by_category(per_game, "team"),
            "per_snap_rates": order_by_category(per_snap, "team"),
            "opponent_context": opponents,
        },
        references={"rating_companions": []},
    )


def qb_index(frame: pl.DataFrame) -> MetricTableOut:
    """Return the quarterback table: ratings, totals, per-game and per-dropback stats, opponents.

    ``rating_companions`` names the unadjusted rate and the dropbacks behind the rating, which stay
    in their own groups.
    """
    identity = existing(frame.columns, QB_IDENTITY)
    ratings = existing(frame.columns, QB_RATING_COLUMNS)
    opponents = [column for column in frame.columns if column.startswith(("opp_", "qopp_"))]
    per_game = [column for column in frame.columns if column.endswith("_per_game")]
    per_dropback = [
        column
        for column in frame.columns
        if not column.startswith(("opp_", "qopp_"))
        and column not in ratings
        and (column in QB_PER_DROPBACK_RATE_COLUMNS or column.endswith("_per_dropback"))
    ]
    placed = {*identity, *ratings, *opponents, *per_game, *per_dropback}
    totals = [column for column in frame.columns if column not in placed]
    return metric_table(
        frame,
        {
            "identity": identity,
            "ratings": ratings,
            "raw_totals": order_by_category(totals, "qb"),
            "per_game_rates": order_by_category(per_game, "qb"),
            "per_dropback_rates": order_by_category(per_dropback, "qb"),
            "opponent_context": opponents,
        },
        references={
            "rating_companions": existing(frame.columns, QB_RATING_COMPANION_COLUMNS),
        },
    )


def index_table(kind: EntityKind, frame: pl.DataFrame) -> MetricTableOut:
    """Return the team or quarterback index table of one season's ``combined`` table."""
    return team_index(frame) if kind == "teams" else qb_index(frame)


def categories(kind: EntityKind) -> list[CategoryOut]:
    """Return the entity's metric categories in the registry's display order."""
    return [
        CategoryOut(
            name=category.name,
            description=category.description,
            subcategories=list(category.subcategories),
        )
        for category in get_registry().categories(_REGISTRY_ENTITY[kind])
    ]


def games_per_team(season: int) -> int:
    """Return how many regular-season games each team plays: 16 before 2021, 17 since."""
    return constants.get_regular_season_weeks(season) - _BYE_WEEKS


def season_progress(
    team_frame: pl.DataFrame, season: int, current_season: int
) -> tuple[bool, int | None]:
    """Return whether ``season`` is under way, and the most games any team has played.

    Only the current season can be in progress, and only while a team has games left; a completed
    season with a cancelled game stays complete. Without ``games_played`` the current season
    counts as in progress.
    """
    if "games_played" not in team_frame.columns or team_frame.is_empty():
        return season == current_season, None
    games = team_frame.get_column("games_played").cast(pl.Int64)
    most = games.max()
    fewest = games.min()
    most_played = most if isinstance(most, int) else None
    in_progress = season == current_season and (
        not isinstance(fewest, int) or fewest < games_per_team(season)
    )
    return in_progress, most_played
