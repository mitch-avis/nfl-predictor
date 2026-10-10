"""Assembly of the project registry: categories and the cached builder.

Category order here is the display order for both the index pages and the
detail pages — the project's own schedule-adjusted ratings always lead.
"""

from __future__ import annotations

from functools import lru_cache

from nfl_predictor.season_stats.metrics.pair_metrics import PAIR_METRICS
from nfl_predictor.season_stats.metrics.qb_metrics import QB_METRICS
from nfl_predictor.season_stats.metrics.registry import MetricRegistry
from nfl_predictor.season_stats.metrics.schema import CategoryDef
from nfl_predictor.season_stats.metrics.team_metrics import TEAM_METRICS
from nfl_predictor.season_stats.metrics.unit_rank_metrics import UNIT_RANK_METRICS

TEAM_CATEGORIES: tuple[CategoryDef, ...] = (
    CategoryDef(
        name="Schedule-Adjusted Ratings",
        entity="team",
        description=(
            "The project's own ratings, in points per game against an average team and adjusted "
            "for the opponents each team actually played, plus their ranks and how firm those "
            "ranks are. Start here when ranking teams."
        ),
    ),
    CategoryDef(
        name="Overall",
        entity="team",
        description=(
            "Who played whom and how games turned out: record, points, and whole-team margins."
        ),
    ),
    CategoryDef(
        name="Offense",
        entity="team",
        description=(
            "What the offense did with the ball: passing, rushing, receiving, scoring, "
            "conversions, drives, turnovers, and penalties."
        ),
        subcategories=(
            "Total",
            "Passing",
            "Rushing",
            "Receiving",
            "Scoring",
            "Downs & Conversions",
            "Drives & Field Position",
            "Turnovers",
            "Penalties",
        ),
    ),
    CategoryDef(
        name="Defense",
        entity="team",
        description="Everything the team allowed, plus the plays its defense made.",
        subcategories=(
            "Total",
            "Passing",
            "Rushing",
            "Receiving",
            "Scoring",
            "Downs & Conversions",
            "Drives & Field Position",
            "Turnovers",
            "Pressure & Playmaking",
            "Penalties",
        ),
    ),
    CategoryDef(
        name="Special Teams",
        entity="team",
        description="Kicks, punts, returns, field goals, and extra points.",
    ),
)

QB_CATEGORIES: tuple[CategoryDef, ...] = (
    CategoryDef(
        name="Schedule-Adjusted Ratings",
        entity="qb",
        description=(
            "The project's own quarterback ratings, in EPA per dropback adjusted for the defenses "
            "each quarterback actually faced, plus their ranks and how firm those ranks are. "
            "Start here when ranking QBs."
        ),
    ),
    CategoryDef(
        name="Identity & Availability",
        entity="qb",
        description=(
            "Who the quarterback is, how much he played, and whether he played enough to be ranked."
        ),
    ),
    CategoryDef(
        name="Passing Volume",
        entity="qb",
        description=(
            "Raw passing production (attempts, completions, yards, touchdowns, interceptions, and "
            "passing EPA)."
        ),
    ),
    CategoryDef(
        name="Passing Efficiency",
        entity="qb",
        description="Quality per play: the rates that separate good QBs from busy ones.",
    ),
    CategoryDef(
        name="Pressure, Sacks & Pocket",
        entity="qb",
        description=(
            "Sacks taken, the yards and fumbles lost on them, and how often the quarterback "
            "scrambled."
        ),
    ),
    CategoryDef(
        name="Rushing",
        entity="qb",
        description="Quarterback runs: designed carries, scrambles, and their value.",
    ),
    CategoryDef(
        name="Scoring, Clutch & Outcomes",
        entity="qb",
        description=(
            "Results and late-game moments: wins, comebacks, and game-winning drives. "
            "Context stats — they never feed the performance ratings."
        ),
    ),
    CategoryDef(
        name="Turnovers & Ball Security",
        entity="qb",
        description=(
            "Fumbles on quarterback runs and the touchdown-minus-interception margin. "
            "Interceptions are under Passing Volume, and sack fumbles under Pressure, Sacks & "
            "Pocket."
        ),
    ),
)


def build_registry() -> MetricRegistry:
    """Build and validate the full project registry."""
    return MetricRegistry(
        metrics=TEAM_METRICS + UNIT_RANK_METRICS + QB_METRICS + PAIR_METRICS,
        categories=TEAM_CATEGORIES + QB_CATEGORIES,
    )


@lru_cache(maxsize=1)
def get_registry() -> MetricRegistry:
    """Build, validate, and cache the project registry."""
    return build_registry()
