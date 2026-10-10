"""Team metric definitions: every team column the pipeline publishes, in catalog order.

The definitions live in one module per category (`team_rating_metrics`, `team_overall_metrics`,
`team_offense_metrics`, `team_passing_metrics`, `team_defense_metrics`); this module assembles
them in the order the catalog and the app show them. Human-readable companion:
[docs/stats-catalog.md](../../docs/stats-catalog.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfl_predictor.season_stats.metrics.team_defense_metrics import (
    DEFENSE_METRICS,
)
from nfl_predictor.season_stats.metrics.team_offense_metrics import (
    OFFENSE_DOWNS_METRICS,
    OFFENSE_DRIVES_METRICS,
    OFFENSE_PENALTY_METRICS,
    OFFENSE_RUSHING_METRICS,
    OFFENSE_SCORING_METRICS,
    OFFENSE_TOTAL_METRICS,
    OFFENSE_TURNOVER_METRICS,
)
from nfl_predictor.season_stats.metrics.team_overall_metrics import (
    OVERALL_METRICS,
)
from nfl_predictor.season_stats.metrics.team_passing_metrics import (
    OFFENSE_PASSING_METRICS,
    OFFENSE_RECEIVING_METRICS,
)
from nfl_predictor.season_stats.metrics.team_rating_metrics import (
    RATING_METRICS,
    SPECIAL_TEAMS_METRICS,
)

if TYPE_CHECKING:
    from nfl_predictor.season_stats.metrics.schema import MetricDef

TEAM_METRICS: tuple[MetricDef, ...] = (
    RATING_METRICS
    + OVERALL_METRICS
    + OFFENSE_TOTAL_METRICS
    + OFFENSE_PASSING_METRICS
    + OFFENSE_RUSHING_METRICS
    + OFFENSE_RECEIVING_METRICS
    + OFFENSE_SCORING_METRICS
    + OFFENSE_DOWNS_METRICS
    + OFFENSE_DRIVES_METRICS
    + OFFENSE_TURNOVER_METRICS
    + OFFENSE_PENALTY_METRICS
    + DEFENSE_METRICS
    + SPECIAL_TEAMS_METRICS
)

__all__ = [
    "DEFENSE_METRICS",
    "OFFENSE_DOWNS_METRICS",
    "OFFENSE_DRIVES_METRICS",
    "OFFENSE_PASSING_METRICS",
    "OFFENSE_PENALTY_METRICS",
    "OFFENSE_RECEIVING_METRICS",
    "OFFENSE_RUSHING_METRICS",
    "OFFENSE_SCORING_METRICS",
    "OFFENSE_TOTAL_METRICS",
    "OFFENSE_TURNOVER_METRICS",
    "OVERALL_METRICS",
    "RATING_METRICS",
    "SPECIAL_TEAMS_METRICS",
    "TEAM_METRICS",
]
