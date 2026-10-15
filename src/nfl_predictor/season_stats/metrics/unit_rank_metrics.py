"""Unit rank metrics: each team's place in the league by offense, defense, and special teams.

The ``rating_ranges`` file carries them with their game-bootstrap percentiles (``_q025`` through
``_q975``). Kept apart from the team definition module, which is already long.
"""

from __future__ import annotations

from nfl_predictor.season_stats.metrics.schema import MetricDef, section

_ratings = section("team", "Schedule-Adjusted Ratings")


def _unit_rank(unit: str, label: str, rating_label: str) -> MetricDef:
    """Return the published rank metric of one unit rating."""
    return _ratings(
        name=f"{unit}_rank",
        label=f"{label} Rank",
        full_name=f"{rating_label} Rank",
        description=(
            f"The team's place in the league by {rating_label}, 1 for the best. Teams with equal "
            "ratings share the better rank."
        ),
        shape="score",
        polarity="lower",
        source="D",
        since=1999,
    )


UNIT_RANK_METRICS: tuple[MetricDef, ...] = (
    _unit_rank("offense", "Off", "Offense Rating"),
    _unit_rank("defense", "Def", "Defense Rating"),
    _unit_rank("special_teams", "ST", "Special Teams Rating"),
)

__all__ = ["UNIT_RANK_METRICS"]
