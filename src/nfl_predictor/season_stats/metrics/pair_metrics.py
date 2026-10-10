"""Head-to-head pair metrics: how often one team or quarterback is rated above another.

Each row of the ``rating_pairs`` and ``qb_rating_pairs`` files compares two teams or two
quarterbacks across the same game-bootstrap resamples that give the rank ranges. Kept apart from
the team and QB definition modules, which are already long.
"""

from __future__ import annotations

from nfl_predictor.season_stats.metrics.schema import MetricDef, section

_team = section("team", "Schedule-Adjusted Ratings")
_qb = section("qb", "Schedule-Adjusted Ratings")

TEAM_PAIR_METRICS: tuple[MetricDef, ...] = (
    _team(
        name="other_team",
        label="Compared Team",
        full_name="Compared Team",
        description=(
            "The other team in a head-to-head comparison, by its standard NFL abbreviation. The "
            "two teams need not have played each other."
        ),
        shape="id",
        polarity="neutral",
        source="D",
    ),
    _team(
        name="team_rated_above_probability",
        label="Rated-Above Chance",
        full_name="Chance of Being Rated Above the Compared Team",
        description=(
            "Among 1,000 redrawn seasons (the season's games drawn at random, with repeats) that "
            "include both teams, the share in which this team's Team Rating beat the compared "
            "team's. A tie counts half."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples with both teams",
        since=1999,
        percent=True,
    ),
    _team(
        name="team_rating_gap",
        label="Rating Gap",
        full_name="Team Rating Minus the Compared Team's",
        description=(
            "This team's Team Rating minus the compared team's, in points per game, in each "
            "redrawn season (the season's games drawn at random, with repeats). Negative means the "
            "compared team was rated higher."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _team(
        name="team_pair_share",
        label="Both-Teams Share",
        full_name="Share of Redrawn Seasons With Both Teams",
        description=(
            "Share of 1,000 redrawn seasons in which both teams had at least one game, so they "
            "could be compared. It is 1 for a finished season and slightly lower early in a "
            "season."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
)

QB_PAIR_METRICS: tuple[MetricDef, ...] = (
    _qb(
        name="other_qb_id",
        label="Compared QB ID",
        full_name="Compared Quarterback ID",
        description=(
            "The league's player ID for the other quarterback in the comparison. The two need not "
            "have faced each other."
        ),
        shape="id",
        polarity="neutral",
        source="D",
    ),
    _qb(
        name="qb_rated_above_probability",
        label="Rated-Above Chance",
        full_name="Chance of Being Rated Above the Compared Quarterback",
        description=(
            "Among 1,000 redrawn seasons (the season's games drawn at random, with repeats) that "
            "include both quarterbacks, the share in which this one's Adjusted EPA Per Dropback "
            "beat the other's. A tie counts half."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples with both quarterbacks",
        since=1999,
        percent=True,
    ),
    _qb(
        name="qb_rating_gap",
        label="Rating Gap",
        full_name="Adjusted EPA Per Dropback Minus the Compared Quarterback's",
        description=(
            "This quarterback's Adjusted EPA Per Dropback minus the compared quarterback's, in "
            "each redrawn season that includes both. Negative means the other was rated higher; "
            "EPA per dropback values are small decimals (about -0.3 to +0.3)."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _qb(
        name="qb_pair_share",
        label="Both-QBs Share",
        full_name="Share of Redrawn Seasons With Both Quarterbacks",
        description=(
            "Share of 1,000 redrawn seasons that include games by both quarterbacks, so they could "
            "be compared. Only qualifying quarterbacks are paired, so it drops below 1 mainly "
            "early in a season."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
)

PAIR_METRICS: tuple[MetricDef, ...] = TEAM_PAIR_METRICS + QB_PAIR_METRICS

__all__ = ["PAIR_METRICS", "QB_PAIR_METRICS", "TEAM_PAIR_METRICS"]
