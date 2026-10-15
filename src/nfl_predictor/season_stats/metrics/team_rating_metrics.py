"""Team rating and special-teams definitions.

The published ratings, schedule strength, and SRS, plus the special-teams play counts the
ratings use.

One part of the team registry that `team_metrics.TEAM_METRICS` assembles in catalog order;
human-readable companion: [docs/stats-catalog.md](../../docs/stats-catalog.md).
"""

from __future__ import annotations

from nfl_predictor.season_stats.metrics.schema import MetricDef, section

_ratings = section("team", "Schedule-Adjusted Ratings")
_special_teams = section("team", "Special Teams")

RATING_METRICS: tuple[MetricDef, ...] = (
    _ratings(
        name="team_rating",
        label="Team Rating",
        full_name="Team Rating",
        description=(
            "Points per game better (+) or worse (-) than an average team on a neutral field, "
            "built from expected points added (EPA) per play and adjusted for every opponent "
            "faced. 0 is average; most teams fall between about -8 and +8."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        formula="Offense Rating + Defense Rating + Special Teams Rating",
        note=(
            "Until a team has played 9 games, its offense and defense also lean on last season's "
            "(Offense Prior and Defense Prior), a little less after each game."
        ),
    ),
    _ratings(
        name="offense_rating",
        label="Off Rating",
        full_name="Offense Rating",
        description=(
            "Points per game the offense added compared with an average offense, from its EPA per "
            "run or pass play, adjusted for the defenses it faced. Pace does not count: every team "
            "is scaled to the league-average number of plays."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        formula=(
            "Opponent-adjusted EPA per scrimmage play above average x league-average scrimmage "
            "plays per team-game"
        ),
        note=(
            "Until a team has played 9 games, its offense and defense also lean on last season's "
            "(Offense Prior and Defense Prior), a little less after each game."
        ),
    ),
    _ratings(
        name="defense_rating",
        label="Def Rating",
        full_name="Defense Rating",
        description=(
            "Points per game the defense saved compared with an average defense, from the EPA per "
            "run or pass play it allowed, adjusted for the offenses it faced."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        formula=(
            "Opponent-adjusted EPA per scrimmage play prevented x league-average scrimmage plays "
            "per team-game"
        ),
        note=(
            "Until a team has played 9 games, its offense and defense also lean on last season's "
            "(Offense Prior and Defense Prior), a little less after each game."
        ),
    ),
    _ratings(
        name="special_teams_rating",
        label="ST Rating",
        full_name="Special Teams Rating",
        description=(
            "Points per game gained on kicking plays (kickoffs, punts, field goals, extra points, "
            "and their returns) compared with an average team, counting plays with and without the "
            "ball, adjusted for the opponents faced."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        formula=(
            "(Opponent-adjusted EPA per kicking play with the ball + EPA per kicking play "
            "prevented without it) x league-average kicking plays per team-game"
        ),
    ),
    _ratings(
        name="SRS",
        label="SRS",
        full_name="Simple Rating System",
        description=(
            "Average point margin per game, adjusted for the strength of the opponents played, "
            "with each opponent also judged by its point margin. Built from final scores rather "
            "than plays, as a check beside Team Rating. 0 is average."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        formula=(
            "Average point margin + average SRS of the opponents played, solved for every team at "
            "once and centered at 0"
        ),
    ),
    _ratings(
        name="sos",
        label="SoS",
        full_name="Strength of Schedule",
        description=(
            "The average Team Rating of the opponents played, one entry per game, so a team faced "
            "twice counts twice. Each opponent is rated with this team's games left out. Higher "
            "means a tougher schedule; 0 is average."
        ),
        shape="score",
        polarity="higher",
        source="D",
        since=1999,
        contextual=True,
        formula=(
            "Average, over games played, of each opponent's Team Rating refit without this team's "
            "games; opponents with no other games yet are skipped"
        ),
    ),
    _ratings(
        name="team_rank",
        label="Rank",
        full_name="Team Rating Rank",
        description=(
            "The team's place in the league by Team Rating, 1 for the best. Teams with equal "
            "ratings share the better rank."
        ),
        shape="score",
        polarity="lower",
        source="D",
        since=1999,
    ),
    _ratings(
        name="team_rank_missing_share",
        label="Unranked Share",
        full_name="Share of Redrawn Seasons Without the Team",
        description=(
            "Share of 1,000 redrawn seasons (the season's games drawn at random, with repeats) in "
            "which the team had no games and so no rank. Essentially zero for a finished season; "
            "it grows early in a season."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="team_rank_top5_probability",
        label="Top-5 Chance",
        full_name="Chance of a Top-5 Rank",
        description=(
            "Share of 1,000 redrawn seasons (the season's games drawn at random, with repeats) in "
            "which the team ranked in the top five by Team Rating. Shows how much the ranking "
            "depends on which games happened to be played."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="team_rank_top10_probability",
        label="Top-10 Chance",
        full_name="Chance of a Top-10 Rank",
        description=(
            "Share of 1,000 redrawn seasons (the season's games drawn at random, with repeats) in "
            "which the team ranked in the top ten by Team Rating. Shows how much the ranking "
            "depends on which games happened to be played."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="team_rank_probabilities",
        label="Rank Chances",
        full_name="Chance of Each Rank",
        description=(
            "For each rank from 1st down, the share of 1,000 redrawn seasons (the season's games "
            "drawn at random, with repeats) in which the team finished at exactly that rank by "
            "Team Rating."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="excluded_team",
        label="Left-Out Team",
        full_name="Team Left Out of the Refit",
        description=(
            "For a schedule-strength refit, the team whose games are left out, this season's and "
            "last season's; blank for the season fit itself."
        ),
        shape="id",
        polarity="neutral",
        source="D",
        since=2003,
    ),
    _ratings(
        name="offense_prior",
        label="Offense Prior",
        full_name="Preseason Prior: Offense",
        description=(
            "The per-play offense value the early-season fit pulls this team toward instead of "
            "average: its offense last season times how much of it usually carries over, fading "
            "to nothing by its 9th game, then shifted so the league averages zero."
        ),
        shape="score",
        polarity="neutral",
        source="D",
        since=2003,
        formula=(
            "max(0, 1 - games played / 9) x carryover slope x last season's per-play offense "
            "effect, minus the league average of the same"
        ),
    ),
    _ratings(
        name="defense_prior",
        label="Defense Prior",
        full_name="Preseason Prior: Defense",
        description=(
            "The per-play defense value the early-season fit pulls this team toward instead of "
            "average: its defense last season times how much of it usually carries over, fading "
            "to nothing by its 9th game, then shifted so the league averages zero."
        ),
        shape="score",
        polarity="neutral",
        source="D",
        since=2003,
        formula=(
            "max(0, 1 - games played / 9) x carryover slope x last season's per-play defense "
            "effect, minus the league average of the same"
        ),
    ),
)

SPECIAL_TEAMS_METRICS: tuple[MetricDef, ...] = (
    _special_teams(
        name="st_plays",
        label="ST Plays",
        full_name="Special Teams Plays",
        description=(
            "Special-teams plays with the ball: punts, field goals, and extra points, plus "
            "kickoffs received."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _special_teams(
        name="st_epa",
        label="ST EPA",
        full_name="Special Teams EPA",
        description=(
            "Expected points added on special-teams plays with the ball: punts, field goals, extra "
            "points, and kickoffs received. Punt returns and kickoff coverage are not included."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
)

__all__ = [
    "RATING_METRICS",
    "SPECIAL_TEAMS_METRICS",
]
