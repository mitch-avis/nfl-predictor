"""Team overall definitions.

Results, margins, and the stats that span offense and defense.

One part of the team registry that `team_metrics.TEAM_METRICS` assembles in catalog order;
human-readable companion: [docs/stats-catalog.md](../../docs/stats-catalog.md).
"""

from __future__ import annotations

from nfl_predictor.season_stats.metrics.schema import MetricDef, section

_overall = section("team", "Overall")

OVERALL_METRICS: tuple[MetricDef, ...] = (
    _overall(
        name="team",
        label="Team",
        full_name="Team",
        description="The team's standard NFL abbreviation, such as KC or PHI.",
        shape="id",
        polarity="neutral",
        source="SCH",
    ),
    _overall(
        name="game_id",
        label="Game ID",
        full_name="Game ID",
        description=(
            "A unique code for one game: season, week, away team, and home team, for example "
            "2025_01_ARI_NO."
        ),
        shape="id",
        polarity="neutral",
        source="SCH",
    ),
    _overall(
        name="week",
        label="Week",
        full_name="Week",
        description=(
            "The regular-season week: on a game row, the week the game was played; on a "
            "rating-history row, the last week of games the rating includes."
        ),
        shape="id",
        polarity="neutral",
        source="SCH",
    ),
    _overall(
        name="opponent_team",
        label="Opponent",
        full_name="Opponent Team",
        description="The opposing team in this game or in the summarized opponent row.",
        shape="id",
        polarity="neutral",
        source="SCH",
    ),
    _overall(
        name="is_home",
        label="Home",
        full_name="Home Game",
        description=(
            "Whether the team was the home team in this game. At a neutral site, this is the team "
            "the schedule lists as home."
        ),
        shape="flag",
        polarity="neutral",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="games_played",
        label="Games Played",
        full_name="Games Played",
        description=(
            "Regular-season games played (on a week-by-week history row, games through that week). "
            "Per-game figures are season totals divided by this number."
        ),
        shape="count",
        polarity="neutral",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="games",
        label="Games",
        full_name="Games",
        description=(
            "How many games this row covers: 1 for a single game, more for a summary row such as "
            "all games against one opponent."
        ),
        shape="count",
        polarity="neutral",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="wins",
        label="Wins",
        full_name="Wins",
        description="Regular-season games won. Treated as an outcome, not a rating input.",
        shape="count",
        polarity="higher",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="losses",
        label="Losses",
        full_name="Losses",
        description="Regular-season games lost. Treated as an outcome, not a rating input.",
        shape="count",
        polarity="lower",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="ties",
        label="Ties",
        full_name="Ties",
        description="Regular-season games that ended tied after overtime.",
        shape="count",
        polarity="neutral",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="win_pct",
        label="Win %",
        full_name="Win Percentage",
        description=("Share of games won, with a tie counted as half a win."),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="games played",
        since=1999,
        formula="(Wins + 0.5 x ties) ÷ games played",
        percent=True,
    ),
    _overall(
        name="win_value",
        label="Win Value",
        full_name="Win Value",
        description=(
            "A game's result as a number: 1 for a win, 0.5 for a tie, 0 for a loss. A row covering "
            "several games shows the average, which works like a win percentage."
        ),
        shape="avg",
        polarity="higher",
        source="D",
        denominator="games included",
        since=1999,
    ),
    _overall(
        name="points_for",
        label="Points For",
        full_name="Points Scored",
        description="Points scored, from every source: offense, defense, and special teams.",
        shape="count",
        polarity="higher",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="points_allowed",
        label="Points Allowed",
        full_name="Points Allowed",
        description=(
            "Points given up, counting every score by the other side, not only those against the "
            "defense."
        ),
        shape="count",
        polarity="lower",
        source="SCH",
        since=1999,
    ),
    _overall(
        name="point_margin",
        label="Point Margin",
        full_name="Point Margin",
        description=("Points scored minus points allowed."),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
        formula=None,
        note=None,
    ),
    _overall(
        name="turnover_margin",
        label="TO Margin",
        full_name="Turnover Margin",
        description=(
            "Takeaways minus giveaways: interceptions made and opponent fumbles recovered, minus "
            "interceptions thrown and fumbles lost."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="points_per_offensive_snap",
        label="Points/Off Snap",
        full_name="Points Per Offensive Snap",
        description=(
            "Points scored per offensive snap (each run or pass play), counting every score, "
            "including defensive and special-teams touchdowns."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="offensive snaps",
        since=1999,
    ),
    _overall(
        name="points_allowed_per_defensive_snap",
        label="Pts Allowed/Snap",
        full_name="Points Allowed Per Defensive Snap",
        description=(
            "Points given up per defensive snap (each run or pass play faced), counting every "
            "score by the other side."
        ),
        shape="rate",
        polarity="lower",
        source="D",
        denominator="defensive snaps",
        since=1999,
    ),
    _overall(
        name="total_yards_differential",
        label="Total Yds Diff",
        full_name="Total Yards Differential",
        description=(
            "Passing plus rushing yards gained minus passing plus rushing yards allowed. Passing "
            "yards are counted before sack losses."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="penalty_differential",
        label="Penalty Diff",
        full_name="Penalty Differential",
        description=(
            "Penalties drawn (called on the opponent) minus penalties committed, across offense, "
            "defense, and special teams."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="penalty_yards_differential",
        label="Pen Yds Diff",
        full_name="Penalty Yards Differential",
        description=(
            "Penalty yards drawn (called on the opponent) minus penalty yards committed, across "
            "offense, defense, and special teams."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="epa_margin_per_play",
        label="EPA Margin/Play",
        full_name="EPA Margin Per Play",
        description=(
            "Offensive EPA per play minus defensive EPA per play allowed, not adjusted for "
            "opponents. EPA (expected points added) measures how much a play changed the offense's "
            "expected points."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="scrimmage snaps",
        since=1999,
    ),
    _overall(
        name="success_rate_margin",
        label="Success Margin",
        full_name="Success Rate Margin",
        description=(
            "Share of the offense's plays that succeeded minus the share that succeeded against "
            "the defense. A play succeeds when it raises the offense's expected points."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="scrimmage snaps",
        since=1999,
        percent=True,
    ),
    _overall(
        name="wp_unit",
        label="Unit",
        full_name="Play Unit",
        description=(
            "Which plays this row counts: the team's runs and passes (scrimmage), or the kicking "
            "plays where it had the ball (special teams)."
        ),
        shape="id",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="wp_bin",
        label="WP Bin",
        full_name="Win-Probability Bin",
        description=(
            "How close the game was before the snap: the underdog's chance of winning at that "
            "moment, in whole percentage points, rounded down. 50 is a toss-up; 0 means one side "
            "was over 99% likely to win."
        ),
        shape="id",
        polarity="neutral",
        source="PBP",
        since=1999,
        formula=(
            "100 x the smaller of the offense's win probability and 1 minus it, rounded down. "
            "Plays without a win probability get no bin, and every filter keeps them."
        ),
    ),
    _overall(
        name="wp_kept_play_share",
        label="Kept Plays",
        full_name="Share of Plays the Filter Keeps",
        description=(
            "Share of the team's own plays (runs, passes, and kicking plays with the ball) that "
            "the garbage-time filter keeps at the chosen setting. 100% means no play was dropped."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="scrimmage and special-teams plays",
        since=1999,
        percent=True,
    ),
    _overall(
        name="wp_bin_plays",
        label="Plays",
        full_name="Plays in Win-Probability Bin",
        description=(
            "Plays the team ran with the ball in one game, play unit, and win-probability bin. "
            "Added up over all bins, they equal the game's total for that unit."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _overall(
        name="wp_bin_epa",
        label="EPA",
        full_name="EPA in Win-Probability Bin",
        description=(
            "Expected points added on the team's plays with the ball in one game, play unit, and "
            "win-probability bin. Added up over all bins, it equals the game's total for that "
            "unit."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
)

__all__ = [
    "OVERALL_METRICS",
]
