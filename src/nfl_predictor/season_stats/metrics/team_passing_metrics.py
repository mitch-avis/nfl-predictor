"""Team offense passing and receiving definitions.

Passing volume, efficiency, depth, and the receiving side of the same plays.

One part of the team registry that `team_metrics.TEAM_METRICS` assembles in catalog order;
human-readable companion: [docs/stats-catalog.md](../../docs/stats-catalog.md).
"""

from __future__ import annotations

from nfl_predictor.season_stats.metrics.schema import MetricDef, section

_off_pass = section("team", "Offense", "Passing")
_off_recv = section("team", "Offense", "Receiving")

_TARGETS_GAP_NOTE = (
    "Blank in 2003-2008, when nflverse play-by-play names the intended receiver on almost no "
    "incomplete passes."
)

OFFENSE_PASSING_METRICS: tuple[MetricDef, ...] = (
    _off_pass(
        name="passing_yards",
        label="Pass Yds",
        full_name="Passing Yards",
        description=(
            "Yards gained on completed passes, before taking away yards lost on sacks. Net Passing "
            "Yards subtracts sacks, as the NFL's official team stats do."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_epa",
        label="Pass EPA",
        full_name="Passing Expected Points Added",
        description=(
            "EPA (expected points added) on pass attempts and sacks; scrambles count as runs. EPA "
            "is how much a play raised or lowered the offense's expected points, given down, "
            "distance, and field position."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_tds",
        label="Pass TDs",
        full_name="Passing Touchdowns",
        description="Touchdown passes thrown by the team.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_first_downs",
        label="Pass 1Ds",
        full_name="Passing First Downs",
        description=(
            "First downs gained by completed passes. Scrambles count as runs, and first downs "
            "awarded by penalty are left out."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_cpoe",
        label="Pass CPOE",
        full_name="Completion Percentage Above Expectation",
        description=(
            "Completion percentage compared with what an average passer would complete on the same "
            "throws, given how hard each was, in percentage points: 0 is as expected, +3 is 3 "
            "points above."
        ),
        shape="avg",
        polarity="higher",
        source="PBP +TS",
        denominator="pass attempts (model-expected completions)",
        since=2006,
        formula=(
            "Actual minus expected completion chance on each pass attempt, averaged over the "
            "attempts, in percentage points"
        ),
    ),
    _off_pass(
        name="sacks_suffered",
        label="Sacks Taken",
        full_name="Sacks Suffered",
        description="Times the team's quarterbacks were sacked.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_interceptions",
        label="INTs Thrown",
        full_name="Interceptions Thrown",
        description="Passes the defense intercepted.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="sack_fumbles_lost",
        label="Sack Fum Lost",
        full_name="Sack Fumbles Lost",
        description="Fumbles lost to the defense on sack plays (strip-sacks).",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="attempts",
        label="Att",
        full_name="Pass Attempts",
        description=(
            "Official pass attempts: every throw, complete or not, including interceptions and "
            "spikes. Sacks and two-point tries are left out, as in the league's official stats."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=1999,
        formula=None,
    ),
    _off_pass(
        name="completions",
        label="Comp",
        full_name="Completions",
        description="Completed passes by the team.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="completion_pct",
        label="Comp %",
        full_name="Completion Percentage",
        description="The share of official pass attempts that were completed.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts",
        since=1999,
        percent=True,
    ),
    _off_pass(
        name="net_passing_yards",
        label="Net Pass Yds",
        full_name="Net Passing Yards",
        description=(
            "Passing yards minus the yards lost on sacks. This is how the NFL's official team "
            "stats count passing yards."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        formula=None,
    ),
    _off_pass(
        name="dropbacks",
        label="Dropbacks",
        full_name="Dropbacks",
        description=("Every play that started as a pass: pass attempts, sacks, and scrambles."),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="sack_yards_lost",
        label="Sack Yds Lost",
        full_name="Sack Yards Lost",
        description="Yards lost on sacks, shown as a positive number.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
        note=None,
    ),
    _off_pass(
        name="sack_rate_per_dropback",
        label="Sack Rate",
        full_name="Sack Rate",
        description="The share of dropbacks that ended in a sack.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="dropbacks",
        since=1999,
        percent=True,
    ),
    _off_pass(
        name="scrambles",
        label="Scrambles",
        full_name="QB Scrambles",
        description=(
            "Times the quarterback dropped back to pass and then ran with the ball instead."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="scramble_yards",
        label="Scramble Yds",
        full_name="Scramble Yards",
        description="Rushing yards gained on quarterback scrambles.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="passing_air_yards",
        label="Air Yds",
        full_name="Passing Air Yards",
        description=(
            "Total distance the ball traveled past the line of scrimmage on all throws, "
            "including incompletions — a measure of how far downfield the team attacks."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=2006,
    ),
    _off_pass(
        name="passing_yards_after_catch",
        label="YAC",
        full_name="Passing Yards After Catch",
        description="Yards receivers gained after catching the ball.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=2006,
    ),
    _off_pass(
        name="air_yards_per_attempt",
        label="Air Yds/Att",
        full_name="Air Yards Per Attempt",
        description=(
            "How far past the line of scrimmage the average pass attempt traveled in the air, "
            "throwaways included. A style stat, not a quality grade."
        ),
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="pass attempts",
        since=2006,
    ),
    _off_pass(
        name="yac_per_completion",
        label="YAC/Comp",
        full_name="Yards After Catch Per Completion",
        description="Average yards gained after the catch on completed passes.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="completions",
        since=2006,
    ),
    _off_pass(
        name="yards_per_attempt",
        label="Y/A",
        full_name="Yards Per Attempt",
        description=(
            "Average passing yards per pass attempt. Sacks don't count as attempts, and their lost "
            "yards are not subtracted."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts",
        since=1999,
    ),
    _off_pass(
        name="net_yards_per_attempt",
        label="NY/A",
        full_name="Net Yards Per Attempt",
        description=(
            "Yards per pass play with sacks counted: each sack adds a play, and its lost yards "
            "come off the total."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts + sacks",
        since=1999,
        formula="(Passing yards - sack yards lost) ÷ (pass attempts + sacks)",
    ),
    _off_pass(
        name="adjusted_net_yards_per_attempt",
        label="ANY/A",
        full_name="Adjusted Net Yards Per Attempt",
        description=(
            "Net yards per attempt with a 20-yard bonus for each touchdown pass and a 45-yard "
            "charge for each interception, so scoring and turnovers count along with yards."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts + sacks",
        since=1999,
        formula=(
            "(Passing yards + 20 x TD passes - 45 x interceptions - sack yards lost) ÷ (pass "
            "attempts + sacks)"
        ),
    ),
    _off_pass(
        name="yards_per_dropback",
        label="Yds/Dropback",
        full_name="Yards Per Dropback",
        description=(
            "Passing yards per dropback. Sacks and scrambles count as dropbacks but add no passing "
            "yards, and sack losses are not subtracted."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
    ),
    _off_pass(
        name="epa_per_dropback",
        label="EPA/Dropback",
        full_name="EPA Per Dropback",
        description=(
            "Average EPA per dropback, scrambles included: the core measure of passing efficiency. "
            "EPA is how much a play raised or lowered the offense's expected points. Team seasons "
            "mostly run -0.2 to +0.25."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
    ),
    _off_pass(
        name="pass_success_rate",
        label="Pass Success %",
        full_name="Passing Success Rate",
        description=(
            "The share of dropbacks with positive EPA, meaning the play left the offense better "
            "placed to score than before. It rewards steady gains, not just big plays."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
        percent=True,
    ),
    _off_pass(
        name="team_passer_rating",
        label="Passer Rating",
        full_name="Team Passer Rating",
        description=(
            "The NFL's official passer rating formula (scale 0 to 158.3), built from completion %, "
            "yards, touchdowns, and interceptions per attempt, applied to all the team's passes."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="official NFL formula over attempts",
        since=1999,
        formula=(
            "Four parts, each held between 0 and 2.375: (completions ÷ attempts - 0.3) x 5, (yards "
            "÷ attempts - 3) x 0.25, TD passes ÷ attempts x 20, and 2.375 - interceptions ÷ "
            "attempts x 25. Rating = their sum ÷ 6 x 100."
        ),
    ),
    _off_pass(
        name="passing_td_rate_per_attempt",
        label="TD %",
        full_name="Passing Touchdown Rate",
        description="The share of pass attempts that scored touchdowns.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="pass attempts",
        since=1999,
        percent=True,
    ),
    _off_pass(
        name="int_rate_per_attempt",
        label="INT %",
        full_name="Interception Rate",
        description="The share of pass attempts that were intercepted.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="pass attempts",
        since=1999,
        percent=True,
    ),
    _off_pass(
        name="explosive_pass_rate",
        label="Explosive Pass %",
        full_name="Explosive Pass Rate",
        description="The share of dropbacks that produced a completion of 20 or more yards.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="dropbacks",
        since=1999,
        percent=True,
    ),
    _off_pass(
        name="deep_attempt_rate",
        label="Deep Att %",
        full_name="Deep Attempt Rate",
        description=(
            "The share of pass attempts thrown 16 or more yards past the line of scrimmage. It "
            "shows how often a team throws deep, not how well."
        ),
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="pass attempts",
        since=2006,
        percent=True,
    ),
    _off_pass(
        name="longest_pass",
        label="Longest Pass",
        full_name="Longest Completed Pass",
        description=(
            "The longest completed pass, in yards: the season's longest in season tables, that "
            "game's longest in game logs."
        ),
        shape="max",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_pass(
        name="sack_fumbles",
        label="Sack Fumbles",
        full_name="Sack Fumbles",
        description="Fumbles on sack plays, whether or not the team lost the ball.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="passing_2pt_conversions",
        label="2-Pt Pass Conv",
        full_name="Passing Two-Point Conversions",
        description="Successful two-point conversions thrown.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_pass(
        name="air_epa_total",
        label="Air EPA",
        full_name="Expected Points Added Through the Air",
        description=(
            "EPA credited to the distance each pass traveled in the air, summed over every pass "
            "attempt. Incompletions and interceptions count the value the pass would have added if "
            "caught."
        ),
        shape="count",
        # It tracks how deep a team throws more than how well: neutral, like the air-yards stats.
        polarity="neutral",
        source="PBP",
        since=2006,
    ),
    _off_pass(
        name="yac_epa_total",
        label="YAC EPA",
        full_name="Expected Points Added After the Catch",
        description=(
            "EPA added after the catch on completed passes: the value receivers created with the "
            "ball in their hands."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=2006,
        formula="Sum over completed passes of the play's EPA minus its air EPA",
    ),
    _off_pass(
        name="xyac_per_completion",
        label="Exp YAC/Comp",
        full_name="Expected Yards After Catch Per Completion",
        description=(
            "The yards after the catch an average receiver would be expected to gain on the same "
            "catches, from nflverse's expected-yards model."
        ),
        shape="avg",
        polarity="neutral",
        source="PBP",
        denominator="completions",
        since=2006,
    ),
    _off_pass(
        name="yac_over_expected_per_completion",
        label="YAC Over Exp",
        full_name="Yards After Catch Over Expected",
        description=(
            "How many more yards after the catch the team gained per catch than an average "
            "receiver would on the same catches. Zero means as expected."
        ),
        shape="avg",
        polarity="higher",
        source="PBP",
        denominator="completions",
        since=2006,
    ),
)

OFFENSE_RECEIVING_METRICS: tuple[MetricDef, ...] = (
    _off_recv(
        name="targets",
        label="Targets",
        full_name="Targets",
        description=(
            "Pass attempts thrown to an intended receiver. Throwaways and spikes are attempts "
            "but not targets."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=1999,
        note=_TARGETS_GAP_NOTE,
    ),
    _off_recv(
        name="receptions",
        label="Receptions",
        full_name="Receptions",
        description="Team catches — the same number as completions, receiving-side view.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        duplicate_of="completions",
    ),
    _off_recv(
        name="receiving_yards",
        label="Rec Yds",
        full_name="Receiving Yards",
        description=(
            "Yards gained on catches, including yards after the catch; the same number as passing "
            "yards."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        duplicate_of="passing_yards",
    ),
    _off_recv(
        name="receiving_tds",
        label="Rec TDs",
        full_name="Receiving Touchdowns",
        description="Touchdown catches — the same number as passing touchdowns.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        duplicate_of="passing_tds",
    ),
    _off_recv(
        name="receiving_air_yards",
        label="Rec Air Yds",
        full_name="Receiving Air Yards",
        description=(
            "How far past the line of scrimmage passes were thrown, summed over every attempt; the "
            "same number as passing air yards."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=2006,
        duplicate_of="passing_air_yards",
    ),
    _off_recv(
        name="receiving_yards_after_catch",
        label="Rec YAC",
        full_name="Receiving Yards After Catch",
        description=(
            "Yards receivers gained after the catch; the same number as passing yards after catch."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=2006,
        duplicate_of="passing_yards_after_catch",
    ),
    _off_recv(
        name="receiving_first_downs",
        label="Rec 1Ds",
        full_name="Receiving First Downs",
        description="First downs on catches — the same number as passing first downs.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        duplicate_of="passing_first_downs",
    ),
    _off_recv(
        name="receiving_fumbles",
        label="Rec Fumbles",
        full_name="Receiving Fumbles",
        description="Fumbles by receivers after the catch, whether or not lost.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_recv(
        name="receiving_fumbles_lost",
        label="Rec Fum Lost",
        full_name="Receiving Fumbles Lost",
        description="Receivers' fumbles after the catch that the defense recovered.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_recv(
        name="catch_rate",
        label="Catch %",
        full_name="Catch Rate",
        description=(
            "The share of targets that were caught. Throwaways and spikes are not targets, so "
            "unlike completion percentage they do not count as misses."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="targets",
        since=1999,
        note=_TARGETS_GAP_NOTE,
        percent=True,
    ),
)

__all__ = [
    "OFFENSE_PASSING_METRICS",
    "OFFENSE_RECEIVING_METRICS",
]
