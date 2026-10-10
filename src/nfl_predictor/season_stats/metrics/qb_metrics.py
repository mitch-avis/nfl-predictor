"""Quarterback metric definitions: every QB column the pipeline publishes.

Human-readable companion: [docs/qb-stats-catalog.md](../../docs/qb-stats-catalog.md).
"""

from __future__ import annotations

from nfl_predictor.season_stats.metrics.schema import MetricDef, section

_ratings = section("qb", "Schedule-Adjusted Ratings")
_identity = section("qb", "Identity & Availability")
_volume = section("qb", "Passing Volume")
_efficiency = section("qb", "Passing Efficiency")
_pressure = section("qb", "Pressure, Sacks & Pocket")
_rushing = section("qb", "Rushing")
_clutch = section("qb", "Scoring, Clutch & Outcomes")
_turnovers = section("qb", "Turnovers & Ball Security")

QB_RATING_METRICS: tuple[MetricDef, ...] = (
    _ratings(
        name="adj_qb_epa_per_dropback",
        label="Adj EPA/Dropback",
        full_name="Adjusted EPA Per Dropback",
        description=(
            "Expected points added (how much each play raised or lowered the offense's expected "
            "points) per dropback, adjusted for the pass defenses faced. Same scale as raw "
            "EPA/Dropback; every rating is pulled toward average, small samples most."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
        formula=(
            "One fit over every passer-game, weighted by dropbacks: EPA per dropback = league "
            "average + passer strength - defense strength + home field. Shown: league average + "
            "passer strength."
        ),
    ),
    _ratings(
        name="qb_faced_pass_defense",
        label="Pass Defense Faced",
        full_name="Strength of Pass Defenses Faced",
        description=(
            "How strong the pass defenses faced were, in EPA per dropback they held passers below "
            "average, weighted by the quarterback's dropbacks against each. Higher means tougher "
            "defenses; 0 is average."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
        contextual=True,
        formula=(
            "Dropback-weighted average of each opponent's pass-defense strength, each from a refit "
            "that leaves out this quarterback's dropbacks; defenses that have faced no other "
            "passer yet are skipped."
        ),
    ),
    _ratings(
        name="qb_rank",
        label="Rank",
        full_name="QB Rank",
        description=(
            "The quarterback's place among qualifying quarterbacks (14 pass attempts per team "
            "game) by Adjusted EPA Per Dropback; 1 is best. Tied ratings share the better rank."
        ),
        shape="score",
        polarity="lower",
        source="D",
        since=1999,
    ),
    _ratings(
        name="qb_rank_missing_share",
        label="No-Dropback Share",
        full_name="Share of Redrawn Seasons Without the QB",
        description=(
            "How often the quarterback had no dropbacks, and so no rank, across 1,000 redrawn "
            "seasons (the season's games drawn at random, with repeats). Part-time starters miss "
            "more often; his rank range uses only the seasons he appears in."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="qb_rank_top5_probability",
        label="Top-5 Chance",
        full_name="Chance of a Top-5 Rank",
        description=(
            "How often the quarterback ranked in the top five qualifying quarterbacks across 1,000 "
            "redrawn seasons (the season's games drawn at random, with repeats). Shows how much "
            "the ranking rests on which games were played."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="qb_rank_top10_probability",
        label="Top-10 Chance",
        full_name="Chance of a Top-10 Rank",
        description=(
            "How often the quarterback ranked in the top ten qualifying quarterbacks across 1,000 "
            "redrawn seasons (the season's games drawn at random, with repeats). Shows how much "
            "the ranking rests on which games were played."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
    _ratings(
        name="qb_rank_probabilities",
        label="Rank Chances",
        full_name="Chance of Each Rank",
        description=(
            "For each rank from 1 down, how often the quarterback finished exactly there among "
            "qualifying quarterbacks across 1,000 redrawn seasons (the season's games drawn at "
            "random, with repeats). Adds to under 100% when he is missing from some."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="bootstrap resamples",
        since=1999,
        percent=True,
    ),
)

QB_IDENTITY_METRICS: tuple[MetricDef, ...] = (
    _identity(
        name="qb_id",
        label="QB ID",
        full_name="Quarterback ID",
        description=(
            "The league's official player ID for this quarterback (for example 00-0023459), used "
            "to link his rows across tables. If that ID is missing, another source's ID or his "
            "name stands in."
        ),
        shape="id",
        polarity="neutral",
        source="PBP",
    ),
    _identity(
        name="qb_name",
        label="QB",
        full_name="Quarterback",
        description="The quarterback's display name.",
        shape="id",
        polarity="neutral",
        source="PBP",
    ),
    _identity(
        name="qb_games_played",
        label="QB Games",
        full_name="QB Games Played",
        description=(
            "Games in which the quarterback dropped back at least once or, in seasons with snap "
            "counts, played at least one offensive snap. Per-game stats divide by this count."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _identity(
        name="qb_offense_snaps",
        label="QB Snaps",
        full_name="QB Offensive Snaps",
        description=(
            "Offensive snaps the quarterback was on the field for, from the league's snap counts, "
            "which this data has from 2013 on."
        ),
        shape="count",
        polarity="neutral",
        source="SNP",
        since=2013,
        note="Empty before 2013: nflverse's 2012 snap-count file has no rows.",
    ),
    _identity(
        name="qb_dropbacks",
        label="Dropbacks",
        full_name="QB Dropbacks",
        description=(
            "Pass attempts plus sacks: the plays where the quarterback dropped back and threw or "
            "went down. Scrambles are not counted here; they count as carries."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
        formula="Pass attempts (two-point tries included, spikes left out) + sacks",
    ),
    _identity(
        name="qb_is_eligible",
        label="Qualified",
        full_name="Qualified for Ranking",
        description=(
            "Whether the quarterback has enough pass attempts to be ranked: 14 for every game his "
            "team has played (his main team, if he changed teams). Below that he is listed but "
            "unranked."
        ),
        shape="flag",
        polarity="neutral",
        source="D",
    ),
    _identity(
        name="qb_attempt_qualifier",
        label="Att to Qualify",
        full_name="Pass Attempts Needed to Qualify",
        description=(
            "The pass attempts the quarterback needs to be ranked: 14 for every game his team has "
            "played so far (the team he played the most games for, if he changed teams)."
        ),
        shape="count",
        polarity="neutral",
        source="D",
        formula="14 x team games played (238 in a 17-game season)",
    ),
)

QB_VOLUME_METRICS: tuple[MetricDef, ...] = (
    _volume(
        name="qb_attempts",
        label="Att",
        full_name="QB Pass Attempts",
        description="Official pass attempts (sacks and two-point tries excluded).",
        shape="count",
        polarity="neutral",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_completions",
        label="Comp",
        full_name="QB Completions",
        description="Passes completed to a teammate.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_pass_yards",
        label="Pass Yds",
        full_name="QB Passing Yards",
        description=(
            "Yards gained on completed passes, including yards after the catch. Yards lost on "
            "sacks are not subtracted."
        ),
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_pass_touchdowns",
        label="Pass TDs",
        full_name="QB Passing Touchdowns",
        description="Touchdown passes thrown.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_interceptions",
        label="INTs",
        full_name="QB Interceptions",
        description="Passes intercepted by the defense.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_passing_epa",
        label="Pass EPA",
        full_name="QB Passing EPA",
        description=(
            "Expected points added (how much each play raised or lowered the offense's expected "
            "points) on pass attempts, sacks, and spikes, as nflverse's official passing EPA "
            "counts them. Scrambles count in rushing EPA instead."
        ),
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _volume(
        name="qb_spike_epa",
        label="Spike EPA",
        full_name="QB Spike EPA",
        description=(
            "Expected points added on the quarterback's spikes, throwing the ball into the ground "
            "to stop the clock. Passing EPA includes them; EPA per dropback and the QB rating "
            "leave them out, as a spike is a called incompletion, not a dropback."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _volume(
        name="wp_kept_dropback_share",
        label="Kept Dropbacks",
        full_name="Share of Dropbacks the Filter Keeps",
        description=(
            "The share of the quarterback's dropbacks the garbage-time filter keeps: those where "
            "the offense's chance to win before the snap was between the threshold and 100% minus "
            "it. 100% means none were dropped."
        ),
        shape="rate",
        polarity="neutral",
        source="D",
        denominator="dropbacks",
        since=1999,
        percent=True,
    ),
    _volume(
        name="qb_wp_bin_dropbacks",
        label="WP Bin Dropbacks",
        full_name="Dropbacks in Win-Probability Bin",
        description=(
            "The quarterback's dropbacks in one game within one 1-point band of win probability "
            "(how far from decided the game was before the snap). Added across all bands, they "
            "equal his dropbacks in that game."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _volume(
        name="qb_wp_bin_epa",
        label="WP Bin Pass EPA",
        full_name="Passing EPA in Win-Probability Bin",
        description=(
            "Play-by-play passing EPA on the quarterback's dropbacks in one game within one "
            "1-point band of win probability. The garbage-time filter adds up the bands it keeps."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
)

QB_EFFICIENCY_METRICS: tuple[MetricDef, ...] = (
    _efficiency(
        name="qb_epa_per_dropback",
        label="EPA/Dropback",
        full_name="QB EPA Per Dropback",
        description=(
            "Expected points added (how much each play raised or lowered the offense's expected "
            "points) per dropback, before any adjustment for opponents. Spikes and kneel-downs "
            "are not dropbacks and are left out. League average is usually a little above 0."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
        formula="(Passing EPA - spike EPA) ÷ dropbacks (pass attempts + sacks)",
    ),
    _efficiency(
        name="qb_pass_yards_per_dropback",
        label="Pass Yds/Dropback",
        full_name="QB Passing Yards Per Dropback",
        description=(
            "Passing yards per dropback. Unlike yards per attempt, it counts each sack as a play "
            "with no gain; the yards lost on sacks are not subtracted."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
        formula="Passing yards ÷ (pass attempts + sacks)",
    ),
    _efficiency(
        name="qb_td_int_margin_rate",
        label="TD-INT/Dropback",
        full_name="QB TD-INT Margin Per Dropback",
        description=(
            "Touchdown passes minus interceptions, per dropback: a per-play version of TD-INT "
            "differential. Typical qualifying starters land between 0 and +0.03."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="dropbacks",
        since=1999,
        formula="(TD passes - interceptions) ÷ (pass attempts + sacks)",
    ),
    _efficiency(
        name="qb_any_a",
        label="ANY/A",
        full_name="QB Adjusted Net Yards Per Attempt",
        description=(
            "Yards per pass play with bonuses and penalties: +20 yards for each touchdown pass, "
            "-45 for each interception, and sacks counted as plays with their lost yards "
            "subtracted."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts + sacks",
        since=1999,
        formula=(
            "(Passing yards + 20 x TD passes - 45 x interceptions - sack yards lost) ÷ (pass "
            "attempts + sacks)"
        ),
        note=None,
    ),
    _efficiency(
        name="qb_completion_percentage_above_expectation",
        label="CPOE",
        full_name="QB Completion Percentage Above Expectation",
        description=(
            "How much higher the completion rate was than an average passer's on the same throws, "
            "given how hard each was, in percentage points (0 = as expected)."
        ),
        shape="avg",
        polarity="higher",
        source="PLS",
        denominator="pass attempts (model-expected completions)",
        since=2006,
        formula=(
            "Actual minus expected completion chance on each pass attempt, averaged over the "
            "attempts, in percentage points"
        ),
    ),
    _efficiency(
        name="qb_passer_rating",
        label="Passer Rating",
        full_name="QB Passer Rating",
        description=(
            "The NFL's passer rating (0 to 158.3), built from completion %, yards, touchdowns, and "
            "interceptions per attempt."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="official NFL formula over attempts",
        since=1999,
        note=None,
        formula=(
            "Four parts, each held between 0 and 2.375: (completions ÷ attempts - 0.3) x 5, (yards "
            "÷ attempts - 3) x 0.25, TD passes ÷ attempts x 20, and 2.375 - interceptions ÷ "
            "attempts x 25. Rating = their sum ÷ 6 x 100."
        ),
    ),
    _efficiency(
        name="qb_yards_per_attempt",
        label="Y/A",
        full_name="QB Yards Per Attempt",
        description="Passing yards divided by official pass attempts.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts",
        since=1999,
    ),
    _efficiency(
        name="qb_touchdown_rate",
        label="TD %",
        full_name="QB Touchdown Rate",
        description="The share of pass attempts that scored touchdowns.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts",
        since=1999,
        percent=True,
    ),
    _efficiency(
        name="qb_interception_rate",
        label="INT %",
        full_name="QB Interception Rate",
        description="The share of pass attempts that were intercepted.",
        shape="rate",
        polarity="lower",
        source="D",
        denominator="pass attempts",
        since=1999,
        percent=True,
    ),
    _efficiency(
        name="qb_completion_pct",
        label="Comp %",
        full_name="QB Completion Percentage",
        description="Completions divided by official pass attempts.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="pass attempts",
        since=1999,
        percent=True,
    ),
)

QB_PRESSURE_METRICS: tuple[MetricDef, ...] = (
    _pressure(
        name="qb_sacks",
        label="Sacks",
        full_name="QB Sacks Taken",
        description="Times the quarterback was sacked. Avoiding sacks is a QB skill.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
    _pressure(
        name="qb_sack_yards_lost",
        label="Sack Yds Lost",
        full_name="QB Sack Yards Lost",
        description="Yards lost on sacks, shown as a positive number.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
        note=None,
    ),
    _pressure(
        name="qb_sack_rate",
        label="Sack Rate",
        full_name="QB Sack Rate",
        description=(
            "The share of dropbacks (pass attempts plus sacks) that ended in a sack. Sack "
            "avoidance tracks quarterbacks more than offensive lines."
        ),
        shape="rate",
        polarity="lower",
        source="D",
        denominator="dropbacks",
        since=1999,
        percent=True,
    ),
    _pressure(
        name="qb_sack_fumbles_lost",
        label="Sack Fumbles Lost",
        full_name="QB Sack Fumbles Lost",
        description="Strip-sack fumbles the defense recovered.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
    _pressure(
        name="qb_scramble_rate",
        label="Scramble %",
        full_name="QB Scramble Rate",
        description=(
            "How often the quarterback took off running on a called pass play: scrambles per "
            "dropback."
        ),
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="dropbacks",
        since=1999,
        percent=True,
        formula="Scrambles ÷ (pass attempts + sacks)",
    ),
)

QB_RUSHING_METRICS: tuple[MetricDef, ...] = (
    _rushing(
        name="qb_carries",
        label="Carries",
        full_name="QB Carries",
        description="Official rushing attempts, including scrambles and kneel-downs.",
        shape="count",
        polarity="neutral",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_yards",
        label="Rush Yds",
        full_name="QB Rushing Yards",
        description="Rushing yards, scrambles and kneel-downs included.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_yards_per_carry",
        label="Rush Yds/Carry",
        full_name="QB Yards Per Carry",
        description=(
            "Rushing yards per carry, scrambles and kneel-downs included, so clock-killing kneels "
            "pull it down."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="carries",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_tds",
        label="Rush TDs",
        full_name="QB Rushing Touchdowns",
        description="Touchdowns scored on the ground.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_first_downs",
        label="Rush 1st Downs",
        full_name="QB Rushing First Downs",
        description="First downs gained on quarterback runs, scrambles included.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_epa",
        label="Rush EPA",
        full_name="QB Rushing EPA",
        description=(
            "Expected points added on runs, scrambles and kneel-downs included. A quarterback's "
            "scrambling value shows up here, not in passing EPA."
        ),
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _rushing(
        name="qb_epa_per_carry",
        label="EPA/Carry",
        full_name="QB EPA Per Carry",
        description="Rushing EPA per carry, scrambles and kneel-downs included.",
        shape="rate",
        polarity="higher",
        source="D",
        denominator="carries",
        since=1999,
    ),
    _rushing(
        name="qb_designed_carries",
        label="Designed Carries",
        full_name="QB Designed Carries",
        description=(
            "Runs called for the quarterback, such as sneaks and option keepers. Scrambles, "
            "kneel-downs, and two-point tries are left out."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_designed_rush_yards",
        label="Designed Rush Yds",
        full_name="QB Designed Rush Yards",
        description=(
            "Yards gained on runs called for the quarterback (scrambles, kneel-downs, and "
            "two-point tries left out)."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_designed_rush_epa",
        label="Designed Rush EPA",
        full_name="QB Designed Rush EPA",
        description=(
            "Expected points added on runs called for the quarterback (scrambles, kneel-downs, and "
            "two-point tries left out)."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        formula=(None),
    ),
    _rushing(
        name="qb_designed_yards_per_carry",
        label="Designed Yds/Carry",
        full_name="QB Designed-Rush Yards Per Carry",
        description=(
            "Average yards on runs called for the quarterback (scrambles and kneel-downs left out)."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="designed carries",
        since=1999,
    ),
    _rushing(
        name="qb_designed_epa_per_carry",
        label="Designed EPA/Carry",
        full_name="QB Designed-Rush EPA Per Carry",
        description=(
            "Average EPA on runs called for the quarterback (scrambles and kneel-downs left out)."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="designed carries",
        since=1999,
    ),
    _rushing(
        name="qb_scrambles",
        label="Scrambles",
        full_name="QB Scrambles",
        description="Pass plays on which the quarterback took off and ran instead of throwing.",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_scramble_yards",
        label="Scramble Yds",
        full_name="QB Scramble Yards",
        description="Yards gained on scrambles.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_yards_per_scramble",
        label="Yds/Scramble",
        full_name="QB Yards Per Scramble",
        description="Average yards gained per scramble.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="scrambles",
        since=1999,
    ),
    _rushing(
        name="qb_kneels",
        label="Kneels",
        full_name="QB Kneel-Downs",
        description=(
            "Kneel-downs to run out the clock. Counted in carries and the per-carry rates, "
            "left out of the designed-run stats."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _rushing(
        name="qb_rushing_2pt_conversions",
        label="2-Pt Rush Conv",
        full_name="QB Rushing Two-Point Conversions",
        description="Successful two-point conversions run in.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
)

QB_CLUTCH_METRICS: tuple[MetricDef, ...] = (
    _clutch(
        name="qb_wins",
        label="QB Wins",
        full_name="QB Wins",
        description=(
            "Team wins in games where he was the main quarterback (most snaps, or most dropbacks "
            "before snap counts). A team result shown for context; it never feeds a rating."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_losses",
        label="QB Losses",
        full_name="QB Losses",
        description=(
            "Team losses in games where he was the main quarterback (most snaps, or most dropbacks "
            "before snap counts)."
        ),
        shape="count",
        polarity="lower",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_ties",
        label="QB Ties",
        full_name="QB Ties",
        description=(
            "Ties in games where he was the main quarterback (most snaps, or most dropbacks before "
            "snap counts)."
        ),
        shape="count",
        polarity="neutral",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_win_pct",
        label="QB Win %",
        full_name="QB Win Percentage",
        description=(
            "Share of primary-QB games won, counting a tie as half a win; empty for a "
            "quarterback who was never the primary passer. A team outcome, shown for context — "
            "never a rating input."
        ),
        shape="rate",
        polarity="higher",
        source="D",
        denominator="primary-QB games",
        since=1999,
        percent=True,
    ),
    _clutch(
        name="qb_fourth_quarter_comeback",
        label="4th-Qtr Comeback",
        full_name="QB Fourth-Quarter Comeback",
        description=(
            "1 if the team trailed in the fourth quarter or overtime and came back to win; the "
            "credit goes to the game's main quarterback. Any deficit counts, not just one score."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_fourth_quarter_comebacks",
        label="4th-Qtr Comebacks",
        full_name="QB Fourth-Quarter Comebacks",
        description=(
            "Wins in which the team trailed in the fourth quarter or overtime, counted in games "
            "where he was the main quarterback. Any deficit counts, not just one score."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_game_winning_drive",
        label="Game-Winning Drive",
        full_name="QB Game-Winning Drive",
        description=(
            "1 if the team, in a game it won, scored to go from tied or behind to ahead in the "
            "fourth quarter or overtime; the credit goes to the game's main quarterback."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _clutch(
        name="qb_game_winning_drives",
        label="Game-Winning Drives",
        full_name="QB Game-Winning Drives",
        description=(
            "Wins in which the team scored to go from tied or behind to ahead in the fourth "
            "quarter or overtime, counted in games where he was the main quarterback."
        ),
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
)

QB_TURNOVER_METRICS: tuple[MetricDef, ...] = (
    _turnovers(
        name="qb_td_int_differential",
        label="TD-INT Diff",
        full_name="QB Touchdown-Interception Differential",
        description="Touchdown passes minus interceptions across the season.",
        shape="count",
        polarity="higher",
        source="D",
        since=1999,
    ),
    _turnovers(
        name="qb_rushing_fumbles",
        label="Rush Fumbles",
        full_name="QB Rushing Fumbles",
        description="Fumbles on quarterback runs, whether or not lost.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
    _turnovers(
        name="qb_rushing_fumbles_lost",
        label="Rush Fumbles Lost",
        full_name="QB Rushing Fumbles Lost",
        description="Fumbles on quarterback runs that the defense recovered.",
        shape="count",
        polarity="lower",
        source="PLS",
        since=1999,
    ),
)

QB_METRICS: tuple[MetricDef, ...] = (
    QB_RATING_METRICS
    + QB_IDENTITY_METRICS
    + QB_VOLUME_METRICS
    + QB_EFFICIENCY_METRICS
    + QB_PRESSURE_METRICS
    + QB_RUSHING_METRICS
    + QB_CLUTCH_METRICS
    + QB_TURNOVER_METRICS
)
