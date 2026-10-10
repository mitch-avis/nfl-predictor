"""Team offense definitions other than passing and receiving.

Totals, rushing, scoring, downs, drives, turnovers, and penalties.

One part of the team registry that `team_metrics.TEAM_METRICS` assembles in catalog order;
human-readable companion: [docs/stats-catalog.md](../../docs/stats-catalog.md).
"""

from __future__ import annotations

from nfl_predictor.season_stats.metrics.schema import MetricDef, section

_off_total = section("team", "Offense", "Total")
_off_rush = section("team", "Offense", "Rushing")
_off_score = section("team", "Offense", "Scoring")
_off_downs = section("team", "Offense", "Downs & Conversions")
_off_drives = section("team", "Offense", "Drives & Field Position")
_off_to = section("team", "Offense", "Turnovers")
_off_pen = section("team", "Offense", "Penalties")

OFFENSE_TOTAL_METRICS: tuple[MetricDef, ...] = (
    _off_total(
        name="offensive_snaps",
        label="Off Snaps",
        full_name="Offensive Snaps",
        description=(
            "Plays the offense ran from scrimmage: dropbacks (pass attempts, sacks, and "
            "scrambles), designed runs, kneel-downs, and spikes. Punts, kicks, and kickoffs are "
            "not included."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_total(
        name="total_yards",
        label="Total Yds",
        full_name="Total Yards",
        description=(
            "Passing yards plus rushing yards. Yards lost on sacks are not subtracted, so this "
            "runs higher than the NFL's official total yards."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
        note=None,
    ),
    _off_total(
        name="yards_per_offensive_snap",
        label="Yds/Off Snap",
        full_name="Yards Per Offensive Snap",
        description=(
            "Average yards gained per offensive snap, the simplest measure of how well an offense "
            "moves the ball."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="offensive snaps",
        since=1999,
        formula="(Passing yards + rushing yards) ÷ offensive snaps; sack losses are not subtracted",
    ),
    _off_total(
        name="first_downs",
        label="First Downs",
        full_name="First Downs",
        description="First downs gained by run, pass, or a penalty on the defense.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_total(
        name="scrimmage_tds",
        label="Scrimmage TDs",
        full_name="Scrimmage Touchdowns",
        description="Touchdowns scored by the offense: passing plus rushing touchdowns.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_total(
        name="offensive_epa",
        label="Off EPA",
        full_name="Offensive Expected Points Added",
        description=(
            "EPA (expected points added) on the offense's snaps. EPA measures how much a play "
            "raised or lowered the offense's expected points, given down, distance, and field "
            "position, so it credits the situation, not just yards."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_total(
        name="epa_per_offensive_snap",
        label="EPA/Off Snap",
        full_name="Expected Points Added Per Offensive Snap",
        description=(
            "Average EPA (expected points added) per offensive snap, the core measure of offensive "
            "efficiency. Team seasons usually fall between about -0.2 and +0.1."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="offensive snaps",
        since=1999,
    ),
    _off_total(
        name="success_rate",
        label="Success %",
        full_name="Offensive Success Rate",
        description=(
            "Share of offensive snaps with positive EPA, meaning the play left the offense better "
            "placed to score than before. It rewards consistency and ignores how big each gain "
            "was."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="offensive snaps",
        since=1999,
        percent=True,
    ),
    _off_total(
        name="explosive_play_rate",
        label="Explosive %",
        full_name="Explosive Play Rate",
        description=(
            "Share of offensive snaps that were big plays: completions of 20 or more yards, or "
            "carries of 10 or more yards with scrambles included."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="offensive snaps",
        since=1999,
        percent=True,
        formula="(Completions of 20+ yards + carries of 10+ yards) ÷ offensive snaps",
    ),
    _off_total(
        name="no_huddle_rate",
        label="No-Huddle %",
        full_name="No-Huddle Rate",
        description="Share of offensive snaps run without a huddle. A style stat, not a grade.",
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="offensive snaps",
        since=2003,
        note=(
            "Blank before 2003, when nflverse flags almost no snap as no-huddle. The flag follows "
            "the play text, which marks no-huddle snaps more often in later seasons, so compare "
            "teams within a season."
        ),
        percent=True,
    ),
    _off_total(
        name="shotgun_rate",
        label="Shotgun %",
        full_name="Shotgun Rate",
        description=(
            "Share of offensive snaps taken from the shotgun formation. A style stat, not a grade."
        ),
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="offensive snaps",
        since=1999,
        percent=True,
    ),
    _off_total(
        name="pass_rate",
        label="Pass %",
        full_name="Pass Rate",
        description=(
            "Share of offensive snaps that were dropbacks (pass attempts, sacks, and scrambles): "
            "how pass-heavy the offense is."
        ),
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="offensive snaps",
        since=1999,
        percent=True,
    ),
    _off_total(
        name="early_down_pass_rate",
        label="Early-Down Pass %",
        full_name="Early-Down Pass Rate",
        description=(
            "Share of first- and second-down snaps that were dropbacks: how pass-first the "
            "play-calling is before third down."
        ),
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="early-down snaps",
        since=1999,
        percent=True,
    ),
    _off_total(
        name="pass_rate_over_expected",
        label="Pass Rate Over Exp",
        full_name="Pass Rate Over Expected",
        description=(
            "How much more (or less) often the offense dropped back to pass than a model expects "
            "for the situation (down, distance, field position, score, and time), in percentage "
            "points. Available from 2006."
        ),
        shape="avg",
        polarity="neutral",
        source="PBP",
        denominator="offensive snaps (model-expected pass rate)",
        since=2006,
        formula=(
            "Average over offensive snaps of 100 x (1 for a dropback, else 0, minus the model's "
            "dropback chance)"
        ),
    ),
    _off_total(
        name="offensive_wpa",
        label="Off Win Prob Added",
        full_name="Offensive Win Probability Added",
        description=(
            "Change in the team's chance of winning from its offensive snaps, as a fraction: +0.10 "
            "means 10 percentage points of win probability gained. Plays late in close games move "
            "it most."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
)

OFFENSE_RUSHING_METRICS: tuple[MetricDef, ...] = (
    _off_rush(
        name="rushing_yards",
        label="Rush Yds",
        full_name="Rushing Yards",
        description="Yards gained on carries, including quarterback scrambles and kneel-downs.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_rush(
        name="rushing_epa",
        label="Rush EPA",
        full_name="Rushing Expected Points Added",
        description=(
            "EPA (expected points added) on carries, including quarterback scrambles and "
            "kneel-downs. Most teams finish below zero, because the average run loses a little "
            "expected value."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_rush(
        name="rushing_tds",
        label="Rush TDs",
        full_name="Rushing Touchdowns",
        description="Touchdowns scored on the ground.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_rush(
        name="rushing_first_downs",
        label="Rush 1st Downs",
        full_name="Rushing First Downs",
        description="First downs gained on the ground.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_rush(
        name="rushing_fumbles_lost",
        label="Rush Fumbles Lost",
        full_name="Rushing Fumbles Lost",
        description="Fumbles lost to the defense on rushing plays.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_rush(
        name="carries",
        label="Carries",
        full_name="Carries",
        description="Official rushing attempts, including scrambles and kneel-downs.",
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=1999,
    ),
    _off_rush(
        name="designed_carries",
        label="Designed Carries",
        full_name="Designed Carries",
        description=(
            "Called running plays: carries that were not quarterback scrambles or kneel-downs."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_rush(
        name="yards_per_carry",
        label="Yds/Carry",
        full_name="Yards Per Carry",
        description="Average yards per carry, including quarterback scrambles and kneel-downs.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="carries",
        since=1999,
    ),
    _off_rush(
        name="epa_per_carry",
        label="EPA/Carry",
        full_name="Expected Points Added Per Carry",
        description=(
            "Average EPA (expected points added) per carry, scrambles and kneel-downs included: "
            "how much value the running game produced per attempt."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="carries",
        since=1999,
    ),
    _off_rush(
        name="rush_success_rate",
        label="Rush Success %",
        full_name="Rushing Success Rate",
        description=(
            "Share of called running plays with positive EPA, meaning the run left the offense "
            "better placed to score. Scrambles and kneel-downs are left out."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="designed carries",
        since=1999,
        percent=True,
    ),
    _off_rush(
        name="explosive_rush_rate",
        label="Explosive Rush %",
        full_name="Explosive Rush Rate",
        description="Share of carries that gained 10 or more yards, scrambles included.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="carries",
        since=1999,
        percent=True,
    ),
    _off_rush(
        name="stuffed_run_rate",
        label="Stuffed %",
        full_name="Stuffed Run Rate",
        description=("The share of carries other than kneel-downs stopped for no gain or a loss."),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="carries other than kneel-downs",
        since=1999,
        percent=True,
    ),
    _off_rush(
        name="rushing_fumbles",
        label="Rush Fumbles",
        full_name="Rushing Fumbles",
        description="Fumbles on rushing plays, whether or not the team lost the ball.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_rush(
        name="longest_rush",
        label="Longest Rush",
        full_name="Longest Rush",
        description=(
            "Longest single carry, in yards, scrambles included. Game logs show the game's "
            "longest; the season table shows the season's longest."
        ),
        shape="max",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_rush(
        name="rushing_2pt_conversions",
        label="2-Pt Rush Conv",
        full_name="Rushing Two-Point Conversions",
        description="Successful two-point conversions run in.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
)

OFFENSE_SCORING_METRICS: tuple[MetricDef, ...] = (
    _off_score(
        name="total_tds",
        label="Total TDs",
        full_name="Total Touchdowns",
        description="All touchdowns the team scored: offense, defense, and special teams.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_score(
        name="offensive_tds",
        label="Off TDs",
        full_name="Offensive Touchdowns",
        description="Passing plus rushing touchdowns.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        duplicate_of="scrimmage_tds",
    ),
    _off_score(
        name="points_per_drive",
        label="Pts/Drive",
        full_name="Points Per Drive",
        description=(
            "Average points the offense scored per drive, counting the extra point or two-point "
            "try after a touchdown."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="drives",
        since=1999,
    ),
    _off_score(
        name="red_zone_trips",
        label="Red Zone Trips",
        full_name="Red Zone Trips",
        description="Drives that got inside the opponent's 20-yard line.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_score(
        name="red_zone_td_pct",
        label="Red Zone TD %",
        full_name="Red Zone Touchdown Percentage",
        description=(
            "Share of red-zone trips (drives that got inside the opponent's 20) that ended in a "
            "touchdown."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="red-zone trips",
        since=1999,
        percent=True,
    ),
    _off_score(
        name="points_per_red_zone_trip",
        label="Pts/Red Zone Trip",
        full_name="Points Per Red Zone Trip",
        description=(
            "Average points scored on drives that got inside the opponent's 20, counting the extra "
            "point or two-point try."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="red-zone trips",
        since=1999,
    ),
    _off_score(
        name="goal_to_go_td_pct",
        label="Goal-to-Go TD %",
        full_name="Goal-to-Go Touchdown Percentage",
        description=(
            "Share of goal-to-go series (first-and-goal and the downs after it) that ended in a "
            "touchdown."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="goal-to-go series",
        since=1999,
        percent=True,
    ),
    _off_score(
        name="two_pt_attempts",
        label="2-Pt Att",
        full_name="Two-Point Attempts",
        description="Two-point conversion tries.",
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=1999,
    ),
    _off_score(
        name="two_pt_conversions",
        label="2-Pt Made",
        full_name="Two-Point Conversions",
        description="Successful two-point conversions.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _off_score(
        name="two_pt_conversion_rate",
        label="2-Pt %",
        full_name="Two-Point Conversion Rate",
        description="Share of two-point tries that succeeded.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="two-point attempts",
        since=1999,
        percent=True,
    ),
)

OFFENSE_DOWNS_METRICS: tuple[MetricDef, ...] = (
    _off_downs(
        name="first_downs_penalty",
        label="Penalty 1st Downs",
        full_name="First Downs by Penalty",
        description="First downs the offense gained because of a penalty on the defense.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_downs(
        name="third_down_attempts",
        label="3rd Down Att",
        full_name="Third Down Attempts",
        description="Third downs faced by the offense.",
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_downs(
        name="third_down_conversions",
        label="3rd Down Conv",
        full_name="Third Down Conversions",
        description="Third downs converted into a first down or touchdown.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_downs(
        name="third_down_pct",
        label="3rd Down %",
        full_name="Third Down Conversion Percentage",
        description="Share of third downs the offense converted into a first down or touchdown.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="third-down attempts",
        since=1999,
        percent=True,
    ),
    _off_downs(
        name="third_down_avg_distance",
        label="3rd Down Dist",
        full_name="Average Third Down Distance",
        description=(
            "Average yards to go on third down. A short third down usually means the offense "
            "gained good yardage on first and second down."
        ),
        shape="avg",
        polarity="lower",
        source="PBP",
        denominator="third downs faced",
        since=1999,
    ),
    _off_downs(
        name="fourth_down_attempts",
        label="4th Down Att",
        full_name="Fourth Down Attempts",
        description=(
            "Fourth downs on which the offense ran a play instead of punting or kicking a field "
            "goal."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_downs(
        name="fourth_down_conversions",
        label="4th Down Conv",
        full_name="Fourth Down Conversions",
        description="Fourth-down tries that gained a first down or touchdown.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _off_downs(
        name="fourth_down_pct",
        label="4th Down %",
        full_name="Fourth Down Conversion Percentage",
        description="Share of fourth-down tries the offense converted.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="fourth-down attempts",
        since=1999,
        percent=True,
    ),
    _off_downs(
        name="fourth_down_go_rate",
        label="4th Down Go %",
        full_name="Fourth Down Go Rate",
        description=(
            "How often the offense went for it on fourth down instead of punting or kicking a "
            "field goal. A coaching-style stat, not a grade."
        ),
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="fourth downs faced",
        since=1999,
        percent=True,
        formula="Fourth-down tries ÷ fourth downs faced (tries, punts, and field-goal attempts)",
    ),
    _off_downs(
        name="fourth_down_aggressiveness",
        label="4th & Short Go %",
        full_name="Fourth-and-Short Go-for-It Rate",
        description=(
            "How often the offense went for it on fourth down with 2 or fewer yards to go. "
            "Analytics research generally favors going for it in these spots."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="fourth-and-short situations",
        since=1999,
        percent=True,
        formula=(
            "Fourth-down tries with 2 or fewer yards to go ÷ fourth downs faced with 2 or fewer "
            "yards to go (tries, punts, and field-goal attempts)"
        ),
    ),
    _off_downs(
        name="series",
        label="Series",
        full_name="Offensive Series",
        description=(
            "Sets of downs the offense started: each possession and each new first down begins one."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_downs(
        name="series_conversion_rate",
        label="Series Conv %",
        full_name="Series Conversion Rate",
        description=(
            "Share of the offense's sets of downs that earned a new first down or a touchdown. "
            "Steadier than third-down rate because every set of downs counts."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="series",
        since=1999,
        percent=True,
    ),
    _off_downs(
        name="three_and_out_rate",
        label="3-and-Out %",
        full_name="Three-and-Out Rate",
        description="Share of drives that ended in a punt without gaining a first down.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="drives",
        since=1999,
        percent=True,
    ),
    _off_downs(
        name="turnovers_on_downs",
        label="Turnovers on Downs",
        full_name="Turnovers on Downs",
        description="Failed fourth-down tries that handed the ball over.",
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
)

OFFENSE_DRIVES_METRICS: tuple[MetricDef, ...] = (
    _off_drives(
        name="drives",
        label="Drives",
        full_name="Offensive Drives",
        description=(
            "Possessions the offense had, from taking over the ball until it scored, gave the ball "
            "up, or the half ended."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _off_drives(
        name="yards_per_drive",
        label="Yds/Drive",
        full_name="Yards Per Drive",
        description=(
            "Average net yards per drive, counting yards lost on sacks and yards gained or lost "
            "through penalties."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="drives",
        since=1999,
    ),
    _off_drives(
        name="plays_per_drive",
        label="Plays/Drive",
        full_name="Plays Per Drive",
        description="Average number of offensive plays per drive.",
        shape="avg",
        polarity="neutral",
        source="PBP",
        denominator="drives",
        since=1999,
    ),
    _off_drives(
        name="time_per_drive",
        label="Time/Drive",
        full_name="Time Per Drive",
        description="Average game-clock time per drive, in seconds (180 = 3 minutes).",
        shape="avg",
        polarity="neutral",
        source="PBP",
        denominator="drives",
        since=1999,
    ),
    _off_drives(
        name="first_downs_per_drive",
        label="1st Downs/Drive",
        full_name="First Downs Per Drive",
        description="Average first downs gained per possession.",
        shape="avg",
        polarity="higher",
        source="PBP",
        denominator="drives",
        since=1999,
    ),
    _off_drives(
        name="score_pct_per_drive",
        label="Score %/Drive",
        full_name="Scoring Rate Per Drive",
        description="Share of drives that ended in a touchdown or field goal.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="drives",
        since=1999,
        percent=True,
    ),
    _off_drives(
        name="punt_pct_per_drive",
        label="Punt %/Drive",
        full_name="Punt Rate Per Drive",
        description="Share of drives that ended in a punt.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="drives",
        since=1999,
        percent=True,
    ),
    _off_drives(
        name="turnover_pct_per_drive",
        label="Turnover %/Drive",
        full_name="Turnover Rate Per Drive",
        description=(
            "Share of drives that ended in a giveaway: an interception or a lost fumble, including "
            "those returned for a touchdown."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="drives",
        since=1999,
        percent=True,
    ),
    _off_drives(
        name="avg_starting_field_position",
        label="Avg Drive Start",
        full_name="Average Starting Field Position",
        description=(
            "Average yard line where drives started, counted from the team's own goal line: 25 "
            "means its own 25, 50 means midfield, 60 means the opponent's 40."
        ),
        shape="avg",
        polarity="higher",
        source="PBP",
        denominator="drives",
        since=1999,
    ),
    _off_drives(
        name="long_field_score_pct",
        label="Long-Field Score %",
        full_name="Long-Field Scoring Rate",
        description=(
            "Share of drives starting at or inside the team's own 25 that ended in a touchdown or "
            "field goal."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="long-field drives",
        since=1999,
        percent=True,
    ),
    _off_drives(
        name="drive_penalty_yards",
        label="Net Drive Pen Yds",
        full_name="Net Penalty Yards on Drives",
        description=(
            "Net penalty yards on the team's drives: yards the defense's fouls gave the offense "
            "minus yards its own fouls cost it."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=2001,
    ),
)

OFFENSE_TURNOVER_METRICS: tuple[MetricDef, ...] = (
    _off_to(
        name="giveaways",
        label="Giveaways",
        full_name="Giveaways",
        description="Interceptions thrown plus fumbles lost on offensive snaps.",
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
    _off_to(
        name="fumbles",
        label="Fumbles",
        full_name="Fumbles",
        description="All offensive fumbles, whether recovered or lost.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_to(
        name="fumbles_lost",
        label="Fumbles Lost",
        full_name="Fumbles Lost",
        description="Offensive fumbles the defense recovered.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_to(
        name="giveaway_rate_per_offensive_snap",
        label="Giveaway %",
        full_name="Giveaways Per Offensive Snap",
        description="Share of offensive snaps that ended in an interception or lost fumble.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="offensive snaps",
        since=1999,
        percent=True,
    ),
    _off_to(
        name="giveaways_per_drive",
        label="Giveaways/Drive",
        full_name="Giveaways Per Drive",
        description="Average giveaways per drive.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="drives",
        since=1999,
    ),
    _off_to(
        name="turnover_epa",
        label="Turnover EPA",
        full_name="Expected Points Added on Giveaways",
        description=(
            "EPA (expected points added), from the team's side, on its interceptions and lost "
            "fumbles, including muffs and fumbles on kickoff and punt returns: how costly its "
            "giveaways were, not just how many. Almost always negative."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        note=None,
    ),
)

OFFENSE_PENALTY_METRICS: tuple[MetricDef, ...] = (
    _off_pen(
        name="penalties",
        label="Penalties",
        full_name="Penalties",
        description="Accepted penalties called on the team on offense, defense, and special teams.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pen(
        name="penalty_yards",
        label="Penalty Yds",
        full_name="Penalty Yards",
        description=(
            "Yards marked off against the team on its accepted penalties, across all units."
        ),
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _off_pen(
        name="offensive_penalties",
        label="Off Penalties",
        full_name="Offensive Penalties",
        description=(
            "Penalties committed on plays where the team had the ball, including its punts, "
            "field goals, extra points, and kickoff returns."
        ),
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
    _off_pen(
        name="offensive_penalty_yards",
        label="Off Pen Yds",
        full_name="Offensive Penalty Yards",
        description=(
            "Penalty yards assessed on plays where the team had the ball, including its punts, "
            "field goals, extra points, and kickoff returns."
        ),
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
    _off_pen(
        name="presnap_penalty_rate",
        label="Pre-Snap Pen %",
        full_name="Pre-Snap Penalty Rate",
        description=(
            "False starts and delay-of-game penalties per offensive snap, a measure of pre-snap "
            "discipline. Other pre-snap fouls, such as illegal formation, are not counted."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="offensive snaps",
        since=1999,
        percent=True,
    ),
    _off_pen(
        name="penalty_rate_per_offensive_snap",
        label="Off Pen %",
        full_name="Offensive Penalty Rate",
        description=(
            "Offensive penalties (special-teams plays with the ball included) divided by "
            "offensive snaps."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="offensive snaps",
        since=1999,
        percent=True,
    ),
)

__all__ = [
    "OFFENSE_DOWNS_METRICS",
    "OFFENSE_DRIVES_METRICS",
    "OFFENSE_PENALTY_METRICS",
    "OFFENSE_RUSHING_METRICS",
    "OFFENSE_SCORING_METRICS",
    "OFFENSE_TOTAL_METRICS",
    "OFFENSE_TURNOVER_METRICS",
]
