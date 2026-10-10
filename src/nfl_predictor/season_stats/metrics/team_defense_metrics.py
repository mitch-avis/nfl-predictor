"""Team defense definitions.

What the defense allowed and the plays it made.

One part of the team registry that `team_metrics.TEAM_METRICS` assembles in catalog order;
human-readable companion: [docs/stats-catalog.md](../../docs/stats-catalog.md).
"""

from __future__ import annotations

from nfl_predictor.season_stats.metrics.schema import MetricDef, section

_def_total = section("team", "Defense", "Total")
_def_pass = section("team", "Defense", "Passing")
_def_rush = section("team", "Defense", "Rushing")
_def_recv = section("team", "Defense", "Receiving")
_def_score = section("team", "Defense", "Scoring")
_def_downs = section("team", "Defense", "Downs & Conversions")
_def_drives = section("team", "Defense", "Drives & Field Position")
_def_to = section("team", "Defense", "Turnovers")
_def_press = section("team", "Defense", "Pressure & Playmaking")
_def_pen = section("team", "Defense", "Penalties")

_TARGETS_GAP_NOTE = (
    "Blank in 2003-2008, when nflverse play-by-play names the intended receiver on almost no "
    "incomplete passes."
)
_QB_HITS_GAP_NOTE = (
    "Blank in 2003-2005, when nflverse records no QB hits; in 1999-2002 it records them only on "
    "sacks."
)
_TACKLES_FOR_LOSS_GAP_NOTE = (
    "Blank in 2003-2011, when nflverse's weekly player stats credit no tackles for loss."
)

DEFENSE_METRICS: tuple[MetricDef, ...] = (
    _def_total(
        name="defensive_snaps",
        label="Def Snaps",
        full_name="Defensive Snaps",
        description=(
            "Plays opponents ran from scrimmage against the defense: dropbacks (pass attempts, "
            "sacks, and scrambles), designed runs, kneel-downs, and spikes."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=1999,
    ),
    _def_total(
        name="total_yards_allowed",
        label="Total Yds Allowed",
        full_name="Total Yards Allowed",
        description=(
            "Passing plus rushing yards opponents gained against the defense. Passing yards here "
            "are before sack losses, so this runs higher than the NFL's official total yards."
        ),
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
        note=None,
    ),
    _def_pass(
        name="passing_yards_allowed",
        label="Pass Yds Allowed",
        full_name="Passing Yards Allowed",
        description=(
            "Passing yards opponents gained against the defense, before subtracting yards lost on "
            "sacks."
        ),
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _def_pass(
        name="passing_epa_allowed",
        label="Pass EPA Allowed",
        full_name="Passing EPA Allowed",
        description=(
            "Expected points added (EPA) on opponents' pass plays: how much those plays raised or "
            "lowered their expected points, given down, distance, and field position. League "
            "average is usually a little above zero."
        ),
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
        formula="Sum of opponent EPA on pass attempts and sacks (QB scrambles not included)",
    ),
    _def_pass(
        name="passing_tds_allowed",
        label="Pass TDs Allowed",
        full_name="Passing Touchdowns Allowed",
        description="Touchdown passes given up.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _def_pass(
        name="passing_first_downs_allowed",
        label="Pass 1Ds Allowed",
        full_name="Passing First Downs Allowed",
        description="First downs given up through the air.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _def_pass(
        name="passing_cpoe_allowed",
        label="CPOE Allowed",
        full_name="Completion Percentage Above Expectation Allowed",
        description=(
            "How much more often opponents completed passes than expected, given how hard each "
            "throw was, in percentage points. Below zero means they completed fewer than expected; "
            "league average is near zero."
        ),
        shape="avg",
        polarity="lower",
        source="PBP +TS",
        denominator="opponent pass attempts (model-expected completions)",
        since=2006,
        formula=(
            "Actual minus expected completion chance on each opponent pass attempt, averaged over "
            "the attempts, in percentage points"
        ),
    ),
    _def_rush(
        name="rushing_yards_allowed",
        label="Rush Yds Allowed",
        full_name="Rushing Yards Allowed",
        description=(
            "Rushing yards opponents gained against the defense, with quarterback scrambles and "
            "kneel-downs counted as official stats count them."
        ),
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _def_rush(
        name="rushing_epa_allowed",
        label="Rush EPA Allowed",
        full_name="Rushing EPA Allowed",
        description=(
            "Expected points added (EPA) on opponents' runs, including quarterback scrambles and "
            "kneel-downs. League average is below zero, because the average run loses a little "
            "expected value."
        ),
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _def_rush(
        name="rushing_tds_allowed",
        label="Rush TDs Allowed",
        full_name="Rushing Touchdowns Allowed",
        description="Rushing touchdowns given up.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _def_rush(
        name="rushing_first_downs_allowed",
        label="Rush 1Ds Allowed",
        full_name="Rushing First Downs Allowed",
        description="First downs given up on the ground.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _def_press(
        name="def_sacks",
        label="Def Sacks",
        full_name="Defensive Sacks",
        description="Times the defense sacked the opposing quarterback.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _def_press(
        name="def_qb_hits",
        label="Def QB Hits",
        full_name="Defensive QB Hits",
        description="Times the defense's players hit the opposing quarterback, sacks included.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
        note=_QB_HITS_GAP_NOTE,
        formula="Sum of QB hits credited to defenders (a hit shared by two defenders counts twice)",
    ),
    _def_press(
        name="def_tackles_for_loss",
        label="Def TFL",
        full_name="Defensive Tackles for Loss",
        description=(
            "Tackles behind the line of scrimmage credited to the defense's players. Most sacks "
            "also count as a tackle for loss."
        ),
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
        note=_TACKLES_FOR_LOSS_GAP_NOTE,
    ),
    _def_press(
        name="def_pass_defended",
        label="Passes Defended",
        full_name="Passes Defended",
        description=(
            "Passes the defense's players broke up, tipped, or intercepted. Interceptions count as "
            "passes defended."
        ),
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
        formula=(
            "Sum of passes defended credited to defenders (a pass defended by two players counts "
            "twice)"
        ),
    ),
    _def_press(
        name="def_fumbles_forced",
        label="Forced Fumbles",
        full_name="Defensive Fumbles Forced",
        description=(
            "Fumbles the defense's players forced, whether or not the defense recovered the ball."
        ),
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _def_press(
        name="def_safeties",
        label="Def Safeties",
        full_name="Defensive Safeties",
        description="Safeties forced by the defense (two points each).",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    _def_to(
        name="def_interceptions",
        label="Def INTs",
        full_name="Defensive Interceptions",
        description="Passes intercepted by the defense.",
        shape="count",
        polarity="higher",
        source="PLS",
        since=1999,
    ),
    # Planned defense expansion (mirrors and defense-only stats).
    _def_total(
        name="first_downs_allowed",
        label="1Ds Allowed",
        full_name="First Downs Allowed",
        description="Total first downs given up by rush, pass, or penalty.",
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
    _def_total(
        name="yards_per_defensive_snap_allowed",
        label="Yds/Snap Allowed",
        full_name="Yards Allowed Per Defensive Snap",
        description="Yards opponents gained per scrimmage play against the defense.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="defensive snaps",
        since=1999,
        formula="(Opponent passing yards before sack losses + rushing yards) ÷ defensive snaps",
    ),
    _def_total(
        name="epa_per_defensive_snap_allowed",
        label="EPA/Snap Allowed",
        full_name="EPA Allowed Per Defensive Snap",
        description=(
            "Expected points added (EPA) per opponent scrimmage play: how much a typical play "
            "raised or lowered their expected points. Team seasons usually fall between about "
            "-0.15 and +0.1."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="defensive snaps",
        since=1999,
    ),
    _def_total(
        name="success_rate_allowed",
        label="Success % Allowed",
        full_name="Success Rate Allowed",
        description=(
            "Share of opponent scrimmage plays that raised their expected points (positive EPA)."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="defensive snaps",
        since=1999,
        percent=True,
    ),
    _def_total(
        name="explosive_play_rate_allowed",
        label="Explosive % Allowed",
        full_name="Explosive Play Rate Allowed",
        description=(
            "Share of opponent scrimmage plays that were big gains: completions of 20+ yards or "
            "runs of 10+ yards."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="defensive snaps",
        since=1999,
        percent=True,
        formula="(Opponent completions of 20+ yards + carries of 10+ yards) ÷ defensive snaps",
    ),
    _def_pass(
        name="attempts_faced",
        label="Att Faced",
        full_name="Pass Attempts Faced",
        description=(
            "Passes opponents threw against the defense. Sacks and two-point tries are not counted "
            "as attempts."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=1999,
    ),
    _def_pass(
        name="completions_allowed",
        label="Comp Allowed",
        full_name="Completions Allowed",
        description="Opponent passes completed against this defense.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
    ),
    _def_pass(
        name="completion_pct_allowed",
        label="Comp % Allowed",
        full_name="Completion Percentage Allowed",
        description="The share of opponent attempts completed against this defense.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent pass attempts",
        since=1999,
        percent=True,
    ),
    _def_pass(
        name="net_passing_yards_allowed",
        label="Net Pass Yds Allowed",
        full_name="Net Passing Yards Allowed",
        description="Passing yards allowed minus opponent sack yardage lost.",
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
    _def_pass(
        name="epa_per_dropback_allowed",
        label="EPA/Dropback Allowed",
        full_name="EPA Per Dropback Allowed",
        description=(
            "Expected points added (EPA) per opponent dropback (pass attempt, sack, or scramble): "
            "how much a typical pass play raised or lowered their expected points. Team seasons "
            "usually fall between about -0.15 and +0.2."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent dropbacks",
        since=1999,
    ),
    _def_pass(
        name="any_a_allowed",
        label="ANY/A Allowed",
        full_name="Adjusted Net Yards Per Attempt Allowed",
        description=(
            "Opposing passers' yards per pass play after adding a bonus for touchdowns and "
            "subtracting for interceptions and sack losses."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent pass attempts + sacks",
        since=1999,
        formula=(
            "(Opponent passing yards + 20 x TD passes - 45 x interceptions - sack yards lost) ÷ "
            "(opponent pass attempts + sacks)"
        ),
    ),
    _def_pass(
        name="explosive_pass_rate_allowed",
        label="Expl Pass % Allowed",
        full_name="Explosive Pass Rate Allowed",
        description="Share of opponent dropbacks that ended in a completion of 20+ yards.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent dropbacks",
        since=1999,
        percent=True,
    ),
    _def_pass(
        name="air_yards_allowed",
        label="Air Yds Allowed",
        full_name="Air Yards Allowed",
        description=(
            "How far opponents' passes traveled past the line of scrimmage, added up over every "
            "attempt, caught or not. Shows how far downfield opponents threw."
        ),
        shape="count",
        polarity="neutral",
        source="PBP",
        since=2006,
    ),
    _def_pass(
        name="yac_allowed",
        label="YAC Allowed",
        full_name="Yards After Catch Allowed",
        description=(
            "Yards opponents' receivers gained after the catch on completed passes; a rough read "
            "on tackling and pursuit."
        ),
        shape="count",
        polarity="lower",
        source="PBP",
        since=2006,
    ),
    _def_pass(
        name="team_passer_rating_allowed",
        label="Passer Rtg Allowed",
        full_name="Passer Rating Allowed",
        description=(
            "The NFL passer rating of all opposing quarterbacks combined, on the usual 0 to 158.3 "
            "scale."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="official NFL formula over opponent attempts",
        since=1999,
        formula=(
            "The passer rating formula applied to opponents' combined completions, attempts, "
            "passing yards, TD passes, and interceptions"
        ),
    ),
    _def_pass(
        name="deep_attempt_rate_faced",
        label="Deep Att % Faced",
        full_name="Deep Attempt Rate Faced",
        description=(
            "Share of opponent pass attempts thrown deep (about 16+ yards past the line of "
            "scrimmage). Describes how opponents attacked the defense, not how well it played."
        ),
        shape="rate",
        polarity="neutral",
        source="PBP",
        denominator="opponent pass attempts",
        since=2006,
        percent=True,
    ),
    _def_rush(
        name="carries_faced",
        label="Carries Faced",
        full_name="Carries Faced",
        description=(
            "Opponent rushing attempts against the defense, including quarterback scrambles and "
            "kneel-downs."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=1999,
    ),
    _def_rush(
        name="yards_per_carry_allowed",
        label="Yds/Carry Allowed",
        full_name="Yards Per Carry Allowed",
        description=(
            "Rushing yards opponents averaged per carry, with scrambles and kneel-downs counted as "
            "official stats count them."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent carries",
        since=1999,
    ),
    _def_rush(
        name="rush_success_rate_allowed",
        label="Rush Success Allowed",
        full_name="Rushing Success Rate Allowed",
        description=(
            "Share of opponent designed runs (not scrambles or kneel-downs) that raised their "
            "expected points."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent designed carries",
        since=1999,
        percent=True,
    ),
    _def_rush(
        name="explosive_rush_rate_allowed",
        label="Expl Rush % Allowed",
        full_name="Explosive Rush Rate Allowed",
        description="Share of opponent carries that gained 10+ yards.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent carries",
        since=1999,
        percent=True,
        formula=(
            "Opponent carries of 10+ yards ÷ opponent carries (scrambles and kneel-downs included)"
        ),
    ),
    _def_recv(
        name="targets_faced",
        label="Targets Faced",
        full_name="Targets Faced",
        description=(
            "Opponent pass attempts thrown to an intended receiver. Throwaways and spikes are "
            "attempts but not targets."
        ),
        shape="count",
        polarity="neutral",
        source="PBP +TS",
        since=1999,
        note=_TARGETS_GAP_NOTE,
    ),
    _def_recv(
        name="receptions_allowed",
        label="Rec Allowed",
        full_name="Receptions Allowed",
        description="Opponent catches — the same number as completions allowed.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
        duplicate_of="completions_allowed",
    ),
    _def_recv(
        name="receiving_yards_allowed",
        label="Rec Yds Allowed",
        full_name="Receiving Yards Allowed",
        description="Opponent receiving yards — equals passing yards allowed.",
        shape="count",
        polarity="lower",
        source="PBP +TS",
        since=1999,
        duplicate_of="passing_yards_allowed",
    ),
    _def_recv(
        name="catch_rate_allowed",
        label="Catch % Allowed",
        full_name="Catch Rate Allowed",
        description=(
            "The share of opponent targets that were caught. Throwaways and spikes are not "
            "targets, so unlike completion percentage allowed they do not count as misses."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent targets",
        since=1999,
        note=_TARGETS_GAP_NOTE,
        percent=True,
    ),
    _def_score(
        name="points_per_drive_allowed",
        label="Pts/Drive Allowed",
        full_name="Points Per Drive Allowed",
        description=(
            "Points opponents scored per possession against the defense, extra points and "
            "two-point tries included."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent drives",
        since=1999,
    ),
    _def_score(
        name="red_zone_td_pct_allowed",
        label="RZ TD % Allowed",
        full_name="Red Zone Touchdown Percentage Allowed",
        description="The share of opponent red-zone trips that ended in touchdowns.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent red-zone trips",
        since=1999,
        percent=True,
    ),
    _def_score(
        name="goal_to_go_td_pct_allowed",
        label="G2G TD % Allowed",
        full_name="Goal-To-Go Touchdown Percentage Allowed",
        description=(
            "Share of opponent goal-to-go series (sets of downs where the line to gain is the goal "
            "line) that ended in a touchdown."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent goal-to-go series",
        since=1999,
        percent=True,
    ),
    _def_score(
        name="two_pt_conversion_rate_allowed",
        label="2-Pt % Allowed",
        full_name="Two-Point Conversion Rate Allowed",
        description="Opponent two-point conversion success rate against this defense.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent two-point attempts",
        since=1999,
        percent=True,
    ),
    _def_downs(
        name="third_down_pct_allowed",
        label="3rd Down % Allowed",
        full_name="Third Down Percentage Allowed",
        description="Opponent third-down conversion rate against this defense.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent third-down attempts",
        since=1999,
        percent=True,
    ),
    _def_downs(
        name="fourth_down_pct_allowed",
        label="4th Down % Allowed",
        full_name="Fourth Down Percentage Allowed",
        description="Share of opponent fourth-down go-for-it plays that converted.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent fourth-down attempts",
        since=1999,
        percent=True,
    ),
    _def_downs(
        name="series_conversion_rate_allowed",
        label="Series Conv Allowed",
        full_name="Series Conversion Rate Allowed",
        description=(
            "Share of opponent series (sets of downs) that earned a new first down or a touchdown. "
            "A series that ends in a field goal, punt, or turnover counts as a stop."
        ),
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent series",
        since=1999,
        percent=True,
    ),
    _def_downs(
        name="three_and_outs_forced_rate",
        label="3-and-Outs Forced %",
        full_name="Three-and-Outs Forced Rate",
        description="Share of opponent drives that ended in a punt without gaining a first down.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="opponent drives",
        since=1999,
        percent=True,
    ),
    _def_drives(
        name="score_pct_per_drive_allowed",
        label="Score % Allowed",
        full_name="Scoring Rate Per Drive Allowed",
        description="Share of opponent drives that ended in a touchdown or field goal.",
        shape="rate",
        polarity="lower",
        source="PBP",
        denominator="opponent drives",
        since=1999,
        percent=True,
    ),
    _def_drives(
        name="punts_forced_pct",
        label="Punts Forced %",
        full_name="Punts Forced Rate",
        description="The share of opponent possessions that ended in a punt.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="opponent drives",
        since=1999,
        percent=True,
    ),
    _def_drives(
        name="avg_starting_field_position_allowed",
        label="Avg Start Allowed",
        full_name="Opponent Average Starting Field Position",
        description=(
            "Where opponent drives started on average, in yards from their own goal line (25 means "
            "their own 25). Driven mostly by this team's kickoffs, punts, and turnovers."
        ),
        shape="avg",
        polarity="lower",
        source="PBP",
        denominator="opponent drives",
        since=1999,
    ),
    _def_to(
        name="takeaways",
        label="Takeaways",
        full_name="Takeaways",
        description="Interceptions plus opponent fumbles recovered.",
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _def_to(
        name="def_interception_yards",
        label="INT Ret Yds",
        full_name="Interception Return Yards",
        description="Yards gained returning interceptions.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _def_to(
        name="fumble_recovery_opp",
        label="Fumbles Recovered",
        full_name="Opponent Fumbles Recovered",
        description="Opponent fumbles this defense recovered.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _def_to(
        name="takeaway_rate_per_defensive_snap",
        label="Takeaway %",
        full_name="Takeaways Per Defensive Snap",
        description=(
            "Share of opponent scrimmage plays that ended in an interception or a fumble the "
            "defense recovered."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="defensive snaps",
        since=1999,
        percent=True,
    ),
    _def_to(
        name="takeaways_per_drive",
        label="Takeaways/Drive",
        full_name="Takeaways Per Drive",
        description="Takeaways divided by opponent possessions.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="opponent drives",
        since=1999,
    ),
    _def_to(
        name="takeaway_epa",
        label="Takeaway EPA",
        full_name="Takeaway EPA",
        description=(
            "Expected points opponents lost on plays where the team took the ball away "
            "(interceptions and opponent fumbles it recovered, including muffed punts and "
            "return fumbles), so each takeaway is weighed by how costly it was."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
        formula="-(Opponent Turnover EPA)",
    ),
    _def_to(
        name="def_tds",
        label="Def & Return TDs",
        full_name="Touchdowns on Opponent Possessions",
        description=(
            "Touchdowns this team scored while the opponent had the ball: interception and fumble "
            "returns, plus scores on the opponent's punts and kicks, such as punt returns."
        ),
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _def_to(
        name="fumble_recovery_tds",
        label="Fumble Return TDs",
        full_name="Fumble Return Touchdowns",
        description="Touchdowns scored returning recovered fumbles.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _def_press(
        name="def_sack_yards",
        label="Sack Yds Forced",
        full_name="Sack Yards Forced",
        description="Yards opponents lost to this defense's sacks.",
        shape="count",
        polarity="higher",
        source="PBP +TS",
        since=1999,
    ),
    _def_press(
        name="def_sack_rate_per_dropback",
        label="Sack Rate Forced",
        full_name="Sack Rate Forced",
        description="Share of opponent dropbacks that ended in a sack.",
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="opponent dropbacks",
        since=1999,
        percent=True,
        formula="Sacks ÷ opponent dropbacks (pass attempts, sacks, and scrambles)",
    ),
    _def_press(
        name="qb_pressure_events_rate",
        label="Sack+Hit Rate",
        full_name="Sacks Plus QB Hits Per Dropback",
        description=(
            "Sacks plus quarterback hits per opponent dropback. Most sacks are also logged as "
            "hits, so they usually count twice; this is not the share of dropbacks with pressure."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="opponent dropbacks",
        since=1999,
        note=_QB_HITS_GAP_NOTE,
        percent=True,
        formula="(Sacks + plays with a QB hit) ÷ opponent dropbacks",
    ),
    _def_press(
        name="stuff_rate",
        label="Stuff %",
        full_name="Run Stuff Rate",
        description=(
            "The share of opponent carries other than kneel-downs stopped for no gain or a loss."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="opponent carries other than kneel-downs",
        since=1999,
        percent=True,
    ),
    _def_press(
        name="havoc_rate",
        label="Havoc %",
        full_name="Havoc Rate",
        description=(
            "Share of opponent scrimmage plays with a disruptive play by the defense: a tackle for "
            "loss, forced fumble, interception, or pass defended. Sacks are not counted unless "
            "they also force a fumble."
        ),
        shape="rate",
        polarity="higher",
        source="PBP",
        denominator="defensive snaps",
        since=1999,
        percent=True,
        formula=(
            "Opponent plays with a tackle for loss, forced fumble, interception, or pass defended "
            "÷ defensive snaps"
        ),
    ),
    _def_press(
        name="defensive_2pt_conversions",
        label="Def 2-Pt Returns",
        full_name="Defensive Two-Point Conversions",
        description=(
            "Times the defense returned a blocked extra point or a turnover on a two-point try to "
            "the other end zone for two points."
        ),
        shape="count",
        polarity="higher",
        source="PBP",
        since=1999,
    ),
    _def_pen(
        name="defensive_penalties",
        label="Def Penalties",
        full_name="Defensive Penalties",
        description=(
            "Penalties committed on plays where the opponent had the ball, including the "
            "team's kickoffs, punt returns, and field-goal and extra-point defense."
        ),
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
    _def_pen(
        name="defensive_penalty_yards",
        label="Def Pen Yds",
        full_name="Defensive Penalty Yards",
        description=(
            "Penalty yards assessed on plays where the opponent had the ball, including the "
            "team's kickoffs, punt returns, and field-goal and extra-point defense."
        ),
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
    _def_pen(
        name="defensive_pass_interference",
        label="DPI",
        full_name="Defensive Pass Interference Penalties",
        description="Defensive pass interference penalties committed.",
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
    _def_pen(
        name="penalty_first_downs_allowed",
        label="Pen 1Ds Allowed",
        full_name="Penalty First Downs Allowed",
        description=(
            "First downs opponents got from this team's penalties rather than by running or "
            "passing."
        ),
        shape="count",
        polarity="lower",
        source="PBP",
        since=1999,
    ),
)

__all__ = [
    "DEFENSE_METRICS",
]
