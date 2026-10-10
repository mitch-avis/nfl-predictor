# NFL QB Stats Catalog

Generated from the metric registry (`src/nfl_predictor/season_stats/metrics/`) by
`.venv/bin/python -m nfl_predictor.season_stats.metrics.catalog_docs`; do not edit it by hand.
Companion catalog: [stats-catalog.md](stats-catalog.md).

Every column below is regular season only. Data files and the analyst app add a prefix or suffix to
these base names: `opp_` (the team's head-to-head-excluded opponent profile), `qopp_` (what the
defenses a quarterback faced allowed to other passers), `_per_game`, `_total`, the per-snap
denominators `_per_offensive_snap` and `_per_defensive_snap`, rank-range percentiles `_q025` through
`_q975`, and, in the analyst app's garbage-time filter view, `filtered_` (refit on the plays the
filter keeps) and `_change` (filtered minus unfiltered). Shapes: `count` totals, `rate` ratios with
their own denominator, `avg` per-event averages, `max` the largest single value (such as the longest
play), `score` model outputs on their own scale, `flag` booleans, and `id` identity fields.

## Sources

| Code | Meaning |
| --- | --- |
| PBP | Computed by this project from nflverse play-by-play (`load_pbp`). |
| PBP +TS | Computed from play-by-play and also published in nflverse weekly team stats, which serves as a cross-check. |
| PLS | nflverse weekly player stats (`load_player_stats`). |
| SCH | nflverse schedules and final scores (`load_schedules`). |
| SNP | nflverse snap counts (`load_snap_counts`, 2013 onward; the 2012 file is empty). |
| D | Derived by this project from the other columns, including every rating. |

## Schedule-Adjusted Ratings

The project's own quarterback ratings, in EPA per dropback adjusted for the defenses each
quarterback actually faced, plus their ranks and how firm those ranks are. Start here when ranking
QBs.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `adj_qb_epa_per_dropback` | Adj EPA/Dropback | rate | dropbacks | 1999 | D | Expected points added (how much each play raised or lowered the offense's expected points) per dropback, adjusted for the pass defenses faced. Same scale as raw EPA/Dropback; every rating is pulled toward average, small samples most. Formula: One fit over every passer-game, weighted by dropbacks: EPA per dropback = league average + passer strength - defense strength + home field. Shown: league average + passer strength. |
| `qb_faced_pass_defense` | Pass Defense Faced | rate | dropbacks | 1999 | D | How strong the pass defenses faced were, in EPA per dropback they held passers below average, weighted by the quarterback's dropbacks against each. Higher means tougher defenses; 0 is average. Formula: Dropback-weighted average of each opponent's pass-defense strength, each from a refit that leaves out this quarterback's dropbacks; defenses that have faced no other passer yet are skipped. |
| `qb_rank` | Rank | score | - | 1999 | D | The quarterback's place among qualifying quarterbacks (14 pass attempts per team game) by Adjusted EPA Per Dropback; 1 is best. Tied ratings share the better rank. |
| `qb_rank_missing_share` | No-Dropback Share | rate | bootstrap resamples | 1999 | D | How often the quarterback had no dropbacks, and so no rank, across 1,000 redrawn seasons (the season's games drawn at random, with repeats). Part-time starters miss more often; his rank range uses only the seasons he appears in. |
| `qb_rank_top5_probability` | Top-5 Chance | rate | bootstrap resamples | 1999 | D | How often the quarterback ranked in the top five qualifying quarterbacks across 1,000 redrawn seasons (the season's games drawn at random, with repeats). Shows how much the ranking rests on which games were played. |
| `qb_rank_top10_probability` | Top-10 Chance | rate | bootstrap resamples | 1999 | D | How often the quarterback ranked in the top ten qualifying quarterbacks across 1,000 redrawn seasons (the season's games drawn at random, with repeats). Shows how much the ranking rests on which games were played. |
| `qb_rank_probabilities` | Rank Chances | rate | bootstrap resamples | 1999 | D | For each rank from 1 down, how often the quarterback finished exactly there among qualifying quarterbacks across 1,000 redrawn seasons (the season's games drawn at random, with repeats). Adds to under 100% when he is missing from some. |
| `other_qb_id` | Compared QB ID | id | - | - | D | The league's player ID for the other quarterback in the comparison. The two need not have faced each other. |
| `qb_rated_above_probability` | Rated-Above Chance | rate | bootstrap resamples with both quarterbacks | 1999 | D | Among 1,000 redrawn seasons (the season's games drawn at random, with repeats) that include both quarterbacks, the share in which this one's Adjusted EPA Per Dropback beat the other's. A tie counts half. |
| `qb_rating_gap` | Rating Gap | score | - | 1999 | D | This quarterback's Adjusted EPA Per Dropback minus the compared quarterback's, in each redrawn season that includes both. Negative means the other was rated higher; EPA per dropback values are small decimals (about -0.3 to +0.3). |
| `qb_pair_share` | Both-QBs Share | rate | bootstrap resamples | 1999 | D | Share of 1,000 redrawn seasons that include games by both quarterbacks, so they could be compared. Only qualifying quarterbacks are paired, so it drops below 1 mainly early in a season. |

## Identity & Availability

Who the quarterback is, how much he played, and whether he played enough to be ranked.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `qb_id` | QB ID | id | - | - | PBP | The league's official player ID for this quarterback (for example 00-0023459), used to link his rows across tables. If that ID is missing, another source's ID or his name stands in. |
| `qb_name` | QB | id | - | - | PBP | The quarterback's display name. |
| `qb_games_played` | QB Games | count | - | 1999 | PBP | Games in which the quarterback dropped back at least once or, in seasons with snap counts, played at least one offensive snap. Per-game stats divide by this count. |
| `qb_offense_snaps` | QB Snaps | count | - | 2013 | SNP | Offensive snaps the quarterback was on the field for, from the league's snap counts, which this data has from 2013 on. Empty before 2013: nflverse's 2012 snap-count file has no rows. |
| `qb_dropbacks` | Dropbacks | count | - | 1999 | PBP | Pass attempts plus sacks: the plays where the quarterback dropped back and threw or went down. Scrambles are not counted here; they count as carries. Formula: Pass attempts (two-point tries included, spikes left out) + sacks |
| `qb_is_eligible` | Qualified | flag | - | - | D | Whether the quarterback has enough pass attempts to be ranked: 14 for every game his team has played (his main team, if he changed teams). Below that he is listed but unranked. |
| `qb_attempt_qualifier` | Att to Qualify | count | - | - | D | The pass attempts the quarterback needs to be ranked: 14 for every game his team has played so far (the team he played the most games for, if he changed teams). Formula: 14 x team games played (238 in a 17-game season) |

## Passing Volume

Raw passing production (attempts, completions, yards, touchdowns, interceptions, and passing EPA),
plus the win-probability data behind the garbage-time filter.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `qb_attempts` | Att | count | - | 1999 | PLS | Official pass attempts (sacks and two-point tries excluded). |
| `qb_completions` | Comp | count | - | 1999 | PLS | Passes completed to a teammate. |
| `qb_pass_yards` | Pass Yds | count | - | 1999 | PLS | Yards gained on completed passes, including yards after the catch. Yards lost on sacks are not subtracted. |
| `qb_pass_touchdowns` | Pass TDs | count | - | 1999 | PLS | Touchdown passes thrown. |
| `qb_interceptions` | INTs | count | - | 1999 | PLS | Passes intercepted by the defense. |
| `qb_passing_epa` | Pass EPA | count | - | 1999 | PLS | Expected points added (how much each play raised or lowered the offense's expected points) on pass attempts, sacks, and spikes, as nflverse's official passing EPA counts them. Scrambles count in rushing EPA instead. |
| `qb_spike_epa` | Spike EPA | count | - | 1999 | PBP | Expected points added on the quarterback's spikes, throwing the ball into the ground to stop the clock. Passing EPA includes them; EPA per dropback and the QB rating leave them out, as a spike is a called incompletion, not a dropback. |
| `wp_kept_dropback_share` | Kept Dropbacks | rate | dropbacks | 1999 | D | The share of the quarterback's dropbacks the garbage-time filter keeps: those where the offense's chance to win before the snap was between the threshold and 100% minus it. 100% means none were dropped. |
| `qb_wp_bin_dropbacks` | WP Bin Dropbacks | count | - | 1999 | PBP | The quarterback's dropbacks in one game within one 1-point band of win probability (how far from decided the game was before the snap). Added across all bands, they equal his dropbacks in that game. |
| `qb_wp_bin_epa` | WP Bin Pass EPA | count | - | 1999 | PBP | Play-by-play passing EPA on the quarterback's dropbacks in one game within one 1-point band of win probability. The garbage-time filter adds up the bands it keeps. |

## Passing Efficiency

Quality per play: the rates that separate good QBs from busy ones.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `qb_epa_per_dropback` | EPA/Dropback | rate | dropbacks | 1999 | D | Expected points added (how much each play raised or lowered the offense's expected points) per dropback, before any adjustment for opponents. Spikes and kneel-downs are not dropbacks and are left out. League average is usually a little above 0. Formula: (Passing EPA - spike EPA) ÷ dropbacks (pass attempts + sacks) |
| `qb_pass_yards_per_dropback` | Pass Yds/Dropback | rate | dropbacks | 1999 | D | Passing yards per dropback. Unlike yards per attempt, it counts each sack as a play with no gain; the yards lost on sacks are not subtracted. Formula: Passing yards ÷ (pass attempts + sacks) |
| `qb_td_int_margin_rate` | TD-INT/Dropback | rate | dropbacks | 1999 | D | Touchdown passes minus interceptions, per dropback: a per-play version of TD-INT differential. Typical qualifying starters land between 0 and +0.03. Formula: (TD passes - interceptions) ÷ (pass attempts + sacks) |
| `qb_any_a` | ANY/A | rate | pass attempts + sacks | 1999 | D | Yards per pass play with bonuses and penalties: +20 yards for each touchdown pass, -45 for each interception, and sacks counted as plays with their lost yards subtracted. Formula: (Passing yards + 20 x TD passes - 45 x interceptions - sack yards lost) ÷ (pass attempts + sacks) |
| `qb_completion_percentage_above_expectation` | CPOE | avg | pass attempts (model-expected completions) | 2006 | PLS | How much higher the completion rate was than an average passer's on the same throws, given how hard each was, in percentage points (0 = as expected). Formula: Actual minus expected completion chance on each pass attempt, averaged over the attempts, in percentage points |
| `qb_passer_rating` | Passer Rating | rate | official NFL formula over attempts | 1999 | D | The NFL's passer rating (0 to 158.3), built from completion %, yards, touchdowns, and interceptions per attempt. Formula: Four parts, each held between 0 and 2.375: (completions ÷ attempts - 0.3) x 5, (yards ÷ attempts - 3) x 0.25, TD passes ÷ attempts x 20, and 2.375 - interceptions ÷ attempts x 25. Rating = their sum ÷ 6 x 100. |
| `qb_yards_per_attempt` | Y/A | rate | pass attempts | 1999 | D | Passing yards divided by official pass attempts. |
| `qb_touchdown_rate` | TD % | rate | pass attempts | 1999 | D | The share of pass attempts that scored touchdowns. |
| `qb_interception_rate` | INT % | rate | pass attempts | 1999 | D | The share of pass attempts that were intercepted. |
| `qb_completion_pct` | Comp % | rate | pass attempts | 1999 | D | Completions divided by official pass attempts. |

## Pressure, Sacks & Pocket

Sacks taken, the yards and fumbles lost on them, and how often the quarterback scrambled.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `qb_sacks` | Sacks | count | - | 1999 | PLS | Times the quarterback was sacked. Avoiding sacks is a QB skill. |
| `qb_sack_yards_lost` | Sack Yds Lost | count | - | 1999 | PLS | Yards lost on sacks, shown as a positive number. |
| `qb_sack_rate` | Sack Rate | rate | dropbacks | 1999 | D | The share of dropbacks (pass attempts plus sacks) that ended in a sack. Sack avoidance tracks quarterbacks more than offensive lines. |
| `qb_sack_fumbles_lost` | Sack Fumbles Lost | count | - | 1999 | PLS | Strip-sack fumbles the defense recovered. |
| `qb_scramble_rate` | Scramble % | rate | dropbacks | 1999 | PBP | How often the quarterback took off running on a called pass play: scrambles per dropback. Formula: Scrambles ÷ (pass attempts + sacks) |

## Rushing

Quarterback runs: designed carries, scrambles, and their value.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `qb_carries` | Carries | count | - | 1999 | PLS | Official rushing attempts, including scrambles and kneel-downs. |
| `qb_rushing_yards` | Rush Yds | count | - | 1999 | PLS | Rushing yards, scrambles and kneel-downs included. |
| `qb_yards_per_carry` | Rush Yds/Carry | rate | carries | 1999 | D | Rushing yards per carry, scrambles and kneel-downs included, so clock-killing kneels pull it down. |
| `qb_rushing_tds` | Rush TDs | count | - | 1999 | PLS | Touchdowns scored on the ground. |
| `qb_rushing_first_downs` | Rush 1st Downs | count | - | 1999 | PLS | First downs gained on quarterback runs, scrambles included. |
| `qb_rushing_epa` | Rush EPA | count | - | 1999 | PLS | Expected points added on runs, scrambles and kneel-downs included. A quarterback's scrambling value shows up here, not in passing EPA. |
| `qb_epa_per_carry` | EPA/Carry | rate | carries | 1999 | D | Rushing EPA per carry, scrambles and kneel-downs included. |
| `qb_designed_carries` | Designed Carries | count | - | 1999 | PBP | Runs called for the quarterback, such as sneaks and option keepers. Scrambles, kneel-downs, and two-point tries are left out. |
| `qb_designed_rush_yards` | Designed Rush Yds | count | - | 1999 | PBP | Yards gained on runs called for the quarterback (scrambles, kneel-downs, and two-point tries left out). |
| `qb_designed_rush_epa` | Designed Rush EPA | count | - | 1999 | PBP | Expected points added on runs called for the quarterback (scrambles, kneel-downs, and two-point tries left out). |
| `qb_designed_yards_per_carry` | Designed Yds/Carry | rate | designed carries | 1999 | D | Average yards on runs called for the quarterback (scrambles and kneel-downs left out). |
| `qb_designed_epa_per_carry` | Designed EPA/Carry | rate | designed carries | 1999 | D | Average EPA on runs called for the quarterback (scrambles and kneel-downs left out). |
| `qb_scrambles` | Scrambles | count | - | 1999 | PBP | Pass plays on which the quarterback took off and ran instead of throwing. |
| `qb_scramble_yards` | Scramble Yds | count | - | 1999 | PBP | Yards gained on scrambles. |
| `qb_yards_per_scramble` | Yds/Scramble | rate | scrambles | 1999 | PBP | Average yards gained per scramble. |
| `qb_kneels` | Kneels | count | - | 1999 | PBP | Kneel-downs to run out the clock. Counted in carries and the per-carry rates, left out of the designed-run stats. |
| `qb_rushing_2pt_conversions` | 2-Pt Rush Conv | count | - | 1999 | PLS | Successful two-point conversions run in. |

## Scoring, Clutch & Outcomes

Results and late-game moments: wins, comebacks, and game-winning drives. Context stats — they never
feed the performance ratings.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `qb_wins` | QB Wins | count | - | 1999 | D | Team wins in games where he was the main quarterback (most snaps, or most dropbacks before snap counts). A team result shown for context; it never feeds a rating. |
| `qb_losses` | QB Losses | count | - | 1999 | D | Team losses in games where he was the main quarterback (most snaps, or most dropbacks before snap counts). |
| `qb_ties` | QB Ties | count | - | 1999 | D | Ties in games where he was the main quarterback (most snaps, or most dropbacks before snap counts). |
| `qb_win_pct` | QB Win % | rate | primary-QB games | 1999 | D | Share of primary-QB games won, counting a tie as half a win; empty for a quarterback who was never the primary passer. A team outcome, shown for context — never a rating input. |
| `qb_fourth_quarter_comeback` | 4th-Qtr Comeback | count | - | 1999 | D | 1 if the team trailed in the fourth quarter or overtime and came back to win; the credit goes to the game's main quarterback. Any deficit counts, not just one score. |
| `qb_fourth_quarter_comebacks` | 4th-Qtr Comebacks | count | - | 1999 | D | Wins in which the team trailed in the fourth quarter or overtime, counted in games where he was the main quarterback. Any deficit counts, not just one score. |
| `qb_game_winning_drive` | Game-Winning Drive | count | - | 1999 | D | 1 if the team, in a game it won, scored to go from tied or behind to ahead in the fourth quarter or overtime; the credit goes to the game's main quarterback. |
| `qb_game_winning_drives` | Game-Winning Drives | count | - | 1999 | D | Wins in which the team scored to go from tied or behind to ahead in the fourth quarter or overtime, counted in games where he was the main quarterback. |

## Turnovers & Ball Security

Fumbles on quarterback runs and the touchdown-minus-interception margin. Interceptions are under
Passing Volume, and sack fumbles under Pressure, Sacks & Pocket.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `qb_td_int_differential` | TD-INT Diff | count | - | 1999 | D | Touchdown passes minus interceptions across the season. |
| `qb_rushing_fumbles` | Rush Fumbles | count | - | 1999 | PLS | Fumbles on quarterback runs, whether or not lost. |
| `qb_rushing_fumbles_lost` | Rush Fumbles Lost | count | - | 1999 | PLS | Fumbles on quarterback runs that the defense recovered. |
