# NFL Team Stats Catalog

Generated from the metric registry (`src/nfl_predictor/season_stats/metrics/`) by
`.venv/bin/python -m nfl_predictor.season_stats.metrics.catalog_docs`; do not edit it by hand.
Companion catalog: [qb-stats-catalog.md](qb-stats-catalog.md).

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

The project's own ratings, in points per game against an average team and adjusted for the opponents
each team actually played, plus their ranks and how firm those ranks are. Start here when ranking
teams.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `team_rating` | Team Rating | score | - | 1999 | D | Points per game better (+) or worse (-) than an average team on a neutral field, built from expected points added (EPA) per play and adjusted for every opponent faced. 0 is average; most teams fall between about -8 and +8. Formula: Offense Rating + Defense Rating + Special Teams Rating Until a team has played 9 games, its offense and defense also lean on last season's (Offense Prior and Defense Prior), a little less after each game. |
| `offense_rating` | Off Rating | score | - | 1999 | D | Points per game the offense added compared with an average offense, from its EPA per run or pass play, adjusted for the defenses it faced. Pace does not count: every team is scaled to the league-average number of plays. Formula: Opponent-adjusted EPA per scrimmage play above average x league-average scrimmage plays per team-game Until a team has played 9 games, its offense and defense also lean on last season's (Offense Prior and Defense Prior), a little less after each game. |
| `defense_rating` | Def Rating | score | - | 1999 | D | Points per game the defense saved compared with an average defense, from the EPA per run or pass play it allowed, adjusted for the offenses it faced. Formula: Opponent-adjusted EPA per scrimmage play prevented x league-average scrimmage plays per team-game Until a team has played 9 games, its offense and defense also lean on last season's (Offense Prior and Defense Prior), a little less after each game. |
| `special_teams_rating` | ST Rating | score | - | 1999 | D | Points per game gained on kicking plays (kickoffs, punts, field goals, extra points, and their returns) compared with an average team, counting plays with and without the ball, adjusted for the opponents faced. Formula: (Opponent-adjusted EPA per kicking play with the ball + EPA per kicking play prevented without it) x league-average kicking plays per team-game |
| `SRS` | SRS | score | - | 1999 | D | Average point margin per game, adjusted for the strength of the opponents played, with each opponent also judged by its point margin. Built from final scores rather than plays, as a check beside Team Rating. 0 is average. Formula: Average point margin + average SRS of the opponents played, solved for every team at once and centered at 0 |
| `sos` | SoS | score | - | 1999 | D | The average Team Rating of the opponents played, one entry per game, so a team faced twice counts twice. Each opponent is rated with this team's games left out. Higher means a tougher schedule; 0 is average. Formula: Average, over games played, of each opponent's Team Rating refit without this team's games; opponents with no other games yet are skipped |
| `team_rank` | Rank | score | - | 1999 | D | The team's place in the league by Team Rating, 1 for the best. Teams with equal ratings share the better rank. |
| `team_rank_missing_share` | Unranked Share | rate | bootstrap resamples | 1999 | D | Share of 1,000 redrawn seasons (the season's games drawn at random, with repeats) in which the team had no games and so no rank. Essentially zero for a finished season; it grows early in a season. |
| `team_rank_top5_probability` | Top-5 Chance | rate | bootstrap resamples | 1999 | D | Share of 1,000 redrawn seasons (the season's games drawn at random, with repeats) in which the team ranked in the top five by Team Rating. Shows how much the ranking depends on which games happened to be played. |
| `team_rank_top10_probability` | Top-10 Chance | rate | bootstrap resamples | 1999 | D | Share of 1,000 redrawn seasons (the season's games drawn at random, with repeats) in which the team ranked in the top ten by Team Rating. Shows how much the ranking depends on which games happened to be played. |
| `team_rank_probabilities` | Rank Chances | rate | bootstrap resamples | 1999 | D | For each rank from 1st down, the share of 1,000 redrawn seasons (the season's games drawn at random, with repeats) in which the team finished at exactly that rank by Team Rating. |
| `excluded_team` | Left-Out Team | id | - | 2003 | D | For a schedule-strength refit, the team whose games are left out, this season's and last season's; blank for the season fit itself. |
| `offense_prior` | Offense Prior | score | - | 2003 | D | The per-play offense value the early-season fit pulls this team toward instead of average: its offense last season times how much of it usually carries over, fading to nothing by its 9th game, then shifted so the league averages zero. Formula: max(0, 1 - games played / 9) x carryover slope x last season's per-play offense effect, minus the league average of the same |
| `defense_prior` | Defense Prior | score | - | 2003 | D | The per-play defense value the early-season fit pulls this team toward instead of average: its defense last season times how much of it usually carries over, fading to nothing by its 9th game, then shifted so the league averages zero. Formula: max(0, 1 - games played / 9) x carryover slope x last season's per-play defense effect, minus the league average of the same |
| `offense_rank` | Off Rank | score | - | 1999 | D | The team's place in the league by Offense Rating, 1 for the best. Teams with equal ratings share the better rank. |
| `defense_rank` | Def Rank | score | - | 1999 | D | The team's place in the league by Defense Rating, 1 for the best. Teams with equal ratings share the better rank. |
| `special_teams_rank` | ST Rank | score | - | 1999 | D | The team's place in the league by Special Teams Rating, 1 for the best. Teams with equal ratings share the better rank. |
| `other_team` | Compared Team | id | - | - | D | The other team in a head-to-head comparison, by its standard NFL abbreviation. The two teams need not have played each other. |
| `team_rated_above_probability` | Rated-Above Chance | rate | bootstrap resamples with both teams | 1999 | D | Among 1,000 redrawn seasons (the season's games drawn at random, with repeats) that include both teams, the share in which this team's Team Rating beat the compared team's. A tie counts half. |
| `team_rating_gap` | Rating Gap | score | - | 1999 | D | This team's Team Rating minus the compared team's, in points per game, in each redrawn season (the season's games drawn at random, with repeats). Negative means the compared team was rated higher. |
| `team_pair_share` | Both-Teams Share | rate | bootstrap resamples | 1999 | D | Share of 1,000 redrawn seasons in which both teams had at least one game, so they could be compared. It is 1 for a finished season and slightly lower early in a season. |

## Overall

Who played whom and how games turned out: record, points, and whole-team margins.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `team` | Team | id | - | - | SCH | The team's standard NFL abbreviation, such as KC or PHI. |
| `game_id` | Game ID | id | - | - | SCH | A unique code for one game: season, week, away team, and home team, for example 2025_01_ARI_NO. |
| `week` | Week | id | - | - | SCH | The regular-season week: on a game row, the week the game was played; on a rating-history row, the last week of games the rating includes. |
| `opponent_team` | Opponent | id | - | - | SCH | The opposing team in this game or in the summarized opponent row. |
| `is_home` | Home | flag | - | 1999 | SCH | Whether the team was the home team in this game. At a neutral site, this is the team the schedule lists as home. |
| `games_played` | Games Played | count | - | 1999 | SCH | Regular-season games played (on a week-by-week history row, games through that week). Per-game figures are season totals divided by this number. |
| `games` | Games | count | - | 1999 | SCH | How many games this row covers: 1 for a single game, more for a summary row such as all games against one opponent. |
| `wins` | Wins | count | - | 1999 | SCH | Regular-season games won. Treated as an outcome, not a rating input. |
| `losses` | Losses | count | - | 1999 | SCH | Regular-season games lost. Treated as an outcome, not a rating input. |
| `ties` | Ties | count | - | 1999 | SCH | Regular-season games that ended tied after overtime. |
| `win_pct` | Win % | rate | games played | 1999 | D | Share of games won, with a tie counted as half a win. Formula: (Wins + 0.5 x ties) ÷ games played |
| `win_value` | Win Value | avg | games included | 1999 | D | A game's result as a number: 1 for a win, 0.5 for a tie, 0 for a loss. A row covering several games shows the average, which works like a win percentage. |
| `points_for` | Points For | count | - | 1999 | SCH | Points scored, from every source: offense, defense, and special teams. |
| `points_allowed` | Points Allowed | count | - | 1999 | SCH | Points given up, counting every score by the other side, not only those against the defense. |
| `point_margin` | Point Margin | count | - | 1999 | D | Points scored minus points allowed. |
| `turnover_margin` | TO Margin | count | - | 1999 | PBP | Takeaways minus giveaways: interceptions made and opponent fumbles recovered, minus interceptions thrown and fumbles lost. |
| `points_per_offensive_snap` | Points/Off Snap | rate | offensive snaps | 1999 | D | Points scored per offensive snap (each run or pass play), counting every score, including defensive and special-teams touchdowns. |
| `points_allowed_per_defensive_snap` | Pts Allowed/Snap | rate | defensive snaps | 1999 | D | Points given up per defensive snap (each run or pass play faced), counting every score by the other side. |
| `total_yards_differential` | Total Yds Diff | count | - | 1999 | PBP | Passing plus rushing yards gained minus passing plus rushing yards allowed. Passing yards are counted before sack losses. |
| `penalty_differential` | Penalty Diff | count | - | 1999 | PBP | Penalties drawn (called on the opponent) minus penalties committed, across offense, defense, and special teams. |
| `penalty_yards_differential` | Pen Yds Diff | count | - | 1999 | PBP | Penalty yards drawn (called on the opponent) minus penalty yards committed, across offense, defense, and special teams. |
| `epa_margin_per_play` | EPA Margin/Play | rate | scrimmage snaps | 1999 | PBP | Offensive EPA per play minus defensive EPA per play allowed, not adjusted for opponents. EPA (expected points added) measures how much a play changed the offense's expected points. |
| `success_rate_margin` | Success Margin | rate | scrimmage snaps | 1999 | PBP | Share of the offense's plays that succeeded minus the share that succeeded against the defense. A play succeeds when it raises the offense's expected points. |
| `wp_unit` | Unit | id | - | 1999 | PBP | Which plays this row counts: the team's runs and passes (scrimmage), or the kicking plays where it had the ball (special teams). |
| `wp_bin` | WP Bin | id | - | 1999 | PBP | How close the game was before the snap: the underdog's chance of winning at that moment, in whole percentage points, rounded down. 50 is a toss-up; 0 means one side was over 99% likely to win. Formula: 100 x the smaller of the offense's win probability and 1 minus it, rounded down. Plays without a win probability get no bin, and every filter keeps them. |
| `wp_kept_play_share` | Kept Plays | rate | scrimmage and special-teams plays | 1999 | D | Share of the team's own plays (runs, passes, and kicking plays with the ball) that the garbage-time filter keeps at the chosen setting. 100% means no play was dropped. |
| `wp_bin_plays` | Plays | count | - | 1999 | PBP | Plays the team ran with the ball in one game, play unit, and win-probability bin. Added up over all bins, they equal the game's total for that unit. |
| `wp_bin_epa` | EPA | count | - | 1999 | PBP | Expected points added on the team's plays with the ball in one game, play unit, and win-probability bin. Added up over all bins, it equals the game's total for that unit. |

## Offense

What the offense did with the ball: passing, rushing, receiving, scoring, conversions, drives,
turnovers, and penalties.

### Offense: Total

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `offensive_snaps` | Off Snaps | count | - | 1999 | PBP | Plays the offense ran from scrimmage: dropbacks (pass attempts, sacks, and scrambles), designed runs, kneel-downs, and spikes. Punts, kicks, and kickoffs are not included. |
| `total_yards` | Total Yds | count | - | 1999 | PBP +TS | Passing yards plus rushing yards. Yards lost on sacks are not subtracted, so this runs higher than the NFL's official total yards. |
| `yards_per_offensive_snap` | Yds/Off Snap | rate | offensive snaps | 1999 | PBP | Average yards gained per offensive snap, the simplest measure of how well an offense moves the ball. Formula: (Passing yards + rushing yards) ÷ offensive snaps; sack losses are not subtracted |
| `first_downs` | First Downs | count | - | 1999 | PBP | First downs gained by run, pass, or a penalty on the defense. |
| `scrimmage_tds` | Scrimmage TDs | count | - | 1999 | PBP +TS | Touchdowns scored by the offense: passing plus rushing touchdowns. |
| `offensive_epa` | Off EPA | count | - | 1999 | PBP | EPA (expected points added) on the offense's snaps. EPA measures how much a play raised or lowered the offense's expected points, given down, distance, and field position, so it credits the situation, not just yards. |
| `epa_per_offensive_snap` | EPA/Off Snap | rate | offensive snaps | 1999 | PBP | Average EPA (expected points added) per offensive snap, the core measure of offensive efficiency. Team seasons usually fall between about -0.2 and +0.1. |
| `success_rate` | Success % | rate | offensive snaps | 1999 | PBP | Share of offensive snaps with positive EPA, meaning the play left the offense better placed to score than before. It rewards consistency and ignores how big each gain was. |
| `explosive_play_rate` | Explosive % | rate | offensive snaps | 1999 | PBP | Share of offensive snaps that were big plays: completions of 20 or more yards, or carries of 10 or more yards with scrambles included. Formula: (Completions of 20+ yards + carries of 10+ yards) ÷ offensive snaps |
| `no_huddle_rate` | No-Huddle % | rate | offensive snaps | 2003 | PBP | Share of offensive snaps run without a huddle. A style stat, not a grade. Blank before 2003, when nflverse flags almost no snap as no-huddle. The flag follows the play text, which marks no-huddle snaps more often in later seasons, so compare teams within a season. |
| `shotgun_rate` | Shotgun % | rate | offensive snaps | 1999 | PBP | Share of offensive snaps taken from the shotgun formation. A style stat, not a grade. |
| `pass_rate` | Pass % | rate | offensive snaps | 1999 | PBP | Share of offensive snaps that were dropbacks (pass attempts, sacks, and scrambles): how pass-heavy the offense is. |
| `early_down_pass_rate` | Early-Down Pass % | rate | early-down snaps | 1999 | PBP | Share of first- and second-down snaps that were dropbacks: how pass-first the play-calling is before third down. |
| `pass_rate_over_expected` | Pass Rate Over Exp | avg | offensive snaps (model-expected pass rate) | 2006 | PBP | How much more (or less) often the offense dropped back to pass than a model expects for the situation (down, distance, field position, score, and time), in percentage points. Available from 2006. Formula: Average over offensive snaps of 100 x (1 for a dropback, else 0, minus the model's dropback chance) |
| `offensive_wpa` | Off Win Prob Added | count | - | 1999 | PBP | Change in the team's chance of winning from its offensive snaps, as a fraction: +0.10 means 10 percentage points of win probability gained. Plays late in close games move it most. |

### Offense: Passing

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `passing_yards` | Pass Yds | count | - | 1999 | PBP +TS | Yards gained on completed passes, before taking away yards lost on sacks. Net Passing Yards subtracts sacks, as the NFL's official team stats do. |
| `passing_epa` | Pass EPA | count | - | 1999 | PBP +TS | EPA (expected points added) on pass attempts and sacks; scrambles count as runs. EPA is how much a play raised or lowered the offense's expected points, given down, distance, and field position. |
| `passing_tds` | Pass TDs | count | - | 1999 | PBP +TS | Touchdown passes thrown by the team. |
| `passing_first_downs` | Pass 1Ds | count | - | 1999 | PBP +TS | First downs gained by completed passes. Scrambles count as runs, and first downs awarded by penalty are left out. |
| `passing_cpoe` | Pass CPOE | avg | pass attempts (model-expected completions) | 2006 | PBP +TS | Completion percentage compared with what an average passer would complete on the same throws, given how hard each was, in percentage points: 0 is as expected, +3 is 3 points above. Formula: Actual minus expected completion chance on each pass attempt, averaged over the attempts, in percentage points |
| `sacks_suffered` | Sacks Taken | count | - | 1999 | PBP +TS | Times the team's quarterbacks were sacked. |
| `passing_interceptions` | INTs Thrown | count | - | 1999 | PBP +TS | Passes the defense intercepted. |
| `sack_fumbles_lost` | Sack Fum Lost | count | - | 1999 | PBP +TS | Fumbles lost to the defense on sack plays (strip-sacks). |
| `attempts` | Att | count | - | 1999 | PBP +TS | Official pass attempts: every throw, complete or not, including interceptions and spikes. Sacks and two-point tries are left out, as in the league's official stats. |
| `completions` | Comp | count | - | 1999 | PBP +TS | Completed passes by the team. |
| `completion_pct` | Comp % | rate | pass attempts | 1999 | PBP | The share of official pass attempts that were completed. |
| `net_passing_yards` | Net Pass Yds | count | - | 1999 | PBP | Passing yards minus the yards lost on sacks. This is how the NFL's official team stats count passing yards. |
| `dropbacks` | Dropbacks | count | - | 1999 | PBP | Every play that started as a pass: pass attempts, sacks, and scrambles. |
| `sack_yards_lost` | Sack Yds Lost | count | - | 1999 | PBP +TS | Yards lost on sacks, shown as a positive number. |
| `sack_rate_per_dropback` | Sack Rate | rate | dropbacks | 1999 | PBP | The share of dropbacks that ended in a sack. |
| `scrambles` | Scrambles | count | - | 1999 | PBP | Times the quarterback dropped back to pass and then ran with the ball instead. |
| `scramble_yards` | Scramble Yds | count | - | 1999 | PBP | Rushing yards gained on quarterback scrambles. |
| `passing_air_yards` | Air Yds | count | - | 2006 | PBP +TS | Total distance the ball traveled past the line of scrimmage on all throws, including incompletions — a measure of how far downfield the team attacks. |
| `passing_yards_after_catch` | YAC | count | - | 2006 | PBP +TS | Yards receivers gained after catching the ball. |
| `air_yards_per_attempt` | Air Yds/Att | rate | pass attempts | 2006 | PBP | How far past the line of scrimmage the average pass attempt traveled in the air, throwaways included. A style stat, not a quality grade. |
| `yac_per_completion` | YAC/Comp | rate | completions | 2006 | PBP | Average yards gained after the catch on completed passes. |
| `yards_per_attempt` | Y/A | rate | pass attempts | 1999 | PBP | Average passing yards per pass attempt. Sacks don't count as attempts, and their lost yards are not subtracted. |
| `net_yards_per_attempt` | NY/A | rate | pass attempts + sacks | 1999 | PBP | Yards per pass play with sacks counted: each sack adds a play, and its lost yards come off the total. Formula: (Passing yards - sack yards lost) ÷ (pass attempts + sacks) |
| `adjusted_net_yards_per_attempt` | ANY/A | rate | pass attempts + sacks | 1999 | PBP | Net yards per attempt with a 20-yard bonus for each touchdown pass and a 45-yard charge for each interception, so scoring and turnovers count along with yards. Formula: (Passing yards + 20 x TD passes - 45 x interceptions - sack yards lost) ÷ (pass attempts + sacks) |
| `yards_per_dropback` | Yds/Dropback | rate | dropbacks | 1999 | PBP | Passing yards per dropback. Sacks and scrambles count as dropbacks but add no passing yards, and sack losses are not subtracted. |
| `epa_per_dropback` | EPA/Dropback | rate | dropbacks | 1999 | PBP | Average EPA per dropback, scrambles included: the core measure of passing efficiency. EPA is how much a play raised or lowered the offense's expected points. Team seasons mostly run -0.2 to +0.25. |
| `pass_success_rate` | Pass Success % | rate | dropbacks | 1999 | PBP | The share of dropbacks with positive EPA, meaning the play left the offense better placed to score than before. It rewards steady gains, not just big plays. |
| `team_passer_rating` | Passer Rating | rate | official NFL formula over attempts | 1999 | PBP | The NFL's official passer rating formula (scale 0 to 158.3), built from completion %, yards, touchdowns, and interceptions per attempt, applied to all the team's passes. Formula: Four parts, each held between 0 and 2.375: (completions ÷ attempts - 0.3) x 5, (yards ÷ attempts - 3) x 0.25, TD passes ÷ attempts x 20, and 2.375 - interceptions ÷ attempts x 25. Rating = their sum ÷ 6 x 100. |
| `passing_td_rate_per_attempt` | TD % | rate | pass attempts | 1999 | PBP | The share of pass attempts that scored touchdowns. |
| `int_rate_per_attempt` | INT % | rate | pass attempts | 1999 | PBP | The share of pass attempts that were intercepted. |
| `explosive_pass_rate` | Explosive Pass % | rate | dropbacks | 1999 | PBP | The share of dropbacks that produced a completion of 20 or more yards. |
| `deep_attempt_rate` | Deep Att % | rate | pass attempts | 2006 | PBP | The share of pass attempts thrown 16 or more yards past the line of scrimmage. It shows how often a team throws deep, not how well. |
| `longest_pass` | Longest Pass | max | - | 1999 | PBP | The longest completed pass, in yards: the season's longest in season tables, that game's longest in game logs. |
| `sack_fumbles` | Sack Fumbles | count | - | 1999 | PBP +TS | Fumbles on sack plays, whether or not the team lost the ball. |
| `passing_2pt_conversions` | 2-Pt Pass Conv | count | - | 1999 | PBP +TS | Successful two-point conversions thrown. |
| `air_epa_total` | Air EPA | count | - | 2006 | PBP | EPA credited to the distance each pass traveled in the air, summed over every pass attempt. Incompletions and interceptions count the value the pass would have added if caught. |
| `yac_epa_total` | YAC EPA | count | - | 2006 | PBP | EPA added after the catch on completed passes: the value receivers created with the ball in their hands. Formula: Sum over completed passes of the play's EPA minus its air EPA |
| `xyac_per_completion` | Exp YAC/Comp | avg | completions | 2006 | PBP | The yards after the catch an average receiver would be expected to gain on the same catches, from nflverse's expected-yards model. |
| `yac_over_expected_per_completion` | YAC Over Exp | avg | completions | 2006 | PBP | How many more yards after the catch the team gained per catch than an average receiver would on the same catches. Zero means as expected. |

### Offense: Rushing

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `rushing_yards` | Rush Yds | count | - | 1999 | PBP +TS | Yards gained on carries, including quarterback scrambles and kneel-downs. |
| `rushing_epa` | Rush EPA | count | - | 1999 | PBP +TS | EPA (expected points added) on carries, including quarterback scrambles and kneel-downs. Most teams finish below zero, because the average run loses a little expected value. |
| `rushing_tds` | Rush TDs | count | - | 1999 | PBP +TS | Touchdowns scored on the ground. |
| `rushing_first_downs` | Rush 1st Downs | count | - | 1999 | PBP +TS | First downs gained on the ground. |
| `rushing_fumbles_lost` | Rush Fumbles Lost | count | - | 1999 | PBP +TS | Fumbles lost to the defense on rushing plays. |
| `carries` | Carries | count | - | 1999 | PBP +TS | Official rushing attempts, including scrambles and kneel-downs. |
| `designed_carries` | Designed Carries | count | - | 1999 | PBP | Called running plays: carries that were not quarterback scrambles or kneel-downs. |
| `yards_per_carry` | Yds/Carry | rate | carries | 1999 | PBP | Average yards per carry, including quarterback scrambles and kneel-downs. |
| `epa_per_carry` | EPA/Carry | rate | carries | 1999 | PBP | Average EPA (expected points added) per carry, scrambles and kneel-downs included: how much value the running game produced per attempt. |
| `rush_success_rate` | Rush Success % | rate | designed carries | 1999 | PBP | Share of called running plays with positive EPA, meaning the run left the offense better placed to score. Scrambles and kneel-downs are left out. |
| `explosive_rush_rate` | Explosive Rush % | rate | carries | 1999 | PBP | Share of carries that gained 10 or more yards, scrambles included. |
| `stuffed_run_rate` | Stuffed % | rate | carries other than kneel-downs | 1999 | PBP | The share of carries other than kneel-downs stopped for no gain or a loss. |
| `rushing_fumbles` | Rush Fumbles | count | - | 1999 | PBP +TS | Fumbles on rushing plays, whether or not the team lost the ball. |
| `longest_rush` | Longest Rush | max | - | 1999 | PBP | Longest single carry, in yards, scrambles included. Game logs show the game's longest; the season table shows the season's longest. |
| `rushing_2pt_conversions` | 2-Pt Rush Conv | count | - | 1999 | PBP +TS | Successful two-point conversions run in. |

### Offense: Receiving

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `targets` | Targets | count | - | 1999 | PBP +TS | Pass attempts thrown to an intended receiver. Throwaways and spikes are attempts but not targets. Blank in 2003-2008, when nflverse play-by-play names the intended receiver on almost no incomplete passes. |
| `receptions` | Receptions | count | - | 1999 | PBP +TS | Team catches — the same number as completions, receiving-side view. Same value as `completions`. |
| `receiving_yards` | Rec Yds | count | - | 1999 | PBP +TS | Yards gained on catches, including yards after the catch; the same number as passing yards. Same value as `passing_yards`. |
| `receiving_tds` | Rec TDs | count | - | 1999 | PBP +TS | Touchdown catches — the same number as passing touchdowns. Same value as `passing_tds`. |
| `receiving_air_yards` | Rec Air Yds | count | - | 2006 | PBP +TS | How far past the line of scrimmage passes were thrown, summed over every attempt; the same number as passing air yards. Same value as `passing_air_yards`. |
| `receiving_yards_after_catch` | Rec YAC | count | - | 2006 | PBP +TS | Yards receivers gained after the catch; the same number as passing yards after catch. Same value as `passing_yards_after_catch`. |
| `receiving_first_downs` | Rec 1Ds | count | - | 1999 | PBP +TS | First downs on catches — the same number as passing first downs. Same value as `passing_first_downs`. |
| `receiving_fumbles` | Rec Fumbles | count | - | 1999 | PBP +TS | Fumbles by receivers after the catch, whether or not lost. |
| `receiving_fumbles_lost` | Rec Fum Lost | count | - | 1999 | PBP +TS | Receivers' fumbles after the catch that the defense recovered. |
| `catch_rate` | Catch % | rate | targets | 1999 | PBP | The share of targets that were caught. Throwaways and spikes are not targets, so unlike completion percentage they do not count as misses. Blank in 2003-2008, when nflverse play-by-play names the intended receiver on almost no incomplete passes. |

### Offense: Scoring

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `total_tds` | Total TDs | count | - | 1999 | PBP | All touchdowns the team scored: offense, defense, and special teams. |
| `offensive_tds` | Off TDs | count | - | 1999 | PBP | Passing plus rushing touchdowns. Same value as `scrimmage_tds`. |
| `points_per_drive` | Pts/Drive | rate | drives | 1999 | PBP | Average points the offense scored per drive, counting the extra point or two-point try after a touchdown. |
| `red_zone_trips` | Red Zone Trips | count | - | 1999 | PBP | Drives that got inside the opponent's 20-yard line. |
| `red_zone_td_pct` | Red Zone TD % | rate | red-zone trips | 1999 | PBP | Share of red-zone trips (drives that got inside the opponent's 20) that ended in a touchdown. |
| `points_per_red_zone_trip` | Pts/Red Zone Trip | rate | red-zone trips | 1999 | PBP | Average points scored on drives that got inside the opponent's 20, counting the extra point or two-point try. |
| `goal_to_go_td_pct` | Goal-to-Go TD % | rate | goal-to-go series | 1999 | PBP | Share of goal-to-go series (first-and-goal and the downs after it) that ended in a touchdown. |
| `two_pt_attempts` | 2-Pt Att | count | - | 1999 | PBP +TS | Two-point conversion tries. |
| `two_pt_conversions` | 2-Pt Made | count | - | 1999 | PBP +TS | Successful two-point conversions. |
| `two_pt_conversion_rate` | 2-Pt % | rate | two-point attempts | 1999 | PBP | Share of two-point tries that succeeded. |

### Offense: Downs & Conversions

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `first_downs_penalty` | Penalty 1st Downs | count | - | 1999 | PBP | First downs the offense gained because of a penalty on the defense. |
| `third_down_attempts` | 3rd Down Att | count | - | 1999 | PBP | Third downs faced by the offense. |
| `third_down_conversions` | 3rd Down Conv | count | - | 1999 | PBP | Third downs converted into a first down or touchdown. |
| `third_down_pct` | 3rd Down % | rate | third-down attempts | 1999 | PBP | Share of third downs the offense converted into a first down or touchdown. |
| `third_down_avg_distance` | 3rd Down Dist | avg | third downs faced | 1999 | PBP | Average yards to go on third down. A short third down usually means the offense gained good yardage on first and second down. |
| `fourth_down_attempts` | 4th Down Att | count | - | 1999 | PBP | Fourth downs on which the offense ran a play instead of punting or kicking a field goal. |
| `fourth_down_conversions` | 4th Down Conv | count | - | 1999 | PBP | Fourth-down tries that gained a first down or touchdown. |
| `fourth_down_pct` | 4th Down % | rate | fourth-down attempts | 1999 | PBP | Share of fourth-down tries the offense converted. |
| `fourth_down_go_rate` | 4th Down Go % | rate | fourth downs faced | 1999 | PBP | How often the offense went for it on fourth down instead of punting or kicking a field goal. A coaching-style stat, not a grade. Formula: Fourth-down tries ÷ fourth downs faced (tries, punts, and field-goal attempts) |
| `fourth_down_aggressiveness` | 4th & Short Go % | rate | fourth-and-short situations | 1999 | PBP | How often the offense went for it on fourth down with 2 or fewer yards to go. Analytics research generally favors going for it in these spots. Formula: Fourth-down tries with 2 or fewer yards to go ÷ fourth downs faced with 2 or fewer yards to go (tries, punts, and field-goal attempts) |
| `series` | Series | count | - | 1999 | PBP | Sets of downs the offense started: each possession and each new first down begins one. |
| `series_conversion_rate` | Series Conv % | rate | series | 1999 | PBP | Share of the offense's sets of downs that earned a new first down or a touchdown. Steadier than third-down rate because every set of downs counts. |
| `three_and_out_rate` | 3-and-Out % | rate | drives | 1999 | PBP | Share of drives that ended in a punt without gaining a first down. |
| `turnovers_on_downs` | Turnovers on Downs | count | - | 1999 | PBP | Failed fourth-down tries that handed the ball over. |

### Offense: Drives & Field Position

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `drives` | Drives | count | - | 1999 | PBP | Possessions the offense had, from taking over the ball until it scored, gave the ball up, or the half ended. |
| `yards_per_drive` | Yds/Drive | rate | drives | 1999 | PBP | Average net yards per drive, counting yards lost on sacks and yards gained or lost through penalties. |
| `plays_per_drive` | Plays/Drive | avg | drives | 1999 | PBP | Average number of offensive plays per drive. |
| `time_per_drive` | Time/Drive | avg | drives | 1999 | PBP | Average game-clock time per drive, in seconds (180 = 3 minutes). |
| `first_downs_per_drive` | 1st Downs/Drive | avg | drives | 1999 | PBP | Average first downs gained per possession. |
| `score_pct_per_drive` | Score %/Drive | rate | drives | 1999 | PBP | Share of drives that ended in a touchdown or field goal. |
| `punt_pct_per_drive` | Punt %/Drive | rate | drives | 1999 | PBP | Share of drives that ended in a punt. |
| `turnover_pct_per_drive` | Turnover %/Drive | rate | drives | 1999 | PBP | Share of drives that ended in a giveaway: an interception or a lost fumble, including those returned for a touchdown. |
| `avg_starting_field_position` | Avg Drive Start | avg | drives | 1999 | PBP | Average yard line where drives started, counted from the team's own goal line: 25 means its own 25, 50 means midfield, 60 means the opponent's 40. |
| `long_field_score_pct` | Long-Field Score % | rate | long-field drives | 1999 | PBP | Share of drives starting at or inside the team's own 25 that ended in a touchdown or field goal. |
| `drive_penalty_yards` | Net Drive Pen Yds | count | - | 2001 | PBP | Net penalty yards on the team's drives: yards the defense's fouls gave the offense minus yards its own fouls cost it. |

### Offense: Turnovers

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `giveaways` | Giveaways | count | - | 1999 | PBP | Interceptions thrown plus fumbles lost on offensive snaps. |
| `fumbles` | Fumbles | count | - | 1999 | PBP +TS | All offensive fumbles, whether recovered or lost. |
| `fumbles_lost` | Fumbles Lost | count | - | 1999 | PBP +TS | Offensive fumbles the defense recovered. |
| `giveaway_rate_per_offensive_snap` | Giveaway % | rate | offensive snaps | 1999 | PBP | Share of offensive snaps that ended in an interception or lost fumble. |
| `giveaways_per_drive` | Giveaways/Drive | rate | drives | 1999 | PBP | Average giveaways per drive. |
| `turnover_epa` | Turnover EPA | count | - | 1999 | PBP | EPA (expected points added), from the team's side, on its interceptions and lost fumbles, including muffs and fumbles on kickoff and punt returns: how costly its giveaways were, not just how many. Almost always negative. |

### Offense: Penalties

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `penalties` | Penalties | count | - | 1999 | PBP +TS | Accepted penalties called on the team on offense, defense, and special teams. |
| `penalty_yards` | Penalty Yds | count | - | 1999 | PBP +TS | Yards marked off against the team on its accepted penalties, across all units. |
| `offensive_penalties` | Off Penalties | count | - | 1999 | PBP | Penalties committed on plays where the team had the ball, including its punts, field goals, extra points, and kickoff returns. |
| `offensive_penalty_yards` | Off Pen Yds | count | - | 1999 | PBP | Penalty yards assessed on plays where the team had the ball, including its punts, field goals, extra points, and kickoff returns. |
| `presnap_penalty_rate` | Pre-Snap Pen % | rate | offensive snaps | 1999 | PBP | False starts and delay-of-game penalties per offensive snap, a measure of pre-snap discipline. Other pre-snap fouls, such as illegal formation, are not counted. |
| `penalty_rate_per_offensive_snap` | Off Pen % | rate | offensive snaps | 1999 | PBP | Offensive penalties (special-teams plays with the ball included) divided by offensive snaps. |

## Defense

Everything the team allowed, plus the plays its defense made.

### Defense: Total

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `defensive_snaps` | Def Snaps | count | - | 1999 | PBP | Plays opponents ran from scrimmage against the defense: dropbacks (pass attempts, sacks, and scrambles), designed runs, kneel-downs, and spikes. |
| `total_yards_allowed` | Total Yds Allowed | count | - | 1999 | PBP +TS | Passing plus rushing yards opponents gained against the defense. Passing yards here are before sack losses, so this runs higher than the NFL's official total yards. |
| `first_downs_allowed` | 1Ds Allowed | count | - | 1999 | PBP | Total first downs given up by rush, pass, or penalty. |
| `yards_per_defensive_snap_allowed` | Yds/Snap Allowed | rate | defensive snaps | 1999 | PBP | Yards opponents gained per scrimmage play against the defense. Formula: (Opponent passing yards before sack losses + rushing yards) ÷ defensive snaps |
| `epa_per_defensive_snap_allowed` | EPA/Snap Allowed | rate | defensive snaps | 1999 | PBP | Expected points added (EPA) per opponent scrimmage play: how much a typical play raised or lowered their expected points. Team seasons usually fall between about -0.15 and +0.1. |
| `success_rate_allowed` | Success % Allowed | rate | defensive snaps | 1999 | PBP | Share of opponent scrimmage plays that raised their expected points (positive EPA). |
| `explosive_play_rate_allowed` | Explosive % Allowed | rate | defensive snaps | 1999 | PBP | Share of opponent scrimmage plays that were big gains: completions of 20+ yards or runs of 10+ yards. Formula: (Opponent completions of 20+ yards + carries of 10+ yards) ÷ defensive snaps |

### Defense: Passing

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `passing_yards_allowed` | Pass Yds Allowed | count | - | 1999 | PBP +TS | Passing yards opponents gained against the defense, before subtracting yards lost on sacks. |
| `passing_epa_allowed` | Pass EPA Allowed | count | - | 1999 | PBP +TS | Expected points added (EPA) on opponents' pass plays: how much those plays raised or lowered their expected points, given down, distance, and field position. League average is usually a little above zero. Formula: Sum of opponent EPA on pass attempts and sacks (QB scrambles not included) |
| `passing_tds_allowed` | Pass TDs Allowed | count | - | 1999 | PBP +TS | Touchdown passes given up. |
| `passing_first_downs_allowed` | Pass 1Ds Allowed | count | - | 1999 | PBP +TS | First downs given up through the air. |
| `passing_cpoe_allowed` | CPOE Allowed | avg | opponent pass attempts (model-expected completions) | 2006 | PBP +TS | How much more often opponents completed passes than expected, given how hard each throw was, in percentage points. Below zero means they completed fewer than expected; league average is near zero. Formula: Actual minus expected completion chance on each opponent pass attempt, averaged over the attempts, in percentage points |
| `attempts_faced` | Att Faced | count | - | 1999 | PBP +TS | Passes opponents threw against the defense. Sacks and two-point tries are not counted as attempts. |
| `completions_allowed` | Comp Allowed | count | - | 1999 | PBP +TS | Opponent passes completed against this defense. |
| `completion_pct_allowed` | Comp % Allowed | rate | opponent pass attempts | 1999 | PBP | The share of opponent attempts completed against this defense. |
| `net_passing_yards_allowed` | Net Pass Yds Allowed | count | - | 1999 | PBP | Passing yards allowed minus opponent sack yardage lost. |
| `epa_per_dropback_allowed` | EPA/Dropback Allowed | rate | opponent dropbacks | 1999 | PBP | Expected points added (EPA) per opponent dropback (pass attempt, sack, or scramble): how much a typical pass play raised or lowered their expected points. Team seasons usually fall between about -0.15 and +0.2. |
| `any_a_allowed` | ANY/A Allowed | rate | opponent pass attempts + sacks | 1999 | PBP | Opposing passers' yards per pass play after adding a bonus for touchdowns and subtracting for interceptions and sack losses. Formula: (Opponent passing yards + 20 x TD passes - 45 x interceptions - sack yards lost) ÷ (opponent pass attempts + sacks) |
| `explosive_pass_rate_allowed` | Expl Pass % Allowed | rate | opponent dropbacks | 1999 | PBP | Share of opponent dropbacks that ended in a completion of 20+ yards. |
| `air_yards_allowed` | Air Yds Allowed | count | - | 2006 | PBP | How far opponents' passes traveled past the line of scrimmage, added up over every attempt, caught or not. Shows how far downfield opponents threw. |
| `yac_allowed` | YAC Allowed | count | - | 2006 | PBP | Yards opponents' receivers gained after the catch on completed passes; a rough read on tackling and pursuit. |
| `team_passer_rating_allowed` | Passer Rtg Allowed | rate | official NFL formula over opponent attempts | 1999 | PBP | The NFL passer rating of all opposing quarterbacks combined, on the usual 0 to 158.3 scale. Formula: The passer rating formula applied to opponents' combined completions, attempts, passing yards, TD passes, and interceptions |
| `deep_attempt_rate_faced` | Deep Att % Faced | rate | opponent pass attempts | 2006 | PBP | Share of opponent pass attempts thrown deep (about 16+ yards past the line of scrimmage). Describes how opponents attacked the defense, not how well it played. |

### Defense: Rushing

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `rushing_yards_allowed` | Rush Yds Allowed | count | - | 1999 | PBP +TS | Rushing yards opponents gained against the defense, with quarterback scrambles and kneel-downs counted as official stats count them. |
| `rushing_epa_allowed` | Rush EPA Allowed | count | - | 1999 | PBP +TS | Expected points added (EPA) on opponents' runs, including quarterback scrambles and kneel-downs. League average is below zero, because the average run loses a little expected value. |
| `rushing_tds_allowed` | Rush TDs Allowed | count | - | 1999 | PBP +TS | Rushing touchdowns given up. |
| `rushing_first_downs_allowed` | Rush 1Ds Allowed | count | - | 1999 | PBP +TS | First downs given up on the ground. |
| `carries_faced` | Carries Faced | count | - | 1999 | PBP +TS | Opponent rushing attempts against the defense, including quarterback scrambles and kneel-downs. |
| `yards_per_carry_allowed` | Yds/Carry Allowed | rate | opponent carries | 1999 | PBP | Rushing yards opponents averaged per carry, with scrambles and kneel-downs counted as official stats count them. |
| `rush_success_rate_allowed` | Rush Success Allowed | rate | opponent designed carries | 1999 | PBP | Share of opponent designed runs (not scrambles or kneel-downs) that raised their expected points. |
| `explosive_rush_rate_allowed` | Expl Rush % Allowed | rate | opponent carries | 1999 | PBP | Share of opponent carries that gained 10+ yards. Formula: Opponent carries of 10+ yards ÷ opponent carries (scrambles and kneel-downs included) |

### Defense: Receiving

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `targets_faced` | Targets Faced | count | - | 1999 | PBP +TS | Opponent pass attempts thrown to an intended receiver. Throwaways and spikes are attempts but not targets. Blank in 2003-2008, when nflverse play-by-play names the intended receiver on almost no incomplete passes. |
| `receptions_allowed` | Rec Allowed | count | - | 1999 | PBP +TS | Opponent catches — the same number as completions allowed. Same value as `completions_allowed`. |
| `receiving_yards_allowed` | Rec Yds Allowed | count | - | 1999 | PBP +TS | Opponent receiving yards — equals passing yards allowed. Same value as `passing_yards_allowed`. |
| `catch_rate_allowed` | Catch % Allowed | rate | opponent targets | 1999 | PBP | The share of opponent targets that were caught. Throwaways and spikes are not targets, so unlike completion percentage allowed they do not count as misses. Blank in 2003-2008, when nflverse play-by-play names the intended receiver on almost no incomplete passes. |

### Defense: Scoring

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `points_per_drive_allowed` | Pts/Drive Allowed | rate | opponent drives | 1999 | PBP | Points opponents scored per possession against the defense, extra points and two-point tries included. |
| `red_zone_td_pct_allowed` | RZ TD % Allowed | rate | opponent red-zone trips | 1999 | PBP | The share of opponent red-zone trips that ended in touchdowns. |
| `goal_to_go_td_pct_allowed` | G2G TD % Allowed | rate | opponent goal-to-go series | 1999 | PBP | Share of opponent goal-to-go series (sets of downs where the line to gain is the goal line) that ended in a touchdown. |
| `two_pt_conversion_rate_allowed` | 2-Pt % Allowed | rate | opponent two-point attempts | 1999 | PBP | Opponent two-point conversion success rate against this defense. |

### Defense: Downs & Conversions

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `third_down_pct_allowed` | 3rd Down % Allowed | rate | opponent third-down attempts | 1999 | PBP | Opponent third-down conversion rate against this defense. |
| `fourth_down_pct_allowed` | 4th Down % Allowed | rate | opponent fourth-down attempts | 1999 | PBP | Share of opponent fourth-down go-for-it plays that converted. |
| `series_conversion_rate_allowed` | Series Conv Allowed | rate | opponent series | 1999 | PBP | Share of opponent series (sets of downs) that earned a new first down or a touchdown. A series that ends in a field goal, punt, or turnover counts as a stop. |
| `three_and_outs_forced_rate` | 3-and-Outs Forced % | rate | opponent drives | 1999 | PBP | Share of opponent drives that ended in a punt without gaining a first down. |

### Defense: Drives & Field Position

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `score_pct_per_drive_allowed` | Score % Allowed | rate | opponent drives | 1999 | PBP | Share of opponent drives that ended in a touchdown or field goal. |
| `punts_forced_pct` | Punts Forced % | rate | opponent drives | 1999 | PBP | The share of opponent possessions that ended in a punt. |
| `avg_starting_field_position_allowed` | Avg Start Allowed | avg | opponent drives | 1999 | PBP | Where opponent drives started on average, in yards from their own goal line (25 means their own 25). Driven mostly by this team's kickoffs, punts, and turnovers. |

### Defense: Turnovers

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `def_interceptions` | Def INTs | count | - | 1999 | PLS | Passes intercepted by the defense. |
| `takeaways` | Takeaways | count | - | 1999 | PBP | Interceptions plus opponent fumbles recovered. |
| `def_interception_yards` | INT Ret Yds | count | - | 1999 | PBP +TS | Yards gained returning interceptions. |
| `fumble_recovery_opp` | Fumbles Recovered | count | - | 1999 | PBP +TS | Opponent fumbles this defense recovered. |
| `takeaway_rate_per_defensive_snap` | Takeaway % | rate | defensive snaps | 1999 | PBP | Share of opponent scrimmage plays that ended in an interception or a fumble the defense recovered. |
| `takeaways_per_drive` | Takeaways/Drive | rate | opponent drives | 1999 | PBP | Takeaways divided by opponent possessions. |
| `takeaway_epa` | Takeaway EPA | count | - | 1999 | PBP | Expected points opponents lost on plays where the team took the ball away (interceptions and opponent fumbles it recovered, including muffed punts and return fumbles), so each takeaway is weighed by how costly it was. Formula: -(Opponent Turnover EPA) |
| `def_tds` | Def & Return TDs | count | - | 1999 | PBP +TS | Touchdowns this team scored while the opponent had the ball: interception and fumble returns, plus scores on the opponent's punts and kicks, such as punt returns. |
| `fumble_recovery_tds` | Fumble Return TDs | count | - | 1999 | PBP +TS | Touchdowns scored returning recovered fumbles. |

### Defense: Pressure & Playmaking

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `def_sacks` | Def Sacks | count | - | 1999 | PLS | Times the defense sacked the opposing quarterback. |
| `def_qb_hits` | Def QB Hits | count | - | 1999 | PLS | Times the defense's players hit the opposing quarterback, sacks included. Formula: Sum of QB hits credited to defenders (a hit shared by two defenders counts twice) Blank in 2003-2005, when nflverse records no QB hits; in 1999-2002 it records them only on sacks. |
| `def_tackles_for_loss` | Def TFL | count | - | 1999 | PLS | Tackles behind the line of scrimmage credited to the defense's players. Most sacks also count as a tackle for loss. Blank in 2003-2011, when nflverse's weekly player stats credit no tackles for loss. |
| `def_pass_defended` | Passes Defended | count | - | 1999 | PLS | Passes the defense's players broke up, tipped, or intercepted. Interceptions count as passes defended. Formula: Sum of passes defended credited to defenders (a pass defended by two players counts twice) |
| `def_fumbles_forced` | Forced Fumbles | count | - | 1999 | PLS | Fumbles the defense's players forced, whether or not the defense recovered the ball. |
| `def_safeties` | Def Safeties | count | - | 1999 | PLS | Safeties forced by the defense (two points each). |
| `def_sack_yards` | Sack Yds Forced | count | - | 1999 | PBP +TS | Yards opponents lost to this defense's sacks. |
| `def_sack_rate_per_dropback` | Sack Rate Forced | rate | opponent dropbacks | 1999 | PBP | Share of opponent dropbacks that ended in a sack. Formula: Sacks ÷ opponent dropbacks (pass attempts, sacks, and scrambles) |
| `qb_pressure_events_rate` | Sack+Hit Rate | rate | opponent dropbacks | 1999 | PBP | Sacks plus quarterback hits per opponent dropback. Most sacks are also logged as hits, so they usually count twice; this is not the share of dropbacks with pressure. Formula: (Sacks + plays with a QB hit) ÷ opponent dropbacks Blank in 2003-2005, when nflverse records no QB hits; in 1999-2002 it records them only on sacks. |
| `stuff_rate` | Stuff % | rate | opponent carries other than kneel-downs | 1999 | PBP | The share of opponent carries other than kneel-downs stopped for no gain or a loss. |
| `havoc_rate` | Havoc % | rate | defensive snaps | 1999 | PBP | Share of opponent scrimmage plays with a disruptive play by the defense: a tackle for loss, forced fumble, interception, or pass defended. Sacks are not counted unless they also force a fumble. Formula: Opponent plays with a tackle for loss, forced fumble, interception, or pass defended ÷ defensive snaps |
| `defensive_2pt_conversions` | Def 2-Pt Returns | count | - | 1999 | PBP | Times the defense returned a blocked extra point or a turnover on a two-point try to the other end zone for two points. |

### Defense: Penalties

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `defensive_penalties` | Def Penalties | count | - | 1999 | PBP | Penalties committed on plays where the opponent had the ball, including the team's kickoffs, punt returns, and field-goal and extra-point defense. |
| `defensive_penalty_yards` | Def Pen Yds | count | - | 1999 | PBP | Penalty yards assessed on plays where the opponent had the ball, including the team's kickoffs, punt returns, and field-goal and extra-point defense. |
| `defensive_pass_interference` | DPI | count | - | 1999 | PBP | Defensive pass interference penalties committed. |
| `penalty_first_downs_allowed` | Pen 1Ds Allowed | count | - | 1999 | PBP | First downs opponents got from this team's penalties rather than by running or passing. |

## Special Teams

Kicks, punts, returns, field goals, and extra points.

| Column | Label | Shape | Per | Since | Source | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `st_plays` | ST Plays | count | - | 1999 | PBP | Special-teams plays with the ball: punts, field goals, and extra points, plus kickoffs received. |
| `st_epa` | ST EPA | count | - | 1999 | PBP | Expected points added on special-teams plays with the ball: punts, field goals, extra points, and kickoffs received. Punt returns and kickoff coverage are not included. |
