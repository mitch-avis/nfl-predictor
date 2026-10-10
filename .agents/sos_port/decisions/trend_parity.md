# Decisions in giving a game's completed and upcoming rows the same trends (roadmap step 6)

An independent review of the step-6 integration branch (absorbing `nfl-sos-ratings`) found that a
game's `scoring_margin_4wk_trend` and `turnover_margin_4wk_trend` (each side, and their
away-minus-home differences) differ between its completed row and its upcoming row: with 2025
week 18 made unplayed, in all 16 games, by up to 15.75 points (22.625 on the difference). The
model trains on completed rows and predicts upcoming ones, so it learned one version of these
columns and predicted with the other. The reviewer saw the same at the integration branch's base,
so the defect predates step 6. This file records the cause, each decision with its reasoning and
data, and the measured movement. The fix changes what a weekly run produces (the upcoming rows),
so it is held for the user's approval with the rest of step 6.

Commands run from this repository's root with the worktree's environment. Every number below
comes from one of three scripts beside this file, each with its output checked in beside it.
None writes outside a temporary directory, and the two that build do so offline (the network
refused) from read-only copies of the frozen step-6 inputs (`~/scratch/step6/data/`, manifest
`~/scratch/step6_data_snapshot_20261010.sha256`) and of the integration build's play-by-play
cache (`/tmp/step6_int/cache/`, byte-identical to the reviewer's `/tmp/rev_ab77/cache/` and,
for the schedules and team stats, to the frozen inputs). That cache is needed because this branch
requests about 90 play-by-play columns the frozen cache predates, so the ETL would download every
season again; `--download-stale-pbp` does that into the temporary copy when the cache is gone.

- **The scenario**, `trend_parity_check.py`: builds 2025 through the ETL twice, as played and with
  week 18 unplayed (scores, result, total and overtime blanked in the schedule, the week's 32
  team-stat rows and 2,636 plays removed; the ELO file, TeamRankings and the nfelo lines kept,
  since an upcoming week has them before kickoff), and compares every column of the 16 games. It
  also checks every published completed regular-season row for its own game, and counts the ELO
  trends' fallback in the latest published season. Output with the code before the fix
  (`--base a000802`, the integration branch's tip when this work started):
  `trend_parity_check_before_output.txt`; with the fix: `trend_parity_check_output.txt`.
- **The movement measurement**, `trend_parity_movement.py --base a000802`: builds 1999-2025 twice
  from one copy of the inputs, with the code before the fix and with the fix, and compares every
  column of every game row. Output: `trend_parity_movement_output.txt`.
- **The audit's reach**, `trend_parity_audit_reach.py`: how strongly the scoring-margin trend
  tracks the game margin, as published and in a deliberately leaking version. Output:
  `trend_parity_audit_reach_output.txt`.

## The cause

1. **Which version is correct.** The completed row's. A week-`N` trend may use only the team's
   games strictly before week `N`, and the completed row's value is exactly that: the mean of the
   team's last four earlier games minus its season-to-date mean over all earlier games. The
   upcoming row's `0.0` is not a trend at all: it is the default the week builder writes when it
   finds no trend row.
2. **Whether the completed row includes its own game.** No, so this is not leakage. In the code,
   `build_team_stat_trends` shifts each stat back one game within the team before the rolling and
   cumulative means (`shifted = pl.col(stat).shift(1)`,
   `src/nfl_predictor/utils/polars/features.py` line 1095 at `a000802`), so a row's own game never
   enters its value. On the data, every completed regular-season row of the frozen inputs'
   published `all_data_ml.csv` (1999-2026, 14,034 team-games) equals the trend from the team's
   earlier games recomputed from the schedule's scores; of the 10,471 rows where counting the row's
   own game would change the value, none matches that version (the scenario, section 5).
3. **What differs between the two code paths.** The trend frame's keys. `build_team_stat_trends`
   (`features.py` lines 1048-1109 at `a000802`) wrote one row per row of the per-team-game frame,
   and that frame holds only completed regular-season games: `build_team_game_skeleton`
   (`src/nfl_predictor/utils/polars/loaders.py` lines 677-705) keeps games with both scores
   (line 699) of type `REG` (line 703), and the ETL loads regular-season team stats only
   (`src/nfl_predictor/data_collection.py` line 703). The week builder joins the frame on season,
   week and team exactly (`_merge_team_trends`,
   `src/nfl_predictor/utils/polars/week_rows.py` lines 560-561). A completed regular-season game
   finds its own row, computed from the earlier games; an upcoming game finds none, and neither
   does any playoff game, played or not, so `_fill_trend_and_coach_defaults` (`week_rows.py` line
   428) writes `0.0` for both sides and the difference.
4. **Which other trend or rolling columns share the path.** The scenario compares every column of
   the 16 games. Before the fix, only the six team-stat trend columns, the two scores and five line
   columns differ; after it, only the scores and the line columns. The line columns differ by
   design: an unplayed game takes nfelo's latest line and a completed one its opener
   (`src/nfl_predictor/utils/polars/pick_time_lines.py`), the approximation `AGENTS.md` rule 11
   records; for a past game made unplayed the latest line is its closing line, hence gaps of up to
   7 points of spread. Of the other trend and rolling families:
   - the ELO trend (`elo_4wk_trend`) and the two quarterback trends (`qb_elo_4wk_trend`,
     `qb_value_4wk_trend`) take the same exact join (`week_rows.py` lines 521 and 523; the
     quarterback trends on season, week and quarterback name, lines 589-590), but their frames
     are keyed by the rows of `qb_elos.csv`, which lists the upcoming week's games with pre-game
     ratings, and the value uses only pre-game ratings, so they agree whenever the file carries
     the week (decision 3);
   - the TeamRankings rating trend (`last_5_games_rating_trend`) is the difference of two ratings
     from the week's scraped TeamRankings file (`week_rows.py` line 668), the same file for a
     completed and an upcoming row;
   - the coach priors are keyed by every scheduled game, played or not (`build_coach_features`,
     `features.py` line 1160), so they were already built the way this fix builds the trends;
   - the season-to-date statistics, records, standings, lookahead, schedule-adjusted strength,
     schedule strength and quarterback rates are computed for each week from games strictly
     before it and joined on the team alone.

## 1. Every scheduled team-week gets its trend from the team's earlier games

- **What differs**: the trend frame had a row only for a completed regular-season team-game, and
  every other game of the schedule took `0.0`.
- **Options**: (a) key `build_team_stat_trends` on the season's schedule: every scheduled
  team-week, played or not, takes the team's trends after its last game strictly before that week
  (a backward as-of join that never matches the week itself, the pattern the quarterback rates
  use in `src/nfl_predictor/utils/polars/qb_stats.py`), with the schedule passed in by the one
  caller, `data_collection._with_season_features`; (b) leave the frame keyed by completed games
  and have the week builder look up each team's trends after its last earlier game, so the
  frame's `week` would mean the week of the game the trends follow rather than the week they are
  used in; (c) have `build_team_stat_trends` write every week from 1 through the last playoff week
  for every team, with no schedule; (d) compute the trends inside the week builder from the
  team-stat rows before the week, as the season-to-date statistics are, and drop the precomputed
  season frame; (e) give the completed rows `0.0` too.
- **Decision**: (a).
- **Reasoning**: the frame keeps its meaning, the trend a team carries into its game in week `N`
  from games strictly before `N`, and the week builder's exact join stays as it is. A completed
  row's value is computed from the same games with the same windows as before (the trends after
  game `j` are the shifted values at game `j + 1`, nulls included), so no completed regular-season
  row moves, which the movement measurement confirms. The coach priors already cover every
  scheduled game the same way. (b) would leave a frame whose row for week `N` includes week `N`'s
  game, the leak this investigation looked for, one exact join away. (c) needs the last playoff
  week as a constant and writes weeks without a game. (d) moves the work into the week builder and
  changes more of `data_collection.py`, which another branch is changing now. (e) makes the rows
  agree by throwing the signal away and moves every training row with a non-zero trend.
- **Data**: in the scenario before the fix, all 16 upcoming games carry `0.0` in the four side
  columns and both differences while their completed rows carry up to 15.75 (22.625 on the
  difference), the reviewer's numbers; after it, no trend column differs and each game's four
  values match its completed row's (section 4 of each output). The played build reproduces the
  published values of the week exactly, before and after (largest gap 0.0), so the temporary build
  is the ETL that published them. `tests/test_trend_parity.py` builds a small season twice through
  `data_collection.process_season`, played and with one week and everything after it unplayed, and
  requires every trend column of that week to match, for the last regular-season week and for the
  playoff round; on the code before the fix both cases fail (the first on the values, the second
  because the completed playoff rows carry no trend). `tests/test_trend_features.py` adds
  `test_build_team_stat_trends_gives_an_unplayed_week_the_trend_from_earlier_games`,
  `test_build_team_stat_trends_gives_a_playoff_week_the_full_regular_season_trend` and
  `test_build_team_stat_trends_is_zero_for_a_team_without_an_earlier_game`; the existing test of
  the trend definition passes unchanged apart from the schedule it now passes.
  `test_build_team_stat_trends_keeps_a_game_without_a_value_in_the_window` pins the missing-value
  handling, which the code before the fix shares for every played week: a game without a value
  fills a slot of the four-game window but counts in neither mean, and the week right after it
  has no trend (published as `0.0`).
- **Feature-value impact**: no training row moves. Over 1999-2025 none of the 6,952 completed
  regular-season rows changes in any season, and no column other than the six team-stat trend
  columns changes anywhere (the movement measurement, sections 2 and 3). A walk-forward without
  `--include-postseason` trains and scores on exactly those rows, so no benchmark result changes:
  the benchmark already measured the model with these trends, and the fix makes production predict
  with them too (the gap `AGENTS.md` rule 11 forbids, closed without a new walk-forward). Every
  upcoming row changes: it carried `0.0` and now carries the value its completed row will carry.
  Over every season's completed regular-season rows that value is `0.0` through week 5 by
  construction (with four or fewer earlier games, the last four are the whole season), non-zero for
  72.8% of team sides in week 6 (mostly teams that have had their bye) and 98.8-99.9% from week 7
  on; the mean absolute scoring-margin trend is 1.76 points in week 6, 2.766-4.1 in weeks 7-12 and
  4.349-5.055 in weeks 13-18 (largest 22.0, week 15), and the mean absolute turnover-margin trend
  0.248-0.645 (largest 3.017, week 17) (section 4). In 2025 week 18 the upcoming rows move by up to
  15.75 on a side and 22.625 on the difference (the scenario). A weekly run's predictions change
  from week 6, the first week a team can have five earlier games, so the next affected run is 2026
  week 6. The season-build characterization snapshot changes in two rows of its unplayed week: the
  home-side trends and the two differences.
- **How to reverse it**: restore `build_team_stat_trends` to its shifted computation over the
  per-team-game rows, drop its `schedule_df` argument and the caller's, and rewrite the
  season-build characterization snapshot (`NFLP_UPDATE_SNAPSHOTS=1`).
- **Review**: not yet reviewed.

## 2. A playoff week takes the trend over the full regular season

- **What differs**: the per-team-game frame holds no playoff games, so before the fix every
  playoff row, completed or not, carried `0.0`. Under decision 1 a playoff week is a scheduled
  team-week like any other and takes the trends after the team's last earlier game in the frame:
  its last regular-season game.
- **Options**: (a) let it, so a playoff row carries the trend over the full regular season; (b)
  keep playoff rows at `0.0` by keying only the regular-season games; (c) count playoff games
  too.
- **Decision**: (a).
- **Reasoning**: this repository's rule for playoff rows is that they use the full regular season
  (`.agents/TODO.md`, "Guardrails checklist"), which the season-to-date statistics and the
  strength features already follow. A constant `0.0` is a missing value presented as "no trend".
  Training uses regular-season rows only by default (`include_postseason: false` in
  `config/weekly_run.yaml`, the walk-forward and the final fit), where late-season rows carry real
  trends, so a playoff prediction with `0.0` sits outside what the model learned. (b) keeps that
  mismatch. (c) needs playoff team stats, which the ETL does not load at all; the season-to-date
  features leave them out too.
- **Data**: `test_build_team_stat_trends_gives_a_playoff_week_the_full_regular_season_trend` and
  the playoff case of `tests/test_trend_parity.py`. In the season-build characterization fixture
  the playoff round did not move, because that fixture writes team-stat rows for its playoff game
  (the ETL never does), so the old code already computed it from the earlier games.
- **Feature-value impact**: all 309 completed playoff rows of 1999-2025 move, from `0.0` to the
  trend over the full regular season (11 a season through 2019, 13 from 2020). Per season, the
  scoring-margin difference moves by 4.343891-13.340909 on average and by up to 28.705883 (2024);
  the turnover-margin difference by 0.283937-1.719318 on average and by up to 4.8125 (2003) (the
  movement measurement, section 2). These rows are in `completed_games_ml.csv`, but training
  leaves them out by default (`include_postseason: false`), so no default training or
  walk-forward row changes; with `include_postseason: true` they would. Playoff predictions
  change the same way.
- **How to reverse it**: keep only `game_type == "REG"` games in `_scheduled_team_weeks`
  (`features.py`); playoff rows then find no trend row and take `0.0` again.
- **Review**: not yet reviewed.

## 3. The ELO, quarterback and TeamRankings trends stay as they are

- **What differs**: the ELO trend and the two quarterback trends go through the same exact join,
  keyed by the rows of `qb_elos.csv`; the TeamRankings trend comes from the week's scraped file.
- **Options**: (a) leave them; (b) give them the as-of treatment of decision 1.
- **Decision**: (a).
- **Reasoning**: their inputs exist for an upcoming week before kickoff (`../nfeloqb` lists the
  upcoming week's games with pre-game ratings, and a week's TeamRankings file is scraped before
  the week), and the week-`N` value uses only those pre-game inputs, so a completed and an
  upcoming row agree whenever the input carries the week; in the scenario none of them differs.
  When `qb_elos.csv` lacks the predicted week, the ELO levels fall back to each team's latest
  pre-game rating (`_merge_elo`, `week_rows.py` line 304), the team ELO trend becomes `0.0` and
  the quarterback trends null (`_merge_qb_trends` finds no quarterback names, `week_rows.py`
  line 576). That is a stale input rather than this defect, and an as-of rule would not repair it,
  because the week-`N` ELO trend needs the week-`N` pre-game rating.
- **Data**: the scenario's section 3 lists no ELO, quarterback or TeamRankings trend column. Its
  section 6 shows the fallback in the frozen inputs' published `all_data_ml.csv` (built
  2026-10-09): none of the 14 unplayed 2026 week-5 games, which `qb_elos.csv` lists, has a `0.0`
  team ELO trend on both sides or a missing quarterback trend, while every unplayed game of weeks
  6-18, which it does not list yet, has both.
- **Feature-value impact**: none.
- **How to reverse it**: nothing to reverse.
- **Review**: not yet reviewed.

## 4. The leakage audit could not have caught this, and it is not changed here

- **What differs**: `nfl-predictor leakage-audit` (`src/nfl_predictor/ml/leakage_audit.py`)
  reads one completed-games CSV and checks for target columns among the features, suspicious
  names, and a correlation with the home margin or total of at least 0.995 in absolute value (a
  flag; 0.999999 fails) or near-equality with a score. The third check `.agents/modeling_spec.md`
  requires ("validates season-to-date features exclude the current game row") has no code in it.
- **Options**: (a) no audit change in this task, with the parity test pinning the property and a
  question for the user; (b) add a completed-against-upcoming parity check to the audit, as the
  scenario script does; (c) implement the spec's third check by recomputing each season-to-date
  family with and without the row's own game.
- **Decision**: (a).
- **Reasoning**: the audit should not have caught this defect and could not: it is a mismatch
  between completed and upcoming rows, not leakage, and the audit never sees an upcoming row. It
  would not have caught the leak the reviewer feared either: the published trend difference
  correlates -0.0244 with the home margin, and a version recomputed to include each game's own
  result only -0.4237, far from the 0.995 flag. Only the unimplemented third check, or a parity
  check, could see either. Both are audit features with their own design (the audit reads one CSV
  today; a parity check needs the ETL's inputs), so they go to the user rather than into this fix.
- **Data**: `trend_parity_audit_reach_output.txt` (7,017 completed regular-season games of
  1999-2026); `leakage_audit.py`, `_audit_column` and `run_leakage_audit`. The audit itself,
  `.venv/bin/nfl-predictor leakage-audit --data-path ~/scratch/step6/data/completed_games_ml.csv
  --out-json /tmp/trend_parity/leakage_audit.json`, passes on the frozen inputs' published
  completed games (`ok`, 7,326 rows, 461 features, no flagged column, no failure): the defect is
  not in that file at all, only in the upcoming rows the audit never reads.
- **Feature-value impact**: none.
- **How to reverse it**: nothing to reverse.
- **Review**: not yet reviewed.
