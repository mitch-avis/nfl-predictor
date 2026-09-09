# nflreadpy PBP Schema Survey

Written 2026-09-09. Produced by loading each season individually via
`nflreadpy.load_pbp(seasons=[season])` (nflreadpy `0.1.5`, in-memory cache only) with a
throwaway script in the scratch directory, then hand-writing this report from the captured
stdout. No files were written under `data/`, no repo Python file was edited.

## Per-season summary

| Season | Rows  | Columns | `season_type` values | Load seconds |
| ------ | ----- | ------- | -------------------- | ------------ |
| 1999   | 46136 | 372     | POST, REG            | 0.59         |
| 2003   | 46811 | 372     | POST, REG            | 0.47         |
| 2006   | 46299 | 372     | POST, REG            | 0.57         |
| 2015   | 48122 | 372     | POST, REG            | 0.56         |
| 2024   | 49492 | 372     | POST, REG            | 0.55         |

All five seasons loaded without error. `season_type` is present in every season and only ever
takes `POST` / `REG` (no `PRE` rows in PBP). Total column count is a stable `372` across all five
seasons, so the schema shape has not changed structurally since 1999 (individual columns can
still be all-null in older seasons; see below).

Load times above are for a single in-process run in this environment (effectively a fast
download, no on-disk cache reuse across the two scripts). Do not assume this speed holds under
different network conditions; the one-season-at-a-time constraint is about memory, not time.

## Candidate columns, part 1 (game_id .. down)

Cell format is `dtype / null_rate` (4 decimals) or `-` if the column is absent.

| Column | 1999 | 2003 | 2006 | 2015 | 2024 |
| --- | --- | --- | --- | --- | --- |
| game_id | String / 0.0000 | String / 0.0000 | String / 0.0000 | String / 0.0000 | String / 0.0000 |
| season | Int32 / 0.0000 | Int32 / 0.0000 | Int32 / 0.0000 | Int32 / 0.0000 | Int32 / 0.0000 |
| season_type | String / 0.0000 | String / 0.0000 | String / 0.0000 | String / 0.0000 | String / 0.0000 |
| week | Int32 / 0.0000 | Int32 / 0.0000 | Int32 / 0.0000 | Int32 / 0.0000 | Int32 / 0.0000 |
| posteam | String / 0.0114 | String / 0.0514 | String / 0.0521 | String / 0.0510 | String / 0.0548 |
| defteam | String / 0.0114 | String / 0.0514 | String / 0.0521 | String / 0.0510 | String / 0.0548 |
| home_team | String / 0.0000 | String / 0.0000 | String / 0.0000 | String / 0.0000 | String / 0.0000 |
| away_team | String / 0.0000 | String / 0.0000 | String / 0.0000 | String / 0.0000 | String / 0.0000 |
| posteam_type | String / 0.0114 | String / 0.0514 | String / 0.0521 | String / 0.0510 | String / 0.0548 |
| play_type | String / 0.0812 | String / 0.0304 | String / 0.0302 | String / 0.0293 | String / 0.0292 |
| qb_dropback | Float64 / 0.0812 | Float64 / 0.0304 | Float64 / 0.0302 | Float64 / 0.0293 | Float64 / 0.0292 |
| qb_kneel | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 |
| qb_spike | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 |
| rush | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 |
| pass | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 |
| epa | Float64 / 0.0112 | Float64 / 0.0115 | Float64 / 0.0117 | Float64 / 0.0113 | Float64 / 0.0115 |
| qb_epa | Float64 / 0.0112 | Float64 / 0.0115 | Float64 / 0.0117 | Float64 / 0.0113 | Float64 / 0.0115 |
| success | Float64 / 0.0112 | Float64 / 0.0115 | Float64 / 0.0117 | Float64 / 0.0113 | Float64 / 0.0115 |
| yards_gained | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| air_yards | Float64 / 1.0000 | Float64 / 1.0000 | Float64 / 0.6305 | Float64 / 0.6031 | Float64 / 0.6246 |
| cpoe | Float64 / 1.0000 | Float64 / 1.0000 | Float64 / 0.6582 | Float64 / 0.6081 | Float64 / 0.6414 |
| xpass | Float64 / 1.0000 | Float64 / 1.0000 | Float64 / 0.2410 | Float64 / 0.2397 | Float64 / 0.2385 |
| pass_oe | Float64 / 1.0000 | Float64 / 1.0000 | Float64 / 0.2647 | Float64 / 0.2609 | Float64 / 0.2611 |
| down | Float64 / 0.1525 | Float64 / 0.1518 | Float64 / 0.1527 | Float64 / 0.1547 | Float64 / 0.1618 |

## Candidate columns, part 2 (ydstogo .. passer_player_name)

| Column | 1999 | 2003 | 2006 | 2015 | 2024 |
| --- | --- | --- | --- | --- | --- |
| ydstogo | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 |
| yardline_100 | Float64 / 0.0120 | Float64 / 0.0690 | Float64 / 0.0695 | Float64 / 0.0680 | Float64 / 0.0716 |
| third_down_converted | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| third_down_failed | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| fourth_down_converted | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| fourth_down_failed | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| two_point_attempt | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| two_point_conv_result | String / 0.9980 | String / 0.9985 | String / 0.9991 | String / 0.9979 | String / 0.9970 |
| td_team | String / 0.9739 | String / 0.9733 | String / 0.9734 | String / 0.9716 | String / 0.9705 |
| sack | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| interception | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| fumble_lost | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| complete_pass | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| pass_touchdown | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| rush_touchdown | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| first_down | Float64 / 0.0377 | Float64 / 0.0311 | Float64 / 0.0310 | Float64 / 0.0304 | Float64 / 0.0307 |
| special | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 |
| special_teams_play | Float64 / 1.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 |
| fixed_drive | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 | Float64 / 0.0000 |
| fixed_drive_result | String / 0.0050 | String / 0.0001 | String / 0.0000 | String / 0.0000 | String / 0.0000 |
| drive_start_yard_line | String / 0.0027 | String / 0.0151 | String / 0.0141 | String / 0.0128 | String / 0.0127 |
| passer_player_id | String / 0.5937 | String / 0.6061 | String / 0.6029 | String / 0.5754 | String / 0.5942 |
| passer_player_name | String / 0.5937 | String / 0.6061 | String / 0.6029 | String / 0.5754 | String / 0.5942 |

Every one of the 47 candidate columns is **present** (never absent) in all five surveyed
seasons. The `special_teams_play / 1.0000` cell for 1999 means the column exists but every
value is null that year; see the flag section below for why that matters.

## Special-teams flag

`special` is `Float64`, encoding true as `1.0` / false as `0.0`, and is fully populated
(0% null) in every surveyed season, 1999 through 2024. `special_teams_play` is also
`Float64` with the same `1.0`/`0.0` encoding when it works, but:

- **1999**: `special_teams_play` is present but **100% null** — unusable that year.
- **2003**: `special_teams_play` is present, 0% null, but its only observed value is `0.0`
  (`frac_flagged_true = 0.0`) — the column exists but never fires, so it is also unusable
  that year (looks like an upstream data gap, not a real "no special teams plays" season).
- **2006, 2015, 2024**: `special_teams_play` is present, populated, and has real `0.0`/`1.0`
  values.

Per-season fraction of rows flagged true:

| Season | `special` true rate | `special_teams_play` true rate |
| ------ | ------------------- | ------------------------------ |
| 1999   | 0.1571              | n/a (all null)                 |
| 2003   | 0.1568              | 0.0000 (always false)          |
| 2006   | 0.1582              | 0.1369                         |
| 2015   | 0.1575              | 0.1362                         |
| 2024   | 0.1525              | 0.1290                         |

Even in the seasons where both work (2006, 2015, 2024), the two columns are **not**
interchangeable — `special_teams_play` flags roughly 1.3 points fewer than `special`
(about 12-13% fewer special-teams rows), so they encode slightly different definitions of
"special teams play."

**Recommendation**: prefer `special` first, then fall back to `special_teams_play` only if
`special` is absent (it never was, in this survey). Do not treat the two as synonyms if both
are present — pick one and document which.

## Extra columns of interest (2024)

2024 returns `372` total columns. Columns whose name contains `epa`, `success`,
`explosive`, `special`, `kick`, `punt`, or `field_goal`, beyond the candidate list (54 found;
`explosive` matched none):

- **EPA breakdowns**: `air_epa`, `comp_air_epa`, `comp_yac_epa`, `yac_epa`, `xyac_epa`,
  `xyac_success`, `total_away_epa`, `total_home_epa`, `total_away_pass_epa`,
  `total_home_pass_epa`, `total_away_rush_epa`, `total_home_rush_epa`,
  `total_away_comp_air_epa`, `total_home_comp_air_epa`, `total_away_comp_yac_epa`,
  `total_home_comp_yac_epa`, `total_away_raw_air_epa`, `total_home_raw_air_epa`,
  `total_away_raw_yac_epa`, `total_home_raw_yac_epa`
- **Success**: `series_success`
- **Kickoff**: `kick_distance`, `kicker_player_id`, `kicker_player_name`,
  `kickoff_attempt`, `kickoff_downed`, `kickoff_fair_catch`, `kickoff_in_endzone`,
  `kickoff_inside_twenty`, `kickoff_out_of_bounds`, `kickoff_returner_player_id`,
  `kickoff_returner_player_name`, `lateral_kickoff_returner_player_id`,
  `lateral_kickoff_returner_player_name`, `own_kickoff_recovery`,
  `own_kickoff_recovery_player_id`, `own_kickoff_recovery_player_name`,
  `own_kickoff_recovery_td`, `home_opening_kickoff`
- **Punt**: `punt_attempt`, `punt_blocked`, `punt_downed`, `punt_fair_catch`,
  `punt_in_endzone`, `punt_inside_twenty`, `punt_out_of_bounds`,
  `punt_returner_player_id`, `punt_returner_player_name`,
  `lateral_punt_returner_player_id`, `lateral_punt_returner_player_name`,
  `punter_player_id`, `punter_player_name`
- **Field goal**: `field_goal_attempt`, `field_goal_result`

No column name matched `explosive`; explosive-play flags would have to be self-computed
(e.g. thresholding `yards_gained` or `epa`), not read directly from PBP.

## Sanity anchors (2024)

- Mean `epa` where `qb_dropback == 1`: `0.05225`
- Mean `epa` where `rush == 1`: `-0.06941`
- Mean `success` overall: `0.45139`

## Findings that affect implementation

- All 47 candidate columns are present in every surveyed season back to 1999; none need a
  "column missing before season X" guard for existence — but several are effectively
  unusable in specific seasons because they're all-null or all-zero (see below), so an
  existence check alone is not sufficient.
- `special_teams_play` is unusable in 1999 (100% null) and 2003 (always `0.0`). Use
  `special` as the primary flag; it is clean (0% null) across all five seasons.
- `air_yards`, `cpoe`, `xpass`, `pass_oe` are **100% null in 1999 and 2003** — treat any
  cpoe/xpass/pass_oe-derived feature as unavailable before 2006 at the earliest (their null
  rate is still 24-66% from 2006 onward, but that floor is structural: these fields are only
  defined on pass attempts, not every play).
- `two_point_conv_result` (~99.7-99.9% null), `td_team` (~97% null), and
  `passer_player_id`/`passer_player_name` (~58-61% null) are high-null by construction —
  they only populate on the play subset they describe (two-point tries, scoring plays, pass
  plays respectively), not because of a data quality problem.
- `down` is null on 15-16% of rows in every season (kickoffs, PATs, timeouts, etc. have no
  down) — expected, but any per-down aggregation must filter nulls explicitly.
- Total column count is stable at 372 across all five seasons, so no wholesale schema
  version changes to plan around over 1999-2024.
- No load failures occurred for any of the five seasons in this environment; the survey
  script has an explicit try/except per season that was never exercised on the failure path,
  so failure-path behavior (partial report, continue) is untested by this run.

## Open questions

- Why is `special_teams_play` all-null in 1999 and all-zero in 2003 while `special` is clean
  in both years? This looks like an nflverse/nflfastR upstream data gap specific to that
  derived column, not something fixable on our side — worth a quick look at nflverse's own
  changelog if it ever matters which exact column we standardize on.
- `special` and `special_teams_play` disagree by ~1.3 points of true-rate even in seasons
  where both are populated (2006, 2015, 2024). Neither this survey nor a quick schema read
  can say which definition ("all special teams snaps" vs. some narrower subset) is intended
  without reading nflverse's PBP field documentation or diffing example rows where they
  disagree — flagged here rather than guessed at.
- This survey did not check seasons between 2015 and 2024 (e.g. did `special_teams_play`
  stay reliable every year in between, or was 2003 a one-off gap); if the eventual PBP loader
  needs to support arbitrary seasons rather than just the five sampled here, a fuller
  per-season audit of just the `special`/`special_teams_play` pair would be cheap to run.
