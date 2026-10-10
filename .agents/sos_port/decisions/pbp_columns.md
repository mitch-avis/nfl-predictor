# Decisions: the play-by-play cache columns and the stadium table

Two small chunks of roadmap step 6 (absorbing `nfl-sos-ratings`) that feed the batch of
feature-value changes: the play-by-play cache gains every column the port needs, without any
feature reading them yet, and the stadium table gains the venues the schedules use but it lacked.
Each judgment call below has the parts `.agents/sos_port/DECISIONS.md` asks for. Commits on branch
`worktree-agent-aee4f463ac1335152`: `973458d` (characterization pins), `954e2ca` (the column
change), `f5a3f1e` (stadium location derived on every load), `fcf9243` (the stadium entries).

Scripts and their captured output sit beside this file:

- `pbp_columns_inventory.py` and `pbp_columns_inventory_output.txt`: which play-by-play columns
  `nfl-sos-ratings` reads at `9081bce`, by AST, and which of them `constants.PBP_COLUMNS` lacks.
- `pbp_release_check.py` and `pbp_release_check_output.txt`: a fresh nflverse download of
  1999-2025 against the main checkout's cache on every shared column, and the cache size before
  and after.
- `pbp_column_facts.py` and `pbp_column_facts_output.txt`: the dtypes that vary between seasons,
  the team codes normalization changes in each team-code and yard-line column, and what the play
  text and the credit columns add to a season's cache file.
- `stadium_venues.py` and `stadium_venues_output.txt`: every venue id of the cached 1999-2026
  schedules against `constants.STADIUMS`, the games at the missing ones, and the venue fixture
  of `tests/test_stadiums.py`.
- `stadium_elevations.py` and `stadium_elevations_output.txt`: the sources of the new entries'
  coordinates and elevations.

## Play-by-play columns

### 1. Which columns: every play-by-play read, the analysis helpers included

- **What differs**: `nfl-sos-ratings` at `9081bce` reads 134 play-by-play columns; this repo's
  cache held 55, 50 of them among those 134. The 84 missing ones are listed in run 1 of
  `pbp_columns_inventory_output.txt`.
- **Options**: (a) the reads of the ten modules the step plan names plus the columns it asks for
  by name (132 columns); (b) those plus the reads of its analysis helpers in
  `.agents/findings_2026_10_08/` (134); (c) all 372 columns nflverse publishes.
- **Decision**: (b). `constants.PBP_COLUMNS` now holds 139 columns: the 55 it had and 84 added.
- **Reasoning**: every later addition makes every cached season download again, and each such
  refresh can bring in whatever nflverse has republished since, the confound this measurement
  exists to rule out; one refresh for the whole batch is cheaper than several. Option (b) adds
  only two numeric columns over (a) (`half_seconds_remaining` and `kickoff_attempt`, read by the
  source-coverage and lost-fumble analyses); `play_id` and `aborted_play`, also found only in the
  helpers, were requested by name anyway. Option (c) would roughly triple the size again for
  columns nothing reads.
- **Data**: method in the script's docstring: each Python file of the package and of the findings
  helpers is read at `9081bce` with `git show` (68 files, 10 primary), every string literal is
  collected with `ast` and intersected with the 372 column names of the 1999 and 2025 downloads;
  a matched name is a play-by-play read unless `NOT_PLAY_BY_PLAY` names it with the frame it is
  read from (8 names: `away_score` and `home_score` from the schedule, `receiving_yards` an output
  alias, `result` the per-drive alias of `fixed_drive_result`, `total` a local frame, `name` and
  `rusher` output aliases in the helpers, `id` a metric-registry shape). No f-string in the
  primary modules builds a column name (five, all messages or local names; listed in the output),
  and no generic selector (`pl.all()`, `pl.exclude`, selectors) touches a play-by-play frame
  there. Command (output in `pbp_columns_inventory_output.txt`):

  ```bash
  .venv/bin/python -I .agents/sos_port/decisions/pbp_columns_inventory.py \
      --sos-repo /home/mitch/workspace/nfl-sos-ratings
  ```

  ```text
  Matched names: 142
  Not play-by-play reads: 8
  Play-by-play reads: 134 (130 in the primary modules, 4 only in the others)
  Requested by name: 9 (9 also found by the scan)
  Columns the port needs: 134; in constants.PBP_COLUMNS: 134; missing: 0
  constants.PBP_COLUMNS: 139 columns
  ```

  Before the change the same script reports `in constants.PBP_COLUMNS: 50; missing: 84` and
  exits 1. Five cached columns are not read there: this repo's aggregations read four of them
  (`posteam_type`, `yardline_100`, `first_down_pass`, `first_down_rush`), and `xpass` is read by
  neither repository (left as it was).
- **Feature-value impact**: none. No consumer of the cached frame names an added column (the
  three are `pbp.aggregate_pbp_team_game_stats`, `pbp.aggregate_pbp_team_box_score_stats` and
  `qb_stats.aggregate_qb_game_stats`), and `tests/test_pbp_cache_characterization.py`, pinned on
  the code before the change, passes unedited after it: a synthetic world whose raw rows carry
  every added column, with values that would move an aggregation reading them (pre-2006 style
  scrambles, `passing_yards` and `rushing_yards` unequal to `yards_gained`, botched snaps,
  fumble and tackle credits). Each pin failed when its aggregation was broken on purpose, and
  the team-game pin failed when the snap definition was made to read `qb_scramble`.
- **How to reverse it**: remove entries from `constants.PBP_COLUMNS`. Cache files written with
  the longer list stay valid (a cache read keeps only the requested columns).
- **Review**: not yet reviewed.

### 2. Columns that `nfl-sos-ratings` reads only to normalize or repair

- **What differs**: its loader (`data_loader.load_pbp_data`) reads 33 added columns that no
  statistic there uses: three team-code columns it only normalizes (`side_of_field`,
  `timeout_team`, `return_team`), three yard-line texts it only normalizes (`yrdln`,
  `end_yard_line`, `drive_end_yard_line`), and 27 player-team and player-id columns that only its
  repair of Jacksonville's 2001-2002 home games reads, where nflverse credits every player to the
  visiting team (nflverse-pbp issue 92): the touchdown and penalty player ids, the second
  fumbler's team, and the fumble-recovery, forced-fumble and tackle credits.
  `PBP_PLAYER_TEAM_IDS` in its `player_team_repair.py` lists them with the five columns its
  statistics also read (`td_team`, `penalty_team`, `fumbled_1_team` and the two fumbler ids).
- **Options**: cache them, or cache only the columns statistics read.
- **Decision**: cache them, under entry 1's rule.
- **Reasoning**: the port has to carry that repair (`.agents/sos_port_source.md`, "nflverse quirks
  its loaders handle"), and its detection counts the teams named across all of those columns in a
  game; leaving some out changes which games it flags. They are cheap: the 28 fumble and tackle
  credit columns are 3% of a season's new cache file.
- **Data**: 2024 cache file with every requested column 4,217,398 bytes, without the 28 fumble and
  tackle credit columns 4,073,921 bytes, both written with the loader's own cache writer
  (`pbp_column_facts_output.txt`, from
  `.venv/bin/python -I .agents/sos_port/decisions/pbp_column_facts.py --work-dir "$(mktemp -d)"`).
- **Feature-value impact**: none (entry 1).
- **How to reverse it**: remove them from `constants.PBP_COLUMNS` (and, for team-code columns,
  from `constants.PBP_TEAM_COLUMNS`).
- **Review**: not yet reviewed.

### 3. The play text (`desc`)

- **What differs**: `player_team_repair.repair_pbp_player_teams` parses `desc` for the penalized
  team (`PENALTY on ([A-Z]{2,3})`) when a penalty in a repaired game names no player; its
  botched-snap analysis prints `desc` samples but classifies by `fumbled_1_player_id`.
- **Options**: cache `desc`, or leave it out and repair those penalties some other way.
- **Decision**: cache it.
- **Reasoning**: a rule there parses it, which is the condition the step plan set. It is the
  largest added column, but the whole cache stays near 100 MB.
- **Data**: 2024 cache file 4,217,398 bytes with `desc`, 3,275,462 without (22% of the file;
  `pbp_column_facts_output.txt`); all seasons together 104,039,588 bytes with every requested
  column (table in "Cache size").
- **Feature-value impact**: none.
- **How to reverse it**: remove `desc` from `constants.PBP_COLUMNS` and `_PBP_STRING_COLUMNS`.
- **Review**: not yet reviewed.

### 4. Every team-code column is normalized when the cache is written

- **What differs**: this repo normalized six play-by-play team columns (`posteam`, `defteam`,
  `home_team`, `away_team`, `td_team`, `penalty_team`) through `constants.ALIAS_TO_CANONICAL`;
  `nfl-sos-ratings` normalizes 23 (`_PBP_TEAM_COLUMNS` in its `data_loader.py`). nflverse writes
  the Rams as `LA` and Washington as `WAS` in every one of them; this repo's codes are `LAR` and
  `WSH`.
- **Options**: normalize all 23 at load time (new `constants.PBP_TEAM_COLUMNS`), or cache the 17
  added ones raw and normalize where they are read.
- **Decision**: normalize all 23 at load time.
- **Reasoning**: the ported rules compare these columns with `posteam` and `defteam` (the
  lost-fumble team, the penalized team, the touchdown team), which are already canonical, and the
  repo rule is to normalize every team identifier. A test checks that every cached column whose
  name ends in `team`, and `side_of_field`, is in the list.
- **Data**: in today's 1999-2025 downloads (regular season), the only codes normalization
  changes are `LA` and `WAS`, in every one of the 23 columns that has values, for example 73,005
  `posteam` values, 1,258 `fumbled_1_team`, 9,100 `return_team` and 35,308 `solo_tackle_1_team`;
  `tackle_with_assist_2_team` has no such value. `side_of_field` also holds `''`, `50` and `MID`,
  and `posteam` holds `''`, which pass through unchanged (`pbp_column_facts_output.txt`). Tests:
  `tests/test_pbp_cache_columns.py::test_load_pbp_normalizes_the_added_team_code_columns` and
  `::test_pbp_team_columns_name_every_cached_team_code_column`.
- **Feature-value impact**: none; the six earlier columns are normalized exactly as before.
- **How to reverse it**: shorten `constants.PBP_TEAM_COLUMNS`, then refresh the cache
  (`--refresh-nflreadpy`), since cached values stay normalized until rewritten.
- **Review**: not yet reviewed.

### 5. Yard-line text is cached as nflverse writes it

- **What differs**: `nfl-sos-ratings` rewrites the team code that opens a yard line (`LA 25`) in
  `yrdln`, `drive_start_yard_line`, `drive_end_yard_line` and `end_yard_line`; this repo caches
  `drive_start_yard_line` raw, and no consumer here reads it.
- **Options**: normalize the code in all four; normalize it only in the three added ones; keep
  all four raw and let the port normalize where it compares a yard line with `posteam`.
- **Decision**: keep all four raw.
- **Reasoning**: normalizing `drive_start_yard_line` would change the cached values of a column
  the cache already held, which this chunk promised not to do, and treating the added three
  differently from it would make the four disagree. The port must normalize at the point of use:
  its drive-start field position compares this code with `posteam`.
- **Data**: in today's 1999-2025 downloads (regular season), the code is one that normalization
  changes (`LA` or `WAS`; canonical `LAR` and `WSH`) on 75,362 plays in `yrdln`, 76,210 in
  `drive_start_yard_line`, 75,028 in `drive_end_yard_line` and 4,578 in `end_yard_line`
  (`pbp_column_facts_output.txt`). Test:
  `tests/test_pbp_cache_columns.py::test_load_pbp_keeps_yard_line_text_as_published`.
- **Feature-value impact**: none now. For the ETL integration: a ported field-position stat that
  skips the normalization misreads the Rams' and Washington's drive starts.
- **How to reverse it**: add the yard-line normalization to `loaders._prepare_pbp` (a change to
  cached values of `drive_start_yard_line`, so a cache refresh).
- **Review**: not yet reviewed.

### 6. One dtype per column across seasons

- **What differs**: nflverse publishes `goal_to_go` as `Float64` in 1999-2002, 2020, 2024 and 2025
  and as `Int32` in the other seasons (`xyac_median_yardage` varies too, and is not cached). The
  loader stacks seasons with a strict diagonal concat, so adding `goal_to_go` as published made
  every multi-season load fail with `SchemaError: type Float64 is incompatible with expected type
  Int32`.
- **Options**: cast each cached column to the loader's dtype map before caching; or stack seasons
  with a relaxed concat (the column type would then depend on which seasons were loaded).
- **Decision**: cast every cached column except `season` and `week` to `_PBP_COLUMN_DTYPES`
  (`loaders._cast_pbp_value_columns`); `goal_to_go` is cached as `Float64`. A column that already
  has its mapped dtype is left alone, and `season`/`week` keep nflverse's `Int32`.
- **Reasoning**: a stable type per column is the cache's contract (the empty frame already uses
  the map). Every one of the 55 earlier columns already had its mapped dtype, so the cast changes
  none of them.
- **Data**: of the 372 raw columns in the 27 downloads, only `goal_to_go` and
  `xyac_median_yardage` change dtype between seasons (`pbp_column_facts_output.txt`). Test:
  `tests/test_pbp_cache_columns.py::test_load_pbp_stacks_seasons_that_type_goal_to_go_differently`;
  with the cast removed it fails with the `SchemaError` above, and so does the characterization
  test, whose fixture types `goal_to_go` per season the same way.
- **Feature-value impact**: none.
- **How to reverse it**: drop the cast call in `loaders._prepare_pbp` (and drop `goal_to_go`, or
  multi-season loads fail again).
- **Review**: not yet reviewed.

### 7. Where the added columns sit in the list

- **What differs**: the step plan asks to keep the list's existing order and sections.
- **Options**: append the 84 columns at the end; or place each in its section, with three new
  sections (game state, series, the other players' credits) and the play text last.
- **Decision**: place each in its section; the 55 earlier columns keep their relative order
  (checked by `test_pbp_columns_keep_the_earlier_list_and_add_every_ported_read`).
- **Reasoning**: the list is read as documentation of the cache. Consumers select columns by name,
  so the new column order in cache files and loaded frames changes no output (the pins pass).
- **Data**: none needed beyond the test.
- **Feature-value impact**: none.
- **How to reverse it**: reorder the list; nothing depends on the order.
- **Review**: not yet reviewed.

### 8. A fresh download equals the cache, so the candidate build may refresh it

- **What differs**: the candidate build will download every season again (each cache file lacks
  the added columns), while the reference build read the old cache. A republished season would
  confound the comparison.
- **Options**: accept fresh downloads after checking them against the cache; or keep the old
  files and add the new columns from a separate download.
- **Decision**: accept fresh downloads. Rerun `pbp_release_check.py` right before the candidate
  build's ETL (it takes about a minute) and keep its output with the build.
- **Reasoning**: on 2026-10-10 a fresh download of each season 1999-2025 equals the main
  checkout's cache on all 55 shared columns, row for row, and every season's release file was
  last updated before its cache file was written. nflverse can still republish before the build,
  hence the recheck.
- **Data**: command and full output in `pbp_release_check_output.txt`; per season (rows cached,
  rows fresh, shared columns, columns with differing values, values differing, largest EPA
  difference):

  ```text
  1999 | 44164 | 44164 | 55 | 0 | 0 | 0      2013 | 46255 | 46255 | 55 | 0 | 0 | 0
  2000 | 43576 | 43576 | 55 | 0 | 0 | 0      2014 | 45673 | 45673 | 55 | 0 | 0 | 0
  2001 | 43074 | 43074 | 55 | 0 | 0 | 0      2015 | 46141 | 46141 | 55 | 0 | 0 | 0
  2002 | 45326 | 45326 | 55 | 0 | 0 | 0      2016 | 45707 | 45707 | 55 | 0 | 0 | 0
  2003 | 44850 | 44850 | 55 | 0 | 0 | 0      2017 | 45268 | 45268 | 55 | 0 | 0 | 0
  2004 | 44793 | 44793 | 55 | 0 | 0 | 0      2018 | 45120 | 45120 | 55 | 0 | 0 | 0
  2005 | 44973 | 44973 | 55 | 0 | 0 | 0      2019 | 45339 | 45339 | 55 | 0 | 0 | 0
  2006 | 44321 | 44321 | 55 | 0 | 0 | 0      2020 | 45406 | 45406 | 55 | 0 | 0 | 0
  2007 | 44411 | 44411 | 55 | 0 | 0 | 0      2021 | 47651 | 47651 | 55 | 0 | 0 | 0
  2008 | 43982 | 43982 | 55 | 0 | 0 | 0      2022 | 47157 | 47157 | 55 | 0 | 0 | 0
  2009 | 44595 | 44595 | 55 | 0 | 0 | 0      2023 | 47399 | 47399 | 55 | 0 | 0 | 0
  2010 | 44991 | 44991 | 55 | 0 | 0 | 0      2024 | 47274 | 47274 | 55 | 0 | 0 | 0
  2011 | 45455 | 45455 | 55 | 0 | 0 | 0      2025 | 46452 | 46452 | 55 | 0 | 0 | 0
  2012 | 45829 | 45829 | 55 | 0 | 0 | 0
  Seasons identical on every shared column: 27 of 27
  ```

  The latest release update is 2020's, at 2026-08-26T07:33:10Z; the cache files were written on
  2026-09-22 (2002-2025) and 2026-10-09 (1999-2001).
- **Feature-value impact**: none from republication as of 2026-10-10.
- **How to reverse it**: not applicable; if the recheck finds a difference, list the affected
  seasons and columns and decide then whether to pin the old values.
- **Review**: not yet reviewed.

### Cache size

Each season's prepared frame from the same fresh download, written with the loader's own cache
writer once with the cached file's 55 columns and once with all 139 (from
`pbp_release_check_output.txt`; the first column is the main checkout's file today, which matches
the 55-column write to within the 921-byte column record that files written before 2026-10-09
lack):

```text
season | cached file today | 55 columns | 139 columns | after / before
1999 | 1083915 | 1083915 | 3005336 | 2.77
2005 | 1111227 | 1112148 | 3228326 | 2.90
2006 | 1502237 | 1503158 | 3929599 | 2.61
2024 | 1605991 | 1606912 | 4217398 | 2.62
2025 | 1583045 | 1583966 | 4182340 | 2.64
total 1999-2025 | 38698844 | 38720948 | 104039588 | 2.69
```

The full cache grows from about 39 MB to about 104 MB.

## The stadium table

### 9. Which venues: every venue id of the cached schedules, 2026 included

- **What differs**: `constants.STADIUMS` had no entry for 14 venue ids of the cached
  1999-2026 schedules. Nine are 1999-2002 home stadiums used in 158 games (Foxboro Stadium
  `BOS99`, Memorial Stadium in Champaign `CHI99`, Cinergy Field `CIN99`, Mile High Stadium
  `DEN99`, the Pontiac Silverdome `DET99`, Veterans Stadium `PHI99`, Three Rivers Stadium
  `PIT99`, the Kingdome `SEA98`, Husky Stadium `SEA99`); five are 2026 international venues with
  one game each (the Bernabéu `MAD01`, the Melbourne Cricket Ground `MEL00`, the Allianz Arena as
  "FC Bayern Munich Stadium" `MUN01`, the Stade de France `PAR00`, the Maracanã `RIO00`). Their
  games got `stadium_elevation` 0 (the default for an unknown venue) and no city or state. The
  step plan counted 14 ids "used by 1999-2025 schedules"; 1999-2025 has nine of them.
- **Options**: (a) only the nine ids of 1999-2025; (b) all 14.
- **Decision**: (b). The test checks every venue id of 1999-2026 (67 ids).
- **Reasoning**: the 2026 games are the ones being predicted now; three of the five are still to
  be played (Paris in week 7, Madrid in week 9, Munich in week 10). The schedule files of
  1995-1998 in the cache hold no games, so no other id is missing.
- **Data**: run 1 of `stadium_venues_output.txt` (exit 1: `67 venue ids, 14 without a
  constants.STADIUMS entry, 163 games at them`); run 2 after the change exits 0 with none
  missing. Command:

  ```bash
  .venv/bin/python -I .agents/sos_port/decisions/stadium_venues.py \
      --cache-dir /home/mitch/workspace/nfl-predictor/data/cache/nflreadpy \
      --first-season 1999 --last-season 2026 --fixture tests/fixtures/schedule_venue_ids.json
  ```

- **Feature-value impact**: 163 games (playoff games included) change `stadium_elevation` from 0
  to the new entry's value, and gain `stadium_city` and `stadium_state` (reference columns, not
  features); `stadium_name` comes from the schedule and does not change. Per season: 1999 57
  games, 2000 49, 2001 34, 2002 18, 2026 5 (every game id in `stadium_venues_output.txt`). By
  venue, elevation 0 becomes:

  ```text
  BOS99  313 ft  1999-2001  25 games      SEA98   18 ft  1999        9 games
  CHI99  732 ft  2002        8 games      SEA99   32 ft  2000-2001  16 games
  CIN99  505 ft  1999        8 games      MAD01 2315 ft  2026        1 game  (2026_09_CIN_ATL)
  DEN99 5210 ft  1999-2000  16 games      MEL00   54 ft  2026        1 game  (2026_01_SF_LA)
  DET99  892 ft  1999-2001  24 games      MUN01 1617 ft  2026        1 game  (2026_10_NE_DET)
  PHI99   21 ft  1999-2002  36 games      PAR00  138 ft  2026        1 game  (2026_07_PIT_NO)
  PIT99  731 ft  1999-2000  16 games      RIO00   38 ft  2026        1 game  (2026_03_BAL_DAL)
  ```

  Batched with the other feature-value changes: it reaches a weekly run only when the batch
  merges.
- **How to reverse it**: remove the entries from `constants.STADIUMS` (and the ids from
  `tests/fixtures/schedule_venue_ids.json`).
- **Review**: not yet reviewed.

### 10. The source of each new entry's values

- **What differs**: the table holds a name, a city, a state or country code and `elevation_ft`
  (it holds no roof or surface: the schedule supplies those for every game). The existing
  entries record no source; of the nine US entries the script also checked, eight are within
  29 ft of USGS ground elevation at their coordinates, and `DEN00` holds the nominal mile (5,280
  against 5,195).
- **Options**: elevation from (a) USGS 3DEP at the venue's coordinates in the United States and,
  elsewhere, the median of Copernicus GLO-90 on a 300 m ring around the site; (b) Copernicus
  GLO-90 at the venue's coordinates everywhere; (c) the city's elevation from an encyclopedia;
  and for `DEN99`, (d) the nominal 5,280 ft `DEN00` uses.
- **Decision**: (a), with the coordinates from each venue's English Wikipedia article (revision
  ids in `stadium_elevations_output.txt`), rounded to the foot; `DEN99` gets its measured
  5,210 ft. Names are the ones the venue had during its NFL use (`Cinergy Field`, not Riverfront
  Stadium; `Memorial Stadium (Champaign)`, as the schedule names it); city and state come from the
  same articles, with an ISO country code for non-US venues (`ES`, `AU`, `DE`, `FR`, `BR`), as
  most existing international entries have (Toronto's `ON` is the exception).
- **Reasoning**: 3DEP is a bare-earth, lidar-based model at about 10 m. Copernicus GLO-90 is a
  surface model, so a stadium's stands and roof raise its value at the center (the Superdome
  reads 75 ft there, 2 ft in USGS and 7 ft as the ring median); on 18 US sites the ring median is
  within -16 to +34 ft of USGS. The step plan asks for sourced values, which rules out (d); for
  the model, Denver at 5,210 or 5,280 ft is the same high-altitude venue.
- **Data**: command and output in `stadium_elevations_output.txt`
  (`.venv/bin/python -I .agents/sos_port/decisions/stadium_elevations.py --existing`), one line
  per venue, for example:

  ```text
  DEN99 | Mile High Stadium (revision 1372216043) | 39.74611, -105.02167 | USGS 3DEP 5210 ft | Copernicus GLO-90 5220 ft, ring median 5244 ft | table None
  MAD01 | Bernabéu (stadium) (revision 1376542686) | 40.45306, -3.68835 | USGS 3DEP - ft | Copernicus GLO-90 2343 ft, ring median 2315 ft | table None
  NOR00 | Caesars Superdome (revision 1378776769) | 29.95083, -90.08111 | USGS 3DEP 2 ft | Copernicus GLO-90 75 ft, ring median 7 ft | table 3.0
  ```

- **Feature-value impact**: as entry 9.
- **How to reverse it**: change the values in `constants.STADIUMS`.
- **Review**: not yet reviewed.

### 11. The Allianz Arena under two ids

- **What differs**: nflverse files the Allianz Arena as `GER00` ("Allianz Arena", 2022 and 2024)
  and as `MUN01` ("FC Bayern Munich Stadium", its name at UEFA and NFL events, 2026).
- **Options**: give `MUN01` the values `GER00` holds (1,617 ft); or its own measurement (the
  ring median, 1,621 ft).
- **Decision**: the `GER00` values: name "Allianz Arena", Munich, `DE`, 1,617 ft.
- **Reasoning**: one stadium should not have two elevations; the existing value agrees with the
  measurement within 4 ft.
- **Data**: `stadium_elevations_output.txt`: the same Wikipedia article and coordinates for both
  ids, Copernicus GLO-90 1,611 ft at the center and 1,621 ft as the ring median.
- **Feature-value impact**: one game, `2026_10_NE_DET`.
- **How to reverse it**: change `MUN01` in `constants.STADIUMS`.
- **Review**: not yet reviewed.

### 12. Stadium city and state derived on every load

- **What differs**: the schedule loader derived `stadium_city`, `stadium_state` and a missing
  `stadium_name` from the table only when it built a frame from a download; a cached schedule
  kept the values it was written with, while `stadium_elevation` was derived again on every load.
  The main checkout's cached 1999-2026 schedules were written before the 14 entries existed, so a
  build from that cache would have given those games their elevation but no city or state, and a
  build from fresh downloads both.
- **Options**: (a) derive the table's columns on every load; (b) leave the loader and refresh the
  schedule cache (`--refresh-nflreadpy`) whenever the table changes.
- **Decision**: (a) (`loaders._apply_schedule_enrichments`).
- **Reasoning**: a build should not depend on when its cache was written, and (b) relies on a
  manual step that nothing enforces. For venues the table already held, the values derived are
  the ones cached, so nothing else changes (the loader and characterization suites pass
  unchanged).
- **Data**: `tests/test_stadiums.py::test_load_schedule_locates_a_cached_venue_the_table_gained_later`
  failed before the change with `(None, None, 1234.0) == ('Testville', 'TS', 1234.0)`: the cached
  game had the elevation of the venue the table gained, but not its city and state.
- **Feature-value impact**: no model feature (city and state are reference columns); the 163
  games of entry 9 get their location from a cached schedule too.
- **How to reverse it**: restore the `"stadium_city" not in df.columns` condition in
  `_apply_schedule_enrichments`.
- **Review**: not yet reviewed.

### 13. Observed, not changed: nflverse's roofs for the 2026 international venues

- **What differs**: the 2026 schedule lists the Melbourne Cricket Ground, the Allianz Arena
  (`MUN01`) and the Stade de France as `dome` and leaves the Bernabéu's roof empty, while it lists
  the Allianz Arena as `outdoors` under `GER00` in 2022 and 2024. None of the first three has a
  roof over the field; the Bernabéu has had a retractable one since its renovation. The roof
  feeds the `stadium_type` feature (`dome`, `open`, `retractable`, or `unknown` when empty).
- **Options**: (a) leave the schedule's values: the current season's schedule is downloaded
  again on every ETL run, so an nflverse correction arrives by itself; (b) add a roof field to
  the table that overrides the schedule.
- **Decision**: (a), and nothing changed here: the table holds no roof, and adding one is beyond
  this chunk. Raised as a question for the user, with the recommendation to recheck the
  schedule before the week-7 Paris game.
- **Reasoning**: three of the four games are still to be played, and an override would be a new
  table field and loader rule for three games whose source may still be corrected.
- **Data**: the roof column of the "Every venue id" table in run 1 of
  `stadium_venues_output.txt` (the main checkout's cached schedules): `GER00` (2022-2024)
  `outdoors`; `MUN01`, `MEL00` and `PAR00` (2026) `dome`; `MAD01` (2026) empty; `RIO00` (2026)
  `open`.
- **Feature-value impact**: none from this chunk; if left, those games keep `stadium_type`
  `dome` (three) and `unknown` (Madrid).
- **How to reverse it**: not applicable.
- **Review**: not yet reviewed.
