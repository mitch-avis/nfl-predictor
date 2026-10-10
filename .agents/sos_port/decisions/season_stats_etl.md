# Decisions: the season statistics in the ETL

Roadmap step 6, the ETL step that writes the season statistics tables ported from
`nfl-sos-ratings` (`src/nfl_predictor/season_stats/`), so the web app's team and quarterback pages
have their data. Each judgment call below has the parts `.agents/sos_port/DECISIONS.md` asks for.
Commits on branch `worktree-agent-ad5396b63790d07a8` (off `feat/step6-sos-port` at `b6f85ed`,
`0.54.0`): `a962f1b` (the season cache's code fingerprint), `e3dee51` (the step and its ratings
hook), `735109e` (the ETL writes the tables and keys the cache on them), `10698f5` (the parity and
timing scripts), `d01df69` (the missing-source check), `c511e50` (the READMEs), `c378123` (the web
empty states). After the branch's independent review: `7f916fd` (the determinism tests' labeled
phases), `90f21f2` (the row-order guard over the season statistics package, entry 9), `7b37868` and
`631187c` (the download and fallback tests, entry 2), `3ff547f` (a reused season's game rows,
entry 7), `fc07d16` and `66b732c` (one schema for every table, entry 6), `3dfb4c6` (the season
cache round trip in the parity script), and the reruns of the parity and missing-source checks.

Scripts and their outputs, run from the repository root with the worktree's environment; each
writes only to a temporary directory under `~/scratch`, reading the snapshot of the nflreadpy
cache `nfl-sos-ratings` read (`~/scratch/nflreadpy_cache_snapshot_20261008/`) through this
repository's loaders, with downloads blocked (`snapshot_cache.py`, which now also writes the
official weekly team stats cache from the snapshot's team stats files):

- `.agents/sos_port/season_stats_etl_parity.py` writes `SEASON_STATS_ETL_PARITY.md` and
  `season_stats_etl_parity.json` beside it: every season of 1999-2026 built through the ETL's
  step, written by the ETL's writer, read back and compared with the published files; then every
  season's frames stored in a season cache, loaded back, reused as an incremental run reuses them
  and written again, each file compared byte for byte with the fresh build's.
- `.agents/sos_port/season_stats_etl_schema.py` writes
  `src/nfl_predictor/utils/polars/season_stats_schema.py`, the columns and types of every table's
  full build (entry 6); with `--check` it only says whether that module is up to date.
- `.agents/sos_port/season_stats_etl_timing.py` and its captured output in this directory,
  `season_stats_etl_timing_output.txt`: what the step adds to a full and to an incremental run.
- `.agents/sos_port/decisions/season_stats_etl_degrade.py` and its captured output,
  `season_stats_etl_degrade_output.txt`: what a season's tables lose without each source.

The timings were measured while the phase-4 walk-forwards ran on the same machine (from 10:08,
GPU and CPU), so they are inflated; each output records the load averages beside its numbers.

## 1. The ratings build runs the step, right after each season's fit

- **What differs**: the plan names an ETL step for the season statistics but not its place.
  `nfl-sos-ratings`' `main.run_season` builds a season's statistics and its ratings in one pass,
  and the `combined` and `qb_combined` tables take the season's ratings.
- **Options**: (a) a new module, `utils/polars/season_stats_build.py`, whose build
  `RatingsBuild.build` runs right after each season's fit, recording the tables among the
  season's ratings frames; (b) a separate build that `data_collection._build_season` calls after
  the ratings, with a field of its own in the season cache's entries; (c) the step inside
  `ratings_build.py`.
- **Decision**: (a).
- **Reasoning**: the tables need the season's fit, which only the ratings build holds while it
  fits. Recorded among its frames, they share the ratings' writer (one temporary file per table,
  then a move), their season cache entry, their reuse and their listing of published tables with
  no new plumbing, so `data_collection.py` changes in two places only, as the brief asked. (b)
  needs a second entry field, a new entry format and more of `_build_season`; (c) would add
  loading code to a 715-line module about the ratings.
- **Data**: `git diff --stat b6f85ed -- src/nfl_predictor/data_collection.py`: 31 lines added,
  5 removed (the build constructed in `_load_sources`, the key's frames in `_season_frames`).
  `tests/test_season_stats_build.py`:
  `test_build_adds_each_seasons_statistics_tables_to_its_frames`,
  `test_published_tables_name_the_season_statistics_tables_by_season_and_kind`,
  `test_reused_season_frames_publish_the_same_season_statistics_tables`;
  `tests/test_data_collection_helpers.py`:
  `test_load_sources_builds_the_season_statistics_beside_the_ratings_from_their_sources`.
- **Feature-value impact**: none; the tables feed no feature.
- **How to reverse it**: drop the `season_stats` argument where `data_collection._load_sources`
  builds the `RatingsBuild` (the ETL then writes the ratings tables only), or revert `735109e` and
  `e3dee51` for the whole step.
- **Review**: not yet reviewed.

## 2. No source is read twice; the official weekly team stats are the one new load

- **What differs**: `season_stats.sources.load_season_inputs` loads six sources for each season;
  an ETL run already holds five of them (play-by-play, schedules, weekly player stats, snap counts,
  and the players file and weekly rosters behind the quarterback identity).
- **Options**: (a) reuse the frames the ratings loaded and load the official weekly team stats
  once, for the run's seasons, through their own cached source (`OFFICIAL_TEAM_STATS`, cache files
  `official_team_stats_<season>_reg.parquet`); (b) call `load_season_inputs` for every season,
  asking every source again; (c) take the official stats from the ETL's team-stat loader
  (`loaders.load_team_stats`), which reads the same nflverse file.
- **Decision**: (a).
- **Reasoning**: (b) asks for every source twice. (c) would change the team-stat loader, which
  sums away the separate first downs and lost fumbles the override reads, and whose columns feed
  the model's features. With (a), the current season's team stats file is asked for by both
  loaders, with the same address and parameters, and nflreadpy's in-memory cache (its default)
  answers the second request: one download per run. A finished season's official file is
  downloaded once, on the first run after this change, then read from the cache.
- **Data**: `tests/test_season_stats_build.py`:
  `test_sources_load_the_official_team_stats_of_the_runs_seasons_once` (one load, the run's
  seasons), `test_sources_reuse_every_other_frame_the_ratings_loaded` (every other loader fails
  the test if called),
  `test_the_current_seasons_team_stats_file_is_downloaded_once_for_both_loaders` (one HTTP request
  for both loaders) and its control,
  `test_the_team_stats_file_is_downloaded_again_once_nflreadpy_forgets_it` (two requests when
  nflreadpy's cache is cleared between the loaders, so the single request is the cache's doing).
  The official stats keep the cache-then-degrade contract: when the current season's download
  fails with a `ConnectionError` or a `ValueError` from `nfl.load_team_stats`,
  `test_a_failed_current_season_download_falls_back_to_the_cached_official_team_stats` reads the
  cached file and
  `test_a_failed_current_season_download_without_a_cache_leaves_no_official_team_stats` gets no
  rows with the source's columns. Caveat: an environment that turns nflreadpy's cache off
  (`NFLREADPY_CACHE=off`) downloads the current season's file twice.
- **Feature-value impact**: none.
- **How to reverse it**: in `season_stats_build.load_season_stats_sources`, load the other
  sources per season instead of taking them from the ratings' frames.
- **Review**: not yet reviewed.

## 3. The ETL's play-by-play: repaired first, then prepared, and only the columns read

- **What differs**: `load_season_inputs`, like the source, prepares a season's play-by-play (the
  fields nflverse lacks that season set to null, empty team codes to null, canonical codes in the
  yard lines) and then repairs the games nflverse credits to one team (Jacksonville's 2001-2002
  home games). The ETL repaired its play-by-play when it loaded it, so the step prepares after
  the repair. The step also hands the builders only `SEASON_STATS_PBP_COLUMNS`, where
  `load_season_inputs` hands them every cached column.
- **Options**: (a) prepare the ETL's repaired play-by-play, cut to the columns the builders read;
  (b) load the season again through `load_season_inputs` (a second request for the play-by-play and
  the rosters); (c) prepare the ETL's frame and repair it a second time.
- **Decision**: (a).
- **Reasoning**: the order can only matter if an empty team code hid a game from the repair's
  test (one team named in every player-team column), and the parity run below shows no table
  moves in any season, the two seasons with repaired games included. The cut columns are the ones
  the builders read (`load_season_inputs` refuses play-by-play without them, and the season
  statistics audit showed the builders read nothing else); handing over only those makes the
  season cache's key hold exactly the builders' play-by-play (entry 7).
- **Data**: `SEASON_STATS_ETL_PARITY.md`: 280 tables, 1999-2026, every cell bit-identical,
  2001 and 2002 included. `tests/test_season_stats_build.py`:
  `test_tables_equal_those_built_from_the_season_statistics_own_loader` (each synthetic season's
  tables equal those built from `load_season_inputs` over the same frames).
- **Feature-value impact**: none.
- **How to reverse it**: in `SeasonStatsBuild.season_inputs`, keep every column of the plays.
- **Review**: not yet reviewed.

## 4. The rating columns come from the season's fit, as `main.run_season` fed them

- **What differs**: `main.run_season` joins `combined` to the season's `ratings` table (less
  `games_played`) and `qb_combined` to `build_qb_ratings`, the quarterback fit's rating and faced
  pass defense of every passer it rates; the port's `build_season_tables` takes both as arguments.
- **Options**: (a) the ratings build's `ratings` table, and `build_qb_season`'s `qb_season`, every
  passer's row of the same quarterback fit; (b) the published `qb_ratings` table; (c) a second fit
  in the step.
- **Decision**: (a).
- **Reasoning**: (a) is the fit whose tables the ETL publishes, for every passer, as there. (b)
  lists qualified passers only, so every other passer's rating would be null, unlike the source;
  (c) fits twice.
- **Data**: `SEASON_STATS_ETL_PARITY.md`: the rating columns this run fitted, 9,592 cells (5,358
  in `combined`, 4,234 in `qb_combined`), all bit-identical to the published files.
  `tests/test_season_stats_build.py`:
  `test_combined_takes_the_rating_columns_of_the_seasons_team_ratings` and
  `test_qb_combined_takes_the_rating_columns_of_the_seasons_quarterback_fit` (the backup, who
  never qualifies, keeps his rating).
- **Feature-value impact**: none.
- **How to reverse it**: pass other frames as `team_ratings` and `qb_ratings` in
  `RatingsBuild.build`.
- **Review**: not yet reviewed.

## 5. A season without a qualified passer leaves the passers' ratings null

- **What differs**: the ratings build fits no quarterbacks in a season with no passer qualified for
  the rankings (`ratings_build._has_qualified_passer`), at worst a season in its first days;
  `main.run_season` fitted every season.
- **Options**: (a) build `qb_combined` with its rating columns null and log it; (b) fit the
  quarterbacks for `qb_combined` alone; (c) leave the season without tables.
- **Decision**: (a).
- **Reasoning**: no published season lacks a qualified passer, so the case is a season's first
  days, when a rating from a game or two says little; (c) would hide the season's stats.
- **Data**: `SEASON_STATS_ETL_PARITY.md`: every season of 1999-2026 has its quarterback ratings.
  `test_a_season_without_a_qualified_passer_leaves_the_passer_ratings_null`.
- **Feature-value impact**: none.
- **How to reverse it**: give `RatingsBuild._tables` a quarterback fit for such a season.
- **Review**: not yet reviewed.

## 6. A missing source: build without it, keep every column, and say what is missing

- **What differs**: `load_season_inputs` lets a missing current-season source become no rows
  without saying what the tables lose; `build_season_tables` refuses a season without a game. The
  package's builders also drop the columns a missing source fills (a 2025 build without the
  weekly player stats lost up to 34 columns of `qb_combined`) and the opponent-profile columns of
  `combined` and `qb_combined` in a season where no opponent has played anyone else yet, and give
  `qb_game_logs.week` the play-by-play's `Int32` before the snap counts begin (2013) and `Int64`
  once they join. `AGENTS.md`'s "Missing Data Rules" ask the ETL for an invariant schema in every
  run, a missing source becoming nulls or defined defaults.
- **Options**: (a) build the season without the source and warn, naming what is missing; log and
  skip a season before its first game; conform every table built to the columns and types of a
  full build, with nulls of each column's type, in the ETL's step, leaving the reviewed package as
  ported; (b) as (a), each table keeping the columns its build gives it (the step before the
  review); (c) as (a), also writing the opponent-profile tables empty in a season too early for a
  profile, where the builders make none; (d) skip a season's tables while any source is missing;
  (e) fail the ETL.
- **Decision**: (a); the branch's independent review raised the missing columns against the rule,
  and the delegating session chose to conform in the step. The reference is `SEASON_STATS_SCHEMAS`
  in `src/nfl_predictor/utils/polars/season_stats_schema.py`, generated by
  `.agents/sos_port/season_stats_etl_schema.py` from the full builds of every season of 1999-2026,
  which have the same columns in the same order and each column one type, apart from
  `qb_game_logs.week`, written as the published files' `Int32` in every season.
  `season_stats_build.conform_table` adds a missing column as nulls of its type, in its place, casts
  a column to the reference's type, and refuses a column a full build lacks rather than dropping it.
- **Reasoning**: the task asks that a missing current-season source not fail the ETL and that what
  is skipped be logged, so (d) and (e) are out: (d) would leave the pages without the current season
  while nflverse catches up each week. Under (b) a file's columns would change as nflverse
  publishes, against the rule. Conforming in the step keeps the package's parity with
  `nfl-sos-ratings` as it was proven. The reference comes from real builds because the synthetic
  test league gives some columns other types than nflverse's files; it cannot go stale silently: a
  test fails when the builders' columns drift from it, and the generator's `--check` compares its
  types with the real builds again. Against (c): `nfl-sos-ratings` wrote no profile file then
  either (`nfl_sos_ratings/main.py` at `9081bce` writes one only when the profile exists), an
  absent table already reads as not computed (the pages hide a season's missing section), the case
  lasts only until the season's second week of games, and an empty table would show an empty
  section instead. Rows a missing source supplies are still missing (without the snap counts, the
  passers' games without a dropback), and the warning says so.
- **Data**: `season_stats_etl_degrade_output.txt` (2025, rerun at `3dfb4c6`): every variant with
  plays has every table with a full build's columns. Without the official team stats the
  play-by-play totals stand in; without the weekly player stats the columns they fill are null
  (the 14 defense box-score columns of the team tables, 11 of `qb_game_logs`, 34 of
  `qb_combined`); without the snap counts `qb_game_logs` has 635 of the full build's 687 rows,
  `qb_per_game_stats` and `qb_combined` 78 of 82, and `qb_opponent_profiles` 65 against 64;
  without the season's play-by-play there is no table. `SEASON_STATS_ETL_PARITY.md` (at
  `3dfb4c6`): every one of the 280 published tables still matches cell for cell, and
  `qb_game_logs.week` no longer differs in type from the published files in 2013-2026. Tests in
  `tests/test_season_stats_build.py`:
  `test_a_build_missing_a_source_has_every_column_of_a_full_build` (each of the official team
  stats, weekly player stats and snap counts),
  `test_a_build_without_player_stats_leaves_their_columns_null`,
  `test_qb_game_logs_week_has_one_type_with_or_without_snap_counts`,
  `test_a_season_whose_opponents_played_no_one_else_keeps_every_combined_column`,
  `test_the_reference_schema_names_every_season_statistics_kind`,
  `test_a_full_build_has_exactly_the_reference_columns_before_it_is_conformed`,
  `test_conforming_a_full_build_keeps_every_value`,
  `test_conforming_refuses_a_column_a_full_build_does_not_have`,
  `test_a_season_without_a_game_has_no_tables`,
  `test_a_played_season_without_a_source_logs_what_its_tables_lack`,
  `test_a_season_whose_snap_counts_nflverse_publishes_logs_their_absence`,
  `test_a_season_before_nflverse_has_snap_counts_logs_nothing` (nflverse's 2012 snap-count file
  is empty, so the warning starts in 2013).
- **Feature-value impact**: none.
- **How to reverse it**: drop the `conform_table` call in `SeasonStatsBuild.tables` (the tables
  then keep the builders' columns and types); `season_stats_build.log_missing_sources` and the
  early return in `SeasonStatsBuild.tables` for the rest.
- **Review**: not yet reviewed.

## 7. The season cache keeps a finished season's tables, keyed on what they read

- **What differs**: the task asks to keep finished seasons' tables in the season cache if that
  fits its design, and otherwise to rebuild them every run.
- **Options**: (a) keep them with the season's ratings, keyed on the season's play-by-play, game
  rows and played schedule (`build_season_tables`' arguments besides the ratings, which the
  ratings' part of the key covers), building the game rows again on every incremental run for the
  key; (b) key on the raw sources (play-by-play, player stats, snap counts, official team stats,
  schedule) and the season's quarterback identity; (c) key on the raw sources and the identity
  cut to the players the season's sources name, handing the builders that cut identity; (d) no
  cache: rebuild every season every run.
- **Decision**: (a).
- **Reasoning**: a season's tables read only its own frames and its ratings, so they fit the
  design. The players file is downloaded again on every in-season run and gains players every
  week, and every season's identity lists every player in it, so (b) would change every finished
  season's key every week and rebuild its ratings too (they share the entry). (a) keys on the game
  rows, which move only when a player the season's sources name changes. (c) would cut the key's
  cost to a few seconds but hands the builders a trimmed identity, a second difference from the
  source to prove and keep proven. (d) costs every incremental run the full build.
- **Data**: `season_stats_etl_timing_output.txt`, run 1 (the ETL's threads, load averages from 4.7
  to 21.5): the key of the 27 finished seasons 31.9 s and the season in progress 3.3 s, against
  113.1 s for every season's tables. `tests/test_data_collection_incremental.py`, whose every cold,
  warm and full comparison now covers the season statistics tables too:
  `test_a_changed_play_only_the_season_statistics_read_rebuilds_that_season` and
  `test_a_new_player_in_the_players_file_keeps_every_finished_season_cached`.
  `tests/test_season_stats_build.py`:
  `test_key_frames_ignore_a_players_file_row_no_source_of_the_season_names` and
  `test_key_frames_change_when_the_players_file_renames_a_passer_of_the_season`. After the review,
  a season reused from the cache also drops the game rows built for its key, which only building
  the season's tables used to release (`RatingsBuild.reuse`;
  `test_reusing_a_season_drops_the_game_rows_its_key_kept`). `SEASON_STATS_ETL_PARITY.md`,
  "Season cache round trip" (at `3dfb4c6`): every season of 1999-2026 stored in a season cache,
  loaded back and reused; the 534 tables written from them are byte-identical to the fresh
  build's.
- **Feature-value impact**: none.
- **How to reverse it**: for (d), build the tables after the season loop of
  `data_collection.collect_all_data` rather than in `RatingsBuild.build`, and drop the season
  statistics' part of `_season_frames`.
- **Review**: not yet reviewed.

## 8. Those frames are keyed as the season's own, not through the season

- **What differs**: the season cache keys a run-wide frame on its rows through the season and a
  frame of one season on its rows whole (`season_cache.SeasonKeys`).
- **Options**: (a) the season statistics' frames as the season's own frames
  (`data_collection._season_frames`, from `RatingsBuild.season_key_frames`); (b) as run-wide
  frames through `RatingsBuild.key_frames`, with no change to `data_collection.py`.
- **Decision**: (a).
- **Reasoning**: a season's tables read no other season, so under (b) a change to one season's
  frames would rebuild every later season and its ratings, and every key would digest every
  earlier season's play-by-play again; (a) costs one helper in `data_collection.py`.
- **Data**: `test_a_changed_season_statistics_play_changes_only_that_seasons_key` (a 2019 change
  moves 2019's key and keeps 2020's).
- **Feature-value impact**: none.
- **How to reverse it**: return the frames from `RatingsBuild.key_frames` instead.
- **Review**: not yet reviewed.

## 9. Determinism: a two-run test, and the row-order guard over the package as ported

- **What differs**: the task asks for a test that two runs write byte-identical tables, and to
  extend the ETL's row-order guard (`tests/test_etl_determinism.py`) if the new code has
  group-bys.
- **Options**: for the guard, (a) leave it: the new module has no group-by or unique and lies
  under `utils/polars/`, which the guard already scans; (b) extend it to the season statistics
  package and add `maintain_order=True` to each of its calls; (c) extend it to the package and list
  the package's 33 calls that leave row order open, counted by file, enclosing function, call and
  problem, so a new one fails the test while the package stays as ported.
- **Decision**: (c). The step first chose (a); the branch's independent review asked for the guard
  over the package without editing its calls.
- **Reasoning**: (b) edits the reviewed package's 33 group-bys and uniques, and its README notes
  that Polars sums a group's floats in another order on another code path, so the change could
  move published bits. Under (a) a new unordered call in the package would pass unnoticed. The
  listed calls stay because the package fixes its row order by sorting before every reduction (its
  README, "Determinism"), the parity audits match the published files bit for bit, and the tests
  below cover its output through the step.
- **Data**: `tests/test_etl_determinism.py`: `_ETL_PATHS` takes every module under
  `src/nfl_predictor/season_stats/`, and `_REVIEWED_UNORDERED_CALLS` lists the 33 calls (24 keys
  in six files); `test_etl_group_by_and_unique_calls_keep_a_deterministic_order` fails on a call
  outside the list and on a listed call that is gone (an unordered `group_by` added to `wp_bins.py`
  failed it). `tests/test_season_stats_build.py`:
  `test_two_runs_write_byte_identical_season_statistics_tables` (the second run fed whole games
  and every other source in another row order). `season_stats_etl_timing_output.txt`: the 280
  tables written with one Polars thread and with 24, in two processes, have the same digest.
- **Feature-value impact**: none.
- **How to reverse it**: drop the package from `_ETL_PATHS` and the list from the test.
- **Review**: not yet reviewed.

## 10. The web empty states say the ETL writes the tables, in words only

- **What differs**: the team, quarterback, rank-range and head-to-head pages and the Schedules
  page said no job writes their tables.
- **Options**: (a) text: the ETL writes them (`nfl-predictor data`, the Jobs page's Full ETL
  rebuild), and to run it when the folder holds none; (b) the same with an admin-only "Run ETL"
  button, the Data & ETL page's `RunJobButton`.
- **Decision**: (a). The Schedules page also tells an empty folder (run the ETL) from one holding
  only seasons still in progress, which its ranking leaves out (nothing to run).
- **Reasoning**: (b) would put a rebuild of several minutes one click from a read-only page and
  make every ratings page read the job catalog; the Data & ETL and Jobs pages already launch it.
- **Data**: `web/src/pages/Ratings.test.tsx` and `web/src/pages/Schedules.test.tsx`.
- **Feature-value impact**: none.
- **How to reverse it**: the texts in `web/src/components/ratings/RatingsStates.tsx` and
  `web/src/pages/Schedules.tsx`.
- **Review**: not yet reviewed.

## 11. The time the step adds to the ETL

- **What differs**: not a difference; the task asks for the time.
- **Options**: not applicable.
- **Decision**: recorded, not acted on further. What is left of the time is Polars' thread-pool
  overhead on the package's many small queries: the same tables build about 8.5 times faster with
  one Polars thread. Vectorizing the package's per-team and per-passer loops would remove most of
  it, a behavior-preserving change to the reviewed package to check against the parity reports,
  left as a follow-up like the ratings' (`integration_ratings.md`, entry 18).
- **Reasoning**: a full rebuild is rare, and the weekly run is incremental.
- **Data**: `season_stats_etl_timing_output.txt`, run 1 (the ETL's threads, on a machine shared
  with the walk-forwards): a full rebuild about 114 s (every season's tables 113.1 s, the
  official team stats 0.1 s, writing 280 tables 0.6 s); an incremental run about 37 s (the key of
  the 27 finished seasons 31.9 s, the season in progress 3.3 s, the season cache's extra input
  and output 1.0 s, loading 0.1 s, writing 0.6 s). Run 2 (one Polars thread): every season's
  tables 13.3 s, the key 9.4 s.
- **Feature-value impact**: none.
- **How to reverse it**: not applicable.
- **Review**: not yet reviewed.
