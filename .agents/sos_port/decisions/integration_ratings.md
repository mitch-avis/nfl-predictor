# Decisions: the ported ratings in the ETL

Roadmap step 6, the ratings integration: an adapter from this repository's play-by-play to the
ratings package's input rows, the published ratings tables written by the ETL, and the pre-week
rating features in the game dataset. Each judgment call below has the parts
`.agents/sos_port/DECISIONS.md` asks for. Commits on branch `worktree-agent-a0778ee61e6367a09`:
`1632444` (the adapter), `354c8c0` (passers known before the week), `1a9944b` (one starter
function), `5124712` (the feature group), `0e65af0` (the build), `dd9583e` (the season cache),
`7ad13d2` and `5828a1c` (row order), `e16fbcf` (the ETL wiring), `ff0de1a` (the ETL-path leakage
test), `e21da94` (the design builder's speed), `3334075` (the adapter narrows the play-by-play
before filtering it), `5912f73` (the adapter parity script), `8b1b9c7` (a cache entry without
its ratings frames is rebuilt).

Scripts and their reports, run from the repository root with the worktree's environment:

- `.agents/sos_port/adapter_parity.py` writes `.agents/sos_port/ADAPTER_PARITY.md` and
  `adapter_parity.json`: the snapshot of the nflreadpy cache `nfl-sos-ratings` read
  (`~/scratch/nflreadpy_cache_snapshot_20261008/`) loaded through this repository's loaders, the
  adapter's rows against that project's published game logs, every season's tables rebuilt from
  them against its published tables, and how often the identity known before a week differs from
  the season's. `.agents/sos_port/snapshot_cache.py` turns the snapshot into cache files.
- `.agents/sos_port/decisions/integration_ratings_facts.py` and its captured
  `integration_ratings_facts_output.txt`: neutral-site games, null home flags in the published
  logs, empty team codes, and the qualified passers of each published season.
- `.agents/sos_port/ratings_etl_timing.py` and its captured output in this directory,
  `ratings_etl_timing_output.txt`: the time the ratings add to a full and to an incremental ETL
  run, on the snapshot;
  `.agents/sos_port/ratings_build_profile.py` profiles one season's build.

## 1. Where the adapter and the ETL step live

- **What differs**: the plan names an adapter and an ETL step but not their place. The ratings
  package reads no files ("every function takes Polars frames and returns them"), while the ETL
  step loads sources and caches.
- **Options**: (a) the adapter in the package (`ratings/pbp_inputs.py`, pure frames) and the ETL
  step in `utils/polars/ratings_build.py`, beside the other feature builders; (b) both under
  `utils/polars/`; (c) both in the package.
- **Decision**: (a).
- **Reasoning**: the adapter is the play-by-play counterpart of `published_logs.py` and keeps
  the package's no-I/O rule, so its tests need only frames; the loading, the season cache and the
  feature attach are ETL work like `strength_table.py` and `qb_stats.py`. Both directories are in
  the season cache's code fingerprint (entry 15).
- **Data**: `grep -n "import" src/nfl_predictor/ratings/pbp_inputs.py` shows Polars, the
  package and `utils/polars/player_identity.py` only.
- **Feature-value impact**: none.
- **How to reverse it**: move the modules; the imports in `ratings_build.py`,
  `data_collection.py` and the tests follow.
- **Review**: not yet reviewed.

## 2. The adapter keeps `nfl-sos-ratings`' definitions, not this repository's

- **What differs**: this repository's play-by-play family (`utils/polars/pbp.py`) defines snaps,
  dropbacks and special-teams plays its own way (two-point tries out of dropbacks, kneel-downs and
  spikes out of the EPA sums, both sides' special-teams plays). The ratings were fit, and
  published, with `nfl-sos-ratings`' definitions.
- **Options**: (a) build the rows with that project's definitions inside the adapter; (b) feed
  the fits this repository's aggregates.
- **Decision**: (a), decided by the plan ("using the source's definitions internally, so its
  ratings reproduce the published files"). The adapter implements: scrimmage snaps (dropbacks,
  runs, kneel-downs, spikes) with both teams known, a run play flagged `qb_scramble` counting as a
  dropback, `offensive_epa` over every snap, special-teams plays (`special`) for the team with the
  ball, the schedule's margin and home side; and per passer his dropbacks, the pass attempts
  among them, his `qb_epa` with his spikes' left out of the dropback EPA, the weekly player stats'
  attempts and passing EPA for a passer they or the identity list at quarterback, the snap
  counts' quarterbacks without a dropback, the identity's names, and the identity rule.
- **Reasoning**: only (a) reproduces the published files, which is how the port and the adapter
  are checked; the integration design keeps this repository's features as they are and gives the
  ratings their own rows.
- **Data**: `ADAPTER_PARITY.md`, 1999-2025: the adapter's team-game and quarterback-game rows
  equal the published game logs' on every column the fits consume, row for row and in order:
  305,830 cells, all bit-identical; the tables rebuilt from them: 741,455 cells, 741,295
  bit-identical, 160 within 1e-12 (the raw season rate `qb_epa_per_dropback`, a display column
  the ratings parity audit already explains), none outside, no structural difference.
- **Feature-value impact**: the rating features are the ported method's values.
- **How to reverse it**: not applicable while the ratings are the ported ones.
- **Review**: not yet reviewed.

## 3. A neutral site keeps the schedule's home team

- **What differs**: the input contract allows a null `is_home` (no home-field term) for a neutral
  site; `nfl-sos-ratings` sets `is_home` from the schedule's home and away teams for every game.
- **Options**: (a) the schedule's home team is home, as there; (b) null at a neutral site.
- **Decision**: (a).
- **Reasoning**: (b) would change every fit with a neutral-site game and break the reproduction;
  whether the home-field term belongs at a neutral site is a modeling question for a measured arm,
  not for the port.
- **Data**: `integration_ratings_facts_output.txt`: 66 of 6,967 regular-season games of 1999-2025
  were at a neutral site (none before 2003, up to 7 a season since), and the published team-game
  logs carry 0 null home flags.
- **Feature-value impact**: the home-field term reaches those 66 games (132 team-game rows).
- **How to reverse it**: in `pbp_inputs._team_scores`, set `is_home` null where the schedule's
  `neutral` is 1, then rebuild.
- **Review**: not yet reviewed.

## 4. Empty team codes and the fields blanked in early seasons

- **What differs**: `.agents/sos_port/decisions/pbp_columns.md` entries 14 and 15: that project
  turns an empty team code into null and blanks nine fields in the seasons nflverse lacks them;
  this repository's cache keeps both as published, to be handled where the ported code reads them.
- **Options**: for the empty codes, (a) treat an empty `posteam` or `defteam` as missing in the
  adapter, or (b) let it through. For the blanked fields, none to choose.
- **Decision**: (a). The adapter reads none of the nine blanked fields (`air_yards`, `air_epa`,
  `pass_length`, `yards_after_catch`, `yac_epa`, `xyac_mean_yardage`, `qb_hit`, `no_huddle`,
  `drive_yards_penalized`), so entry 15 needs nothing here.
- **Reasoning**: an empty `posteam` would become a team called `""` (the input checks refuse it);
  missing is what the source reads.
- **Data**: `integration_ratings_facts_output.txt`: 39 plays in 1999 and 64 in 2000 have an
  empty `posteam`, none an empty `defteam`; `pbp_inputs._TEAM_PLAY_COLUMNS` and
  `_PASSER_PLAY_COLUMNS` list the columns the adapter reads. Pinned by
  `test_team_games_leave_out_plays_without_both_teams`.
- **Feature-value impact**: none (those plays are no-plays, so no snap is dropped).
- **How to reverse it**: drop `_known_team` from `pbp_inputs`.
- **Review**: not yet reviewed.

## 5. This repository's team codes

- **What differs**: Washington is `WAS` there and `WSH` here.
- **Options**: (a) this repository's codes in the rows and tables; (b) that project's.
- **Decision**: (a), as the task says.
- **Reasoning**: the tables and features join this repository's game rows. The codes decide the
  fits' label order and the rows' order, and `WAS` and `WSH` both sort after every other code,
  so no row or column moves and the fits are bit-identical.
- **Data**: `ADAPTER_PARITY.md` maps the published team columns through
  `constants.ALIAS_TO_CANONICAL` before comparing and finds no difference.
- **Feature-value impact**: none.
- **How to reverse it**: not applicable.
- **Review**: not yet reviewed.

## 6. The quarterback identity known before a week

- **What differs**: the season's identity (`player_identity.qb_identity_crosswalk`) counts every
  regular-season roster week, so a listing in week 12 makes a player a quarterback in week 1
  (`.agents/sos_port/decisions/sources.md`, "Forward notes for the integration").
- **Options**: (a) the season's identity for the published tables, and for the pre-week ratings
  of week `w` an identity built from the players file and the roster weeks before `w` only;
  (b) the season's identity everywhere; (c) the roster weeks up to each game's own week.
- **Decision**: (a), as the task says. Before week `w`, a passer counts when the players file or a
  roster week before `w` lists him at quarterback, or when neither places him at any position;
  the identity's part in choosing the official attempts and passing EPA uses the same weeks; and
  only the games before `w` are returned (`quarterback_rows(..., before_week=w)`, read by
  `RatingsHistory` through its new `load_qb_games_before`). The players file, one current
  snapshot, counts in both.
- **Reasoning**: (b) lets a later roster listing reach an earlier week's feature; (c) is stricter
  than the task's rule and would drop rows the identity knew about by the week being rated.
- **Data**: `ADAPTER_PARITY.md`, "The quarterback identity known before a week": over 464
  season-weeks of 1999-2025 (148,608 passer-games, summed over the weeks) the two identities keep
  the same passer-games, 0 kept by one only, and choose the same official values, 0 differences.
  So the rule changes no feature on the snapshot; it guards against the case.
  `test_features_of_a_week_never_see_that_week_or_later` rewrites every later roster week (and a
  receiver listed at quarterback only from week 5) and fails when the season's identity is wired
  in instead (checked by a temporary edit, not kept).
- **Feature-value impact**: none measured on the snapshot.
- **How to reverse it**: in `ratings_build.RatingsBuild`, drop `load_qb_games_before`.
- **Review**: not yet reviewed.
- **After the review** (`4fb3905`): the leakage case now runs on the synthetic league with weekly
  player stats and snap counts filled in, and rewrites from the week on the plays' EPA, passer
  ids and names and dropback, pass, run, kneel, spike and special-teams flags, the player
  stats, the snap counts, the roster weeks and the scores; the ETL-level case moved from week 2,
  where `ratings_sos` is null by construction, to week 3 and rewrites the same play values and
  the scores.

## 7. The pre-week passer rows are ordered by passer id

- **What differs**: that project's quarterback fits read a season's rows by team, week, game and
  passer name, and the season tables keep that order. The name comes from the players file, then
  the weekly roster, which can be a later week's.
- **Options**: (a) the name order for the season tables and team, week, game and passer id for
  the pre-week rows; (b) the name order everywhere; (c) the id order everywhere.
- **Decision**: (a).
- **Reasoning**: the fits sum in row order, so the order reaches their last bits; a name taken
  from a later roster week could change a pre-week fit's order, the id cannot. (c) would cost the
  tables their bit-for-bit reproduction.
- **Data**: the season tables are bit-identical to the published ones (`ADAPTER_PARITY.md`);
  `test_quarterback_rows_before_a_week_sort_by_team_week_game_and_id` pins the pre-week order.
- **Feature-value impact**: in the last bits of a pre-week quarterback rating at most.
- **How to reverse it**: the `order` lists in `pbp_inputs.quarterback_rows`.
- **Review**: not yet reviewed.

## 8. The expected starter is the quarterback family's

- **What differs**: the feature needs each game's expected starter; the quarterback family maps
  `away_qb` / `home_qb` to GSIS ids inside `qb_stats.attach_qb_features`, and a parallel stream
  is changing which starter every quarterback feature reads.
- **Options**: (a) one function, `qb_stats.expected_starter_ids`, that the family and the ratings
  both read; (b) call the family's private id resolution from the ratings; (c) a resolution of
  their own.
- **Decision**: (a). `attach_qb_features` now reads its starters from it; its tests pass
  unchanged and fail when the function is broken.
- **Reasoning**: one place decides the starter, so the starter definition the parallel stream
  lands reaches both families at once; (b) and (c) could drift apart without a test noticing.
- **Data**: `tests/test_qb_starter_ids.py`; `tests/test_qb_stats.py` passes before and after.
- **Feature-value impact**: none for the quarterback family.
- **How to reverse it**: inline the function back into `attach_qb_features`.
- **Review**: not yet reviewed.
- **After the merge with the play-by-play integration**: the function takes the run's
  play-by-play starters, so a completed game's side reads its play-by-play starter and every
  other side its named one (`2c5d028`; `integration_pbp_elo.md`, entry 1.7), and its name
  fallback reads only the passers of games before the row's week (`06e9863`): a later passer
  sharing an abbreviation (a 2026 rookie `J.Daniels`, say) can no longer make an earlier name
  ambiguous, and a debutant's own debut can no longer name him in a training row while the same
  name stays unknown before kickoff. With today's identity file neither changes a published
  value.

## 9. The quarterback feature's evidence column: dropbacks before the week

- **What differs**: the task asks for the starter's pre-week rating "with whatever count column
  carries its evidence".
- **Options**: (a) his dropbacks this season before the week, the fit's weight; (b) his games
  played this season; (c) his career dropbacks.
- **Decision**: (a), as `ratings_qb_dropbacks`: 0 for an identified starter without a dropback
  this season (week 1 included), null when the starter is unidentified or the season has no
  quarterback fit (1999).
- **Reasoning**: the ridge weights each passer-game by its dropbacks, so they measure how far
  the rating has moved from average; (c) belongs to the quarterback family
  (`qb_history_dropbacks`), whose rating is a career rate, while this one is season-to-date.
- **Data**: `test_attach_gives_a_starter_without_a_dropback_yet_no_rating_and_zero_dropbacks`,
  `test_attach_leaves_an_unknown_starters_quarterback_columns_null`,
  `test_attach_gives_week_one_starters_no_rating_and_zero_dropbacks`.
- **Feature-value impact**: defines the column.
- **How to reverse it**: `_QB_FEATURES` in `ratings_build.py`.
- **Review**: not yet reviewed.

## 10. The feature names and the group's markers

- **What differs**: `FEATURE_GROUP_COLUMN_MARKERS` matches substrings. The source's names collide:
  `sos` is inside `sos_played_adj`, `team_rating` and `*_rating` sit beside TeamRankings'
  `*_rating` columns, and `adj_qb_epa_per_dropback` contains the play-by-play marker
  `epa_per_dropback`.
- **Options**: (a) one prefix no other name contains, `ratings_`; (b) the source's names with a
  suffix; (c) the source's names as they are.
- **Decision**: (a): `ratings_offense`, `ratings_defense`, `ratings_special_teams`,
  `ratings_team`, `ratings_sos`, `ratings_qb`, `ratings_qb_dropbacks`, each per side with a
  `_diff`, the group `ratings`.
- **Reasoning**: (c) would put the ratings in the `pbp` group and let `sos` catch the strength
  family; a prefix makes the group's columns obvious in a feature-importance list.
- **Data**: `tests/test_ratings_feature_group.py`: the group resolves to exactly the 21 columns,
  no other group reaches them, and no marker matches any other name of the final schema or of any
  list, tuple, set or mapping in `constants.py`.
- **Feature-value impact**: none.
- **How to reverse it**: `constants.RATINGS_FEATURE_STATS`.
- **Review**: not yet reviewed.

## 11. Which tables the ETL publishes

- **What differs**: the port builds a `qb_season` table (the rating columns of that project's
  `qb_combined` file) that has no file of its own there, and that project writes weekly rank
  ranges only for the season in progress.
- **Options**: for the kinds, (a) the nine season files and the two weekly ones, named as there,
  or (b) also `qb_season` under a new name; for a finished season's weekly files, (c) leave them
  as last written, or (d) delete them.
- **Decision**: (a) and (c).
- **Reasoning**: the task asks for the kinds named as that project names them; `qb_combined`'s
  statistics belong to the season-statistics stream. A season's weekly ranges are rewritten by
  every run while it is the current season, so its last such run leaves the complete file; the
  ETL deletes nothing in a directory the web stream shares.
- **Data**: `ratings_build.PUBLISHED_KINDS` and `IN_PROGRESS_KINDS`;
  `test_published_tables_are_named_by_season_and_kind`.
- **Feature-value impact**: none.
- **How to reverse it**: add `qb_season` in `RatingsBuild._tables`.
- **Review**: not yet reviewed.
- **After the review** (`37599f3`): every table is written to a temporary file in the same
  directory and moved over the published one, as `lines_refresh._write_atomic` does, so a web
  page that reads `data/ratings/` during a web-launched ETL sees the old file or the new one,
  never half of one; a failed write leaves the old file and no temporary file
  (`test_a_failed_table_write_leaves_the_published_file_whole`).

## 12. A season without a qualified passer gets no quarterback tables

- **What differs**: the package's rank summary fails when no passer qualifies for the rankings
  (it returns a frame without columns, which the join then misses).
- **Options**: (a) skip the season's quarterback tables; (b) fix the package's summary to return
  an empty table; (c) let the ETL fail.
- **Decision**: (a).
- **Reasoning**: only a season in progress right after its first game can have no qualified
  passer, and an empty table carries nothing; (b) changes the reviewed package for a case its
  published files never meet.
- **Data**: `integration_ratings_facts_output.txt`: every published season ranks 30 to 37
  qualified passers. `test_build_of_a_season_without_a_qualified_passer_publishes_only_team_tables`.
- **Feature-value impact**: none (the features read the pre-week tables).
- **How to reverse it**: `_has_qualified_passer` in `ratings_build.py`.
- **Review**: not yet reviewed.
- **After the review** (`9f2e7a3`): the guard asks for a qualified passer among the passers the
  fit rates (a game with a dropback). A season whose qualified passers had no dropback recorded
  passed the old guard and then stopped the ETL with `ColumnNotFoundError` in the rank summary,
  for the same reason as the season without a qualified passer
  (`test_quarterback_tables_need_a_qualified_passer_the_fit_rates`).

## 13. The weeks that get pre-week ratings

- **What differs**: the package rates any week asked for.
- **Options**: (a) every scheduled week of the season, a postseason week reading the week after
  the regular season (the whole regular season), fitted once; (b) every week from 1 to 22.
- **Decision**: (a).
- **Reasoning**: those are the weeks game rows have; the postseason weeks share one fit.
- **Data**: `test_pre_week_teams_cover_every_team_in_every_scheduled_week`,
  `test_a_postseason_week_takes_the_whole_regular_season`.
- **Feature-value impact**: defines the postseason rows' values.
- **How to reverse it**: `RatingsBuild._weeks`.
- **Review**: not yet reviewed.

## 14. How the ratings' sources load

- **What differs**: the ratings need every season from 1999 (penalties from the season before,
  the prior from every earlier one), whatever `--min-season` says, and the nflverse player
  sources the ETL did not read until now.
- **Options**: (a) reuse the ETL's play-by-play and schedules, read the earlier seasons from the
  cache without a refresh, read each player source only for the seasons it publishes, and refresh
  the players file when the run reaches the current season, falling back to its cached copy;
  (b) load everything again for the ratings.
- **Decision**: (a). The play-by-play and schedules go through `polars_utils`, the seam the rest
  of the ETL reads them through. A run without any play leaves the ratings columns null and reads
  no other source.
- **Reasoning**: the quarterback family already reads its history this way; the player sources'
  cache contract and the identity's players-file rule come from the sources stream
  (`decisions/sources.md`).
- **Data**: `tests/test_ratings_build_sources.py`.
- **Feature-value impact**: none.
- **How to reverse it**: `ratings_build.load_ratings_sources`.
- **Review**: not yet reviewed.
- **After the merge with the play-by-play integration**: the loader asks only for the
  play-by-play seasons before the first one the ETL loaded, so the current season, which has no
  plays before kickoff, is requested once (`0cb7c00`;
  `test_sources_never_ask_again_for_a_season_the_etl_asked_for`). The ETL now loads the sources
  through `load_ratings_sources` itself and hands the players file and weekly rosters it loaded
  to every season's quarterback identity, so a current-season run downloads each of them once
  (`d9a2c6b`; `nflverse_sources.QbIdentitySources`).

## 15. The ratings in the season cache

- **What differs**: the season cache stored a season's game rows and strength snapshots only.
- **Options**: (a) store each season's ratings frames (published tables and pre-week tables) in
  its entry, key the entry on the ratings' team-game and passer-game rows through the season, and
  add the ratings package to the code fingerprint; (b) a cache of the ratings' own; (c) no cache,
  rebuilding the ratings every run.
- **Decision**: (a); the entry format moves to version 2, so older entries read as misses once,
  and an entry whose manifest lists no ratings frames is rebuilt, like a damaged one.
- **Reasoning**: a season's ratings read only it and earlier seasons, as its game rows do, so they
  fit the cache's design; (c) costs every incremental weekly run the full rebuild's time
  (entry 18).
- **Data**: `tests/test_data_collection_incremental.py`: a cold and a warm incremental run write
  the same CSVs and ratings tables as a full rebuild; a warm run fits only the current season; a
  changed play in a finished season rebuilds that season and later ones; a changed play of the
  current season keeps every finished season's key; an entry whose manifest lists no ratings
  frames is rebuilt with the output unchanged. `tests/test_season_cache_ratings.py`.
- **Feature-value impact**: none.
- **How to reverse it**: drop the `ratings` field of `season_cache.SeasonBuild` and the ratings
  frames from `data_collection._open_season_cache`.
- **Review**: not yet reviewed.

## 16. One BLAS thread, a fixed row order, and the order guard

- **What differs**: the task asks for one pinned BLAS thread, a fixed row order, and the ETL's
  row-order guard extended to the ratings package, whose group-bys and uniques did not ask for
  their order.
- **Options**: for the threads, (a) `threadpoolctl.threadpool_limits(1, "blas")` around each
  season's build, or (b) around every fit; for the guard, add `maintain_order=True` to every
  Polars group-by and unique in the package.
- **Decision**: (a), with `threadpoolctl` now a direct dependency; and the guard as asked, with
  `numpy.unique` exempt (always sorted, no such argument).
- **Reasoning**: entering the limit costs a library scan, so once per season; every one of the
  22 group-bys and uniques is sorted afterward, joined by key, or reduced to a set, a maximum or
  a dict, so keeping order changes no value.
- **Data**: `test_build_fits_with_one_blas_thread` (two threads outside, one inside);
  `test_two_runs_write_byte_identical_tables` and `test_two_runs_attach_identical_features` (the
  second run fed its games, schedule, rosters and players in another order; each game keeps
  nflverse's play order, which the EPA sums follow); `ADAPTER_PARITY.md`'s counts are the same
  before and after the change.
- **Feature-value impact**: none.
- **How to reverse it**: drop the `threadpool_limits` context in `RatingsBuild.build`.
- **Review**: not yet reviewed.

## 17. Building the ridge designs from plain values

- **What differs**: `ridge.build_unit_design` read each column through a Polars expression, and a
  season's pre-week ratings build about 2,300 designs. With Polars' default thread pool (every
  core) on a busy machine, each such query cost about 2 ms.
- **Options**: (a) read each column once as plain values and index them in Python and numpy;
  (b) build the ratings in a subprocess with one Polars thread; (c) accept the cost.
- **Decision**: (a).
- **Reasoning**: it yields the same labels, matrix, weights and response, so no value moves, and
  it removes most of the cost; (b) adds a process boundary to the ETL; (c) would add about half an
  hour to a full rebuild.
- **Data**: `ratings_etl_timing_output.txt`, runs 2 and 3: a full rebuild of 1999-2001 took
  173.8 s with the default 24 Polars threads and 29.8 s with one before this change, and 61.9 s
  with the default threads after it, on a machine with load averages between 7 and 17. A
  profile of the 2001 build (`ratings_build_profile.py`) before the change spent most of its time
  in Polars queries made by `build_unit_design`. The bit-exact parity tests and
  `ADAPTER_PARITY.md`'s counts are unchanged.
- **Feature-value impact**: none.
- **How to reverse it**: revert `e21da94`.
- **Review**: not yet reviewed.

## 18. The time the ratings add to the ETL

- **What differs**: not a difference; the task asks for the time.
- **Options**: not applicable.
- **Decision**: recorded, not acted on further: a full rebuild is rare and the weekly run is
  incremental. What is left of a season's time is Polars' thread-pool overhead on the package's
  per-team and per-passer refits (about 10,000 small filters and joins a season); building each
  week's design once and refitting it through numpy masks would remove most of it, a larger change
  to the reviewed package left as a follow-up.
- **Reasoning**: the ratings fit every week of every season with head-to-head refits, which no
  other ETL step does.
- **Data**: `ratings_etl_timing_output.txt`, run 1 (1999-2026, the ETL's default threads, load
  averages 8 to 12): loading the other sources and building the adapter's rows 9.0 s, building
  every season 442.8 s (a full rebuild), building only 2026 (the season in progress, with its
  weekly rank ranges) 21.4 s (an incremental run), the season cache's extra input and output
  0.6 s, attaching the features 0.2 s, writing the 254 tables 0.2 s. The incremental step grows
  through the season, since the season in progress refits its weekly rank ranges for every week
  played: the independent review timed 2025 built as the season in progress at 51.5 s against
  19.5 s for the same season finished, so late in a season an incremental run spends about 60 s
  on the ratings.
- **Feature-value impact**: none.
- **How to reverse it**: not applicable.
- **Review**: not yet reviewed.

## 19. A team whose games have no final score keeps its ratings row

- **What differs**: the published `ratings` table inner-joined the score-based SRS (and the
  schedule strength and game count) onto the fit's teams, and the SRS solve drops every game
  without a final score, so a team whose played games all lacked one vanished from the table.
- **Options**: (a) left-join them and leave such a team's SRS null, with a warning; (b) warn and
  keep the inner join.
- **Decision**: (a) (`519444c`; `ratings.season.build_team_ratings`).
- **Reasoning**: the fit rates the team from its plays, so the table should carry it; only the
  score-based column has nothing to report.
- **Data**: `test_build_team_season_keeps_teams_whose_games_have_no_final_score`; the published
  parity test is unchanged, so every table with its scores present is the same.
- **Feature-value impact**: none (the features read the pre-week tables).
- **How to reverse it**: the inner joins in `build_team_ratings`.
- **Review**: not yet reviewed.
