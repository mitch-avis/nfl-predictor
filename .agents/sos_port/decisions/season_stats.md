# Decisions in the season statistics port (roadmap step 6)

Every judgment call made while porting `nfl-sos-ratings`' season statistics into
`src/nfl_predictor/season_stats/` (its team and quarterback game logs, season rows, opponent
profiles, win-probability bins, schedule leaderboard, and metric registry), and every place the
port departs from that project at its frozen commit `9081bce`. The parity audit rebuilds every
published table of these kinds for 1999-2025 from the snapshot of nflreadpy's disk cache that
project's files were built from, read through this repository's loaders, and compares them cell by
cell: 270 tables, 11,754,571 cells, all bit-identical, 0 column or row differences
(`.agents/sos_port/SEASON_STATS_PARITY.md`). So the departures below change how inputs are read,
how the code is organized, or a column's integer width, never a published value.

Commands run from this repository's root with the worktree's environment:

- "The audit": `.venv/bin/python .agents/sos_port/season_stats_parity.py` (writes
  `SEASON_STATS_PARITY.md` and `season_stats_parity.json` beside itself);
- "the facts": `.venv/bin/python .agents/sos_port/decisions/season_stats_facts.py`, output in
  `season_stats_facts_output.txt` beside this file;
- "the parity tests": `.venv/bin/python -m pytest -p no:cacheprovider -n 0 --no-cov
  tests/test_season_stats_published_parity.py`;
- "the comparator check": `.venv/bin/python .agents/sos_port/decisions/season_stats_audit_check.py`,
  output in `season_stats_audit_check_output.txt` beside this file.

## 1. Where the package and the metric registry live

- **What differs**: that project keeps the statistics in `nfl_sos_ratings/` and the registry in
  `nfl_sos_ratings/metrics/`, which its ETL, API, and frontend all read. Here they are a new
  package, `src/nfl_predictor/season_stats/`, with the registry at
  `nfl_predictor.season_stats.metrics` and the column names in its own modules.
- **Options**: (a) one package with the registry inside it; (b) the registry as its own top-level
  package (`nfl_predictor.metrics`); (c) the registry under the web API
  (`nfl_predictor.api.registry`, beside the existing column registry); (d) the modules under
  `utils/polars/` and the names in `constants.py`.
- **Decision**: (a).
- **Reasoning**: the registry validates the tables' columns when they are built and serves the
  web pages, so both the ETL side and the API must import it; inside the statistics package it
  imports none of the builders, so the API can import it alone, and the ETL never imports the API
  (option (c) would make it). Option (d) would change the incremental ETL's
  season-cache fingerprint (`utils/` is in it) and every walk-forward checkpoint fingerprint
  (`constants.py`) for code that is not wired in, the reason the ratings port gave (its decision
  1). The integration stream can move feature-facing names into `constants.py` when it wires the
  tables in.
- **Data**: nothing outside the package imports it: `grep -rln
  'nfl_predictor\.season_stats\|from nfl_predictor import season_stats' src/ | grep -v
  '^src/nfl_predictor/season_stats/'` prints nothing. Inside `metrics/`, the only import from
  outside the registry is the project logger, in the catalog generator (`grep -h 'from
  nfl_predictor\.' src/nfl_predictor/season_stats/metrics/*.py | grep -v season_stats.metrics`).
- **Feature-value impact**: none; no fingerprint or cache key changes.
- **How to reverse it**: move `metrics/` to another package and update its imports.
- **Review**: not yet reviewed.

## 2. The rank-range settings the registry describes are restated, not imported

- **What differs**: that project's registry imports `BOOTSTRAP_RESAMPLES`, `RANGE_QUANTILES` and
  `quantile_suffix` from its `rating_ranges` module, which the ratings port carries as
  `nfl_predictor.ratings.rating_ranges`. Here `metrics/rank_ranges.py` restates the three.
- **Options**: (a) restate them; (b) import them from the ratings package.
- **Decision**: (a).
- **Reasoning**: the ratings package is on another branch that has not merged into the
  integration branch, so (b) would leave this branch unable to import its own registry; restating
  three values also keeps the registry free of the ratings code's numpy imports. A test pins the
  values to the published column suffixes.
- **Data**: `test_rank_range_settings_match_the_published_rank_ranges` in
  `tests/test_season_stats_registry.py` (1000 resamples; suffixes `_q025` through `_q975`); the
  ratings branch's `rating_ranges.py` has the same values (`git show
  worktree-agent-af021dc0836482a44:src/nfl_predictor/ratings/rating_ranges.py | sed -n 27,29p`).
- **Feature-value impact**: none.
- **How to reverse it**: once both are merged, import the three from
  `nfl_predictor.ratings.rating_ranges` in `metrics/registry.py` and delete `rank_ranges.py`.
- **Review**: not yet reviewed.

## 3. The inputs come through this repository's loaders, plus what that project's loaders added

- **What differs**: that project reads nflverse straight through nflreadpy and prepares each
  source in its `data_loader.py`. Here `sources.load_season_inputs` reads play-by-play through
  `loaders.load_pbp`, the schedule through `loaders.load_schedule`, and weekly player stats, snap
  counts, and the quarterback identity through `utils/polars/nflverse_sources.py`, and then does
  what those loaders do not: every play-by-play team code made canonical in all 23 team columns and
  in the yard-line text, an empty code read as null, the fields nflverse never charted in a season
  set to null, and the player teams of Jacksonville's 2001-2002 home games rebuilt from the season
  roster (the repair `player_team_repair.repair_pbp_player_teams` ports, which `load_pbp` does not
  call). The builders themselves (`game_logs.py`) take frames and read nothing.
- **Options**: (a) the repository's loaders plus the source's remaining steps in `sources.py`;
  (b) the source's loaders ported whole, reading nflreadpy directly; (c) extend the repository's
  loaders with the steps.
- **Decision**: (a).
- **Reasoning**: the task asks for this repository's loaders, which carry the cache contract
  (historical seasons from the cache, the current season degrading to it); (b) would bypass it.
  (c) edits `utils/`, which changes feature-facing ETL code and its fingerprint; the play-by-play
  cache's column list and team-code normalization are another stream's change already.
- **Data**: the audit's play-by-play build reproduces every published table bit for bit through
  these loaders; the parity tests do the same for five games served as nflreadpy served them.
- **Feature-value impact**: none.
- **How to reverse it**: replace `load_season_inputs` with direct nflreadpy reads.
- **Review**: not yet reviewed.

## 4. The official weekly team stats have their own cached source

- **What differs**: that project reads the official weekly team stats with
  `nfl.load_team_stats(summary_level="week")` each run. This repository's team-stat loader
  (`loaders.load_team_stats`) renames columns and sums the passing and rushing first downs and the
  sack and rushing lost fumbles into totals, dropping the parts the override reads. Here
  `sources.OFFICIAL_TEAM_STATS` is a second source on the nflverse player-source loader
  (`nflverse_sources.SeasonSource`), cached as `official_team_stats_<season>_reg.parquet`, with the
  19 columns the override reads.
- **Options**: (a) a new cached source; (b) keep the parts in `loaders.load_team_stats`; (c) read
  nflreadpy directly.
- **Decision**: (a).
- **Reasoning**: (b) changes the columns of a cache the ETL reads, invalidating it, for a change
  that wires nothing in; (c) bypasses the cache contract. (a) reuses the reviewed loader without
  editing it.
- **Data**: the facts, section 3: `fumbles_lost` sums `sack_fumbles_lost` and
  `rushing_fumbles_lost`, and `first_downs` sums `passing_first_downs` and `rushing_first_downs`,
  all four read by the override (`game_logs.OFFICIAL_TEAM_STAT_COLUMNS`).
- **Feature-value impact**: none; once the integration stream runs it, one new cache file per
  season.
- **How to reverse it**: point `load_season_inputs` at another team-stat loader.
- **Review**: not yet reviewed.

## 5. Washington is `WSH` in every table

- **What differs**: that project writes Washington as `WAS`; this repository's canonical code is
  `WSH` (decided for the new sources, `sources.md`), and the loaders normalize to it. Game ids keep
  nflverse's `WAS`, as there.
- **Options**: keep `WSH`; write `WAS` in these tables.
- **Decision**: keep `WSH`.
- **Reasoning**: one code per team across the repository; the web pages will join these tables to
  the predictions. It is a label, so the audit reads the published `WAS` as `WSH` in the team-code
  columns and counts those cells; both codes sort after every other team, so no row moves.
- **Data**: the audit: 36,060 published team-code cells read as `WSH`. The comparator check:
  writing `WAS` back into the `team` column of the 2024 team game logs fails on 17 cells,
  Washington's 17 games.
- **Feature-value impact**: none.
- **How to reverse it**: map `WSH` to `WAS` on the tables' team columns before writing them.
- **Review**: not yet reviewed.

## 6. Whole numbers from this repository's loaders are 64-bit

- **What differs**: nflverse publishes the weekly player stats' defensive counts and the snap
  counts' week as 32-bit integers, which that project's files keep; this repository's loaders read
  every whole-number column as a 64-bit integer. So six `team_game_logs` columns (`def_*`) are
  `Int64` here and `Int32` there in every season, and `qb_game_logs.week` in 2013-2025.
- **Options**: keep 64-bit; cast to the published widths.
- **Decision**: keep 64-bit.
- **Reasoning**: a width is not a value: the audit compares every value of these columns and all
  are equal. Casting would add code to imitate a type nflverse could change.
- **Data**: the audit's "Column types that differ" section lists the seven columns and seasons.
- **Feature-value impact**: none.
- **How to reverse it**: cast those columns in `game_logs.build_team_game_logs` and
  `build_qb_game_logs`.
- **Review**: not yet reviewed.

## 7. The row order every float sum reads is fixed

- **What differs**: that project leaves the order of the team-game and passer-game rows to its
  joins, which Polars documents as unspecified without `maintain_order`, and its season rows and
  opponent profiles sum floats in that order. Here the builders return the rows sorted by team,
  week, and game (passers then by name and id), the two profile joins keep the left rows' order
  and then the right's, and `build_season_tables` sorts the rows again on the way in. Plays are
  not re-sorted.
- **Options**: (a) pin the orders; (b) leave them as that project does.
- **Decision**: (a).
- **Reasoning**: the same inputs must give the same bits on every run, as the ratings port's
  review required. Polars also sums a group's floats in another order when the frame is flagged
  sorted by the group key, so sorting inside `build_season_tables` gives every caller, including
  the published-log adapter, the same order and the same flag. The play order is the one nflverse
  publishes and the cache keeps; re-sorting plays by `play_id` would change real values, since
  `play_id` does not increase within thousands of games. No linear algebra is involved, so there is
  no BLAS thread count to pin.
- **Data**: the audit is bit-identical at one Polars thread (the committed report) and at 24 (the
  audit run with `POLARS_MAX_THREADS=24` and `--out-dir` pointing elsewhere: 11,754,571 of
  11,754,571 cells, on the committed code). Before the sort in
  `build_season_tables`, reversed input rows gave other bits in `team_per_game_stats`,
  `opponent_profiles`, and `combined`
  (`test_build_season_tables_sum_each_teams_games_in_one_order_whatever_the_input_order` fails with
  that sort removed); with neither that sort nor the published-log adapter's, the season rows
  built from the 2002 and 2024 published logs differed from the published ones in 137 and 167
  cells of `team_per_game_stats` (and the same cells of `combined`), by at most 3.6e-15 and
  7.1e-15 (`--seasons 2002 2024` with both sorts removed). A
  name tie between two passers in one game is now ordered by id
  (`test_build_qb_game_stats_breaks_a_name_tie_by_passer_id`). The facts, section 1: 2,279 games
  of 1999-2025 have a `play_id` that does not increase in publication order (6 in 1999; 272 of 272
  in 2023-2025); section 4: no numpy, scipy, or sklearn import in the package.
- **Feature-value impact**: none; every published cell is reproduced bit for bit.
- **How to reverse it**: drop the sorts in `game_logs.build_weekly_team_stats`,
  `build_qb_game_stats`, and `season.build_season_tables`, and the `maintain_order` arguments in
  `team_stats.compute_team_stats_excluding_opponents` and
  `qb_opponent_stats._qb_allowed_stats_by_pair`.
- **Review**: not yet reviewed.

## 8. The published-log check covers what the published logs determine

- **What differs**: the task asks for an adapter that reads the published game logs to check the
  season tables. The published logs leave out every rate's hidden numerator and denominator, which
  the season rows and opponent profiles pool over games. `published_logs.py` rebuilds exactly the
  parts the published columns determine, as the code defines them (the per-snap rates, the offense
  ratios of published columns, and their defensive mirrors from the opponent's row), and the
  audit's second comparison checks every column that does not need another rate's parts.
- **Options**: (a) rebuild the determined parts and leave the rest uncompared; (b) estimate the
  missing parts (a published rate times a published denominator where one exists); (c) no
  published-log layer, only the play-by-play build.
- **Decision**: (a).
- **Reasoning**: (b) would invent values that project never had, with rounding error, and still
  could not cover rates whose denominator the logs do not carry (passes with a completion
  probability over expected, third-down plays, drives with a starting yard line). The full check of
  those columns is the play-by-play build, which reproduces the published game logs bit for bit
  and every season table from them; the published-log layer adds an independent check of the
  season-table code on the published inputs.
- **Data**: the audit's second comparison: 1,125,975 cells compared, all bit-identical; 276,432
  cells not determined by the published logs; 77 team rates and 1 quarterback rate whose parts the
  logs do not determine, listed in the report.
- **Feature-value impact**: none.
- **How to reverse it**: extend `published_logs._own_specs` if a later published log carries more
  parts.
- **Review**: not yet reviewed.

## 9. The published-log check takes each team's opponents from the schedule

- **What differs**: a first version of the adapter rebuilt the schedule from the published team
  logs. Three regular-season games have a final score but no play-by-play, so no log rows, yet that
  project counts their teams as opponents (its profiles read the schedule). The adapter now takes
  no schedule; the audit and the parity tests use the loaders' played schedule, as that project's
  build did.
- **Options**: the loaders' schedule; a schedule rebuilt from the logs.
- **Decision**: the loaders' schedule.
- **Reasoning**: it is the input that project read. A schedule from the logs silently loses a
  team's only meeting with an opponent.
- **Data**: the facts, section 2: `1999_01_BAL_STL`, `2000_03_SD_KC`, `2000_06_BUF_MIA`. With the
  schedule rebuilt from the logs, the audit found 300 cells outside tolerance in 1999
  (`opponent_profiles` and the `opp_` columns of `combined`, Baltimore's and the Rams'; the two
  2000 games are division rivals' second meetings, so their profiles kept the opponent); with the
  loaders' schedule, none.
  `test_the_season_without_a_games_play_by_play_still_counts_it_as_an_opponent` pins it.
- **Feature-value impact**: none.
- **How to reverse it**: not advised; it would drop opponents with games without play-by-play.
- **Review**: not yet reviewed.

## 10. The play-by-play columns the build reads

- **What differs**: that project reads every play-by-play column nflverse publishes. Here
  `sources.SEASON_STATS_PBP_COLUMNS` names the 122 the builders and the Jacksonville repair read,
  and `load_season_inputs` refuses play-by-play that lacks one, rather than reading a missing
  column as zeros (the builders' expressions default an absent column to 0).
- **Options**: a declared list with a guard; read whatever the cache holds.
- **Decision**: the declared list with the guard.
- **Reasoning**: a missing column would silently zero every statistic built on it. The audit sets
  the play-by-play cache's column list to exactly these columns and reproduces every table, so the
  list is sufficient.
- **Data**: the audit's "Play-by-play columns" section: 74 of the 122 are missing from this
  branch's `constants.PBP_COLUMNS`; all 122 are in the 139-column list of the play-by-play cache
  stream's branch at `3870a39` (`git show
  worktree-agent-aee4f463ac1335152:src/nfl_predictor/constants.py`, compared with
  `SEASON_STATS_PBP_COLUMNS`). Until that list merges, `load_season_inputs` raises for any season
  with plays; the audit and the parity tests set the list themselves.
- **Feature-value impact**: none.
- **How to reverse it**: drop the guard in `load_season_inputs`.
- **Review**: not yet reviewed.

## 11. What was not ported, and why

- **What differs**: these parts of that project's code are not in the package:
  - `wp_filter.py`: no published table depends on it (its web view refits the ratings on the plays
    a garbage-time filter keeps, from the bins this package builds); left for the web stream.
  - `srs.py`: `SRS` belongs to the ratings table, which the ratings package builds with this
    repository's `utils/polars/adjusted_strength.solve_srs` (that port's decision 5).
  - In `data_loader.py`: ESPN QBR (ported with the new sources as `utils/polars/espn_qbr.py`), the
    quarterback identity crosswalk (`utils/polars/player_identity.py`), the postseason loaders (its
    validation checks only), and the switch to nflreadpy's disk cache (this repository caches
    every source itself).
  - In `main.py`: the file writes and the rating outputs. `_write_data_file` became
    `season.finalize_table` (registry check and row order, no write); the ratings come from the
    ratings package.
  - `schedules.py`'s command: only its ranking is ported, as a function of each season's ratings
    (`schedule_ranks.rank_schedules`).
- **Options**: port each; leave each to the stream that needs it.
- **Decision**: leave them.
- **Reasoning**: the task covers the per-season statistics tables and the registry, with no wiring
  and no commands; each part above either exists here already or belongs to the web or
  integration stream.
- **Data**: `main.py` at `9081bce` never names `wp_filter` (`grep -c wp_filter
  nfl_sos_ratings/main.py` in an archive of that commit prints 0); the published file kinds are the
  ten the audit compares plus the ratings files.
- **Feature-value impact**: none.
- **How to reverse it**: port a part when a stream needs it.
- **Review**: not yet reviewed.

## 12. The stats catalogs live beside the package

- **What differs**: that project generates `docs/stats-catalog.md` and `docs/qb-stats-catalog.md`
  with its `catalog` command. Here they are `stats-catalog.md` and `qb-stats-catalog.md` beside
  the package README, regenerated with `.venv/bin/python -m
  nfl_predictor.season_stats.metrics.catalog_docs` (no front-door command), which logs each file
  it writes instead of printing.
- **Options**: beside the package; a new top-level `docs/`.
- **Decision**: beside the package.
- **Reasoning**: this repository documents a package beside it (the ratings package does the
  same); the module command keeps the CLI unchanged.
- **Data**: the generated bodies equal that project's files from the first line of prose on
  (`diff` of each pair from the line starting "Every column below", run against an archive of
  `9081bce`); only the header lines differ.
- **Feature-value impact**: none.
- **How to reverse it**: change `catalog_docs.CATALOG_PATHS`.
- **Review**: not yet reviewed.

## 13. Smaller adaptations of the ported code

- **What differs**:
  - Division rivals in the opponent profiles' details (a side output, not a published column)
    come from this repository's modern `constants.TEAM_TO_DIVISION`, as that project used its
    modern map for every season, not the season-aware pre-2002 map this repository also has.
  - The defensive mirror tables, the offense ratio specs, the giveaways-per-drive ratio, and the
    quarterback log context columns became module constants so the adapter reads the code's own
    definitions.
  - The source's "Tier 1" label was dropped from docstrings (no plan labels in code).
  - Tests that read that project's `data/` directory (marked `published_data` there) read the
    parity fixtures instead; its check that every season's bins leave at most 0.1% of plays
    without a bin is covered by the audit reproducing every bin.
- **Options**: as above, or verbatim.
- **Decision**: as above.
- **Reasoning**: each keeps a published value unchanged while following this repository's rules.
- **Data**: the audit (every cell bit-identical).
- **Feature-value impact**: none.
- **How to reverse it**: per item, in the named modules.
- **Review**: not yet reviewed.

## 14. Parity: what is compared, the tolerance, and the fixtures

- **What differs**: nothing in the port; these are choices about how parity is judged.
- **Options**: exact equality everywhere; a float tolerance with bit-identical cells counted.
- **Decision**: floats within 1e-9 absolute (chosen before the first run), integers, text, flags,
  columns, and row order exactly; a column's type difference reported, its values still compared
  (exactly where the published column holds whole numbers); bit-identical cells counted. The tests
  use the same rule on fixtures: five games' nflverse inputs (Washington's and the Rams' codes in
  2024, scrambles coded as runs in 2004, Jacksonville's 2001 home game credited to the visitor, a
  1999 game) served as nflreadpy served them and built through the loaders, and the 1999 and 2024
  published logs for the season tables. Fixtures are Parquet (935 KB, 47 files), re-included by a
  `.gitignore` line at the end of the file, with provenance and hashes.
- **Reasoning**: bit identity depends on the CPU's kernels and library versions, so the tolerance
  keeps the check meaningful elsewhere, and the counts show the result is stronger. A sum of plays'
  EPA differs from another summation order by about 1e-15; 1e-9 is far below any single play's
  value.
- **Data**: the audit: 0 cells outside the tolerance and 0 within it but not bit-identical, in both
  comparisons. The comparator check: a copy of the 2024 team game logs fails with a text, an
  integer, a null, or a float moved by 1e-6 changed in one cell, with its rows reversed or cut, its
  columns reversed, or Washington's code written back, and passes with a float moved by 1e-12 (one
  cell within tolerance) or unchanged. Two runs of
  `.venv/bin/python .agents/sos_port/season_stats_fixtures.py` write identical files.
- **Feature-value impact**: none.
- **How to reverse it**: change `FLOAT_TOLERANCE` in the audit and the parity tests.
- **Review**: not yet reviewed.
