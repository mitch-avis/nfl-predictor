# Season statistics package

`nfl_predictor.season_stats` holds the season statistics of the separate, read-only
`nfl-sos-ratings` project, ported from its frozen commit `9081bce`: the per-season tables its web
pages show beside the ratings, and the metric registry that names and describes every column. The
port reproduces the tables that project published, cell by cell (see "Checking the port" below).

`nfl-predictor data` builds its tables for every season of the run and writes them to
`data/ratings/` beside the ratings (see "In the ETL" below). Nothing in the game dataset, the model,
or the weekly run's predictions uses them, and no default depends on them. The web app's team and
quarterback pages (`src/nfl_predictor/api/ratings/`) read the tables and describe their columns
from the package's metric registry.

The tables are whole-season display data, not values known before a week's games: a season row
pools every game played so far, and an opponent profile pools every played game of each
opponent against everyone else, later weeks included. A model feature built from them needs its
own pre-week version, solved from the games strictly before each week, as `AGENTS.md` requires
of every schedule- or opponent-adjusted value.

The season cache's code fingerprint (`_ETL_SOURCE_DIRECTORIES` in `utils/season_cache.py`) covers
this package, so an edit here rebuilds every season an incremental ETL run had cached.

## What it computes

- **Game logs** (`game_logs.py`): one row per team-game from play-by-play (`team_stats.py`,
  `team_stats_expanded.py`), with the passing, rushing, first-down, sack, interception, and
  lost-fumble totals replaced by nflverse's official weekly team stats, the defense-only box score
  from weekly player stats, and the score from the schedule; and one row per passer-game from
  play-by-play and snap counts (`qb_stats.py`), with the official passing line and rushing fields
  from weekly player stats. Every rate carries its numerator and denominator as hidden columns
  (`pooled_rates.py`), so rows for several games pool it as their summed numerator over their
  summed denominator, never a mean of rates.
- **Season rows**: one row per team (counts per game, rates pooled over the games, longest plays
  as season maxima, the record) and one per passer (totals, per-game counts, pooled rates, and the
  ranking qualifier of 14 attempts per game of his primary team).
- **Head-to-head-excluded opponent profiles** (`opponent_stats.py`, `qb_opponent_stats.py`): what
  a team's opponents did against everyone else, and what the defenses a passer faced allowed to
  other passers, each opponent measured without its games against the team.
- **Win-probability bins** (`wp_bins.py`): each team-game's and passer-game's plays and EPA by
  win-probability bin, the inputs of a garbage-time filter view.
- **The schedule-strength leaderboard** (`schedule_ranks.py`): every completed team-season ranked
  on `sos`, from each season's ratings.
- **The metric registry** (`metrics/`): every column's label, full name, layman description,
  shape, polarity, source, and category, the prefix and suffix rules for derived columns (`opp_`,
  `qopp_`, per game, per snap, rank-range percentiles), the validation that refuses a column it
  cannot resolve, and the payload a web API can serve. The stats catalogs beside this README,
  [stats-catalog.md](stats-catalog.md) and [qb-stats-catalog.md](qb-stats-catalog.md), are
  generated from it (`.venv/bin/python -m nfl_predictor.season_stats.metrics.catalog_docs`), and a
  test fails when they drift.

## Inputs

`sources.load_season_inputs(season)` reads one season through this repository's loaders and
returns its six inputs (`SeasonInputs`):

- play-by-play from `loaders.load_pbp`, then every team code made canonical in the team columns
  and the yard-line text (an empty code becomes null), the fields nflverse never charted in a
  season set to null (`PBP_FIELD_GAPS`), and the player teams of the games nflverse credits to one
  team (Jacksonville's 2001-2002 home games) rebuilt from the season roster. It refuses
  play-by-play that lacks a column the builders read (`SEASON_STATS_PBP_COLUMNS`), which
  `constants.PBP_COLUMNS` must therefore hold;
- weekly player stats, snap counts, and the season's quarterback identity from
  `utils/polars/nflverse_sources.py`;
- the schedule from `loaders.load_schedule`, regular-season games under the team names the
  builders read;
- the official weekly team stats from their own cached source (`OFFICIAL_TEAM_STATS`, cache files
  `official_team_stats_<season>_reg.parquet`), because the team-stat loader sums away the separate
  first downs and lost fumbles the override reads.

Washington is `WSH` here, as everywhere in this repository; that project writes `WAS`.

## Outputs

`season.build_season_games(inputs)` returns the game rows and the played schedule, and
`season.build_season_tables(games, pbp, team_ratings=..., qb_ratings=...)` every table, under the
names of that project's files (`{season}_<name>.parquet` there), each validated against the
registry and in its fixed row order (`row_order.py`): `team_game_logs`, `qb_game_logs`,
`team_per_game_stats`, `qb_per_game_stats`, `qb_opponent_profiles`, `opponent_profiles`,
`combined`, `qb_combined`, `team_wp_bins`, and `qb_wp_bins`. The combined tables take their
rating columns from the caller (`TEAM_RATING_INPUT_COLUMNS`, `QB_RATING_INPUT_COLUMNS`), as the
ratings package (`nfl_predictor.ratings`) builds them. Before kickoff the current season has a
schedule and no game rows: `SeasonGames.played` is false, and `build_season_tables` refuses it
rather than tabulate nothing.

`published_logs.py` turns the game logs that project published back into the rows the season
tables read. The published logs leave out each rate's hidden parts; the adapter rebuilds those it
can read off the published columns as the code defines them, a whole-number total as its
published rate times its published denominator, rounded (`rebuilt_team_rates`). A season row
built from published logs therefore matches the published one only on the columns those parts
cover.

## Determinism

The same inputs give the same bits. The tables sum floats over rows, so their order is fixed
rather than left to Polars: the builders sort the team-game and passer-game rows, and the two
profile joins ask Polars to keep their inputs' order, which it leaves unspecified otherwise (a
guard: Polars 2.0 gives the same bits without it). Polars also sums a group's floats in another
order when the frame is flagged sorted by the group key, so `build_season_tables` sorts the game
rows itself and gives every caller the same order and flag. Play order is the order nflverse
publishes, which the play-by-play cache keeps; `play_id` is not monotonic within every game, so
plays are not re-sorted. No linear algebra is involved, so the BLAS thread count does not matter.

## In the ETL

`utils/polars/season_stats_build.py` runs the package inside `nfl-predictor data`:

- It reads no nflverse source twice. A season's inputs are the ones `load_season_inputs` reads,
  cut from the frames the run already holds: the play-by-play the ETL loaded (its games credited
  to one team already repaired from the season rosters), kept to `SEASON_STATS_PBP_COLUMNS` and
  prepared as `prepare_pbp` prepares it; the schedules, weekly player stats, snap counts, players
  file and weekly rosters the ratings loaded. The official weekly team stats are its one source of
  its own, loaded once for the run's seasons under the cache-then-degrade contract of the other
  per-season sources; in the current season they are the file the team-stat loader fetched earlier
  in the run, which nflreadpy serves again from its in-memory cache.
- The ratings build (`utils/polars/ratings_build.py`) builds each season's tables right after the
  season's fit, so `combined` takes `TEAM_RATING_INPUT_COLUMNS` from the season's `ratings` table
  and `qb_combined` takes `QB_RATING_INPUT_COLUMNS` from every passer's row of the quarterback fit
  (`qb_season`), as `nfl-sos-ratings`' `main.run_season` fed them. A season without a qualified
  passer has no quarterback fit, so its `qb_combined` leaves those columns null.
- The tables are written to `data/ratings/{season}_{kind}.parquet` with the ratings tables, each
  file through a temporary file of its own. A season before its first game gets none, and a season
  whose plays exist without official team stats, weekly player stats or snap counts (the current
  season before nflverse publishes them) is built without them; both are logged.
- Every table keeps the columns and types of a full build whatever is missing: the step conforms
  it (`conform_table`) to `utils/polars/season_stats_schema.py`, which
  `.agents/sos_port/season_stats_etl_schema.py` generates from the full builds of 1999-2026. A
  column that a missing source or a season too early for an opponent profile leaves out is null,
  and `qb_game_logs.week` is `Int32` in every season, as published. Rows a missing source supplies
  stay missing: without the snap counts, the passers' games without a dropback. The opponent
  profile tables themselves are absent until a profile can be computed.
- Each slice of a frame that spans seasons is rechunked, so a later season's rows cannot move where
  a sum's chunks split, and two runs, the second fed its games and every other source in another
  row order (each game's plays in nflverse's order), write the same bytes
  (`tests/test_season_stats_build.py`).
- An incremental run keeps a finished season's tables in its season cache entry, with its ratings.
  The tables read only the season's play-by-play, its game rows and its played schedule besides the
  ratings, so the entry's key holds exactly those (`SeasonStatsBuild.key_frames`). The game rows
  are built again on every incremental run for that: they read the players file, which a run in
  season downloads again with new players each week, and only a change to a player the season's
  sources name moves them.
- `.agents/sos_port/season_stats_etl_parity.py` builds every published season through this step
  on the snapshot that project's files were built from, compares what the ETL writes with the
  published files, and checks that the tables written from the season cache's entries equal the
  fresh ones byte for byte (`.agents/sos_port/SEASON_STATS_ETL_PARITY.md`);
  `.agents/sos_port/season_stats_etl_timing.py` times what the step adds to a full and an
  incremental run.

## Checking the port

- `tests/test_season_stats_*.py`: the unit tests, ported from that project's tests with additions,
  and `tests/test_season_stats_published_parity.py`, which builds five games through the loaders
  from nflverse inputs cut from the snapshot that project's files were built from, and two seasons'
  tables from published logs, against fixtures cut from the published files
  (`tests/fixtures/sos_season_stats_parity/`, with their provenance and hashes).
- `.agents/sos_port/season_stats_parity.py`: rebuilds every published season of these tables
  (1999-2026, 2026 through the weeks played at that commit) from that snapshot through the
  loaders, compares each table cell by cell, builds the season tables a second way from the
  published logs, compares the schedule leaderboard with the one that project's own code ranks,
  and writes `.agents/sos_port/SEASON_STATS_PARITY.md` with the counts and timings. The judgment
  calls and departures are in `.agents/sos_port/decisions/season_stats.md`.
