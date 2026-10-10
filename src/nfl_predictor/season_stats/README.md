# Season statistics package

`nfl_predictor.season_stats` holds the season statistics of the separate, read-only
`nfl-sos-ratings` project, ported from its frozen commit `9081bce`: the per-season tables its web
pages show beside the ratings, and the metric registry that names and describes every column. The
port reproduces the tables that project published, cell by cell (see "Checking the port" below).

Nothing in the ETL, the game dataset, the model, the weekly run, or the web API uses the package
yet, and no default depends on it.

The tables are whole-season display data, not values known before a week's games: a season row
pools every game played so far, and an opponent profile pools every played game of each
opponent against everyone else, later weeks included. A model feature built from them needs its
own pre-week version, solved from the games strictly before each week, as `AGENTS.md` requires
of every schedule- or opponent-adjusted value.

When an ETL step starts reading the package, add `season_stats` to `_ETL_SOURCE_DIRECTORIES` in
`utils/season_cache.py` in the same change: the season cache's code fingerprint covers
`data_collection.py`, `constants.py`, and `utils/` only, so a change here would otherwise leave
cached seasons built from the old code.

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
