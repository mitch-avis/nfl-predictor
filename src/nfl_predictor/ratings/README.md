# Ratings package

`nfl_predictor.ratings` holds the team and quarterback ratings of the separate, read-only
`nfl-sos-ratings` project, ported from its frozen commit `9081bce`: its head-to-head-excluded
opponent profiling and the simultaneous ridge that generalizes it. The package reproduces every
ratings file that project published, bit for bit apart from one display column (see "Checking
the port" below).

`nfl-predictor data` fits them on this repository's play-by-play every run, writes their tables
to `data/ratings/`, and adds pre-week rating features to the game dataset as the `ratings`
feature group (see "In the ETL" below). The schedule-adjusted strength snapshot
(`utils/polars/strength_snapshot.py`) stays in the dataset beside them.

## What it computes

- **Team ratings** (`team_rating.py`), in points per game against an average team on a neutral
  field. One weighted ridge fit explains every team-game's scrimmage EPA per play as league
  average plus offense minus the opposing defense plus home field, each row weighted by its plays;
  `offense_rating` and `defense_rating` are the per-play effects times the league's average
  scrimmage plays per team-game. A second fit over special-teams plays rates each team's
  possession and coverage units, which add up to `special_teams_rating`. `team_rating` is the sum.
- **The ridge solver** (`ridge.py`): the penalty is chosen by five-fold cross-validation with whole
  games held out. A season's fit takes the penalties cross-validated on the whole previous season
  (1999, the first season of play-by-play, cross-validates its own).
- **The preseason prior** (`team_prior.py`): early in a season the scrimmage penalty pulls each
  team toward last season's effect times a carryover slope (each season regressed on the one
  before, through the origin, pooled over every earlier pair), faded linearly to nothing by the
  ninth game and centered over the league. The first prior comes in 2003, once three season pairs
  exist.
- **Schedule strength** (`sos`): the mean `team_rating` of a team's opponents, one entry per game,
  each opponent rated by a refit without the team's games, its prior taken from the previous
  season refit without that team.
- **Quarterback ratings** (`qb_rating.py`): `adj_qb_epa_per_dropback` from one ridge of passers
  against the defenses they faced, over EPA per dropback with spikes left out, weighted by
  dropbacks; `qb_faced_pass_defense` from refits without the passer's own dropbacks; and the
  season summary with the rankings qualifier (14 attempts per game of the passer's primary team).
- **Histories, rank ranges, and pairs** (`season.py`, `rating_ranges.py`): the ratings refit on the
  games through each week with the season's penalties; a 1000-resample game bootstrap per season
  (seed 0) for rating and rank percentiles, rank chances, and head-to-head chances; and, for a
  season in progress, each week's rank ranges.
- **SRS**: the score-based reference beside the team rating, from this repository's existing
  solver.

## Inputs

Every function takes Polars frames and returns them; nothing reads files or the network. The fits
consume two kinds of regular-season rows, documented column by column in `inputs.py`:

- team-game rows (`TEAM_GAME_SCHEMA`): `game_id`, `week`, `team`, `opponent_team`, `is_home`,
  `offensive_snaps`, `offensive_epa`, `st_plays`, `st_epa`, `point_margin`;
- quarterback-game rows (`QB_GAME_SCHEMA`): `game_id`, `week`, `team`, `opponent_team`, `is_home`,
  `qb_id`, `qb_name`, `qb_dropbacks`, `qb_dropback_epa`, `qb_attempts`.

`team_game_rows` and `qb_game_rows` check a frame against these and keep its row order, and they
refuse empty ids and team codes. `published_logs.py` adapts the game logs `nfl-sos-ratings`
published, in the order its fits read them, and `pbp_inputs.py` builds the same rows from this
repository's play-by-play and nflverse player sources with that project's definitions:

- team-game rows (`team_games_from_pbp`): a scrimmage snap is a dropback, a run, a kneel-down or a
  spike with both teams known (an empty team code counts as unknown), a run play flagged
  `qb_scramble` counting as a dropback (nflverse leaves most scrambles before 2006 unflagged);
  `offensive_epa` sums the snaps' EPA in the plays' order; special-teams plays (`special`) go to
  the team with the ball; the margin and the home side come from the schedule, whose home team is
  the home team at a neutral site too, as there;
- quarterback rows (`passer_games_from_pbp`, then `quarterback_rows`): a passer's dropbacks (sacks
  included), the pass attempts among them, and their `qb_epa` with his spikes' EPA left out of the
  dropback EPA; the weekly player stats' attempts and passing EPA for a passer they or the
  identity list at quarterback; the snap counts' quarterbacks without a dropback; names from the
  players file, then the weekly roster, then the play-by-play; and the identity rule: a passer
  stays when the players file or a regular-season roster week lists him at quarterback, or when
  no source places him at all. `quarterback_rows(..., before_week=w)` applies the rule with only
  the roster weeks before week `w` and returns only the games before it, sorted by team, week,
  game and passer id, for the pre-week ratings; without `before_week` every roster week counts
  and the rows come in that project's fit order (team, week, game, name), for the season tables.

The exact bits of a fit depend on more than its inputs, so a rebuild that must be byte-identical
fixes all of these:

- Row order: the fits sum over rows in the order given, so an adapter should emit a fixed order
  (team, week, and game; for passers also the name). `pbp_inputs.py` does, and keeps each game's
  plays in nflverse's order, which its EPA sums follow.
- BLAS threads: with one thread, as the tests and the parity audit run and as `nfl-sos-ratings`
  built its files, the quarterback fits are reproducible; with more, they can differ in their last
  bits.
- The BLAS kernel and the library versions: with numpy 2.5.3 (OpenBLAS 0.3.34 on its SkylakeX
  kernel) and Polars 2.0.0 the port reproduces the published files bit for bit. Another kernel
  moves fitted floats in their last bits (a few 1e-14) and can move `SRS`, rounded to six
  decimals, by one step, so the parity tests compare floats within 1e-12 (`SRS` within 1.5e-6)
  and keep ranks, rank and pair chances, and text exact; they pass on the SkylakeX, Sandybridge,
  and Nehalem kernels.

## Outputs

- `season.build_team_season` and `season.build_qb_season` return one season's tables, named and
  ordered like that project's files: `ratings`, `team_prior`, `ratings_by_week`, `rating_ranges`,
  `rating_pairs`, `rating_ranges_by_week`, and the quarterback `qb_season`, `qb_ratings`,
  `qb_ratings_by_week`, `qb_rating_ranges`, `qb_rating_pairs`, `qb_rating_ranges_by_week`.
- `pre_week.RatingsHistory.pre_week(season, week, teams)` is the entry point for model features.
  It fits only the games before the week, with the previous season's cross-validated penalties for
  teams and quarterbacks alike and the preseason prior as of those games, and returns one row per
  team (`TEAM_PRE_WEEK_COLUMNS`) and one per passer with a dropback before the week
  (`QB_PRE_WEEK_COLUMNS`). For the teams that have played, the team ratings equal the published
  history of the week before. Week 1 rates every team at its centered preseason prior on the
  previous season's scale (null before 2003) with `sos` null and no quarterback rows; 1999 has no
  earlier season for its penalties and so returns nulls; a postseason week takes the whole regular
  season. `pre_week_teams` and `pre_week_quarterbacks` return the two parts on their own. Given
  `load_qb_games_before`, the quarterback part reads a season's rows as known before the week
  (who counts as a passer then); the penalties still come from the previous season's whole rows.

The quarterback history `nfl-sos-ratings` published uses the season's own cross-validated penalty,
which depends on games after the week, so it is for display only; the pre-week ratings differ from
it slightly in the seasons whose penalty changed.

## In the ETL

`utils/polars/ratings_build.py` runs the package inside `nfl-predictor data`:

- It loads every season from 1999 through the run's last one (the penalties and the preseason
  prior need every earlier season): the play-by-play and schedules the ETL already loaded, the
  earlier seasons from the cache, and the nflverse player sources under their cache contract.
- For each season of the run it builds the published tables (`build_team_season` and
  `build_qb_season`), and for the current season the weekly rank ranges too, and the pre-week
  team and quarterback ratings of every scheduled week (a postseason week reads the whole regular
  season). A season with no passer qualified for the rankings, such as a season in progress after
  its first game, gets no quarterback tables. `nfl-predictor data` writes the tables to
  `data/ratings/{season}_{kind}.parquet`.
- It attaches the `ratings` feature group (`constants.RATINGS_FEATURE_STATS`) to every game row:
  both teams' pre-week ratings and `sos` for the row's week, and the starter's
  `adj_qb_epa_per_dropback` and dropbacks before the week. The starter is the one every
  quarterback feature reads (`qb_stats.expected_starter_ids`: a completed game's play-by-play
  starter, else the starter the ETL names, the projected starter of an upcoming game), and the
  passers are counted as the identity knew them before the week. What the early weeks hold: week
  1 has each team's preseason prior (from 2003), no `sos`, and no quarterback rating with 0
  dropbacks; `sos` is still null in week 2, since no opponent has played anyone else yet; 1999 is
  null throughout.
- The fits run with one BLAS thread (`threadpoolctl`), the rows reach them in a fixed order, and
  each season's slices are rechunked, so two runs on the same inputs write the same bytes. A
  season's outputs read only it and earlier seasons, so the incremental ETL keeps them in the
  season cache with the season's build (`utils/season_cache.py`, keyed on the ratings' rows and on
  this package's code).

## Checking the port

- `tests/test_ratings_*.py`: the unit tests, ported from that project's tests with additions, and
  `tests/test_ratings_published_parity.py`, which fits small fixtures cut from its published files
  (`tests/fixtures/sos_ratings_parity/`, with their provenance and hashes) and expects the
  published tables exactly, the 1000-resample rank ranges and pairs included.
- `.agents/sos_port/ratings_parity.py`: rebuilds every published season (1999-2025 and the 2026
  season in progress) from that project's game logs, compares each ratings file cell by cell, runs
  the pre-week entry point for every week, and writes `.agents/sos_port/RATINGS_PARITY.md` with the
  counts and timings. The judgment calls and departures are in
  `.agents/sos_port/decisions/ratings.md`.
- `.agents/sos_port/adapter_parity.py`: loads every season of the snapshot of the nflreadpy cache
  that project read through this repository's loaders, compares the adapter's team-game and
  quarterback-game rows with its published game logs, rebuilds every season's tables from them,
  and writes `.agents/sos_port/ADAPTER_PARITY.md`; the integration's judgment calls are in
  `.agents/sos_port/decisions/integration_ratings.md`.
- `tests/test_ratings_pbp_inputs.py`, `tests/test_ratings_build.py` and the ratings cases of
  `tests/test_data_collection_incremental.py`: the adapter's definitions, the build and its
  features, two runs that write identical bytes, and the incremental ETL. The leakage cases
  rewrite every input from a week on and check that the week's features do not move: in the
  build (weeks 1, 3, 4 and 6), the plays' EPA, passer ids and names and dropback, pass, run,
  kneel, spike and special-teams flags, the weekly player stats, the snap counts, the weekly
  rosters and the scores; through the ETL (week 3, the first week with every rating feature),
  the plays and the scores. A rewrite of the week before does move them.
