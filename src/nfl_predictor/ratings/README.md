# Ratings package

`nfl_predictor.ratings` holds the team and quarterback ratings of the separate, read-only
`nfl-sos-ratings` project, ported from its frozen commit `9081bce`: its head-to-head-excluded
opponent profiling and the simultaneous ridge that generalizes it. The package reproduces every
ratings file that project published, bit for bit apart from one display column (see "Checking
the port" below).

Nothing in the ETL, the game dataset, the model, or the weekly run uses the package, and no
default depends on it: the model's strength features come from the schedule-adjusted strength
snapshot (`utils/polars/strength_snapshot.py`).

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

`team_game_rows` and `qb_game_rows` check a frame against these and keep its row order. The fits
sum over rows in the order given, so an adapter should emit a fixed order (team, week, and game;
for passers also the name) to make rebuilds byte-identical. The BLAS thread count matters the
same way: with one thread, as the tests and the parity audit run and as `nfl-sos-ratings` built
its files, the port reproduces those files bit for bit; with more, the quarterback fits can differ
in their last bits. `published_logs.py` adapts the game logs `nfl-sos-ratings` published, in the
order its fits read them.

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
  season. `pre_week_teams` and `pre_week_quarterbacks` return the two parts on their own.

The quarterback history `nfl-sos-ratings` published uses the season's own cross-validated penalty,
which depends on games after the week, so it is for display only; the pre-week ratings differ from
it slightly in the seasons whose penalty changed.

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
