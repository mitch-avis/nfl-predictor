# Absorbing nfl-sos-ratings: the inventory and the play-by-play parity check

Two audit scripts compare `nfl-sos-ratings` at its frozen commit `9081bce` (see
`../sos_port_source.md`) with this repository, for roadmap step 6. Both are report-only: they read
everything read-only and write only their own reports in this directory. Run them from the
repository root with this repository's virtual environment.

| script | writes | judgments |
| --- | --- | --- |
| `inventory.py` | `INVENTORY.md`, `inventory.json` | `mapping.yaml` |
| `parity.py` | `PARITY.md`, `parity.json`, `parity_attributions.parquet` | `parity.yaml` |

`decisions/phase1_parity.md` logs each difference the first reports found, with the options, the
implementer's recommended decision and its data, and `decisions/phase1_refresh.md` how the
judgments were refreshed for the integrated code. `DECISIONS.md` records the decision taken on
each, and `FINDINGS.md` the short list of changes; the delegating session writes both.

The checked-in reports compare `nfl-sos-ratings` with this repository's code on the step-6
branch (`feat/step6-sos-port` at `9bba627`, `0.56.0`, with these scripts refreshed) and the
candidate build `~/scratch/step6/data/` (built at `0.55.0`: the extended play-by-play cache, the
ratings tables and `all_data_ml.csv`; the ETL writes the season-statistics tables only from
`0.56.0`, so the inventory reads their columns from the code). They show the port's progress:
every output column of `nfl-sos-ratings` is now written by this repository's ETL, and the classes
say where this repository's own model inputs still define a column differently. The main
checkout's `data/` predates the integration, so a run on it stops: the play-by-play cache lacks
columns the code now reads, and `data/ratings/` does not exist yet.

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/inventory.py --data-dir ~/scratch/step6/data
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/parity.py --data-dir ~/scratch/step6/data
```

`--data-dir` names the data directory of this repository to read; without it both scripts read
the main checkout's `data/`. Each report records the directory and the sha256 of every file it
read there. `PYTHONDONTWRITEBYTECODE=1` keeps the runs from leaving `__pycache__` beside the
scripts. The inventory takes a few seconds; the parity check re-aggregates every season once per
cause combination it tries and takes about a minute and a half. Both exit `0` only when every
check holds, and both reproduce their reports byte for byte.

## Report sizes

The repository's `check-added-large-files` hook rejects a file over 500 KB. A text table of every
mismatching team-game would exceed it, so the parity check writes that table as
`parity_attributions.parquet`, which `.gitignore` re-includes past its `*.parquet` rule.
`inventory.json` stays under the limit (about 460 KiB) because it stores the column table as rows
of fields per file kind (a column's seasons are its kind's unless `partial_seasons` lists it), the
registry's metrics as rows of fields, and the constants catalog and the ported tables' columns as
counts.

## Inputs and provenance

- `nfl-sos-ratings` code is read at `9081bce` with `git show` and `git ls-tree`, never from its
  working tree, and every git call passes `--no-optional-locks` so it writes nothing there. The
  inventory imports its metric registry (`nfl_sos_ratings.metrics` and `rating_ranges`, pure
  modules) from a copy that `git archive` extracts into a new directory under `~/scratch` and
  removes afterwards, even when the run fails. Its pipeline, loaders and command line are never
  run.
- Its outputs are read in place from `../nfl-sos-ratings/data/`: every `{season}_{kind}.parquet`
  for 1999-2025, plus the two `*_rating_ranges_by_week.parquet` files it writes only for the season
  in progress (2026). They were built on 2026-10-09 from `4d23890`.
- The play-by-play it read comes from a read-only snapshot of nflreadpy's on-disk cache, taken
  after that build: `~/scratch/nflreadpy_cache_snapshot_20261008/`, with the manifest
  `~/scratch/nflreadpy_cache_snapshot_20261008.sha256`. The parity check finds each season's file
  by its content (the play-by-play columns and a single `season` value) and checks it against the
  manifest.
- Both scripts fail unless that checkout's `HEAD` is `9081bce`, its tracked files are unchanged,
  and `git diff 4d23890 9081bce` is empty for `nfl_sos_ratings/`, `pyproject.toml` and `uv.lock`
  (the build commit's output code is the frozen commit's).
- This repository's data is read, read-only, from the `--data-dir` directory, by default the main
  checkout's `data/` (the first entry of `git worktree list`): the play-by-play cache
  `cache/nflreadpy/pbp_{season}_reg.parquet`, read through this repository's own loader, with the
  season rosters (`rosters_{season}.parquet`) its Jacksonville repair reads and the schedule
  cache; the cached weekly team stats schema; the header of `all_data_ml.csv`; and the ratings
  tables `ratings/{season}_{kind}.parquet` of the last completed season (2025) and, for the two
  rank-range-by-week kinds, of the season in progress (2026). The parity check blocks every
  nflreadpy download, so a cache miss fails the run instead of writing to the data directory.
  This repository's code is the checkout that runs the script.
- Every file read is listed with its sha256 under `inputs` in each JSON report, and the data
  directory under `data_dir`.

## The inventory

`inventory.py` finds every item itself and classifies it with `mapping.yaml`:

- **Data sources**: nflreadpy `load_*` calls (through the names bound to `nflreadpy`), network
  fetches, web addresses and data files named under the data directory, from each repository's
  Python syntax tree (`inventory_scan.py`), with file, line and function. The report states what
  the scan does not cover.
- **Output columns**: every column of every output file kind, unioned over the seasons read and
  resolved through its own metric registry to a base metric and its prefix and suffix rules. Each
  column is `matched` (same definition here, named), `different` (a counterpart here with a
  different definition, said how) or `missing` (the step-6 phase that ports it, or none planned).
  For the team game logs, where each column comes from (play-by-play, nflverse's official weekly
  team stats, the schedule, weekly player stats) is read from its loader's code
  (`sos_sources.py`).
- **Metric registry**: every entry, read from the registry object.
- **Ratings**: its rating files against this repository's strength snapshot, pre-week rating
  features and quarterback columns.
- **Web**: its API routes (route decorators in `ui_api.py`), page routes (`web/src/router.tsx`) and
  page and detail-page components, against this app's (its pages, the ratings pages' components
  in `web/src/components/ratings/` and the shared table in `web/src/components/table/`).

A column's counterparts are of two kinds. `ratings:<column>` is the same column of the table of
the same kind that this repository's ETL writes to `data/ratings/{season}_{kind}.parquet` (the
ported ratings and season statistics; `port_catalog` reads the ratings kinds' columns from the
data directory and the season-statistics kinds' from `SEASON_STATS_SCHEMAS`, the schema the ETL
conforms them to). The inventory names it first for every matched or different column the port
writes. The others are this repository's own model inputs: its play-by-play aggregates, cached
team stats, quarterback games, game dataset and strength snapshot. The ported tables equal the
published files cell for cell, Washington's code apart (`SEASON_STATS_ETL_PARITY.md`,
`RATINGS_PARITY.md`), so a column is `matched` only when every counterpart named shares its
definition, and `different` when one of this repository's own inputs defines it otherwise. A
counterpart on the opponent's row of the same game marks its catalog `[opponent]`
(`box_score[opponent]:rush_attempts`); only `team_game`, `box_score` and `team_stats`, whose rows
carry a team and its opponent, take the marker, and the inventory fails it on any other catalog.

Rules match columns on file kind, name, registry base metric and affixes. A `rest` rule takes the
columns of its kinds that no other rule matches and states how many (`expect`); any rule may pin
its count the same way. A mirror rule (`mirror_kind`) gives each of its columns the counterparts
the same column has in another file kind (with `mirror_base`, its base metric's column there),
less that column's ported table, so a season value names what its game-log column names. Source
and web rules may name an item's own key as `{key}`.

Guards then check every item left `missing`. The port guard fails a column the ported tables
write under its kind and name. The same-name guard fails one with a same-name counterpart in this
repository's other catalogs (its aggregates' columns, the cached team stats, the quarterback
games, the game dataset, the strength snapshot and every list of names in `constants.py`), or a
season total or per-game value whose base metric has one. The cross-kind guard fails one whose
name, or a total's or per-game value's base metric, is `matched` or `different` in another file
kind, unless `cross_kind_exceptions` in `mapping.yaml` says why. Opponents-faced profile columns
are exempt from the last two, because a same-name column here describes the team itself. A source
or web item left `missing` fails when this repository has an item of the same key.

## The parity check

`parity.py` reads this repository's cached play-by-play for 1999-2025 the way its ETL does (a
cache hit through `loaders.load_pbp`, then the Jacksonville 2001-2002 player-team repair,
`nflverse_sources.repair_one_team_pbp_games`), aggregates it with `aggregate_pbp_team_game_stats`
and `aggregate_pbp_team_box_score_stats`, and compares every team-game with
`{season}_team_game_logs.parquet` on the pairs `parity.yaml` judges to share a definition: counts
must be equal, sums and rates within `1e-6`. Only game-log columns its own code builds from
play-by-play can be judged; the report lists every other column with its source. A pair with
`row: opponent` compares this repository's value for a team with the game-log value on its
opponent's row of the same game (`carries_faced`, `attempts_faced` and `completions_allowed`, what
the defense faced); a team-game whose opponent's row is missing fails the check. Both sides are
keyed in canonical team codes, so a team-game one side writes with the Rams as `LA` is compared
like any other.

For each season the check prepares the snapshot's file with this repository's own `_prepare_pbp`,
which keeps the rows in order, and requires it to equal the cache row by row on every cached
column, `play_id`, `qb_scramble` and the play text `desc` included. That comparison is the
play-by-play test of a difference between nflverse releases: with the snapshot equal to the cache,
no team-game can owe a difference to the release.

A mismatching team-game is attributed to a named cause only when that cause's adjustment
(`parity_rules.py`), alone or with the fewest other causes the pair allows, makes it match exactly.
An adjustment changes one rule in this repository's play-by-play before re-running its own
aggregates (for example its rule that a two-point try is neither a dropback nor a carry); a
combination applies them in one fixed order (`adjust`). The design also allows a cause that undoes
a repair on the other side, though none does today. For each cause the report gives both what it
explains and its own effect: its adjustment applied alone to every team-game. Every mismatching
team-game, with its values and causes, is in `parity_attributions.parquet`. Beside the cause for
runs a penalty wiped out, the report counts the plays nflverse records as `no_play` by the flags
they keep.

## What makes a run fail

Both scripts exit non-zero, with every reason listed under Failures in their report, when:

- the `nfl-sos-ratings` checkout is not at `9081bce`, has changed tracked files, or its output
  code changed between `4d23890` and `9081bce`;
- an input file is missing (a ratings table of the data directory included), or a search that
  must find something (an nflreadpy load, an API route, a page, a component) finds nothing.

`inventory.py` also fails when a column, source or web item matches no rule or more than one,
when a rule matches nothing, when a rule takes a different number of columns than it expects,
when a rule's class lacks its counterpart or phase, when a counterpart named for this repository
does not exist, when a mirror rule mirrors a column that has no counterpart, is a mirror itself or
has no counterpart but its ported table, when a guard finds an item left `missing` that has a
counterpart, when a cross-kind exception is not needed, or when a column does not resolve in its
registry.

`parity.py` also fails when a mismatching team-game is unexplained, when a team-game exists on
one side only, when a played game lacks rows or a listed game without play-by-play has them, when
a season's cached schedule is empty, when a season's cache file is missing or older than the
columns the code reads, when a season roster the repair needs is not cached, when a season's
snapshot file is missing, ambiguous, not in its manifest or not equal to the cache row by row,
when an evidence pattern in `parity.yaml` no longer holds, when a pair names an unknown kind or
row, a column that does not exist or a game-log column that does not come from play-by-play, or
when a cause is described without an implementation or implemented without a description.
