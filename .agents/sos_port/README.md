# Absorbing nfl-sos-ratings: the inventory and the play-by-play parity check

Two audit scripts compare `nfl-sos-ratings` at its frozen commit `9081bce` (see
`../sos_port_source.md`) with this repository, for roadmap step 6. Both are report-only: they read
everything read-only and write only their own reports in this directory. Run them from the
repository root with this repository's virtual environment.

| script | writes | judgments |
| --- | --- | --- |
| `inventory.py` | `INVENTORY.md`, `inventory.json` | `mapping.yaml` |
| `parity.py` | `PARITY.md`, `parity.json`, `parity_attributions.parquet` | `parity.yaml` |

`decisions/phase1_parity.md` logs each difference the reports found, with the options, the
implementer's recommended decision and its data. `DECISIONS.md` records the decision taken on each
(it is on the step-6 integration branch until that branch merges), and `FINDINGS.md` the short
list of proposed changes; the delegating session writes both.

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/inventory.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/parity.py
```

`PYTHONDONTWRITEBYTECODE=1` keeps the runs from leaving `__pycache__` beside the scripts. The
inventory takes a few seconds; the parity check re-aggregates every season once per cause
combination it tries and takes about six minutes. Both exit `0` only when every check holds.

## Report sizes

The repository's `check-added-large-files` hook rejects a file over 500 KB. A text table of every
mismatching team-game would exceed it, so the parity check writes that table as
`parity_attributions.parquet`, which `.gitignore` re-includes past its `*.parquet` rule.
`inventory.json` is close to the limit (about 490 KiB), although it already stores each column as
a row of fields and the constants catalog as a count.

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
- This repository's data is read from the main checkout (the first entry of `git worktree list`),
  read-only: the play-by-play cache `data/cache/nflreadpy/pbp_{season}_reg.parquet` and schedule
  cache, the cached weekly team stats schema, and the header of `data/all_data_ml.csv`. This
  repository's code is the checkout that runs the script.
- Every file read is listed with its sha256 under `inputs` in each JSON report.

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
- **Ratings**: its rating files against this repository's strength snapshot and quarterback
  columns.
- **Web**: its API routes (route decorators in `ui_api.py`), page routes (`web/src/router.tsx`) and
  page and detail-page components, against this app's.

Rules match columns on file kind, name, registry base metric and affixes. A `rest` rule takes the
columns of its kinds that no other rule matches and states how many (`expect`); any rule may pin
its count the same way. A mirror rule (`mirror_kind`) gives each of its columns the counterparts
the same column has in another file kind (with `mirror_base`, its base metric's column there), so
a season value names what its game-log column names.

Two guards then check every column left `missing`. The same-name guard fails one with a same-name
counterpart in this repository's catalogs (its aggregates' columns, the cached team stats, the
quarterback games, the game dataset, the strength snapshot and every list of names in
`constants.py`), or a season total or per-game value whose base metric has one. The cross-kind
guard fails one whose name, or a total's or per-game value's base metric, is `matched` or
`different` in another file kind, unless `cross_kind_exceptions` in `mapping.yaml` says why.
Opponents-faced profile columns are exempt from both, because a same-name column here describes
the team itself.

## The parity check

`parity.py` aggregates this repository's cached play-by-play with `aggregate_pbp_team_game_stats`
and `aggregate_pbp_team_box_score_stats` for 1999-2025 and compares every team-game with
`{season}_team_game_logs.parquet` on the pairs `parity.yaml` judges to share a definition: counts
must be equal, sums and rates within `1e-6`. Only game-log columns its own code builds from
play-by-play can be judged; the report lists every other column with its source. Both sides are
keyed in canonical team codes, so a team-game one side writes with the Rams as `LA` is compared
like any other.

This repository's cache leaves out `qb_scramble`, `play_id` and the play text `desc`. For each
season the check prepares the snapshot's file with this repository's own `_prepare_pbp`, which
keeps the rows in order, and requires it to equal the cache row by row on every cached column;
only then are the three columns attached by row position. Rows equal on every cached column cannot
be told apart that way, so the run also requires each group of such rows to carry one
`qb_scramble` value, and each `play_id` to appear once within its game. That comparison is also
the play-by-play test of a difference between nflverse releases: with the snapshot equal to the
cache, no team-game can owe a difference to the release.

A mismatching team-game is attributed to a named cause only when that cause's adjustment
(`parity_rules.py`), alone or with the fewest other causes the pair allows, makes it match exactly.
An adjustment changes one rule in this repository's play-by-play before re-running its own
aggregates (for example its rule that a run play flagged `qb_scramble` is a dropback), or undoes
one repair on its side (Jacksonville's 2001-2002 player teams). For each cause the report gives both
what it explains and its own effect: its adjustment applied alone to every team-game. Every
mismatching team-game, with its values and causes, is in `parity_attributions.parquet`. For the
carry decision, the report also counts the plays nflverse records as `no_play` by the flags they
keep (and, for those without the penalty flag, by what their play text says), this repository's
carries without the `rush_attempt` flag by play type, and how carries built from `rush_attempt`,
with and without its `~qb_scramble` term, differ from this repository's carries.

## What makes a run fail

Both scripts exit non-zero, with every reason listed under Failures in their report, when:

- the `nfl-sos-ratings` checkout is not at `9081bce`, has changed tracked files, or its output
  code changed between `4d23890` and `9081bce`;
- an input file is missing, or a search that must find something (an nflreadpy load, an API route,
  a page, a component) finds nothing.

`inventory.py` also fails when a column, source or web item matches no rule or more than one,
when a rule matches nothing, when a rule takes a different number of columns than it expects,
when a rule's class lacks its counterpart or phase, when a counterpart named for this repository
does not exist, when a mirror rule mirrors a column that has no counterpart or is a mirror itself,
when either guard finds a column left `missing` that has a counterpart, when a cross-kind
exception is not needed, or when a column does not resolve in its registry.

`parity.py` also fails when a mismatching team-game is unexplained, when a team-game exists on
one side only, when a played game lacks rows or a listed game without play-by-play has them, when
a season's snapshot file is missing, ambiguous, not in its manifest or not equal to the cache row
by row, when a `play_id` repeats within a game or rows equal on every cached column differ in
`qb_scramble`, when an evidence pattern in `parity.yaml` no longer holds, when a pair names a
column that does not exist or a game-log column that does not come from play-by-play, or when a
cause is described without an implementation or implemented without a description.
