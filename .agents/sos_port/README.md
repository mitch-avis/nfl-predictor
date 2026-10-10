# Absorbing nfl-sos-ratings: the inventory and the play-by-play parity check

Two audit scripts compare `nfl-sos-ratings` at its frozen commit `9081bce` (see
`../sos_port_source.md`) with this repository, for roadmap step 6. Both are report-only: they read
everything read-only and write only their own reports in this directory. Run them from the
repository root with this repository's virtual environment.

| script | writes | judgments |
| --- | --- | --- |
| `inventory.py` | `INVENTORY.md`, `inventory.json` | `mapping.yaml` |
| `parity.py` | `PARITY.md`, `parity.json` | `parity.yaml` |

`FINDINGS.md` lists the proposed changes the reports point to, and `decisions/phase1_parity.md` logs
each difference with the options, a recommended decision and its data.

## Commands

```bash
.venv/bin/python .agents/sos_port/inventory.py
.venv/bin/python .agents/sos_port/parity.py
```

The inventory takes about a minute; the parity check re-aggregates every season once per cause
combination it tries and takes about five minutes. Both exit `0` only when every check holds.

## Inputs and provenance

- `nfl-sos-ratings` code is read at `9081bce` with `git show` and `git ls-tree`, never from its
  working tree. The inventory imports its metric registry (`nfl_sos_ratings.metrics` and
  `rating_ranges`, pure modules) from a copy that `git archive` extracts into a new directory under
  `~/scratch` and removes afterwards. Its pipeline, loaders and command line are never run.
- Its outputs are read in place from `../nfl-sos-ratings/data/`: every `{season}_{kind}.parquet`
  for 1999-2025, plus the two `*_rating_ranges_by_week.parquet` files it writes only for the season
  in progress (2026). They were built on 2026-10-09 from `4d23890`.
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
  Python syntax tree (`inventory_scan.py`), with file, line and function.
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
columns of its kinds that no other rule matches and states how many (`expect`).

## The parity check

`parity.py` aggregates this repository's cached play-by-play with `aggregate_pbp_team_game_stats`
and `aggregate_pbp_team_box_score_stats` for 1999-2025 and compares every team-game with
`{season}_team_game_logs.parquet` on the pairs `parity.yaml` judges to share a definition: counts
must be equal, sums and rates within `1e-6`. Only game-log columns its own code builds from
play-by-play can be judged; the report lists every other column with its source.

A mismatching team-game is attributed to a named cause only when that cause's adjustment
(`parity_rules.py`), alone or with the fewest other causes the pair allows, makes it match exactly.
An adjustment changes one rule in this repository's play-by-play before re-running its own
aggregates (for example its rule that a run play flagged `qb_scramble` is a dropback), or undoes
one repair on its side (Jacksonville's 2001-2002 player teams). The check also matches team-game
keys (re-keying the Rams written `LA`) and checks every played game of the cached schedule against
both sides, apart from the three games nflverse has no play-by-play for.

This repository's play-by-play cache does not carry `qb_scramble`, so the scramble adjustment uses
nflverse's scramble coding (a rush attempt coded as a pass play) as a stand-in. An attribution that
holds only with the stand-in's plays is reported as provisional, and the run fails on it: the
scramble rule can be confirmed only with the flag itself. A difference between nflverse
play-by-play releases can be shown only play by play (same game, same play ids, different
values), and neither input has play ids, so that cause is listed as not evaluable and explains
nothing.

## What makes a run fail

Both scripts exit non-zero, with every reason listed under Failures in their report, when:

- the `nfl-sos-ratings` checkout is not at `9081bce`, has changed tracked files, or its output
  code changed between `4d23890` and `9081bce`;
- an input file is missing, or a search that must find something (an nflreadpy load, an API route,
  a page, a component) finds nothing.

`inventory.py` also fails when a column, source or web item matches no rule or more than one,
when a rule matches nothing, when a `rest` rule takes a different number of columns than it
expects, when a rule's class lacks its counterpart or phase, when a counterpart named for this
repository does not exist, or when a column does not resolve in its registry.

`parity.py` also fails when a mismatching team-game is unexplained or explained only with the
scramble stand-in, when a team-game exists on one side only, when a played game lacks rows or a
listed game without play-by-play has them, when an evidence pattern in `parity.yaml` no longer
holds, when a pair names a column that does not exist or a game-log column that does not come from
play-by-play, or when a cause is described without an implementation or implemented without a
description.
