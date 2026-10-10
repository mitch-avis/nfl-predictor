# Decisions: the nflverse player sources, the models directory and `compare` ties

Absorbing `nfl-sos-ratings` (roadmap step 6), the nflverse-sources phase (phase 2; stream S2 in
`.agents/step6_plan.md`) with the two items batched into it: the models-directory constant and the
`compare` tie rule. Written by the phase's implementer on 2026-10-10 on the branch
`worktree-agent-ac0e0051f63130c11` (from `feat/step6-sos-port` at `3ab0bd1`). The source is
`nfl-sos-ratings` at its frozen commit `9081bce`. Review: not yet reviewed.

Two scripts beside this file reproduce the data, and neither reads or writes `data/` beyond the
play-by-play cache the first one reads:

```bash
# The sources themselves (live nflverse downloads; nflreadpy's day cache under /tmp):
NFLREADPY_CACHE=filesystem NFLREADPY_CACHE_DIR=/tmp/nflreadpy_cache \
    .venv/bin/python .agents/sos_port/decisions/sources_data.py \
    --pbp-cache /home/mitch/workspace/nfl-predictor/data/cache/nflreadpy
# The new loaders against the live files (their cache in a temporary directory):
.venv/bin/python .agents/sos_port/decisions/sources_live_check.py
```

Both ran on 2026-10-10 (08:21 and 08:42 UTC) with nflreadpy `0.1.5` and Polars `2.0.0`. nflverse
revises its files, so a later run can differ in the last rows.

## The nflverse player sources

### Team codes follow this repository's canonical codes

- **What differs**: `nfl-sos-ratings` normalizes team codes with `config.TEAM_ABBR_ALIASES`
  (`JAC`, `LA`, `OAK`, `SD`, `STL`, and `WSH` to `WAS`), so Washington is `WAS` there, and its
  `data_loader._normalize_team_abbreviations` turns an empty code into null. Here every team
  column goes through `constants.ALIAS_TO_CANONICAL` (`normalize_team_column` in
  `utils/scraping_utils.py`), which makes Washington `WSH`, with no empty-code rule. The port also
  normalizes the snap counts' `opponent` column, which the source leaves as published.
- **Options**: this repository's codes; or the source's, which would put `WAS` beside `WSH` in
  this repository's tables.
- **Decision**: this repository's canonical codes.
- **Reasoning**: every table here joins on `ALIAS_TO_CANONICAL` codes (`AGENTS.md`, "Column &
  Schema Rules"). The empty-code rule protects play-by-play `posteam` on non-plays; none of these
  sources has an empty code.
- **Data** (`sources_data.py`, 1999-2025, QBR 2006-2026; the codes the mapping renames or does not
  know, by source and row count):

  ```text
  QBR season           OAK -> LV 24, SD -> LAC 24, STL -> LAR 16
  QBR week             OAK -> LV 454, SD -> LAC 366, STL -> LAR 204
  season rosters       ARZ 896, BLT 903, CLV 942, HST 935, SL 931 -> unknown;
                       LA -> LAR 902, STL -> LAR 198, OAK -> LV 1463, SD -> LAC 1226,
                       WAS -> WSH 2145
  snap counts          LA -> LAR 16566, OAK -> LV 10126, SD -> LAC 5864, STL -> LAR 4298,
                       WAS -> WSH 19841
  weekly player stats  LA -> LAR 30036, WAS -> WSH 29161, null 7
  weekly rosters       ARZ 13467, BLT 14153, CLV 13388, HST 13499, SL 13502 -> unknown;
                       LA -> LAR 14605, OAK -> LV 18708, SD -> LAC 14920, WAS -> WSH 27988
  empty or null codes  weekly player stats: team null 1, opponent_team null 6; no empty string
  ```

- **Feature-value impact**: none; the ETL reads none of these sources.
- **How to reverse it**: map through another table in `SeasonSource.prepare`
  (`src/nfl_predictor/utils/polars/nflverse_sources.py`).

### The league team codes of the 2002-2015 rosters

- **What differs**: the source maps the league (GSIS) codes `ARZ`, `BLT`, `CLV`, `HST` and `SL` in
  `data_loader._ROSTER_TEAM_CODES`, described as "such as 2002's", for the season rosters its
  Jacksonville repair reads. The data shows them in every season roster and weekly roster from
  2002 through 2015. Here they are `constants.NFLVERSE_ROSTER_TEAM_CODES` (to nflverse's `ARI`,
  `BAL`, `CLE`, `HOU`, `STL`), applied to season and weekly rosters before the canonical codes.
- **Options**: a roster-only map; or the five codes as aliases in `constants.TEAM_MAPPING`, so
  every source would map them.
- **Decision**: a roster-only map, applied to both roster sources.
- **Reasoning**: no other source uses these codes (the table above), and the global alias table is
  read by every ETL source and the TeamRankings lookups; widening it gains nothing. Mapping the
  weekly rosters too means a later reader of their `team` column needs no second rule.
- **Data**: `sources_data.py`, "Seasons whose season or weekly rosters use league (GSIS) team
  codes": 2002 through 2015 in both, 313-349 season-roster and 4,567-5,082 weekly-roster rows a
  season. `sources_live_check.py`: the 2002 and 2016 season rosters load with the 32 canonical
  codes.
- **Feature-value impact**: none.
- **How to reverse it**: move the five codes into `constants.TEAM_MAPPING` and drop `team_codes`
  from `ROSTERS` and `WEEKLY_ROSTERS` in `nflverse_sources.py`.

### Jacksonville 2001-2002: canonical sides, and only player stats repaired for now

- **What differs**:
  - The source's `player_team_repair._game_sides` reads the two teams from the game id and compares
    them, as written, with normalized roster codes; that works there because its Washington code is
    nflverse's `WAS`. Here `player_team_repair.game_sides` and the team read from a penalty's play
    text go through `ALIAS_TO_CANONICAL`.
  - `game_player_teams` here takes rosters of several seasons, each game reading its own season's
    roster; the source passes one season.
  - The source repairs play-by-play and weekly player stats and drops one-team games from the
    weekly team stats (`two_team_games_only`). Here the play-by-play repair is ported and tested
    (`repair_pbp_player_teams`) but `loaders.load_pbp` does not call it; weekly player stats are
    repaired in `load_weekly_player_stats`, after the cache. `two_team_games_only` is not ported:
    `loaders.attach_team_stats_to_schedule` already nulls the box score of a team-stat row that
    covers both teams.
- **Options**: compare the sides as written or canonical; wire the play-by-play repair into
  `load_pbp` now, or with the phase-3 feature-value changes.
- **Decision**: canonical sides; play-by-play repair ported but not wired; weekly player stats
  repaired.
- **Reasoning**: sides as written leave 35 more rows without a team here, all in
  `2002_10_WAS_JAX`. Wiring the play-by-play repair needs the player-id columns in
  `constants.PBP_COLUMNS` (a refetch of every cached season) and changes `td_team` and
  `penalty_team` in the 16 games, which feed this repository's red-zone touchdown and penalty
  aggregates (`utils/polars/pbp.py`): a feature-value change, which this phase may not make.
- **Data**:

  ```text
  sources_data.py   16 games (8 in 2001, 8 in 2002, each a Jacksonville home game), 1,027 rows
    sides normalized:             1,025 placed on one side, 1 on both rosters, 2 left unknown
    sides as written in game_id:    991 placed, 36 left unknown; 2002_10_WAS_JAX 33 of 68 placed
  sources_live_check.py (the loader itself): the same 16 games, 2 rows without a team
    (2001_06_BUF_JAX and 2002_10_WAS_JAX, 1 each), 0 games naming one team after the repair
  ```

- **Feature-value impact**: none now. Wiring the play-by-play repair later changes `td_team` and
  `penalty_team` in those 16 games, so the red-zone and penalty columns of Jacksonville and its
  visitors in those weeks.
- **How to reverse it**: drop the repair from `load_weekly_player_stats`; read the sides as
  written by removing `constants.normalize_team_abbr` from `player_team_repair.game_sides`.

### Stats nflverse never charted are null, after the cache

- **What differs**: the rule is the source's (`data_loader._PLAYER_STAT_GAPS`: tackles for loss
  2003-2011, quarterback hits 2003-2005), here `constants.NFLREADPY_PLAYER_STAT_GAPS`. The source
  has no cache of its own and blanks at load; here `blank_player_stat_gaps` runs after the cache,
  which keeps nflverse's zeros.
- **Options**: blank before writing the cache, or after reading it.
- **Decision**: after.
- **Reasoning**: the cache's staleness test checks columns only, so a later change to the gap list
  would not reach files written under the old one.
- **Data** (`sources_data.py`, regular-season totals): tackles for loss 1,878-2,202 a season in
  1999-2002, 0 in 2003-2005, 1 in 2006, 0 in 2007-2011, 2,221 in 2012; quarterback hits 1,056-1,154
  in 1999-2002, 0 in 2003-2005, 2,055 or more from 2006.
- **Feature-value impact**: none.
- **How to reverse it**: edit or empty `constants.NFLREADPY_PLAYER_STAT_GAPS`.

### The cache contract, the requested columns and the first seasons

- **What differs**: the source loads whole files through nflreadpy (its one-day filesystem cache),
  always the regular season. Here each source is cached per season like play-by-play: the columns
  listed in `constants` (weekly player stats 34 of 150, snap counts 11 of 16, players 5 of 39,
  rosters 8 of 36), every frame with every requested column and a fixed type (null where a
  season's file lacks one), the regular season by default with `_reg` or `_all` in the file name,
  and no request below a source's first season (player stats 1999, snap counts 2012, season
  rosters 1920, weekly rosters 2002, QBR 2006).
- **Options**: whole files or column lists; first seasons in code or a failed request.
- **Decision**: column lists and first seasons in `constants`.
- **Reasoning**: the play-by-play loader's pattern (`AGENTS.md`: column lists are constants). Under
  the cache contract a failed historical season stops the run, and nflreadpy refuses a season
  below its range with a `ValueError`. A longer column list later costs one download per season,
  since each file records the columns it was written for.
- **Data**: `sources_data.py`, a season after the current one: player stats `ConnectionError`
  (404), snap counts `ValueError` "Season must be between 2012 and 2026", weekly rosters
  `ValueError` "... 2002 and 2026", season rosters `ValueError` "... 1920 and 2026". The 2012 snap
  counts file has no rows (2013 has 22,821 regular-season rows). `sources_live_check.py`: 2026
  loads for every source; 2027 is skipped with a warning by each.
- **Feature-value impact**: none.
- **How to reverse it**: the lists and `*_MIN_SEASON` constants in `constants.py`.

### Quarterback identity: the players file and the weekly roster, as the source

- **What differs**: the rule is the source's (`load_qb_identity_crosswalk`), with its column names
  (`qb_id`, `snap_player_id`, `qb_name`, `qb_position`) so the quarterback stats ported next read
  it unchanged. The first name and Pro Football Reference id are chosen deterministically here
  (stable sort, ordered group); the source's sort is not stable, so between two roster rows of one
  player its pick could vary. The judgment call: nflverse has no weekly roster before 2002.
- **Options**: the players file alone before 2002, as the source; or season rosters standing in
  for the missing weekly rosters.
- **Decision**: as the source.
- **Reasoning**: season rosters would mark no additional passer in any season.
- **Data** (`sources_data.py`, regular-season play-by-play passers by where a source lists them at
  quarterback): season-roster-only passers 0 in every season 1999-2025; passers only the weekly
  roster lists at quarterback 0 a season through 2011, 0-2 a season from 2012 (2020, 2021 and 2023
  have 2). `sources_live_check.py`: in 2022 Taysom Hill (`00-0033357`, listed `TE` in the players
  file) is `QB`.
- **Feature-value impact**: none.
- **How to reverse it**: pass season rosters to `player_identity.qb_identity_crosswalk` as a third
  source.

### The players file: one cached file, refreshed for the current season

- **What differs**: the source downloads the players file for every identity load. Here
  `load_players` caches `players.parquet`; an earlier season's identity reads it (a failed first
  download is raised), and the current season's downloads it again and, on failure, falls back
  to the cached copy or to no players file.
- **Options**: refresh by age (nflreadpy's one day), on every load, or by the season rule the
  other sources follow.
- **Decision**: the season rule.
- **Reasoning**: one contract for every source, and deterministic for finished seasons. Without a
  players file the weekly roster still marks the season's quarterbacks.
- **Data**: none needed beyond the tests (`tests/test_nflverse_sources.py`).
- **Feature-value impact**: none.
- **How to reverse it**: the `load_players` arguments in `load_qb_identity`.

### ESPN QBR: validation only, one download split into per-season files

- **What differs**: the source downloads the asset with `urllib` (60 s timeout), keeps
  `"Regular"`, normalizes `team_abb` and `opp_abb` with its aliases (Washington to `WAS`) and joins
  QBR to its quarterbacks by season, team and normalized name. Here `espn_qbr.load_espn_qbr`
  downloads with `requests` (60 s, the repository's nfelo pattern), reports failures as
  `ConnectionError` or `ValueError` as nflreadpy does, caches each season from the one download
  under the same contract, keeps only the regular season, leaves Washington `WSH`, and keeps ESPN's
  `player_id` while the players file keeps `espn_id`, so a later check can join by id. A test runs
  the dataset, model, weekly-run and command-line modules in a fresh interpreter and requires that
  none imports it.
- **Options**: keep a postseason option (dropped: the source never uses one); a name join (left to
  the chunk that ports the validation).
- **Decision**: as described.
- **Reasoning**: the task makes QBR a validation reference only; the import test enforces it.
- **Data**: `sources_live_check.py`: season level 2006 56 rows and 2007 57 (2005 skipped, no
  request), week level 2016 512 rows with the 32 canonical codes, 2026 51 rows.
- **Feature-value impact**: none, by construction.
- **How to reverse it**: not applicable.
