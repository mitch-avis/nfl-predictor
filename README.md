# nfl-predictor

NFL game score and outcome prediction with a margin/total ML model, calibrated win probabilities,
and tooling for Pick 'Em and Confidence Pools.

This repo is geared toward:

- **Pick 'Em** and **Confidence Pools** (primary)
- sports-betting research (secondary; no profit claims)

## Table of Contents

- [nfl-predictor](#nfl-predictor)
  - [Table of Contents](#table-of-contents)
  - [What this project produces](#what-this-project-produces)
  - [Setup](#setup)
    - [Python environment](#python-environment)
      - [Recommended: uv project workflow](#recommended-uv-project-workflow)
      - [Alternative: create the venv manually](#alternative-create-the-venv-manually)
    - [Run tests](#run-tests)
  - [Data collection (Polars + nflreadpy)](#data-collection-polars--nflreadpy)
  - [Data sources + missing data](#data-sources--missing-data)
  - [Training + prediction](#training--prediction)
    - [Quickstart (train + predict)](#quickstart-train--predict)
    - [Splits: train and holdout](#splits-train-and-holdout)
  - [Modeling approach](#modeling-approach)
    - [Margin/Total targets (canonical)](#margintotal-targets-canonical)
    - [Win probability](#win-probability)
    - [Market integration (optional, recommended)](#market-integration-optional-recommended)
  - [Backtesting](#backtesting)
  - [Weekly workflow (canonical)](#weekly-workflow-canonical)
    - [High-level stages](#high-level-stages)
    - [How postseason games enter today](#how-postseason-games-enter-today)
    - [Authoritative weekly workflow (runs, in this order)](#authoritative-weekly-workflow-runs-in-this-order)
    - [Outputs and conventions](#outputs-and-conventions)
  - [Command line](#command-line)
  - [Web UI](#web-ui)
  - [Validation](#validation)
  - [Leakage audit](#leakage-audit)
  - [Changelog](#changelog)
  - [Artifacts](#artifacts)
  - [Confidence pool rules (implemented)](#confidence-pool-rules-implemented)
  - [Score rounding / realism (optional)](#score-rounding--realism-optional)
  - [Repository layout notes](#repository-layout-notes)
  - [Implemented feature areas](#implemented-feature-areas)
  - [Open work](#open-work)
  - [Development notes](#development-notes)
  - [Safety and claims](#safety-and-claims)

## What this project produces

- **Predicted margin and total** for each game (canonical targets).
- **Predicted home/away scores** derived from margin/total.
- **Calibrated win probabilities** for confidence ranking.
- **Weekly confidence ranks** (1..N unique values) suitable for pool submission.
- **Backtest summaries** for pool points and probability calibration.

## Setup

### Python environment

This project targets **Python 3.14+** (see `pyproject.toml`).

#### Recommended: uv project workflow

This repo uses `pyproject.toml` plus `uv.lock` for reproducible runs:

- `pyproject.toml` is the source of truth for runtime and development dependencies
- `uv.lock` is the lockfile used to sync environments reproducibly

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

To refresh the lockfile and sync the active virtual environment:

```bash
./update_requirements.sh
```

The helper expects `uv` on your `PATH` and an activated project virtual environment. If `.venv` is
missing, it offers to create one with `uv venv .venv` and then exits so you can activate the
environment before re-running it.

The lockfile installs the CPU-only LightGBM wheel. On a Linux machine with a CUDA toolkit, the
helper instead keeps a CUDA build of the same LightGBM version: it syncs with the flags that
`nfl-lightgbm-cuda-install uv-args` prints, then runs `nfl-lightgbm-cuda-install install`, which
does nothing when LightGBM already trains on the GPU and otherwise reinstalls uv's cached CUDA
build (compiling from source, about three minutes, only when no cached build exists). The CUDA
build needs NVIDIA's NCCL for the toolkit's CUDA major version (`libnccl2` and `libnccl-dev`
tagged `+cuda13.x` for CUDA 13, from `developer.download.nvidia.com/compute/cuda/repos`); with
Ubuntu's own NCCL, which is built for CUDA 12, the flags are withheld and LightGBM stays
CPU-only. `nfl-lightgbm-cuda-install status` reports which build is installed. On this
repo's data, LightGBM trains faster on the CPU than with CUDA, so the CUDA build is optional:
when it fails, the helper prints a warning, keeps the CPU build and finishes the refresh.

For a manual upgrade without the helper script:

```bash
uv lock --upgrade
uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)
.venv/bin/nfl-lightgbm-cuda-install install
```

#### Alternative: create the venv manually

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

uv sync
```

### Run tests

```bash
python -m pytest
```

By default, pytest runs with coverage enabled (configured in `pyproject.toml`). To disable coverage
for a quick local run:

```bash
python -m pytest --no-cov
```

`tests/conftest.py` runs Polars, OpenMP (XGBoost) and BLAS on one thread each unless
`POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` or `MKL_NUM_THREADS` is already
set. The suite builds thousands of tiny frames and models, where pools sized to every core spend
their time waking and spinning threads: on a 24-core machine it took about 174 s with the library
defaults and about 42 s with one thread each. `pytest-xdist` then spreads the suite across one
worker per core by default (`-n=auto` in `pyproject.toml`; about 15 s on the same machine). Pass
`-n 0` to run in a single process, for example to use `pdb` or to run one test quickly.

The suite also stays out of the checkout's `data/` and `models/` and off the network
(`tests/data_isolation.py`, installed by `tests/conftest.py` before any test module imports the
package). `constants.DATA_PATH` and the nflreadpy and nfelo caches point at a session temporary
directory, so every path a module copies from them at import time starts out redirected (to that
shared session directory), and each test gets a fresh directory on top. `constants.MODELS_PATH`,
which the run directories, the walk-forward checkpoints and this checkout's web run index read when
they run, points at a second temporary directory the same way. Every TCP connection and every
lookup of a host name other than `localhost` is refused, as on an offline machine, for the whole
session, fixtures included; a refusal fails the next test to finish, or the session. A test that
checks the packaged paths themselves asks for the `packaged_data_paths` or `packaged_models_path`
fixture, and tests in `tests/test_conftest.py` scan every module for a path still inside `data/`
or `models/`. Three things are not redirected, because none of them reads those constants; tests
that touch them pass their own paths:

- `api.settings.Settings()` built without a root, which fills its data, state and database paths
  from `constants.ROOT_DIR` (the checkout's `data/web/app.db`, the running web app's database); its
  models directory follows `constants.MODELS_PATH`;
- the floor-sigma reference runs (`constants.FLOOR_SIGMA_REFERENCE_RUNS`, by default
  `models/step6/ratings_anchored_seed*`), which `ml/floor_sigma.py` resolves against
  `constants.ROOT_DIR`: the default of `nfl-predictor train --floor-sigma-reference-runs` and the
  weekly run's `floor_sigma_reference_runs`, read when a model is trained;
- the relative `data/...` defaults of `nfl-predictor rankings` (`--data-ml`, `--data-schedule`,
  `--out-dir`), resolved against the current directory.

To run explicitly with coverage (same behavior as the default config):

```bash
python -m pytest --cov=nfl_predictor --cov-report=term-missing
```

Pytest now enforces the repo coverage floor of **90%** by default via `pyproject.toml`. To try a
stricter local target, add `--cov-fail-under` with a higher value. The preseason hardening target
remains **90% or higher**, with **100%** as the aspirational ceiling:

```bash
python -m pytest --cov-fail-under=95
```

Current validated local baseline as of 2026-09-25 (version `0.27.1`): `1020 passed` with `92.15%`
coverage. `scripts/gate.sh` is the source of truth for this number; re-run it rather than trusting
this line as the project grows.

## Data collection (Polars + nflreadpy)

The authoritative data build pipeline is:

```bash
nfl-predictor data
```

The default season range is controlled by `constants.MIN_SEASON` (currently 1999).

This writes datasets under `data/` (paths are defined in `src/nfl_predictor/constants.py`). Pass
`--data-dir` to write them somewhere else; the upstream inputs and caches the ETL reads still come
from `data/`.

Typical outputs:

- `data/all_data.csv` and `data/all_data_ml.csv`
- `data/completed_games.csv` and `data/completed_games_ml.csv`
- `data/predict/week_XX_games_to_predict.csv`
- `data/strength_snapshots.csv`: pre-week schedule-adjusted strength per team, one row for every
  team on each season's schedule and every processed week (teams on a bye included, plus the week
  after the regular season), keyed by `season`, `week` and `team_abbr`. The values equal the
  `away_`/`home_` strength columns on that week's game rows, plus the league-wide home-field term
  `adj_hfa`. Power rankings read it.
- `data/own_elo_metadata.json`: how the own Elo columns were built (each preset's switches and
  fitted weights, the first and last built season, and the win-total file the run read: its URL,
  origin and SHA-256). A build without the own Elo writes `{}`, so a record never outlives the
  build it describes.
- `data/ratings/{season}_{kind}.parquet`: the team and quarterback ratings ported from
  `nfl-sos-ratings`, one file per season and table, named as that project names its files
  (`ratings`, `team_prior`, `ratings_by_week`, `rating_ranges`, `rating_pairs`, `qb_ratings`,
  `qb_ratings_by_week`, `qb_rating_ranges`, `qb_rating_pairs`), and for the current season each
  week's rank ranges (`rating_ranges_by_week`, `qb_rating_ranges_by_week`). A finished season's
  weekly ranges stay as its last build as the current season wrote them. Beside them, the season
  statistics ported from the same project, under its names too: `team_game_logs`,
  `qb_game_logs`, `team_per_game_stats`, `qb_per_game_stats`, `opponent_profiles`,
  `qb_opponent_profiles`, `combined` and `qb_combined` (the season rows with the opponent profiles
  and the season's ratings), `team_wp_bins` and `qb_wp_bins`; a season gets them once its first
  game is played (the two opponent-profile tables once a profile can be computed, from week 2),
  and each has the same columns and types in every season, null where what fills them is missing
  (a source nflverse has not published yet, an opponent profile too early in the season in
  `combined` and `qb_combined`). [The ratings README](src/nfl_predictor/ratings/README.md) and
  [the season statistics README](src/nfl_predictor/season_stats/README.md) describe every table.

Note: the `data/` directory is gitignored by default; generate it via the data collection step
above. Note: `*_ml.csv` files include model-ready engineered features.

Game rows are ordered newest first, with games on the same date ordered by `game_id`. Two runs
with the same code, locked dependencies and inputs, on the same machine with the same Polars
thread count (`POLARS_MAX_THREADS`), write byte-identical files.

Building each season's weekly feature rows is nearly all of the ETL's run time.
`nfl-predictor data --incremental` keeps each finished season's rows, strength snapshots, ratings
tables and season statistics tables under `<data dir>/cache/etl_seasons/` and reuses them when
nothing they are built from has changed; the season in progress is always rebuilt, and every
input is still loaded. An entry is keyed on:

- the source of the ETL modules (`data_collection.py`, `constants.py` and everything under
  `src/nfl_predictor/utils/`, `src/nfl_predictor/ratings/` and `src/nfl_predictor/season_stats/`),
  the Polars, Polars runtime and NumPy versions, the Python version, the CPU architecture and the
  Polars thread count;
- the options that change a season's rows (`--min-season`, the two prior-blend switches,
  `--strength-prior-blend-games`, `--stat-prior-blend-games`, `--team-stats-source`,
  `--tr-stats-source` and `--line-source`);
- the loaded schedule, team stats and Elo rows and the ratings' team-game and passer-game rows
  from that season and every earlier one, and that season's and the previous season's
  TeamRankings, compared by value;
- that season's own play-by-play, team-game and passer-game rows and played schedule as the
  season statistics read them, compared by value. The season statistics' game rows are built
  again on every incremental run for this: they read the nflverse players file, which a run in
  season downloads again, so only a change to a player the season's sources name moves them.

So a changed input row rebuilds its season and every later one (a row only the season statistics
read rebuilds its season alone), and an edit to the ETL code rebuilds everything. A stale,
damaged or unreadable entry is rebuilt and rewritten, never an error, and the directory is safe to
delete. An input with a column the by-value comparison cannot render (list, struct, array,
duration, binary or object) leaves its seasons uncached. Without the flag, which is
`nfl-predictor data`'s default, the ETL rebuilds every season and neither reads nor writes the
cache. The weekly run's refresh passes it (`data_collection_args: "--incremental"` in
`config/weekly_run.yaml`; "Weekly workflow (canonical)" below says how your own arguments combine
with it).

A reused season equals a full rebuild of the same inputs byte for byte, also after the season in
progress (or any later season) gains rows: a season's build reads nothing from later seasons,
not even through where Polars splits the loaded frames into chunks, which moves with their total
length. For that reason the league means behind the week-1 prior rechunk their season's slice of
the team-stat frame before reducing it.

Historical seasons load from cached artifacts where available. nflreadpy outputs are cached per
season under `data/cache/nflreadpy` (schedule, team stats, and play-by-play). Current/future
seasons are always refreshed to keep upcoming games and lines current. Each schedule, team-stat and
play-by-play cache file records the columns the loader asked for when it was written; a file that
lacks a column the code now asks for (a column added to `constants.PBP_COLUMNS`, to the team-stat
mapping, or to the schedule selection or renames) is a cache miss and that season downloads again,
so a historical schedule refetch can pull revised lines; a column the source never
published does not force a download on every run. Such a column is not checked again: if nflverse
publishes it for that season later, rerun the ETL with `--refresh-nflreadpy` to rewrite the cache.
Use
`--min-season`/`--max-season` to override the default season window (defaults to
`constants.MIN_SEASON` through the current NFL season).

For completed regular-season games, the ETL now builds the per-team-game frame from the schedule
first: each completed game contributes exactly two team rows, and nflverse team stats plus the
play-by-play counts are left-joined onto that skeleton. When nflverse misses one side of a game,
the row still exists with null box-score stats, so schedule-driven counts such as
`strength_games_played` continue to follow the schedule instead of the stats source's coverage.
The ETL logs every `(season, team)` whose nflverse team-stat row count differs from the schedule.

Play-by-play is the largest of those sources (roughly 1.2M regular-season plays for 1999-2025). It
is fetched one season at a time, reduced to the column list in `constants.PBP_COLUMNS`, filtered to
the regular season, team-normalized, and cached as `data/cache/nflreadpy/pbp_<season>_reg.parquet`.
The list also holds every play-by-play column the ratings ported from `nfl-sos-ratings` read
(play ids, scramble and botched-snap flags, rusher, receiver, fumble and tackle credits, game
state, drive and series results, and the play text); the ratings' rows read the ones they need.
Every column that holds a team code (`constants.PBP_TEAM_COLUMNS`) is normalized; yard-line text
such as `LA 25` is kept as nflverse writes it. Each column keeps one type across seasons (nflverse
publishes `goal_to_go` as a whole number in some seasons and a float in others; the cache stores
floats). Before kickoff the current season has no play-by-play published at all; that is
non-fatal, and the ETL falls back to cache or continues without it.

The nflverse player sources (`src/nfl_predictor/utils/polars/nflverse_sources.py`: weekly player
stats, snap counts, the players file, and season and weekly rosters) load under the same cache
contract. Each season is kept to its column list in `constants.py`, filtered to the regular season
where the source has a season type, given canonical team codes (the 2002-2015 rosters' league codes
`ARZ`, `BLT`, `CLV`, `HST` and `SL` included), and cached as
`data/cache/nflreadpy/<source>_<season>[_reg|_all].parquet`; the players file, which covers every
season, is one `players.parquet`. Weekly player stats leave null the stats nflverse never charted
in a season (tackles for loss in 2003-2011, quarterback hits in 2003-2005), and in Jacksonville's
2001-2002 home games, which nflverse credits wholly to the visitors (nflverse-pbp issue 92), each
player's team comes back from that season's roster (`utils/polars/player_team_repair.py`), in the
weekly player stats and in the play-by-play the ETL aggregates: who scored, fumbled, recovered,
tackled or drew a penalty, so each penalty is charged to the penalized player's team. A player is
a quarterback in a season when the players file or any regular-season week of that season's
weekly roster lists him there, matched by GSIS id and never by name
(`utils/polars/player_identity.py`). ESPN QBR (`utils/polars/espn_qbr.py`) is cached the same way
as a validation reference only: nothing that builds the dataset or the model imports it. The ETL
reads the season rosters of 2001 and 2002, for that repair, and the player sources for the
ratings, every season from 1999 (so the first run that does downloads each season once); it
downloads the players file again whenever the run reaches the current season, falling back to its
cached copy on an outage. Every season's quarterback identity, for the starter rule below, reads
the players file and weekly rosters the ratings loaded, so a run loads each of them once.

Every quarterback feature keys on one starter: the quarterback family and the ratings'
`ratings_qb` read it from `qb_stats.expected_starter_ids`, and the own Elo applies the same rule
(the `nfl-sos-ratings` rule, `utils/polars/qb_starters.py`). A completed game's starter comes from
its play-by-play: the quarterback on the team's first first-quarter snap, by `play_id`, whose
passer or rusher has a dropback for the team in that game and is a quarterback that season (or is
listed by no identity source), the passer before the rusher on one play. A game the play-by-play
names no starter for (an upcoming game, every postseason game, a game without play-by-play, and
the six 1999-2025 team-games without a first-quarter quarterback snap) takes the starter the ETL
names (`away_qb`/`home_qb`: the projected starter of an upcoming game). nflverse's schedule
columns `away_qb_id`/`home_qb_id` are not read. A named starter maps to a GSIS id through
`data/meta_data.csv`, a read-only copy of `../nfeloqb/Other Data/meta_data.csv` made the same way
as `data/qb_elos.csv`, else through the play-by-play passer name (`F.Last`) when exactly one
passer of the games before the row's week has it; without the file only the play-by-play
starters and those passer names identify quarterbacks, and the ETL logs a warning. Career rates
need every earlier season, so the ETL reads the cached play-by-play from 1999 even when
`--min-season` is later.

TeamRankings data is cached under `data/<season>/` as week-level CSVs; enable debug logging to see
cache hits. Use `--timing` to log per-step runtimes and `--debug-logs` for detailed ETL diagnostics.
Use `--refresh-nflreadpy` to force refresh nflreadpy data even when cache exists.

Two flags choose where per-team-game stats come from; both default to `pbp` since 2026-09-21.
`--team-stats-source pbp` derives the box-score families (passing, rushing, penalties, first downs,
sacks and interceptions) from play-by-play and overlays them on the nflverse rows, so nflverse still
fills whatever play-by-play cannot derive; pass the flag with `nflverse` to use the scraped/nflverse
sources instead. `--tr-stats-source pbp` derives the eight situational percentages (third down,
fourth down, red zone and two-point, for and allowed) from the play-by-play counts instead of the
TeamRankings scrape; pass `scrape` to use the scrape. TeamRankings supplies its ratings either way.
The derived percentages use the scraped columns' 0-100 scale, and `red_zone_td_pct` divides
touchdown drives by red-zone trips, which is what the scraped "red zone scoring %" measures -- not
touchdowns per red-zone snap. A red-zone trip is a drive nflverse flags as getting inside the
opponent's 20 (`drive_inside20` on any of its plays) in which the team ran at least one offensive
snap, a try not counting as one; with rare exceptions the flag leaves out the try after a touchdown,
which is snapped inside the 20 by rule. The snap is a departure from `nfl-sos-ratings`, whose
definition (every flagged drive) the ported season statistics keep for display: nflverse files a
kickoff the kicking team recovers under the receiving team, with the kicking team's next drive and
its flag, and the try after a return or defensive touchdown as a drive of the scorer's own, and
neither is a trip here. `passing_epa` sums `qb_epa` (nflverse's own quarterback EPA attribution)
rather than `epa`, and the box-score counts and yardage use nflverse's own
`pass_attempt`/`rush_attempt` raw flags rather than `play_type`; both were verified against nflverse
team stats at 99%+ agreement on the columns they touch
(`models/pbp_vs_nflverse_m54_2/COMPARISON.md`). `fumbles`/`fumbles_lost` exclude special-teams plays
to match nflverse's offense-only fumble stat, with a residual gap on aborted-snap fumbles;
`2pt_conversions` still disagrees with nflverse on a meaningful share of team-games, but this is not
a derivation defect: nflverse's own team-stats table has a confirmed, systematic bug that roughly
doubles the count of successful two-point conversions on the games it gets wrong (see
`models/pbp_vs_nflverse_m54_2/COMPARISON.md`), so the play-by-play value is the correct one where
they differ. Both flags change feature values at ETL time, so compare them with two dataset builds
rather than with `--disable-feature-groups`.

Early-season handling: Week 1 has no in-season games, so every season-to-date team stat (the
nflreadpy families and the play-by-play counts) falls back to the previous regular season regressed
one third of the way toward the league mean (`constants.WEEK1_REGRESSION_FACTOR`). From week 2 on,
each team's per-game means are blended toward that same prior, weighting the in-season sample
`games / (games + K)` with `K = constants.PRIOR_BLEND_GAMES` (`4`): one game is 20% of the value,
four games 50%, twelve games 75%. Rates are recomputed from the blended sums. Use
`--stat-prior-blend-games` to change `K` and `--no-stat-prior-blend` to publish plain in-season
means instead; both change feature values, so compare them with two dataset builds, not with
`--disable-feature-groups`. The first season in a run has no prior and uses in-season means, and the
schedule-adjusted strength family carries its own blend of the same form, with its own `K`
(`--strength-prior-blend-games`, default `constants.PRIOR_BLEND_GAMES`) and its own ablation
(`--no-strength-prior-blend`).

Market lines: `--line-source` chooses which line each game's five line columns (`total_line`, the
spreads and the moneylines) carry. Those columns are the market features and the anchor of both
heads, in training and in production alike. The default (`constants.DEFAULT_LINE_SOURCE`) is
`pick_time`, so the weekly run's data refresh and the web ETL and weekly jobs build it too.

- `pick_time` (the default): the line known when picks are made, from nfelo's market-lines file
  (`greerreNFL/nfelomarket_data`, `Data/lines.csv`), joined on season, week and the canonical
  team codes. A completed game takes nfelo's opening spread where the opener is real (2007-2021
  and 2023 on, when present; about 5% of 2024 has none), its opening moneylines when nfelo has
  both (2024 on), otherwise moneylines derived from the opening spread, and its opening total,
  or the stored total before nfelo published one. Every other completed game (1999-2006, 2022, a
  missing opener) keeps its stored line and is counted as a fallback row. An upcoming game takes
  nfelo's latest line, then nflverse's, then SurvivorGrid's. A game with no line at all still
  stops the model run, as with `stored`. The closing line is never a feature under this source.
  Derived moneylines come from a spread-to-moneyline map fitted to the market's own prices
  (`utils/polars/moneyline_map.py`): for each half-point spread, the mean implied probability of
  the favorite's and the underdog's moneylines over games with a real spread and both real
  moneylines, kept monotone, so the key numbers 3 and 7 keep the market's jumps. Each season's map
  is fitted on earlier seasons only (prices, never scores), from the cached schedules back to
  1999; a season with fewer than `constants.MONEYLINE_MAP_MIN_GAMES` earlier priced games (2006
  and before) keeps the fixed conversion.
- `stored`: the nflverse schedule lines, one late, probably closing, snapshot. Missing
  moneylines are derived from the spread with the fixed conversion
  (`game_utils.spread_to_moneyline`: a normal curve at `SCORE_DIFF_STD_DEV` with a flat 5% vig),
  and future games without a line take SurvivorGrid's spread.

nfelo's file is rewritten several times a day. Each pick-time run downloads it, keeps the last
good copy in `data/cache/nfelo/lines.csv` and every copy a run used in
`data/cache/nfelo/snapshots/`, named by its SHA-256, and falls back to the cached copy (or, with
none, to the stored lines) instead of failing. The run writes `market_lines_metadata.json` beside
the datasets: the snapshot's URL, origin and hash, per-season game, match, opener and fallback-row
counts, how many moneylines each season derived and how, and every season's map. Two diagnostic
counts per season flag possible data errors without filtering anything: openers and upcoming nfelo
lines on the other side of the stored line, both at least 3 points (`opener_sign_flips`,
`upcoming_nfelo_sign_flips`). A `stored` build fetches nothing new and writes
`{"line_source": "stored"}` to the same file, so a record never outlives the build it describes.
The option changes feature values, so compare the two sources with two dataset builds, not with
`--disable-feature-groups`.

`nfl-predictor lines --season <season> --week <week>` (also the web "Refresh market lines" job)
refreshes only the five line columns of one season in `all_data_ml.csv`, `all_data.csv` and the
week's `predict/week_<NN>_games_to_predict.csv`, without rerunning the ETL, and follows the line
source the data was built with, read from `market_lines_metadata.json`. On a pick-time build it
writes what a rebuild would: the pick-time line order above, with missing moneylines priced by the
season's map from the record. On a stored build it reads the nflverse schedule with the fixed
conversion. A data directory without the record predates it and is refreshed as a stored build,
with a warning; an unreadable record, an unknown line source, or a pick-time record without a map
entry for the season stops the refresh before anything is written. Completed-game datasets are
never touched. Re-predict the week after a refresh, because the lines feed the model.

## Data sources + missing data

This project is designed to keep an invariant output schema across seasons, even when some sources
are missing historically.

Primary sources:

- `nflreadpy` (NFLverse): schedules, results, team-level stats, and play-by-play.
- nflverse player sources (weekly player stats, snap counts, the players file, rosters) and ESPN
  QBR as a validation reference: cached loaders the ETL does not read yet (see "Data collection"
  above).
- nfelo's market-lines file (`greerreNFL/nfelomarket_data`) for the default
  `--line-source pick_time`.
- Local cached CSVs under `data/` for Elo/market data when present.
- `data/qb_elos.csv` (copied from `../nfeloqb`) for the Elo columns. A week the copy does not list
  yet takes each team's latest listed rating, a team Elo trend of 0.0 and no quarterback trends;
  the ETL warns when that week is the one being predicted, so refresh the copy before a weekly run.
- `data/meta_data.csv` (copied from `../nfeloqb`) for the quarterback name-to-id bridge. A name
  missing from it falls back to the play-by-play passer name (`F.Last`) when that is unique; a
  quarterback still unmatched gets null quarterback features, and the ETL logs the unmatched rate.
- TeamRankings web scrape for select ratings and stats not available in NFLverse (see ETL logs).
- nfelosrs's preseason win-total ratings (`greerreNFL/nfelosrs`, `wt_ratings.csv`, 2003 on), read
  by the own Elo's win-total preset (see "Implemented feature areas"). Like nfelo's lines, the
  last good download is cached in `data/cache/nfelo/win_totals/wt_ratings.csv`, every copy a run
  used is kept under `snapshots/` by its SHA-256, and a failed download falls back to the cache.
  A build whose win totals (downloaded or cached) cover none of its seasons stops with an error
  instead of publishing that preset without its pull; a built season the file lacks (the current
  one, before nfelosrs publishes it) is logged, and its preseason ratings get no pull.

Missing data policy (high level):

- ETL emits all expected columns; missing sources become nulls and/or defined defaults.
- The ML pipeline is expected to tolerate nulls (imputation and/or model-native missing handling).

## Training + prediction

Training and prediction run through `nfl-predictor train` and `nfl-predictor predict`, which take
the same options (`predict` needs `--model-in`, a saved model):

```bash
nfl-predictor train --help
```

### Quickstart (train + predict)

Train a margin/total model and generate predictions for a weekly input file:

```bash
nfl-predictor train \
  --model-kind margin_total \
  --holdout-seasons 0 \
  --win-prob-calibration none \
  --tune \
  --tune-timeout 600 \
  --tune-objective expected_points \
  --xgb-tree-method hist \
  --predict-path data/predict/week_17_games_to_predict.csv
```

This prints a weekly summary and writes `*_predictions.csv` next to the input file.

If you want a minimal run without tuning (and with explicit input paths):

```bash
nfl-predictor train \
  --data-path data/completed_games_ml.csv \
  --model-kind margin_total \
  --holdout-seasons 1 \
  --predict-path data/predict/week_17_games_to_predict.csv
```

### Splits: train and holdout

Splits are time-aware by season: the model trains on every eligible completed game except the
evaluation holdout seasons, the newest completed week included, exactly as each walk-forward fold
trains on every game before its week.

By default, training/evaluation uses **regular season** games only when the input data includes a
`game_type` column (i.e., postseason rows are filtered out). You can still generate predictions for
playoff games as long as the feature row exists.

To include postseason games in training, pass `--include-postseason`. To emphasize postseason games,
also set `--postseason-weight` (e.g., `--postseason-weight 1.5`). Optional recency weighting is
available via `--recency-half-life-seasons` to apply exponential decay by season to training
samples. It is off by default and in the shipped
weekly config: the six-season, two-seed measurement below found no gain from it.

- `--holdout-seasons` reserves the most recent whole seasons for evaluation only (`train`
  defaults to 2, the weekly run to 0); their metrics are logged and written to
  `metrics_report.json`.
- No fit hands XGBoost an eval frame: every head runs its full tree budget, and no calibrator is
  fitted (see "Win probability").
- The run log lists the training and holdout seasons, and `metadata.json` records them under
  `splits`.

Example: hold out the most recent season for evaluation:

```bash
nfl-predictor train \
  --model-kind margin_total \
  --holdout-seasons 1
```

`--disable-feature-groups` leaves feature groups out of the fit, exactly the columns the
backtest's option of the same name drops (see "Backtesting"), and the model's `metadata.json`
records the groups and those columns. Its default is the weekly run's,
`strength,own_elo,own_elo_wt` (see "Backtesting"); an empty value (`--disable-feature-groups ''`)
leaves no group out. A saved model predicts with the columns it was trained on, so
`nfl-predictor predict` ignores an option given and logs that it does.

## Modeling approach

### Margin/Total targets (canonical)

We predict:

- `margin = home_score - away_score`
- `total  = home_score + away_score`

Then derive scores:

- `home = (total + margin) / 2`
- `away = (total - margin) / 2`

This keeps score predictions internally consistent and makes win probability derivation
straightforward.

### Win probability

Every run type (the weekly run, `backtest`, `train`, `predict`) maps the predicted margin to a
home win probability through one curve, the deterministic floor: `Phi(margin / sigma)`, the
normal CDF of the predicted margin over sigma, the spread of the model's own errors. For a game
in season `s`, week `w`, sigma is the root-mean-square of (actual minus predicted margin) over
every out-of-fold prediction strictly before `(s, w)`: earlier seasons, and season `s` weeks
before `w` (`src/nfl_predictor/ml/floor_sigma.py`). It is one value per week, so it changes how
confident the probabilities are but never a pick or a confidence rank. Until that pool spans
three earlier seasons, any weeks of them (`FLOOR_SIGMA_MIN_POOL_SEASONS`), sigma is the fixed
`SCORE_DIFF_STD_DEV` (`14.21`, in `src/nfl_predictor/constants.py`), and the outputs say it fell back.
Nothing is fitted on top of the floor and the market line never enters it; the pick and its
confidence come from it.

Where the pool comes from:

- A walk-forward (`nfl-predictor backtest`, and the weekly run's stage 1) pools its own earlier
  weeks with any supplied history: `backtest --floor-sigma-reference-runs <run dirs>` (none by
  default), and the weekly run's reference runs in stage 1. A week the run predicts itself
  replaces the history's errors for that week. Every week's `floor_sigma`,
  `floor_sigma_fallback` and `floor_sigma_pool_games` are in its per-week metrics, the
  prediction frame carries `floor_sigma` and `floor_sigma_fallback`, and the report's
  `floor_sigma` block names the history.
- The production final fit (the weekly run, `nfl-predictor train`) pools the out-of-fold errors
  of the reference runs, `floor_sigma_reference_runs` in `config/weekly_run.yaml` and
  `--floor-sigma-reference-runs` for `train` (default: the two seeds of the reviewed reference
  walk-forward of production's settings, market anchored with the strength snapshot and both
  own-Elo presets left out, `models/step6/ratings_anchored_seed42` and
  `ratings_anchored_seed7`, whose fold checkpoints are only read), with, in the weekly run,
  stage 1's errors before the predicted week at every `(season, week)` the reference runs have
  no rows for: the seasons after the reference and the predicted season's earlier weeks (where
  both have a week, the reference's errors are used).
  Two seeds are averaged per game, so each game counts once (over the same games
  this equals pooling every row). The sigma is estimated for the predicted week (for `train`
  without a single-week `--predict-path`, the week after the newest game in the data) and
  recorded in the saved model and in its `metadata.json` (`floor_sigma`: the value, whether it
  fell back, the week, the pool's size and seasons, and the runs it came from). Prediction from
  a saved model (`nfl-predictor predict --model-in`) uses the recorded value and reads no
  reference run; a model saved before the sigma was recorded predicts with the constant. A
  configured reference run that is missing stops the run with an error naming it; an empty list
  (`floor_sigma_reference_runs: []`, or `--floor-sigma-reference-runs` with no paths) runs
  without reference errors, and the record then says whether the sigma fell back.
- Weekly predictions carry `floor_sigma` and `floor_sigma_fallback` columns, and the projected
  standings use the model's recorded sigma for the remaining games.

`--win-prob-calibration` (`train`, `predict`) and `--calibration` (`backtest`) take `auto`, the
floor, and the default; `none` is accepted and means the same. A model saved with a fitted or
Elo calibrator, a market blend or a clamp still loads: the run log names what it ignores, and
the model predicts the floor of its own margins. The walk-forward report carries the floor and
the market-implied probability side by side, with paired intervals.

### Market integration (optional, recommended)

If spreads/totals/moneylines are present, you can:

- use market-derived features (`--market-transform`)
- train on residuals vs market baselines (`--market-anchor`) so the model learns deviations rather
  than re-learning what the market already priced

The market-implied home win probability (no-vig moneylines, else the spread) is scored beside the
model in every walk-forward as the yardstick; it never enters the submitted probability.

The stats-only mode means "no game lines", not "no market information". It comes with two
stricter variants kept as test benches. A group list given replaces the default
(`strength,own_elo,own_elo_wt`), so each variant names the whole list:

- No game lines (the mode):
  `nfl-predictor backtest --exclude-market --no-market-anchor --no-market-transform`. No line
  column (the spreads, the total and the moneylines) and no column computed from them
  (`market_home_margin`, `market_total_line`, `home_market_prob`, `away_market_prob`) reaches the
  feature matrix, and nothing is anchored to the lines. With no group option it leaves out
  production's groups and keeps the ported ratings and the `elo` group, with its preseason
  win-total prior.
- No market information, with the own Elo: the same options plus
  `--disable-feature-groups strength,elo,own_elo_wt` (`[strength, elo, own_elo_wt]` in the weekly
  config). It keeps the market-free own Elo (`own_elo`); the reference runs are
  `models/step6/elo_marketfree_seed42` and `models/step6/elo_marketfree_seed7`.
- No market information and no Elo at all: the same options plus
  `--disable-feature-groups strength,elo,own_elo,own_elo_wt`, the default list plus `elo`; the
  reference runs are `models/step6/ratings_marketfree_seed42` and
  `models/step6/ratings_marketfree_seed7`.

The `elo` group is every column built from `data/qb_elos.csv` (18: the team Elo, quarterback Elo
and quarterback value, each side and their difference, and their four-week trends).
`../nfeloqb` pulls each team's preseason Elo toward a rating built from betting win totals
(2003-2023 in its `wt_ratings.csv`), and the own Elo's win-total preset (`own_elo_wt`, 4 columns)
pulls toward nfelosrs's win-total ratings, so those columns carry preseason win-total priors.

Every variant still scores the market-implied probability as the yardstick. A run without game
lines that keeps either market-informed group (`constants.MARKET_INFORMED_FEATURE_GROUPS`) logs
a "No game lines" info message naming it and how to leave it out; with the default groups it
names `elo`. The weekly run's `--wf-market-mode none` is the mode for stage 1 and the final fit
(no game-line column, no anchor and no transform), and `nfl-predictor train --exclude-market` is
too, since its anchor and transform are off by default. Either becomes a stricter variant with
its group list, in `disable_feature_groups` in `config/weekly_run.yaml` or as
`--disable-feature-groups` on `nfl-predictor weekly` or `nfl-predictor train`.
`.agents/market_free/MARKET_COLUMNS.md`, generated by the script beside it, lists every column
derived from betting lines in a build and every column of the market-informed groups, and checks
that the mode admits no line-derived column and the stricter variants no column of either kind.
It reads a dataset build, so it is regenerated after each rebuild: the copy in the repository
comes from a build without the own Elo's columns and checks the stricter variants with `elo`
alone.

## Backtesting

Evaluate with walk-forward (rolling-origin) backtesting, which scores every week out of
sample, confidence-pool points included:

```bash
nfl-predictor backtest --help
```

This is the **canonical evaluation protocol** for model selection. By default it evaluates the last
N seasons (regular season only) and scores the model's probability, the deterministic floor,
against the market-implied home win probability on the same games.
Use `--include-postseason` if you want postseason folds included. Optional recency weighting
is available via `--recency-half-life-seasons`. XGBoost trains on the GPU by default:
`--xgb-device auto` (the default) uses CUDA when the installed XGBoost build has it and a usable
GPU is present, and the CPU otherwise; `--xgb-device cpu` or `--xgb-device cuda` picks one. The
resolved device is part of the fold-checkpoint fingerprint, so CPU and GPU runs never resume each
other's weeks, and the run's `metadata.json` records it as `config.xgb_device`. If the latest
season is incomplete, either pass `--wf-exclude-incomplete-seasons` or specify `--eval-seasons`
explicitly; the metrics report includes the evaluated window and any exclusions. Each fold's
trees fit on every completed game before its week, as the final fit trains on every completed
game, with no eval frame; the summary table includes deterministic-minus-market bootstrap
intervals for week 1, week 2, weeks 3-18, and all weeks; and every fold runs the full
`n_estimators` budget (no in-season early stopping, in walk-forward or in production), with
`best_iteration` recorded per head.

The report also carries a stability view, `metrics.stability` in `metrics_report.json`, and the run
logs it as Markdown tables when it finishes. For week 1, week 2, weeks 3-18 and all weeks it gives
an all-seasons row and one row per season with the deterministic Brier, log loss, pick accuracy,
margin and total MAE, confidence-pool points, market Brier, and the deterministic-minus-market
Brier with a 95% game-bootstrap interval. The rows use the definitions and bootstrap defaults of
`nfl-predictor compare` (below), so they equal what `compare` reports for the run on the same
games. A row with fewer than two games has no interval. The view scores pick accuracy and pool
points on the deterministic probability, and `metrics.overall` and `metrics.per_season` in the
same file score the submitted `home_win_prob`; both are the floor, so the two agree.

The report also says how the run differs from production: `settings_versus_production` in
`metrics_report.json`, logged under "Settings versus production" when the run finishes. The
production side is what `nfl-predictor weekly` would use today, read through the weekly run's own
parser and `config/weekly_run.yaml` merged with the code defaults, for both of its halves: stage 1's
walk-forward and the final fit. For each half, `differences` lists every model-affecting setting
where the run's resolved value differs (the XGBoost device and parameters, built by the resolver
training uses, market features, anchoring and transform resolved against the dataset's lines,
recency weighting, calibration, postseason games and their weight, disabled feature groups,
trend-feature ablation, pruning, holdout seasons, tuning). Production's disabled feature groups are
the weekly config's `disable_feature_groups`, the same in both halves, and the groups are compared
as a set, so the same groups in another order show no difference. A setting one side applies
implicitly is compared at that value: production never drops pruning or trend features, and a
walk-forward fold holds out no season, never tunes and weights postseason games like any other. What
remains on one side only is listed under `run_only` or `production_only` (every `tune_*` option when
production tunes), and `not_recorded` names production settings an older run's metadata lacks (its
resolved XGBoost parameters, which the backtest now records as `config.xgb_params`); settings an
older run recorded that no longer exist are listed under `retired` with their values. A half reads
"no model-affecting differences" only when all four lists are empty and nothing is retired; the run
then trains what production trains. Scope (the scored seasons and weeks, the seed, the dataset and
checkpoint paths, production's dataset and data-collection arguments) and runtime settings that
change no prediction (XGBoost threads and verbosity, quantile models) are listed apart, never as
differences.

Every finished week logs its position, running time, and an estimate of the time remaining
(`Walk-forward fold 37/54 done: season 2024 week 5 (14 games, Brier 0.2213), 2410s elapsed, about
1107s remaining`). The estimate averages the weeks trained so far, so it runs a little low late in a
run as training sets grow. Measured on 2026-09-20/21 on a 24-core machine, training on the CPU
(these runs predate the GPU default): a from-week-1 run over
three seasons takes about 50 minutes on an idle machine and about 110 minutes when anything else
loads it; over six seasons it takes about 100 minutes idle at the default 200-tree budget, about
3.3 hours at 400 and about 4.8 hours at 598 under load (run directories
`models/wf_m55_7_2020_2025_trees*/`). Run one walk-forward at a time: XGBoost uses every core, and
two concurrent runs slow each other down far more than twofold. When anything else is busy on the
machine, OpenMP's passive wait policy (`OMP_WAIT_POLICY=PASSIVE`, the walk-forward commands'
default since `0.24.0`, below) matters: XGBoost's OpenMP threads otherwise spin while waiting for a
preempted peer. On 2026-09-10, with other jobs loading the machine, one walk-forward week took `730s`
with the default policy and `185s` with `PASSIVE`. On an idle machine keep the default: with
`PASSIVE` the threads sleep between XGBoost's many small parallel steps and waking them costs more
than it saves, so an idle week took `~142s` against `~82s` for the default. The setting changes
scheduling only, never results, so switching it mid-run and resuming is safe.

Walk-forward runs are **resumable**. Each finished week is saved under
`models/wf_checkpoints/<fingerprint>/`, where the fingerprint covers the input rows, the full config,
the modelling source code, and the installed library versions. Re-running an identical command
after a stop (deliberate or not) restores the finished weeks and trains only the rest; a resumed run
returns exactly the numbers an uninterrupted one would, which a test pins. Anything that changes the
fingerprint starts fresh, so stale results are never mixed in. `--no-resume` retrains every week
and `--checkpoint-dir` moves the root. The same checkpointing runs inside the run directories of
`nfl-predictor weekly` (its existing `--resume`). The metrics report records how many weeks
were restored and how many were trained. Checkpoints are small (a few hundred KB per run) and safe to
delete once a report is written. Nothing prunes them: `nfl-predictor checkpoints` lists every
checkpoint directory with its size and whatever names it (a run's report, a review, a launcher,
`AGENTS.md` or `.agents/`), and `--unreferenced-only` lists the ones nothing names. It never
deletes anything.

`weekly` and `backtest` run XGBoost with OpenMP's passive wait policy
(`OMP_WAIT_POLICY=PASSIVE`) unless the environment sets one: when other work shares the machine,
the default policy's spinning threads stall and a week can take several times longer. On a
dedicated, idle machine the default is faster; set `OMP_WAIT_POLICY=` (empty) to keep it. The
policy changes scheduling only, never results.

Trend/season-phase ablation (drop trend + season-phase features while keeping everything else
identical) is available via `--disable-trend-features`. Example 2x2 comparison matrix:

```bash
nfl-predictor backtest
nfl-predictor backtest --recency-half-life-seasons 16
nfl-predictor backtest --disable-trend-features
nfl-predictor backtest --disable-trend-features --recency-half-life-seasons 16
```

Named feature groups can be ablated the same way with `nfl-predictor backtest
--disable-feature-groups`. Group names come from
`constants.FEATURE_GROUP_COLUMN_MARKERS`; a column belongs to a group when any of that group's
markers is a substring of the column name, which catches the `away_`/`home_` prefixes and the
`_diff` suffix at once. An unknown group name is a hard error. The dropped column list is recorded
in the run's metrics report. The list replaces the default, so an ablation next to a plain
backtest names the default groups too:

```bash
nfl-predictor backtest --disable-feature-groups strength,own_elo,own_elo_wt,pbp
```

Production leaves groups out the same way, so an arm that disables groups can be adopted without
a gap between the benchmark and production: `nfl-predictor train --disable-feature-groups` takes
the same list, and the weekly run reads `disable_feature_groups` from `config/weekly_run.yaml` (a
list; `--disable-feature-groups` on `nfl-predictor weekly` overrides it) for both its stage-1
walk-forward and its final fit. Production leaves out the strength snapshot (`strength`) and both
presets of this repository's own Elo (`own_elo`, `own_elo_wt`), so it trains on the ported team
ratings (`ratings`) and the `qb_elos.csv` Elo (`elo`): that is the default of the weekly config,
`train` and the backtest alike (`constants.PRODUCTION_DISABLED_FEATURE_GROUPS`), so a backtest
without the option leaves out what production leaves out. An empty list
(`disable_feature_groups: []`, or `--disable-feature-groups ''`) leaves no group out, and a list
given replaces the default rather than adding to it (`--disable-feature-groups pbp` keeps the
strength and own-Elo columns). Every one of them turns groups into columns through one function
(`feature_spec.resolve_feature_group_columns`), so the same groups remove the same columns
everywhere, and each saved model's `metadata.json` names the groups it was trained without
(`disabled_feature_groups`) and the columns they removed (`dropped_feature_group_columns`, the key
the backtest records in its config). `train` and the weekly run check the group names before any
data loads. A group that matches no column of the data drops nothing, and every run logs it as a
warning, so a build without a group's columns does not quietly train differently from the arm.
Leaving out `strength` on a build that has the strength snapshot but no ported ratings would
train on neither. A ratings column null in every row, as a build without play-by-play leaves it,
counts as missing. `train` stops there; the weekly run stops right after the data refresh or its
skip, before it creates the run directory (a dry run that would refresh the data first only
warns), and stage 1 and the final fit check again; the backtest warns. The message names the
two remedies: rebuild the dataset (`nfl-predictor data`), or pass a group list that keeps
`strength`.

`--wf-n-estimators` overrides the XGBoost tree budget for the run (every in-season fit runs the full
budget), which is how the budget itself is measured against the default. The six-season ladder
measured with it (`200`, `400`, `598`) is recorded in `.agents/benchmarks.md` under "Tree-budget
ladder".

Season weighting was measured on the current `pbp`-default build
(`data/completed_games_ml.m54_flip_through_2025.csv`): nine six-season arms (`2020-2025`, from
week 1, `auto`, `market_anchor` on, default `200`-tree budget) at seeds `42` and `7`, run
directories `models/wf_m55_8_2020_2025_*/`, independently reviewed in
`models/wf_m55_8_review/INDEPENDENT_REVIEW.md`. Weeks 3-18 (`1423` games; market Brier `0.2095`):

| setting | det Brier, seed 42 / 7 | margin MAE, seed 42 / 7 | two-seed det Brier vs unweighted |
| --- | --- | --- | --- |
| unweighted | `0.2107` / `0.2096` | `9.9361` / `9.9105` | - |
| half-life 4 | `0.2115` / `0.2118` | `10.0051` / `10.0121` | `+0.0015` `[-0.0001, +0.0031]` |
| half-life 8 | `0.2107` / - | `9.9673` / - | - (one seed) |
| half-life 16 | `0.2102` / `0.2099` | `9.9522` / `9.9299` | `-0.0001` `[-0.0010, +0.0009]` |
| half-life 32 | `0.2110` / `0.2098` | `9.9571` / `9.9304` | `+0.0002` `[-0.0006, +0.0010]` |

Half-life `4` loses (two-seed margin MAE `+0.0853` `[+0.0188, +0.1535]`, and worse beyond its
intervals over all weeks); half-lives `16` and `32` tie unweighted. Since `0.18.0` the weekly run
trains unweighted in both its walk-forward stage and its final fit. The same arms show that on six
seasons re-seeding alone moves Brier by up to about `0.0013` and pick accuracy by about `0.008`,
so a single-seed difference that small is not a result.

Evaluation rule: "Model selection is based on time-aware walk-forward evaluation; random CV is not
authoritative." Season-blocked CV is used for hyperparameter tuning only; walk-forward remains the
source of truth.

### Comparing two runs

`nfl-predictor compare` rescores two walk-forward runs from their fold checkpoints (never from a
run's `metrics_report.json`) and compares them game by game on the same games:

```bash
nfl-predictor compare \
  --candidate models/<candidate_run> \
  --reference models/<reference_run> \
  --out-md models/<candidate_run>/compare.md
```

A run is a run directory (its `metadata.json` names the checkpoint directory and adds the dataset
hash, the git commit and the settings that differ) or a checkpoint directory. For week 1, week 2,
weeks 3-18 and all weeks it reports each run's deterministic Brier, log loss, pick accuracy, margin
and total MAE, confidence-pool points (ranked as in "Confidence pool rules" below) and market
Brier, and the candidate-minus-reference difference with a 95% bootstrap interval (5,000
resamples over games, over (season, week) blocks for pool points). The report names the unit: its
first lines say so, and the paired tables head the pool column `pool pts (week blocks)`. A pick is
the home side at `p >= 0.5`, as everywhere else ("Confidence pool rules" below).
A closing "Stability by season" section repeats each window per season, for every run and for the
paired difference; week 1 or week 2 of one season is a single week, so its pool-points difference
has no interval (`[n/a]`). Everything before that section matches the report `compare` wrote
before the section existed, apart from those unit labels, and the JSON is a strict superset of it
(each window gains a `seasons` entry). Repeat `--candidate` and `--reference` once per seed, in the
same order, to combine seeds: the per-game differences are averaged over the seed pairs before the
bootstrap.
By default each run's market Brier and its det - market interval use that run's own
`market_home_win_prob`. `--market-from <run>` (a run or checkpoint directory, given once) takes
it from another run instead, matched on `game_id`, for every compared run: a run anchored to the
pick-time line is then also measured against the closing line from a stored-line run on the same
games (run it once without the option for the pick-time yardstick). The model columns do not
change, and each run's market Brier is still scored on the run's own results. It stops with exit
code 2 when the option is repeated, or when the market run lacks a compared game, repeats one or
has no market value for one. The report says which market it used (JSON key `market_source`,
with the market run's provenance) and counts the compared games whose result differs in the
market run (`outcome_mismatches`, a printed warning when nonzero: separate builds can carry a
revised score); without the option the output is unchanged.
A last "Settings versus production" section gives each run's differences from today's production
weekly run, as the walk-forward report does (JSON key `settings_versus_production`, by run),
rebuilt from the run's `metadata.json` and its dataset's columns. Nothing is inferred from
today's defaults: an older run that recorded only its XGBoost overrides has the rest listed as
not recorded. A run with no metadata, whose dataset is gone, or whose recorded configuration no
longer loads (a retired calibration method) says why instead.
It reproduces the independent task 55.8 rescore exactly (`.agents/m60/verify_compare.py`) and
replaces the per-run `compare_to_benchmark.py` copies under `models/`.

## Weekly workflow (canonical)

The canonical "do everything for this week" entrypoint is:

```bash
nfl-predictor weekly --help
```

### High-level stages

1. (optional) refresh data (`nfl-predictor data --incremental` with the shipped config, which
   reuses unchanged finished seasons)
2. walk-forward of the production configuration over the recent seasons, so the run reports how
   production would have scored against the market (stage 1)
3. train the production configuration, the model the week's picks come from (stage 2)
4. generate weekly predictions + betting outputs + (optional) power rankings

The weekly run has one production configuration and chooses nothing by itself: the
probabilities it submits are the deterministic floor (`Phi(margin / sigma)`, with sigma
estimated from the reference runs' and stage 1's earlier out-of-fold errors as in "Win
probability" above, and no fitted calibrator or market blend), from a model trained with
`--wf-market-mode` (`hybrid` by default: market lines as features and the model anchored to
them; `features`, `anchor` and `none`, no game lines, are the others).

Notes:

- `--wf-*` flags control the stage-1 walk-forward. Stage 1 starts at week 1 (`--wf-start-week`,
  default `1` in the weekly run), so the errors of weeks 1 and 2 reach the final fit's sigma pool;
  `nfl-predictor backtest` keeps its own default of week 3. `--wf-market-mode` also sets the final fit's
  market mode, and `--wf-n-estimators`, `--wf-max-depth` and `--wf-learning-rate` (the tree
  budget, depth and learning rate) train both stage 1 and the final fit, so stage 1 measures the
  model the weekly run submits. With `--tune`, the tuned values replace them in the final fit.
- `disable_feature_groups` in the config (`--disable-feature-groups`, comma-separated, on the
  command line) names the feature groups that stage 1 and the final fit both leave out, removing
  the columns `nfl-predictor backtest --disable-feature-groups` removes for those groups (see
  "Backtesting"). It defaults to `[strength, own_elo, own_elo_wt]`, the strength snapshot and
  both own-Elo presets, and an empty list (`[]`, or `--disable-feature-groups ''`) leaves no
  group out. An unknown group stops the run before the data refresh, and the model's
  `metadata.json` records the groups and the columns they removed.
- `--train-*` flags control the **final training** fit that produces the weekly outputs.
- `--xgb-*` flags control XGBoost runtime and apply to both the walk-forward and final training.
  `--xgb-device` (`xgb_device` in the config) defaults to `auto`: the GPU when one is usable, else
  the CPU. It is resolved once per run, so stage 1 and the final fit share one device, and the
  model's `metadata.json` records it as `xgb_device`.
- Outputs are written under the run directory (default: `models/<run_id>/`) unless `--output-dir` is
  provided.
- Stage 1 is resumable: its finished weeks are saved under `wf_compare/wf_folds/` in the run
  directory. Its summary row is written as `wf_compare.csv` and `wf_best.json`, and its per-week,
  per-season and overall metrics with the reliability table as
  `wf_compare/wf_candidate_<key>.json`.
- `--data-collection-args` forwards extra arguments to the data refresh stage as one
  shell-quoted string, for example
  `--data-collection-args "--min-season 2010 --stat-prior-blend-games 4"`. It is split
  shell-style and handed to the data refresh (`nfl-predictor data`) as its argument list; when it is
  unset the refresh runs with that command's own defaults, a full rebuild. The same value is a
  config key (`data_collection_args`), and the shipped `config/weekly_run.yaml` sets it to
  `--incremental`, so the weekly refresh reuses each unchanged finished season (see "Data
  collection (Polars + nflreadpy)" above) and writes the same files a full rebuild writes. A
  value given on the command line replaces the configured string rather than adding to it: keep
  `--incremental` in it to keep the reuse
  (`--data-collection-args "--incremental --min-season 2010"`). A file read with `--config`
  replaces the shipped one, so it runs a full rebuild unless it sets `data_collection_args`
  itself. The web UI's weekly job passes its form's values as options over the shipped file, so
  its refresh is incremental unless its "ETL arguments" field replaces the string.

### How postseason games enter today

This is the current behavior, recorded for reference; none of it is a recommendation.

- Walk-forward and training filter to `game_type == "REG"` by default: `--include-postseason`
  and `--wf-include-postseason` are off, and `--postseason-weight` is `1.0`.
- The shipped `config/weekly_run.yaml` now keeps those regular-season defaults for the weekly run:
  `wf_include_postseason: false` and `include_postseason: false`. It still ships
  `postseason_weight: 1.3`, but that weight is inert unless postseason training is explicitly
  enabled, and `power_rankings_include_postseason: false` (the code default too) keeps postseason
  games out of the rankings step.
- The schedule-adjusted strength composite built in ETL never includes postseason games.
- The power rankings default through-week is the week before the prediction week, clamped to the
  last regular-season week when the prediction week is postseason, because strength snapshots stop
  one week after the regular season. Pass `--power-rankings-through-week` to override it.

### Authoritative weekly workflow (runs, in this order)

1. Refresh data (ETL)

   ```bash
   nfl-predictor data
   ```

   The bare command rebuilds every season. `nfl-predictor weekly` runs this step itself with
   `--incremental`, which writes the same files and reuses each unchanged finished season.

2. Canonical evaluation + model selection (walk-forward)

   ```bash
   nfl-predictor backtest --help
   ```

3. Train + predict for the upcoming week (writes predictions + artifacts)

   ```bash
   nfl-predictor train --help
   ```

4. Power rankings + projected standings

   ```bash
   nfl-predictor rankings --help
   ```

### Outputs and conventions

- Run artifacts (model/metrics/metadata/feature importance) land under `models/<run_id>/` by
  default.
- Weekly prediction outputs live next to the input prediction file (e.g., `data/predict/`).
- `metadata.json` includes dataset fingerprint, the XGBoost device the model trained on
  (`xgb_device`, read from every head: `mixed` when a head fell back to another device), the
  feature groups it was trained without and the columns they removed (`disabled_feature_groups`,
  `dropped_feature_group_columns`), tuned params, and Optuna summary when tuning runs.
- Power rankings outputs:
  - `power_rankings_season_XXXX_week_YY.csv`
  - `projected_standings_season_XXXX_week_YY.csv`
  - `projected_division_standings_season_XXXX_week_YY.csv`
- Power rankings measure **current-season** strength by default (`--method composite`): how strong
  each team is going into the next week. Teams are ranked on the ETL's schedule-adjusted strength
  composite for the week after `--through-week`, read from `data/strength_snapshots.csv`. That
  snapshot is solved only from games before the week it describes and has a row for every team on
  the schedule, so a team on a bye is ranked exactly and no later result reaches a historical
  rerun. The composite (a weighted mean of within-week z-scores) is converted to points with the
  week's own SRS slope, then to a win probability against an average team through the model's
  margin curve, and placed on the 1-10 and 0-10 scales. Each row also carries the composite,
  `points_vs_average`, the components (`adj_off_*`, `adj_def_*`, `st_rating`, `adj_srs`) and
  `strength_games_played`. A higher `adj_def_*` is a better defense.
- The composite weights per-snap passing and rushing EPA (offense and defense) and special teams;
  wins, losses and points are not in it, so a head-to-head result moves it only through that
  game's EPA. Early in a season it leans on last season: each component blends the in-season
  solve with the previous season's full-season solve regressed by one third, weighted
  `games / (games + 4)`, so after two games last season still sets most of the order (an open
  follow-up in `.agents/TODO.md`). The trained model does not set the ranking; it supplies only
  the projected standings.
- `--method bradley_terry` ranks on a Bradley-Terry fit to game results instead: the current and
  previous season only (`--ratings-window-seasons 2`), prior-season games weighted `0.25`
  (`--ratings-prior-season-weight`), completed games scored by margin (`--ratings-target margin`),
  and the model's forecasts for future games kept out of the fit (`--ratings-include-future` is
  off). `--legacy-franchise-fit` restores the older all-seasons, equal-weight **franchise** ranking,
  which describes a club's history more than its current team, and implies this method.
- `--method ported`, `model` and `blend` rank on a rating in points known before the week after
  `--through-week` (`src/nfl_predictor/reporting/rating_sources.py`). `ported` is the ported team
  rating the ETL publishes: each team's pre-week value from the game rows of `--data-ml`, and the
  weekly history under `--ratings-dir` (`data/ratings`) for a team on a bye. `model` is a ridge fit
  of team strengths and a home term to a walk-forward's pre-game predicted margins for the
  season's earlier games (`--model-predictions`: run or fold checkpoint directories that cover
  every completed regular-season game before the week; in week 1, the previous season's last fit
  carried forward by the factor the ported prior applies to a team rating). `blend` is
  `--blend-weight` times the ported rating plus the rest times the model rating. Each is mapped
  onto the scales like the composite and published with `team_rating`, `points_vs_average`,
  `rating_week` and `rating_source`. The rankings command's default stays the composite; the
  weekly run's is the blend (below).
- Projected standings are the same under every method: current record plus the model's win
  probabilities for the remaining games.
- `nfl-predictor weekly` accepts the same options (`--power-rankings-method`,
  `--power-rankings-strength-snapshots`, `--power-rankings-ratings-dir`,
  `--power-rankings-model-predictions`, `--power-rankings-blend-weight`, `--ratings-*`,
  `--legacy-franchise-fit`) and writes the same files. It ranks on the even blend by default
  (`power_rankings_method` left unset ranks on `blend`, with `power_rankings_blend_weight: 0.5` in
  `config/weekly_run.yaml`): the user chose it after the rankings measurement, and `0.5` is the
  weight its pre-registered rule picks for 2026 (`.agents/sos_port/decisions/power_rankings.md`,
  decision 12). Without `--power-rankings-model-predictions`, the model rating reads the ranked
  season from the run's own stage-1 fold checkpoints (`wf_compare/`, whose `metadata.json` names
  them; a resumed run that reuses stage 1 writes it when missing, finding the folds under
  `wf_compare/wf_folds/` by name or, for a stage 1 run on other code or library versions, by the
  one `manifest.json` that records the production configuration and the data's row and column
  counts; with none or several matching it writes nothing and warns to rerun with
  `--no-resume`) and earlier seasons, which
  only week 1's carried-forward fit reads, from the `floor_sigma_reference_runs`, averaged per
  game in the order listed as one source; later weeks read the stage-1 checkpoints alone, so a
  missing reference run blocks only week 1. Explicit paths replace both.
  `power_rankings_method: composite` restores the composite, and `--legacy-franchise-fit` with
  no method set implies Bradley-Terry; with any other method set, on the command line or in the
  config, the run refuses to start. When the snapshot or a rating source for the requested week
  is missing or does not cover the games, it skips the rankings with an error naming what is
  missing, ranks on no other method in their place, and still writes the projected standings;
  the run still exits `0`, and its reports-stage marker records the skip, so `--resume` runs
  that stage again.

**Moneyline edges, split at the spread.** The betting report's moneyline edge (`edge_home_prob`) is
the model's home win probability minus the market's no-vig moneyline probability. The weekly run
passes the report its floor sigma, and the report splits that edge in two:
`market_home_prob_spread` is the floor's curve at the market's margin, `Phi(-home_spread / sigma)`;
`edge_beyond_spread_home_prob` (the model's probability minus that) is what the model adds beyond
the spread; and `spread_moneyline_gap_home_prob` (that minus the moneyline probability) is the gap
between the spread and the moneyline, with no information from the model's margin. The two parts
sum to the edge. The confidence score and the action label still read the whole edge. The three
columns are empty for a game without a spread, or when the report is built without a sigma. Like
the rest of the report, they are diagnostics, not a profitability claim.

**Totals are diagnostic-only.** The total (over/under) columns of the betting report
(`total_value_side`, `total_edge_points`) come from the model's total head. Since version `0.6.2`
that head learns again (before, it stopped after one tree and predicted about 44 points for every
game), but in the 2023-2025 walk-forward it still trails the market's own total line: in weeks 3-18,
total MAE is `10.3152` in the production configuration (no market anchoring) and `10.2295` with
anchoring, against `10.0847` for the line itself. Treat an over/under lean as a diagnostic, not a
betting signal. Spreads, moneylines and win probabilities are unaffected.

Model selection hierarchy (default):

- Primary: deterministic probability quality and deterministic-minus-market intervals.
- Secondary: confidence pool expected points and stability.
- Tertiary: margin/total MAE (plus market-relative residual MAE when anchoring).

Metrics reports include a summary table (with metric priority + direction), plus optional
diagnostics such as season win totals (expected vs actual) and calibration drift by season/week.
Walk-forward reports also record the evaluation window, the calibration method, and any excluded
incomplete seasons.

## Command line

Every task runs through one command, `nfl-predictor <command>` (installed into `.venv/bin` by
`uv sync`; `python -m nfl_predictor <command>` is the same). `nfl-predictor --help` lists the
commands by group, and `nfl-predictor <command> --help` shows a command's options:

| group | commands |
| --- | --- |
| weekly | `weekly` (refresh, walk-forward of the production configuration, train, predict, reports; resumable, JSON/YAML config) |
| research | `backtest` (walk-forward, the benchmark), `compare` (paired comparison of two walk-forward runs), `explain` (SHAP attribution for a saved model), `checkpoints` (read-only listing of walk-forward checkpoints) |
| data | `data` (the ETL), `validate` (`--live` compares against the schedule), `leakage-audit`, `lines`, `build-week` |
| models by hand | `train`, `predict` (`--model-in`), `rankings` |
| web | `web`, `users` |

The code lives in the package: the weekly run in `src/nfl_predictor/weekly_run/`, the other
commands in `src/nfl_predictor/cli/`. The per-module forms (`python -m nfl_predictor.data_collection`,
`python -m nfl_predictor.ml_model` and the others) keep working. The old `scripts/<name>.py`
paths were removed in `0.27.0`; `scripts/gate.sh`, the check CI runs, is the one script left.

### Reproducing an old run

A run records the git commit it ran on (`git_commit_hash` in its `metadata.json`), and its
`launch.sh`, when it has one, names the commands as they were then, including `scripts/<name>.py`
paths and option spellings that no longer exist. To rerun it exactly, check out that commit in a
separate worktree with its own environment, and run the launcher as written from there:

```bash
git worktree add ../nfl-predictor-old <commit>
cd ../nfl-predictor-old
uv sync                                    # the environment that commit locked
ln -s ../nfl-predictor/data data           # the datasets are not in git
mkdir -p models/<run_id>
cp ../nfl-predictor/models/<run_id>/launch.sh models/<run_id>/
bash models/<run_id>/launch.sh
```

A launcher changes to the repository two levels above itself, so it runs inside the worktree and
writes its outputs there. Check that the dataset fingerprint in the new run's metadata matches the
old one. Remove the worktree with `git worktree remove ../nfl-predictor-old` when done.

`weekly` config example (JSON):

```json
{
  "wf_eval_last_n_seasons": 3,
  "wf_market_mode": "hybrid",
  "predict_path": "data/predict/week_03_games_to_predict.csv"
}
```

Run it with:

```bash
nfl-predictor weekly --config path/to/weekly_run.json
```

Every XGBoost command (`backtest`, `weekly`, `train`) defaults to `--xgb-device auto`,
which trains on the GPU when the XGBoost build has CUDA and a usable GPU is present, and on the
CPU otherwise. Pass `--xgb-device cpu` to force the CPU; the accepted values are `auto`, `cpu`,
`cuda` and `cuda:N` (`gpu` means `cuda`), and anything else is a usage error. A CUDA request
without a usable GPU runs on the CPU with a warning. A CUDA error during a fit retries that fit on
the CPU with a warning, and the saved model's metadata records the device it was fitted on; in a
walk-forward the run stops instead of checkpointing that week, so one checkpoint directory never
mixes devices, and rerunning resumes from the weeks already saved.

If you see great performance on the exact data a model trained on, that is not evidence the model
generalizes. Prefer holdout and walk-forward metrics.

## Web UI

A FastAPI backend (`src/nfl_predictor/api/`) and a React single-page app (`web/`) browse the
project's outputs and, for admins, run its jobs from a browser. The backend indexes
`models/*/metadata.json`, lets an admin mark one run **active**, and serves that run's
predictions, confidence picks, betting table (totals are never actionable), power rankings,
model metrics and data status; the Jobs pages run the ETL, a lines-only refresh
(`src/nfl_predictor/lines_refresh.py`), future-week inputs (`src/nfl_predictor/week_builder.py`),
training, prediction, walk-forward and the reports as streamed background subprocesses. A week
predicted ahead with the Predictions page's Generate action lands in the active run beside the
run's own week, and both stay selectable; once a newer weekly run predicts that week as its own,
the newer run's predictions stand for it.

The Pool page (`src/nfl_predictor/api/pool/`, routes under `/api/pool`) turns a week's
predictions into pool entries and never changes a probability, pick or rank:

- **Confidence sheet**: the picks with the points to assign, read from the existing confidence
  ranks (ranks that are not the unique `1..N` are reported, not repaired), the expected points,
  the lock (the week's first kickoff, since picks are single-shot) with a warning when the
  predictions were written after it, and, once games end, the points won under the pool rules
  below (final scores from the predictions file or `data/all_data.csv`). When the predictions were
  rewritten after the lock (a lines refresh chains a new prediction) and the weekly run's picks
  file was written before it, the sheet, its points and the picks download come from that picks
  file, and the newer ranks show only as notes. Copy it, or download it as CSV
  (`GET /api/predictions/picks?format=csv`).
- **Tiebreakers**: whole-point total, home and road entries for any two games (the latest Sunday
  and Monday kickoffs preselected, since the pool names its games each week), with p10 to p90
  bands from the quantile models, and the week's highest and lowest team scores, estimated as
  the extremes of normal team scores rather than the best and worst projections.
- **Survivor**: one team a week, each team once, a loss or a tie ending the entry. For each
  possible pick this week, every later week with predictions is assigned exactly (the
  assignment of weeks to distinct teams that maximizes the chance of surviving them all,
  `scipy.optimize.linear_sum_assignment`), so a team worth more later is saved; the next few
  weeks show as near-term picks. Weeks without predictions are listed; predict them first to plan
  for them. The teams already used are kept in the browser, per season.

The team and quarterback pages (`src/nfl_predictor/api/ratings/`, routes under `/api/ratings`)
read each season's tables from `data/ratings/{season}_{kind}.parquet`, with the kinds and columns
of the `nfl-sos-ratings` files that `src/nfl_predictor/ratings/` and
`src/nfl_predictor/season_stats/` were ported from. The ETL writes those tables
(`nfl-predictor data`: the Jobs page's Full ETL rebuild, or the weekly run's refresh); while the
folder holds none, the pages say where they look for them and to run the ETL. Every column's
label, description and shading direction comes from the metric registry; green shades better or
worse, slate shades context such as schedule strength, deeper for tougher. They only read, and a
season built without a table hides that section.

- **Teams** and **Quarterbacks**: a season's table by view (ratings, per game, per play, season
  totals, and the opponents faced per game and per play) and registry category, with a search, the
  quarterback qualifier, and up to four rows compared side by side. Each row opens a page with the
  ratings and their ranks, the rank range across 1,000 redraws of the season's games, head-to-head
  chances, the rating by week (and, for a season in progress, the rank range by week), the season
  stats by view, the game log with each opponent's season ratings, and one row per opponent faced.
- **Schedules**: every completed team-season ranked on schedule strength (`sos`), softest and
  hardest first.
- **Rank ranges**: where every team or qualifying quarterback ranks across those redraws.
- **Head to head**: how often one team or quarterback was rated above another, and both side by
  side.
- The **Glossary** describes these stats from the registry beside the model's own columns.

Power Rankings stays the model's own team strengths and projected standings; it and the Teams page
link to each other.

```bash
nfl-predictor users create-user <name> --role admin   # once
nfl-predictor web                                     # http://127.0.0.1:8000
```

The frontend needs Node 26 (`source ~/.nvm/nvm.sh`) and pnpm;
`cd web && pnpm install --frozen-lockfile && pnpm run build` writes `web/dist/`, which the backend
serves. Configuration is by `NFLP_*` environment variables
(`NFLP_DATA_DIR`, `NFLP_MODELS_DIR`, `NFLP_STATE_DIR`, `NFLP_PORT`, ...). Launch jobs from a
server started without `--reload`: a reload restart marks running jobs failed. Details (including
when to use `--reload` and the Vite dev server), the check commands and the layout are in
`web/README.md`; the design and phase status are in `.agents/web_ui_plan.md`.

## Validation

Offline validation:

```bash
nfl-predictor validate
```

Live validation (may require network access):

```bash
nfl-predictor validate --live
```

Both read `data/all_data.csv` unless `--data-dir` points them at another dataset directory, and
both exit non-zero when the input data file is missing or the validation fails, so they are safe to
use in shell automation.

Canonical local validation sequence, as one command:

```bash
scripts/gate.sh          # add --web when web/ changed; --quick skips pytest
```

It runs the steps below and reports every one before exiting non-zero:

```bash
uv lock --check
uv sync --check --active
ruff format --check
ruff check
ty check                      # src/ and tests/, from [tool.ty.src]
pyright                       # src/ and tests/, from [tool.pyright]
python -m pytest
coverage report --include="src/*" --fail-under=90   # src/ alone; the total counts tests/
markdownlint .
nfl-predictor --help          # then every command's --help, listed by the front door
```

CI installs `markdownlint-cli` for the `markdownlint .` step; on a machine that has
`markdownlint-cli2` instead, the equivalent is:

```bash
markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings" "#web/node_modules" "#web/dist" "#.agents/*transcript*.md"
```

GitHub Actions (`.github/workflows/validation.yml`) sets up the environment, runs the
editable-install smoke check, and then runs `scripts/gate.sh` itself, so CI and a local run check
the same things; the frontend steps run there as a separate `web` job.

## Leakage audit

To detect obvious feature leakage patterns:

```bash
nfl-predictor leakage-audit \
  --data-path data/completed_games_ml.csv \
  --out-json models/leakage_audit.json
```

## Changelog

Release history lives in `CHANGELOG.md` and follows the [Common
Changelog](https://common-changelog.org/) format. The historical baseline is `0.1.0` from `main`.

Update `CHANGELOG.md` as work lands, one entry per fix, feature group, or changed default, each
under its own incremented `## [VERSION] - YYYY-MM-DD` heading at the top of the file. There is no
`[Unreleased]` section: bump the patch version for fixes and small additions and the minor version
for new feature families, changed defaults, or schema changes, and set `pyproject.toml` to the same
version in the same change, then run `uv lock` and
`uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)` (which keeps a local CUDA build of
LightGBM). Keep the change groups in this order:

- `Changed`
- `Added`
- `Removed`
- `Fixed`

Keep each change to a single imperative line and skip routine formatting noise. The project is
private, so versions are not tagged and no GitHub release is published;
`.github/workflows/release.yml` only runs when a `0.x.y` or `v0.x.y` tag is pushed.

## Artifacts

Training/backtests can write a run directory containing reproducible artifacts.

- Use `--run-dir` to write `model.joblib`, `metadata.json`, and (when evaluated)
  `metrics_report.json`.
- `feature_importance.json` records XGBoost importance per model head and per encoded column
  (`gain`, the average loss reduction per split; `total_gain`, the loss reduction summed over
  every split; `weight`, the split count), and per base feature under `base_features`: total gain
  and splits summed over the feature's one-hot columns, per head and `combined` over both heads.
  Margin/total fits (the weekly run and `nfl-predictor train`'s default model kind) also record
  `mean_abs_shap` (`schema_version` 3): the mean absolute SHAP value in points from XGBoost's
  exact TreeSHAP, over the final model's own tree-training rows (the `shap` block gives the row
  count and seasons; holdout rows are left out). A market-anchored head is
  explained on its output, the residual over the market line, so the value measures how far a
  feature moves the model's adjustment to the line. A base feature's value sums each row's signed
  contributions over its one-hot columns before the absolute value, and `combined` adds the
  margin and total heads. Its `measures` block describes each key. The web Model page
  ranks base features by combined mean |SHAP|; runs without SHAP fall back to combined total
  gain, and runs written before `schema_version` 2 to the per-split average summed over columns
  (which favors features with many categories), each labelled as such. A "Margin head only"
  switch ranks every feature by the same measure for the margin head alone, the head that sets
  the win probabilities and picks. `nfl-predictor explain`
  uses the same TreeSHAP values for one head, per encoded column, on a sample of any dataset.
- Metadata includes timestamp, dataset fingerprint/hash, key package versions, training config/CLI
  args, the resolved XGBoost device, the floor's recorded sigma (`floor_sigma`), feature list, the
  feature groups the model was trained without and the columns they removed
  (`disabled_feature_groups`, `dropped_feature_group_columns`), and tuning/early-stopping info
  (when used). `models/` and `optuna.db` are gitignored by default, so keep run artifacts local
  unless you copy them elsewhere.

## Confidence pool rules (implemented)

- Each week assigns unique confidence values `1..N` to each picked winner, least confident first.
  Confidence is `|p - 0.5|` rounded to 12 decimals (`CONFIDENCE_DECIMALS` in
  `src/nfl_predictor/ml/metrics.py`), and equal rounded confidences are ordered by `game_id`. The
  rounding makes mathematically equal confidences (a home favorite and a home underdog by the
  same spread) almost always tie instead of differing by floating-point noise. The weekly picks,
  walk-forward pool points, the prediction log and `nfl-predictor compare` all rank this way.
- The pick is the home team when the unrounded home win probability is at least `0.5`
  (`p >= 0.5`, so an exact coin flip picks home), else the away team (`picks_home` in
  `src/nfl_predictor/ml/metrics.py`); the walk-forward, the training pool summaries, the weekly
  `predicted_winner` and `nfl-predictor compare` all use this rule.
- The weekly picks and ranks use the unrounded probability, as the walk-forward does: the
  published `home_win_prob` and `away_win_prob` are rounded to 4 decimals, but a game published
  at `0.5000` still picks the side its unrounded probability favors, two games that share a
  published value still rank by their real difference, and the published `confidence_strength`
  is the unrounded confidence (rounded to the 12 decimals above), so it agrees with the rank.
- Max weekly points: `N*(N+1)/2`.
- Realized points: `sum(confidence_value * 1[pick_correct])`.
- Ties count as incorrect.
- Picks are single-shot: made once, before the week's first game, with no in-week updates. The web
  app's confidence sheet scores a week by these rules and flags predictions written after the
  first kickoff.

## Score rounding / realism (optional)

When generating predictions, you can optionally post-process **display scores** without changing
training targets, win probabilities, or pool ranking logic:

- `--score-rounding none|int|half|nfl`

Use `nfl` to snap to common NFL score patterns for reporting.

## Repository layout notes

The package lives under `src/nfl_predictor/` and builds with the `uv_build` backend. `uv sync`
installs it in editable mode, so the tests and every command import it from `src/`; the data,
models and config directories stay at the repository root (`constants.ROOT_DIR`).

Some modules are split to keep files under lint `max-module-lines` limits while preserving legacy
import paths.

- Polars ETL helpers live under `src/nfl_predictor/utils/polars/` with a compatibility facade at
  `src/nfl_predictor/utils/polars_utils.py`.
- `src/nfl_predictor/data_collection.py` runs `nfl-predictor data`: the options, the source
  loading, the season loop (`process_season`) and the written files. Each week's game rows come
  from `process_week` in `utils/polars/week_rows.py`, and each week's schedule-adjusted strength
  table from `utils/polars/strength_table.py`.
- ML implementation lives under `src/nfl_predictor/ml/` with a compatibility facade at
  `src/nfl_predictor/ml_model.py`.
- `src/nfl_predictor/ratings/` holds the team and quarterback ratings ported from the sibling
  `nfl-sos-ratings` project (offense, defense, and special-teams ratings, head-to-head-excluded
  schedule strength, quarterback ratings, weekly histories, and bootstrap rank ranges), with a
  leakage-safe pre-week entry point for model features and the adapter from this repository's
  play-by-play (`pbp_inputs.py`). The ETL runs them through `utils/polars/ratings_build.py` and
  writes their tables to `data/ratings/`, which the web app's team and quarterback pages read;
  [its README](src/nfl_predictor/ratings/README.md) covers the methods, inputs, and parity checks.
- `src/nfl_predictor/season_stats/` holds the season statistics ported from the sibling
  `nfl-sos-ratings` project (team and quarterback game logs, season rows, head-to-head-excluded
  opponent profiles, win-probability bins) and its metric registry, with the stats catalogs
  generated from it. Its tables are whole-season display data, not values known before a week's
  games. The ETL builds them beside the ratings (`utils/polars/season_stats_build.py`) and writes
  them to `data/ratings/`, where the web app's team and quarterback pages read them with the
  registry; the model does not use them.
  [Its README](src/nfl_predictor/season_stats/README.md) covers the inputs, outputs, and parity
  checks.

## Implemented feature areas

All engineered features are defined so they apply to **every matchup**, not only end-of-season
games.

- Invariant-schema missing-data handling across seasons.
- Season-to-date record features (overall/division/conference), with the pre-2002 alignment for
  the three historical seasons before realignment.
- Divisional rivalry indicator, also season-aware across the 2002 realignment.
- Lookahead / next-week context features.
- Standings-based motivation proxy features (clinch/elimination proxies).
- Stadium metadata features (roof/surface/type, elevation, venue location).
- Head coach prior record features (career and team-specific).
- Play-by-play per-snap efficiency features (`constants.PBP_STATS`): offensive and allowed EPA per
  snap, EPA per dropback and per carry, success rates, explosive pass/rush rates, stuffed-run rate,
  early-down pass rate, snap volume, and special-teams EPA margin per play. Defensive metrics are
  named explicitly (`def_*` / `*_allowed`) rather than relying on the generic `opponent_` mirror.
  Counts and sums are carried through season-to-date aggregation and every rate is computed
  afterwards as a ratio of sums, never a mean of per-game rates; a count no published rate reads
  is not carried. The plays behind them (`utils/polars/pbp.py`, regular season only): a dropback
  is a play flagged `qb_dropback`, or a run play flagged `qb_scramble` (nflverse left the dropback
  flag off scrambles before 2006), never a kneel-down, spike or two-point try; a carry is a
  designed run, the `nfl-sos-ratings` rule: a rush attempt (`rush_attempt`) that is not a
  scramble, a kneel-down or a two-point try, and a play flagged as both a dropback and a rush
  counts once, as a dropback. A scrimmage snap (`offensive_snaps`, the per-snap denominators) is a
  dropback, a play flagged `rush`, a kneel-down or a spike. A run a penalty wiped out nearly
  always keeps the `rush` flag but not `rush_attempt`, so it is a snap but not a carry, and its
  EPA stays out of the rushing sums. Third-down, fourth-down and two-point counts read their flags
  on every play, and a down flagged both converted and failed counts once, as a conversion.
- Schedule-adjusted team strength (`constants.ADJUSTED_STRENGTH_STATS`): a pre-week snapshot
  solved for every `(season, week)` from a simultaneous ridge that estimates one offense and one
  defense coefficient per team plus a shared home-field term, on per-snap pass and rush EPA
  responses. Published per team as `adj_off_pass_epa_snap`, `adj_off_rush_epa_snap`,
  `adj_def_pass_epa_snap`, `adj_def_rush_epa_snap`, a point-margin SRS companion (`adj_srs`), a
  special-teams rating (`st_rating`), a standardized display composite
  (`adj_strength_composite`), and the games behind the solve (`strength_games_played`, per side
  only: the two teams have almost always played the same number of games, so its diff would be
  zero outside bye weeks). A **higher** `adj_def_*` value means a **better** defense: the solve
  models a team-game as `offense[team] - defense[opponent]`, so the defense coefficient is what
  suppresses the opponent's output.
- Schedule strength in two lenses (`constants.SCHEDULE_STRENGTH_STATS`): `sos_played_adj` and
  `sos_remaining_adj`, the mean pre-week composite of the opponents already faced and still to
  come; and `sos_played_raw`, the one-hop companion that profiles each faced opponent from only
  its games **against the rest of the league**, excluding every head-to-head game with the subject.
  Both are restricted to the regular season: the regular-season schedule is fixed before kickoff
  and is legitimately known, but the postseason bracket is an outcome of the season being
  predicted and must never reach a pre-week feature. `sos_played_raw` is therefore null through
  week 2, because a week-2 opponent's only prior game is the one against the subject.
- The team and quarterback ratings ported from `nfl-sos-ratings`
  (`constants.RATINGS_FEATURE_STATS`, built by `src/nfl_predictor/utils/polars/ratings_build.py`),
  the `ratings` feature group (every name starts with `ratings_`). Per side with a `_diff`
  companion: `ratings_offense`, `ratings_defense`, `ratings_special_teams` and `ratings_team`
  (points per game against an average team on a neutral field, from opponent-adjusted EPA) and
  `ratings_sos` (the mean team rating of the opponents played, each rated without the team's
  games), each fit only on the season's games
  before the row's week with the previous season's penalties and a preseason prior from earlier
  seasons; and `ratings_qb` (the starter's adjusted EPA per dropback, the starter every
  quarterback feature reads) with `ratings_qb_dropbacks`, his dropbacks this season before the
  week, the passers counted as the players file and the roster weeks before the week list them.
  Week 1 holds each team's preseason prior (from 2003; null before), no `ratings_sos`, and no
  quarterback rating with 0 dropbacks; `ratings_sos` is still null in week 2, when no opponent has
  played anyone else; 1999 is null throughout; a postseason row reads the whole regular season;
  a starter without a dropback yet this season has no rating and 0 dropbacks, an unidentified one
  nulls. The strength family above stays in the build beside it, so a walk-forward can compare
  the two with `--disable-feature-groups`; production trains on the ratings and leaves the
  strength family out.

- Quarterback per-dropback production for each side's starter (`constants.QB_PBP_STATS`, in
  `src/nfl_predictor/utils/polars/qb_stats.py`; the starter is the one of "Data collection": the
  play-by-play starter of a completed game, the named one otherwise), from that quarterback's
  regular-season dropbacks in every earlier week across teams and seasons (never the game's own
  week). Career EPA per dropback (`qb_dropback_epa`), CPOE (2006+), sack rate and ANY/A are shrunk
  toward the league with `K = constants.QB_PRIOR_DROPBACKS` pseudo-dropbacks,
  `(sum + K * league_rate) / (count + K)`, so a first start gets the league rate; the last
  `constants.QB_RECENT_GAMES` games (`qb_dropback_epa_recent`, `qb_any_a_recent`) are shrunk
  toward the career rate the same way; `qb_history_dropbacks` tells the model how much evidence
  stands behind them. A scramble, which nflverse files without a passer id, is credited to its
  rusher, the quarterback who scrambled.

- This repository's own Elo, computed from game results with FiveThirtyEight's published method
  instead of read from `data/qb_elos.csv`. The ETL publishes both of its presets on every game
  row (`src/nfl_predictor/utils/polars/own_elo_build.py`): `market_free`
  (`constants.OWN_ELO_COLUMNS`: `away_own_elo`, `home_own_elo`, `own_elo_diff`,
  `own_elo_home_win_prob`) and `win_totals` (`constants.OWN_WT_ELO_COLUMNS`, the same names with
  `own_wt_elo`), and records how it built them in `data/own_elo_metadata.json`. The ratings run
  over every game from 1999 on (the seeding, the quarterback model's first-start mean and the
  rolling home field count from its first season), upcoming games included; a run that builds
  fewer seasons reads the earlier schedules from the cache and names their starters from
  `qb_elos.csv`. `build_own_elo_features` in `src/nfl_predictor/utils/polars/elo_ratings.py`
  computes one preset: per game, each side's pre-game rating with its quarterback adjustment, away
  minus home, and the home win probability, plus metadata naming the preset. Every rating and
  league-wide value comes from earlier games, with one exception shared with the `qb_elos.csv`
  columns and the quarterback family: a completed game's quarterback adjustment uses that game's
  play-by-play starter (see "Data collection"), known only at kickoff, while an upcoming game uses
  the projected starter the ETL names (`away_qb`/`home_qb`), else the team's last one. The
  framework is 538's (K 20, home field 65, a third of the way back to 1505 each offseason, the
  margin-of-victory multiplier), and `tests/test_elo_ratings.py` reproduces its published ratings
  and probabilities for 2017-2019 to floating precision. `EloSettings` switches on 538's later
  additions one at a time: rest (`elo_game_context.py`, 25 points off a bye week), travel (a
  55-point base plus 4 points per 1,000 miles of the visitor's trip), the playoff boost, a rolling
  ten-season home field, the quarterback adjustment (538's VALUE model on play-by-play box scores,
  `elo_quarterbacks.py`: passing from the quarterback's dropbacks, rushing from every rush attempt
  credited to his rusher id, two-point tries out) and the preseason pull toward win-total ratings
  (market information; its weight is fitted per season on earlier seasons). The `market_free`
  variant uses `MARKET_FREE_SETTINGS` (travel, rolling home field, quarterback) and `win_totals`
  uses `WIN_TOTAL_SETTINGS` (the same plus the win-total pull): the components a forward selection
  on the 2007-2025 regular season kept. `.agents/sos_port/ELO_EVAL.md`, generated by
  `.agents/sos_port/elo_eval.py`, holds the numbers, including the presets scored with each
  team's previous starter instead of the actual one.

The strength family is ablatable as the `strength` feature group, which production leaves out by
default together with both own-Elo groups. Its early-season prior blend weights the in-season solve
`games / (games + K)` against the previous season's final snapshot regressed by
`constants.WEEK1_REGRESSION_FACTOR`; at ETL time `--strength-prior-blend-games` sets `K` (default
`constants.PRIOR_BLEND_GAMES`, `4`) and `--no-strength-prior-blend` ablates the blend. Both change
feature values and the strength snapshot file behind the composite power rankings, so compare them
with two dataset builds. The quarterback family is the `qb` group, and the 18 Elo columns from
`data/qb_elos.csv` are the `elo` group (market-informed, see "Market integration"). The own Elo's
market-free columns are the `own_elo` group and its win-total columns the `own_elo_wt` group,
which is market-informed too and left out by default; a run with no market information also
leaves out `elo` (`--disable-feature-groups strength,elo,own_elo_wt`, or
`strength,elo,own_elo,own_elo_wt` for no Elo at all). The three are disjoint, so one build
carries all of them. The rare-event noise family is the `rare_events` group (`special_teams_tds`,
`def_fumbles`, `fumble_recovery_tds`, `2pt_conversions`, `def_safeties`, `def_tds`). The
`next_opponent_identity` group drops only the `away_next_opponent_abbr` and
`home_next_opponent_abbr` pair, which would reach the model as one one-hot column per team on each
side, and keeps the rest of the lookahead family, `*_next_opponent_win_pct` included. No group
overlaps another.

The model leaves that pair out by default. It is on the pruning list
(`constants.PRUNED_FEATURE_COLUMNS`, the known-weak columns every fit leaves out: `train`,
`backtest` and the weekly run, and so the models `predict` loads), so a default run fits exactly
the matrix that `--disable-feature-groups next_opponent_identity` gave before, and only the
backtest's `--disable-pruning` brings the pair back. The ETL still writes both columns.

## Open work

Active tasks (milestones + guardrails) are tracked in `.agents/TODO.md`, and completed milestones
live in `.agents/ARCHIVE.md`; both are tracked in git alongside the cross-repo feature crosswalk
(`.agents/feature_crosswalk.md`) and the next-session handoff prompt. The current direction is
stronger feature engineering around play-by-play EPA and schedule-adjusted team strength, porting
the head-to-head-excluded opponent-profile method from the sibling `nfl-sos-ratings` project and
the simultaneous ridge that generalizes it, with XGBoost margin/total remaining the primary model.

## Development notes

- ETL and feature engineering run in Polars.
- All NFLverse data is pulled via `nflreadpy`.
- Logging uses the project logger; avoid `print`.
- Formatting is enforced via Ruff format (line length 100).
- Linting and import sorting are enforced via Ruff (includes isort rules).
- Type checking runs through both Pyright and Ty; both are required local validation gates.
- Development dependencies are declared in `pyproject.toml` and synced via `uv.lock`.
- Claude Code subagents are defined in `.claude/agents/`: `implementer` works in its own git
  worktree under `.claude/worktrees/` (branched from the current `HEAD`, per
  `.claude/settings.json`), and `reviewer` checks finished work it did not produce. Both follow
  `AGENTS.md`. Inside a worktree, run `uv sync` once, then run the gate with
  `VIRTUAL_ENV="$PWD/.venv"` so it uses the worktree's environment.

For users reading this documentation: commands are shown assuming your project virtual environment
is already activated. Agent-specific files keep the fully qualified `.venv/bin/...` forms for
automation reliability.

See the Validation section above for the canonical local validation sequence.

## Safety and claims

This project outputs statistical forecasts and backtest metrics. It does not guarantee accuracy,
profit, or betting success.
