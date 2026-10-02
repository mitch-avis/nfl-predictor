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

Note: the `data/` directory is gitignored by default; generate it via the data collection step
above. Note: `*_ml.csv` files include model-ready engineered features.

Game rows are ordered newest first, with games on the same date ordered by `game_id`. Two runs
with the same code, locked dependencies and inputs, on the same machine with the same Polars
thread count (`POLARS_MAX_THREADS`), write byte-identical files.

Building each season's weekly feature rows is nearly all of the ETL's run time.
`nfl-predictor data --incremental` keeps each finished season's rows and strength snapshots under
`<data dir>/cache/etl_seasons/` and reuses them when nothing they are built from has changed; the
season in progress is always rebuilt, and every input is still loaded. An entry is keyed on:

- the source of the ETL modules (`data_collection.py`, `constants.py` and everything under
  `src/nfl_predictor/utils/`), the Polars, Polars runtime and NumPy versions, the Python
  version, the CPU architecture and the Polars thread count;
- the options that change a season's rows (`--min-season`, the two prior-blend switches,
  `--strength-prior-blend-games`, `--stat-prior-blend-games`, `--team-stats-source`,
  `--tr-stats-source` and `--line-source`);
- the loaded schedule, team stats and Elo rows from that season and every earlier one, and that
  season's and the previous season's TeamRankings, compared by value.

So a changed input row rebuilds its season and every later one, and an edit to the ETL code
rebuilds everything. A stale, damaged or unreadable entry is rebuilt and rewritten, never an error,
and the directory is safe to delete. An input with a column the by-value comparison cannot
render (list, struct, array, duration, binary or object) leaves its seasons uncached. Without the
flag, which is `nfl-predictor data`'s default, the ETL rebuilds every season and neither reads
nor writes the cache. The weekly run's refresh passes it (`data_collection_args: "--incremental"`
in `config/weekly_run.yaml`; "Weekly workflow (canonical)" below says how your own arguments
combine with it).

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
Before kickoff the current season has no play-by-play published at all; that is non-fatal, and the
ETL falls back to cache or continues without it.

The quarterback family needs `data/meta_data.csv`, a read-only copy of
`../nfeloqb/Other Data/meta_data.csv` made the same way as `data/qb_elos.csv`. It maps the Elo
quarterback names to GSIS ids, which are also the play-by-play passer ids. Without the file the
quarterback columns are null and the ETL logs a warning. Career rates need every earlier season, so
the ETL reads the cached play-by-play from 1999 even when `--min-season` is later.

TeamRankings data is cached under `data/<season>/` as week-level CSVs; enable debug logging to see
cache hits. Use `--timing` to log per-step runtimes and `--debug-logs` for detailed ETL diagnostics.
Use `--refresh-nflreadpy` to force refresh nflreadpy data even when cache exists.

Two flags choose where per-team-game stats come from; both default to `pbp` since 2026-09-21.
`--team-stats-source pbp` derives the box-score families (passing, rushing, penalties, first
downs, sacks and interceptions) from play-by-play and overlays them on the nflverse rows, so
nflverse still fills whatever play-by-play cannot derive; pass the flag with `nflverse` to use
the scraped/nflverse sources instead. `--tr-stats-source pbp` derives the eight situational
percentages (third down, fourth down, red zone and two-point, for and allowed) from the
play-by-play counts instead of the TeamRankings scrape; pass `scrape` to use the scrape.
TeamRankings supplies its ratings either way. The derived percentages use the scraped columns'
0-100 scale, and `red_zone_td_pct` divides touchdown drives by red-zone trips (drives that
reached the 20, from `fixed_drive`), which is what the scraped "red zone scoring %" measures --
not touchdowns per red-zone snap. `passing_epa` sums `qb_epa` (nflverse's own quarterback
EPA attribution) rather than `epa`, and the box-score counts and yardage use nflverse's own
`pass_attempt`/`rush_attempt` raw flags rather than `play_type`; both were verified against
nflverse team stats at 99%+ agreement on the columns they touch
(`models/pbp_vs_nflverse_m54_2/COMPARISON.md`). `fumbles`/`fumbles_lost` exclude special-teams
plays to match nflverse's offense-only fumble stat, with a residual gap on aborted-snap fumbles;
`2pt_conversions` still disagrees with nflverse on a meaningful share of team-games, but this is
not a derivation defect: nflverse's own team-stats table has a confirmed, systematic bug that
roughly doubles the count of successful two-point conversions on the games it gets wrong (see
`models/pbp_vs_nflverse_m54_2/COMPARISON.md`), so the play-by-play value is the correct one where
they differ. Both flags change feature values at ETL time, so compare them with two dataset
builds rather than with `--disable-feature-groups`.

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
heads, in training and in production alike.

- `stored` (the default): the nflverse schedule lines, one late, probably closing, snapshot.
  Missing moneylines are derived from the spread with the fixed conversion
  (`game_utils.spread_to_moneyline`: a normal curve at `SCORE_DIFF_STD_DEV` with a flat 5% vig),
  and future games without a line take SurvivorGrid's spread.
- `pick_time`: the line known when picks are made, from nfelo's market-lines file
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

  The file is rewritten several times a day. Each run downloads it, keeps the last good copy in
  `data/cache/nfelo/lines.csv` and every copy a run used in `data/cache/nfelo/snapshots/`, named
  by its SHA-256, and falls back to the cached copy (or, with none, to the stored lines) instead of
  failing. The run writes `market_lines_metadata.json` beside the datasets: the snapshot's URL,
  origin and hash, per-season game, match, opener and fallback-row counts, how many moneylines
  each season derived and how, and every season's map. Two diagnostic counts per season flag
  possible data errors without filtering anything: openers and upcoming nfelo lines on the other
  side of the stored line, both at least 3 points (`opener_sign_flips`,
  `upcoming_nfelo_sign_flips`). A `stored` build fetches nothing new and writes
  `{"line_source": "stored"}` to the same file, so a record never outlives the build it
  describes. The option changes feature values, so compare the two sources with two dataset
  builds, not with `--disable-feature-groups`.

## Data sources + missing data

This project is designed to keep an invariant output schema across seasons, even when some sources
are missing historically.

Primary sources:

- `nflreadpy` (NFLverse): schedules, results, team-level stats, and play-by-play.
- nfelo's market-lines file (`greerreNFL/nfelomarket_data`), only with `--line-source pick_time`.
- Local cached CSVs under `data/` for Elo/market data when present.
- `data/meta_data.csv` (copied from `../nfeloqb`) for the quarterback name-to-id bridge. A name
  missing from it falls back to the play-by-play passer name (`F.Last`) when that is unique; a
  quarterback still unmatched gets null quarterback features, and the ETL logs the unmatched rate.
- TeamRankings web scrape for select ratings and stats not available in NFLverse (see ETL logs).

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
  `--floor-sigma-reference-runs` for `train` (default: the two seeds of the GPU reference
  walk-forward, `models/step3_gpu_reference/l1_seed42` and `l2_seed7`, whose fold checkpoints
  are only read), with, in the weekly run, stage 1's errors before the predicted week at every
  `(season, week)` the reference runs have no rows for: the seasons after the reference and the
  predicted season's earlier weeks (where both have a week, the reference's errors are used).
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
parser and `config/weekly_run.yaml` merged with the code defaults, for both of its halves: stage
1's walk-forward and the final fit. For each half, `differences` lists every model-affecting
setting where the run's resolved value differs (the XGBoost device and parameters, built by the
resolver training uses, market features, anchoring and transform resolved against the dataset's
lines, recency weighting, calibration, postseason games and their weight, disabled feature
groups, trend-feature ablation, pruning, holdout seasons, tuning). A setting one side applies
implicitly is compared at that value: production never drops pruning, feature groups or trend
features, and a walk-forward fold holds out no season, never tunes and weights postseason games
like any other. What remains on one side only is listed under `run_only` or `production_only`
(every `tune_*` option when production tunes), and `not_recorded` names production settings an
older run's metadata lacks (its resolved XGBoost parameters, which the backtest now records as
`config.xgb_params`); settings an older run recorded that no longer exist are listed under
`retired` with their values. A half reads "no model-affecting differences" only when all four
lists are empty and nothing is retired; the run then trains what production trains. Scope (the
scored seasons and weeks, the seed, the dataset and checkpoint paths, production's dataset and
data-collection arguments) and runtime settings that change no prediction (XGBoost threads and
verbosity, quantile models) are listed apart, never as differences.

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
in the run's metrics report.

```bash
nfl-predictor backtest --disable-feature-groups pbp
```

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
resamples over games, over weeks for pool points).
A closing "Stability by season" section repeats each window per season, for every run and for the
paired difference; week 1 or week 2 of one season is a single week, so its pool-points difference
has no interval (`[n/a]`). Everything before that section is byte-identical to the report
`compare` wrote before the section existed, and the JSON is a strict superset of it (each window
gains a `seasons` entry). Repeat `--candidate` and `--reference` once per seed, in the same order,
to combine seeds: the per-game differences are averaged over the seed pairs before the bootstrap.
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
them).

Notes:

- `--wf-*` flags control the stage-1 walk-forward. Stage 1 starts at week 1 (`--wf-start-week`,
  default `1` in the weekly run), so the errors of weeks 1 and 2 reach the final fit's sigma pool;
  `nfl-predictor backtest` keeps its own default of week 3. `--wf-market-mode` also sets the final fit's
  market mode, and `--wf-n-estimators`, `--wf-max-depth` and `--wf-learning-rate` (the tree
  budget, depth and learning rate) train both stage 1 and the final fit, so stage 1 measures the
  model the weekly run submits. With `--tune`, the tuned values replace them in the final fit.
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
  (`xgb_device`, read from every head: `mixed` when a head fell back to another device), tuned
  params, and Optuna summary when tuning runs.
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
- Projected standings are the same under both methods: current record plus the model's win
  probabilities for the remaining games.
- `nfl-predictor weekly` accepts the same options (`--power-rankings-method`,
  `--power-rankings-strength-snapshots`, `--ratings-*`, `--legacy-franchise-fit`) and writes the
  same files. It skips the rankings with a warning when the snapshot for the requested week is
  missing.

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

**Totals are diagnostic-only.** The total (over/under) columns of the betting report
(`total_value_side`, `total_edge_points`) come from the model's total head. Since version `0.6.2`
that head learns again (before, it stopped after one tree and predicted about 44 points for every
game), but in the 2023-2025 walk-forward it still trails the market's own total line: in weeks 3-18,
total MAE is `10.3152` in the production configuration (no market anchoring) and `10.2295` with
anchoring, against `10.0847` for the line itself. Treat an over/under lean as a diagnostic, not a
betting signal. Spreads, moneylines and win probabilities are unaffected.

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
training, prediction, walk-forward and the reports as streamed background subprocesses.

```bash
nfl-predictor users create-user <name> --role admin   # once
nfl-predictor web                                     # http://127.0.0.1:8000
```

The frontend needs Node 26 (`source ~/.nvm/nvm.sh`); `cd web && npm ci && npm run build` writes
`web/dist/`, which the backend serves. Configuration is by `NFLP_*` environment variables
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
ruff format --check .
ruff check .
ty check .
pyright .
python -m pytest
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
version in the same change (then `uv lock` and `uv sync`). Keep the change groups in this order:

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
  (which favors features with many categories), each labelled as such. `nfl-predictor explain`
  uses the same TreeSHAP values for one head, per encoded column, on a sample of any dataset.
- Metadata includes timestamp, dataset fingerprint/hash, key package versions, training config/CLI
  args, the resolved XGBoost device, the floor's recorded sigma (`floor_sigma`), feature list, and
  tuning/early-stopping info (when used).
  `models/` and `optuna.db` are gitignored by default, so keep run artifacts local unless you
  copy them elsewhere.

## Confidence pool rules (implemented)

- Each week assigns unique confidence values `1..N` to each picked winner, least confident first.
  Confidence is `|p - 0.5|` rounded to 12 decimals (`CONFIDENCE_DECIMALS` in
  `src/nfl_predictor/ml/metrics.py`), and equal rounded confidences are ordered by `game_id`. The
  rounding makes mathematically equal confidences (a home favorite and a home underdog by the
  same spread) almost always tie instead of differing by floating-point noise. The weekly picks,
  walk-forward pool points, the prediction log and `nfl-predictor compare` all rank this way.
- The pick is the home team when the unrounded home win probability is at least `0.5`
  (`p >= 0.5`, so an exact coin flip picks home), else the away team (`picks_home` in
  `src/nfl_predictor/ml/metrics.py`); the walk-forward, the training pool summaries and the weekly
  `predicted_winner` all use this rule.
- The weekly picks and ranks use the unrounded probability, as the walk-forward does: the
  published `home_win_prob` and `away_win_prob` are rounded to 4 decimals, but a game published
  at `0.5000` still picks the side its unrounded probability favors, two games that share a
  published value still rank by their real difference, and the published `confidence_strength`
  is the unrounded confidence (rounded to the 12 decimals above), so it agrees with the rank.
- Max weekly points: `N*(N+1)/2`.
- Realized points: `sum(confidence_value * 1[pick_correct])`.
- Ties count as incorrect.

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
  afterwards as a ratio of sums, never a mean of per-game rates.
- Schedule-adjusted team strength (`constants.ADJUSTED_STRENGTH_STATS`): a pre-week snapshot
  solved for every `(season, week)` from a simultaneous ridge that estimates one offense and one
  defense coefficient per team plus a shared home-field term, on per-snap pass and rush EPA
  responses. Published per team as `adj_off_pass_epa_snap`, `adj_off_rush_epa_snap`,
  `adj_def_pass_epa_snap`, `adj_def_rush_epa_snap`, a point-margin SRS companion (`adj_srs`), a
  special-teams rating (`st_rating`), a standardized display composite
  (`adj_strength_composite`), and the games behind the solve (`strength_games_played`).
  A **higher** `adj_def_*` value means a **better** defense: the solve models a team-game as
  `offense[team] - defense[opponent]`, so the defense coefficient is what suppresses the
  opponent's output.
- Schedule strength in two lenses (`constants.SCHEDULE_STRENGTH_STATS`): `sos_played_adj` and
  `sos_remaining_adj`, the mean pre-week composite of the opponents already faced and still to
  come; and `sos_played_raw`, the one-hop companion that profiles each faced opponent from only
  its games **against the rest of the league**, excluding every head-to-head game with the subject.
  Both are restricted to the regular season: the regular-season schedule is fixed before kickoff
  and is legitimately known, but the postseason bracket is an outcome of the season being
  predicted and must never reach a pre-week feature. `sos_played_raw` is therefore null through
  week 2, because a week-2 opponent's only prior game is the one against the subject.

- Quarterback per-dropback production for the expected starter (`constants.QB_PBP_STATS`, in
  `src/nfl_predictor/utils/polars/qb_stats.py`): for `away_qb` and `home_qb`, from that quarterback's
  regular-season dropbacks in every earlier week across teams and seasons (never the game's own
  week). Career EPA per dropback (`qb_dropback_epa`), CPOE (2006+), sack rate and ANY/A are shrunk
  toward the league with `K = constants.QB_PRIOR_DROPBACKS` pseudo-dropbacks,
  `(sum + K * league_rate) / (count + K)`, so a first start gets the league rate; the last
  `constants.QB_RECENT_GAMES` games (`qb_dropback_epa_recent`, `qb_any_a_recent`) are shrunk
  toward the career rate the same way; `qb_history_dropbacks` tells the model how much evidence
  stands behind them. Scrambles are credited to the team-game's primary passer, because the
  play-by-play cache keeps the passer id but not the rusher id.
- Defense-adjusted quarterback EPA per dropback (`constants.QB_DEF_ADJ_STATS`, same module):
  each earlier game of the expected starter is credited with
  `qb_epa_sum + dropbacks * adj_def_pass_epa_snap`, the faced defense's pre-week strength value
  for that game's week (solved from games strictly before it), so EPA earned against a strong
  defense counts for more. The career (`qb_def_adj_epa`) and last-`QB_RECENT_GAMES`
  (`qb_def_adj_epa_recent`) rates use the same `K` shrinkage and windows as `qb_dropback_epa`.
  The coefficient is pass EPA per offensive snap and is applied per dropback as it stands, not
  rescaled by the pass share of snaps. A defense with no pre-week value counts as average: a
  season the run did not build (before its `--min-season`), and any team with neither an earlier
  game that season nor a previous-season value, which means every team in week 1 of the first
  season built, teams that have not played yet in 1999 weeks 2-3, Houston in 2002 week 1 (the
  expansion season), and every season's week 1 under `--no-strength-prior-blend`. Unlike the
  rest of the quarterback family these two columns therefore depend on the run's first season,
  and they move with `--strength-prior-blend-games`, because the `adj_def_pass_epa_snap` they read
  is blended with that `K`; the ETL logs how many quarterback games count a defense as average,
  and training reports the group's null cells and rows like the other groups.

The strength family is ablatable as the `strength` feature group
(`--disable-feature-groups strength`). Its early-season prior blend weights the in-season solve
`games / (games + K)` against the previous season's final snapshot regressed by
`constants.WEEK1_REGRESSION_FACTOR`; at ETL time `--strength-prior-blend-games` sets `K` (default
`constants.PRIOR_BLEND_GAMES`, `4`) and `--no-strength-prior-blend` ablates the blend. Both change
feature values and the strength snapshot file behind the default power rankings, so compare them
with two dataset builds. The quarterback family is the `qb`
group (`--disable-feature-groups qb`), and the defense-adjusted quarterback rate is the
`qb_def_adj` group (`--disable-feature-groups qb_def_adj`); with it disabled the model trains on
exactly the feature matrix of a dataset without those columns. The rare-event noise family is
the `rare_events` group (`special_teams_tds`, `def_fumbles`, `fumble_recovery_tds`,
`2pt_conversions`, `def_safeties`, `def_tds`). The `next_opponent_identity` group drops only the
`away_next_opponent_abbr` and `home_next_opponent_abbr` pair, which the model sees as one one-hot
column per team on each side, and keeps the rest of the lookahead family,
`*_next_opponent_win_pct` included. No group overlaps another.

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
