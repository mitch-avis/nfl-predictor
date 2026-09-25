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
    - [Splits: train, calibration, holdout](#splits-train-calibration-holdout)
  - [Modeling approach](#modeling-approach)
    - [Margin/Total targets (canonical)](#margintotal-targets-canonical)
    - [Win probability calibration](#win-probability-calibration)
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
repo's data, LightGBM trains faster on the CPU than with CUDA, so the CUDA build is optional.

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

Current validated local baseline as of 2026-09-22 (version `0.16.2`): `889 passed` with `92.45%`
coverage. `scripts/gate.sh` is the source of truth for this number; re-run it rather than trusting
this line as the project grows.

## Data collection (Polars + nflreadpy)

The authoritative data build pipeline is:

```bash
python -m nfl_predictor.data_collection
```

The default season range is controlled by `constants.MIN_SEASON` (currently 1999).

This writes datasets under `data/` (paths are defined in `nfl_predictor/constants.py`). Pass
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

Historical seasons load from cached artifacts where available. nflreadpy outputs are cached per
season under `data/cache/nflreadpy` (schedule, team stats, and play-by-play). Current/future
seasons are always refreshed to keep upcoming games and lines current. Use
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
schedule-adjusted strength family carries its own blend (`--no-strength-prior-blend`).

## Data sources + missing data

This project is designed to keep an invariant output schema across seasons, even when some sources
are missing historically.

Primary sources:

- `nflreadpy` (NFLverse): schedules, results, team-level stats, and play-by-play.
- Local cached CSVs under `data/` for Elo/market data when present.
- `data/meta_data.csv` (copied from `../nfeloqb`) for the quarterback name-to-id bridge. A name
  missing from it falls back to the play-by-play passer name (`F.Last`) when that is unique; a
  quarterback still unmatched gets null quarterback features, and the ETL logs the unmatched rate.
- TeamRankings web scrape for select ratings and stats not available in NFLverse (see ETL logs).

Missing data policy (high level):

- ETL emits all expected columns; missing sources become nulls and/or defined defaults.
- The ML pipeline is expected to tolerate nulls (imputation and/or model-native missing handling).

## Training + prediction

The primary entrypoint is:

```bash
python -m nfl_predictor.ml_model --help
```

### Quickstart (train + predict)

Train a margin/total model and generate predictions for a weekly input file:

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --holdout-seasons 0 \
  --calibration-seasons 0 \
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
python -m nfl_predictor.ml_model \
  --data-path data/completed_games_ml.csv \
  --model-kind margin_total \
  --holdout-seasons 1 \
  --calibration-seasons 1 \
  --win-prob-calibration isotonic \
  --predict-path data/predict/week_17_games_to_predict.csv
```

### Splits: train, calibration, holdout

Splits are time-aware by season and (optionally) by in-season week.

By default, training/evaluation uses **regular season** games only when the input data includes a
`game_type` column (i.e., postseason rows are filtered out). You can still generate predictions for
playoff games as long as the feature row exists.

To include postseason games in training, pass `--include-postseason`. To emphasize postseason games,
also set `--postseason-weight` (e.g., `--postseason-weight 1.5`). Optional recency weighting is
available via `--recency-half-life-seasons` to apply exponential decay by season to training
and calibration samples. It is off by default and in the shipped
weekly config: the six-season, two-seed measurement below found no gain from it.

- `--holdout-seasons` reserves the most recent seasons for evaluation only.
- `--calibration-seasons` and `--calibration-weeks` still gate whether a fitted post-processing
  calibrator is allowed to run, but the fitted calibration pool itself is now the previous two
  seasons plus the completed weeks of the current season.

Example: hold out the most recent season for evaluation and calibrate on the season before it:

```bash
python -m nfl_predictor.ml_model \
  --model-kind margin_total \
  --holdout-seasons 1 \
  --calibration-seasons 1
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

### Win probability calibration

Win probabilities are derived from the predicted margin, then optionally calibrated using a
time-aware calibration split (seasons and/or weeks immediately preceding the holdout window).

`--win-prob-calibration` options:

- `none`: deterministic Normal-CDF mapping using `constants.SCORE_DIFF_STD_DEV`.
- `elo`: deterministic Elo-style logistic mapping (no fitting).
- `sigma`: Normal-CDF mapping with one sigma estimated from the margin residuals of the
  calibration frame instead of the fixed `SCORE_DIFF_STD_DEV`.
- `platt`: Platt scaling on the calibration frame, with `C` chosen from a small grid on the
  latest pre-eval season of that frame.
- `isotonic`: isotonic regression on the calibration frame; below the 200-row threshold it falls
  back to `sigma`.
- `auto`: the deterministic floor (`none`). No fitted calibrator has beaten it on the walk-forward
  instrument, so `auto` does not fit anything.
- `logistic`: alias for `platt`.

The calibration frame for the fitted methods is the previous two seasons plus the completed weeks
of the current season, strictly before the predicted week. Its rows are in-sample for the model
that predicts them (an out-of-fold pool is an open follow-up), which is one reason the fitted
methods have not beaten the deterministic floor. The walk-forward report carries the configured,
deterministic and market-implied probability columns side by side so the choice can be measured.

### Market integration (optional, recommended)

If spreads/totals/moneylines are present, you can:

- use market-derived features (`--market-transform`)
- train on residuals vs market baselines (`--market-anchor`) so the model learns deviations rather
  than re-learning what the market already priced

Win probability can also be blended or clamped vs market-implied home win probability via
`--market-prob-weight` / `--market-prob-clamp` (`--market-prob-blend` is the older spelling). Use
`--market-prob-source raw|novig` to choose implied-prob handling and `--market-prob-blend-method
prob|logit` to blend in probability or log-odds space.

## Backtesting

Evaluate with walk-forward (rolling-origin) backtesting, which scores every week out of
sample, confidence-pool points included:

```bash
python scripts/walk_forward_backtest.py --help
```

This is the **canonical evaluation protocol** for model selection. By default it evaluates the last
N seasons (regular season only) and reports three probability views on the same games: the
configured calibrator, the deterministic margin map, and the market-implied home win probability.
Use `--include-postseason` if you want postseason folds included. Optional recency weighting
is available via `--recency-half-life-seasons`. GPU
acceleration is optional: add `--xgb-tree-method hist --xgb-device cuda`. If the latest season is
incomplete, either pass `--exclude-incomplete-seasons` or specify `--eval-seasons` explicitly; the
metrics report includes the evaluated window and any exclusions. Walk-forward calibration uses the
calibration frame described above for fitted calibrators; `auto` is the deterministic floor; the
summary table includes deterministic-minus-market bootstrap intervals for week 1, week 2,
weeks 3-18, and all weeks; and every fold runs the full `n_estimators` budget (no in-season early
stopping, in walk-forward or in production), with `best_iteration` recorded per head.

Every finished week logs its position, running time, and an estimate of the time remaining
(`Walk-forward fold 37/54 done: season 2024 week 5 (14 games, Brier 0.2213), 2410s elapsed, about
1107s remaining`). The estimate averages the weeks trained so far, so it runs a little low late in a
run as training sets grow. Measured on 2026-09-20/21 on a 24-core machine: a from-week-1 run over
three seasons takes about 50 minutes on an idle machine and about 110 minutes when anything else
loads it; over six seasons it takes about 100 minutes idle at the default 200-tree budget, about
3.3 hours at 400 and about 4.8 hours at 598 under load (run directories
`models/wf_m55_7_2020_2025_trees*/`). Run one walk-forward at a time: XGBoost uses every core, and
two concurrent runs slow each other down far more than twofold. When anything else is busy on the
machine, set `OMP_WAIT_POLICY=PASSIVE` (for example `OMP_WAIT_POLICY=PASSIVE python
scripts/walk_forward_backtest.py ...`): XGBoost's OpenMP threads otherwise spin while waiting for a
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
and `--checkpoint-dir` moves the root. The same checkpointing runs in `scripts/wf_compare.py`
(`--resume`, `--checkpoint-dir`) and inside the run directories of `scripts/weekly_run.py` (its
existing `--resume`). The metrics report records how many weeks
were restored and how many were trained. Checkpoints are small (a few hundred KB per run) and safe to
delete once a report is written.

Trend/season-phase ablation (drop trend + season-phase features while keeping everything else
identical) is available via `--disable-trend-features`. Example 2x2 comparison matrix:

```bash
python scripts/walk_forward_backtest.py
python scripts/walk_forward_backtest.py --recency-half-life-seasons 16
python scripts/walk_forward_backtest.py --disable-trend-features
python scripts/walk_forward_backtest.py --disable-trend-features --recency-half-life-seasons 16
```

Named feature groups can be ablated the same way with `--disable-feature-groups` (available on both
`scripts/walk_forward_backtest.py` and `scripts/wf_compare.py`). Group names come from
`constants.FEATURE_GROUP_COLUMN_MARKERS`; a column belongs to a group when any of that group's
markers is a substring of the column name, which catches the `away_`/`home_` prefixes and the
`_diff` suffix at once. An unknown group name is a hard error. The dropped column list is recorded
in the run's metrics report.

```bash
python scripts/walk_forward_backtest.py --disable-feature-groups pbp
```

`--n-estimators` overrides the XGBoost tree budget for the run (every in-season fit runs the full
budget), which is how the budget itself is measured against the default. The six-season ladder
measured with it (`200`, `400`, `598`) is recorded in `AGENTS.md` under "Tree-budget ladder".

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

## Weekly workflow (canonical)

The canonical "do everything for this week" entrypoint is:

```bash
python scripts/weekly_run.py --help
```

### High-level stages

1. (optional) refresh data (`python -m nfl_predictor.data_collection`)
2. (optional) walk-forward compare to choose market/calibration/prob-postprocess variants
3. train + calibrate the selected configuration
4. generate weekly predictions + betting outputs + (optional) power rankings

Notes:

- `--wf-*` flags control **walk-forward comparison** behavior (model selection).
- `--train-*` flags control **final training/calibration** for the model used to produce weekly
  outputs.
- `--xgb-*` flags control XGBoost runtime (GPU/CPU), and should be used for both comparison and
  final training.
- Outputs are written under the run directory (default: `models/<run_id>/`) unless `--output-dir` is
  provided.
- Stage 1 walk-forward comparison is resumable and writes `wf_compare/` artifacts under the run
  directory (including `wf_summary.csv` and per-candidate results).
- `--data-collection-args` forwards extra arguments to the data refresh stage as one
  shell-quoted string, for example
  `--data-collection-args "--min-season 2010 --stat-prior-blend-games 4"`. It is split
  shell-style and handed to `nfl_predictor.data_collection` as its argument list; when it is
  unset the refresh runs exactly as before. The same value is a config key
  (`data_collection_args` in `config/weekly_run.yaml`).

### How postseason games enter today

This is the current behavior, recorded for reference; none of it is a recommendation.

- Walk-forward and training filter to `game_type == "REG"` by default: `--include-postseason`
  and `--wf-include-postseason` are off, and `--postseason-weight` is `1.0`.
- The shipped `config/weekly_run.yaml` now keeps those regular-season defaults for the weekly run:
  `wf_include_postseason: false` and `include_postseason: false`. It still ships
  `postseason_weight: 1.3`, but that weight is inert unless postseason training is explicitly
  enabled, and `power_rankings_include_postseason: true` remains on for the rankings step.
- The schedule-adjusted strength composite built in ETL never includes postseason games.
- The power rankings default through-week is the week before the prediction week, clamped to the
  last regular-season week when the prediction week is postseason, because strength snapshots stop
  one week after the regular season. Pass `--power-rankings-through-week` to override it.

### Authoritative weekly workflow (runs, in this order)

1. Refresh data (ETL)

   ```bash
   python -m nfl_predictor.data_collection
   ```

2. Canonical evaluation + model selection (walk-forward)

   ```bash
   python scripts/walk_forward_backtest.py --help
   ```

3. Train + predict for the upcoming week (writes predictions + artifacts)

   ```bash
   python -m nfl_predictor.ml_model --help
   ```

4. Power rankings + projected standings

   ```bash
   python scripts/power_rankings.py --help
   ```

### Outputs and conventions

- Run artifacts (model/metrics/metadata/feature importance) land under `models/<run_id>/` by
  default.
- Weekly prediction outputs live next to the input prediction file (e.g., `data/predict/`).
- `metadata.json` includes dataset fingerprint, tuned params, and Optuna summary when tuning runs.
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
- `--method bradley_terry` ranks on a Bradley-Terry fit to game results instead: the current and
  previous season only (`--ratings-window-seasons 2`), prior-season games weighted `0.25`
  (`--ratings-prior-season-weight`), completed games scored by margin (`--ratings-target margin`),
  and the model's forecasts for future games kept out of the fit (`--ratings-include-future` is
  off). `--legacy-franchise-fit` restores the older all-seasons, equal-weight **franchise** ranking,
  which describes a club's history more than its current team, and implies this method.
- Projected standings are the same under both methods: current record plus the model's win
  probabilities for the remaining games.
- `scripts/weekly_run.py` accepts the same options (`--power-rankings-method`,
  `--power-rankings-strength-snapshots`, `--ratings-*`, `--legacy-franchise-fit`) and writes the
  same files. It skips the rankings with a warning when the snapshot for the requested week is
  missing.

Model selection hierarchy (default):

- Primary: deterministic probability quality and deterministic-minus-market intervals.
- Secondary: confidence pool expected points and stability.
- Tertiary: margin/total MAE (plus market-relative residual MAE when anchoring).

Metrics reports include a summary table (with metric priority + direction), plus optional
diagnostics such as season win totals (expected vs actual) and calibration drift by season/week.
Walk-forward reports also record the evaluation window, calibration window, and any excluded
incomplete seasons.

## Command line

Every task runs through one command, `nfl-predictor <command>` (installed into `.venv/bin` by
`uv sync`; `python -m nfl_predictor <command>` is the same). `nfl-predictor --help` lists the
commands by group, and `nfl-predictor <command> --help` shows a command's options:

| group | commands |
| --- | --- |
| weekly | `weekly` (refresh, stage-1 selection, train, predict, reports; resumable, JSON/YAML config) |
| research | `backtest` (walk-forward, the benchmark), `sweep` (calibration and market-probability variants), `explain` (SHAP attribution for a saved model) |
| data | `data` (the ETL), `validate` (`--live` compares against the schedule), `leakage-audit`, `lines`, `build-week` |
| models by hand | `train`, `predict` (`--model-in`), `rankings` |
| web | `web`, `users` |

The code lives in the package: the weekly run in `nfl_predictor/weekly_run/`, the other
commands in `nfl_predictor/cli/`. The per-module forms (`python -m nfl_predictor.data_collection`,
`python -m nfl_predictor.ml_model` and the others) keep working, and so do the old
`scripts/<name>.py` paths, which now only call the package; `scripts/gate.sh` is the check that
runs everything CI runs.

**Totals are diagnostic-only.** The total (over/under) columns of the betting report
(`total_value_side`, `total_edge_points`) come from the model's total head. Since version `0.6.2`
that head learns again (before, it stopped after one tree and predicted about 44 points for every
game), but in the 2023-2025 walk-forward it still trails the market's own total line: in weeks 3-18,
total MAE is `10.3152` in the production configuration (no market anchoring) and `10.2295` with
anchoring, against `10.0847` for the line itself. Treat an over/under lean as a diagnostic, not a
betting signal. Spreads, moneylines and win probabilities are unaffected.

`wf_compare` examples:

```bash
python scripts/wf_compare.py \
  --eval-last-n-seasons 3 \
  --market-mode hybrid \
  --market-prob-source raw \
  --market-prob-blend-method prob
```

```bash
python scripts/wf_compare.py \
  --eval-last-n-seasons 3 \
  --market-mode all \
  --market-prob-source both \
  --market-prob-blend-method both
```

Uncertainty-aware comparison:

```bash
python scripts/wf_compare.py \
  --eval-last-n-seasons 3 \
  --win-prob-uncertainty both
```

`weekly_run` config example (JSON):

```json
{
  "wf_eval_last_n_seasons": 3,
  "wf_market_mode": "hybrid",
  "wf_market_prob_source": "raw",
  "wf_market_prob_blend_method": "prob",
  "predict_path": "data/predict/week_03_games_to_predict.csv"
}
```

Run it with:

```bash
python scripts/weekly_run.py --config path/to/weekly_run.json
```

GPU note (XGBoost 2.x): prefer `--xgb-tree-method hist --xgb-device cuda`.

If you see great performance on the exact data a model trained on, that is not evidence the model
generalizes. Prefer holdout and walk-forward metrics.

## Web UI

A FastAPI backend (`nfl_predictor/api/`) and a React single-page app (`web/`) browse the
project's outputs and, for admins, run its jobs from a browser. The backend indexes
`models/*/metadata.json`, lets an admin mark one run **active**, and serves that run's
predictions, confidence picks, betting table (totals are never actionable), power rankings,
model metrics and data status; the Jobs pages run the ETL, a lines-only refresh
(`nfl_predictor/lines_refresh.py`), future-week inputs (`nfl_predictor/week_builder.py`),
training, prediction, walk-forward and the reports as streamed background subprocesses.

```bash
python -m nfl_predictor.api.auth.cli create-user <name> --role admin   # once
python -m nfl_predictor.api                                            # http://127.0.0.1:8000
```

The frontend needs Node 26 (`source ~/.nvm/nvm.sh`); `cd web && npm ci && npm run build` writes
`web/dist/`, which the backend serves. Configuration is by `NFLP_*` environment variables
(`NFLP_DATA_DIR`, `NFLP_MODELS_DIR`, `NFLP_STATE_DIR`, `NFLP_PORT`, ...). Details, the check
commands and the layout are in `web/README.md`; the design and phase status are in
`.agents/web_ui_plan.md`.

## Validation

Offline validation:

```bash
python scripts/validate_offline.py
```

Live validation (may require network access):

```bash
python scripts/validate_live.py
```

Both read `data/all_data.csv` unless `--data-dir` points them at another dataset directory, and
both exit non-zero when the input data file is missing or the validation fails, so they are safe to
use in shell automation.

Canonical local validation sequence, as one command:

```bash
scripts/gate.sh          # add --web when web/ changed; --quick skips pytest
```

It runs the steps below in CI's order and reports every one before exiting non-zero:

```bash
ruff format --check .
ruff check .
pyright .
ty check .
python -m pytest
markdownlint .
uv lock --check
uv sync --check --active
```

CI installs `markdownlint-cli` for the `markdownlint .` step; on a machine that has
`markdownlint-cli2` instead, the equivalent is
`markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings" "#.agents/skills"`. The
`.agents/skills/` folder is an optional, gitignored clone of agent skills; ruff and
`markdownlint .` skip it through `pyproject.toml` and `.markdownlintignore`.

GitHub Actions mirrors this gate in `.github/workflows/validation.yml` and also runs the
editable-install smoke check plus `--help` smoke checks for `nfl_predictor.ml_model`,
`scripts/weekly_run.py`, `scripts/power_rankings.py`, and `nfl_predictor.api`.

## Leakage audit

To detect obvious feature leakage patterns:

```bash
python scripts/leakage_audit.py
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
- `feature_importance.json` includes XGBoost gain/weight importance per model head.
- Metadata includes timestamp, dataset fingerprint/hash, key package versions, training config/CLI
  args, feature list, and tuning/early-stopping info (when used). `models/` and `optuna.db` are
  gitignored by default, so keep run artifacts local unless you copy them elsewhere.

## Confidence pool rules (implemented)

- Each week assigns unique confidence values `1..N` to each picked winner.
- Max weekly points: `N*(N+1)/2`.
- Realized points: `sum(confidence_value * 1[pick_correct])`.
- Ties count as incorrect.

## Score rounding / realism (optional)

When generating predictions, you can optionally post-process **display scores** without changing
training targets, win probabilities, or pool ranking logic:

- `--score-rounding none|int|half|nfl`

Use `nfl` to snap to common NFL score patterns for reporting.

## Repository layout notes

Some modules are split to keep files under lint `max-module-lines` limits while preserving legacy
import paths.

- Polars ETL helpers live under `nfl_predictor/utils/polars/` with a compatibility facade at
  `nfl_predictor/utils/polars_utils.py`.
- ML implementation lives under `nfl_predictor/ml/` with a compatibility facade at
  `nfl_predictor/ml_model.py`.

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
  `nfl_predictor/utils/polars/qb_stats.py`): for `away_qb` and `home_qb`, from that quarterback's
  regular-season dropbacks in every earlier week across teams and seasons (never the game's own
  week). Career EPA per dropback (`qb_dropback_epa`), CPOE (2006+), sack rate and ANY/A are shrunk
  toward the league with `K = constants.QB_PRIOR_DROPBACKS` pseudo-dropbacks,
  `(sum + K * league_rate) / (count + K)`, so a first start gets the league rate; the last
  `constants.QB_RECENT_GAMES` games (`qb_dropback_epa_recent`, `qb_any_a_recent`) are shrunk
  toward the career rate the same way; `qb_history_dropbacks` tells the model how much evidence
  stands behind them. Scrambles are credited to the team-game's primary passer, because the
  play-by-play cache keeps the passer id but not the rusher id.

The strength family is ablatable as the `strength` feature group
(`--disable-feature-groups strength`), and the early-season prior blend can be ablated
independently at ETL time with `--no-strength-prior-blend`. The quarterback family is the `qb`
group (`--disable-feature-groups qb`). The rare-event noise family is the `rare_events` group
(`special_teams_tds`, `def_fumbles`, `fumble_recovery_tds`, `2pt_conversions`, `def_safeties`,
`def_tds`). No group overlaps another.

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

For users reading this documentation: commands are shown assuming your project virtual environment
is already activated. Agent-specific files keep the fully qualified `.venv/bin/...` forms for
automation reliability.

See the Validation section above for the canonical local validation sequence.

## Safety and claims

This project outputs statistical forecasts and backtest metrics. It does not guarantee accuracy,
profit, or betting success.
