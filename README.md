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
    - [Authoritative weekly workflow (runs, in this order)](#authoritative-weekly-workflow-runs-in-this-order)
    - [Outputs and conventions](#outputs-and-conventions)
  - [Scripts](#scripts)
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

For a manual upgrade without the helper script:

```bash
uv lock --upgrade
uv sync
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

Current validated local baseline as of 2026-09-09: `471 passed` with `90.51%` coverage.

## Data collection (Polars + nflreadpy)

The authoritative data build pipeline is:

```bash
python -m nfl_predictor.data_collection
```

The default season range is controlled by `constants.MIN_SEASON` (currently 1999).

This writes datasets under `data/` (paths are defined in `nfl_predictor/constants.py`).

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

Play-by-play is the largest of those sources (roughly 1.2M regular-season plays for 1999-2025). It
is fetched one season at a time, reduced to the column list in `constants.PBP_COLUMNS`, filtered to
the regular season, team-normalized, and cached as `data/cache/nflreadpy/pbp_<season>_reg.parquet`.
Before kickoff the current season has no play-by-play published at all; that is non-fatal, and the
ETL falls back to cache or continues without it.

TeamRankings data is cached under `data/<season>/` as week-level CSVs; enable debug logging to see
cache hits. Use `--timing` to log per-step runtimes and `--debug-logs` for detailed ETL diagnostics.
Use `--refresh-nflreadpy` to force refresh nflreadpy data even when cache exists.

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
  --tune-metric expected_points \
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
available via `--recency-half-life-weeks` or `--recency-half-life-seasons` (use only one) to apply
exponential decay to training and calibration samples.

- `--holdout-seasons` reserves the most recent seasons for evaluation only.
- `--calibration-seasons` reserves seasons just before the holdout for calibration/blending.
- `--calibration-weeks` reserves the most recent weeks from the latest season for calibration.

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
- `platt`: Platt scaling via logistic regression fit on the calibration split.
- `isotonic`: isotonic regression fit on the calibration split.
- `auto`: use isotonic when calibration data is large enough; otherwise fall back to Platt.
- `logistic`: alias for `platt`.

Calibration is time-aware: it fits only on historical data relative to the evaluation window.

### Market integration (optional, recommended)

If spreads/totals/moneylines are present, you can:

- use market-derived features (`--market-transform`)
- train on residuals vs market baselines (`--market-anchor`) so the model learns deviations rather
  than re-learning what the market already priced

Win probability can also be blended or clamped vs market-implied home win probability via
`--market-prob-blend` / `--market-prob-clamp` (alias: `--market-prob-weight`). Use
`--market-prob-source raw|novig` to choose implied-prob handling and `--market-prob-blend-method
prob|logit` to blend in probability or log-odds space.

## Backtesting

Backtest and produce weekly confidence ranks and summary metrics:

```bash
python scripts/backtest_predictions.py \
  --model-in models/your_model.joblib \
  --model-kind margin_total \
  --data-path data/completed_games_ml.csv \
  --output-dir data/backtest
```

For the most realistic evaluation, use walk-forward (rolling-origin) backtesting:

```bash
python scripts/walk_forward_backtest.py --help
```

This is the **canonical evaluation protocol** for model selection. By default it evaluates the last
N seasons (regular season only) with time-aware calibration from the last K weeks of each eval
season. Use `--include-postseason` if you want postseason folds included. Optional recency weighting
is available via `--recency-half-life-weeks` or `--recency-half-life-seasons` (use only one). GPU
acceleration is optional: add `--xgb-tree-method hist --xgb-device cuda`. If the latest season is
incomplete, either pass `--exclude-incomplete-seasons` or specify `--eval-seasons` explicitly; the
metrics report includes the evaluated window and any exclusions. Walk-forward calibration uses the
last K weeks strictly before the eval week; if insufficient weeks or outcomes are available,
calibration is skipped for that fold.

Every finished week logs its position, running time, and an estimate of the time remaining
(`Walk-forward fold 37/54 done: season 2024 week 5 (14 games, Brier 0.2213), 2410s elapsed, about
1107s remaining`). The estimate averages the weeks trained so far, so it runs a little low late in a
run as training sets grow. A from-week-1 run over three seasons takes about 75 minutes on a 24-core
machine. Run one walk-forward at a time: XGBoost uses every core, and two concurrent runs slow each
other down far more than twofold. When anything else is busy on the machine, set
`OMP_WAIT_POLICY=PASSIVE` (for example `OMP_WAIT_POLICY=PASSIVE python
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
(`--resume`, `--checkpoint-dir`), `scripts/golden_command.py` (`--wf-resume`,
`--wf-checkpoint-dir`), and inside the run directories of `scripts/weekly_run.py` and
`scripts/betting_pipeline.py` (their existing `--resume`). The metrics report records how many weeks
were restored and how many were trained. Checkpoints are small (a few hundred KB per run) and safe to
delete once a report is written.

Trend/season-phase ablation (drop trend + season-phase features while keeping everything else
identical) is available via `--disable-trend-features`. Example 2x2 comparison matrix:

```bash
python scripts/walk_forward_backtest.py
python scripts/walk_forward_backtest.py --recency-half-life-seasons 2
python scripts/walk_forward_backtest.py --disable-trend-features
python scripts/walk_forward_backtest.py --disable-trend-features --recency-half-life-seasons 2
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

Recent ablation example (2003-2025 seasons, include postseason, calibration=platt, recency half-life
seasons=2):

```text
Setting                         Brier    LogLoss  MarginMAE  TotalMAE  ActualPts
Trends on, recency off          0.2325   0.7525   10.0159    10.1254   213.35
Trends on, recency on           0.2804   1.9525   10.0691    10.0670   213.55
Trends off, recency off         0.2372   0.7772   10.0987    10.0759   210.55
Trends off, recency on          0.2827   1.9775   10.2003    10.0706   211.50
```

Interpretation:

- Trend features improve probability metrics (Brier/log loss) and margin MAE, with a small tradeoff
  in total MAE.
- Recency weighting (half-life seasons=2) hurts probability metrics in this run; keep it off unless
  a future ablation shows improvement.

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
  missing. `scripts/golden_command.py` writes a separate `model_rating_rankings.csv` from one
  model's per-game ratings; it is a diagnostic, not the power ranking.

Model selection hierarchy (default):

- Primary: probability quality (Brier, log loss, reliability).
- Secondary: confidence pool expected points and stability.
- Tertiary: margin/total MAE (plus market-relative residual MAE when anchoring).

Metrics reports include a summary table (with metric priority + direction), plus optional
diagnostics such as season win totals (expected vs actual) and calibration drift by season/week.
Walk-forward reports also record the evaluation window, calibration window, and any excluded
incomplete seasons.

## Scripts

Repo utilities under `scripts/`:

- `scripts/betting_pipeline.py`: end-to-end orchestration (walk-forward compare -> resumable tuning
  -> final train -> weekly predictions + betting_report.csv). If `--predict-path` is omitted, the
  newest `data/predict/week_XX_games_to_predict.csv` file is selected automatically. `--dry-run`
  previews the planned paths/stages even in a clean checkout before local `data/` files exist. See
  `--help`.
- `scripts/objective_compare_models.py`: objective walk-forward comparison of two saved models by
  retraining per fold under identical splits.
- `scripts/betting_report_excel.py`: generate an Excel betting template/report.
- `scripts/golden_command.py`: convenience orchestration for walk-forward + training + prediction
  and artifact stamping.
- `scripts/shap_analysis.py`: optional SHAP feature attribution for a saved model (requires `shap`).
- `scripts/wf_compare.py`: sweep calibration + market-prob post-processing variants and summarize
  walk-forward metrics.
- `scripts/weekly_run.py`: weekly orchestration (refresh -> wf compare -> train -> predictions +
  reports), resumable with optional JSON/YAML config.
- `scripts/backtest_predictions.py`: run a backtest using a saved model artifact.

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

## Validation

Offline validation:

```bash
python scripts/validate_offline.py
```

Live validation (may require network access):

```bash
python scripts/validate_live.py
```

Both validation scripts exit non-zero when the input data file is missing or the validation fails,
so they are safe to use in shell automation.

Canonical local validation sequence:

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
`scripts/weekly_run.py`, and `scripts/power_rankings.py`.

## Leakage audit

To detect obvious feature leakage patterns:

```bash
python scripts/leakage_audit.py
```

## Changelog

Release history lives in `CHANGELOG.md` and follows the [Common
Changelog](https://common-changelog.org/) format. The historical baseline is `0.1.0` from `main`.

When preparing the next release, add a new `## VERSION - YYYY-MM-DD` entry at the top of the file
and keep the change groups in this order:

- `Changed`
- `Added`
- `Removed`
- `Fixed`

Keep each change to a single imperative line, link the most relevant commit or PR, and skip routine
formatting noise. Update `CHANGELOG.md` whenever user-facing behavior, tooling expectations, or the
operating workflow changes. Pushing a `0.x.y` or `v0.x.y` tag triggers
`.github/workflows/release.yml`, which extracts the matching `CHANGELOG.md` section and creates or
updates the GitHub release. Keep git tags aligned with changelog versions.

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
- Season-to-date record features (overall/division/conference).
- Divisional rivalry indicator.
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

The strength family is ablatable as the `strength` feature group
(`--disable-feature-groups strength`), and the early-season prior blend can be ablated
independently at ETL time with `--no-strength-prior-blend`.

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
