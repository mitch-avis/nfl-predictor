# Next Agent Session Prompt

You are the orchestrating agent (Claude Opus 5) for a session in the `nfl-predictor` workspace
(`/home/mitch/workspace/nfl-predictor`). Deliverables, in this order:

1. **The 2026 Week 2 weekly run** (Phase 0), if the session falls between Monday night
   2026-09-14 (DEN at KC ends Week 1) and Thursday 2026-09-17 (DET at BUF opens Week 2). It is the
   first weekly run on the fixed code: the calibration window crosses the season boundary, the
   total head learns, the betting report labels totals `diagnostic_only`, and the dataset carries
   the quarterback family (kept by user decision, see section 1). Outside that window, skip to
   deliverable 2 and say so.
2. **Milestone 53, task 53.6: the quarterback schedule lenses** (Phase 1). The quarterback family
   itself (53.1-53.5) is done, measured, and the user decided 2026-09-11 to keep it in production;
   this is not an open decision, do not re-litigate it.
3. Otherwise, or once 53.6 lands, move to the Milestone 49 `games_played` follow-up (Phase 2).

Work stays on branch `fix/calibration-window-total-head` **or `main` if it has been merged**
(`git branch --show-current`, `git log --oneline -5 main` — check whether the branch's tip commit
is an ancestor of `main`). The branch was committed and pushed at the end of the 2026-09-11
session (5 commits, see section 4 for the split); if the user has since merged it to `main` and
deleted the branch, work on `main` instead and skip straight past section 4.

## Calendar

- Today's date is in your environment. Week 1 finishes with DEN at KC on Monday 2026-09-14. Week 2
  opens Thursday 2026-09-17 (DET at BUF).
- The Week 2 run must start after Monday night's game is final and finish before Thursday
  kickoff. It needs a data refresh (ETL), so it takes roughly 10 minutes of ETL, about 18 minutes
  of walk-forward comparison (9 candidates), and a few minutes of training and reports.

## 0. Read first, in this order

1. `AGENTS.md`: non-negotiables, the changelog rules (a new incremented version per landed
   change, never `[Unreleased]`, `pyproject.toml` in step, no tags or releases), the benchmark
   table, and the walk-forward operating notes.
2. `CHANGELOG.md` entries `0.6.0` to `0.7.0`: what the last session shipped.
3. `.agents/TODO.md`: Milestone 53 (tasks 53.1-53.5 are done or measured; 53.6 and 53.7 are
   open), Milestone 55's note about re-tuning, and the open follow-ups (the early-stopping window
   from task 56.4, the Milestone 49 `games_played` item).
4. `.agents/ARCHIVE.md`: Milestone 52 (the total head: fixed, still behind the market line) and
   Milestone 56 (partial) (the calibration window).
5. `nfl_predictor/utils/polars/qb_stats.py` and `tests/test_qb_stats.py`: the quarterback family,
   its formulas and its strictly-before rule.

## 1. Facts to trust unless your verification disproves them

- Version `0.7.0` in `pyproject.toml`, `uv.lock` aligned; no tag and no GitHub release exist or may
  be created. Gate at the end of the 2026-09-11 session: `649 passed`, coverage `91.21%`, ruff,
  pyright, ty, markdownlint, `uv lock --check` and `uv sync --check --active` all clean. Committed
  as 5 commits on `fix/calibration-window-total-head` and pushed to origin the same session.
- **Quarterback family: kept, decision closed.** The user reviewed the on/off walk-forward table
  (below) on 2026-09-11 and chose to keep the family in production with no code change: week 2
  pick accuracy improved meaningfully (48 games) and the weeks-3-18 headline loss sits inside its
  95% interval either way, so there was no evidence against keeping it. A training-time
  disabled-feature-group switch was considered and explicitly rejected as unneeded complexity for
  this decision (`walk_forward_backtest.py` / `wf_compare.py` already support
  `--disable-feature-groups` for any future ablation study). Do not reopen this question or build
  that switch unless a later task genuinely needs to search over feature-group inclusion
  (Milestone 55's sweep is the plausible future home for that, not before).
- **The walk-forward baseline shift is explained, not a bug.** The QB-off arm scored weeks 3-18
  Brier `0.2302` / log loss `0.7577` against the benchmark's `0.2284` / `0.7406` on an earlier
  build, same config and code. Cause: the user ran a full `--refresh-nflreadpy` ETL overnight
  before this session, which re-pulls every season's nflverse cache; nflverse periodically
  republishes corrected historical values, so a full refresh can legitimately move historical rows
  with no code change. The quarterback identity files (`data/qb_elos.csv`, `data/qb_meta_data.csv`)
  were verified byte-identical to their `../nfeloqb` sources both before and after this session, so
  the shift is not a quarterback-feature defect. See `.agents/ARCHIVE.md`, Milestone 53, "Resolved
  after review."
- **Calibration window (task 56.4, `0.6.1`).** In-season calibration takes the newest completed
  `(season, week)` pairs across the pool, so the Week 2 run calibrates on 2026 week 1 plus 2025
  weeks 16-18 and records them in `metadata.json` under `splits.calibration_inseason.pairs`.
  Final training also early-stops on that window (about 50-64 games); in the Week-2 smoke run the
  anchored margin head stopped at iteration 1. That predates the fix and is an open follow-up, not
  a regression.
- **Total head (`0.6.2`, `0.6.3`).** Each XGBoost fit now gets its own early stopping. The healthy
  total head still trails the market line in 2023-2025 walk-forward (weeks 3-18 total MAE `10.3152`
  unanchored, `10.2295` anchored, `10.0847` for the line; its deviation from the line has no
  signal), so the betting report carries `total_signal = diagnostic_only`. The fix did not
  improve walk-forward total MAE (the pre-fix unanchored head scores `10.3009`, a tie).
- **Data.** The user refreshed `data/` outside the session at 06:03 on 2026-09-11 and removed the
  older backups and the benchmark input `data/completed_games_ml.m49_on_through_2025.csv`; the
  benchmark in `AGENTS.md` stays auditable from `models/wf_checkpoints/c39db4f843175eaab09f/`. The
  06:47 rebuild with the quarterback family is the live build: `data/completed_games_ml.csv`
  (`7263` rows, `519` columns, `acaa2892...`), its cut `data/completed_games_ml.m53_through_2025.csv`
  (`7261` rows, `06a7a34d...`), and the pre-rebuild copy in `data/backup_pre_m53/`. Leakage audit
  on it: `484` features, `0` flags.
- **Quarterback family (`0.7.0`).** Seven stats per side plus diffs (`constants.QB_PBP_STATS`),
  career rates shrunk toward the league with `K = 300` pseudo-dropbacks, recent (last 8 games)
  rates shrunk toward the career, `qb_history_dropbacks`. Identity through `data/qb_meta_data.csv`
  (a read-only copy of `../nfeloqb/Other Data/meta_data.csv`; data is gitignored, so it must exist
  locally); 0 of 7533 rows unmatched. A real-data check found every 2024 week-10 game identical
  when computed from play-by-play cut before week 10. It is the `qb` feature group.
- **Quarterback walk-forward** (benchmark config: anchored, from week 1, 2023-2025, on the m53
  cut; `models/wf_qb_2023_2025_{on,off}/`, checkpoints `cb41507aa5be425f3c3f` on and
  `6024b0fcaebe9c480dd4` off): weeks 3-18 Brier `0.2327` on against `0.2302` off, log loss
  `0.7708` against `0.7577`, pick accuracy `0.6819` against `0.6806`, margin MAE `9.9839` against
  `9.9324`. Paired bootstrap, on minus off: Brier `+0.0025` `[-0.0025, +0.0074]`, log loss
  `+0.0131` `[-0.0083, +0.0346]`; weeks 1-2 lean the other way (week 2 pick accuracy `0.6458`
  against `0.5833` on 48 games). Full table in `.agents/ARCHIVE.md`, Milestone 53 (partial).
- The production training paths (`ml_model`, `weekly_run.py`, `golden_command.py`) have no
  `--disable-feature-groups` by design; only the walk-forward tools do, and that is intentional
  (see the quarterback decision above), not a gap to fill.
- `../nfeloqb` had uncommitted changes of its own on 2026-09-11 (`meta_data.csv`,
  `package_meta.json`, `qb_elos.csv`); they are the user's. Never modify that repo.

## 2. Decided (do not relitigate; record deviations)

- Totals stay diagnostic-only until a total model beats the closing line in walk-forward.
- The quarterback family stays in production (user decision, 2026-09-11); no feature-group switch
  in the training CLIs unless Milestone 55's sweep needs one.
- Tuning is out of scope; any future tuning starts from scratch (old studies scored a crippled
  total head).
- New work that is not part of an existing milestone takes number 58 onward.
- Walk-forward runs one at a time; `uptime` first; default OpenMP policy when idle.

## 3. Non-negotiables

- TDD, no leakage, time-aware evaluation, comparisons only within one build and code version.
- All Python tooling via `.venv/bin/...`; `uv` from PATH.
- Update `CHANGELOG.md` as each chunk lands, under a new incremented version, with
  `pyproject.toml` bumped and `uv lock` / `uv sync` run; never `[Unreleased]`, never a tag.
- Do not modify `../nfeloqb` or `../nfl-sos-ratings`; `.agents/skills/` is not this repo's.

## 4. If the user asks for commits

One logical change per commit, Conventional Commits, body with what and why, the harness
attribution line. A split that matches the changelog:

1. `docs(changelog): version the power-rankings entry as 0.6.0` plus the changelog-rule docs
   (`AGENTS.md`, `README.md` changelog section, `.agents/TODO.md` execution loop).
2. `fix(ml): roll the in-season calibration window across seasons` (`ml_model_core.py`,
   `ml_model_training.py`, their tests; `0.6.1`).
3. `fix(ml): give every XGBoost fit its own early stopping` (`ml_model_xgb_utils.py`, tests;
   `0.6.2`).
4. `feat(reporting): label betting-report totals diagnostic-only` (`scripts/betting_pipeline.py`,
   test, README caveat; `0.6.3`).
5. `feat(etl): add quarterback per-dropback features` (`qb_stats.py`, `pbp.py` helpers,
   `constants.py`, `finalize.py`, `data_collection.py`, tests, README; `0.7.0`).
6. `docs(agents): record milestones 52, 53 and task 56.4` (`.agents/*`, `AGENTS.md` benchmark).

Version bumps and `uv.lock` go with the commit whose changelog entry they belong to.

## Phase 0 - The Week 2 weekly run (time-boxed)

1. Ask the user whether `data/qb_elos.csv` and `data/qb_meta_data.csv` have been refreshed from
   `../nfeloqb` after Week 1; the ETL takes the Week 2 starters from `qb_elos.csv`.
2. `uptime`, then `.venv/bin/python scripts/weekly_run.py --run-id weekly_2026_week_02` (data
   refresh on; `--dry-run` first if in doubt). Resume with the same command after a stop.
3. Check: the ETL log's `QB features` unmatched rates; `metadata.json`
   `splits.calibration_inseason.pairs` (2026 week 1 plus 2025 weeks 16-18); the total head's
   `best_iteration` recorded; the betting report's `total_signal` column; predictions for all
   Week 2 games; power rankings through week 1.
4. Record the run in `.agents/TODO.md` (a short note is enough) and report the picks, the
   confidence ranking and any anomaly to the user.

## Phase 1 - Task 53.6, the quarterback schedule lenses

`qb_faced_pass_def_adj`: the dropback-weighted mean of the faced defenses' pre-week ridge
pass-defense coefficient (`adj_def_pass_epa_snap` from `data/strength_snapshots.csv`, the snapshot
of the week each game was played). `qb_faced_pass_def_raw`: the faced defenses' EPA per dropback
allowed from prior-week games, excluding games against the quarterback's team (the one-hop,
head-to-head-excluded method from `feature_crosswalk.md` section 3.1; credit the user's
`nfl-sos-ratings` method by name). Tests first, strictly-before rule as in `qb_stats.py`, then one
rebuild and an on/off walk-forward inside the `qb` group.

## Phase 2 - Milestone 49 follow-up: `games_played` as evidence

`games_played` publishes `17` for a week-1 fallback row and `1` for a blended week-2 row. Add an
effective-games column or a `stat_prior_weight` (see `.agents/TODO.md`), rebuild, and measure from
week 1 with weeks 1, 2 and 3-18 reported separately.

## 5. Coordinating with the web UI worktree

`../nfl-predictor-web` (branch `feat/web-ui`) is a separate worktree building the FastAPI+React
app; see `[[web-ui-worktree]]`/`.agents/web_ui_plan.md`. It reads this repo's `data/` and `models/`
directories directly (the user pointed its running instance at this workspace's paths, not its
own). Before that branch merges to `main`, it should rebase onto (or merge) whatever this branch
landed as, especially the task 56.4 calibration-window fix: the user already hit exactly that
crash (`Not enough weeks ... for calibration`) running `weekly_run` from the web UI, before this
fix existed there. If the user asks about merge order, recommend: keep this repo's ML/data-pipeline
work on short-lived branches merged to `main` frequently (each milestone or fix, not batched), and
have `feat/web-ui` rebase onto `main` right after each such merge rather than diverging for long
stretches, since it depends on this repo's model/data outputs being current and bug-free.

## Final report to the user

1. The Week 2 run: done or not, and its outputs.
2. Task 53.6 outcome, with its own walk-forward table.
3. What landed after that, with walk-forward tables and run directories.
4. What is left, and the recommendation for the next session.
