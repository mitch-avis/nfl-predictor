# Next Agent Session Prompt

You are the orchestrating agent (Claude Opus 5, or Sonnet 5 for a bounded task) for a session in
the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`). Deliverables, in this
order:

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

**Start on a new branch off `main`.** Everything from the 2026-09-11 sessions is merged and
pushed: `main` is at `7c105bc` (version `0.8.0`) and contains both
`fix/calibration-window-total-head` and `feat/web-ui`. Run `git switch main && git pull`, confirm
`git log --oneline -1` shows that hash or a descendant, then
`git switch -c <type>/<short-name>` (for example `feat/qb-schedule-lenses`, or
`chore/weekly-2026-week-02` for the weekly run's doc and config notes). Do not work on `main`
directly and do not reuse the two merged branches; they stay in place only as history.

## Calendar

- Today's date is in your environment. Week 1 finishes with DEN at KC on Monday 2026-09-14. Week 2
  opens Thursday 2026-09-17 (DET at BUF). This prompt was written on Friday 2026-09-11, so the
  window has not opened yet; check the date before deciding.
- The Week 2 run must start after Monday night's game is final and finish before Thursday
  kickoff. It needs a data refresh (ETL), so it takes roughly 10 minutes of ETL, about 18 minutes
  of walk-forward comparison (9 candidates), and a few minutes of training and reports.

## 0. Read first, in this order

1. `AGENTS.md`: non-negotiables, the changelog rules (a new incremented version per landed
   change, never `[Unreleased]`, `pyproject.toml` in step, no tags or releases), the benchmark
   table, the walk-forward operating notes, and the new web-UI paragraph under "Project Shape".
2. `CHANGELOG.md` entries `0.6.0` to `0.8.0`: what the last sessions shipped, including the
   `0.7.1` review fixes and the `0.8.0` web-UI landing.
3. `.agents/TODO.md`: Milestone 53 (tasks 53.1-53.5 are done or measured; 53.6 and 53.7 are
   open), Milestone 58 (the web UI's open phases), Milestone 55's note about re-tuning, and the
   open follow-ups (the early-stopping window from task 56.4, the Milestone 49 `games_played`
   item, and the items the 2026-09-11 code review added).
4. `.agents/ARCHIVE.md`: Milestone 52 (the total head: fixed, still behind the market line),
   Milestone 53 (partial), Milestone 56 (partial) (the calibration window) and Milestone 58
   (partial) (the web UI, phases 0-3).
5. `nfl_predictor/utils/polars/qb_stats.py` and `tests/test_qb_stats.py`: the quarterback family,
   its formulas and its strictly-before rule.

## 1. Facts to trust unless your verification disproves them

- Version `0.8.0` in `pyproject.toml`, `uv.lock` aligned; no tag and no GitHub release exist or may
  be created. Gate on the merged `main` at the end of the 2026-09-11 review session:
  `814 passed`, coverage `92.90%`, ruff, pyright, ty, markdownlint, `uv lock --check` and
  `uv sync --check --active` all clean. The frontend gate (`web/`: lint, typecheck, `22` vitest
  tests, build) was run in the `../nfl-predictor-web` worktree on the same tree and passes; the
  primary checkout has no `web/node_modules`, so run it there or `npm ci` first.
- The web server libraries are now core dependencies (version `0.8.0`), so a plain `uv sync`
  installs everything `tests/api/` imports. The `web` extra still exists but adds nothing.
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
  before the 2026-09-11 session, which re-pulls every season's nflverse cache; nflverse
  periodically republishes corrected historical values, so a full refresh can legitimately move
  historical rows with no code change. The quarterback identity files (`data/qb_elos.csv`,
  `data/qb_meta_data.csv`) were verified byte-identical to their `../nfeloqb` sources before and
  after that session, so the shift is not a quarterback-feature defect. See `.agents/ARCHIVE.md`,
  Milestone 53, "Resolved after review."
- **Calibration window (task 56.4, `0.6.1`, guard relaxed in `0.7.1`).** In-season calibration
  takes the newest completed `(season, week)` pairs across the pool, so the Week 2 run calibrates
  on 2026 week 1 plus 2025 weeks 16-18 and records them in `metadata.json` under
  `splits.calibration_inseason.pairs`; the training log now prints the same pairs. Final training
  also early-stops on that window (about 50-64 games); in the Week-2 smoke run the anchored margin
  head stopped at iteration 1. That predates the fix and is an open follow-up, not a regression.
  The walk-forward's own calibration selection (`walk_forward.select_calibration_data`) still
  takes weeks from the eval season only, so folds for weeks 2-4 run without a calibration frame
  while production rolls back; recorded as a follow-up in `TODO.md`, not fixed.
- **Total head (`0.6.2`, `0.6.3`).** Each XGBoost fit now gets its own early stopping. The healthy
  total head still trails the market line in 2023-2025 walk-forward (weeks 3-18 total MAE `10.3152`
  unanchored, `10.2295` anchored, `10.0847` for the line; its deviation from the line has no
  signal), so the betting report carries `total_signal = diagnostic_only`. The fix did not
  improve walk-forward total MAE (the pre-fix unanchored head scores `10.3009`, a tie).
- **Data.** The live build is the 2026-09-11 06:47 rebuild with the quarterback family:
  `data/completed_games_ml.csv` (`7263` rows, `519` columns, `acaa2892...`), its cut
  `data/completed_games_ml.m53_through_2025.csv` (`7261` rows, `06a7a34d...`),
  `data/all_data_ml.csv` (`7533` rows, `519` columns) and `data/strength_snapshots.csv` (`18818`
  rows); the pre-rebuild copy is in `data/backup_pre_m53/`. Leakage audit on it: `484` features,
  `0` flags. The benchmark input `data/completed_games_ml.m49_on_through_2025.csv` no longer
  exists; the benchmark in `AGENTS.md` stays auditable from `models/wf_checkpoints/`.
- **Quarterback family (`0.7.0`).** Seven stats per side plus diffs (`constants.QB_PBP_STATS`),
  career rates shrunk toward the league with `K = 300` pseudo-dropbacks, recent (last 8 games)
  rates shrunk toward the career, `qb_history_dropbacks`. Identity through `data/qb_meta_data.csv`
  (a read-only copy of `../nfeloqb/Other Data/meta_data.csv`; data is gitignored, so it must exist
  locally; without it the abbreviated passer-name fallback still matches most starters); 0 of
  7533 rows unmatched. A real-data check found every 2024 week-10 game identical when computed
  from play-by-play cut before week 10. It is the `qb` feature group. Since `0.7.1` the history
  seasons it loads for career rates always come from the per-season cache, so
  `--refresh-nflreadpy` refreshes only the seasons an ETL run processes.
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
- New work that is not part of an existing milestone takes number 59 onward (58 is the web UI).
- Walk-forward runs one at a time; `uptime` first; default OpenMP policy when idle. The web UI's
  job runner serializes its own walk-forward jobs, but it does not know about a walk-forward you
  start from a shell, so check `pgrep -af walk_forward` and the Jobs page before starting one.

## 3. Non-negotiables

- TDD, no leakage, time-aware evaluation, comparisons only within one build and code version.
- All Python tooling via `.venv/bin/...`; `uv` from PATH.
- Update `CHANGELOG.md` as each chunk lands, under a new incremented version, with
  `pyproject.toml` bumped and `uv lock` / `uv sync` run; never `[Unreleased]`, never a tag.
- Do not modify `../nfeloqb` or `../nfl-sos-ratings`; `.agents/skills/` is not this repo's.
- Commits (when the user asks): one logical change per commit, Conventional Commits subjects,
  a body with what and why, the harness attribution line. Version bumps and `uv.lock` go with the
  commit whose changelog entry they belong to. Merge finished work to `main` promptly rather than
  batching it on a long branch.

## Phase 0 - The Week 2 weekly run (time-boxed)

1. Ask the user whether `data/qb_elos.csv` and `data/qb_meta_data.csv` have been refreshed from
   `../nfeloqb` after Week 1; the ETL takes the Week 2 starters from `qb_elos.csv`.
2. `uptime`, then `.venv/bin/python scripts/weekly_run.py --run-id weekly_2026_week_02` (data
   refresh on; `--dry-run` first if in doubt). Resume with the same command after a stop. The
   user may prefer to launch it from the web UI's Jobs page (`weekly_run` template); either way,
   never start a second walk-forward while it runs.
3. Check: the ETL log's `QB features` unmatched rates; `metadata.json`
   `splits.calibration_inseason.pairs` (2026 week 1 plus 2025 weeks 16-18); the total head's
   `best_iteration` recorded; the betting report's `total_signal` column; predictions for all
   Week 2 games; power rankings through week 1. If the run was launched from the web UI,
   activate the run there afterwards so the pages read it.
4. Record the run in `.agents/TODO.md` (a short note is enough) and report the picks, the
   confidence ranking and any anomaly to the user.

## Phase 1 - Task 53.6, the quarterback schedule lenses

`qb_faced_pass_def_adj`: the dropback-weighted mean of the faced defenses' pre-week ridge
pass-defense coefficient (`adj_def_pass_epa_snap` from `data/strength_snapshots.csv`, the snapshot
of the week each game was played). `qb_faced_pass_def_raw`: the faced defenses' EPA per dropback
allowed from prior-week games, excluding games against the quarterback's team (the one-hop,
head-to-head-excluded method from `feature_crosswalk.md` section 3.1; credit the user's
`nfl-sos-ratings` method by name). Tests first, strictly-before rule as in `qb_stats.py`, then one
rebuild and an on/off walk-forward inside the `qb` group. While in `qb_stats.py`, the review's
reuse notes in `TODO.md` (shared `pbp` helpers, `calculate_stat_differentials`) are cheap to take
along; do not let them grow the task.

## Phase 2 - Milestone 49 follow-up: `games_played` as evidence

`games_played` publishes `17` for a week-1 fallback row and `1` for a blended week-2 row. Add an
effective-games column or a `stat_prior_weight` (see `.agents/TODO.md`), rebuild, and measure from
week 1 with weeks 1, 2 and 3-18 reported separately.

## 5. The web UI now lives on `main`

`feat/web-ui` merged into `main` on 2026-09-11 (version `0.8.0`, `CHANGELOG.md`): the FastAPI
backend in `nfl_predictor/api/`, the React app in `web/`, `nfl_predictor/lines_refresh.py`,
`nfl_predictor/week_builder.py`, `tests/api/`, and the `web` CI job. Its design and phase status
are in `.agents/web_ui_plan.md` (phases 0-3 done; 4-6 open as Milestone 58 in `TODO.md`).
`.agents/web_ui_session_prompt.md` is the historical Phase 2 prompt. The worktree
`../nfl-predictor-web` still exists on `feat/web-ui`, now equal to `main`; future web work should
start from a fresh branch off `main` (in that worktree or here), and keep merging to `main` after
each phase. Do not delete either merged branch without the user asking.

The user runs the API from that worktree on port 8765 against this checkout's `data/`, `models/`
and `reports/` (`NFLP_*` environment variables; state in `../nfl-predictor-web/data/web/`). The
instance running at the end of the review session was started at 02:40 on 2026-09-11 with the
pre-merge API code (its jobs already execute the merged CLIs, because they are subprocesses of
the worktree's files). A restart picks up the merged package; ask before bouncing it. Its job
history holds the `weekly_run` job that failed on the old `Not enough weeks in season 2026`
error, which the current code fixes. Tests under `tests/api/` are part of the normal gate now;
touching `scripts/weekly_run.py` flags or `metadata.json` keys can affect the API's readers and
job catalog (`nfl_predictor/api/jobs/catalog.py`), so run the whole suite.

## Final report to the user

1. The Week 2 run: done or not, and its outputs.
2. Task 53.6 outcome, with its own walk-forward table.
3. What landed after that, with walk-forward tables and run directories.
4. What is left, and the recommendation for the next session.
