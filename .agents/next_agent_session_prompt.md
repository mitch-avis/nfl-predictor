# Next Agent Session Prompt

You are the orchestrating agent for a session in the `nfl-predictor` workspace
(`/home/mitch/workspace/nfl-predictor`). Deliverables, in this order:

1. **The 2026 Week 2 weekly run is done** (Phase 0 closed). It completed 2026-09-17 18:36 MDT
   (`scripts/weekly_run.py --run-id weekly_2026_week_02`; ETL ran standalone first, then chained
   in with `--skip-data-refresh`). Started too late (~18:04 MDT) for that night's DET-at-BUF
   kickoff (~18:15 MDT) — that game had no model pick in time — but predictions, the betting
   report and power rankings exist for all 16 Week 2 games under `models/weekly_2026_week_02/`;
   see `.agents/TODO.md` Milestone 53 for the run summary. Nothing to do here unless something
   about that run looks wrong on inspection. For any future weekly run, use `weekly_run.py`
   directly (not `python -m nfl_predictor.data_collection` alone, which is only its ETL stage).
2. **The quarterback schedule lenses keep-or-drop decision** (Milestone 53, task 53.6) is open
   and belongs to the user, not this agent. It landed in version `0.9.0` (measured: no gain in
   walk-forward, every headline point estimate leans against the two columns). Do not decide it
   unilaterally; if the session includes that conversation, follow the user's call and update
   `TODO.md`/`ARCHIVE.md`/`constants.py` accordingly. Otherwise leave the columns in place and
   move on.
3. Otherwise, or once the schedule-lens decision is resolved, move to the Milestone 49
   `games_played` follow-up (Phase 2).

**Start on a new branch off `main`** for any new code work; the weekly run itself can run from
whatever branch is checked out (it does not commit anything). `feat/qb-schedule-lenses` (this
branch) holds task 53.6, pushed to `origin`. Confirm `git log --oneline -1` and `git status`
before starting; do not work on `main` directly.

## Calendar

- Today's date is in your environment. Week 2 opened Thursday 2026-09-17 (DET at BUF); the Week 2
  weekly run completed that evening (see Phase 0) — done, not something to redo.
- A weekly run needs a data refresh (ETL, roughly 10 minutes from the nflreadpy cache), about 3
  minutes of walk-forward comparison when it can reuse a warm cache (9 candidates; budget more,
  up to 18 minutes, on a cold one), and a few minutes of training and reports — the 2026-09-17
  run finished ETL-to-reports in about 23 minutes total; budget at least 30 for a future one
  that starts cold.

## 0. Read first, in this order

1. `AGENTS.md`: non-negotiables, the changelog rules (a new incremented version per landed
   change, never `[Unreleased]`, `pyproject.toml` in step, no tags or releases), the benchmark
   table, the walk-forward operating notes, and the new web-UI paragraph under "Project Shape".
2. `CHANGELOG.md` entries `0.6.0` to `0.9.0`: what the last sessions shipped, including the
   `0.7.1` review fixes, the `0.8.0` web-UI landing, and the `0.9.0` schedule lenses.
3. `.agents/TODO.md`: Milestone 53 (tasks 53.1-53.6 are done or measured, 53.6's keep-or-drop
   decision still open, 53.7 not started), Milestone 58 (the web UI's open phases), Milestone
   55's note about re-tuning, and the open follow-ups (the early-stopping window from task 56.4,
   the Milestone 49 `games_played` item, and the `_ratio` dedup left out of 53.6).
4. `.agents/ARCHIVE.md`: Milestone 52 (the total head: fixed, still behind the market line),
   Milestone 53 (through 53.6), Milestone 56 (partial) (the calibration window) and Milestone 58
   (partial) (the web UI, phases 0-3).
5. `nfl_predictor/utils/polars/qb_stats.py` and `tests/test_qb_stats.py`: the quarterback family
   including the schedule lenses, their formulas and the strictly-before rule.

## 1. Facts to trust unless your verification disproves them

- Version `0.9.0` in `pyproject.toml`, `uv.lock` aligned; no tag and no GitHub release exist or may
  be created. Gate on `feat/qb-schedule-lenses` as of 2026-09-17: `821 passed`, coverage `92.96%`,
  ruff, pyright, ty, markdownlint (on the changed docs) all clean. Re-run the frontend gate
  (`web/`: lint, typecheck, vitest, build) if this session touches `nfl_predictor/api/` or `web/`;
  it was last verified passing at the `0.8.0` merge, not re-checked since.
- The web server libraries are now core dependencies (version `0.8.0`), so a plain `uv sync`
  installs everything `tests/api/` imports. The `web` extra still exists but adds nothing.
- **Quarterback family (the per-dropback EPA stats, `constants.QB_PBP_STATS`): kept, decision
  closed.** The user reviewed the on/off walk-forward table on 2026-09-11 and chose to keep the
  family in production with no code change. Do not reopen this question.
- **Quarterback schedule lenses (`constants.QB_SCHEDULE_STATS`, task 53.6, version `0.9.0`):
  keep-or-drop is open, not closed.** Built and measured 2026-09-11: no gain anywhere in
  walk-forward (weeks 3-18 Brier `0.2312` on / `0.2282` off, paired diff `+0.0030`
  `[-0.0019, +0.0080]`; every headline point estimate leans against the lenses; only early pick
  accuracy, on a 96-game sample, clears its interval against them). They ship live in the `qb`
  group today only because no decision has been made yet, not because they are validated. The
  `qb_schedule` group ablates only these two columns if a training run needs to test without
  them. Full table: `.agents/ARCHIVE.md`, Milestone 53, "53.6 Schedule lenses."
- A training-time `--disable-feature-groups` switch for production paths (`ml_model`,
  `weekly_run.py`, `golden_command.py`) was considered and explicitly rejected as unneeded
  complexity; only the walk-forward tools carry it. Do not build one unless a later task
  genuinely needs to search over feature-group inclusion (Milestone 55's sweep is the plausible
  future home for that, not before).
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
- **Data.** The 2026-09-11 18:17 rebuild carries the schedule lenses: `data/completed_games_ml.csv`
  (`7263` rows, `525` columns, `4cf48985...`), its cut `data/completed_games_ml.m53_6_through_2025.csv`
  (`940cbbf4...`); the pre-rebuild `519`-column copy is in `data/backup_pre_m53_6/`. Leakage audit:
  `490` features, `0` flags. The 2026-09-17 evening ETL (started for the Week 2 weekly run, see
  Phase 0) refreshes `data/completed_games_ml.csv` again from the current nflreadpy cache plus
  the user's `../nfeloqb` refresh; check its fingerprint and row count rather than trusting the
  numbers above once that run has landed.
- **Quarterback family (`0.7.0`).** Seven stats per side plus diffs (`constants.QB_PBP_STATS`),
  career rates shrunk toward the league with `K = 300` pseudo-dropbacks, recent (last 8 games)
  rates shrunk toward the career, `qb_history_dropbacks`. Identity through `data/qb_meta_data.csv`
  (a read-only copy of `../nfeloqb/Other Data/meta_data.csv`; data is gitignored, so it must exist
  locally; without it the abbreviated passer-name fallback still matches most starters); 0 of
  7533 rows unmatched. It is the `qb` feature group (schedule lenses included since `0.9.0`; see
  the schedule-lens note above and `.agents/ARCHIVE.md` Milestone 53 for both walk-forward tables).
- `../nfeloqb` is the user's; never modify it. It was refreshed for Week 2 on 2026-09-17.

## 2. Decided (do not relitigate; record deviations)

- Totals stay diagnostic-only until a total model beats the closing line in walk-forward.
- The quarterback per-dropback family stays in production (user decision, 2026-09-11); no
  feature-group switch in the training CLIs unless Milestone 55's sweep needs one. The schedule
  lenses (task 53.6) are the one open exception — see section 1.
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

## Phase 0 - The Week 2 weekly run (done; reference only)

Completed 2026-09-17 18:36 MDT under `models/weekly_2026_week_02/`. Selected config
`hybrid_raw_prob_base_elo_blend0.20_clamp0.10`; calibration window `[[2025,16],[2025,17],
[2025,18],[2026,1]]` as expected; `total_signal = diagnostic_only` throughout; predictions,
confidence picks, betting report and power rankings all present for all 16 Week 2 games. The
Thursday-night DET-at-BUF game has a prediction on file but had no model pick in time for that
kickoff. Nothing left to do here; if the next weekly run is due (Week 3, opening after Monday
night 2026-09-21's game), follow the pattern above with a fresh `--run-id` and start it early
enough — the 2026-09-17 run took about 23 minutes end to end, budget at least 30.

## Phase 1 - The quarterback schedule lenses keep-or-drop decision

Task 53.6 already landed the code, tests and walk-forward measurement (version `0.9.0`; see
section 1 and `.agents/ARCHIVE.md` Milestone 53, "53.6 Schedule lenses"). There is no
implementation work left here — only the user's decision on whether to keep
`qb_faced_pass_def_adj` / `qb_faced_pass_def_raw` in the schema. If the user decides to drop
them: remove `QB_SCHEDULE_STATS` from `constants.py` (and its entry in
`FEATURE_GROUP_COLUMN_MARKERS`), the `qb_schedule` group, the ETL wiring in
`data_collection.py` (`defense_games` / `snapshots` args to `_attach_qb_features`), the
`_schedule_lenses` machinery in `qb_stats.py`, and their tests; rebuild and note the schema
change in `CHANGELOG.md` under a new version. If kept, just close out the TODO item recording
the decision. Do not spend implementation time here until the decision is made.

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

1. The Week 2 run: confirmed complete or not, and its outputs (picks, confidence ranking, any
   anomaly, whether Thursday night's game was covered).
2. The schedule-lens decision, if it was made this session, and what changed as a result.
3. What landed after that, with walk-forward tables and run directories.
4. What is left, and the recommendation for the next session.
