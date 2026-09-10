# Next Agent Session Prompt

You are the orchestrating agent (Claude Opus 5) for an implementation session in the `nfl-predictor`
workspace (`/home/mitch/workspace/nfl-predictor`). Your deliverable is **Milestone 49: continuous
early-season shrinkage** from `.agents/TODO.md`, validated with a from-week-1 walk-forward. If it
lands with time to spare, continue into **Milestone 43 phase 2** (43.2 onward), also in
`.agents/TODO.md`.

## Why this order, and the one thing to say in your first response

The 2026-09-09 planning review put Milestone 43 next and flagged the shrinkage defect as higher
value. The 2026-09-10 review reorders them, for a reason of timing rather than value: **2026 Week 2
kicks off Thursday 2026-09-17.** Week 2 is the one week of the season this defect wrecks (pick
accuracy `0.5208` against `0.6958` for weeks 3-18). A fix that lands and validates on history this
week changes the Week 2 picks; a fix that lands next week helps in 2027. Milestone 43 phase 2 is a
reporting refinement with no deadline.

Say this in one sentence at the top of your first response so the user can flip the order if they
disagree. Then start. Do not ask permission to begin.

## 0. Read first, in this order

1. `AGENTS.md`: non-negotiables, command forms, readiness behaviors, the validated baseline, and
   the note that the recorded benchmark's dataset build no longer exists on disk.
2. `.agents/TODO.md`, the Milestone 49 section in full, then Milestone 43 and the follow-ups.
3. `.agents/ARCHIVE.md`, Milestone 46: the strength snapshot already does the blend you are about
   to generalize, and its walk-forward showed the blend is a tie on Brier. Expect that here too.
4. `nfl_predictor/data_collection.py::process_week` (around lines 1042-1078) and
   `nfl_predictor/utils/polars/teamrankings.py::aggregate_team_stats_to_week`,
   `regress_to_mean`, `recompute_derived_metrics`.
5. `nfl_predictor/utils/polars/strength_snapshot.py` lines 85-95 and 305-330: the existing
   `games / (games + PRIOR_BLEND_GAMES)` blend, `PRIOR_BLEND_GAMES = 4.0`.
6. Tests that pin the current fallback: `tests/test_data_collection.py`
   (`test_process_week_uses_fallback_stats_for_week1`,
   `test_week1_fallback_regresses_pbp_rates_toward_the_league_mean`,
   `test_week1_row_features_ignore_every_current_season_play`) and
   `tests/test_data_collection_helpers.py::test_process_week_fallback`.

## 1. Facts to trust unless your verification disproves them

- Baseline 2026-09-10, version `0.4.0`: all gates green, `558 passed`, coverage `90.83%`.
  `uv sync --check --active` passes again (the environment had the `0.3.0` package installed after
  the version bump; a plain `uv sync` fixed it).
- **The dataset on disk is not the build the recorded walk-forward arms ran on.** Every
  `models/wf_strength_2023_2025_*` report carries dataset hash `668368d8...`; the current
  `data/completed_games_ml.csv` (rebuilt 2026-09-09 17:28 by the Week 1 refresh) fingerprints to
  `5b6af6aa...`. So the "off" arm for your comparison does not exist yet. Start it first (Phase 0).
- The defect, verified on the current dataset, 2024 season, `away_games_played` and
  `away_success_rate`:

  | week | games_played | std | min | max |
  | --- | --- | --- | --- | --- |
  | 1 | 17 (regressed prior) | `0.0216` | `0.373` | `0.455` |
  | 2 | 1 | `0.0807` | `0.250` | `0.569` |
  | 3 | 2 | `0.0659` | `0.265` | `0.500` |
  | 16 | 14 | `0.0359` | `0.375` | `0.490` |

- Root cause, verified in code: `aggregate_team_stats_to_week` takes the plain mean of prior
  in-season games; `process_week` builds the regressed prior-season frame only for
  `teams_needing_fallback`, the teams with **zero** in-season games. One game is enough to switch a
  team from 100% prior to 0% prior.
- `aggregate_team_stats_to_week` stores per-game **means** of counts, and the derived rates are
  ratios of those means (equal to ratios of sums because the game count cancels). Blending means
  then calling `recompute_derived_metrics` keeps every rate a ratio of blended sums. Do not blend
  the rates directly.
- The play-by-play counts are already joined into `team_stats_df` before `process_week`, so one
  blend on the aggregated frame covers the nflreadpy stats and the PBP family together.
- Already shrunk, leave alone: the strength family (`prior_strength_snapshot`, its own blend), the
  TeamRankings ratings (`tr_df` / `prev_tr_df`, a separate merge), Elo, trend features, records.
- Timings: full ETL rebuild about `480s` warm; from-week-1 walk-forward at
  `--eval-last-n-seasons 3` about `40 min`. The walk-forward writes to `models/<run_id>/` by
  default; `--out-json` may only ever point inside `models/`.
- The ETL overwrites `data/*.csv` unconditionally. Back them up before each rebuild.
- `markdownlint` here is `markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings"`.

## 2. Design decisions already made (do not relitigate; record deviations)

- **Form.** For every team with a prior-season regressed profile available, publish
  `w * in_season_mean + (1 - w) * regressed_prior_mean` per stat column with
  `w = games_played / (games_played + K)`, then `recompute_derived_metrics`. A team with zero games
  gets `w = 0`, which is exactly today's fallback, so **week-1 rows must be bit-identical before and
  after**. Write that test first; it is your strongest regression guard.
- **K.** Reuse `PRIOR_BLEND_GAMES = 4.0`; promote it to `constants.py` so both blends share one
  named value, rather than introducing a second constant. Expose it as
  `--stat-prior-blend-games` on `nfl_predictor.data_collection` with `--no-stat-prior-blend` to
  ablate, mirroring `--strength-prior-blend`.
- **`games_played` keeps its current semantics** in this cut (in-season count; the regressed prior
  carries `17`). Changing it is a separate, measurable follow-up; record it in `TODO.md`.
- **Build the prior once per season**, not once per week: `process_week` already takes
  `prior_strength_snapshot` computed by the caller, so add a `prior_season_stats` argument the same
  way and fall back to computing it inside when `None` (keeps existing tests working).
- **First season in the run** (`season == min_season`) has no prior; publish the raw in-season
  means as today. Playoff rows use full-season means as today (`w` is then about `0.8`; accept it).
- **Ablation is at ETL time**, not in the walk-forward: `--disable-feature-groups` removes columns,
  it cannot undo a value change. Two dataset builds, two walk-forward runs, both under `models/`.

## 3. Non-negotiables

- TDD: characterization or failing tests first, then production code, small diffs.
- No leakage: week `N` features use games strictly before week `N`; the prior is the previous
  season's regular season only, regressed by `constants.WEEK1_REGRESSION_FACTOR`.
- Polars-first ETL; no pandas in `data_collection.py` or `utils/polars/`.
- Docstrings with the formula; type hints; no milestone numbers in code, comments, or tests; no new
  `noqa` / `type: ignore` / `pragma: no cover` without a real reason.
- All Python tooling via `.venv/bin/...`; `uv` from PATH; never bare `python` / `pytest` / `ruff`.
- XGBoost margin/total is the only model family; no tuning campaigns.
- Do not modify `../nfeloqb` or `../nfl-sos-ratings`.
- Every number you report must be readable from a `metrics_report.json` under `models/`.
- Commit only if the user asks. If asked: one logical change per commit, a Conventional Commits
  subject (`type(scope): imperative summary`, see `AGENTS.md`), a body explaining what and why,
  ending with the attribution line the harness provides.

## Phase 0 - Baseline (start this before reading further code)

1. `cp data/completed_games_ml.csv data/completed_games_ml.pre_m49.csv` (and `all_data_ml.csv`).
2. Launch the **off arm** on the current dataset in the background, so it runs while you implement:

   ```bash
   .venv/bin/python scripts/walk_forward_backtest.py --eval-last-n-seasons 3 --wf-start-week 1 \
     --out-json models/wf_shrink_2023_2025_off/metrics_report.json
   ```

   There is no run-id flag; the named directories under `models/` come from pointing `--out-json`
   at `models/<name>/metrics_report.json`, and `metadata.json` (with the dataset hash) lands next
   to it. The report's `metrics.per_week` block gives the week 1, week 2, and weeks 3-18 windows
   you will report.
3. `.venv/bin/python -m pytest -q` and record the actual count.

## Phase 1 - Implement (49.1 to 49.4)

1. Test first: week-1 rows unchanged; a one-game team's week-2 stat equals `0.2 * in_season +
   0.8 * prior` at `K = 4`; rates are recomputed from blended sums (build a fixture where blending
   the rate directly gives a different number); `--no-stat-prior-blend` reproduces today's output
   exactly; the first season in the run is untouched.
2. Then the production change in `process_week`, the CLI flags, the constant promotion.
3. Rebuild the dataset (about 8 minutes) with the blend on. Verify with a one-off Polars check that
   the 2024 week-2 `away_success_rate` spread has collapsed toward the week-16 spread, and that
   week-1 rows match the backup bit-for-bit.
4. Run the leakage audit.

## Phase 2 - Measure (49.5)

Run the **on arm** from week 1 with the same command and `wf_shrink_2023_2025_on`. Report weeks 1,
2, and 3-18 separately against the off arm; the season aggregate hides the effect because 2 of 18
weeks change. Success is week 2 moving materially toward the weeks 3-18 numbers on Brier and log
loss without weeks 3-18 regressing. Week 1 should be identical by construction; if it is not, your
blend touched a fallback row and that is a bug, not a result.

If the result is a tie or a loss, say so and leave the switch default **off**. The strength blend
was a tie on Brier; this may be too. The honest table is the deliverable either way.

## Phase 3 - Docs and gate (49.6)

Update `README.md` (data sources / early-season handling), `AGENTS.md` (baseline table if the
default changes, the early-season note), `CHANGELOG.md`, and move Milestone 49 to `ARCHIVE.md`
with the table. `CHANGELOG.md` already has an `[Unreleased]` section; add to it rather than
opening a version. Then the full gate.

## Phase 4 - Only if time remains: Milestone 43 phase 2

Read the Milestone 43 section of `.agents/TODO.md`. The facts that matter: rank on
`adj_strength_composite` for `(season, through_week + 1)` rows, never on raw `adj_*`; a higher
`adj_def_*` is a **better** defense; the 2024 pre-week-18 anchor is BAL, DET, PHI, BUF, GB on the
composite and DET, BAL, BUF, GB, PHI on the current Bradley-Terry default. Keep `--method
bradley_terry` and `--legacy-franchise-fit`. Wire the phase 1 flags through `weekly_run.py`. No
ETL rebuild is needed for any of it.

## Final report to the user (structure)

1. Outcome first: what landed, the default you chose for the switch, and gate status.
2. The three-window table, off versus on, with the `models/` paths.
3. Whether Week 2 of 2026 will use the new build, and what the user must run after the Week 1
   games finish on Monday 2026-09-14 (`scripts/weekly_run.py` with the ETL refresh).
4. What was left out or deferred, and why.
5. Your recommendation for the next session: Milestone 43 phase 2 if it did not fit here, then
   the total/over-under investigation (Milestone 50), then Milestone 47.
