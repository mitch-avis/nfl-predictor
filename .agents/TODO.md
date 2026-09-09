# TODO - Active Work for nfl-predictor

This file is the **authoritative worklist** for the repo and contains **active** work only.
Completed milestones live in `ARCHIVE.md` (same directory). Agent workflow and guardrails live in
`../AGENTS.md`. The cross-repo review that motivates the current ordering lives in
`feature_crosswalk.md`.

- Completed work should be moved to `ARCHIVE.md` (with dates/notes).
- New milestones continue numbering from the latest archived milestone (44), so new work starts
  at 45.

---

## Execution loop (required)

For each task:

1. **Understand scope**: read the relevant modules/tests/docs.
1. **Plan**: outline the smallest set of changes needed.
1. **Test-Driven Development**: add/adjust tests for all planned changes. Aim to increase coverage.
   For executable code, confirm the exact behavior/lines you plan to touch are covered first; if
   not, add focused characterization or failing tests before editing production code.
1. **Implement**: make changes incrementally (small diffs, one logical change at a time).
1. **Run checks (venv only)**:
   - `.venv/bin/ruff format .`
   - `.venv/bin/ruff check .`
   - `.venv/bin/pyright .`
   - `.venv/bin/ty check .`
   - `.venv/bin/python -m pytest`
   - `markdownlint .`
   - `uv lock --check`
   - `uv sync --check --active`

1. **Update docs** where behavior changes (README/AGENTS/CHANGELOG), and update TODO/ARCHIVE.

### Always use the repo venv

Agents and humans should not rely on the shell activation state.

- Do **not** run `python`, `pip`, `pytest`, `ruff`, `pyright`, or `ty` without the venv prefix.
- Use `.venv/bin/python ...` or the tool-specific binary under `.venv/bin/`.
- Use `uv ...` from `PATH` for dependency management and environment sync.

### Current validated baseline (2026-09-09)

- `.venv/bin/ruff format .`, `.venv/bin/ruff check .`, `.venv/bin/ty check .`, and
  `.venv/bin/pyright .` pass cleanly.
- `.venv/bin/python -m pytest` passes (`412 passed`) with coverage `90.03%` against the enforced
  `90%` floor.
- `markdownlint .`, `uv lock --check`, and `uv sync --check --active` pass.
- ETL was rerun for `1999-2026`: `data/completed_games_ml.csv` covers `1999-2025` (`7260` rows,
  `384` columns) and `data/predict/week_01_games_to_predict.csv` holds `16` rows for 2026 Week 1.
- The leakage audit passed on the refreshed dataset.
- Reference walk-forward benchmark (seasons `2023-2025`, default config): Brier `0.2312`, log loss
  `0.7352`, pick accuracy `0.6833`, margin MAE `9.8954`, total MAE `10.1021`, reliability ECE
  `0.1308`. Every feature milestone below reports against these numbers.

---

## Guardrails checklist (must stay true)

- [ ] Confirm all new features apply to all matchups (no end-of-season-only logic).
- [ ] Confirm no leakage: all splits, features, and labels are time-aware.
- [ ] Confirm PBP-derived and schedule-adjusted features for week `N` use only games strictly
      before week `N` in that season; playoff rows use the full regular season.
- [ ] Confirm ETL is Polars-first and pulls NFLverse via `nflreadpy`.
- [ ] Confirm every nflreadpy source (PBP included) is cached per season and degrades non-fatally
      for the current season before data is published.
- [ ] Confirm artifacts remain reproducible (run folder, metadata, metrics).
- [ ] Confirm new code includes unit tests and does not reduce coverage.
- [ ] Confirm historical data is cached (TeamRankings + nflreadpy), and re-scrapes are minimized.
- [ ] Confirm the readiness behaviors in `AGENTS.md` (Week 1 detection, non-fatal current-season
      404s, CSV-based prediction-file resolution) still hold.
- [ ] Confirm neighboring repos (`../nfeloqb`, `../nfl-sos-ratings`) are not modified.

---

## Roadmap Status

Milestone 44 completed on 2026-06-13 and lives in `ARCHIVE.md`. On 2026-09-09 the roadmap was
reordered around the feature-engineering workstream (see `feature_crosswalk.md`). Execution order:

1. Milestone 45 - PBP foundation + per-snap team EPA, success, explosive, special-teams families
2. Milestone 46 - Weekly schedule-adjusted team strength (ridge snapshot) + EPA schedule strength
3. Milestone 43 (rewritten) - Power rankings measure current-season strength
4. Milestone 47 - QB per-dropback EPA families for the expected starter
5. Milestone 48 - PBP situational stats replace the TeamRankings stat scrape
6. Milestone 39 (with 40 folded in) - Off-season configuration sweep, after the feature work lands
7. Milestone 41 residuals - orchestration polish
8. Milestone 42 - parked (alternative model families) until the user reopens it

Rules for every feature milestone:

- Land each family as one ablatable feature group with a walk-forward switch (mirror
  `--disable-trend-features`).
- Report a `2023-2025` walk-forward comparison with the group on and off before marking done.
- Keep the invariant output schema: when a source is missing for a season, emit nulls.
- XGBoost margin/total remains the only model family in scope.

---

## Milestone 45 - PBP foundation + per-snap team EPA families

Goal: bring play-by-play into the ETL with caching, and publish the per-snap EPA, success,
explosive, and special-teams families that `nfl-sos-ratings` uses as its rating backbone.

Tasks:

- [ ] 45.1 Cached PBP loader in `nfl_predictor/utils/polars/loaders.py`:
  - explicit column list in `constants.PBP_COLUMNS` (see `feature_crosswalk.md` section 8), with
    every column existence-guarded
  - per-season Parquet cache under `data/cache/nflreadpy/pbp_<season>.parquet`, current-season
    refresh, `force_refresh` support, and non-fatal current-season failures with cache fallback
  - regular-season filter for feature inputs and team normalization for `posteam`, `defteam`,
    `home_team`, `away_team`
- [ ] 45.2 Team-game aggregation (new `nfl_predictor/utils/polars/pbp.py`): one row per
      `(season, week, team_abbr, opponent_abbr)` with snap counts (scrimmage-snap definition:
      dropback, rush, kneel, spike), dropbacks, carries, pass/rush EPA sums for offense and
      allowed, success counts, explosive counts, stuffed runs, early-down pass counts, special-teams
      EPA for/against and play counts, and the existing third/fourth-down, red-zone, and two-point
      counts from `aggregate_pbp_stats`. Formula in every docstring; hand-built fixture tests.
- [ ] 45.3 Join the team-game rows into `team_stats_df` in `collect_all_data` before
      `add_per_game_opponent_stats`; derive rates in `_compute_derived_metrics` as ratio of sums;
      add `constants.PBP_STATS` plus exclusion entries; extend `build_final_column_order`.
- [ ] 45.4 Verify the Week 1 previous-season regression path covers the new columns and that a
      season without PBP still emits the invariant schema.
- [ ] 45.5 Add the walk-forward ablation switch for the PBP feature group in
      `nfl_predictor/ml/walk_forward.py`, `scripts/walk_forward_backtest.py`, and
      `scripts/wf_compare.py`.
- [ ] 45.6 Rebuild `1999-2026`, run `scripts/leakage_audit.py`, run the `2023-2025` walk-forward
      with the group on and off, and record the table in `ARCHIVE.md` and `CHANGELOG.md`.
- [ ] 45.7 Update `README.md` data sources and feature areas; add `rushing_epa` to the published
      stats as part of the same schema change.

Acceptance:

- [ ] `data/completed_games_ml.csv` carries the new families for `1999-2025` with documented null
      rates per season.
- [ ] Leakage audit passes; a test proves a future week's plays do not change an earlier week's
      features.
- [ ] Walk-forward table recorded against the reference baseline; all gates green; coverage stays
      at or above `90%`.

---

## Milestone 46 - Weekly schedule-adjusted team strength

Goal: publish a leakage-safe, pre-week schedule-adjusted offense/defense/special-teams strength per
team as ETL features, plus EPA-based schedule strength for games played and games remaining.

Tasks:

- [ ] 46.1 `nfl_predictor/utils/polars/adjusted_strength.py`: NumPy ridge solve with one offense
      coefficient, one defense coefficient per team, and one home-field term, centered per side
      (port the design of `nfl-sos-ratings/simultaneous_adjustment.solve_team_stat_ridge`). Tests:
      synthetic round-robin recovers known strengths; home-field sign; centering; empty input.
- [ ] 46.2 Weekly snapshot builder: for each `(season, week)` solve on prior-week regular-season
      rows using pass and rush EPA per snap responses; emit `adj_off_pass_epa_snap`,
      `adj_off_rush_epa_snap`, `adj_def_pass_epa_snap`, `adj_def_rush_epa_snap`, `adj_hfa`, an SRS
      companion, and `st_rating` from special-teams EPA margin. Fixed ridge lambda for v1.
- [ ] 46.3 Early-week prior: previous-season final snapshot regressed by
      `WEEK1_REGRESSION_FACTOR`, blended with the in-season solve by `games / (games + K)` with a
      documented `K`. Playoff weeks use the full regular season.
- [ ] 46.4 Composite for display (`adj_strength_composite`) with documented default weights
      (start from the sos published weights over within-season standardized components). Model
      features stay in raw adjusted units.
- [ ] 46.5 Schedule strength: `sos_played_adj` and `sos_remaining_adj` from opponents' pre-week
      composite.
- [ ] 46.6 Merge in `process_week` as `away_`/`home_`/`_diff`; constants and finalization;
      ablation switch; leakage test that perturbs week `N+1` and asserts week `N` is unchanged.
- [ ] 46.7 Walk-forward comparison against the Milestone 45 state; record results.

Acceptance:

- [ ] A late-season snapshot ranks teams consistently with current-season point differential and
      adjusted EPA, not prior seasons.
- [ ] Walk-forward table recorded; all gates green.

---

## Milestone 43 - Power rankings measure current-season strength (rewritten 2026-09-09)

Goal: a Week `N` ranking reflects how strong teams are going into week `N`. Verified current
behavior: a Bradley-Terry fit over every season since 1999 with equal weights, fixed `0.97 / 0.03`
targets, and future games filled with model probabilities.

Tasks:

- [ ] 43.1 Quick defaults in `scripts/power_rankings.py` and `scripts/weekly_run.py`:
  - `--ratings-window-seasons` (default `2`: current plus previous season) and
    `--ratings-prior-season-weight` (default `0.25`) with sample weights in
    `fit_bradley_terry_ratings`
  - margin-based targets via `margin_to_home_win_prob` by default; keep the binary mapping as an
    option
  - exclude future model-probability rows from the strength fit by default (they stay in projected
    standings); add `--legacy-franchise-fit` to reproduce the old output
- [ ] 43.2 After Milestone 46: default the ranking to the ETL adjusted composite for
      `(season, through_week + 1)` rows, map to the existing 1-10 and 0-10 scales, and publish the
      components next to the rank. Keep Bradley-Terry available as `--method bradley_terry`.
- [ ] 43.3 Leave projected standings as record plus model win probabilities. Label or remove
      `golden_command._build_pregame_power_rankings` so one ranking artifact is canonical.
- [ ] 43.4 Tests: recency weighting shifts ratings toward recent results; window logic excludes
      older seasons; a synthetic breakout team ranks first late in the season.
- [ ] 43.5 README and `--help` explain "current-season" versus "franchise" rankings.

Acceptance:

- [ ] Default weekly-run rankings for a late-season week align with current-season results and
      recent form; the franchise view remains reachable by explicit flag.

---

## Milestone 47 - QB per-dropback EPA families for the expected starter

Goal: give the model the expected starter's per-dropback production instead of only the nfeloqb
value/Elo pair.

Tasks:

- [ ] 47.1 Identity bridge: read a copied `data/qb_meta_data.csv` (from
      `../nfeloqb/Other Data/meta_data.csv`, read-only contract) to map `away_qb`/`home_qb` names
      to GSIS ids; fallback name match against PBP `passer_player_name`; log the unmatched rate;
      tests for aliases and misses.
- [ ] 47.2 QB-game aggregation from PBP by `passer_player_id`: dropbacks, attempts, completions,
      pass yards, TDs, INTs, sacks, sack yards, `qb_epa` sum, CPOE mean (2006+). Formulas in
      docstrings; fixture tests.
- [ ] 47.3 Career-to-date and season-to-date rates strictly before the game date, plus a rolling
      dropback window for recency; new-starter fallback to a regressed league mean with a
      `qb_history_dropbacks` column so the model can see sample size.
- [ ] 47.4 Join for away/home plus diffs; constants; finalization; null policy documented.
- [ ] 47.5 Leakage audit and walk-forward ablation; record results.
- [ ] 47.6 Optional phase 2: opponent-adjusted QB EPA via a dropback-weighted ridge against faced
      defenses (design in `nfl-sos-ratings/simultaneous_adjustment.solve_qb_stat_ridge`).

Acceptance:

- [ ] Unmatched-QB rate is reported and below an agreed threshold for `2006+`.
- [ ] Walk-forward table recorded; all gates green.

---

## Milestone 48 - PBP situational stats replace the TeamRankings stat scrape

Goal: compute third/fourth-down, red-zone, and two-point rates (offense and allowed) from the
Milestone 45 counts so those eight columns no longer depend on scraping and extend to 1999.

Tasks:

- [ ] 48.1 Derive the rates from PBP counts in `_compute_derived_metrics`.
- [ ] 48.2 Keep the existing TR column names and switch the source, behind a transitional
      `--tr-stats-source pbp|scrape` option; the TeamRankings ratings scrape is unchanged.
- [ ] 48.3 Sanity-compare PBP values against scraped values for `2010-2025`; walk-forward check;
      update README data sources.

Acceptance:

- [ ] The eight situational columns are populated for `1999+` offline; walk-forward is not worse.

---

## Milestone 39 - Off-season configuration sweep + lock default settings (deferred until 45-47 land)

Goal: run an objective, repeatable sweep of modeling configurations under the canonical evaluation
protocol, then write the selected configuration as the default for weekly runs. Deferred because
new feature families would invalidate sweep results.

Tasks:

- [ ] Define a sweep config schema (JSON or YAML) covering model kind (`margin_total`,
      `blended_margin_total`), calibration, market mode and probability source, blend method,
      weight and clamp grids, uncertainty, tuning, and XGBoost params including GPU preference.
- [ ] Implement `scripts/config_sweep.py` (or `--mode sweep` in `scripts/weekly_run.py`) that runs
      walk-forward per config, writes `sweep_summary.csv` and `best_config.json`, supports resume
      by dataset hash plus config hash, and always includes a baseline row with market blending
      and clamping off with deltas versus that baseline (former Milestone 40).
- [ ] Add `xgb_device=auto` (prefer `cuda`, else CPU) used identically in evaluation and final
      training.
- [ ] Decide the `ScoreModel` fate: document as experimental or deprecate cleanly.
- [ ] Add the stability view by season and week bucket, and a "recommended defaults" section.

Acceptance:

- [ ] One command plus one config file produce the sweep summary and `best_config.json`, and
      `scripts/weekly_run.py --defaults-path best_config.json` runs end to end.

---

## Milestone 41 - Weekly orchestration residuals

- [ ] Add data-refresh pass-through (`--data-min-season` / `--data-max-season` or a generic
      `--data-collection-args`) to `scripts/weekly_run.py`.
- [ ] Decide and document how postseason games enter evaluation and training; when the prediction
      week is postseason, default the power-rankings through-week to the last regular-season week.
- [ ] Wire sweep-selected defaults once Milestone 39 lands; confirm resume behavior.

Acceptance:

- [ ] One command produces the complete weekly package from scratch and re-running skips work
      whose inputs and config did not change.

---

## Milestone 42 - Ensembles and alternative models (parked)

Parked by user direction on 2026-09-09: XGBoost margin/total remains the primary model and
benchmark. Reopen only when the user asks. Original scope: direct win-probability classifier,
logit-space probability ensemble, optional LightGBM/CatBoost extras, season-phase specialization.
