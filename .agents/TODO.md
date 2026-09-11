# TODO - Active Work for nfl-predictor

This file is the **authoritative worklist** for the repo and contains **active** work only.
Completed milestones live in `ARCHIVE.md` (same directory). Agent workflow and guardrails live in
`../AGENTS.md`. The cross-repo review that motivates the current ordering lives in
`feature_crosswalk.md`.

- Completed work moves to `ARCHIVE.md` (with dates and notes); a partly done milestone moves its
  finished tasks there and keeps the rest here.
- Active milestones are numbered in execution order. When the order changes, move the section rather
  than renumbering it; renumber only in a deliberate cleanup that records the old-to-new map in
  `ARCHIVE.md`. Archived numbers never change.
- Milestones up to 52 are archived or retired (the last cleanup, 2026-09-10, is recorded at the top
  of `ARCHIVE.md`). Active milestones run 53 to 57, so new work starts at 58.

---

## Execution loop (required)

For each task:

1. **Understand scope**: read the relevant modules/tests/docs.
2. **Plan**: outline the smallest set of changes needed.
3. **Test-Driven Development**: add/adjust tests for all planned changes. Aim to increase coverage.
   For executable code, confirm the exact behavior/lines you plan to touch are covered first; if
   not, add focused characterization or failing tests before editing production code.
4. **Implement**: make changes incrementally (small diffs, one logical change at a time).
5. **Run checks (venv only)**:
   - `.venv/bin/ruff format .`
   - `.venv/bin/ruff check .`
   - `.venv/bin/pyright .`
   - `.venv/bin/ty check .`
   - `.venv/bin/python -m pytest`
   - `markdownlint .` (CI installs `markdownlint-cli`; the local binary is `markdownlint-cli2`,
     so run `markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings" "#.agents/skills"` here)
   - `uv lock --check`
   - `uv sync --check --active`
6. **Update docs** where behavior changes (README/AGENTS), and update TODO/ARCHIVE. Add a
   `CHANGELOG.md` entry as each task or sizable chunk lands, under a new incremented version
   (never `[Unreleased]`), set `pyproject.toml` to that version, and run `uv lock` and `uv sync`.
   Never create a git tag or GitHub release (see `AGENTS.md`).
7. **Commit** (when asked) with Conventional Commits subjects, `type(scope): summary`, per
   `AGENTS.md`.

### Always use the repo venv

Agents and humans should not rely on the shell activation state.

- Do **not** run `python`, `pip`, `pytest`, `ruff`, `pyright`, or `ty` without the venv prefix.
- Use `.venv/bin/python ...` or the tool-specific binary under `.venv/bin/`.
- Use `uv ...` from `PATH` for dependency management and environment sync.

### Current validated baseline (2026-09-11, version `0.7.0`)

- `.venv/bin/ruff format .`, `.venv/bin/ruff check .`, `.venv/bin/ty check .`, and
  `.venv/bin/pyright .` pass cleanly.
- `.venv/bin/python -m pytest` passes (`649 passed`) with coverage `91.21%` against the enforced
  `90%` floor.
- `markdownlint-cli2`, `uv lock --check`, and `uv sync --check --active` pass.
- `.agents/skills/` is a separate git clone of agent skills. It is gitignored and excluded from
  ruff and markdownlint (`.markdownlintignore`); pyright already skips dot-directories and ty only
  checks `nfl_predictor`, `scripts`, and `tests`.
- `data/completed_games_ml.csv` is the 2026-09-11 06:47 rebuild with the quarterback family:
  `7263` rows (`1999-2025` plus the two completed 2026 Week 1 games), `519` columns, fingerprint
  `acaa2892...`; `data/completed_games_ml.m53_through_2025.csv` is its cut to seasons `<= 2025`
  (`7261` rows, `06a7a34d...`), the input of the Milestone 53 walk-forward arms. The build it
  replaced was a refresh made outside the agent session at 06:03 (`9b8bf303...`, `498` columns),
  backed up in `data/backup_pre_m53/`. That refresh also removed `data/backup_pre_m49/`,
  `data/backup_pre_m51/`, `data/completed_games_ml.pre_m49.csv` and the benchmark input
  `data/completed_games_ml.m49_on_through_2025.csv`, so the benchmark in `AGENTS.md` is auditable
  from its fold checkpoints only.
- The leakage audit passed on the 06:47 rebuild (`484` features, `0` flags,
  `models/wf_qb_2023_2025_on/leakage_audit.json`).
- The walk-forward benchmark lives in `AGENTS.md`: `models/wf_totalfix_2023_2025_anchored/`,
  measured 2026-09-11 with the total-head fix on `data/completed_games_ml.m49_on_through_2025.csv`
  (`5d67ddff...`); its margin and probability metrics equal the 2026-09-10 run
  (`models/wf_shrink_2023_2025_on/`) to four decimals. Compare new feature work only against a
  reference arm run on the same build and code version.

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

Done so far in the feature-engineering workstream (see `ARCHIVE.md`): Milestone 45 (play-by-play
EPA families), 46 (schedule-adjusted strength), 49 (continuous early-season shrinkage), the first
phase of 43 (current-season Bradley-Terry defaults), 51 (power rankings on the adjusted
composite), 52 (the total head: fixed, still behind the market line, totals labelled
diagnostic-only), and task 56.4 (calibration window across the season boundary). Execution order:

1. Milestone 53 - QB per-dropback EPA families for the expected starter (**next**; the 2026 Week 2
   weekly run is due between Monday night 2026-09-14 and Thursday 2026-09-17)
2. Milestone 54 - PBP situational stats replace the TeamRankings stat scrape
3. Milestone 55 - Off-season configuration sweep, after the feature work lands
4. Milestone 56 - Weekly orchestration residuals
5. Milestone 57 - Ensembles and alternative models (parked until the user reopens it)

The open follow-ups below are not milestones. Pick them up when their area is next touched, or
promote one to a milestone when it grows.

Rules for every feature milestone:

- Land each family as one ablatable feature group with a walk-forward switch (mirror
  `--disable-trend-features`).
- Report a `2023-2025` walk-forward comparison with the group on and off before marking done.
- Keep the invariant output schema: when a source is missing for a season, emit nulls.
- XGBoost margin/total remains the only model family in scope.
- The method borrowed from `nfl-sos-ratings` is its head-to-head-excluded opponent profiling
  (`feature_crosswalk.md` section 3.1): it landed as the weekly ridge snapshot (its all-hops form)
  plus a one-hop schedule-strength companion, and the QB milestone carries the same two lenses.
- Keep walk-forward comparison artifacts under `models/`; numbers reported in these files must be
  auditable from disk.

---

## Milestone 53 - QB per-dropback EPA families for the expected starter

Formerly Milestone 47. Goal: give the model the expected starter's per-dropback production instead
of only the nfeloqb value/Elo pair.

Tasks:

Tasks 53.1-53.5 (identity bridge, quarterback-game aggregation, pre-game rates, join and schema,
leakage audit and walk-forward) are done and archived under "Milestone 53 (partial)" in
`ARCHIVE.md` (2026-09-11, version `0.7.0`). The on/off walk-forward is a statistical tie; see the
follow-ups below.

- [ ] 53.6 Schedule lenses (`feature_crosswalk.md` section 3.1 applied to QBs):
      `qb_faced_pass_def_adj`, the dropback-weighted mean of the faced defenses' pre-week ridge
      pass-defense coefficient from Milestone 46 (the sos `QSoS` construct), and the one-hop
      `qb_faced_pass_def_raw`, the faced defenses' EPA per dropback allowed from prior-week games
      excluding games against the QB's team.
- [ ] 53.7 Optional phase 2: opponent-adjusted QB EPA via a dropback-weighted ridge against faced
      defenses (design in `nfl-sos-ratings/simultaneous_adjustment.solve_qb_stat_ridge`).

Acceptance:

- [x] Unmatched-QB rate is reported and below an agreed threshold for `2006+`. The ETL logs it per
      side; on the 2026-09-11 rebuild it is `0` of `7533` rows on both sides, every season.
- [ ] Walk-forward table recorded; all gates green. Recorded for 53.1-53.5 (statistical tie,
      `ARCHIVE.md`); 53.6 adds its own on/off arm before this is checked.

---

## Milestone 54 - PBP situational stats replace the TeamRankings stat scrape

Formerly Milestone 48. Goal: compute third/fourth-down, red-zone, and two-point rates (offense and
allowed) from the Milestone 45 counts so those eight columns no longer depend on scraping and extend
to 1999.

Tasks:

- [ ] 54.1 Derive the rates from PBP counts in `_compute_derived_metrics`. Fix the `red_zone_tds`
      attribution first (see the Milestone 45 follow-ups below).
- [ ] 54.2 Keep the existing TR column names and switch the source, behind a transitional
      `--tr-stats-source pbp|scrape` option; the TeamRankings ratings scrape is unchanged.
- [ ] 54.3 Sanity-compare PBP values against scraped values for `2010-2025`; walk-forward check;
      update README data sources.

Acceptance:

- [ ] The eight situational columns are populated for `1999+` offline; walk-forward is not worse.

---

## Milestone 55 - Off-season configuration sweep + lock default settings

Formerly Milestone 39, with former Milestone 40 folded in. Deferred until the feature milestones
land, because new feature families would invalidate sweep results.

Goal: run an objective, repeatable sweep of modeling configurations under the canonical evaluation
protocol, then write the selected configuration as the default for weekly runs.

Tasks:

- [ ] 55.1 Define a sweep config schema (JSON or YAML) covering model kind (`margin_total`,
      `blended_margin_total`), calibration, market mode and probability source, blend method,
      weight and clamp grids, uncertainty, tuning, and XGBoost params including GPU preference.
- [ ] 55.2 Implement `scripts/config_sweep.py` (or `--mode sweep` in `scripts/weekly_run.py`) that
      runs walk-forward per config, writes `sweep_summary.csv` and `best_config.json`, supports
      resume by dataset hash plus config hash (per-week fold checkpoints already exist), and always
      includes a baseline row with market blending and clamping off, with deltas versus that
      baseline.
- [ ] 55.3 Include `PRIOR_BLEND_GAMES` (`K`, for example `2`, `4`, `8`) in the sweep; it was reused
      from the strength blend rather than chosen (see the Milestone 49 follow-ups).
- [ ] 55.4 Add `xgb_device=auto` (prefer `cuda`, else CPU) used identically in evaluation and final
      training.
- [ ] 55.5 Decide the `ScoreModel` fate: document as experimental or deprecate cleanly.
- [ ] 55.6 Add the stability view by season and week bucket, and a "recommended defaults" section.

Note for any tuning: until the shared early-stopping callback was fixed (version `0.6.2`), every
Optuna trial's total head stopped after one round, so the `combined_mae` objective scored a
crippled total and every existing tuned parameter set was in effect chosen on the margin head
alone. Do not reuse old `tune_best_params_out` files or Optuna studies; re-tune from scratch.

Acceptance:

- [ ] One command plus one config file produce the sweep summary and `best_config.json`, and
      `scripts/weekly_run.py --defaults-path best_config.json` runs end to end.

---

## Milestone 56 - Weekly orchestration residuals

Formerly Milestone 41.

- [ ] 56.1 Add data-refresh pass-through (`--data-min-season` / `--data-max-season` or a generic
      `--data-collection-args`) to `scripts/weekly_run.py`; the same pass-through carries the
      `--stat-prior-blend*` flags, which it cannot set today.
- [ ] 56.2 Decide and document how postseason games enter evaluation and training; when the
      prediction week is postseason, default the power-rankings through-week to the last
      regular-season week.
- [ ] 56.3 Wire sweep-selected defaults once Milestone 55 lands; confirm resume behavior.

Task 56.4 (the in-season calibration window rolls back across the season boundary) is done and
archived under "Milestone 56 (partial)" in `ARCHIVE.md`.

Acceptance:

- [ ] One command produces the complete weekly package from scratch and re-running skips work
      whose inputs and config did not change.

---

## Milestone 57 - Ensembles and alternative models (parked)

Formerly Milestone 42. Parked by user direction on 2026-09-09: XGBoost margin/total remains the
primary model and benchmark. Reopen only when the user asks. Original scope: direct win-probability
classifier, logit-space probability ensemble, optional LightGBM/CatBoost extras, season-phase
specialization.

---

## Open follow-ups from completed milestones

Each group names the archived milestone it came from; the milestone's full record is in
`ARCHIVE.md`. Resolved items have moved there.

### From Milestone 45 (play-by-play EPA families)

- [ ] `red_zone_tds` counts a touchdown by either team on a red-zone play, so a pick-six is credited
      to the offense. It is computed but not published today; fix before Milestone 54 publishes it.
- [ ] Half of `PBP_COUNT_COLUMNS` (third/fourth down, red zone, two-point, total plays) is joined,
      aggregated and regressed but never published. Publish it in Milestone 54 or stop carrying it.
- [ ] The play-by-play cache key has no schema version, so growing `constants.PBP_COLUMNS` will not
      invalidate existing per-season caches. Same latent issue as `load_team_stats`.

### From Milestone 46 (schedule-adjusted strength)

- [ ] `strength_games_played_diff` has a gain-based importance of exactly `0.0`: the two teams in a
      game have almost always played the same number of games, so the diff is a constant zero
      outside bye weeks. Drop the `_diff` companion (keep the per-team columns, which do rank) the
      next time the strength schema is touched.
- [ ] The raw `adj_off_*` / `adj_def_*` columns drift in scale across a season because the frozen
      ridge penalty shrinks harder when fewer games have been played (about 30% of true magnitude
      at 4 games, about 50% by 17). They rank far below the standardized composite. Consider either
      a games-aware penalty or publishing only the composite plus `adj_srs`, and measure it.
- [ ] `sos_played_raw` is null for every week-1 and week-2 row (11.7% of the dataset) because a
      week-2 opponent's only prior game is the one against the subject. That is the method being
      correct, but a documented fallback (prior-season profile, or the adjusted lens) would make
      the column usable in the two weeks where schedule strength is least knowable.
- [ ] Schedule-strength columns are not bit-reproducible across identical rebuilds: Polars parallel
      `group_by` summation order moves the last 1-2 ULP. The ridge and SRS columns are exactly
      stable. This is pre-existing (`aggregate_team_stats_to_week` has the same property) but it
      does mean the dataset fingerprint in the model artifact contract changes across identical
      runs. Worth a line in the artifact contract docs.

### From Milestone 49 (continuous early-season shrinkage)

- [ ] `games_played` publishes `17` for a week-1 fallback row (the prior's game count) and `1` for
      a blended week-2 row whose values are 80% prior, so it tells the model the opposite of how
      much evidence stands behind the numbers. Candidates: an effective-games column such as
      `games + K * (1 - WEEK1_REGRESSION_FACTOR)`, or a separate `stat_prior_weight`. Needs its own
      ETL rebuild and from-week-1 walk-forward. The strongest candidate for the next feature slot.
- [ ] Weeks 3-18 margin MAE got worse with the blend (`9.8952` to `9.9578`) and season Brier is
      slightly worse in 2024 and 2025. The `K` check is task 55.3.
- [ ] Choose the OpenMP wait policy automatically in the walk-forward entry points (default when
      idle, `PASSIVE` under load) instead of relying on the operator; see `AGENTS.md`. It changes
      scheduling only, never results.
- [ ] `models/wf_checkpoints/` grows by a few hundred KB per distinct run and is never pruned. Any
      edit under `nfl_predictor/ml/` changes the fingerprint by design, so stale directories pile
      up. Add a cleanup note or command once it matters.

### From task 56.4 (rolling calibration window)

- [ ] Final training early-stops every head on the calibration frame, so a weekly run with the
      default four in-season calibration weeks and no calibration season stops on roughly 50-64
      games. The Week-2 smoke run (`models/smoke_20260911`, market-anchored) stopped its margin
      head at iteration `1` on the 50-game window; the workaround run, calibrated on the 2025 and
      2026 seasons (274 games), stopped at `189` (`models/smoke_20260911_workaround/`), and
      `models/weekly_2025_week_22` (8 weeks) at `9`. This predates the window fix (four weeks of
      one season are just as small). Candidate: early-stop on a larger window than the
      calibrator uses, for example the last full season plus the calibration window, and
      measure it in walk-forward.
