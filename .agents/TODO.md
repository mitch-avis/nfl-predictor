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
- Milestones up to 50 are archived or retired (the last cleanup, 2026-09-10, is recorded at the top
  of `ARCHIVE.md`). Active milestones run 51 to 57, so new work starts at 58.

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
6. **Update docs** where behavior changes (README/AGENTS/CHANGELOG), and update TODO/ARCHIVE.
7. **Commit** (when asked) with Conventional Commits subjects, `type(scope): summary`, per
   `AGENTS.md`.

### Always use the repo venv

Agents and humans should not rely on the shell activation state.

- Do **not** run `python`, `pip`, `pytest`, `ruff`, `pyright`, or `ty` without the venv prefix.
- Use `.venv/bin/python ...` or the tool-specific binary under `.venv/bin/`.
- Use `uv ...` from `PATH` for dependency management and environment sync.

### Current validated baseline (2026-09-10, version `0.5.0`)

- `.venv/bin/ruff format .`, `.venv/bin/ruff check .`, `.venv/bin/ty check .`, and
  `.venv/bin/pyright .` pass cleanly.
- `.venv/bin/python -m pytest` passes (`590 passed`) with coverage `91.04%` against the enforced
  `90%` floor.
- `markdownlint-cli2`, `uv lock --check`, and `uv sync --check --active` pass.
- `.agents/skills/` is a separate git clone of agent skills. It is gitignored and excluded from
  ruff and markdownlint (`.markdownlintignore`); pyright already skips dot-directories and ty only
  checks `nfl_predictor`, `scripts`, and `tests`.
- `data/completed_games_ml.csv` is the stat-prior-blend build (2026-09-10 11:17): `7261` rows
  (`1999-2025` plus the 2026 opener, NE at SEA), `498` columns, fingerprint `e46f1be9...`.
  `data/predict/week_01_games_to_predict.csv` holds the `15` remaining 2026 Week 1 games. The
  pre-change build is kept as `data/completed_games_ml.pre_m49.csv` (`5b6af6aa...`) and
  `data/backup_pre_m49/`.
- The leakage audit passed on this build (`463` features, `0` findings).
- The walk-forward benchmark lives in `AGENTS.md` and was re-measured on this build
  (`models/wf_shrink_2023_2025_on/`). Compare new feature work only against a reference arm run on
  the same build and code version.

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
EPA families), 46 (schedule-adjusted strength), 49 (continuous early-season shrinkage), and the
first phase of 43 (current-season Bradley-Terry defaults). Execution order:

1. Milestone 51 - Power rankings on the adjusted composite (**next**; 51.1 has a design fork the
   user settles first)
2. Milestone 52 - The total (over/under) head carries almost no signal
3. Milestone 53 - QB per-dropback EPA families for the expected starter
4. Milestone 54 - PBP situational stats replace the TeamRankings stat scrape
5. Milestone 55 - Off-season configuration sweep, after the feature work lands
6. Milestone 56 - Weekly orchestration residuals
7. Milestone 57 - Ensembles and alternative models (parked until the user reopens it)

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

## Milestone 51 - Power rankings on the adjusted composite

Formerly phase 2 of Milestone 43. Phase 1 (current-season Bradley-Terry defaults in
`scripts/power_rankings.py`) is archived.

Goal: a Week `N` ranking reflects how strong teams are going into week `N`, read from the ETL's
schedule-adjusted composite instead of a separate Bradley-Terry fit.

Tasks:

- [ ] 51.1 Default the ranking to the adjusted composite for `(season, through_week + 1)`, map it
      to the existing 1-10 and 0-10 scales with a documented transform (the composite is a
      within-snapshot z-score, not a win probability), and publish the components next to the
      rank. Keep Bradley-Terry available as `--method bradley_terry`.
      **Design fork (found 2026-09-10; the user decides before coding):** the composite reaches the
      ranking script today only through game rows, and a team on bye in week `through_week + 1` has
      no game row that week. Its next row is solved from games that include week
      `through_week + 1`, which leaks later results into a historical rerun (harmless live, where
      that week is unplayed). Options:
      (a) use the team's latest row at or before `through_week + 1`: leak-free, no new data path,
      but one game stale for bye teams;
      (b) solve the snapshot inside the script with `strength_snapshot.build_strength_snapshot`:
      exact, but the solver needs the per-team-game frame that only `collect_all_data` assembles
      (nflreadpy team stats, play-by-play aggregation, scoring, opponent columns), so that chain
      must be factored out of the ETL and called from a reporting script, which then needs the
      network for the current season and can disagree with the build the model predicted from;
      (c) have the ETL write the per-team weekly snapshot it already computes (`process_week`
      solves every team on the season's schedule, bye teams included, then keeps only the teams
      with a game) to a new file such as `data/strength_snapshots.csv`: exact, offline and cheap to
      read, and consistent with the model's features by construction, at the cost of one ETL
      rebuild (about 10 minutes, no walk-forward, since training rows do not change) and a new
      artifact to document. Recommended: (c).
- [ ] 51.2 Leave projected standings as record plus model win probabilities. Label or remove
      `golden_command._build_pregame_power_rankings` so one ranking artifact is canonical.
- [ ] 51.3 Tests: the composite ranking ranks an obviously strongest synthetic team first; a bye
      team is handled by the chosen option and never reads a later week; the scale mappings are
      monotone and bounded; `--method bradley_terry` reproduces today's default output; a synthetic
      breakout team ranks first late in the season.
- [ ] 51.4 Wire the ranking flags through `scripts/weekly_run.py`, which today calls
      `_build_games_for_ratings` without them (`--ratings-window-seasons`,
      `--ratings-prior-season-weight`, `--ratings-target`, `--ratings-include-future`,
      `--legacy-franchise-fit`, and the new `--method`).
- [ ] 51.5 README and `--help` explain "current-season" versus "franchise" rankings (the phase-1
      flags are already documented in README and `CHANGELOG.md`; extend, do not redo).

Acceptance:

- [ ] Default weekly-run rankings for a late-season week align with current-season results and
      recent form; the franchise view remains reachable by explicit flag.
- [ ] Sanity anchor, not a test oracle: the 2024 pre-week-18 top five is BAL, DET, PHI, BUF, GB
      on the composite and DET, BAL, BUF, GB, PHI on the current Bradley-Terry default.

---

## Milestone 52 - The total (over/under) head carries almost no signal

Formerly Milestone 50 (found 2026-09-09). Predicted totals for the 2026 Week 1 slate all land
between `43.9` and `44.1` while market totals for the same games range `40.5` to `47.5`. The model
is effectively predicting the league mean for every game. Training holdout `total_mae` is `10.9974`
against a `margin_mae` of `9.8471`.

Consequence: the `total_value_side`, `total_edge_prob`, `total_confidence_1_10` and `total_ev`
columns in the betting workbook are computed from that flat prediction and are not actionable. The
spread and moneyline columns are unaffected. Do not present total-based betting recommendations as
usable until this is resolved.

Tasks:

- [ ] 52.1 Diagnose: feature importance for the total head; whether the total target is being
      learned at all (early-stopping round, train vs holdout MAE); whether the pruning or feature
      selection step is dropping total-relevant columns.
- [ ] 52.2 Test market-total anchoring (residual training against `market_total_line`) in
      walk-forward, and whether the total head deserves a different feature set from the margin head.
- [ ] 52.3 Record the walk-forward table; either fix the default or mark the total columns of the
      betting workbook as diagnostic-only in the report and README.

Acceptance:

- [ ] Weekly predicted totals span a range comparable to the market's, or the total outputs are
      explicitly labelled non-actionable.

---

## Milestone 53 - QB per-dropback EPA families for the expected starter

Formerly Milestone 47. Goal: give the model the expected starter's per-dropback production instead
of only the nfeloqb value/Elo pair.

Tasks:

- [ ] 53.1 Identity bridge: read a copied `data/qb_meta_data.csv` (from
      `../nfeloqb/Other Data/meta_data.csv`, read-only contract) to map `away_qb`/`home_qb` names
      to GSIS ids; fallback name match against PBP `passer_player_name`; log the unmatched rate;
      tests for aliases and misses.
- [ ] 53.2 QB-game aggregation from PBP by `passer_player_id`: dropbacks, attempts, completions,
      pass yards, TDs, INTs, sacks, sack yards, `qb_epa` sum, CPOE mean (2006+). Formulas in
      docstrings; fixture tests.
- [ ] 53.3 Career-to-date and season-to-date rates strictly before the game date, plus a rolling
      dropback window for recency; new-starter fallback to a regressed league mean with a
      `qb_history_dropbacks` column so the model can see sample size.
- [ ] 53.4 Join for away/home plus diffs; constants; finalization; null policy documented.
- [ ] 53.5 Leakage audit and walk-forward ablation; record results.
- [ ] 53.6 Schedule lenses (`feature_crosswalk.md` section 3.1 applied to QBs):
      `qb_faced_pass_def_adj`, the dropback-weighted mean of the faced defenses' pre-week ridge
      pass-defense coefficient from Milestone 46 (the sos `QSoS` construct), and the one-hop
      `qb_faced_pass_def_raw`, the faced defenses' EPA per dropback allowed from prior-week games
      excluding games against the QB's team.
- [ ] 53.7 Optional phase 2: opponent-adjusted QB EPA via a dropback-weighted ridge against faced
      defenses (design in `nfl-sos-ratings/simultaneous_adjustment.solve_qb_stat_ridge`).

Acceptance:

- [ ] Unmatched-QB rate is reported and below an agreed threshold for `2006+`.
- [ ] Walk-forward table recorded; all gates green.

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
