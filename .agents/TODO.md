# TODO - Active Work for nfl-predictor

This file is the **authoritative worklist** for the repo and contains **active** work only.
Completed milestones live in `ARCHIVE.md` (same directory). Agent workflow and guardrails live in
`../AGENTS.md`. The cross-repo review that motivates the current ordering lives in
`feature_crosswalk.md`.

- Completed work should be moved to `ARCHIVE.md` (with dates/notes).
- Milestone numbering is authoritative in `ARCHIVE.md`. Active milestones run through 50, so new
  work starts at 51.

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
   - `markdownlint .` (CI installs `markdownlint-cli`; the local binary is `markdownlint-cli2`,
     so run `markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings"` here)
   - `uv lock --check`
   - `uv sync --check --active`

1. **Update docs** where behavior changes (README/AGENTS/CHANGELOG), and update TODO/ARCHIVE.
1. **Commit** (when asked) with Conventional Commits subjects, `type(scope): summary`, per
   `AGENTS.md`.

### Always use the repo venv

Agents and humans should not rely on the shell activation state.

- Do **not** run `python`, `pip`, `pytest`, `ruff`, `pyright`, or `ty` without the venv prefix.
- Use `.venv/bin/python ...` or the tool-specific binary under `.venv/bin/`.
- Use `uv ...` from `PATH` for dependency management and environment sync.

### Current validated baseline (2026-09-10, version `0.4.0`)

- `.venv/bin/ruff format .`, `.venv/bin/ruff check .`, `.venv/bin/ty check .`, and
  `.venv/bin/pyright .` pass cleanly.
- `.venv/bin/python -m pytest` passes (`558 passed`) with coverage `90.83%` against the enforced
  `90%` floor.
- `markdownlint-cli2`, `uv lock --check`, and `uv sync --check --active` pass.
- `data/completed_games_ml.csv` covers `1999-2025` (`7260` rows, `498` columns) and
  `data/predict/week_01_games_to_predict.csv` holds `16` rows for 2026 Week 1. This build dates
  from the 2026-09-09 17:28 Week 1 refresh and fingerprints to `5b6af6aa...`.
- The leakage audit passed on the `0.4.0` schema (`463` features, `0` findings).
- The walk-forward benchmark table lives in `AGENTS.md`. **Its arms ran on an earlier build**
  (dataset hash `668368d8...`), so before comparing new feature work, re-run the reference arm on
  the build you are measuring against and keep both reports under `models/`.

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

1. Milestone 45 - PBP foundation + per-snap team EPA families (done 2026-09-09)
2. Milestone 46 - Weekly schedule-adjusted team strength (done 2026-09-09)
3. Milestone 49 - Continuous early-season shrinkage (**next**; 2026 Week 2 kicks off 2026-09-17,
   and week 2 is the week this defect wrecks, so it moved ahead of 43 phase 2 on 2026-09-10)
4. Milestone 43 phase 2 - Power rankings on the adjusted composite (43.1 done 2026-09-09)
5. Milestone 50 - The total/over-under head carries almost no signal
6. Milestone 47 - QB per-dropback EPA families for the expected starter
7. Milestone 48 - PBP situational stats replace the TeamRankings stat scrape
8. Milestone 39 (with 40 folded in) - Off-season configuration sweep, after the feature work lands
9. Milestone 41 residuals - orchestration polish
10. Milestone 42 - parked (alternative model families) until the user reopens it

Rules for every feature milestone:

- Land each family as one ablatable feature group with a walk-forward switch (mirror
  `--disable-trend-features`).
- Report a `2023-2025` walk-forward comparison with the group on and off before marking done.
- Keep the invariant output schema: when a source is missing for a season, emit nulls.
- XGBoost margin/total remains the only model family in scope.
- The method borrowed from `nfl-sos-ratings` is its head-to-head-excluded opponent profiling
  (`feature_crosswalk.md` section 3.1): it lands as the weekly ridge snapshot (its all-hops form)
  plus a one-hop schedule-strength companion, and the QB milestone carries the same two lenses.
- Keep walk-forward comparison artifacts under `models/`; numbers reported in these files must be
  auditable from disk.

---

## Follow-ups inherited from Milestone 45 (completed 2026-09-09)

The milestone itself lives in `ARCHIVE.md` with the walk-forward table, ETL timing, null-rate
summary, and the four defects found and fixed along the way.

Outcome to carry forward: the family is leakage-safe, ablatable, and costs almost nothing to build
(0.35s to aggregate 1.2M plays), but it is **not** a win on the primary selection metric. Brier and
log loss are marginally worse with the group on; margin MAE improves in 3 of 3 seasons and
calibration improves slightly. Raw per-snap EPA is unadjusted for opponent, which is the most likely
reason it adds little over the existing Elo and TeamRankings predictive ratings. Milestone 46 is the
direct test of that hypothesis: the same inputs, opponent-adjusted.

Open follow-ups inherited from this milestone:

- [x] Resolved: the recorded reference (Brier `0.2312`, log loss `0.7352`, pick accuracy `0.6833`,
      margin MAE `9.8954`) is **not reproducible** on this machine. Re-running the default config
      against the untouched pre-change dataset gives Brier `0.2300`, log loss `0.7501`, pick
      accuracy `0.6833`, margin MAE `9.9705` - a larger log-loss gap than the rebuilt dataset
      produces. `--xgb-tree-method hist` and `rushing_epa` were both ruled out by controls. The
      milestone's changes did not regress the baseline; the old reference numbers were recorded
      under a configuration or environment that is no longer reproducible.
- [ ] `red_zone_tds` counts a touchdown by either team on a red-zone play, so a pick-six is credited
      to the offense. It is computed but not published today; fix before Milestone 48 publishes it.
- [ ] Half of `PBP_COUNT_COLUMNS` (third/fourth down, red zone, two-point, total plays) is joined,
      aggregated and regressed but never published. Publish it in Milestone 48 or stop carrying it.
- [ ] The play-by-play cache key has no schema version, so growing `constants.PBP_COLUMNS` will not
      invalidate existing per-season caches. Same latent issue as `load_team_stats`.
- [x] Resolved: the leakage perturbation test covered a regular-season week only. Playoff-branch
      and Week-1-fallback equivalents now exist for both the play-by-play family and the
      schedule-adjusted strength family, and each was mutation-verified by breaking the
      corresponding cutoff and confirming the matching test fails.
- [x] The on/off walk-forward arms were written with `--out-json` into a temporary directory and
      are not on disk. A 2026-09-09 review re-ran both arms and reproduced the numbers exactly;
      reports now live in `models/review_wf_2023_2025_pbp_{off,on}/`. Keep future arms under
      `models/`.

---

## Milestone 46 - Weekly schedule-adjusted team strength

**Completed 2026-09-09.** Moved to `ARCHIVE.md` with the walk-forward table, the ridge-penalty
provenance, null-rate summary, sanity checks, and the five defects found and fixed along the way
(two of them leaks found by independent review, not by the original tests).

Outcome to carry forward: **the opponent-adjustment hypothesis held on the primary metric.** With
the strength group on, Brier improves to `0.2277` from `0.2312` with it off and `0.2320` with both
feature groups off, and log loss to `0.7431` from `0.7495` / `0.7493`. Pick accuracy gains 2.2
points against the both-off baseline. This is the first family in the workstream to move Brier and
log loss in the right direction rather than trading them for margin MAE.

Two things did **not** improve and should not be papered over: margin MAE is slightly worse than
the strength-off arm (`9.9006` vs `9.8698`), and reliability ECE is worse than the both-off arm
(`0.1321` vs `0.1237`). The gain comes from probability *ranking*, not from sharper point estimates
or better-calibrated probabilities.

Two further results worth carrying, both of which cut against the milestone's own rationale:

- **The early-season prior blend is a tie on Brier** (`0.2277` with it on and off). It helps only
  log loss (`0.7431` vs `0.7492`) and pick accuracy (`0.6958` vs `0.6847`). It stays on as the
  default on that basis, but it is the weakest-supported piece here and `--no-strength-prior-blend`
  remains so it can be re-measured cheaply.
- **The pass/rush by offense/defense decomposition is not where the gain lives.**
  `adj_strength_composite_diff` ranks 6th and `adj_srs_diff` 7th of 533 features, but the raw
  decomposition components sit at a median rank of 183. The milestone justified itself partly on
  the decomposition giving the model matchup structure; the importance evidence does not support
  that. Follow-up 2 below is the experiment that would settle it.

Open follow-ups inherited from this milestone:

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
- [x] Resolved 2026-09-10: `uv sync --check --active` was failing because the environment still
      had the `0.3.0` package installed after the `0.4.0` version bump. A plain `uv sync` cleared
      it. Re-sync after every version bump.

---

## Milestone 49 - Continuous early-season shrinkage (next; found 2026-09-09)

Goal: season-to-date stat features in weeks 2-4 are mostly the regressed prior season and hand over
to the in-season sample as it accumulates, instead of switching from 100% prior in week 1 to 0%
prior in week 2.

Evidence (`models/wf_strength_2023_2025_from_week1/`, 48 games per week over `2023-2025`):

| window | Brier | log loss | pick acc | margin MAE |
| --- | --- | --- | --- | --- |
| week 1 only | `0.2134` | `0.6151` | `0.6042` | `9.3640` |
| week 2 only | `0.2445` | `0.6846` | `0.5208` | `8.7838` |
| weeks 3-18 | `0.2277` | `0.7431` | `0.6958` | `9.9006` |

Week 1 runs entirely on the previous season regressed toward the league mean and beats the
mid-season benchmark on both probability metrics. Week 2 runs on **one game with no shrinkage** and
drops to `0.5208` pick accuracy.

Root cause, verified in code and on the 2026-09-10 dataset: `aggregate_team_stats_to_week` takes a
plain mean over prior in-season games, and `process_week` builds the regressed prior-season frame
only for `teams_needing_fallback`, the teams with *zero* in-season games. So `away_games_played` is
`17` in week 1 and `1` in week 2, and 2024 `away_success_rate` has std `0.0807` (range
`0.250-0.569`) at week 2 against `0.0359` (`0.375-0.490`) at week 16. This affects every
season-to-date family that flows through `team_stats_df` (nflreadpy stats and the play-by-play
counts alike). The strength family is exempt: its own blend already sits at 80% prior in week 2.

Design (decided 2026-09-10; details in `next_agent_session_prompt.md`): blend the aggregated
per-game means as `w * in_season + (1 - w) * regressed_prior` with
`w = games_played / (games_played + K)`, `K = PRIOR_BLEND_GAMES = 4.0` shared with the strength
snapshot and promoted to `constants.py`, then `recompute_derived_metrics` so rates stay ratios of
blended sums. Zero games gives `w = 0`, so week-1 rows are unchanged by construction.
`games_played` keeps its current semantics in this cut.

Tasks:

- [ ] 49.1 Characterization tests first: week-1 rows bit-identical before and after; a one-game
      team's week-2 stat equals `0.2 * in_season + 0.8 * prior` at `K = 4`; rates are recomputed
      from blended sums, not blended directly; the first season in the run is untouched;
      `--no-stat-prior-blend` reproduces today's output exactly.
- [ ] 49.2 Implement the blend in `process_week`, building the regressed prior once per season
      (a `prior_season_stats` argument alongside `prior_strength_snapshot`, computed inside when
      `None`).
- [ ] 49.3 CLI: `--stat-prior-blend-games` (default `4`) and `--no-stat-prior-blend` on
      `nfl_predictor.data_collection`, mirroring `--strength-prior-blend`.
- [ ] 49.4 Back up `data/*.csv`, rebuild, confirm the 2024 week-2 dispersion collapses toward the
      week-16 spread and week-1 rows match the backup; run the leakage audit.
- [ ] 49.5 Walk-forward from week 1 (`--wf-start-week 1`, `--eval-last-n-seasons 3`), off arm on
      the pre-change build and on arm on the new build, both under `models/wf_shrink_2023_2025_*`.
      Report weeks 1, 2, and 3-18 separately; the aggregate hides a 2-of-18-weeks change.
- [ ] 49.6 README, AGENTS baseline note, CHANGELOG `[Unreleased]`, move to `ARCHIVE.md`.

Acceptance:

- [ ] Week 1 metrics identical on both arms; week 2 Brier and log loss move materially toward the
      weeks 3-18 numbers; weeks 3-18 do not regress. If the result is a tie or a loss, the switch
      ships default-off and the table is recorded anyway.
- [ ] Follow-up recorded: whether `games_played` should publish the effective sample size instead
      of the raw in-season count.

---

## Milestone 50 - The total (over/under) head carries almost no signal (found 2026-09-09)

Predicted totals for the 2026 Week 1 slate all land between `43.9` and `44.1` while market totals
for the same games range `40.5` to `47.5`. The model is effectively predicting the league mean for
every game. Training holdout `total_mae` is `10.9974` against a `margin_mae` of `9.8471`.

Consequence: the `total_value_side`, `total_edge_prob`, `total_confidence_1_10` and `total_ev`
columns in the betting workbook are computed from that flat prediction and are not actionable. The
spread and moneyline columns are unaffected. Do not present total-based betting recommendations as
usable until this is resolved.

Tasks:

- [ ] 50.1 Diagnose: feature importance for the total head; whether the total target is being
      learned at all (early-stopping round, train vs holdout MAE); whether the pruning or feature
      selection step is dropping total-relevant columns.
- [ ] 50.2 Test market-total anchoring (residual training against `market_total_line`) in
      walk-forward, and whether the total head deserves a different feature set from the margin head.
- [ ] 50.3 Record the walk-forward table; either fix the default or mark the total columns of the
      betting workbook as diagnostic-only in the report and README.

Acceptance:

- [ ] Weekly predicted totals span a range comparable to the market's, or the total outputs are
      explicitly labelled non-actionable.

---

## Milestone 43 - Power rankings measure current-season strength (phase 2 pending)

Goal: a Week `N` ranking reflects how strong teams are going into week `N`. Verified current
behavior: a Bradley-Terry fit over every season since 1999 with equal weights, fixed `0.97 / 0.03`
targets, and future games filled with model probabilities.

Tasks:

- [x] 43.1 Landed 2026-09-09 in `scripts/power_rankings.py` and
      `nfl_predictor/reporting/power_rankings.py`: `--ratings-window-seasons` (default `2`),
      `--ratings-prior-season-weight` (default `0.25`) via sample weights in
      `fit_bradley_terry_ratings`, margin-based targets by default (`--ratings-target`), future
      model-probability rows excluded from the strength fit (`--ratings-include-future`), and
      `--legacy-franchise-fit` reproducing the old output exactly (pinned by a test).
      Evidence it mattered: for 2024 through week 18 the old fit ranked a 4-13 New England first;
      the new default ranks DET, BAL, BUF, GB, PHI, matching both the season's results and the
      schedule-adjusted snapshot.
      **Not done for `scripts/weekly_run.py`:** it calls `_build_games_for_ratings` positionally
      without the new arguments, so it silently inherits the new defaults but exposes no flags to
      override them and has no `--legacy-franchise-fit`. Wire them through when convenient.
- [ ] 43.2 Default the ranking to the ETL adjusted composite for
      `(season, through_week + 1)` rows, map to the existing 1-10 and 0-10 scales, and publish the
      components next to the rank. Keep Bradley-Terry available as `--method bradley_terry`.
- [ ] 43.3 Leave projected standings as record plus model win probabilities. Label or remove
      `golden_command._build_pregame_power_rankings` so one ranking artifact is canonical.
- [ ] 43.4 Tests: recency weighting shifts ratings toward recent results; window logic excludes
      older seasons; a synthetic breakout team ranks first late in the season.
- [ ] 43.5 README and `--help` explain "current-season" versus "franchise" rankings (the 43.1
      flags are documented in README and `CHANGELOG.md` as of 2026-09-10; extend, do not redo).

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
- [ ] 47.6 Schedule lenses (`feature_crosswalk.md` section 3.1 applied to QBs):
      `qb_faced_pass_def_adj`, the dropback-weighted mean of the faced defenses' pre-week ridge
      pass-defense coefficient from Milestone 46 (the sos `QSoS` construct), and the one-hop
      `qb_faced_pass_def_raw`, the faced defenses' EPA per dropback allowed from prior-week games
      excluding games against the QB's team.
- [ ] 47.7 Optional phase 2: opponent-adjusted QB EPA via a dropback-weighted ridge against faced
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
