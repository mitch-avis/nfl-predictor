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
  of `ARCHIVE.md`). Active milestones run 53 to 59 (58 is the web UI, added 2026-09-11; 59 is the
  benchmark-instrument follow-up from the 2026-09-18 audit), so new work starts at 60.

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

### Current validated baseline (2026-09-11, version `0.8.0`, `main` at `7c105bc`)

- `.venv/bin/ruff format .`, `.venv/bin/ruff check .`, `.venv/bin/ty check .`, and
  `.venv/bin/pyright .` pass cleanly.
- `.venv/bin/python -m pytest` passes (`814 passed`, `tests/api/` included) with coverage
  `92.90%` against the enforced `90%` floor.
- `markdownlint-cli2`, `uv lock --check`, and `uv sync --check --active` pass. The frontend gate
  in `web/` (lint, typecheck, `22` vitest tests, build) passes after `npm ci` under Node 26.
- `main` contains `fix/calibration-window-total-head` (`0.6.1`-`0.7.1`) and `feat/web-ui`
  (`0.8.0`), both merged and pushed on 2026-09-11. New work starts on a fresh branch off `main`.
- `.agents/skills/` is a separate git clone of agent skills. It is gitignored and excluded from
  ruff and markdownlint (`.markdownlintignore`); pyright already skips dot-directories and ty only
  checks `nfl_predictor`, `scripts`, and `tests`.
- `data/completed_games_ml.csv` is the 2026-09-17 21:39 rebuild on the `0.10.0` schema (schedule
  lenses removed): `7277` rows (`1999-2025` plus the 2026 Week 1 games), `519` columns,
  fingerprint `0cecc2e3...`; leakage audit `484` features, `0` flags. The `525`-column build it
  replaced is in `data/backup_pre_m53_6_drop/`. `data/completed_games_ml.m53_through_2025.csv`
  (`7261` rows, `06a7a34d...`, cut from the 2026-09-11 06:47 build `acaa2892...`) was the input
  of the Milestone 53 walk-forward arms. The 06:47 build
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
diagnostic-only), task 56.4 (calibration window across the season boundary), and the web UI's
phases 0-3 (Milestone 58, merged as `0.8.0`). Execution order:

1. Milestone 59 - Benchmark instrument, calibration and fit parity (user's decision 2026-09-18:
   first, because nothing else can be measured until it lands; tasks 59.1-59.3 before any feature
   work, 59.4-59.6 alongside)
2. Milestone 54 - PBP-first team-game skeleton and situational stats (54.0 first)
3. Milestone 53 - task 53.7 (defense-adjusted quarterback rate), measured on the 59.1 instrument
4. Milestone 55 - Off-season configuration sweep, on the 59.1 instrument
5. Milestone 56 - Weekly orchestration residuals
6. Milestone 57 - Ensembles and alternative models (parked until the user reopens it)
7. Milestone 58 - Web UI, phases 4-6 (pool helpers, team and QB pages, live odds design); runs
   alongside the ML work whenever the user asks for it, on its own branch

The open follow-ups below are not milestones. Pick them up when their area is next touched, or
promote one to a milestone when it grows.

Rules for every feature milestone:

- Land each family as one ablatable feature group with a walk-forward switch (mirror
  `--disable-trend-features`).
- Report a walk-forward comparison with the group on and off before marking done, read on the
  deterministic and market columns (task 59.1), over enough seasons to resolve the effect claimed.
- Keep the invariant output schema: when a source is missing for a season, emit nulls.
- XGBoost margin/total remains the only model family in scope.
- The method borrowed from `nfl-sos-ratings` is its head-to-head-excluded opponent profiling
  (`feature_crosswalk.md` section 3.1): it landed as the weekly ridge snapshot (its all-hops form)
  plus a one-hop schedule-strength companion. The QB milestone tried the same two lenses on
  quarterbacks (task 53.6) and dropped them; its remaining form is the adjusted rate (53.7).
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

- [x] 53.6 Schedule lenses: built and measured 2026-09-11 (`0.9.0`; no gain), dropped from the
      schema 2026-09-17 (`0.10.0`) by the user's decision. Record and reasoning in `ARCHIVE.md`,
      Milestone 53, "53.6". Do not reintroduce them as standalone columns; the idea's next form
      is 53.7.
- [ ] 53.7 Opponent-adjusted quarterback EPA, in two steps. First (cheap) a defense-adjusted
      rate: per quarterback game, `qb_epa_sum - dropbacks * expected_epa_allowed` where the
      expectation is the faced defense's pre-game `adj_def_pass_epa_snap` (or its one-hop
      profile), aggregated with the same `K = 300` shrinkage as `qb_dropback_epa` (career and
      last-8 windows) so it lands next to the existing rates and is measured the same way. The
      quarterback-game rows already carry `opponent_abbr` for this. Only if that shows signal,
      the ridge: dropback-weighted quarterback effects solved jointly with defenses (design in
      `nfl-sos-ratings/simultaneous_adjustment.solve_qb_stat_ridge`). Ceiling to keep in mind:
      the team-level `adj_off_pass_epa` is already opponent-adjusted, so the gain is confined to
      where the quarterback's history diverges from the team's (new and traded starters).

2026 Week 2 weekly run: completed 2026-09-17 18:36 MDT (`scripts/weekly_run.py --run-id
weekly_2026_week_02`, ETL then `--skip-data-refresh` to chain in), started too late (~18:04 MDT)
for the Thursday-night DET-at-BUF kickoff (~18:15 MDT) but predictions exist for all 16 Week 2
games including it. Selected config `hybrid_raw_prob_base_elo_blend0.20_clamp0.10`; weeks-16-2025
through-week-1-2026 calibration window as expected; `total_signal = diagnostic_only` throughout.
Outputs under `models/weekly_2026_week_02/`.

Acceptance:

- [x] Unmatched-QB rate is reported and below an agreed threshold for `2006+`. The ETL logs it per
      side; on the 2026-09-11 rebuild it is `0` of `7533` rows on both sides, every season.
- [ ] Walk-forward table recorded; all gates green. Recorded for 53.1-53.5 (statistical tie) and
      for 53.6 (no gain; `models/wf_qbsched_2023_2025_{on,off}/`), both in `ARCHIVE.md`. Gates
      green on the `0.10.0` removal.

---

## Milestone 54 - PBP-first team-game skeleton and situational stats

Formerly Milestone 48, widened on 2026-09-18 by the user's decision after the audit found that
nflverse team stats lack Jacksonville's 2001-2002 home games while play-by-play has all 16 (see
Milestone 59's findings). Goal: play-by-play becomes the primary per-team-game source; nflverse
team stats fill only what play-by-play cannot derive; the TeamRankings stat scrape goes away.

Tasks:

- [ ] 54.0 Schedule skeleton and coverage check (small; do first). Build the per-team-game frame
      from the schedule (every completed game has exactly two team rows), left-join nflverse team
      stats and the play-by-play counts onto it, and log every `(season, team)` whose team-stats
      row count differs from the schedule. Acceptance: JAX 2001 and 2002 season-to-date rows count
      16 games and `strength_games_played` reads 16 at season end; the leakage audit still passes;
      a walk-forward tie on the 59.1 instrument.
- [ ] 54.1 Derive the eight situational rates (third/fourth down, red zone, two-point; offense and
      allowed) from the play-by-play counts in `_compute_derived_metrics`. Fix the `red_zone_tds`
      attribution first (Milestone 45 follow-ups below).
- [ ] 54.2 Derive the box-score stats now taken from nflverse team stats (yards, attempts,
      completions, touchdowns, sacks, interceptions, fumbles, penalties, first downs, points) from
      play-by-play per team-game, and compare against nflverse team stats for 1999-2025 (they
      should agree where both exist; record the exceptions). Switch the source behind a
      transitional `--team-stats-source pbp|nflverse` option; keep the column names.
- [ ] 54.3 Keep the existing TR column names and switch the situational source, behind
      `--tr-stats-source pbp|scrape`; the TeamRankings ratings scrape is unchanged.
- [ ] 54.4 Sanity-compare PBP values against scraped values for `2010-2025`; walk-forward check on
      the 59.1 instrument; update README data sources and `AGENTS.md`.

Acceptance:

- [ ] No `(season, team)` has fewer season-to-date games than the schedule.
- [ ] The eight situational columns are populated for `1999+` offline; walk-forward is not worse
      on deterministic Brier against the market.

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

## Milestone 58 - Web UI: remaining phases

Added 2026-09-11. Phases 0-3 (scaffolding and auth, the read-only pages, jobs, future-week
predictions) are done and archived under "Milestone 58 (partial)" in `ARCHIVE.md`; the design,
decisions and per-phase status live in `web_ui_plan.md`. Work happens on a fresh branch off
`main` (the `../nfl-predictor-web` worktree is fine for it) and merges back after each phase.

- [ ] 58.1 Phase 4: pool helpers (confidence pool sheet, tiebreakers, survivor optimizer) per
      `web_ui_plan.md`.
- [ ] 58.2 Phase 5: team and QB pages over the `nfl-sos-ratings` Parquet outputs.
- [ ] 58.3 Phase 6 (design only): live betting and live odds.
- [ ] 58.4 Housekeeping: the `web` extra in `pyproject.toml` now duplicates the core dependency
      list; drop it (and the `--extra web` in CI and `web/README.md`) or give it a purpose. The
      `etl_full`, `validate_offline` and `validate_live` job templates read the checkout's own
      `data/` and ignore `NFLP_DATA_DIR` (see the plan's Phase 2 deviations).

Acceptance:

- [ ] Each phase ships with `tests/api/` and vitest coverage, the Python and frontend gates green,
      a `CHANGELOG.md` entry under a new version, and the plan's Status section updated.

---

## Milestone 59 - Benchmark instrument and feature audit follow-ups

Added 2026-09-18 by the review session that audited the `0.10.0` / `0.11.0` sessions and the
feature set. Evidence lives under `models/feature_audit_2026_09_18/` (`fold_importance.json`,
`feature_ranking.json`, `rescored_arms.json`, `deadweight_columns.json`, logs) and
`models/wf_deadweight_2023_2025_pruned/`. Recommended to run **before** Milestone 54, because
it changes how every feature comparison is read.

Findings:

- **The benchmark's probability metrics are dominated by Platt noise, not by features.** The
  walk-forward fits `LogisticRegression` on the last 4 weeks (about 60 games) of the eval season,
  which yields probabilities of exactly `0.0` and `1.0` (25% of weeks-3-18 predictions fall
  outside `[0.05, 0.95]`). On the `0.11.0` arm the same `predicted_margin` scores weeks 3-18 Brier
  `0.2324` / log loss `0.7612` through Platt (worse than a coin flip's `0.693`) and `0.2106` /
  `0.6087` through the deterministic map `Phi(margin / SCORE_DIFF_STD_DEV)`; the market spread
  through the same map scores `0.2099` / `0.6076`. Weeks 1-2, which skip calibration, were never
  affected. Rescoring every checkpointed arm deterministically puts all of them in a Brier band of
  `0.2082` to `0.2135` against the market's `0.2099`, and every paired arm difference the archive
  reports as a gain or a loss (strength prior blend, quarterback family, lenses, `games_played`)
  collapses to within `0.0013` with intervals covering zero (`rescored_arms.json`). Production
  trains with `calibration = elo`, so the benchmark never measured the production probability
  path either.
- **No arm beats the market.** Deterministic weeks-3-18 Brier against the market: benchmark arm
  `-0.0005` `[-0.0041, +0.0032]`, `0.11.0` arm `+0.0006` `[-0.0030, +0.0045]`. Margin MAE: every
  anchored arm is `0.05` to `0.07` points worse than the spread it anchors on (all intervals cover
  zero); the residual head's predictions correlate `0.08` with the actual-minus-spread outcome.
  Feature gain is flat: the median feature's share of total gain equals the uniform `2 / 482`,
  158 features carry half the gain, and the top of the ranking is rare-event counts
  (`special_teams_tds`, `def_fumbles`, `2pt_conversions`, `fumble_recovery_tds`, safeties). That
  is the signature of trees fitting noise on a residual target, so pruning individual columns
  cannot be measured at this sample size; the actionable lever is the instrument and the model's
  regularization, not the column list.
  The one prune the audit landed (`0.12.0`: 20 dead-weight and duplicate columns, `482` to
  `462`) is a tie in every window under both maps (weeks 3-18 deterministic Brier `0.2098`
  after / `0.2106` before, `-0.0007` `[-0.0032, +0.0016]`; table in `CHANGELOG.md`, arm
  `models/wf_deadweight_2023_2025_pruned/`), as expected for columns the trees did not use.
- **Early stopping is inert in walk-forward.** In 15 of the 18 sampled folds with a calibration
  frame the best iteration is `595-597` of `598`; the `n_estimators` cap binds and the 50-round
  patience never fires, while production final training stopped the margin head at iteration `0`
  (Week 2 run). The two paths are not the same model.
- **Upstream nflverse gap.** `load_team_stats` returns only Jacksonville's 8 road games for 2001
  and 2002 (verified live on 2026-09-18), and one game is missing for BAL/LAR 1999 and
  BUF/KC/LAC/MIA 2000. Play-by-play has all 16 JAX games, but the PBP counts left-join onto the
  team-stats skeleton, so every JAX 2001-2002 season-to-date family (stats, EPA, ridge inputs,
  `strength_games_played`, which reads `7` at week 17) is built from road games only. The
  `0.11.0` record's "agreed in 6845 of 6848 rows" compared two columns that shared this
  undercount; against the corrected record count the strength counter disagrees on 58 away and
  54 home rows.
- **Static division map.** `is_divisional_matchup`, the division/conference records, ranks and
  games-behind features use today's `TEAM_TO_DIVISION` for every season, so 1999-2001 rows (pre
  realignment: ARI in the NFC East, SEA in the AFC West, and so on) are mislabeled; the nflverse
  `div_game` flag (`division`) is correct and disagrees with `is_divisional_matchup` on 189 rows,
  all 1999-2001.
- **Near-duplicate mirrors.** `opponent_def_sacks` is `times_sacked` seen from the other sideline
  (`r = 0.998`) and `opponent_times_sacked` is `def_sacks`; `EXCLUDE_FROM_OPPONENT_STATS` already
  removes the analogous interception and turnover mirrors but not these.

Tasks (order agreed with the user on 2026-09-18; 59.1 and 59.2 first, they are the instrument):

- [ ] 59.1 Instrument. Make the walk-forward report score win probability three ways on every
      fold: the configured calibrator (kept as a diagnostic), the deterministic map
      `Phi(margin / sigma)`, and the market-implied probability from the same rows (no-vig
      moneyline, with `Phi(spread / sigma)` as the fallback when moneylines are missing). Print
      the paired model-minus-market Brier with a bootstrap interval per window (weeks 1, 2, 3-18,
      all) in the report and in `wf_compare`. Rebuild the `AGENTS.md` benchmark table from the
      deterministic columns and state the market row beside it. The rescoring script in
      `models/feature_audit_2026_09_18/` (see the `.agents/ARCHIVE.md` review note under
      Milestone 49 for the numbers) is the reference for what the numbers should be.
- [ ] 59.2 Calibration that cannot blow up. Replace the 4-week Platt fit with a calibrator fit on
      pooled out-of-fold predictions: for eval season `S`, use the walk-forward predictions of
      seasons `S-2` and `S-1` (about 540 games) plus completed weeks of `S`, never the last 60 games
      alone. Start with the one-parameter map (estimate `sigma` from those residuals), then test
      Platt on the pooled set with an L2 penalty and isotonic only past the existing 200-game
      threshold. Acceptance: on the 2023-2025 checkpoints, weeks 3-18 log loss within `0.005` of
      the deterministic `0.607`, no probability outside `[0.02, 0.98]` unless the spread exceeds
      14 points. Make `auto` resolve to this path and make production (`weekly_run`,
      `golden_command`, `betting_pipeline`) and walk-forward share one calibration function.
- [ ] 59.3 Fit parity and the early-stopping window. Today production early-stops on a 60-game
      window (the Week 2 margin head stopped at iteration `0`, so production predicted the spread)
      while walk-forward folds run to the `598` cap (patience never fires). Remove early stopping
      from in-season fits and fix `n_estimators` from a time-aware tuning (a whole prior season as
      the eval set, at least 250 games), or early-stop on that season-sized set; either way the
      two paths must call the same fit function and `metadata.json` must record
      `best_iteration` for every head, with a warning when it is `< 10` or at the cap. Re-tuning
      of the other parameters stays in Milestone 55 but must use the 59.1 instrument.
- [ ] 59.4 Season-aware divisions. `division` (nflverse `div_game`) is correct in every season and
      is the flag the model keeps; the division and conference records, ranks, games-behind and
      clinch proxies still use today's map for 1999-2001. Add the pre-2002 alignment to
      `constants` (AFC East BUF IND MIA NE NYJ; AFC Central BAL CIN CLE JAX PIT TEN; AFC West DEN
      KC LV LAC SEA; NFC East ARI DAL NYG PHI WSH; NFC Central CHI DET GB MIN TB; NFC West ATL
      CAR NO SF LAR; no HOU), select the map by season in `features.py`, and assert
      `division == is_divisional_matchup` for every season in a test. Nulling the 1999-2001 rows
      is the fallback if the map proves awkward, but the map is three seasons of known facts.
- [ ] 59.5 Noise-family ablation with the new instrument. The gain ranking is led by rare-event
      counts (`special_teams_tds`, `def_fumbles`, `fumble_recovery_tds`, `2pt_conversions`,
      `def_safeties`, `def_tds`). Run one arm with those families dropped and one with stronger
      regularization (`min_child_weight`, `gamma`) and read both on deterministic Brier against the
      market and on margin MAE against the spread, over at least six eval seasons
      (`--eval-last-n-seasons 6`) so the intervals can resolve `0.002` Brier.
- [ ] 59.6 Add `def_sacks` / `times_sacked` to `EXCLUDE_FROM_OPPONENT_STATS` when the schema is
      next rebuilt (the columns were pruned at training time in `0.12.0` as the interim step).

The team-stats skeleton fix moved to Milestone 54 as task 54.0, by the user's decision.

Acceptance:

- [ ] Every walk-forward report carries deterministic and market columns with paired intervals,
      and the `AGENTS.md` benchmark quotes them.
- [ ] Production and walk-forward fit and calibrate through the same functions; `metadata.json`
      records `best_iteration` for every head.
- [ ] Division context is correct for 1999-2001.

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

- [x] `games_played` published the prior season's count on a fallback row. Closed 2026-09-17
      (version `0.11.0`) by fixing the records join that the column's values were being suffixed
      away by, and pruning both sides as duplicates of `strength_games_played`. Neither proposed
      column was built: `games + K * (1 - WEEK1_REGRESSION_FACTOR)` is an affine shift and
      `K / (games + K)` is monotone in `games`, so neither can change a tree's splits, and the
      strength counter already published the fact correctly. Record, table and bootstrap in
      `ARCHIVE.md`, Milestone 49, "`games_played` as evidence". Do not reopen it as a new column.
- [ ] Weeks 3-18 margin MAE got worse with the blend (`9.8952` to `9.9578`) and season Brier is
      slightly worse in 2024 and 2025. The `K` check is task 55.3.
- [ ] Choose the OpenMP wait policy automatically in the walk-forward entry points (default when
      idle, `PASSIVE` under load) instead of relying on the operator; see `AGENTS.md`. It changes
      scheduling only, never results.
- [ ] `models/wf_checkpoints/` grows by a few hundred KB per distinct run and is never pruned. Any
      edit under `nfl_predictor/ml/` changes the fingerprint by design, so stale directories pile
      up. Add a cleanup note or command once it matters.

### From the 2026-09-11 code review of `0.6.1`-`0.7.0`

Fixed in `0.7.1`: the season guard, the window log line, the quarterback history refresh and
the identity warning. Still open:

- [ ] `walk_forward.select_calibration_data` still takes calibration weeks from the eval season
      only and returns an empty frame when fewer than `calibration_weeks` exist, so walk-forward
      folds for weeks 2-4 fit with no calibration frame (and no early stopping) while production
      training now rolls the window back into the previous season. The backtest therefore does
      not measure the early-season regime the weekly model uses. Roll the walk-forward window
      back the same way and re-measure from week 1.
- [ ] With `--train-calibration-seasons 1` and the default four weeks, the whole calibration
      season is the newest season the window does not touch, so it jumps (for example from 2024
      to 2025) between weeks 4 and 5 of a season and swaps a whole season between train and
      calibration. Correct by construction, but not logged; log the candidates or document it.
- [ ] With `--include-postseason`, the rolled-back window is filled by the previous season's
      playoff weeks (divisional, conference, Super Bowl: seven games), so a week-1 or week-2
      window can hold about 23 games. The weekly default excludes the postseason; consider
      skipping postseason weeks when the window rolls back, or counting games instead of weeks.
- [ ] `_split_train_calibration_holdout` still returns the lossy `(newest season, its weeks)`
      pair and `_inseason_calibration_pairs` re-derives the window from the calibration frame;
      returning `window_pairs` from the split would remove the duplication (touches the two
      destructuring sites in `ml_model_training.py` and two 8-tuple mocks in
      `tests/test_ml_model_training_additional.py`).
- [ ] `_attach_qb_features` requests a not-yet-published current season a second time before
      kickoff (the first `load_pbp` skipped it, so it is "missing"), doubling that network attempt
      and its warning; and every ETL run, however narrow its `--min-season`, needs all
      `1999..max_season` play-by-play seasons cached or downloadable.
- [ ] `qb_stats`: scrambles (dropbacks without a passer id) go to the team-game's plurality
      passer, so a starter who leaves early has his scrambles booked to the backup; the recent
      window counts any week with one dropback as a full game; `completions` and the recent
      `cpoe` sums are computed but never read. Reuse: the play-by-play helpers and
      `calculate_stat_differentials` were taken along with task 53.6 (`0.9.0`; the lenses are gone
      since `0.10.0`, the reuse stays); `_ratio` is still
      a third copy of the null-safe ratio in `teamrankings._safe_ratio` and `strength_snapshot`
      (consolidating it touches three modules, so it was left out of that task).
- [ ] The QB identity chain (Elo name to GSIS id via a copy of the nfeloqb metadata, aliases and
      an `F.Last` fallback) could key completed games on the nflverse schedule's
      `away_qb_id` / `home_qb_id` directly if those columns joined `NFLREADPY_SCHEDULE_COLUMNS`;
      name resolution would then be needed only for future-week rows.
- [ ] `_build_xgb_fit_kwargs` forwards `LogEvalCallback` only when `fit()` accepts `callbacks`,
      which XGBoost 3.4.1 does not, so eval-progress logging is silently dropped (pre-existing).

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
