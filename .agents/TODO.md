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
  benchmark-instrument follow-up from the 2026-09-18 audit), so new work starts at 61
  (60 is the CLI consolidation milestone).

---

## Execution loop (required)

For each task:

1. **Understand scope**: read the relevant modules/tests/docs.
2. **Plan**: outline the smallest set of changes needed.
3. **Test-Driven Development**: add/adjust tests for all planned changes. Aim to increase coverage.
   For executable code, confirm the exact behavior/lines you plan to touch are covered first; if
   not, add focused characterization or failing tests before editing production code.
4. **Implement**: make changes incrementally (small diffs, one logical change at a time).
5. **Run the gate**: `scripts/gate.sh` (add `--web` when `web/` changed). It runs, in CI's
   order, `uv lock --check`, `uv sync --check --active`, `ruff format --check`,
   `ruff check`, `ty check`, `pyright`, `pytest`, markdownlint and the CLI help smoke checks,
   and reports every step before exiting non-zero. Nothing is "done" until it exits `0` on the
   final tree (`AGENTS.md`, "Delegation guardrails"). The individual `.venv/bin/...` commands
   are for iteration only.
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

### Current validated baseline (2026-09-21, version `0.12.15`, `docs/handoff-next-steps`)

- Everything through `0.12.14` is merged into `main` and pushed; `main` is at `3729006`, the
  merge of `docs/rules-and-roadmap`. The current branch `docs/handoff-next-steps` sits off that
  commit and carries the documentation-only `0.12.15` chunk. This chunk changes no executable
  code, so its gate result is the one reported in the session check-in; the last full
  `scripts/gate.sh` exited `0` on the `0.12.13` tree (`862 passed`, coverage `92.68%`), and the
  last `--web` run was `862 passed`, coverage `92.66%`, on `ebb1f8e`. That gate covered ruff
  format, ruff, ty, pyright, pytest with coverage against the enforced `90%` floor,
  markdownlint, `uv lock --check`, `uv sync --check --active` and the CLI help smoke checks,
  plus the frontend gate (lint, typecheck, `22` vitest tests, build). The `web` extra no longer
  exists, so the gate no longer passes `--extra web`.
- Data: the 2026-09-20 rebuild (`db6a78a3...`, `7278` rows, `513` columns; backup of the
  previous build in `data/backup_pre_m59_rebuild/`); walk-forward input for new arms
  `data/completed_games_ml.m59_through_2025.csv` (`cf42ec55...`), reference arm
  `models/wf_m59_rebuild_2023_2025_from_week1/` (see `AGENTS.md`).
- The walk-forward benchmark and its provenance are in `AGENTS.md`; the Milestone 59 record,
  measurements and audit are in `ARCHIVE.md`.
- Everything below is the earlier `0.8.0` baseline, kept for the data-state notes that still
  hold.

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
- [ ] Confirm `scripts/gate.sh` exits `0` on the final tree and that every number written into
      the docs names its run directory (`AGENTS.md`, "Delegation guardrails").

---

## Roadmap Status

Done so far in the feature-engineering workstream (see `ARCHIVE.md`): Milestone 45 (play-by-play
EPA families), 46 (schedule-adjusted strength), 49 (continuous early-season shrinkage), the first
phase of 43 (current-season Bradley-Terry defaults), 51 (power rankings on the adjusted
composite), 52 (the total head: fixed, still behind the market line, totals labelled
diagnostic-only), task 56.4 (calibration window across the season boundary), Milestone 59
(benchmark instrument, fit parity, historical divisions, and the noise-family follow-up; closed
2026-09-19 with two narrowed parts reopened under "From Milestone 59" below and task 55.7), and
the web UI's phases 0-3 (Milestone 58, merged as `0.8.0`). Execution order:

Order set by the user on 2026-09-21, after the tree-budget ladder was reported:

1. Task 55.7 close-out: adopt `200` as the shared default `n_estimators`, align
   `config/weekly_run.yaml` with the production defaults, then run the `100` plateau rung.
2. Task 56.2 config flags: `include_postseason: false` and `wf_include_postseason: false` in
   `config/weekly_run.yaml`. It can share the `0.13.0` chunk with item 1, since both are
   config defaults.
3. Task 54.0 - the parked schedule skeleton plus the box-score repair, the ETL rebuild, and one
   three-season no-breakage arm at the new `200` default.
4. Tasks 54.1 and 54.2 - situational rates and the PBP box-score source.
5. Task 55.8 - season weighting, as six-season arms at the `200` default.
6. Task 55.9 - the Optuna re-tune, after Milestone 54 lands.
7. Milestone 60 - CLI consolidation (read-only audit first, then removals after the user signs
   off); it can run in parallel with any walk-forward as a subagent task.

After those: Milestone 53 task 53.7 (defense-adjusted quarterback rate), tasks 54.3 and 54.4,
the rest of Milestone 55 and Milestone 56, Milestone 58 phases 4-6 whenever the user asks (its
own branch), and Milestone 57 stays parked until the user reopens it.

The open follow-ups below are not milestones. Pick them up when their area is next touched, or
promote one to a milestone when it grows.

Rules for every feature milestone:

- Land each family as one ablatable feature group with a walk-forward switch (mirror
  `--disable-trend-features`).
- Report a walk-forward comparison with the group on and off before marking done, read on the
  deterministic and market columns (task 59.1), over enough seasons to resolve the effect claimed.
  The fit-noise floor (`AGENTS.md`, "Fit-noise floor on the same build") is the yardstick: on
  three seasons a Brier difference under about `0.002`, a pick-accuracy difference under about
  `0.01` or a margin MAE difference under about `0.06` is re-seeding noise. An arm that claims an
  improvement runs on six seasons (`--eval-last-n-seasons 6`, about 100 minutes idle) and, if
  the claim is still near the floor, on a second seed; a three-season arm can only show a tie.
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
Outputs under `models/weekly_2026_week_02/`. Refreshed 2026-09-20 10:13-10:45 MDT for the
Sunday and Monday games (`models/weekly_2026_week_02_refresh/`): market lines refreshed first
(`lines_refresh --season 2026 --week 2`, 9 of 15 games' odds had moved), then
`weekly_run.py --skip-data-refresh` on the 04:31 rebuild; same selected config, 15 games.

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
      a walk-forward tie on the 59.1 instrument. Read the tie as a no-breakage check on three
      seasons against the reference arm on the rebuilt build: the fixed rows are 2001-2002
      training rows, so every margin will move (the fit-noise floor pattern) and no lift is
      expected or claimable.
      Decided 2026-09-21: proceed with the parked branch `feat/m54-0-schedule-skeleton`
      **including** the box-score repair (commits `ae687dd` skeleton and `d34b0ab` repair).
      Merge it into the working branch, write the changelog entry, run the gate; then the
      rebuild, which the user approved for this task: back up `data/*.csv` to
      `data/backup_pre_m54_0/` first, rerun the ETL from the cache, rerun the leakage audit, and
      cut a through-2025 copy `data/completed_games_ml.m54_0_through_2025.csv`. Then one
      three-season from-week-1 arm at the new `200` default, compared against the 2023-2025
      folds of `models/wf_m55_7_2020_2025_trees200/`, which are the `200` reference on the
      previous build (the existing three-season reference arm is at `598`, so it is not the fair
      comparison). Read it as a no-breakage tie check. Expect the ETL to log 9 coverage-gap
      warnings (1999 BAL/LAR, 2000 BUF/KC/LAC/MIA, 2001-2002 JAX, one 1999 null-team row) and 16
      repair warnings.
      Resolved 2026-09-21 (from the 54.0 audit): `blend_with_prior_stats` weights box-score
      columns by the count of scheduled games while their means rest only on the games actually
      covered (JAX 2001-2002: `8` of `16`). Do nothing about it; task 54.2 erases the mismatch
      by deriving those columns from play-by-play for every game.
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
- [ ] 55.7 Choose `n_estimators` time-aware (from task 59.3, not done there). In-season fits run
      the full `598`-tree budget since `0.12.3` with no early stopping anywhere; `598` is the
      old Optuna value, not a measured choice. Tune it on a whole prior season as the eval set
      (at least 250 games), or early-stop on that season-sized set, and apply the result
      identically in walk-forward and production through the shared fit helpers. Read the
      result on the deterministic and market columns over six seasons. Until then, the
      `early_stopping_rounds` config field and the `--wf-early-stopping-rounds` /
      `--train-early-stopping-rounds` flags are recorded but inert for in-season fits. Method:
      a ladder of budgets (for example `200`, `400`, `598`, `800`, `1200`) as separate six-season
      arms of the reference configuration on the rebuilt build, one hypothesis per arm, read on
      the deterministic columns against the fit-noise floor; the winner becomes the shared
      default in walk-forward and production together (a default change: must-ask).
      Progress 2026-09-21: three six-season rungs ran and were independently rescored,
      `models/wf_m55_7_2020_2025_trees598/` (the reference, which reproduces the three-season
      reference arm's 2023-2025 folds bit for bit),
      `models/wf_m55_7_2020_2025_trees200/` and `models/wf_m55_7_2020_2025_trees400/`. The
      aggregate order is monotone toward fewer trees and small: weeks 3-18 deterministic Brier
      `0.2103` / `0.2111` / `0.2117` and margin MAE `9.9281` / `9.9978` / `10.0240` for
      `200` / `400` / `598`, with the `200 - 598` Brier interval covering zero by `+0.0000791`
      and its margin MAE difference `-0.0958` `[-0.1583, -0.0345]` beyond the fit-noise floor.
      The ladder stopped on its own "report all three and ask" branch; the table and the
      pairwise intervals are in `AGENTS.md` under "Tree-budget ladder".
      Decided 2026-09-21 by the user: adopt `200` as the shared default `n_estimators`, in
      walk-forward and production together (the default change is approved). In the same chunk,
      make `config/weekly_run.yaml` consistent: it sets `tune: true` today, which re-runs a
      one-hour Optuna study on every weekly run and was not intended, so set `tune: false`; and
      its walk-forward stage runs `wf_n_estimators: 200`, `wf_max_depth: 4`,
      `wf_learning_rate: 0.03` while the final fit uses `DEFAULT_XGB_PARAMS` (depth `5`,
      learning rate `0.0165`, `598` trees), so align the walk-forward stage's XGBoost params
      with the production defaults and let both stages fit the same model. Then run one more
      rung, `100`, as the plateau check: six seasons, the same ladder configuration, a
      `HYPOTHESIS.md` naming its windows and columns per guardrail rule 4, and an independent
      reviewer rescore. If `100` ties `200`, keep `200`; if `100` is better beyond the
      fit-noise floor, report it and ask. The default change is a minor bump (`0.13.0`). The
      task stays open until `0.13.0` lands and the `100` rung is reviewed.
- [ ] 55.8 Season weighting. Today every training row from 1999 carries the same weight as
      last week's game (`recency_half_life_seasons` is off by default in walk-forward and
      production). The README's recency ablation ("keep it off") is not trustworthy: it was
      measured through Platt calibration, which the 2026-09-18 audit found noise-dominated, on
      a superseded build, with a log loss of `1.95` that reads as a calibration failure rather
      than a model difference, and with an aggressive half-life of `2` seasons. Re-measure on
      the 59.1 instrument over six seasons on the rebuilt build: half-lives of about `4`, `8`
      and `16` seasons via `--recency-half-life-seasons`, plus the unweighted reference,
      hypothesis and decision rule per arm, read against the fit-noise floor. If a weighting
      wins, changing the default is must-ask; either way, replace the README ablation with the
      new measurement and its run directories.
- [ ] 55.9 Optuna re-tune. Added 2026-09-21 by the user's decision, to run **after Milestone 54
      lands**: tuning before the feature set changes would have to be redone. The user would
      like it on a bye week or in the off-season, since it occupies the machine for hours.
      Prerequisites, each its own tested chunk before any trial runs:
      (1) trials must fit exactly the way production fits. `0.12.3` removed in-season early
      stopping from production and walk-forward, but `_score_margin_total_fold` in
      `nfl_predictor/ml/ml_model_core.py` still passes `early_stopping_rounds` to
      `_fit_margin_total_models`, so every trial is scored on a differently fitted model.
      (2) the tuning objective must score what the walk-forward instrument scores: the
      deterministic `Phi(margin / SCORE_DIFF_STD_DEV)` Brier, with the market view alongside,
      not the configured-calibrator Brier.
      (3) plumbing so a tuned parameter set actually reaches the walk-forward. Today
      `weekly_run` takes explicit `wf_*` params and never writes a best-params file;
      `--defaults-path` and `best_config.json` exist only as Milestone 55 acceptance text.
      Then the study itself: the search space as it stands (`n_estimators` `200`-`1200`,
      learning rate `0.01`-`0.3` log, depth `3`-`8`, and the rest), TPE with seed `42`, SQLite
      storage, `300`-`500` trials, about 3-8 hours. The winner is confirmed on a six-season
      walk-forward against the `200` rung and on a second seed before it becomes a default
      (must-ask). The old studies under `models/weekly_2025_week_21/` and
      `models/weekly_2025_week_22/` are void: their trials early-stopped and they predate the
      `0.6.2` guidance.

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

- [x] 56.1 Add data-refresh pass-through (`--data-min-season` / `--data-max-season` or a generic
      `--data-collection-args`) to `scripts/weekly_run.py`; the same pass-through carries the
      `--stat-prior-blend*` flags, which it cannot set today. Done 2026-09-20 (`0.12.11`) as the
      generic `--data-collection-args` string.
- [ ] 56.2 Decide and document how postseason games enter evaluation and training; when the
      prediction week is postseason, default the power-rankings through-week to the last
      regular-season week. Narrowed 2026-09-20 (`0.12.11`): the through-week clamp landed and the
      README documents the current state (code defaults exclude postseason; the shipped
      `config/weekly_run.yaml` includes it at weight `1.3`, so the weekly command and the
      documented defaults disagree). The decision itself is the user's: keep the config's
      inclusion, or align the defaults; a default change is must-ask.
      Direction 2026-09-21: postseason matchups stay in the data and are kept separate from
      regular-season matchups for training and for prediction. A model used for regular-season
      weeks should not train on playoff games, and the prior-season blend should draw on the
      previous regular season only. The design discussion is deferred until the rest of the
      pipeline runs smoothly, and is wanted in time for the 2026 playoffs.
      Decided 2026-09-21: set `include_postseason: false` and `wf_include_postseason: false` in
      `config/weekly_run.yaml` now (approved), so the weekly run matches the rule that a model
      used for regular-season weeks never trains on playoff games. Leave `postseason_weight` in
      place but inert. Verified facts behind the decision: season-to-date stats, adjusted
      strength, records and the prior-season blend are regular-season only; Elo and QB Elo
      (`data/qb_elos.csv`) and the TeamRankings playoff-week snapshots are the only features
      that carry playoff results, and they enter as pre-game ratings, so there is no leakage.
      The playoff-week design itself stays deferred.
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
- [x] 58.4 Housekeeping: the `web` extra in `pyproject.toml` now duplicates the core dependency
      list; drop it (and the `--extra web` in CI and `web/README.md`) or give it a purpose. The
      `etl_full`, `validate_offline` and `validate_live` job templates read the checkout's own
      `data/` and ignore `NFLP_DATA_DIR` (see the plan's Phase 2 deviations).
      Narrowed: the `web` extra is dropped, and the three CLIs now take `--data-dir` with the
      templates passing `NFLP_DATA_DIR`, so every dataset those jobs read or write follows the
      configured tree. The ETL's upstream inputs still resolve from `constants.DATA_PATH`: the
      copied `qb_elos.csv` and quarterback identity file (`nfl_predictor/utils/polars/loaders.py`,
      `data_collection._attach_qb_features`) and the TeamRankings and nflreadpy caches
      (`nfl_predictor/utils/scraping_utils.py`, `nfl_predictor/utils/polars/teamrankings.py`).
      Threading a data directory through those read paths is a separate change; until it lands,
      pointing the API at another checkout's data makes `etl_full` read inputs from this checkout
      and write outputs to the configured one. The narrowed part landed in `0.12.12`
      (`604bcc6`, `e3b828d`); whether the remainder is wanted at all is on the user's question
      list.
      Closed 2026-09-21 by the user's decision: the `NFLP_DATA_DIR` setting stays, because the
      web API resolves its own paths through it and it lets the app run against a copied data
      tree, but the ETL's upstream inputs will not follow it. The `0.12.12` narrowing is the
      final shape of this task; it stays listed here, marked done, while Milestone 58 is open.

Acceptance:

- [ ] Each phase ships with `tests/api/` and vitest coverage, the Python and frontend gates green,
      a `CHANGELOG.md` entry under a new version, and the plan's Status section updated.

---

## Milestone 60 - CLI consolidation

Added 2026-09-21 by the user's direction: the entrypoints' flags have grown bloated, and some of
them no longer do anything.

- [ ] 60.1 Audit every `add_argument` across `nfl_predictor/ml/ml_model_cli.py`,
      `scripts/weekly_run.py`, `scripts/walk_forward_backtest.py`, `scripts/betting_pipeline.py`,
      `scripts/golden_command.py`, `nfl_predictor/data_collection.py`, `scripts/validate_*.py`
      and `scripts/power_rankings.py`. List, per flag: the inert ones (for example the
      early-stopping flags, recorded in the config and the fingerprint but not applied to any
      fit since `0.12.3`), the ones duplicated across entrypoints, and the ones only ever set by
      the config file rather than on a command line. Then propose the removals and a shared
      options module so the surviving flags are declared once. The `--data-dir` and
      `--data-collection-args` options added in `0.12.11` and `0.12.12` belong in the same
      audit.

Acceptance:

- [ ] A written inventory lands in `.agents/`, the user signs off on the removal list, and only
      then do the removals land, each with a deprecation note in `CHANGELOG.md`.

---

## Open follow-ups from completed milestones

Each group names the archived milestone it came from; the milestone's full record is in
`ARCHIVE.md`. Resolved items have moved there.

### From Milestone 59 (benchmark instrument; audited 2026-09-19)

- [ ] Narrowed 59.2: the calibration frame for fitted calibrators is the previous two seasons
      plus the completed weeks of the eval season, but the rows are in-sample (walk-forward: a
      subset of `fold.train_df` predicted by the model trained on it; production:
      `_pooled_calibration_frame` over `train_df + calibration_df`, the same way). The task asked
      for pooled out-of-fold predictions. Only `platt`, `isotonic` and `sigma` are affected;
      `auto` is the deterministic floor and production defaults to `elo`, so nothing shipped on
      it. If a fitted calibrator is ever to beat the floor it needs the out-of-fold pool: in
      walk-forward, the earlier folds' `predictions` frames of the same run (empty for the first
      eval seasons, so the floor stays the fallback); in production, the walk-forward compare
      that `weekly_run` already runs. Measure on the 59.1 instrument before making it default.
- [ ] Narrowed 59.3: `n_estimators` is untuned; task 55.7 carries it.
- [x] ETL rebuild for 59.4 and 59.6: done 2026-09-20 (`0.12.6`), a tie on the instrument;
      record under `ARCHIVE.md`, Milestone 59, "Rebuild". The first pass exposed a `0.12.4`
      defect (the sack exclusion starved `opponent_points_per_play`), fixed before the second
      pass.
- [ ] `WalkForwardConfig.early_stopping_rounds` stays in the config and the fingerprint though
      no in-season fit reads it; removing it changes every fingerprint. Remove it, or wire it to
      the season-sized eval set of task 55.7, when that lands.
- [ ] `scripts/wf_compare.py` prints `deterministic_pick_accuracy` in place of the configured
      `pick_accuracy` in its summary columns; the CSV still has both.

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

- [x] Closed 2026-09-20: superseded by `0.12.3`, which removed in-season early stopping from
      production and walk-forward. No head stops on the calibration frame any more, so the
      defect this item described cannot recur. The historical evidence, kept for the record:
      the Week-2 smoke run (`models/smoke_20260911`, market-anchored) stopped its margin head at
      iteration `1` on a 50-game window; the workaround run calibrated on the 2025 and 2026
      seasons (274 games) stopped at `189` (`models/smoke_20260911_workaround/`), and
      `models/weekly_2025_week_22` (8 weeks) at `9`. Every head now runs the full tree budget,
      and the only question left is how large that budget should be, which is task 55.7.
