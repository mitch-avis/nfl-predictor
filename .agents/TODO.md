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
  of `ARCHIVE.md`). Milestones 54 and 59 are also fully closed and archived; their remaining
  follow-ups live under "Open follow-ups from completed milestones" below. Active milestones with
  open tasks are 53, 55, 56, 58 (the web UI, added 2026-09-11) and 60 (the CLI consolidation
  milestone, added 2026-09-21), so new work starts at 61.

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

### Current validated baseline (2026-09-24, version `0.18.0`, `main` at `703ea25`)

- `feat/m55-8-season-weighting` is merged into `main` and pushed (fast-forward to `703ea25`);
      the remote branch is gone. `scripts/gate.sh` exits `0` on `main` as re-run on the fresh
      `feat/m60-cli-consolidation` branch (`889 passed`, coverage `92.45%`). Coverage measures
      `nfl_predictor/` only; the `scripts/` tree is outside it (Milestone 60, task 60.2).
- The `../nfl-predictor-web`
      worktree directory no longer exists (`git worktree list` marks it prunable), nothing listens
      on port `8765`, and `feat/web-ui` is fully merged into `main`.
- `feat/m54-0-landing` (the `0.14.0`-`0.16.1` chunks) merged into `main` with no conflicts and was
      pushed on 2026-09-22 (merge commit `295d4c4`, version `0.16.2`). `main` and `origin/main` carry
      the `pbp`-default sources, the shared `200`-tree default and, since `0.18.0`, unweighted
      production training.
- Data: the 2026-09-21 full rebuild (`--refresh-nflreadpy`, needed for the new
      `pass_attempt`/`rush_attempt` raw columns) on the `0.16.0` code, now defaulting to
      `--team-stats-source pbp --tr-stats-source pbp`, produced `data/completed_games_ml.csv`
      (`7292` completed rows, `513` columns, `edd6b852...`) with the prior top-level CSVs backed up
      in `data/backup_pre_m54_flip/` and the pre-refresh play-by-play cache in
      `data/cache/nflreadpy/backup_pre_m54_flip/`. The walk-forward input is
      `data/completed_games_ml.m54_flip_through_2025.csv` (`7261` rows, `2d4111a6...`). Leakage
      audit `models/audit_m54_flip_rebuild/leakage_audit.json`: `463` features, `0` flags, same
      shape as every prior 54.x build.
- The user reviewed the four columns that disagreed with nflverse (from the 54.1-54.4 comparison)
      and asked for each to be examined and corrected, then for both source flags to be flipped to
      `pbp` as the default once verified: `passing_epa` now sums `qb_epa` instead of `epa`
      (`69.54%` to `99.33%` match with nflverse); `pass_attempts`/`pass_completions`/`pass_yards`/
      `pass_touchdowns`/`interceptions_thrown`/`rush_attempts`/`rush_yards`/`rush_touchdowns` now
      use nflverse's own `pass_attempt`/`rush_attempt` flags (worst case `pass_attempts` `86.70%`
      to `99.87%`); `rushing_epa` now includes two-point tries (`95.54%` to `99.87%`);
      `fumbles`/`fumbles_lost` now exclude special-teams plays (`73.63%`/`90.35%` to
      `91.95%`/`98.37%`). `2pt_conversions` (`94.80%`) needed no code change: the user manually
      verified a mismatch against the actual game and asked for the rest to be checked, which
      confirmed a systematic nflverse team-stats bug (roughly doubles the true count on the
      games it gets wrong, `95.1%` of `246` mismatches across seven sampled seasons) rather than
      a play-by-play defect. Full numbers: `models/pbp_vs_nflverse_m54_2/COMPARISON.md`.
- Both source flags are now the default (`0.16.0`), including the production fast path
      `scripts/weekly_run.py` uses when it calls `data_collection.main()` with no arguments;
      `nflverse`/`scrape` remain selectable explicitly.
- Task 55.8 closed as `0.18.0` on 2026-09-24 by the user's decision after an independent review
      of nine six-season arms: production trains **unweighted**, so `config/weekly_run.yaml` no
      longer sets `train_recency_half_life_seasons` or `wf_recency_half_life_seasons` (record in
      `ARCHIVE.md`, Milestone 55, "55.8").
- Reference arms on the current default (pre-flip) sources: `models/wf_m54_0_2023_2025_from_week1/`
      (checkpoints `models/wf_checkpoints/d112ebcba3115bafe9d9/`) tied the accepted `200`-tree
      reference slice on weeks 3-18: deterministic Brier `0.2097` vs `0.2090`, diff `+0.0007`
      `[-0.0011, +0.0024]`; margin MAE `9.9166` vs `9.9044`, diff `+0.0122` `[-0.0566, +0.0801]`.
- The default-flip verification arm `models/wf_m54_flip_2023_2025_from_week1/` (checkpoints
      `models/wf_checkpoints/9779c1cbb0701d23661a/`, reviewed in its `REVIEW.md`) tied that
      reference on weeks 3-18: deterministic Brier `0.2106` vs `0.2097`, diff `+0.0009`
      `[-0.0008, +0.0026]`; margin MAE `9.9321` vs `9.9166`, diff `+0.0156` `[-0.0529, +0.0812]`.
- Tasks 54.0-54.4 and the default flip are archived; Milestone 54 is fully closed (`ARCHIVE.md`).
      Task 55.8 closed as `0.18.0` and is on `main`. Merging and pushing remain must-ask, every
      time.

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

Order agreed with the user on 2026-09-24, replacing the 2026-09-21 and 2026-09-23 orders. There
is no deadline: every step runs to satisfactory completion, including its reviews, before the
next starts. Expect roughly two to three weeks, then the web UI (Milestone 58, phases 4-6).

Why this order (the ordering principles, agreed with the user):

1. **Restructure before changing behavior.** Milestone 60 moves code without changing any output,
   pinned by a characterization test of the weekly run; every output-changing task comes after it,
   so the moves stay verifiable and no code is edited just before it moves.
2. **The benchmark must measure production before anything else is measured** (`AGENTS.md`
   rule 11). Parity gaps (device, probability path, line timing) close before new feature or
   tuning measurements.
3. **Features before tuning.** Anything that changes feature values lands before the Optuna
   re-tune, or the tune is fitted to a feature set that no longer exists.
4. **Batch what invalidates checkpoints.** Edits under `nfl_predictor/ml/` and device changes
   invalidate every walk-forward checkpoint, so they are grouped and pay for one new reference.
5. **The weekly run comes first.** Production-changing steps land between game weeks; heavy
   compute runs overnight; never two walk-forwards at once.
6. **Two seeds for any default change** (`AGENTS.md` rule 13).
7. **Each step on its own branch off `main`**, merged (must-ask) when its gate, review and
   check-in are done, so branches stay short (the user's merge-cadence preference).

Steps:

1. **Task 55.8 close-out** (branch `feat/m55-8-season-weighting`). Done 2026-09-24 as `0.18.0`
   apart from the merge (must-ask): nine six-season arms, an independent review of all nine
   (rule 3), and the user's decision for unweighted production training. This
   step's docs chunk also carries this order, the new rules, and the retirement of 55.1/55.2.
2. **Milestone 60 + task 55.5** (new branch, new session). CLI and entrypoint consolidation,
   widened to every file under `scripts/`, behavior-preserving. 55.5 (the `ScoreModel` fate) is
   decided and executed here, because retiring unused code is part of the same cleanup. Absorbs
   the follow-ups marked "(step 2)" below.
3. **Tasks 55.4 + 56.5 + 56.6 + the out-of-fold calibration pool** (new branch). Close the
   production/benchmark parity gaps together, because each changes what production outputs and
   they can share one new reference: the GPU as the default device for every run (55.4, decided
   by the user 2026-09-23, with a GPU determinism check and a two-seed GPU reference); how
   production probabilities are formed (56.5, decided by the user from evidence, mostly by
   rescoring saved predictions); the line-timing yardstick (56.6); and the out-of-fold calibration
   pool (follow-up from 59.2), so fitted calibrators can be judged fairly as 56.5 options.
   Task 55.6 (the per-season stability view) lands here as part of the standard walk-forward
   report. Absorbs the follow-ups marked "(step 3)".
4. **Reproducibility first, then tasks 55.3 + 53.7 + the feature follow-ups** (new branch).
   First the follow-ups that make rebuilds trustworthy (bit-reproducible schedule-strength columns,
   a schema version in the play-by-play cache key), because this step compares dataset builds.
   Then every change to feature values: `K` for the early-season prior blend (55.3), the
   defense-adjusted quarterback rate (53.7), and the follow-ups marked "(step 4)". Each ETL rebuild
   is must-ask; each adoption decision uses two seeds. This is the largest step.
5. **Tasks 55.9 + 56.3** (new branch). The Optuna re-tune on the final features and device, its
   three prerequisites first, then the study, a six-season confirmation on two seeds, and 56.3
   (wiring the chosen settings into the weekly run and the benchmark, one shared source).
6. **Web UI, Milestone 58 phases 4-6**, on a base that Milestone 60 has already settled.

Retired by the user on 2026-09-24: tasks 55.1 and 55.2 (a general configuration-sweep runner). A
runner that tries many settings and keeps the best is the selection bias that broke the weekly
stage 1 (task 56.5), and it conflicts with rule 4 (a hypothesis and decision rule per run).
Hypothesis-driven ladders and the task 55.9 tune cover the purpose. Record in `ARCHIVE.md`,
Milestone 55.

How success is measured from step 3 on (agreed with the user 2026-09-24,
`.agents/m60/PROPOSAL.md` section 8): select on the Brier score of the probabilities actually
submitted, read against the market's Brier on the same games; require no loss on log loss; break
ties on confidence-pool points, then margin MAE; report pick accuracy but never select on it.
Step 3 settles one probability path used identically by every run type and retires the
calibrators it does not use. Step 5 tests the training objective (squared error against
pseudo-Huber) as one choice inside the tune. Betting stays a separate decision layer (edge against
the market, conservative sizing), diagnostic only; no odds-weighted training loss.

Task 55.8 was closed on 2026-09-23 as `0.17.0` and **reopened the same day** by the user after a
second-key review: the numbers were right, but the closing agent had reviewed its own runs
(rule 3), and the close-out left out that the shipped half-life `4` is the weakest arm on every
probability and error metric. The redo closed it on 2026-09-24 as `0.18.0`: unweighted
production training, by the user's decision after an independent review (`ARCHIVE.md`,
Milestone 55, "55.8").

Tasks 54.1-54.4 landed 2026-09-21 (`0.15.0`-`0.15.1`), and the user then approved fixing the
four open exceptions and flipping both source flags to the default once verified, which landed
the same day (`0.16.0`-`0.16.1`); Milestone 54 has no remaining item and is fully archived.
Milestone 57 stays parked until the user reopens it.

The open follow-ups below are not milestones. Each one was assigned on 2026-09-24 to the roadmap
step that absorbs it; none is left for "when the area is next touched":

| step | follow-ups (the group they are listed under) |
| --- | --- |
| 2 (Milestone 60) | `WalkForwardConfig.early_stopping_rounds` in the config and fingerprint; `wf_compare.py` printing the wrong pick-accuracy column (both Milestone 59); the automatic OpenMP wait policy in the walk-forward entry points; pruning `models/wf_checkpoints/` (both Milestone 49); `_build_xgb_fit_kwargs` and `LogEvalCallback` (2026-09-11 review) |
| 3 (parity) | Narrowed 59.2, the out-of-fold calibration pool (Milestone 59); the four calibration-window items: `select_calibration_data`, `--train-calibration-seasons 1`, `--include-postseason` roll-back, `_split_train_calibration_holdout` (2026-09-11 review) |
| 4, first | Schedule-strength columns not bit-reproducible across identical rebuilds (Milestone 46); no schema version in the play-by-play cache key (Milestone 45) |
| 4 (feature values) | Unused `PBP_COUNT_COLUMNS` (Milestone 45); `strength_games_played_diff` zero importance, `adj_*` scale drift, `sos_played_raw` null in weeks 1-2 (Milestone 46); the blend's weeks 3-18 margin MAE cost (Milestone 49, with 55.3); `_attach_qb_features` double request, scramble attribution, the QB identity chain (2026-09-11 review, with 53.7) |

Rules for every feature milestone:

- Land each family as one ablatable feature group with a walk-forward switch (mirror
  `--disable-trend-features`).
- Report a walk-forward comparison with the group on and off before marking done, read on the
  deterministic and market columns (task 59.1), over enough seasons to resolve the effect claimed.
  The fit-noise floor (`AGENTS.md`, "Fit-noise floor on the same build") is the yardstick: on
  three seasons a Brier difference under about `0.002`, a pick-accuracy difference under about
  `0.01` or a margin MAE difference under about `0.06` is re-seeding noise. An arm that claims an
  improvement runs on six seasons (`--eval-last-n-seasons 6`, about 100 minutes idle) and on
  two seeds (`AGENTS.md` rule 13); a three-season arm can only show a tie. The 2026-09-23
  seed-7 pair showed that on six seasons, re-seeding alone can move pick accuracy and pool points
  by amounts whose game-resampled intervals exclude zero, so a single-seed six-season "win" is
  not a result.
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
- [ ] 53.7 (step 4, with 55.3; two seeds per adoption decision) Opponent-adjusted quarterback
      EPA, in two steps. First (cheap) a defense-adjusted
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

## Milestone 55 - Off-season configuration sweep + lock default settings

Formerly Milestone 39, with former Milestone 40 folded in. Deferred until the feature milestones
land, because new feature families would invalidate sweep results.

Goal: run an objective, repeatable sweep of modeling configurations under the canonical evaluation
protocol, then write the selected configuration as the default for weekly runs.

Tasks:

Tasks 55.1 and 55.2 (a general configuration-sweep runner) were retired by the user on
2026-09-24; the reason is under "Roadmap Status" and the record in `ARCHIVE.md`.

- [ ] 55.3 (step 4) Choose `PRIOR_BLEND_GAMES` (`K`) instead of inheriting it from the strength
      blend (see the Milestone 49 follow-ups). With the sweep runner retired, this is a small
      ladder with its hypothesis and rule written first: candidate values (for example `2`, `4`,
      `8`), one ETL build each (must-ask; back up `data/*.csv` first), six seasons from week 1 on
      two seeds, weeks 1, 2 and 3-18 reported separately (the value changes early-season features
      most). Runs after the rebuild-reproducibility follow-ups and shares its rebuild cycle with
      53.7. Context from task 55.8: each two-seed weighting contrast (half-lives `4`, `16`, `32`)
      improved week-2 total MAE by `0.2`-`0.3` points beyond its interval (96 games), the only
      consistent early-season signal that ladder found; report week-2 total MAE for each `K`.
- [ ] 55.4 (step 3) Add `xgb_device=auto` (prefer `cuda`, else CPU) used identically in evaluation
      and final training. **Decided by the user on 2026-09-23: the GPU (CUDA) becomes the default
      device for every XGBoost run, standalone walk-forwards included.** Today only the weekly run
      uses it (`xgb_device: cuda` in `config/weekly_run.yaml`, both stages), while
      `scripts/walk_forward_backtest.py` defaults to the CPU, so every benchmark arm so far trained
      on the CPU (an `AGENTS.md` rule 11 parity gap). Do not change the default while a CPU ladder
      is running: arms of one comparison must share a device. When it lands: a one-fold CPU-vs-GPU
      timing and prediction-difference check, a CPU fallback when no GPU is present, the device
      recorded in run metadata, and a new GPU reference arm, because later arms can no longer be
      compared with the CPU reference numbers in `AGENTS.md`. Added 2026-09-24: a GPU determinism
      check (the same seed twice on one fold must give identical predictions, or checkpoint resume
      and "only one setting differs" comparisons break), and the GPU reference on two seeds, which
      also measures the GPU's own fit-noise floor (rule 13).
- [ ] 55.6 (step 3) Add the stability view by season and week bucket, and a "recommended
      defaults" section. The second-key review script
      (`models/wf_m55_8_review/review_55_8.py`) already produces per-season tables; promote that
      into the standard walk-forward report while step 3 touches reporting.
- [x] 55.7 Choose `n_estimators` time-aware: closed 2026-09-21 (`0.13.0`-`0.13.1`). A four-rung
      ladder (`200`/`400`/`598`, then a `100` plateau check), each rung a six-season arm with its
      own hypothesis and an independent reviewer rescore, found the aggregate order monotone
      toward fewer trees but small; the user adopted `200` as the shared default `n_estimators`
      in walk-forward and production together, and `config/weekly_run.yaml`'s walk-forward stage
      was aligned to the production XGBoost defaults in the same chunk. Full record, the ladder
      table and the pairwise intervals: `ARCHIVE.md`, Milestone 55, "55.7", and `AGENTS.md` under
      "Tree-budget ladder".
- [x] 55.8 Season weighting: closed 2026-09-24 (`0.18.0`). Nine six-season arms (unweighted and
      half-lives `4`, `8`, `16`, `32`; seeds `42` and `7`, half-life `8` at seed 42 only), an
      independent review of all nine (`models/wf_m55_8_review/INDEPENDENT_REVIEW.md`), and the
      user's decision: production trains **unweighted**, in the weekly walk-forward stage and the
      final fit alike. Half-life `4` (shipped since `83ba2a7` and never measured before) lost on
      two seeds; `16` and `32` tied. Record: `ARCHIVE.md`, Milestone 55, "55.8", and `AGENTS.md`
      under "Season weighting and the six-season fit-noise floor".
- [ ] 55.9 (step 5, with 56.3) Optuna re-tune. Added 2026-09-21 by the user's decision, to run
      **after Milestone 54 lands** (tuning before the feature set changes would have to be
      redone). Scheduled 2026-09-24 as roadmap step 5, after every feature-value change (step 4)
      and on the final device (step 3), so the tune is fitted to the feature set that ships. It
      occupies the machine for hours, so it runs overnight. Prerequisites, each its
      own tested chunk before any trial runs: (1) trials must fit exactly the way production fits.
      `0.12.3` removed in-season early stopping from production and walk-forward, but
      `_score_margin_total_fold` in `nfl_predictor/ml/ml_model_core.py` still passes
      `early_stopping_rounds` to `_fit_margin_total_models`, so every trial is scored on a
      differently fitted model. (2) the tuning objective must score what the walk-forward instrument
      scores: the deterministic `Phi(margin / SCORE_DIFF_STD_DEV)` Brier, with the market view
      alongside, not the configured-calibrator Brier. (3) plumbing so a tuned parameter set actually
      reaches the walk-forward. Today `weekly_run` takes explicit `wf_*` params and never writes a
      best-params file; `--defaults-path` and `best_config.json` exist only as Milestone 55
      acceptance text. Then the study itself: the search space as it stands (`n_estimators`
      `200`-`1200`, learning rate `0.01`-`0.3` log, depth `3`-`8`, and the rest), TPE with seed
      `42`, SQLite storage, `300`-`500` trials, about 3-8 hours. The winner is confirmed on a
      six-season walk-forward against the `200` rung and on a second seed before it becomes a
      default (must-ask). The old studies under `models/weekly_2025_week_21/` and
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
- [x] 56.2 How postseason games enter evaluation, training and the rankings: closed 2026-09-21
      (`0.13.0`). Postseason matchups stay in the data, kept separate from regular-season
      matchups for training and prediction; `include_postseason: false` and
      `wf_include_postseason: false` are now set in `config/weekly_run.yaml`, matching the code
      defaults. `postseason_weight` stays in the config but inert. The playoff-week design itself
      (a playoff-specific model or weighting) stays deferred, wanted in time for the 2026
      playoffs. Full record: `ARCHIVE.md`, Milestone 56, "56.2".
- [ ] 56.3 (step 5, with 55.9) Wire the settings chosen by the task 55.9 tune into the weekly run
      and the benchmark from one shared source (today `weekly_run` takes explicit `wf_*` params and
      never reads a best-params file); confirm resume behavior.
- [ ] 56.5 (step 3) The weekly run's stage-1 winner is chosen by list order, not by evidence.
      Found by the 2026-09-23 review. `scripts/weekly_run.py` (`_pick_best_row`) ranks the stage-1 candidates
      by deterministic Brier, then deterministic log loss, but those two columns depend only on
      the predicted margin, which is identical across the nine calibration and market-blend
      candidates of one market mode. Every candidate therefore ties, and the stable sort returns
      whichever candidate the input list puts first; the list is sorted by the configured
      calibrator's Brier, the column the 2026-09-18 audit found noise-dominated. The week-2
      production run (`models/weekly_2026_week_02_refresh/wf_best.json`) selected
      `elo` + `market_prob_weight 0.2` + `market_prob_clamp 0.1`, so the probabilities submitted
      to the pool come from a different path than the one every benchmark measures, chosen
      without an interval, and eight of the nine candidates are trained every week for nothing.
      Related, from rescoring the six-season task 55.8 checkpoints
      (`models/wf_m55_8_review/probability_paths.py`, the producer's analysis only): a market
      blend chosen from earlier seasons appears to settle at high weights, which would mean the
      model's probabilities add little on top of the closing line. That script has **no second
      key** (`INDEPENDENT_REVIEW.md` there, disagreement 8), so 56.5 rescores it independently
      before any of its numbers enter the docs. Deciding how production probabilities should be
      formed (the deterministic floor, a blend weight fixed by a pre-registered walk-forward rule,
      or something else) is a must-ask default change for the user. Direction from the user
      (2026-09-24): one calibration used the same way by every run type (backtest, benchmark,
      weekly run), `auto` or `platt`, and the `walk_forward_backtest --calibration` default
      (`platt`, against the benchmark's `auto`) is settled here too. `auto` resolves to `none`,
      the fixed normal curve with no fitting. Keep it separate from
      Milestone 60's behavior-preserving moves.
- [ ] 56.6 (step 3, with 56.5) Measure the model against the lines available at pick time, not
      only the stored ones. Picks are submitted before the Thursday game, and lines move between
      then and Sunday, sometimes a lot. The ETL's lines come from nflverse schedules
      (`spread_line`, `total_line`, the moneylines), probably closing lines. Backtests both
      **anchor** the model to them and **score** the market against them, while production
      anchors to whatever lines exist at the mid-week refresh. So backtests probably overstate
      production accuracy, and the market yardstick is harder than the one the user actually
      faces (an `AGENTS.md` rule 11 gap).
      Found 2026-09-24: `data/nfl_lines.csv` (7,231 games, 1999-2025, not in git, read by no code;
      `web_ui_plan.md` calls it unused legacy) carries opening and last spreads, moneylines and
      totals, each with a source and timestamp columns:
      - 1999-2006: opening spread equals last spread in every game (no real openers).
      - 2007-2021 and 2023: real openers, source `legacy`, no timestamps; mean absolute
        open-to-last move `1.0`-`1.6` points.
      - 2022: `59%` of openers equal the last line (partly real).
      - 2024-2025: DraftKings or consensus openers with timestamps (2025 has 240 games so far).
      - Opening moneylines only from 2024.
      Work: (a) confirm the file's provenance (likely nfelo) and whether it can be refreshed through
      an existing source (nflverse or nfelo, no new external service); (b) confirm what the
      nflverse schedule lines are (closing, or a snapshot) against this file's `last` columns;
      (c) on 2007-2025, score the model against the opening line as a second market yardstick
      beside the stored one, since pick-time lines sit between open and close; (d) measure how
      much anchoring on opening instead of stored lines changes backtest accuracy, a
      feature-value change measured with two builds and two seeds; (e) record the coverage limits
      (no openers before 2007, uncertain in 2022, undated `legacy` openers) wherever a number
      depends on them. The outcome feeds 56.5: a market blend judged against closing lines
      overstates what it can do at pick time.
      Update 2026-09-24:
      - Provenance. The user confirmed the source as nfelo's `nfelomarket_data` repo
        (`Data/lines.csv`, updated daily) and refreshed `data/nfl_lines.csv` from it (now through
        2026 week 3, 7,324 rows). The user also saved nfelo's
        `output_data/historic_projected_spreads.csv` as `data/historic_odds.csv` (1,463 games,
        2021 to 2026 week 3): opening and closing spreads plus nfelo's own projected spread and
        home win probability.
      - Checks on the refreshed files:
        (i) The two files agree on openers only from 2023 (`93%`-`100%`); in 2021-2022 they agree
        on `35%` of openers, and `historic_odds.csv` has an opener different from its close in
        only `19%`-`22%` of those games, so 2021-2022 openers are doubtful in both.
        (ii) The closes agree from 2025 (`97%`+) but only `26%`-`44%` in 2021-2024 (different
        snapshots).
        (iii) The nflverse spread in our data matches `nfl_lines.csv`'s last line in `64%`-`79%`
        of games from 2010 on (mean gap about `0.2` points), so it is a late snapshot, not the same
        feed.
      - Opening moneylines: the user proposed inferring them from opening spreads, and that is
        sound for this yardstick, done symmetrically. Convert open **and** close spreads to
        probabilities through the same mapping, so an open-versus-close difference measures line
        movement and not the conversion. Fit the mapping from spreads to the market's own no-vig
        moneyline probabilities where both exist; it uses prices, not results, so it cannot leak
        outcomes. Compare the fitted map with the fixed `spread_to_moneyline` conversion the ETL
        already uses to fill missing moneylines. Anchoring (part d) needs only spreads.
      - Candidate extra yardstick (2021 on): nfelo's pre-game `home_probability_nfelo`, a strong
        public model. Confirm it is pre-game before using it.
      - Keep refreshes manual copies, like `data/qb_elos.csv` from `../nfeloqb`; an automated
        fetch from GitHub would be a new network dependency (must-ask).

- [ ] 56.7 (step 3, with 56.5) One production configuration, every setting decided once. Found
      2026-09-24: no weekly run reads `config/weekly_run.yaml` (it needs `--config`, and no
      launcher passes it), so production runs on the code defaults. The YAML's non-default values
      date from its first version (January 2026, commit `83ba2a7`, reformatted in `67d13e2`) and
      no measurement behind them is recorded. The settings fall into three groups:
      - Change nothing a run produces: `wf_checkpoint_per_fold` (resumability only), the thread
        counts, `tune_timeout` / `tune_cv_splits` (tuning is off), `postseason_weight`
        (postseason training is off), `score_rounding` (display only; it never touches
        probabilities or picks).
      - Change numbers slightly, already decided: `xgb_device` / `xgb_tree_method` (the GPU for
        everything, task 55.4). `train_early_stopping_rounds` is recorded only, because
        in-season fits run the full tree budget.
      - Change which probabilities are submitted, to be measured here: `wf_eval_last_n_seasons`
        (how many seasons stage 1 scores; the count includes the unscorable current season, so
        the default `3` scores two), `wf_calibration_weeks` / `train_calibration_weeks` (the
        newest weeks held out of the tree fit for calibration; the benchmark uses `4`),
        `wf_market_prob_source` (`raw` against `novig` moneylines) and
        `wf_market_prob_blend_method` (`prob` against `logit`), plus `market_transform`
        (`auto`, which is on whenever lines exist, against `true`).
      Work: (a) as a Milestone 60 behavior-preserving move, make the weekly run read
      `config/weekly_run.yaml` by default and rewrite the YAML to today's code defaults (so no
      output changes), quoting `off` as `"off"`: YAML reads a bare `off` as the boolean
      `false`, a latent bug in the current file; (b) here in step 3, decide the output-changing
      group with the measures of success under "Roadmap Status", on six seasons and two seeds,
      against the benchmark configuration, so that production and benchmark share every setting
      (rule 11); (c) consider retiring the weekly stage-1 re-selection altogether. Once 56.5
      fixes the probability path by evidence, re-choosing among near-identical candidates each
      week only adds noise (Week 3's two runs chose `none`, then `elo`, a day apart) and costs
      most of the run's time. Changing a default is must-ask.
      Decided by the user 2026-09-24: (a) goes ahead as specified, in the 60.6 front-door chunk;
      landed in `0.23.0` (the weekly run reads the YAML without `--config`, and the file holds the
      code defaults, so production output is unchanged). The rewritten YAML only records today's
      defaults; the user stressed that its settings are still to be optimized, which is part (b)
      here and task 55.4, not something the rewrite settles. How many seasons stage 1 scores is
      settled here too (with (c)), not by widening the window now.

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

LightGBM status (2026-09-24, user's decision): stays installed and stays parked. A proper
comparison with XGBoost, as a replacement or a blend member, waits until this milestone is
reopened. The CUDA build is set up without a shim (version `0.18.3`): NVIDIA's NCCL 2.31.2 for
CUDA 13.3 replaced Ubuntu's CUDA 12 build (the mismatch behind the `cudaGetDeviceProperties_v2`
load failure), and `nfl-lightgbm-cuda-install` keeps a CUDA build of the locked version,
rebuilding only when needed (README, "Recommended: uv project workflow"). The GPT-5.4 session that first
got it working used a shim and a startup hook; both are gone (transcript in
`.agents/GPT-5-4_LightGBM_CUDA_session_transcript.md`).

Device check, 2026-09-24 (`.agents/m57/lightgbm_device_check.py`, output beside it in
`lightgbm_device_check.txt`; informal: one split, train 1999-2024, predict 2025, 491 numeric
features, the shared 200-tree settings, idle machine):

| library | CPU s/fit (12 threads) | CUDA s/fit | same seed twice on CUDA |
| --- | --- | --- | --- |
| LightGBM | `0.54` | `7.22` | differs, max `1.85` points |
| XGBoost | `1.59` | `0.70` | identical |

- XGBoost's CUDA build is faithful: with sampling off its predictions match the CPU's to a mean
  of `0.038` points; with sampling on they differ only as much as a CPU re-seed does.
- LightGBM's CUDA learner is a different approximation, not reproducible, and 13 times slower on
  this data. With sampling off it still differs from its CPU build by a mean `0.72` points (a
  re-seed moves `0.83`), and the same seed twice differs by up to `1.85` points, which breaks
  the "fixed seed, fixed output" rule. When this milestone reopens, LightGBM runs on the CPU;
  the CUDA build is optional.

---

## Milestone 58 - Web UI: remaining phases

Added 2026-09-11. Phases 0-3 (scaffolding and auth, the read-only pages, jobs, future-week
predictions) and task 58.4 (housekeeping) are done and archived under "Milestone 58 (partial)"
in `ARCHIVE.md`; the design, decisions and per-phase status live in `web_ui_plan.md`. Work
happens on a fresh branch off `main` (the `../nfl-predictor-web` worktree is fine for it) and
merges back after each phase.

- [ ] 58.1 Phase 4: pool helpers (confidence pool sheet, tiebreakers, survivor optimizer) per
      `web_ui_plan.md`.
- [ ] 58.2 Phase 5: team and QB pages over the `nfl-sos-ratings` Parquet outputs.
- [ ] 58.3 Phase 6 (design only): live betting and live odds.
- [x] 58.4 Housekeeping: the `web` extra and the ETL's upstream data-directory paths. Closed
      2026-09-21 (`0.12.12`, narrowed and accepted by the user's decision): the `web` extra is
      dropped, and `etl_full`/`validate_offline`/`validate_live` now take `--data-dir` with the
      job templates passing `NFLP_DATA_DIR`. The ETL's upstream inputs (the copied `qb_elos.csv`,
      the quarterback identity file, and the TeamRankings/nflreadpy caches) still resolve from
      `constants.DATA_PATH` and will not follow `NFLP_DATA_DIR`; threading a data directory
      through those read paths would be a separate change, not wanted for now. Full record:
      `ARCHIVE.md`, Milestone 58, "58.4".

Acceptance:

- [ ] Each phase ships with `tests/api/` and vitest coverage, the Python and frontend gates green,
      a `CHANGELOG.md` entry under a new version, and the plan's Status section updated.

---

## Milestone 60 - CLI and entrypoint consolidation

Added 2026-09-21 by the user's direction: the entrypoints' flags have grown bloated, and some of
them no longer do anything. Widened on 2026-09-23 by the user's direction to cover every file under
`scripts/` as well as every flag: each script is retired, moved into `nfl_predictor/` (as a new
module or into an existing one), or kept in `scripts/` for a stated reason.

How it runs (user's direction, 2026-09-23): after task 55.8 is redone, reviewed, committed and
merged, in a **new session on its own branch off `main`**, owned by the reviewing agent (Claude)
rather than delegated. It touches the web API's job runner, CI, the gate, the tests and most of the
docs, so every change is recorded in `CHANGELOG.md` and the web UI plan
(`web_ui_plan.md`, "Milestone 60 impact") is amended in the same chunk that changes the API.

Ground rules for the whole milestone:

- **Behavior-preserving.** Moves, renames and removals must not change a prediction, a
  probability, a pick, a ranking or a report. A characterization test pins the weekly run's
  outputs on fixed inputs before the first move and must pass unchanged after every chunk. A
  behavior defect found along the way is listed and decided separately (see task 56.5), never
  folded into a move.
- **Mid-season.** The weekly run must work every week while the milestone is open, so it lands in
  small chunks, each merged only when the gate (and `scripts/gate.sh --web` when
  `nfl_predictor/api/` or `web/` changed) is green.
- **Evidence by script.** Inventories are generated from the code by a script checked in beside
  them, never written by hand (see the 60.1 note below for why).
- Edits under `nfl_predictor/ml/` change every walk-forward checkpoint fingerprint, so no
  reference walk-forward is started while this milestone is moving ML code.

Tasks 60.1-60.3 (the generated inventories and the user's sign-off), 60.4 (the
characterization tests), 60.5 (the library moves) and 60.6 (the front door, retirements,
renames and removals, `0.19.0`-`0.26.0`) are done and archived under "Milestone 60 (partial)"
in `ARCHIVE.md`; the decisions are in `.agents/m60/PROPOSAL.md`,
"Sign-off". In short: one `nfl-predictor <command>` front door; retire `golden_command`,
`backtest_predictions`, `objective_compare_models` (with `nfl_predictor/ml/model_compare.py`),
`betting_pipeline` (after `build_betting_report` moves) and the whole Excel betting workbook;
remove ScoreModel (task 55.5); move everything else; keep `scripts/gate.sh`; one naming rule
with old spellings kept as aliases; old launchers reproduce from their recorded commit.

- [ ] 60.7 Web API and frontend: update the job templates in
      `nfl_predictor/api/jobs/catalog.py` to the new commands; keep the progress lines the runner
      parses (`Walk-forward fold N/M` from the walk-forward loop and `WF candidate N/M` from the
      weekly run, matched by `PROGRESS_RE` in `nfl_predictor/api/jobs/runner.py`); keep the
      `weekly_run` template's config keys valid (saved job configs under `data/web/job_configs/`
      use them); update `tests/api/`; run `scripts/gate.sh --web`; amend `web_ui_plan.md` and
      `web/README.md`. Also here, test first: fix the model-kind vocabulary defect (canonical
      `blend`, `blended_margin_total` accepted as an alias; three web launches fail today, and
      the user saw the train failure in the web UI). The `betting_xlsx` job, the workbook
      download routes and the Betting page's download button were removed early, in `0.19.0`.
      The API catch-all bug (task 58.5) was fixed first, in `0.26.1` (`ARCHIVE.md`, Milestone
      58, "58.5").
      Found 2026-09-25 (question for the user before fixing): with the vocabulary fixed, the
      power rankings still cannot use a blend model. `_predict_future_games` in
      `nfl_predictor/reporting/power_rankings.py` reads `model.feature_spec`, and a blend keeps
      it on `model.team_model`, so a blend run fails with "Model is missing feature_spec"
      before its blend branch is reached (pinned by `tests/test_blended_model_paths.py`). It
      has never worked. Options: read the team model's spec for a blend, or reject blend runs
      for rankings with a clear message.
- [ ] 60.8 CI, gate and docs: the CLI smoke checks in `scripts/gate.sh`; CI
      (`.github/workflows/validation.yml`) calls `scripts/gate.sh` instead of repeating its
      steps; `README.md` (the Scripts section, every command example, and a "Reproducing an old
      run" note: `git worktree add <commit>`, then the launcher as written); `AGENTS.md` ("Repo
      scripts", "Project Shape", "Dev Workflows", launch instructions); `CHANGELOG.md`;
      `ARCHIVE.md`; the handoff prompt.
- [ ] 60.9 A `compare` command: paired comparison of two walk-forward runs from their fold
      checkpoints (candidate minus reference per game, bootstrap intervals, the windows week 1,
      week 2, weeks 3-18 and all weeks; deterministic Brier, log loss, pick accuracy, margin and
      total MAE, pool points; two seeds combined per game when given), replacing
      `objective_compare_models` and the per-run `compare_to_benchmark.py` copies under
      `models/`. Tooling only; it changes no production output. Its numbers must reproduce an
      existing `REVIEW.md` rescore exactly before it is trusted.

Acceptance:

- [ ] Every flag and every file under `scripts/` has a disposition the user signed off on, in a
      script-generated inventory checked into `.agents/`.
- [ ] The weekly-run characterization test passes unchanged from the first chunk to the last.
- [ ] No production code is imported from `scripts/`, and the moved code counts toward coverage
      with the `90%` floor still met.
- [ ] Every web job template launches and reports progress (tests plus one live check per
      template), and `web_ui_plan.md` records the changes.
- [ ] Every removal or rename has a deprecation or removal note in `CHANGELOG.md`, and the gate
      (with `--web`) is green on the final tree.

---

## Open follow-ups from completed milestones

Each group names the archived milestone it came from; the milestone's full record is in
`ARCHIVE.md`. Resolved items have moved there.

### From the 2026 Week 3 weekly run (2026-09-24)

Found while producing the Week 3 picks. Runs: `models/weekly_2026_week_03_fast/` (the picks the
user submitted) and `models/weekly_2026_week_03_full/` (the rerun with a fresh ETL, for the pick
'em pool). The stopped first run and the two data backups were deleted with the user's approval
on 2026-09-24 (`ARCHIVE.md`, resolved follow-ups).

- [ ] **GPU never used by default.** `xgb_device` defaults to none (the CPU), so stage 1 ran on
      the CPU (about 6 min per candidate against about 1 min on the GPU with `--xgb-device
      cuda`). The user chose the GPU for everything (task 55.4).
- [ ] **The final fit never trains on the newest weeks.** Production and walk-forward both hold
      out the newest 4 completed weeks from the tree fit and use them only for calibration, so
      the Week 3 model's trees did not see 2026 Weeks 1-2, which reach it only through the
      features. Measure a refit on all rows (or a smaller hold-out) in step 3 with the
      out-of-fold calibration pool; the user asked whether this is a design flaw.
- [ ] **Stage-1 selection and pick-time timing.** Already scheduled (task 56.5): this week's
      stage 1 chose `none` (the deterministic map) over 2025 weeks 3-18, Brier `0.2161`, with
      `elo` + blend `0.2173`; last week it chose `elo` + blend. The winner flips on noise.
- [ ] **Stage 1 got slower.** Candidates now fit the shared 200-tree default at learning
      rate `0.0165` (last week's candidate keys show 120 trees at `0.070`). Expected, but it
      makes the GPU default matter more.
- [ ] **ETL rebuilds all 28 seasons every run** (about 9.5 minutes on 2026-09-24: about 19 s
      per season in `collect_all_data`, then QB features). Only the raw downloads are cached;
      every feature row from 1999 on is recomputed. An incremental mode (reprocess the current
      season plus what its week-1 priors need, reuse cached finished seasons keyed on code
      version and input hashes, and prove identical output with a characterization test) fits
      step 4 (rebuild reproducibility). The ETL is Polars on the CPU; the GPU does not help it.
- [ ] **Betting report edges.** The largest moneyline "edges" in Week 3 mostly reflect the gap
      between each game's spread and its moneyline (the market-anchored margin maps through the
      deterministic curve, while the edge compares with the no-vig moneyline). Keep the report
      diagnostic; revisit with task 56.6 (pick-time lines).

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
- [x] Narrowed 59.3: `n_estimators` was untuned; task 55.7 carried it and closed 2026-09-21 with
      a measured tree-budget ladder rather than the season-sized automatic tuning 59.3 asked for
      (`200` adopted as the shared default). See `ARCHIVE.md`, Milestone 55, "55.7".
- [x] ETL rebuild for 59.4 and 59.6: done 2026-09-20 (`0.12.6`), a tie on the instrument;
      record under `ARCHIVE.md`, Milestone 59, "Rebuild". The first pass exposed a `0.12.4`
      defect (the sack exclusion starved `opponent_points_per_play`), fixed before the second
      pass.

### From Milestone 45 (play-by-play EPA families)

- [x] `red_zone_tds` counted a touchdown by either team on a red-zone play, so a pick-six was
      credited to the offense. Fixed as part of Milestone 54 (`pbp.py` now requires
      `td_team == posteam`; see `ARCHIVE.md`, Milestone 54). The raw count still is not published
      as a standalone column, but the corrected count now feeds the published `red_zone_td_pct`
      rate (`--tr-stats-source pbp`).
- [ ] Half of `PBP_COUNT_COLUMNS` (third/fourth down, red zone, two-point, total plays) is joined,
      aggregated and regressed but never published as raw counts. Milestone 54 published derived
      rates from some of them (`red_zone_td_pct`, `two_point_conversion_pct`, and the third/fourth
      down percentages) via `--tr-stats-source pbp`; the raw counts themselves remain unpublished
      intermediates. Publish the raw counts or stop carrying the ones no rate reads.
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

### From task 56.4 (rolling calibration window)

- [x] Closed 2026-09-20: superseded by `0.12.3`, which removed in-season early stopping from
      production and walk-forward. No head stops on the calibration frame any more, so the
      defect this item described cannot recur. The historical evidence, kept for the record:
      the Week-2 smoke run (`models/smoke_20260911`, market-anchored) stopped its margin head at
      iteration `1` on a 50-game window; the workaround run calibrated on the 2025 and 2026
      seasons (274 games) stopped at `189` (`models/smoke_20260911_workaround/`), and
      `models/weekly_2025_week_22` (8 weeks) at `9`. Every head now runs the full tree budget,
      and the only question left is how large that budget should be, which is task 55.7.
