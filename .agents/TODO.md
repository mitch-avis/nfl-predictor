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
  of `ARCHIVE.md`). Milestones 54, 59 and 60 are also fully closed and archived; their remaining
  follow-ups live under "Open follow-ups from completed milestones" below. Active milestones with
  open tasks are 53, 55, 56 and 58 (the web UI, added 2026-09-11), so new work starts at 61.

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
   (never `[Unreleased]`), set `pyproject.toml` to that version, and run `uv lock` and
   `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`.
   Never create a git tag or GitHub release (see `AGENTS.md`).
7. **Commit** (when asked) with Conventional Commits subjects, `type(scope): summary`, per
   `AGENTS.md`.

### Always use the repo venv

Agents and humans should not rely on the shell activation state.

- Do **not** run `python`, `pip`, `pytest`, `ruff`, `pyright`, or `ty` without the venv prefix.
- Use `.venv/bin/python ...` or the tool-specific binary under `.venv/bin/`.
- Use `uv ...` from `PATH` for dependency management and environment sync.

### Current validated baseline (2026-10-16, production as of `0.63.0`)

- Production (`config/weekly_run.yaml`, code defaults): the XGBoost margin and total heads, 200
      trees, unweighted, market mode `hybrid` (line features and the anchor, the transform on),
      the GPU when present, and the deterministic floor `Phi(margin / sigma)`. It leaves out
      `strength, own_elo, own_elo_wt, qb_adj`, so it trains on the ported team ratings
      (`ratings`) and the `qb_elos.csv` Elo (`elo`): 449 features on the through-2025 cut. The
      power rankings rank on the blend of the ported rating and the model rating at weight `0.5`.
- Reference with game lines: `models/qb_elos_measure/new_file_seed{42,7}` (code `c637fc0`,
      `0.60.0`; 2007-2025 from week 1, 328 folds, 4,943 games; independently reviewed), the
      floor-sigma reference since `0.62.0` (final pooled sigma `13.3184` and `13.3182`). It does
      not beat the pick-time line on any column in any window; against it, two seeds, the line
      leads on margin MAE all weeks by `0.0484` `[0.0057, 0.0909]` and on week-1 Brier by
      `0.00511` `[0.00007, 0.01030]` (`.agents/benchmarks.md`, "`qb_elos.csv` measurement").
- Reference without game lines (`weekly --wf-market-mode none`, `train --exclude-market`):
      `models/qb_adj_no_lines/no_lines_seed{42,7}` (code `d791846`, `0.62.0`; leaves out
      `strength, elo, own_elo_wt, qb_adj`, 431 features; independently reviewed), the floor-sigma
      reference of that mode since `0.63.0` (final pooled sigma `13.3969` and `13.4341`). The
      pick-time line leads it all weeks on Brier by `0.00373` `[0.00162, 0.00579]`, log loss by
      `0.00839` `[0.00351, 0.01313]` and margin MAE by `0.1197` `[0.0469, 0.1901]`
      (`.agents/benchmarks.md`, "The quarterback adjustment without game lines").
- Data: the ETL rebuilt on 2026-10-14 on `0.59.0` (7,340 completed rows, 541 columns,
      fingerprint `f3eca626...`, 534 ported-ratings files under `data/ratings/`); leakage audit
      `models/leakage_audit.json`, 490 features, 0 flags. Both references read the walk-forward
      input `models/qb_elos_measure/inputs/new_file.csv` (7,261 rows, 544 columns,
      `b9772e3e...49f0`).
- Run directories that predate step 6 and were deleted on 2026-10-16 are listed in
      `.agents/models_cleanup/`; numbers quoted from them stand as reviewed.

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

Order agreed with the user on 2026-09-24 and extended on 2026-10-08 (steps 5 and 6 added: the
market-free baseline and absorbing `nfl-sos-ratings`; the Optuna re-tune moved after the features
settle; the web UI's remaining phases joined step 6) and on 2026-10-15 (steps 7 and 8 added: the
Elo inputs and the feature ideas, both before the re-tune, now step 9). There is no deadline:
every step runs to satisfactory completion, including its reviews, before the next starts. Step 6
is the largest workstream so far, several weeks of sessions.

Why this order (the ordering principles, agreed with the user):

1. **Restructure before changing behavior.** Milestone 60 moves code without changing any output,
   pinned by a characterization test of the weekly run; every output-changing task comes after it,
   so the moves stay verifiable and no code is edited just before it moves.
2. **The benchmark must measure production before anything else is measured** (`AGENTS.md`
   rule 11). Parity gaps (device, probability path, line timing) close before new feature or
   tuning measurements.
3. **Features before tuning.** Anything that changes feature values lands before the Optuna
   re-tune, or the tune is fitted to a feature set that no longer exists.
4. **Batch what invalidates checkpoints.** Edits under `src/nfl_predictor/ml/` and device changes
   invalidate every walk-forward checkpoint, so they are grouped and pay for one new reference.
5. **The weekly run comes first.** Production-changing steps land between game weeks; heavy
   compute runs overnight; never two walk-forwards at once.
6. **Two seeds for any default change** (`AGENTS.md` rule 13).
7. **Each step on its own branch off `main`**, merged (must-ask) when its gate, review and
   check-in are done, so branches stay short (the user's merge-cadence preference).

Steps:

1. **Task 55.8 close-out** (branch `feat/m55-8-season-weighting`). Done 2026-09-24 as `0.18.0`
   and merged into `main`: nine six-season arms, an independent review of all nine
   (rule 3), and the user's decision for unweighted production training. This
   step's docs chunk also carries this order, the new rules, and the retirement of 55.1/55.2.
2. **Milestone 60 + task 55.5** (new branch, new session). CLI and entrypoint consolidation,
   widened to every file under `scripts/`, behavior-preserving. 55.5 (the `ScoreModel` fate) is
   decided and executed here, because retiring unused code is part of the same cleanup. Absorbs
   the follow-ups marked "(step 2)" below. Done 2026-09-25 as `0.18.1`-`0.28.1` on
   `feat/m60-cli-consolidation` and merged into `main` with the user's approval; archived as
   Milestone 60 in `ARCHIVE.md`; its two leftovers were resolved in step 3 (same
   section, "Leftovers resolved").
3. **Tasks 55.4 + 56.5 + 56.6 + the out-of-fold calibration pool** (new branch). Close the
   production/benchmark parity gaps together, because each changes what production outputs and
   they can share one new reference: the GPU as the default device for every run (55.4, decided
   by the user 2026-09-23, with a GPU determinism check and a two-seed GPU reference); how
   production probabilities are formed (56.5, decided by the user from evidence, mostly by
   rescoring saved predictions); the line-timing yardstick (56.6); and the out-of-fold calibration
   pool (follow-up from 59.2), so fitted calibrators can be judged fairly as 56.5 options.
   Task 55.6 (the per-season stability view) lands here as part of the standard walk-forward
   report. Absorbs the follow-ups marked "(step 3)".
   Plan accepted by the user on 2026-09-27 (branch `feat/step3-parity`): (A) evidence with no
   `ml/` edits: the one-fold CPU-vs-GPU and determinism check (rung L0), the independent rescore
   of the probability paths on the task 55.8 checkpoints (`models/step3_prob_paths/`), and 56.6
   (b)/(c) (`models/step3_open_lines/`); (B) every `ml/` edit in one batch, one implementer
   subagent and one review per chunk; (C) the GPU reference, six seasons from week 1 on seeds
   `42` and `7` (rungs L1-L2); (D) a no-holdout arm (`--wf-calibration-weeks 0`) on two seeds
   (rungs L3-L4) only if (B) leaves the holdout with no purpose. Ladder cap: four six-season runs
   plus L0; each rung's hypothesis and rule are in its run directory. 56.6(d) moved to step 4
   the same day (it needs a dataset build). Changed by the user the same day: L1-L2 cover
   2007-2025 (every season with real opening lines, for 56.6(c)); rung (D) is dropped, because
   walk-forward folds never held weeks out and `0.33.0` moved production to match, so the arm
   would reproduce the reference exactly. Ladder: L0-L2.
   Added by the user on 2026-09-28: measure the spread in the probability floor
   `Phi(margin / sigma)` (today the constant `SCORE_DIFF_STD_DEV = 14.21`) against sigmas estimated
   from the model's own earlier errors, by rescoring the GPU reference
   (`models/step3_sigma/HYPOTHESIS.md`); adopting one is a default change (must-ask).
   The user adopted `expanding` on 2026-09-28 (`0.35.0`), with picks and ranks from the unrounded
   probability and the weekly stage 1 from week 1.
   Standing process item (the user, 2026-09-28): rerun the GPU reference (two seeds, 2007 through
   the last completed season, current code) whenever a step changes the model (steps 4 and 5
   already require a new reference) and once each off-season before week 1, so the floor's sigma
   pool describes the current model; point `floor_sigma_reference_runs` at the new runs.
   Closed by the user on 2026-10-01 (record: `ARCHIVE.md`, "Roadmap step 3").
4. **Reproducibility, then the feature values.** Closed by the user on 2026-10-10 (record:
   `ARCHIVE.md`, "Roadmap step 4"). Its leftovers moved to step 6: 53.7's ridge form, the
   early-season strength-snapshot item and the feature follow-up group to phases 3-4, and the
   betting-report edges to phase 5.
5. **Market-free baseline.** Closed by the user on 2026-10-10 (record: `ARCHIVE.md`, "Roadmap
   step 5"). The test bench for step 6 is the run without betting-line or Elo columns.
6. **Absorb `nfl-sos-ratings` into this repo** (approved 2026-10-08: absorb fully; that repo
   finishes its in-flight work, the port starts from a frozen commit of it, and it then becomes
   read-only). The frozen commit is `9081bce` (2026-10-09); it, what changed there since this
   plan, and what is still open or fragile there are in `.agents/sos_port_source.md`. Each phase
   on its own branch, with an independent review:
   1. a scripted inventory (rule 10) of its data sources, metric registry columns, ratings,
      outputs and web pages against this repo's, plus a check that this repo's play-by-play
      aggregates match its game logs for the same seasons;
      plan accepted by the user on 2026-10-10 (branch `feat/sos-port-inventory`, scripts and
      their generated reports in `.agents/sos_port/`, following `.agents/m60/inventory.py`):
      - the inventory covers everything: the nflreadpy sources each repo loads, every column of
        its per-season output files, its metric registry, its ratings, and its web pages and API
        endpoints. A checked-in mapping file classifies each of its columns (matched, a different
        definition, or missing here), and the script fails on any unclassified column;
      - parity is exact and explained: for 1999-2025, per team-game, counts equal and EPA sums
        within `1e-6` on the columns both repos define the same way, and every mismatching game
        assigned to a named cause (pre-2006 scrambles, Jacksonville's 2001-2002 home games, the
        Rams as `LA`, the three games without play-by-play, a definitional difference); an
        unexplained mismatch fails;
      - report only: phase 1 changes no production code or data. Its differences (the pre-2006
        scramble rule first) become a list of proposed changes for the user, batched with
        phase 3's feature-value changes so they share one rebuild and one new reference.
        Changed by the user later on 2026-10-10 (about 02:00): step 6 runs autonomously, and every
        difference is decided by the session and logged with its reasoning and data in
        `.agents/sos_port/DECISIONS.md` for the user's review; the adopted changes land on the
        step-6 integration branch, and the production rebuild and merge stay held for the user's
        final approval (`.agents/step6_plan.md`, "Authority");
   2. the nflverse sources it uses and this repo lacks (weekly player stats, snap counts, players
      and rosters for quarterback identity; ESPN QBR as a validation reference only), with the
      cache-then-degrade pattern;
   3. its team ratings (points per game against an average team, offense, defense and special
      teams, the cross-validated ridge penalty, the preseason prior inside the fit, the
      head-to-head-excluded `sos` refit), quarterback ratings, weekly histories and rank ranges,
      with tests that match its published outputs; this replaces the current strength snapshot
      and the defense-adjusted quarterback rate;
   4. the ported ratings as model features, measured anchored against the adopted-settings
      reference and market-free on the test bench the user chose 2026-10-10 (no betting-line and
      no Elo columns, `models/step4_reference/marketfree_all_seed*`);
      a candidate in the same phase (approved by the user 2026-10-10): this repo's own Elo,
      computed in the ETL, replacing the `qb_elos.csv` columns if it measures at least as well
      on the test bench and anchored. Build it in this order, each with tests: 538's public
      framework first, pinned by a test that reproduces `elo_prob1` in `fivethirtyeight/
      nfl-elo-game` `data/nfl_games.csv` (K 20, home field 65, a one-third offseason pull to
      1505, the margin-of-victory multiplier); then the 2019 additions (rest and bye, travel,
      playoff boost, rolling home field); then the QB adjustment, 538's published VALUE formula
      first and the ported `nfl-sos-ratings` quarterback rating as the alternative; then the
      preseason pull toward betting win totals as a measured option, read and cached from the
      free `greerreNFL/nfelosrs` `wt_ratings.csv` (2003 on; plain regression before), its weight
      fitted on earlier seasons only. The market-free test bench never gets the win totals.
      After this Elo replaces the `qb_elos.csv` columns (and only then, to keep a production
      input stable mid-season), fix the user's `../nfeloqb` fork to read the upstream win-total
      file instead of its stale local `Manual Data/wt_ratings.csv` (2003-2023), with the user's
      go-ahead at that time. Research: `~/scratch/elo_research/`.
   5. power rankings and projected standings on the ported ratings combined with the trained
      model (the user's direction 2026-10-08; how to combine them is decided by measurement:
      each candidate ranking predicts next week's margins, scored like the `nfl-sos-ratings`
      validation);
   6. its web pages (teams, quarterbacks, schedules, rank ranges, head-to-head) folded into this
      app, together with Milestone 58 phases 4-6.
   The user's decisions of 2026-10-11, 2026-10-12 and 2026-10-14 are recorded in
   `.agents/step6_plan.md` ("User decisions after the run"): the blend default merged last as a
   minor version; the ported ratings replace the strength snapshot in production; the
   play-by-play corrections stay (a tie); the `qb_elos.csv` columns stay and the own Elo stays
   off, tentatively; the configuration without game lines is named "no game lines", keeps the own
   Elo without win totals (measured a tie) and leaves out `strength, elo, own_elo_wt` (with
   `qb_adj` since `0.61.0`); no rating-definition outcome is adopted before a post-merge
   measurement; the merge itself. The reviews' small open findings are listed under "From the
   step-6 integration reviews" below.
   Merged into `main` as `0.59.0` on 2026-10-14 with the user's approval.
   Status 2026-10-16: phases 1-5 are done and independently reviewed (the release index in
   `.agents/sos_port/DECISIONS.md`, `0.46.0`-`0.59.0`), the ported ratings adopted in production
   in `0.59.0`. Phase 6 is done except for two narrowed items: the web pages landed in `0.53.0`
   (two review rounds) and Milestone 58's live-betting design is archived (58.3); 58.1 waits for
   the user's acceptance of the exact survivor planner, and 58.2 for its done-criterion, a first
   look at a current-season team page in a browser (the data rebuild it also waited on is done
   since 2026-10-14). The docs close-out is on `docs/step6-close-out` (2026-10-16, not merged).
   Step 6 closes when 58.1 and 58.2 are accepted.
7. **The Elo inputs** (added by the user 2026-10-15; each on its own branch off `main`):
   - done 2026-10-15 (`0.60.0`-`0.60.1`, independently reviewed, merged into `main`):
     `qb_elos.csv`'s `qb1_adj`/`qb2_adj` published as the `qb_adj` group, off in production until
     measured; `0.60.1` limits the quarterback fill to unplayed games;
   - done 2026-10-15 (independently reviewed, `models/qb_elos_measure/REVIEW.md`; recorded in
     `.agents/benchmarks.md`, "`qb_elos.csv` measurement"): the grouped measurement in
     `models/qb_elos_measure/HYPOTHESIS.md`, the upgraded `qb_elos.csv` against the old one, the
     `qb_adj` arm, and decision 1's ratings swap re-checked on the new file; two builds, eight runs:
     - the new file against the old: a tie, the new file stays;
     - `qb_adj`: a tie, the group stays off;
     - the ratings swap on the new file: the strength snapshot loses, the ported ratings stay;
     - the new-file runs are the floor-sigma reference from `0.62.0` (reviewed, merged);
   - done 2026-10-15 (`0.61.0`, independently reviewed, merged into `main`): a run without game
     lines leaves out `strength,elo,own_elo_wt,qb_adj` by default, and a run counts as one when
     its market features and anchor are off, whatever the transform;
   - done 2026-10-15 (independently reviewed, `models/qb_adj_no_lines/REVIEW.md`; recorded in
     `.agents/benchmarks.md`, "The quarterback adjustment without game lines"): `qb_adj` in the run
     without game lines (approved by the user 2026-10-15), `models/qb_adj_no_lines/HYPOTHESIS.md`,
     sha256 `8c1028438a66594c8ecb1445ed8fcf4403ccf0e7a8c2cd3a1f1ed4c34519ce06`; two arms on the
     grouped measurement's new-file cut, two seeds, four runs, code at `main` `d791846`: a tie, the
     group stays in `constants.NO_GAME_LINES_DISABLED_FEATURE_GROUPS` and no default changes; the
     review's floor-sigma gap is fixed in `0.63.0` (`ARCHIVE.md`);
   - open: `../nfeloqb`, a backup's first start in next-week projections rated as played rows
     rate it (`get_value(team)` in place of `current_value`), test first;
   - open: the fork's win-total file ends at 2023: its own measurement after the grouped one,
     with a fork change and a third build, landing between game weeks.
8. **Feature ideas** (raised by the user 2026-10-08; moved before the re-tune 2026-10-15, to be
   explored quickly and then accepted or dropped): the referee crew (nflverse `referee` since
   1999, full crews in `load_officials`), game-day weather (temperature and wind for outdoor
   games), attendance, and team health (nflverse injury reports since 2009). Each first passes a
   pick-time check (what is known before Thursday for an upcoming game), then a measured arm.
9. **Tasks 55.9 + 56.3** (new branch). The Optuna re-tune on the final features and device, its
   three prerequisites first, then the study, a six-season confirmation on two seeds, and 56.3
   (wiring the chosen settings into the weekly run and the benchmark, one shared source).

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
| 2 (Milestone 60) | All resolved in Milestone 60 (`ARCHIVE.md`, "Step-2 follow-ups, resolved in Milestone 60") |
| 3 (parity) | Narrowed 59.2, the out-of-fold calibration pool (Milestone 59); the four calibration-window items: `select_calibration_data`, `--train-calibration-seasons 1`, `--include-postseason` roll-back, `_split_train_calibration_holdout` (2026-09-11 review); the per-template live web checks and the blend power rankings (Milestone 60); the feature-importance aggregation (2026-09-25 review) |
| 4, first | Done, apart from the low-priority schedule fallback (step-3 merge test group). The play-by-play cache key (Milestone 45), bit-reproducible rebuilds (Milestone 46), the CSV type guessing (step-3 merge test), the incremental ETL (Week 3 run), the split-review follow-ups and the schedule cache were resolved in `0.37.2`-`0.38.2` |
| 6, phases 3-4 (from step 4) | Checked 2026-10-16 by `.agents/sos_port/followup_check.py`: four retired (feed neither the production model nor the default power rankings) and two resolved in `0.54.0`, recorded in `ARCHIVE.md`, "Roadmap step 6 - Follow-ups from step 4". Still open, each feeding production: the early-season stat blend's weeks 3-18 margin MAE cost (Milestone 49, with 55.3 in step 9); what is left of `_attach_qb_features`, of scramble attribution and of the QB identity chain (2026-09-11 review) |

Rules for every feature milestone:

- Land each family as one ablatable feature group with a walk-forward switch (mirror
  `--disable-trend-features`).
- Report a walk-forward comparison with the group on and off before marking done, read on the
  deterministic and market columns (task 59.1), over enough seasons to resolve the effect claimed.
  The fit-noise floor (`.agents/benchmarks.md`, "Fit-noise floor on the same build") is the
  yardstick: on three seasons a Brier difference under about `0.002`, a pick-accuracy difference
  under about `0.01` or a margin MAE difference under about `0.06` is re-seeding noise. An arm that
  claims an improvement runs on six seasons (`--eval-last-n-seasons 6`, about 100 minutes idle) and
  on two seeds (`AGENTS.md` rule 13); a three-season arm can only show a tie. The 2026-09-23 seed-7
  pair showed that on six seasons, re-seeding alone can move pick accuracy and pool points by
  amounts whose game-resampled intervals exclude zero, so a single-seed six-season "win" is not a
  result.
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
- [ ] 53.7 (step 6, phase 3; two seeds per adoption decision) Opponent-adjusted quarterback
      EPA, in two steps. First (cheap) a defense-adjusted
      rate: per quarterback game, `qb_epa_sum - dropbacks * expected_epa_allowed` where the
      expectation is the faced defense's pre-game `adj_def_pass_epa_snap` (or its one-hop
      profile), aggregated with the same `K = 300` shrinkage as `qb_dropback_epa` (career and
      last-8 windows) so it lands next to the existing rates and is measured the same way. The
      quarterback-game rows already carry `opponent_abbr` for this. Only if that shows signal,
      the ridge: dropback-weighted quarterback effects solved jointly with defenses (design in
      `nfl-sos-ratings` `qb_rating.fit_qb_ratings` on its shared `ridge.py` solver). Ceiling to
      keep in mind: the team-level `adj_off_pass_epa` is already opponent-adjusted, so the gain
      is confined to where the quarterback's history diverges from the team's (new and traded
      starters).
      Progress: the first step (`0.40.0`-`0.41.0`) tied the current-settings baseline in the
      defense-adjusted quarterback test (`models/step4/REVIEW.md`); the user dropped it
      (2026-10-02) and it was removed in `0.44.0`. The ridge form is no longer built here: it is
      the `nfl-sos-ratings` quarterback rating, ported in roadmap step 6, phase 3.

2026 Week 2 weekly run: completed 2026-09-17 18:36 MDT (`--run-id
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

## Milestone 55 - Tune the model and lock the default settings

Goal: settle every modeling setting the weekly run ships, each by a walk-forward measurement
under the canonical protocol, with its hypothesis and decision rule written first and two seeds
before any default changes (rules 4 and 13). The settings are the device (55.4), the
early-season prior blend `K` (55.3), the tree budget (55.7) and season weighting (55.8), and
last the Optuna re-tune of the XGBoost hyperparameters on the final features and device (55.9).
The confirmed settings become the one shared default that the weekly run and the benchmark both
read (task 56.3), and the standard report shows how stable they are by season and week (55.6).
The tune runs after the feature-value changes of roadmap step 4, because each of them would
invalidate it.

Tasks:

- [x] 55.7 Choose `n_estimators` time-aware: closed 2026-09-21 (`0.13.0`-`0.13.1`). A four-rung
      ladder (`200`/`400`/`598`, then a `100` plateau check), each rung a six-season arm with its
      own hypothesis and an independent reviewer rescore, found the aggregate order monotone
      toward fewer trees but small; the user adopted `200` as the shared default `n_estimators`
      in walk-forward and production together, and `config/weekly_run.yaml`'s walk-forward stage
      was aligned to the production XGBoost defaults in the same chunk. Full record, the ladder
      table and the pairwise intervals: `ARCHIVE.md`, Milestone 55, "55.7", and `.agents/benchmarks.md`
      under "Tree-budget ladder".
- [x] 55.8 Season weighting: closed 2026-09-24 (`0.18.0`). Nine six-season arms (unweighted and
      half-lives `4`, `8`, `16`, `32`; seeds `42` and `7`, half-life `8` at seed 42 only), an
      independent review of all nine (`models/wf_m55_8_review/INDEPENDENT_REVIEW.md`), and the
      user's decision: production trains **unweighted**, in the weekly walk-forward stage and the
      final fit alike. Half-life `4` (shipped since `83ba2a7` and never measured before) lost on
      two seeds; `16` and `32` tied. Record: `ARCHIVE.md`, Milestone 55, "55.8", and
      `.agents/benchmarks.md` under "Season weighting and the six-season fit-noise floor".
- [ ] 55.9 (step 9, with 56.3) Optuna re-tune. Added 2026-09-21 by the user's decision, to run
      **after Milestone 54 lands** (tuning before the feature set changes would have to be redone).
      Scheduled 2026-09-24 as roadmap step 5 (step 7 from 2026-10-08, step 9 since 2026-10-15),
      after every feature-value change and on the final device (step 3), so the tune is fitted to
      the feature set that ships. It occupies the machine for hours,
      so it runs overnight. Prerequisites, each its own tested chunk before any
      trial runs: (1) trials must fit exactly the way production fits. `0.12.3` removed in-season
      early stopping from production and walk-forward, but `_score_margin_total_fold` in
      `src/nfl_predictor/ml/ml_model_core.py` still passes `early_stopping_rounds` to
      `fit_margin_total_models`, so every trial is scored on a differently fitted model. (2) the
      tuning objective must score what the walk-forward instrument scores: the deterministic
      `Phi(margin / SCORE_DIFF_STD_DEV)` Brier, with the market view alongside, not the
      configured-calibrator Brier. (3) plumbing so a tuned parameter set actually reaches the
      walk-forward. Today `weekly_run` takes explicit `wf_*` params and never writes a best-params
      file; task 56.3 adds the one shared source both read. Then the study itself: the search space
      as it stands (`n_estimators` `200`-`1200`, learning rate `0.01`-`0.3` log, depth `3`-`8`, and
      the rest), TPE with seed `42`, SQLite storage, `300`-`500` trials, about 3-8 hours. The winner
      is confirmed on a six-season walk-forward against the `200` rung and on a second seed before
      it becomes a default (must-ask). The old studies under `models/weekly_2025_week_21/` and
      `models/weekly_2025_week_22/` are void: their trials early-stopped and they predate the
      `0.6.2` guidance.

Note for any tuning: until the shared early-stopping callback was fixed (version `0.6.2`), every
Optuna trial's total head stopped after one round, so the `combined_mae` objective scored a
crippled total and every existing tuned parameter set was in effect chosen on the margin head
alone. Do not reuse old `tune_best_params_out` files or Optuna studies; re-tune from scratch.

Acceptance:

- [ ] The task 55.9 tune runs from one command and one config file; the settings it chooses,
      confirmed on two seeds and approved by the user, live in one shared source that
      `nfl-predictor weekly` and `nfl-predictor backtest` both read (task 56.3); and a weekly
      run on them runs end to end.

---

## Milestone 56 - Weekly orchestration residuals

Formerly Milestone 41.

- [x] 56.1 Add data-refresh pass-through (`--data-min-season` / `--data-max-season` or a generic
      `--data-collection-args`) to `nfl-predictor weekly`; the same pass-through carries the
      `--stat-prior-blend*` flags, which it cannot set today. Done 2026-09-20 (`0.12.11`) as the
      generic `--data-collection-args` string.
- [x] 56.2 How postseason games enter evaluation, training and the rankings: closed 2026-09-21
      (`0.13.0`). Postseason matchups stay in the data, kept separate from regular-season
      matchups for training and prediction; `include_postseason: false` and
      `wf_include_postseason: false` are now set in `config/weekly_run.yaml`, matching the code
      defaults. `postseason_weight` stays in the config but inert. The playoff-week design itself
      (a playoff-specific model or weighting) stays deferred, wanted in time for the 2026
      playoffs. Full record: `ARCHIVE.md`, Milestone 56, "56.2".
- [ ] 56.3 (step 9, with 55.9) Wire the settings chosen by the task 55.9 tune into the weekly run
      and the benchmark from one shared source (today `weekly_run` takes explicit `wf_*` params and
      never reads a best-params file); confirm resume behavior.
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
rebuilding only when needed (README, "Recommended: uv project workflow").

- [ ] Before reopening: on 2026-10-01 (CUDA 13.4, uv 0.12.21) the CUDA build twice ended up as
      the PyPI CPU wheel. Twice in `update_requirements.sh` (17:46 and 17:49; the dist-info shows
      `manylinux_2_27`, `scikit-build-core 1.0.3`) and once after a version-bump sync that passed
      the flags. Two clean scratch builds (`~/scratch/lgb_diag/`) passed `-DUSE_CUDA=ON` and
      trained on the GPU, and the flip did not reproduce there, so the cause is unknown. Known:
      a `uv sync` without `nfl-lightgbm-cuda-install uv-args` swaps the CUDA build for the CPU
      wheel by design. Since
      `0.39.2` a failed CUDA build only warns in `update_requirements.sh`.

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
happens on a fresh branch off `main` and merges back after each phase.

- [ ] 58.1 Phase 4: pool helpers (confidence pool sheet, tiebreakers, survivor optimizer) per
      `web_ui_plan.md`.
      Narrowed (2026-10-10): delivered on the step-6 integration branch as `0.48.0`, with the
      survivor planner solved exactly (an assignment of weeks to teams) in place of the plan's
      beam search, a substitute method that waits for the user's acceptance; the reason and the
      reviews' simulations are in `.agents/sos_port/decisions/web.md`, entry 9, and on the step-6
      approval list in `.agents/sos_port/DECISIONS.md`.
- [ ] 58.2 Phase 5: team and QB pages over the `nfl-sos-ratings` Parquet outputs.
      Narrowed (2026-10-10): the pages, their API and the navigation landed on the step-6
      integration branch as `0.53.0`, over the ported ratings and season statistics, but the
      phase's done-criterion (a current-season team page with heatmaps and a game log) waited on
      the production rebuild of `data/` and a first look in a browser; four source features are
      logged as not ported (`.agents/sos_port/decisions/web_pages.md`, entries 13 and 19-21).
      Status 2026-10-16: the rebuild is done (2026-10-14, `0.59.0`, with the 534 ported-ratings
      files); the browser look is still open.
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

## Open follow-ups from completed milestones

Each group names the archived milestone it came from; the milestone's full record is in
`ARCHIVE.md`. Resolved items have moved there.

### From the step-6 integration reviews (2026-10-14)

Low-priority findings the independent reviews left open; the full list, with the ones already
fixed, is in `.agents/step6_plan.md` ("Small open findings from the integration reviews").

- [ ] No test covers the phase-1 audit's run-loop call to `check_opponent_rows`
      (`.agents/sos_port/parity.py:578`); the tests call the function directly.
- [ ] The open rating-definition tests: the quarterback test's calibration check reads the
      recorded row count, not the fit, and no test shows `run_integrity_checks` calls it; the
      information-set change leaves 1.75 and nulls as they were and the team check does not flip
      later sites; the recorded commit lacks a dirty mark; `season_t_interval` returns NaN bounds
      for one season.
- [ ] The blend ranking default: check the single manifest match against the run itself
      (`wf_best.json` weeks or `created_at`); three older power-ranking skips log at info level
      (`weekly_run/pipeline.py:563-569`).
- [ ] Production adoption: the disabled-group lists are not normalized for order or duplicates;
      no snapshot test of the default fit's predictions; the weekly refresh writes where
      `--data-collection-args` say while the run reads `--data-path`.

### From the step-3 merge test (2026-09-28)

- [ ] (low priority) `reporting/power_rankings.py` `_load_current_records` groups with `group_by`
      without `maintain_order`, so its row order changes between runs (content and rankings are
      identical; found by the `0.37.2` review). The `0.37.3` determinism guard covers only the ETL
      modules. Add `maintain_order=True` (or sort) so "identical output" checks on rankings are
      not noisy.
- [ ] (low priority) `load_schedule` has no current-season fallback: unlike team stats and
      play-by-play, a failed refetch of the current season's schedule (a network error or an
      nflverse outage) fails the whole ETL. Found by the `0.38.2` review; pre-existing. If it is
      fixed, the fallback to the cached current-season schedule must log a loud warning, because
      that copy can hold stale lines and miss games scheduled since.
- [x] (step 4) The `*_next_opponent_abbr` pair enters the model as 32 one-hot columns each (the
      lookahead family); the trees split on them rarely (`importance_aggregation.py`), and
      `*_next_opponent_win_pct` already carries the next opponent's strength. Measure dropping
      the identity pair (on/off, two seeds) with the other feature-value changes. Measured in the
      next-opponent identity test (a tie, `models/step4/REVIEW.md`); dropped from the model by
      the user (2026-10-02) and pruned by default in `0.44.0`. The ETL still writes the pair.

### From the 2026 Week 3 weekly run (2026-09-24)

Found while producing the Week 3 picks. Runs: `models/weekly_2026_week_03_fast/` (the picks the
user submitted) and `models/weekly_2026_week_03_full/` (the rerun with a fresh ETL, for the pick
'em pool). The stopped first run and the two data backups were deleted with the user's approval
on 2026-09-24 (`ARCHIVE.md`, resolved follow-ups).

- [x] **GPU never used by default.** `xgb_device` defaulted to none (the CPU), so stage 1 ran on
      the CPU (about 6 min per candidate against about 1 min on the GPU with `--xgb-device
      cuda`). The user chose the GPU for everything (task 55.4). Fixed in `0.30.0`: every run
      defaults to `auto` (the GPU when usable). The measurement parts stay under task 55.4.
- [x] **The final fit never trains on the newest weeks.** Resolved in `0.33.0` by the user's
      decision: the final fit trains on every eligible completed game, like every walk-forward
      fold. Originally: Production and walk-forward both hold
      out the newest 4 completed weeks from the tree fit and use them only for calibration, so
      the Week 3 model's trees did not see 2026 Weeks 1-2, which reach it only through the
      features. Measure a refit on all rows (or a smaller hold-out) in step 3 with the
      out-of-fold calibration pool; the user asked whether this is a design flaw.
      Correction (2026-09-27, verified by an independent review): only the final fit holds the
      newest 4 weeks out of its trees; walk-forward folds train on every earlier game, so every
      benchmark number measures a model without the hold-out. Which side moves is the user's
      step-3 question (recommendation: production stops holding out).
- [x] **Stage-1 selection and pick-time timing.** Resolved in `0.32.0` (task 56.5): the
      stage-1 re-selection is retired and the floor is submitted. Originally: this week's
      stage 1 chose `none` (the deterministic map) over 2025 weeks 3-18, Brier `0.2161`, with
      `elo` + blend `0.2173`; last week it chose `elo` + blend. The winner flips on noise.
- [ ] (step 6, phase 5) **Betting report edges.** The largest moneyline "edges" in Week 3
      mostly reflect the gap between each game's spread and its moneyline (the market-anchored
      margin maps through the deterministic curve, while the edge compares with the no-vig
      moneyline). Keep the report diagnostic; revisit with task 56.6 (the pick-time line).

### From Milestone 59 (benchmark instrument; audited 2026-09-19)

- [x] Narrowed 59.2 (moot since `0.32.0`: no fitted calibrator remains, task 56.5): the
      calibration frame for fitted calibrators is the previous two seasons
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
      Status 2026-10-16 (`.agents/sos_port/followup_check.py`): still feeds production. The
      strength snapshot's blend no longer does (the `strength` group is left out), but every
      season-to-date stat is blended with the prior season by the same `K`
      (`stat_prior_blend_games` in `utils/polars/week_rows.py`); stays with 55.3 in step 9.

### From the 2026-09-11 code review of `0.6.1`-`0.7.0`

Fixed in `0.7.1`: the season guard, the window log line, the quarterback history refresh and
the identity warning. Still open:

- [x] `walk_forward.select_calibration_data` (retired in `0.33.0` with the calibration frame;
      see the Week 3 hold-out item) still takes calibration weeks from the eval season
      only and returns an empty frame when fewer than `calibration_weeks` exist, so walk-forward
      folds for weeks 2-4 fit with no calibration frame (and no early stopping) while production
      training now rolls the window back into the previous season. The backtest therefore does
      not measure the early-season regime the weekly model uses. Roll the walk-forward window
      back the same way and re-measure from week 1.
      Narrowed (2026-09-27): the premise predates the pooled calibration frame (`0.12.x`). Today
      the frame is the previous two seasons plus the eval season's completed weeks, never empty,
      and walk-forward folds hold nothing out of the tree fit; the real gap is the final fit's
      hold-out (the Week 3 item "The final fit never trains on the newest weeks"), a user
      question. `0.30.1` documents both paths in the help, config and README.
- [x] With `--train-calibration-seasons 1` and the default four weeks, the whole calibration
      season is the newest season the window does not touch, so it jumps (for example from 2024
      to 2025) between weeks 4 and 5 of a season and swaps a whole season between train and
      calibration. Correct by construction, but not logged; log the candidates or document it.
      Done in `0.30.1`: the final fit logs the calibration seasons and why they move, and each
      walk-forward fold logs its calibration frame.
- [x] With `--include-postseason` (moot since `0.33.0`: no window rolls back), the rolled-back
      window is filled by the previous season's
      playoff weeks (divisional, conference, Super Bowl: seven games), so a week-1 or week-2
      window can hold about 23 games. The weekly default excludes the postseason; consider
      skipping postseason weeks when the window rolls back, or counting games instead of weeks.
      Open (2026-09-27): moot if production stops holding weeks out (the user's step-3
      question); otherwise the recommendation is to skip postseason weeks when rolling back.
- [x] `_split_train_calibration_holdout` still returns the lossy `(newest season, its weeks)`
      pair and `_inseason_calibration_pairs` re-derives the window from the calibration frame;
      returning `window_pairs` from the split would remove the duplication (touches the two
      destructuring sites in `ml_model_training.py` and two 8-tuple mocks in
      `tests/test_ml_model_training_additional.py`).
      Done in `0.30.1`: the split returns a `TrainCalibrationSplit` carrying `window_pairs`,
      pinned by characterization tests written before the change.
- [ ] `_attach_qb_features` requests a not-yet-published current season a second time before
      kickoff (the first `load_pbp` skipped it, so it is "missing"), doubling that network attempt
      and its warning; and every ETL run, however narrow its `--min-season`, needs all
      `1999..max_season` play-by-play seasons cached or downloadable.
      Status 2026-10-16: the double request is fixed in `0.54.0` (`requested_seasons`,
      `.agents/sos_port/decisions/integration_pbp_elo.md` entry 1.6); the full-history need is not
      re-checked. Feeds production (21 quarterback-family features).
- [ ] `qb_stats`: scrambles (dropbacks without a passer id) go to the team-game's plurality
      passer, so a starter who leaves early has his scrambles booked to the backup; the recent
      window counts any week with one dropback as a full game; `completions` and the recent
      `cpoe` sums are computed but never read. Reuse: the play-by-play helpers and
      `calculate_stat_differentials` were taken along with task 53.6 (`0.9.0`; the lenses are gone
      since `0.10.0`, the reuse stays); `_ratio` is still
      a third copy of the null-safe ratio in `teamrankings._safe_ratio` and `strength_snapshot`
      (consolidating it touches three modules, so it was left out of that task).
      Status 2026-10-16: scrambles go to the rusher since `0.54.0` (entry 1.4) and the unread
      completion and CPOE sums are gone (entry 1.5); left: the recent window's game count (not
      re-checked) and the `_ratio` copy, still in `utils/polars/qb_stats.py`. Feeds production.
- [ ] The QB identity chain (Elo name to GSIS id via a copy of the nfeloqb metadata, aliases and
      an `F.Last` fallback) could key completed games on the nflverse schedule's
      `away_qb_id` / `home_qb_id` directly if those columns joined `NFLREADPY_SCHEDULE_COLUMNS`;
      name resolution would then be needed only for future-week rows.
      Status 2026-10-16: open; the schedule's starter ids are still not downloaded, and the chain
      feeds production (the quarterback family and the `qb_elos.csv` Elo, 39 features).

### From task 56.4 (rolling calibration window)

- [x] Closed 2026-09-20: superseded by `0.12.3`, which removed in-season early stopping from
      production and walk-forward. No head stops on the calibration frame any more, so the
      defect this item described cannot recur. The historical evidence, kept for the record:
      the Week-2 smoke run (`models/smoke_20260911`, market-anchored) stopped its margin head at
      iteration `1` on a 50-game window; the workaround run calibrated on the 2025 and 2026
      seasons (274 games) stopped at `189` (`models/smoke_20260911_workaround/`), and
      `models/weekly_2025_week_22` (8 weeks) at `9`. Every head now runs the full tree budget,
      and the only question left is how large that budget should be, which is task 55.7.
