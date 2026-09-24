# ARCHIVE - Completed Milestones

This file contains completed milestones and optional enhancements that were previously tracked in
`TODO.md`. Keep this as the audit trail. If future changes regress behavior, re-run the acceptance
checks from the relevant section.

Archived milestone numbers never change. The active worklist in `TODO.md` was renumbered once, on
2026-09-10, so that its milestones run in execution order; the map is below.

---

## Worklist renumbering (2026-09-10)

By 2026-09-10 the active milestones in `TODO.md` ran 50, 43 (phase 2), 47, 48, 39, 41, 42 in
execution order, because milestones had been reordered without renumbering. The finished parts
were archived (Milestone 43 phase 1 below; the resolved follow-ups listed after the map), and the
remaining milestones were renumbered from 51 in execution order. Numbers 39-43, 47, 48 and 50 are
retired for active work: an older document, commit or changelog entry that names one of them means
the old milestone, and the map gives its new home. Subtasks keep their order (old 43.2 is 51.1, old
50.1 is 52.1, and so on).

| old | new | milestone |
| --- | --- | --- |
| 43 phase 2 | 51 | Power rankings on the adjusted composite |
| 50 | 52 | The total (over/under) head carries almost no signal |
| 47 | 53 | QB per-dropback EPA families for the expected starter |
| 48 | 54 | PBP situational stats replace the TeamRankings stat scrape |
| 39 (with 40) | 55 | Off-season configuration sweep + lock default settings |
| 41 | 56 | Weekly orchestration residuals |
| 42 | 57 | Ensembles and alternative models (parked) |

Follow-ups resolved after their milestones closed:

- Milestone 45: the old reference (Brier `0.2312`, log loss `0.7352`, pick accuracy `0.6833`,
  margin MAE `9.8954`) is not reproducible on this machine; the default config on the untouched
  pre-change dataset gives Brier `0.2300`, log loss `0.7501`, pick accuracy `0.6833`, margin MAE
  `9.9705`. `--xgb-tree-method hist` and `rushing_epa` were ruled out by controls.
- Milestone 45: playoff-branch and Week-1-fallback leakage perturbation tests now exist for the
  play-by-play and schedule-adjusted strength families, each mutation-verified.
- Milestone 45: the on/off walk-forward arms were re-run on 2026-09-09 and reproduced exactly;
  reports live in `models/review_wf_2023_2025_pbp_{off,on}/`.
- Milestone 46: `uv sync --check --active` failed after the `0.4.0` bump because the environment
  still had `0.3.0` installed; a plain `uv sync` cleared it. Re-sync after every version bump.

---

## Milestone 54 - PBP-first team-game skeleton and situational stats

Formerly Milestone 48, widened on 2026-09-18 by the user's decision after the audit found that
nflverse team stats lack Jacksonville's 2001-2002 home games while play-by-play has all 16 (see
Milestone 59's findings). Goal: play-by-play becomes the primary per-team-game source; nflverse
team stats fill only what play-by-play cannot derive. Task 54.0 completed 2026-09-21 (version
`0.14.0`). Tasks 54.1-54.4 completed 2026-09-21 (versions `0.15.0`-`0.15.1`). The default flip
to `pbp` (below) completed 2026-09-21 (versions `0.16.0`-`0.16.1`), closing the milestone with no
remaining item: both `--team-stats-source` and `--tr-stats-source` now default to `pbp`;
`nflverse`/`scrape` remain selectable explicitly, and the TeamRankings ratings scrape (not the
situational-stat columns) still runs either way, so it was not retired. `feat/m54-0-landing`
merged into `main` with no conflicts and was pushed on 2026-09-22 (merge commit `295d4c4`,
version `0.16.2`).
Outcome: completed regular-season team-game rows now come from the schedule skeleton, the
collapsed 2001-2002 Jacksonville home rows have their box-score columns nulled, the eight
situational percentages and the per-team-game box score can now be derived from play-by-play
behind `--tr-stats-source pbp` and `--team-stats-source pbp` (both default to the existing
source, so a weekly run is unchanged unless a build opts in), the ETL rebuild passes leakage
checks throughout, and every reviewed three-season arm ties the accepted `200`-tree reference.

### What landed

- `nfl_predictor/utils/polars/loaders.py`: `build_team_game_skeleton`, which projects every
  completed regular-season schedule row to its two team-game rows before any nflverse team stats or
  play-by-play counts are attached; `_log_team_stats_coverage`, which warns for every
  `(season, team)` whose team-stat row count differs from the schedule; `_repair_collapsed_box_scores`,
  which nulls the box-score columns of one-row games where the surviving nflverse row clearly
  covers both teams; and `attach_team_stats_to_schedule`, which applies the repair after the join.
- `nfl_predictor/data_collection.py` now builds the per-team-game frame from the schedule skeleton
  instead of from the nflverse team-stats rows, so missing team-stat rows no longer erase the game
  from schedule-driven counts.
- `nfl_predictor/constants.py`: `TEAM_GAME_NON_BOX_SCORE_COLUMNS`, used to preserve identity,
  schedule-derived scoring columns and play-by-play counts while nulling only the collapsed
  box-score fields.
- Tests: `tests/test_data_collection_helpers.py` and `tests/test_polars_loaders.py` cover the
  schedule skeleton, the coverage-gap warnings and the collapsed-box-score repair.

### Verification

- Branch baseline: `feat/m54-0-landing`, version `0.14.0`, off `main` at `d6795ca`.
- The approved backup of the previous top-level CSVs is `data/backup_pre_m54_0/`.
- Rebuild log: `models/etl_m54_0_rebuild/etl.log`. It logs the expected 9 coverage-gap warnings
  (`1999` null team, `1999` BAL/LAR, `2000` BUF/KC/LAC/MIA, `2001-2002` JAX) and the expected 16
  repair warnings for the collapsed Jacksonville home rows.
- Rebuilt top-level dataset: `data/completed_games_ml.csv` (`7292` completed rows, `513` columns,
  fingerprint `db8b8ff4...`). The through-2025 cut is
  `data/completed_games_ml.m54_0_through_2025.csv` (`7261` rows, `513` columns, fingerprint
  `e914eadf...`), cut by `models/etl_m54_0_rebuild/cut_through_2025.py` with a verbatim-line
  check against the source file.
- Leakage audit: `models/audit_m54_0_rebuild/leakage_audit.json` (`463` features, `0` flags).
- Acceptance fact: `data/strength_snapshots.csv` now shows Jacksonville at
  `strength_games_played = 16.0` at season end in 2001 and 2002.

### Walk-forward (54.0)

Run directory: `models/wf_m54_0_2023_2025_from_week1/`

- Hypothesis and decision rule: `HYPOTHESIS.md`
- Checkpoints: `models/wf_checkpoints/d112ebcba3115bafe9d9/`
- Completion: `Walk-forward fold 54/54 done` at `2026-09-21 14:18:24.011`, `1301s elapsed`,
  followed by `wf exit 0` in `run.log`
- Reference slice: `models/wf_checkpoints/a5e76d54187e27ca7370_2023_2025/`
- Independent review: `models/wf_m54_0_2023_2025_from_week1/REVIEW.md`

Governing weeks 3-18 result, candidate minus reference (`720` games): deterministic Brier
`0.2097` vs `0.2090`, diff `+0.0007` with 95% interval `[-0.0011, +0.0024]`; margin MAE
`9.9166` vs `9.9044`, diff `+0.0122` with 95% interval `[-0.0566, +0.0801]`. By the written
rule, a no-breakage tie. All `816` predicted margins moved (max `4.44` points). A broader-than-
first-hypothesized ETL change also appeared: `836` of `855` scored 2023-2025 rows moved at least
one feature, especially in the `sos_*` and `opponent_*` EPA families, but the review kept the
decision performance-based and still read the run as a tie.

### What landed (54.1-54.4)

- `nfl_predictor/utils/polars/pbp.py`: `red_zone_tds` now requires `td_team == posteam`, so a
  defensive score on a red-zone play no longer credits the offense. New drive-level counts
  `red_zone_trips`/`red_zone_td_drives` (and their `_allowed` mirrors) come from `fixed_drive`
  and `fixed_drive_result`, so the red-zone conversion rate divides touchdown drives by
  red-zone trips rather than touchdowns by red-zone snaps. `aggregate_pbp_team_box_score_stats`
  derives the nflreadpy-named box-score columns (passing, rushing, fumbles, penalties, first
  downs, sacks, interceptions, total yards) from raw plays.
- `nfl_predictor/utils/polars/teamrankings.py`: the eight situational percentages
  (`third_down_pct`, `fourth_down_pct`, `red_zone_td_pct`, `two_point_conversion_pct`, and their
  `opponent_` mirrors) can now be derived from the play-by-play counts on the scraped columns'
  0-100 scale, kept in a `_PBP_PERCENT_RATE_SPECS` table separate from the existing 0-1 rate
  families.
- `nfl_predictor/data_collection.py`: `--team-stats-source {nflverse,pbp}` overlays the
  play-by-play box score onto the nflverse rows, preferring play-by-play where it has a value;
  `--tr-stats-source {scrape,pbp}` switches only the eight situational columns, leaving the
  TeamRankings ratings scrape unchanged either way. Both default to the prior behavior.
- `nfl_predictor/utils/polars/loaders.py`: `load_pbp` now normalizes `td_team` and
  `penalty_team` alongside the other team columns, closing a gap where legacy aliases invented
  team-game rows the schedule had no place for.
- Fixed the box-score aggregation's `total_yards`: nflverse defines it as
  `pass_yards + rush_yards - yards_lost_from_sacks` and stores the sack term as a negative
  number, so the sack yardage is added back, not deducted. The derivation had the sign inverted
  (verified against `13418` of `13418` nflverse team-games of 2000-2025).
- Also fixed a self-join in the box-score aggregation that would have produced `_right`-suffixed
  duplicate columns whenever the offense side produced no rows.
- Tests: `tests/test_polars_pbp.py`, `tests/test_polars_utils.py`,
  `tests/test_teamrankings_helpers.py`, `tests/test_data_collection_helpers.py`,
  `tests/test_polars_loaders.py`, `tests/test_constants.py`.

### Verification (54.1-54.4)

- Candidate build: `data_m54_candidate/completed_games_ml.csv` (`7292` rows, `513` columns) with
  `--team-stats-source pbp --tr-stats-source pbp` on the 54.0 ETL code; the through-2025 cut is
  `data/completed_games_ml.m54_12_through_2025.csv` (`7261` rows, fingerprint `e0b68a0e...`).
  Leakage audit `models/audit_m54_12_candidate/leakage_audit.json`: `463` features, `0` flags,
  matching 54.0.
- Source comparison (task 54.2): `models/pbp_vs_nflverse_m54_2/COMPARISON.md`. Over the `13912`
  team-games of `1999-2025` where both sources exist, seventeen of twenty-one derivable columns
  agree on `94%` or more with a median difference of zero after the `total_yards` fix (it went
  from `14.13%` to `96.62%`). Four columns were recorded as open exceptions at this point:
  `passing_epa` (`69.54%`, no simple definition variant closes it), `fumbles` (`73.63%`, the
  nflverse flag does not attribute which team fumbled), `2pt_conversions` (`94.80%`, disagreeing
  with nflverse's own team-stats table) and `pass_attempts` (`86.70%`, play-type edge cases).
  `passing_epa`, `fumbles` and `pass_attempts` were fixed the same day; `2pt_conversions` was
  found not to need a fix (a confirmed nflverse team-stats bug, not a play-by-play defect) — see
  the "Default flip to `pbp`" subsection below for both.
- Sanity comparison (task 54.4): the same file's second section compares the published,
  season-to-date, prior-blended situational columns for `4363` completed games of `2010-2025`.
  `third_down_pct` agrees closely (median diff `~1.3`); `fourth_down_pct` and `red_zone_td_pct`
  track within a few points in aggregate; `two_point_conversion_pct` does not track the scrape
  well (pbp mean `~47%` against scrape `~32%`), because rare two-point attempts amplify a
  per-game count difference into a large rate swing once blended over a season. Given the
  `2pt_conversions` finding below, this does not establish which side is closer to the truth for
  the rate; it only shows the pbp-derived count itself is correct where it disagrees with
  nflverse's team-stats table.

### Walk-forward (54.1-54.4)

Run directory: `models/wf_m54_12_2023_2025_from_week1/`

- Hypothesis and decision rule: `HYPOTHESIS.md`
- Checkpoints: `models/wf_checkpoints/4e729a9c5751ba978a71/`
- Completion: `Walk-forward fold 54/54 done` at `2026-09-21 18:59:41.706`, `1227s elapsed`,
  followed by `wf exit 0` in `run.log`
- Reference: 54.0's own reference arm `models/wf_m54_0_2023_2025_from_week1/`
  (checkpoints `models/wf_checkpoints/d112ebcba3115bafe9d9/`)
- Independent review: `models/wf_m54_12_2023_2025_from_week1/REVIEW.md`

Governing weeks 3-18 result, candidate (`--team-stats-source pbp --tr-stats-source pbp`) minus
reference (`720` games): deterministic Brier `0.2096` vs `0.2097`, diff `-0.0001` with 95%
interval `[-0.0017, +0.0016]`; margin MAE `9.9090` vs `9.9166`, diff `-0.0075` with 95% interval
`[-0.0760, +0.0611]`; pick accuracy identical at `0.6833` both arms. By the written rule, a
no-breakage tie: the play-by-play source is a safe substitute for the box score and situational
percentages over the seasons where both sources exist, but this tie is not itself grounds to
flip either flag to the default. That remains a separate must-ask decision, argued for by the
`1999-2002` coverage gain (the TeamRankings scrape starts in 2003, leaving `1029` completed
games with null situational percentages; play-by-play fills `1026` of them) and weighed against
the four open exceptions above.

An earlier launch of this arm on an uncorrected build (`total_yards` sign inverted) was stopped
after 6 of 54 folds, before any result was read, once the source comparison above exposed the
defect; its checkpoints were discarded and are not part of this record.

### Default flip to `pbp` (versions `0.16.0`-`0.16.1`, 2026-09-21)

The user reviewed the four open exceptions from the 54.1-54.4 comparison and asked for each to
be examined and corrected, then for both source flags to be flipped to `pbp` as the default once
verified. All four were fixed:

- `passing_epa` now sums `qb_epa` (nflverse's own quarterback-attribution EPA column) instead of
  `epa`, over every `pass_attempt` play including sacks and two-point tries: match rate against
  nflverse rose from `69.54%` to `99.33%` on the full 1999-2025 rebuild (`98.84%` on the
  four-season sample checked before landing, `100%` on 2024 alone).
- `pass_attempts`, `pass_completions`, `pass_yards`, `pass_touchdowns`, `interceptions_thrown`,
  `rush_attempts`, `rush_yards` and `rush_touchdowns` now use nflverse's own canonical
  `pass_attempt`/`rush_attempt` raw flags (added to `constants.PBP_COLUMNS`) instead of
  `play_type`-based conditions. A sack carries `pass_attempt = 1` in the raw data, so nflverse's
  own `pass_attempts` excludes it explicitly; a kneel carries `rush_attempt = 1` despite
  `rush = 0`. `pass_attempts` moved from `86.70%` to `99.87%`; every column in this group now
  matches on `99.68%` or more.
- `rushing_epa` now includes two-point tries: `95.54%` to `99.87%`.
- `fumbles`/`fumbles_lost` now exclude special-teams plays, matching nflverse's offense-only
  fumble stat: `73.63%`/`90.35%` to `91.95%`/`98.37%`. A residual gap on aborted-snap fumbles is
  documented rather than chased further; nflverse's own player-level fumble categories
  (`load_player_stats`'s `sack_fumbles`+`rushing_fumbles`+`receiving_fumbles`) do not cleanly
  attribute those either.
- `2pt_conversions` needed no code change: the match rate (`94.80%` on the earlier partial
  rebuild, `94.35%` on a four-season sample, `94.80%` again on the final full rebuild) was left
  where it stood because the derivation was already correct. The user manually verified one
  mismatch against the actual game (2024 week 17, Green Bay at Minnesota: exactly one two-point
  conversion happened, matching play-by-play exactly) and asked for the rest to be checked. That
  investigation, `models/pbp_vs_nflverse_m54_2/verify_2pt_doubling.py`, found a confirmed,
  systematic nflverse team-stats bug across seven sampled seasons (`2010`, `2015`, `2020`,
  `2022`, `2023`, `2024`, `2025`; `3710` team-games): of `246` mismatches, `234` (`95.1%`) show
  nflverse's count at exactly double the play-by-play count, and zero mismatches go the other
  way (nflverse reporting a nonzero count where play-by-play has none). The play-by-play value is
  the correct one wherever it disagrees with nflverse. Its derived `two_point_conversion_pct`
  still does not track the TeamRankings scrape well once blended (pbp mean `~47%` against scrape
  `~32%` over 2010-2025), but with the doubling bug confirmed on the nflverse side, this
  comparison against a different, unverified third-party source (TeamRankings, not nflverse) no
  longer says which side is closer to the truth for the rate.

Full numbers, formulas and the trace of each residual disagreement:
`models/pbp_vs_nflverse_m54_2/COMPARISON.md`.

With those four fixed, `--team-stats-source` and `--tr-stats-source` were flipped to `pbp` as
the default in `0.16.0`, including the production fast path `scripts/weekly_run.py` uses when it
calls `data_collection.main()` with no arguments (a separate hardcoded default that the CLI
argparse default alone would not have changed). `nflverse`/`scrape` remain selectable
explicitly. Four end-to-end tests' minimal `team_stats_df` fixtures needed `opponent_abbr` added,
since the nflverse default never exercised the PBP overlay's join but the new default does.

The full ETL rebuild that followed (`0.16.1`) needed `--refresh-nflreadpy` to pick up the two
new raw play-by-play columns (`pass_attempt`, `rush_attempt`); the previous top-level CSVs are
backed up in `data/backup_pre_m54_flip/` and the pre-refresh play-by-play cache in
`data/cache/nflreadpy/backup_pre_m54_flip/`. Leakage audit
`models/audit_m54_flip_rebuild/leakage_audit.json` passed (`463` features, `0` flags, same shape
as every prior 54.x build). The rebuild closed the 1999-2002 Jacksonville team-stats coverage
gap that task 54.0's schedule skeleton and box-score repair were built around, as anticipated
when Milestone 54 was widened on 2026-09-18 specifically because play-by-play has both sides of
every JAX game where nflverse's team-stats table does not: the now-default `pbp` overlay fills
those rows before the schedule-skeleton coverage check runs. The ETL logged `0` repair warnings
on this rebuild (down from `16` on the 54.0 rebuild) and `7` coverage-gap warnings, all
pre-existing single-game gaps unrelated to JAX
(`1999` null team, `1999` BAL/LAR, `2000` BUF/KC/LAC/MIA); `strength_games_played` for JAX still
reads `16.0` at both 2001 and 2002 season end, confirming the acceptance criterion still holds.
The schedule-skeleton and repair code remain in place as a safety net for the `nflverse`/
`scrape` configuration.

Run directory: `models/wf_m54_flip_2023_2025_from_week1/` (checkpoints
`models/wf_checkpoints/9779c1cbb0701d23661a/`, hypothesis, `compare_output.txt` and independent
review in the run directory), against the 54.0 pre-flip reference
(`models/wf_checkpoints/d112ebcba3115bafe9d9/`). Governing weeks 3-18 result, candidate minus
reference (`720` games): deterministic Brier `0.2106` vs `0.2097`, diff `+0.0009` with 95%
interval `[-0.0008, +0.0026]`; margin MAE `9.9321` vs `9.9166`, diff `+0.0156` with 95% interval
`[-0.0529, +0.0812]`. By the written rule, a no-breakage tie: the interval covers zero and the
point difference is within the fit-noise floor on both metrics. Every predicted margin moved by
up to `3.144` points. This is a no-breakage confirmation, not a lift claim.

---

## Milestone 59 - Benchmark instrument and feature audit follow-ups

Closed 2026-09-19 across versions `0.12.1` to `0.12.4` by two implementation sessions, then
audited and corrected the same day in `0.12.5` (see "Audit" at the end). The task text and
findings that opened it on 2026-09-18 are in the git history of `TODO.md` (commit `ebdd8a5`).
Outcome: the walk-forward instrument is the deterministic-plus-market view with paired intervals;
`auto` is the deterministic floor; production and walk-forward fit through the same helpers with
no in-season early stopping and `best_iteration` recorded per head; 1999-2001 division context
uses the pre-2002 alignment; sack mirrors are excluded at the ETL source; the noise-family
follow-up found no lift. Two parts were narrowed against the task text and are reopened as
follow-ups ("Narrowed" below), not closed.

### What landed

- 59.1 Instrument: walk-forward reports, `wf_compare.csv`, the `weekly_run` and
  `betting_pipeline` summaries, and the API column registry carry configured, deterministic
  (`Phi(margin / SCORE_DIFF_STD_DEV)`) and market-implied (no-vig moneyline, spread fallback)
  probability columns, with paired deterministic-minus-market Brier and log-loss differences and
  5000-sample bootstrap intervals for week 1, week 2, weeks 3-18 and all weeks. Candidate ranking
  sorts on the deterministic columns.
- 59.2 Calibration: the fitted calibration frame is the previous two seasons plus the completed
  weeks of the eval season (`select_calibration_data`; `_pooled_calibration_frame` in
  production); `sigma` (one residual standard deviation) landed as a method, Platt picks `C`
  from a grid on the latest pre-eval season, undersized isotonic falls back to sigma, and `auto`
  resolves to the deterministic floor because no fitted path beat it (table below).
- 59.3 Fit parity: in-season early stopping removed from production and walk-forward; both call
  the same shared fit helpers and run the full `598`-tree budget. `metadata.json` and the
  walk-forward per-week rows record `best_iteration` per head (since `0.12.5` also
  `early_stopped`, and `at_cap` only when early stopping ran and never fired).
- 59.4 Divisions: `PRE_2002_TEAM_TO_DIVISION` / `_CONFERENCE` and `division_map_for_season` /
  `conference_map_for_season` in `constants`; the divisional flag, record splits, lookahead
  context and standings proxies select the map by season. Verified on the real dataset: the
  recomputed flag agrees with nflverse `division` in every season (the on-disk column disagreed
  on 61, 67 and 61 rows in 1999, 2000 and 2001).
- 59.5 Noise-family ablation: the `rare_events` feature group and `--min-child-weight` /
  `--gamma` on the walk-forward CLI; three six-season arms (table below), all ties.
- 59.6 `def_sacks` and `times_sacked` joined `EXCLUDE_FROM_OPPONENT_STATS`.

### Narrowed (reopened as follow-ups in `TODO.md`, "From Milestone 59", and task 55.7)

- 59.2 asked for a calibrator fit on pooled **out-of-fold** predictions. The landed pool is the
  right rows but in-sample: the model that predicts them was trained on them. Nothing shipped on
  it (`auto` is the floor, production defaults to `elo`), but the acceptance "make `auto`
  resolve to this path" is met only in the degenerate sense that the path is the floor.
- 59.3 asked to fix `n_estimators` from a season-sized time-aware tuning or to early-stop on
  such a set. Neither was done; early stopping was removed and the old `598` cap kept, so every
  head "hits the cap" by construction. Task 55.7.
- The dataset on disk predates 59.4 and 59.6; the ETL rebuild is pending and is a must-ask item.

### Measurements

All arms: `data/completed_games_ml.m49_through_2025.deadweight_cut.csv`, from week 1,
`market_anchor` on, checkpoints under `models/wf_checkpoints/<id>/`. Five three-season arms
(`--eval-last-n-seasons 3`, 54 folds, about 2h20m each under `OMP_WAIT_POLICY=PASSIVE`, run in
sequence 2026-09-18 18:57 to 2026-09-19 07:09) differ only in what `auto` resolved to at the
time; their predicted margins and therefore their deterministic and market columns are identical.

| run directory | `auto` resolved to | checkpoints | configured weeks 3-18 Brier / log loss |
| --- | --- | --- | --- |
| `wf_m59_2023_2025_from_week1/` | isotonic on the pooled frame | `303d85338c828f278a0e` | `0.2378` / `2.0158` |
| `wf_m59_2023_2025_from_week1_auto_sigma/` | sigma (residual RMS) | `3863bd37678f4b9a4dae` | `0.2119` / `0.6157` |
| `wf_m59_2023_2025_from_week1_auto_sigma_centered/` | sigma (residual std) | `27f1cd2a30b2f1a91a9a` | `0.2119` / `0.6157` |
| `wf_m59_2023_2025_from_week1_auto_floor/` | sigma with a floor fallback | `f6ff076066674127b163` | `0.2119` / `0.6157` |
| `wf_m59_2023_2025_from_week1_auto_selected/` | validation-selected among none/sigma/platt/isotonic | `a1e047a11237c39d013d` | `0.2347` / `0.9893` |

The deterministic columns of every arm, and the floor `auto` now ships, weeks 3-18: Brier
`0.2099`, log loss `0.6073`, pick accuracy `0.6861`; market `0.2086` / `0.6042` / `0.6861`;
paired deterministic-minus-market Brier `+0.0012` `[-0.0030, +0.0054]`. The full four-window
table is the benchmark in `AGENTS.md`. Against the `0.12.0` arm (`bae0e56db951d1a890d4`, fit with
early stopping), 247 of 816 predicted margins move by at most `0.34` points; its rescore gives
weeks 3-18 `0.2098` / `0.6072` and margin MAE `9.9600` against `9.9608`: a tie.

Six-season arms (`--eval-last-n-seasons 6`, 107 folds, about 1h40m each on an idle machine,
run in sequence 2026-09-19 10:27 to 15:17), `calibration=auto` (the floor):

| run directory | checkpoints | deterministic Brier | market Brier | paired 95% CI | margin MAE |
| --- | --- | --- | --- | --- | --- |
| `wf_m59_2020_2025_auto_floor_baseline/` | `c641473671fc9f9b2f5f` | `0.2116` | `0.2104` | `[-0.0016, +0.0041]` | `9.9015` |
| `wf_m59_2020_2025_rare_events_off/` | `3f38eaafc58d78b0542b` | `0.2123` | `0.2104` | `[-0.0010, +0.0049]` | `9.9101` |
| `wf_m59_2020_2025_regularized_gamma5_mcw5/` | `8cc0bf9724c5bab18002` | `0.2114` | `0.2104` | `[-0.0018, +0.0039]` | `9.8903` |

Every interval covers zero. The baseline's 2023-2025 folds are bit-identical to the three-season
arms (max predicted-margin difference `0.0`), so they add no information about 2023-2025 and
serve as the reproduction check; its 2020-2022 folds are new. Restored from its checkpoints:
configured probabilities equal the deterministic floor exactly, none leaves `[0.02, 0.98]` when
`|predicted_margin| <= 14`, and every head in all 107 folds ran the full `598`-tree budget.

Reproduction: load `fold_*.joblib` from the checkpoint directory, concatenate `predictions`,
add `deterministic_home_win_prob = ml_model._margin_to_home_win_prob(predicted_margin)` and
`market_home_win_prob = walk_forward._resolve_market_home_win_prob(frame)`, then score with
`walk_forward._probability_window_rows(frame, seed=0)`.

### Rebuild (2026-09-20, versions `0.12.6` and `0.12.7`)

The ETL rebuild for 59.4 and 59.6 ran twice. The first pass, on the `0.12.5` code, published
`away_/home_opponent_points_per_play`, `away_/home_points_per_play_margin` and their diffs as
all-null columns: 59.6 had excluded `times_sacked` from the opponent mirror, and
`opponent_points_per_play` divides `points_allowed` by the opponent's pass attempts, rush
attempts and times sacked (`models/etl_m59_rebuild/etl_defective_first_pass.log`). `0.12.6`
(`2d46083`) builds the mirrors named in `constants.OPPONENT_MIRROR_INTERMEDIATES`
(`times_sacked` only) per game and drops them at schema selection, with tests on the mirror
helper, the formula and the constant. The second pass (`models/etl_m59_rebuild/etl.log`) gives
`data/completed_games_ml.csv` `db6a78a3...` (`7278` rows, `513` columns): against the backup in
`data/backup_pre_m59_rebuild/` (`8bacad41...`) only the 27 division and conference derived
columns move, all in 1999-2001 rows, plus one 2026 `stadium_surface` from the current-season
refresh; the six sack mirrors are gone. Leakage audit `models/audit_m59_rebuild/`: `463`
features, `0` flags.

Tie check: `models/wf_m59_rebuild_2023_2025_from_week1/` (benchmark config from week 1 on the
through-2025 cut `cf42ec55...`; checkpoints `34c17e508ab015a80662`; hypothesis and decision rule
in `HYPOTHESIS.md`; independent rescore in `REVIEW.md`). Weeks 3-18 deterministic Brier
`0.2096` against the benchmark's `0.2099`, paired `-0.0003` `[-0.0028, +0.0023]`; margin MAE
`9.9722` against `9.9608`, `+0.0113` `[-0.0890, +0.1142]`: a tie by the rule. All 816 margins
moved (max `5.16`), and weeks-3-18 pick accuracy fell `0.6861` to `0.6764` (7 games). The full
four-window table is the reference arm in `AGENTS.md`; new arms on the rebuilt build compare
against it.

Noise control (`0.12.8`, 2026-09-20): the reference arm rerun with `--random-seed 7`
(`models/wf_m59_rebuild_2023_2025_from_week1_seed7/`, checkpoints `89dc69c3f18d74ad205b`,
rescored in its `REVIEW.md`) moves every margin (median `0.91` points), flips 46 picks and lands
at weeks-3-18 Brier `0.2116` / pick accuracy `0.6847`, so the rebuild arm's 7-game drop is fit
noise; the floor is recorded beside the reference arm in `AGENTS.md`.

### Audit (2026-09-19, version `0.12.5`)

A review session audited both implementation sessions against the tree and the artifacts. What
held: every run retrained (no stale-fold resume), the division fix, the acceptance checks above,
840 tests passing. What did not, and what `0.12.5` did about it:

- `AGENTS.md` said the benchmark table was rescored from the `0.12.0` checkpoints; it came from
  the retrained arms (fourth-decimal difference, above). Corrected.
- `ruff format --check` failed on `constants.py`; the second session reported a green gate
  without running it, and CI would have failed. Fixed, and `scripts/gate.sh` now runs the whole
  CI gate as one command.
- 59.2 and 59.3 were checked off after being narrowed (above). Reopened as follow-ups.
- Every fold logged an `at_cap` warning that could not be false. The warning now requires early
  stopping to have run.
- `_select_auto_calibration_method` and `_sigma_calibrator_improves_on_floor` were dead after
  the second session removed their call sites. Deleted. `sigma` was reachable only as the
  isotonic fallback; it is now a CLI choice.
- The first session's closing comparison script crashed on a typo, so its "tie with `0.12.0`"
  claim was unverified when made; the rescore above confirms it. Its closing table labelled the
  deterministic columns as the model's result for the `auto_selected` arm without saying that
  arm's configured `auto` scored `0.2337` / `0.9523` over all weeks.
- The "Delegation guardrails" section in `AGENTS.md` records the rules that follow from this.

---

## Milestone 55 (partial) - Off-season configuration sweep + lock default settings

Formerly Milestone 39, with former Milestone 40 folded in. Task 55.7 completed 2026-09-21
(versions `0.13.0`-`0.13.1`). Task 55.8 was closed 2026-09-23 as `0.17.0`, reopened the same day,
and closed again 2026-09-24 as `0.18.0` (see its section below). Tasks 55.1 and 55.2 were retired
2026-09-24. Tasks 55.3-55.6 and 55.9 stay in `TODO.md`.

### 55.1 and 55.2 - Configuration-sweep runner (retired 2026-09-24)

Retired by the user's decision on 2026-09-24 and never started. The tasks asked for a sweep
config schema (55.1) and a runner that ran a walk-forward per configuration and wrote
`sweep_summary.csv` and `best_config.json` (55.2). Why retired: a runner that tries many settings
and keeps the lowest score is a selection-bias machine. It is the failure the 2026-09-23 review
found in the weekly run's stage 1 (task 56.5), and it conflicts with `AGENTS.md` rule 4 (a
hypothesis and a decision rule written before each run) and rule 13 (two seeds for any default
change). What replaces it: hypothesis-driven ladders for single settings (task 55.3 for `K`,
for example), the task 55.9 Optuna tune for the XGBoost hyperparameters, and task 56.3 to wire the
chosen settings into the weekly run and the benchmark from one source.

### 55.7 - Choose `n_estimators` time-aware (versions `0.13.0`-`0.13.1`)

In-season fits had run the full `598`-tree budget since `0.12.3` with no early stopping
anywhere; `598` was the old Optuna value, never a measured choice (task 59.3's narrowed
follow-up). Method: a ladder of budgets as separate six-season arms of the reference
configuration on the rebuilt build, one hypothesis per arm, read on the deterministic columns
against the fit-noise floor; the winner becomes the shared default in walk-forward and
production together (a default change: must-ask).

Three six-season rungs ran 2026-09-20/21 and were independently rescored:
`models/wf_m55_7_2020_2025_trees598/` (the reference, which reproduces the three-season
reference arm's 2023-2025 folds bit for bit), `models/wf_m55_7_2020_2025_trees200/` and
`models/wf_m55_7_2020_2025_trees400/`. The aggregate order is monotone toward fewer trees and
small: weeks 3-18 deterministic Brier `0.2103` / `0.2111` / `0.2117` and margin MAE `9.9281` /
`9.9978` / `10.0240` for `200` / `400` / `598`, with the `200 - 598` Brier interval covering
zero by `+0.0000791` and its margin MAE difference `-0.0958` `[-0.1583, -0.0345]` beyond the
fit-noise floor. The ladder stopped on its own "report all three and ask" branch; the table and
the pairwise intervals are in `AGENTS.md` under "Tree-budget ladder".

Decided 2026-09-21 by the user: adopt `200` as the shared default `n_estimators`, in
walk-forward and production together, conditional on a `100`-tree plateau check (below). In the
same chunk, `config/weekly_run.yaml` was made consistent: it set `tune: true`, which re-ran a
one-hour Optuna study on every weekly run and was not intended, so it is now `tune: false`; and
its walk-forward stage ran `wf_n_estimators: 200`, `wf_max_depth: 4`, `wf_learning_rate: 0.03`
while the final fit used `DEFAULT_XGB_PARAMS` (depth `5`, learning rate `0.0165`, `598` trees at
the time), so the walk-forward stage's XGBoost params were aligned with the production defaults
so both stages fit the same model. Landed 2026-09-21 on `feat/m55-7-default-200` (`0.13.0`, gate
green): `DEFAULT_XGB_PARAMS` now carries `n_estimators = 200`, the bare `weekly_run` parser
falls back to the shared production `n_estimators`, `max_depth` and `learning_rate`, Stage 1 no
longer forces `subsample` / `colsample_bytree` to `0.9`, and `config/weekly_run.yaml` is
aligned.

Closed 2026-09-21 by the reviewed plateau check in `models/wf_m55_7_2020_2025_trees100/`
(`0.13.1`): against the governing `200` rung, weeks 3-18, deterministic Brier `0.2103` vs
`0.2103`, diff `+0.0000` `[-0.0008, +0.0009]`; margin MAE `9.9237` vs `9.9281`, diff `-0.0044`
`[-0.0389, +0.0296]`. By the written rule, a tie, so `200` stays the shared default. The
reviewer also rescored the rung against `598` for ladder continuity (`REVIEW.md` in the same
run directory).

Task 56.2 (the postseason default, landed the same day as part of the same chunk) is archived
under "Milestone 56 (partial)" below.

### 55.8 - Season weighting (versions `0.17.0` and `0.18.0`)

Closed 2026-09-24 as `0.18.0` by the user's decision: **production trains unweighted.**
`config/weekly_run.yaml` no longer sets `train_recency_half_life_seasons: 4` (there since the
file's first commit, `83ba2a7`, 2026-01-25, and never measured before this task) or
`wf_recency_half_life_seasons: 4` (added by the `0.17.0` close-out), and
`tests/test_weekly_run.py` pins both stages as unweighted.

#### The redo (2026-09-23/24)

The `0.17.0` close-out (kept below as written) had correct numbers but kept the shipped half-life
`4`, the weakest arm on every probability and error metric, and its reviews were written by the
agent that produced the runs. The user reopened it the same day. The redo added five six-season
arms on the same build, code path and configuration, all approved in advance with their
hypotheses and rules written before launch:

- the seed-7 pair, unweighted and half-life `4`
  (`models/wf_m55_8_2020_2025_unweighted_seed7/`, `..._half_life4_seed7/`; rule in the first's
  `HYPOTHESIS.md`);
- the long-half-life check, half-life `16` at seed 7 and half-life `32` at seeds 42 and 7
  (`..._half_life16_seed7/`, `..._half_life32/`, `..._half_life32_seed7/`; rule in the first's
  `HYPOTHESIS.md`).

The session that launched them wrote `models/wf_m55_8_review/REVIEW.md`; a separate session that
produced none of the nine runs wrote `models/wf_m55_8_review/INDEPENDENT_REVIEW.md` from the fold
checkpoints (`.venv/bin/python models/wf_m55_8_review/independent_rescore.py`). It reproduced
every number in `REVIEW.md` exactly and confirmed provenance: one dataset (`2d4111a6...`), 107
folds per arm all computed fresh, the full `200` trees, `auto` resolved to the deterministic
floor, and only the seed and half-life differing. Its eight disagreements with `REVIEW.md` (all
accepted, answered at the end of that file) were about completeness and wording, not numbers:
the seed-42 ladder's own rule was never applied, the claim that `wf_recency_half_life_seasons`
did not exist was wrong, the rule-9 list left out weeks 1-2 and the week-2 total-MAE signal, and
the probability-path analysis has no second key.

Weeks 3-18, `1423` games, market Brier `0.20948`:

| arm | det Brier s42 / s7 | log loss s42 / s7 | margin MAE s42 / s7 | pool pts s42 / s7 |
| --- | --- | --- | --- | --- |
| unweighted | `0.21073` / `0.20959` | `0.60914` / `0.60665` | `9.9361` / `9.9105` | `8276` / `8346` |
| half-life 4 | `0.21149` / `0.21180` | `0.61096` / `0.61149` | `10.0051` / `10.0121` | `8272` / `8246` |
| half-life 8 | `0.21072` / - | `0.60919` / - | `9.9673` / - | `8304` / - |
| half-life 16 | `0.21016` / `0.20994` | `0.60787` / `0.60757` | `9.9522` / `9.9299` | `8277` / `8317` |
| half-life 32 | `0.21100` / `0.20975` | `0.60945` / `0.60686` | `9.9571` / `9.9304` | `8251` / `8325` |

Two seeds combined per game (candidate minus unweighted, 5000 game resamples, seed 0), weeks 3-18:

| contrast | det Brier | margin MAE | total MAE | pick acc | pool pts |
| --- | --- | --- | --- | --- | --- |
| half-life 4 | `+0.00148` `[-0.00012, +0.00313]` | `+0.0853` `[+0.0188, +0.1535]` | `+0.0744` `[-0.0028, +0.1510]` | `+0.0042` `[-0.0046, +0.0130]` | `-52` `[-114, +10]` |
| half-life 16 | `-0.00011` `[-0.00104, +0.00088]` | `+0.0177` `[-0.0237, +0.0590]` | `+0.0224` `[-0.0203, +0.0651]` | `+0.0032` `[-0.0025, +0.0091]` | `-14` `[-56, +29]` |
| half-life 32 | `+0.00021` `[-0.00061, +0.00102]` | `+0.0204` `[-0.0139, +0.0556]` | `-0.0093` `[-0.0471, +0.0270]` | `+0.0056` `[+0.0000, +0.0116]` | `-23` `[-60, +15]` |

Outcomes under the rules as written:

- Seed-7 pair: branch B, half-life `4` loses (margin MAE worse beyond its interval, Brier not in
  its favor). Over all weeks it is also worse beyond its intervals on Brier `+0.00171`, log loss
  `+0.00381` and margin MAE `+0.0916`. It is last at both seeds on Brier, log loss, margin MAE and
  total MAE, and worse than unweighted in 5 of 6 seasons.
- Long half-lives: neither `16` nor `32` has a Brier interval below zero, so neither is adopted;
  the half-life `32` pick-accuracy lower bound is exactly zero, not above it.
- Seed-42 ladder (`models/wf_m55_8_2020_2025_unweighted/HYPOTHESIS.md`): its first branch fails
  on the point estimate and its "flat, prefer the shipped `4`" fallback rescues the incumbent, the
  rule-12 case; superseded by the two-seed rules.

Qualifications recorded with the decision: unweighted ties mild weighting rather than beating it;
the one consistent signal in weighting's favor is week-2 total MAE (two-seed `-0.2824`,
`-0.2381`, `-0.2044` for half-lives `4`, `16`, `32`, each interval excluding zero; 96 games, on
the `diagnostic_only` total head, gone by weeks 3-18); and half-life `8` has no second seed.

The six-season fit-noise floor, the first ever measured (same setting, seed 7 minus seed 42, weeks
3-18): deterministic Brier moved by up to `0.00125`, pick accuracy by up to `0.0084` and pool
points by up to `74`, with 32-50 picks flipped; intervals excluding zero appeared from re-seeding
alone (unweighted pick accuracy and pool points, half-life `32` Brier and pool points). This is
the evidence behind `AGENTS.md` rule 13 and closes the tree-budget ladder's open question about a
second seed.

#### The `0.17.0` close-out (2026-09-23, superseded; kept as written)

The shipped weekly config had diverged: `train_recency_half_life_seasons: 4` was already live in
`config/weekly_run.yaml`, but no `wf_recency_half_life_seasons` key existed, so a weekly run
trained the production fit with half-life-4 season weighting while its own walk-forward comparison
stage still evaluated candidates unweighted. The older README recency ablation was also not
trustworthy: it had been measured through Platt calibration on a superseded build and read more
like a calibration failure than a model choice.

Method: a reviewed four-arm six-season ladder on the current `pbp`-default build
(`data/completed_games_ml.m54_flip_through_2025.csv`, seasons `2020-2025`, from week `1`,
calibration `auto`, `market_anchor` on, four calibration weeks, seed `42`, shared default
`200`-tree budget). The arms were:

- `models/wf_m55_8_2020_2025_unweighted/`
- `models/wf_m55_8_2020_2025_half_life4/`
- `models/wf_m55_8_2020_2025_half_life8/`
- `models/wf_m55_8_2020_2025_half_life16/`

Each run directory has its own `REVIEW.md`; the unweighted arm also carries the ladder's
`HYPOTHESIS.md`.

Reviewed metrics from disk:

| arm | weeks 3-18 det Brier | weeks 3-18 margin MAE | all-weeks det Brier | all-weeks margin MAE | weeks 3-18 det-minus-market Brier CI |
| --- | --- | --- | --- | --- | --- |
| unweighted | `0.2107307` | `9.9361` | `0.2113442` | `9.8077` | `[-0.0007381, +0.0033154]` |
| half-life 4 | `0.2114931` | `10.0051` | `0.2122308` | `9.8829` | `[-0.0004344, +0.0044273]` |
| half-life 8 | `0.2107159` | `9.9673` | `0.2117030` | `9.8534` | `[-0.0009993, +0.0035351]` |
| half-life 16 | `0.2101646` | `9.9522` | `0.2111091` | `9.8423` | `[-0.0014890, +0.0028600]` |

Paired bootstrap comparisons from `models/wf_m55_7_2020_2025_trees200/compare_to_benchmark.py`
(candidate minus reference, 5000 resamples, seed `0`):

- Half-life `4` minus unweighted, weeks 3-18: deterministic Brier `+0.0007624`
  `[-0.0011145, +0.0026462]`; margin MAE `+0.0691` `[-0.0080, +0.1465]`
- Half-life `8` minus unweighted, weeks 3-18: deterministic Brier `-0.0000148`
  `[-0.0015453, +0.0015254]`; margin MAE `+0.0313` `[-0.0302, +0.0949]`
- Half-life `16` minus unweighted, weeks 3-18: deterministic Brier `-0.0005661`
  `[-0.0018537, +0.0007247]`; margin MAE `+0.0161` `[-0.0381, +0.0692]`
- Half-life `16` minus half-life `4`, weeks 3-18: deterministic Brier `-0.0013284`
  `[-0.0029136, +0.0002453]`; margin MAE `-0.0530` `[-0.1187, +0.0147]`

Outcome: half-life `16` was the best raw Brier arm, but every governing-window interval against
both the unweighted reference and the shipped half-life `4` arm still covered zero, and half-life
`8` tied the unweighted reference outright. By the ladder rule, the season-weighting choice is
flat within noise rather than a clear default-changing win. The shipped production weighting stays
at `train_recency_half_life_seasons: 4`, and `config/weekly_run.yaml` now also sets
`wf_recency_half_life_seasons: 4` so the walk-forward comparison stage finally measures the same
season weighting the final training stage already uses. The README's old Platt-based recency
ablation is replaced with this ladder.

---

## Milestone 58 (partial) - Web UI: FastAPI backend + React frontend

Phases 0-3 completed 2026-09-10 (on `feat/web-ui`, worktree `../nfl-predictor-web`) and merged
into `main` on 2026-09-11 as version `0.8.0`; task 58.4 completed 2026-09-21 (version `0.12.12`);
phases 4-6 (tasks 58.1-58.3) stay in `TODO.md`. The milestone number was assigned at merge time
(the plan had reserved 51, which the 2026-09-10 renumbering gave to the power rankings). The full
design, the decisions made with the user, and the per-phase status with deviations live in
`web_ui_plan.md`.

### What landed

- `nfl_predictor/api/`: FastAPI app factory (`python -m nfl_predictor.api`), argon2 passwords,
  JWT session cookie with a CSRF header, `viewer` / `admin` roles, a login rate limit, a bootstrap
  CLI (`nfl_predictor.api.auth.cli`), SQLite state under `data/web/`, a run index over
  `models/*/metadata.json` with one admin-selected active run, readers plus a column registry for
  predictions, betting (derived from predictions with the workbook formulas, totals
  `actionable=False`), power rankings with week-over-week movement, model metadata, metrics,
  importance and calibration, and data/ETL status; the built SPA is served from `web/dist/`.
- `nfl_predictor/api/jobs/`: a subprocess runner over the repo CLIs with a SQLite job table,
  persisted logs streamed over server-sent events, progress from the `WF candidate N/M` line,
  cancel by process group, one worker for the walk-forward group and a two-slot pool otherwise,
  and 13 templates (`etl_full`, `lines_refresh`, `weekly_run`, `train`, `predict`,
  `predict_week`, `power_rankings`, `betting_xlsx`, `leakage_audit`, `validate_offline`,
  `validate_live`, `walk_forward_backtest`, `shap_analysis`).
- `nfl_predictor/lines_refresh.py` (lines-only refresh that chains a predict job) and
  `nfl_predictor/week_builder.py` (future-week inputs from `all_data_ml.csv`).
- `web/`: Vite, React 19, TypeScript, Tailwind v4, shadcn, TanStack Table and Query, react-router
  and recharts; pages Overview, Predictions, Power Rankings, Betting, Data & ETL, Model, Runs, Users,
  Jobs, Job detail, Glossary and Login. `tests/api/` (backend) and `web/src/**/*.test.ts`
  (frontend) cover it; CI gained a `web` job (Node 26).
- Config: the web server libraries became core dependencies at merge time (`0.8.0`), `web/` is
  excluded from ruff, pyright and ty, and `web/node_modules/`, `web/dist/` and `data/web/` are
  ignored.

### Merge (2026-09-11)

`main` (`0.7.1`, the calibration-window, total-head, quarterback and review-fix commits) merged
into `feat/web-ui` with no conflicts; the branch's Python gate (`814 passed`, `92.88%`) and
frontend gate passed, and a throwaway API instance on port 8766 served runs, predictions, data
status, the job catalog and power rankings from this checkout's `data/` and `models/`. `main` then
fast-forwarded to the branch tip, and the `0.8.0` changelog, README and `AGENTS.md` entries
followed (`814 passed`, `92.90%`). The user's live instance on port 8765 was left running on the
pre-merge API code.

### 58.4 - Housekeeping: the `web` extra and the ETL's upstream data-directory paths

Started as: the `web` extra in `pyproject.toml` duplicated the core dependency list (drop it and
the `--extra web` in CI and `web/README.md`, or give it a purpose); the `etl_full`,
`validate_offline` and `validate_live` job templates read the checkout's own `data/` and ignored
`NFLP_DATA_DIR` (the plan's Phase 2 deviations).

Narrowed: the `web` extra is dropped, and the three CLIs now take `--data-dir` with the
templates passing `NFLP_DATA_DIR`, so every dataset those jobs read or write follows the
configured tree. The ETL's upstream inputs still resolve from `constants.DATA_PATH`: the copied
`qb_elos.csv` and quarterback identity file (`nfl_predictor/utils/polars/loaders.py`,
`data_collection._attach_qb_features`) and the TeamRankings and nflreadpy caches
(`nfl_predictor/utils/scraping_utils.py`, `nfl_predictor/utils/polars/teamrankings.py`).
Threading a data directory through those read paths is a separate change; until it lands,
pointing the API at another checkout's data makes `etl_full` read inputs from this checkout and
write outputs to the configured one. The narrowed part landed in `0.12.12` (`604bcc6`,
`e3b828d`).

Closed 2026-09-21 by the user's decision: the `NFLP_DATA_DIR` setting stays, because the web API
resolves its own paths through it and it lets the app run against a copied data tree, but the
ETL's upstream inputs will not follow it. The `0.12.12` narrowing is the final shape of this
task.

---

## Milestone 53 (partial) - QB per-dropback EPA families for the expected starter

Tasks 53.1-53.5 completed 2026-09-11 (version `0.7.0`); 53.6 (schedule lenses) and 53.7 (the
optional quarterback ridge) stay in `TODO.md`. Outcome: the family works as designed and passes
every leakage check, but its on/off walk-forward is a statistical tie.

### What landed

- `nfl_predictor/utils/polars/qb_stats.py`: `build_qb_identity` / `load_qb_identity` (nfeloqb
  `name_id` to GSIS id through `data/qb_meta_data.csv`, ambiguous names dropped,
  `constants.QB_NAME_ALIASES` for three Elo spellings, a unique `F.Last` passer-name fallback);
  `aggregate_qb_game_stats` (one row per quarterback game of dropback sums, using the team
  families' dropback definition through the new public `pbp.dropback_condition` and
  `pbp.regular_season_plays`); `attach_qb_features` (strict as-of joins on
  `season * 100 + week`, so a row never sees its own week).
- `constants.QB_PBP_STATS` (7 stats, 21 columns with sides and diffs), `QB_PRIOR_DROPBACKS = 300`,
  `QB_RECENT_GAMES = 8`, `QB_META_DATA_NAME`, and the `qb` feature group, disjoint from `pbp` and
  `strength`. `finalize.build_final_column_order` places the columns; the schema grew from `498`
  to `519`.
- `data_collection._attach_qb_features`, called after `fill_future_qb_data` so future weeks use
  the assigned starter; it loads missing history seasons from 1999 so partial-season runs match a
  full rebuild, and leaves rows without quarterback columns alone.
- Tests: `tests/test_qb_stats.py` (identity, a hand-built play fixture, strictly-before and league
  prior by hand, recent window, a same-week and later-week perturbation, first-week nulls,
  abbreviated fallback and unknown names, schema and group disjointness) and
  `tests/test_data_collection_qb_features.py`.

### Deviations from the plan

- Recency is the last 8 games rather than a season-to-date rate that resets in week 1; career
  rates shrink to the league and recent rates to the career, so a first start gets the league
  rate (no separate rookie prior).
- Column names avoid `epa_per_dropback` so the `pbp` group does not swallow them;
  `qb_td_int_margin_rate` was left out.
- Scrambles (dropbacks without a passer id; the cache has no rusher id) are credited to the
  team-game's primary passer.

### Verification (rebuild of 2026-09-11 06:47)

- `data/completed_games_ml.csv`: `7263` rows, `519` columns, `acaa2892...`; the pre-rebuild build
  (the user's 06:03 refresh, `9b8bf303...`) is in `data/backup_pre_m53/`. The ETL logged 17148
  quarterback games and `0` of `7533` rows unmatched on both sides; CPOE is null before 2006 by
  design. Leakage audit: `484` features, `0` flags.
- Real data: the features of all 14 games of 2024 week 10 are identical when computed from
  play-by-play cut before week 10, and equal the ETL file exactly. Burrow's history before that
  week is `2453` dropbacks against `2358` raw passer-id dropbacks (the rest are credited
  scrambles). Correlation with the home margin: `qb_dropback_epa_diff` `-0.281`,
  `qb_dropback_epa_recent_diff` `-0.308`, `qb_any_a_diff` `-0.267`, `qb_sack_rate_diff`
  `+0.129`, all with the expected sign.

### Walk-forward (53.5)

Benchmark config (anchored, from week 1, `--eval-last-n-seasons 3`, Platt, 4 calibration weeks)
on `data/completed_games_ml.m53_through_2025.csv` (`7261` rows, `06a7a34d...`), one arm at a time,
default OpenMP policy: `models/wf_qb_2023_2025_on/` (checkpoints `cb41507aa5be425f3c3f`, 483
features including all 21 new columns) and `models/wf_qb_2023_2025_off/`
(`--disable-feature-groups qb`, checkpoints `6024b0fcaebe9c480dd4`, 462 features).

| window | games | Brier on / off | log loss on / off | pick acc on / off | margin MAE on / off |
| --- | --- | --- | --- | --- | --- |
| week 1 only | 48 | `0.2023` / `0.2058` | `0.5938` / `0.6012` | `0.7708` / `0.7292` | `8.9241` / `9.0535` |
| week 2 only | 48 | `0.2309` / `0.2353` | `0.6548` / `0.6634` | `0.6458` / `0.5833` | `8.6736` / `8.7179` |
| weeks 3-18 | 720 | `0.2327` / `0.2302` | `0.7708` / `0.7577` | `0.6819` / `0.6806` | `9.9839` / `9.9324` |
| all weeks | 816 | `0.2308` / `0.2291` | `0.7535` / `0.7430` | `0.6850` / `0.6777` | `9.8445` / `9.8092` |

Paired bootstrap over games (5000 resamples, seed 0), on minus off, weeks 3-18: Brier `+0.0025`
`[-0.0025, +0.0074]`, log loss `+0.0131` `[-0.0083, +0.0346]`, pick accuracy `+0.0014`
`[-0.0125, +0.0167]`, margin MAE `+0.0515` `[-0.0479, +0.1496]`. By season (weeks 3-18) Brier
`0.2422 / 0.1939 / 0.2619` on against `0.2516 / 0.1850 / 0.2539` off: better in 2023, worse in 2024
and 2025. Interpretation: no measurable gain; the direction on the headline window is slightly
against, the early weeks slightly for. The production training paths have no feature-group switch,
so the weekly model trains on the family until the user decides (open follow-up in `TODO.md`).

A second finding: the QB-off arm, same config and code as the benchmark, scores weeks 3-18 Brier
`0.2302` / log loss `0.7577` against the benchmark's `0.2284` / `0.7406` on the earlier build. The
inputs changed with the user's refresh; the cause is resolved below.

### Resolved after review (2026-09-11)

- **Decision: keep the quarterback family in production.** The user reviewed the on/off table and
  chose to keep it, no code change: week 2 pick accuracy improved meaningfully (`0.6458` against
  `0.5833`, 48 games) and the headline weeks-3-18 loss is inside its 95% interval either way, so
  there is no evidence against keeping it, only mixed evidence for it. A production
  disabled-feature-group switch was suggested in the first draft of this milestone but rejected as
  unnecessary complexity: the walk-forward tools (`walk_forward_backtest.py`, `wf_compare.py`)
  already support `--disable-feature-groups` for ablation studies, which is all this decision
  needed, and XGBoost's column/row subsampling already down-weights a genuinely low-signal family
  through gain-based splitting; a training-time switch is only worth adding later if the config
  sweep (Milestone 55) needs to search over feature-group inclusion, not for this decision.
- **Baseline shift explained: a full nflreadpy cache refresh, not a bug.** The user ran a full ETL
  with `--refresh-nflreadpy` overnight before this session (the 06:03 build that predates the
  quarterback rebuild), which re-pulls every season's schedule, team-stat and play-by-play cache
  from nflverse rather than reusing the historical cache. nflverse periodically republishes
  corrected historical values (box scores, EPA, market lines), so a full refresh can legitimately
  change many historical rows even with completely unchanged code and walk-forward config. The
  quarterback identity files were already current at both checkpoints (`data/qb_elos.csv` and
  `data/qb_meta_data.csv` matched their `../nfeloqb` sources byte-for-byte throughout), so the
  shift is not a quarterback-feature or identity-bridge defect. No further action needed; the
  `AGENTS.md` benchmark documents which build it was measured on for future audits.
- **Review fixes (2026-09-11, version `0.7.1`).** `_attach_qb_features` no longer passes
  `--refresh-nflreadpy` through to the history seasons it loads for career rates (a one-season
  refresh had become a full 1999+ play-by-play download); they always read the per-season cache.
  The missing-identity-file warning now says quarterbacks are matched by passer name only rather
  than claiming every feature turns null. The remaining review notes (scramble attribution,
  the recent window's one-dropback games, unread sums, reuse of the `pbp` helpers) are follow-ups
  in `TODO.md` to take along with task 53.6.

### 53.6 Schedule lenses (built `0.9.0` 2026-09-11; dropped `0.10.0` 2026-09-17)

- Built: `qb_faced_pass_def_adj` (ridge form, the sos `QSoS` construct) and
  `qb_faced_pass_def_raw` (one-hop, head-to-head excluded, like `sos_played_raw`), per side plus
  `_diff`, in `qb_stats.py`, crediting the `nfl-sos-ratings` method. Scope and weighting follow
  `QSoS`: the quarterback's games earlier in the row's season, dropback-weighted; the ridge value
  of each game comes from the snapshot of the week it was played, and the one-hop profile from
  the faced defense's games before the row's week minus those against the quarterback's team.
  The `qb` group now holds them; `qb_schedule` drops only them. The `pbp` helpers and
  `calculate_stat_differentials` reuse notes were taken along; `_ratio` was not.
- Build: cached ETL rebuild at 18:17 (`525` columns, `4cf48985...`), backup of the `519`-column
  build in `data/backup_pre_m53_6/`. Every pre-existing 1999-2025 value is unchanged (the
  quarterback columns exactly; the `sos_*` columns within `2.2e-16`, float summation order).
  Leakage audit `490` features, `0` flags. Real data: all 14 games of 2024 week 10 get identical
  lenses when recomputed from play-by-play cut before week 10 and snapshots before week 10. The
  lenses are null in week 1 (and the one-hop lens in week 2), about 1-3% null from week 3.
  Correlation with the home margin, weeks 3-18: `qb_faced_pass_def_adj_diff` `-0.055`,
  `qb_faced_pass_def_raw_diff` `+0.047` (both with the expected sign); the two lenses correlate
  `-0.375` with each other (opposite sign conventions).
- Walk-forward, benchmark config (anchored, from week 1, `--eval-last-n-seasons 3`, Platt, 4
  calibration weeks) on `data/completed_games_ml.m53_6_through_2025.csv` (`7261` rows,
  `940cbbf4...`), one arm at a time, default OpenMP policy: `models/wf_qbsched_2023_2025_on/`
  (checkpoints `1ca801a7b58256dc1442`, 489 features) and `models/wf_qbsched_2023_2025_off/`
  (`--disable-feature-groups qb_schedule`, checkpoints `02a3a730da026668dffe`, 483 features).
  Windows are game-weighted from `per_week`; the same scorer reproduces the 53.5 table and its
  bootstrap exactly.

| window | games | Brier on / off | log loss on / off | pick acc on / off | margin MAE on / off |
| --- | --- | --- | --- | --- | --- |
| week 1 only | 48 | `0.2065` / `0.2051` | `0.6015` / `0.5996` | `0.6875` / `0.7708` | `9.2897` / `8.9283` |
| week 2 only | 48 | `0.2280` / `0.2275` | `0.6475` / `0.6457` | `0.6458` / `0.6667` | `8.4192` / `8.5291` |
| weeks 3-18 | 720 | `0.2312` / `0.2282` | `0.7575` / `0.7551` | `0.6764` / `0.6903` | `10.0226` / `9.9708` |
| all weeks | 816 | `0.2295` / `0.2268` | `0.7419` / `0.7395` | `0.6752` / `0.6936` | `9.8852` / `9.8246` |

Paired bootstrap over games (5000 resamples, seed 0), on minus off. Weeks 3-18: Brier `+0.0030`
`[-0.0019, +0.0080]`, log loss `+0.0025` `[-0.0194, +0.0249]`, pick accuracy `-0.0139`
`[-0.0292, +0.0014]`, margin MAE `+0.0519` `[-0.0488, +0.1574]`. Weeks 1-2: Brier `+0.0009`
`[-0.0056, +0.0075]`, pick accuracy `-0.0521` `[-0.1042, -0.0104]` (5 of 96 picks). By season
(weeks 1-18) Brier `0.2363 / 0.2055 / 0.2467` on against `0.2349 / 0.1982 / 0.2473` off.
Interpretation: no gain anywhere; every point estimate leans against the lenses, only the early
pick accuracy clears its interval.

Decision (user, 2026-09-17): drop. Removed in version `0.10.0` from the same branch:
`constants.QB_SCHEDULE_STATS`, the `qb_schedule` group, the `_schedule_lenses` machinery and the
`defense_games` / `snapshots` arguments of `attach_qb_features` / `_attach_qb_features`, and the
tests; `opponent_abbr` on the quarterback-game rows and the `pbp` / `calculate_stat_differentials`
reuse stay. Reasoning, independent of the table: a schedule faced is a nuisance parameter for
estimating the quarterback's skill, not a predictor of the next game, so its value lies entirely
in the subtraction `production - expected production given schedule`; as standalone columns the
lenses left that subtraction for the trees to discover as an interaction. Their season-to-date
window matched neither the career rate (all seasons, `K = 300`) nor the last-8 rate (which reaches
into the prior season until about week 9), so there was no aligned rate to adjust. The team-level
ridge `adj_off_pass_epa` is already opponent-adjusted, so the lenses could add information only
where the quarterback's schedule differs from the team's (mid-season starter changes), exactly
where the `K = 300` prior dominates his rate. And their noisiest weeks (3-5, two to four defenses
with two to four games each) were also their most useful ones, since by mid-season the ridge
snapshots have converged. The two correlated weak columns (`-0.375` with each other, `+-0.05`
with the margin) then cost a little variance through column subsampling, consistent with every
point estimate leaning against them. If the idea returns it goes inside the rate: task 53.7 in
`TODO.md`, the defense-adjusted rate first and the ridge only if that shows signal. The
`0.10.0` rebuild and its verification are recorded under "53.6 removal" below.

### 53.6 removal (2026-09-17, version `0.10.0`)

- Code: `constants.py` (`QB_SCHEDULE_STATS`, the `qb_schedule` group), `finalize.py`,
  `qb_stats.py` (docstring section 4, the lens constants, `_faced_games`,
  `_faced_snapshot_values`, `_faced_one_hop_values`, `_dropback_weighted`, `_schedule_lenses`,
  the `defense_games` / `snapshots` arguments), `data_collection.py` (the same arguments and the
  snapshot / team-game wiring into `_attach_qb_features`), seven tests across
  `tests/test_qb_stats.py` and `tests/test_data_collection_qb_features.py`; 342 lines removed.
- Build: cached ETL rebuild at 21:39 into `data/completed_games_ml.csv` (`7277` rows, `519`
  columns, `0cecc2e3...`); the `525`-column build it replaced (the Week 2 weekly run's 18:13 ETL,
  `3f1db457...`) is in `data/backup_pre_m53_6_drop/`. Aligned on `(season, week, away_abbr,
  home_abbr)`, every shared column is identical (numeric max difference `0`, no null mismatch);
  the only schema change is the six lens columns gone. Quarterback identity still `0` of `7533`
  rows unmatched per side. Leakage audit `484` features, `0` flags
  (`models/audit_m53_6_drop/leakage_audit.json`).
- Gate: `814 passed`, coverage `92.89%`, ruff, pyright, ty and markdownlint (on the changed
  docs) clean. No walk-forward: with the columns absent the model sees exactly the off arm of the
  53.6 measurement (`models/wf_qbsched_2023_2025_off/`), which stands as the number.

---

## Milestone 52 - The total (over/under) head carries almost no signal

Completed 2026-09-11 (fix in version `0.6.2`, report label in `0.6.3`). Outcome: the total head
learns again, but it still trails the market's total line in walk-forward, so the total columns are
labelled diagnostic-only. The original record follows.

Formerly Milestone 50 (found 2026-09-09). Predicted totals for the 2026 Week 1 slate all land
between `43.9` and `44.1` while market totals for the same games range `40.5` to `47.5`. The model
is effectively predicting the league mean for every game. Training holdout `total_mae` is `10.9974`
against a `margin_mae` of `9.8471`.

Consequence: the `total_value_side`, `total_edge_prob`, `total_confidence_1_10` and `total_ev`
columns in the betting workbook are computed from that flat prediction and are not actionable. The
spread and moneyline columns are unaffected. Do not present total-based betting recommendations as
usable until this is resolved.

Tasks:

- [x] 52.1 Diagnose: feature importance for the total head; whether the total target is being
      learned at all (early-stopping round, train vs holdout MAE); whether the pruning or feature
      selection step is dropping total-relevant columns. Done 2026-09-11; findings below. No code
      changed.
- [x] 52.2 Fix the shared early-stopping callback: give each estimator in
      `_fit_margin_total_models` its own `EarlyStopping` instance (a fresh params copy per head),
      with a regression test that the total head's round count does not depend on the margin fit
      (the synthetic reproduction below makes a good fixture). Then run the walk-forward reference
      and fixed arms on one build and code version, anchored (the benchmark config) and unanchored
      (the production config), and record total MAE. Only if a healthy total head still trails
      the market line, test a separate feature set for it. Done 2026-09-11. Deviations: the fix
      drops the explicit callback instead of copying it per head (XGBoost already builds a fresh
      one from the init parameter); the reference anchored arm was not rerun, because the fixed
      anchored arm reproduced the benchmark's margin metrics exactly and the benchmark's own
      checkpoints serve as that reference; the separate feature set was diagnosed, not built.
- [x] 52.3 Record the walk-forward table; either fix the default or mark the total columns of the
      betting workbook as diagnostic-only in the report and README. Done 2026-09-11: the healthy
      head trails the line, so the totals are labelled diagnostic-only.

Findings from 52.1 (2026-09-11):

- **Root cause: the total head shares the margin head's early-stopping callback.** With xgboost
  `3.4.1`, `fit()` no longer takes `early_stopping_rounds`, so `_with_xgb_early_stopping_params`
  puts one `xgb.callback.EarlyStopping` instance into the params dict, and
  `_fit_margin_total_models` builds both `XGBRegressor`s from that dict. The callback keeps its
  best score and patience counter between fits. The total fit therefore starts against the margin
  head's best validation RMSE (about `9.5`, which a total RMSE never beats) with the patience
  counter already spent when the margin head stopped early, and it stops after one round. When the
  margin head runs to `n_estimators` without stopping, the total head gets at most the patience
  left over (up to 50 rounds). `_fit_quantile_models` builds fresh params for every quantile, so the
  quantile heads are healthy (`total_q0.5` stopped at iteration `346` in the 2026 model).
- **Evidence on disk.** `models/week01_2026_refreshed/model.joblib` and
  `models/week01_2026_strength/model.joblib`: margin heads of `283` and `276` trees
  (`best_iteration` `232`, `225`), total heads of **1 tree** with no `best_iteration`, and
  `metadata.json` records early stopping for every head except `total_model`.
  `models/weekly_2025_week_22` has the same 1-tree total head. Runs without early stopping
  (`models/review_*`) keep all `598` trees in both heads. Train versus holdout MAE cannot say more
  than "the total is flat": holdout `total_mae` is `10.9974`, and feature importance for a one-tree
  head is meaningless.
- **Reproduction.** Calling `_fit_margin_total_models` on synthetic data with a planted total
  signal and `early_stopping_rounds=50` gives a 1-tree total head whose predictions span
  `43.70-44.08` (std `0.07`), the 2026 symptom. The same total head fit with its own callback keeps
  `235` trees (std `3.85`) and cuts eval MAE from `8.60` to `8.08`.
- **Scope.** Every caller of `_fit_margin_total_models` that passes an eval set: final training,
  Optuna trials (whose `combined_mae` objective has been scoring a crippled total), walk-forward
  folds, the blended model and `model_compare.py`. The margin head is fit first with a fresh
  callback, so margin predictions, win probabilities, Brier, log loss and pick accuracy are not
  affected; only total predictions and total MAE are (and tuning, through the objective).
- **Why the benchmark hid it.** The walk-forward benchmark runs with `market_anchor` on, so the
  total head predicts a residual on `total_line` and a crippled head gives roughly the market line
  plus a constant. Over the benchmark's 816 games (fold checkpoints in
  `models/wf_checkpoints/5ea347bc3339f5d3a9e3/`, the on arm) total MAE is `10.1000`, against
  `10.1207` for the market line alone and `10.1378` for the p50 quantile head; the within-fold std
  of `predicted_total - total_line` has a median of `0.51`. The production model has
  `market_anchor` off, which is why it prints a flat 44.
- **Not the cause.** Feature pruning or selection (the head never gets past its first round), and
  the total target itself (the quantile heads learn it).

Results of 52.2 (2026-09-11, version `0.6.2`):

- **Fix landed.** `_with_xgb_early_stopping_params` sets only the `early_stopping_rounds` init
  parameter, so XGBoost builds a fresh `EarlyStopping` for every fit. Tests:
  `tests/test_ml_model_margin_total_early_stopping.py` (the paired total head keeps the rounds a
  solo fit keeps and predicts with std above 1; it failed on the old code with a 1-round head) and
  `tests/test_ml_model_xgb_utils.py` (no `callbacks` entry, no shared object).
- **Production config, same live build (`e388dc7a...`).** Old code
  (`models/week01_2026_totalref`, trained from a `HEAD` worktree) against the fix
  (`models/week01_2026_totalfix`): identical margin head (151 trees, best iteration 100); total
  head 1 tree against 235 (best iteration 184); Week 1 totals `43.9-44.2` (std `0.08`) against
  `39.4-48.8` (std `3.01`, correlation `0.956` with market lines of `38.5-50.5`). With the build cut
  to `<= 2025` so the holdout is 2025 (272 games; `models/holdout2025_total{ref,fix}`), holdout
  total MAE is `11.0051` against `10.5387`; the market line scores `10.3934` on the same games.
  Margin MAE, Brier and accuracy are identical.
- **Fixed anchored arm** (`models/wf_totalfix_2023_2025_anchored/`, checkpoints
  `models/wf_checkpoints/c39db4f843175eaab09f/`, benchmark flags on
  `data/completed_games_ml.m49_on_through_2025.csv`). Brier, log loss, pick accuracy and margin MAE
  equal the benchmark to four decimals in every window, so the fix changed nothing on the margin
  side. Total MAE:

  | window | games | crippled (benchmark) | fixed | market line |
  | --- | --- | --- | --- | --- |
  | week 1 only | 48 | `9.9426` | `9.9426` | `10.3333` |
  | week 2 only | 48 | `9.8727` | `9.8727` | `10.4479` |
  | weeks 3-18 | 720 | `10.1257` | `10.2295` | `10.0847` |
  | all weeks | 816 | `10.1000` | `10.1916` | `10.1207` |

  Weeks 1-2 are identical because those folds skip calibration for lack of rows, so they have no
  eval set, no early stopping, and never had the bug. In weeks 3-18 the healthy anchored head is
  worse: against the crippled head `+0.1038` (95% paired bootstrap `[-0.0191, +0.2287]`), against
  the line `+0.1448` (`[-0.0073, +0.2991]`); by season `10.44 / 9.88 / 10.36` against the crippled
  `10.45 / 9.82 / 10.11`, so 2025 carries the gap. The spread of `predicted_total - total_line`
  grows from a median within-fold std of `0.533` to `2.046`: under anchoring the head now learns a
  residual, and on a four-week early-stopping window that residual is mostly noise.
- **Fixed unanchored arm, the production configuration** (`models/wf_totalfix_2023_2025_unanchored/`,
  checkpoints `models/wf_checkpoints/8eb0587a0da4c6ab57cc/`, same build and flags with
  `--no-market-anchor`). Total MAE `9.8223` / `9.8879` / `10.3152` / `10.2610` for week 1, week 2,
  weeks 3-18 and all weeks, against the line's `10.3333` / `10.4479` / `10.0847` / `10.1207`. In
  weeks 3-18 it trails the line by `+0.2305` (95% paired bootstrap `[+0.0725, +0.3881]`) and the
  crippled anchored head by `+0.1895` (`[+0.0493, +0.3287]`); by season `10.48 / 10.05 / 10.42`
  against the line's `10.33 / 9.79 / 10.13`. Median within-fold std of `predicted_total -
  total_line` is `2.091` (the crippled head's was `0.533`). Its probability metrics are not a
  benchmark comparison (anchoring changes the margin head too): Brier `0.2322`, log loss `0.7602`,
  pick accuracy `0.6740`, margin MAE `9.8906` over all weeks.
- **Why a healthy head still trails: its deviation from the line carries no signal.** In weeks
  3-18 the correlation of `predicted_total - total_line` with `actual_total - total_line` is
  `-0.012` (`-0.034` for the p50 quantile head), and blending back toward the line only helps:
  total MAE of `line + k * (prediction - line)` rises monotonically from `10.0847` at `k = 0` to
  `10.3152` at `k = 1` (over all weeks the minimum is `10.1176` at `k = 0.1`). The head learns the
  line, which is one of its features, plus noise. A separate feature set for the total head would
  only help if it carries information the closing total does not (weather, pace, officiating,
  late injury news), and none of today's families was built for that. Not built this session.
- **Reference unanchored arm** (old code; `models/wf_totalref_2023_2025_unanchored/`, checkpoints
  `models/wf_checkpoints/55a1388a301ce116b10a/`). Total MAE `9.8223` / `9.8879` / `10.3009` /
  `10.2485` by window, identical to the fixed arm in weeks 1-2 (no eval set) and **statistically
  tied** with it in weeks 3-18: reference minus fixed `-0.0142` (95% paired bootstrap
  `[-0.1745, +0.1399]`), by season `10.58 / 9.87 / 10.45` against `10.48 / 10.05 / 10.42`. It
  trails the line by `+0.2162` (`[+0.0307, +0.4032]`). The pre-fix head is not a flat 44 in
  walk-forward: when the margin head stops late, the shared callback leaves the total head part of
  its patience, so it is only flatter (median within-fold std of `predicted_total - total_line`
  `2.436` against the fixed `2.091`; a constant prediction would deviate by the line's own spread).
  Its margin and probability metrics equal the fixed arm's, as they must. So the fix restores the
  total head's behaviour (it tracks the market, and the single-split 2025 holdout improves from
  `11.0051` to `10.5387`) but does **not** improve walk-forward total MAE, unanchored or anchored:
  the head has nothing to add to the closing line either way.

52.3 decision and label (version `0.6.3`): the fixed unanchored head trails the line, so every row
of `scripts/betting_pipeline.build_betting_report` (the weekly run's `*_betting_report.csv`)
carries `total_signal = diagnostic_only` next to `total_edge_points`
(`TOTAL_SIGNAL_STATUS`, test `tests/test_betting_pipeline_recs.py`), and README's Scripts section
says the report and workbook totals are diagnostics. The workbook itself is unchanged.

How the walk-forward arms were run and scored:

- One build for every arm: `data/completed_games_ml.m49_on_through_2025.csv` (dataset fingerprint
  `5d67ddff19f8...`, `7260` rows, seasons `<= 2025`). The arms ran one at a time on an idle machine
  with the default OpenMP policy, about 38-40 minutes each. Their `metadata.json` records git
  `6dda1bc` because the fixes were not committed yet; the per-week checkpoint fingerprint, which
  hashes the modelling source, is what separates code versions.
- Fixed anchored: `.venv/bin/python scripts/walk_forward_backtest.py --data-path
  data/completed_games_ml.m49_on_through_2025.csv --eval-last-n-seasons 3 --wf-start-week 1
  --calibration platt --wf-calibration-weeks 4 --market-anchor --market-transform --out-json
  models/wf_totalfix_2023_2025_anchored/metrics_report.json`.
- Fixed unanchored: the same with `--no-market-anchor` and `--out-json
  models/wf_totalfix_2023_2025_unanchored/metrics_report.json`.
- Reference unanchored: the same unanchored command run from a detached `HEAD` worktree with
  `PYTHONPATH` pointing at it, `--out-json models/wf_totalref_2023_2025_unanchored/metrics_report.json`.
  Its checkpoints were written inside the worktree and copied to
  `models/wf_checkpoints/55a1388a301ce116b10a/`. Old code is proven by week 5 of 2023, the first
  fold with an early-stopping eval set: identical margins, total std `1.886` against `4.912`.
- Windows were scored from the per-week checkpoints (games, Brier, log loss, pick accuracy, margin
  and total MAE, the line's total MAE, the p50 head, and the std of `predicted_total -
  total_line`); the same scorer reproduces the `AGENTS.md` benchmark table to four decimals from
  `models/wf_checkpoints/5ea347bc3339f5d3a9e3/`. Paired bootstrap: 5000 resamples of games,
  seed 0. In both anchored arms the 2023 weeks 1-4 folds are identical and every fold from week 5
  differs, so the first four weeks of a season have no early-stopping eval set.

Acceptance:

- [x] Weekly predicted totals span a range comparable to the market's, or the total outputs are
      explicitly labelled non-actionable. Both: the retrained production model's Week 1 totals
      span `39.4-48.8` against market lines of `38.5-50.5`, and the report labels them
      `diagnostic_only`.

---

## Milestone 56 (partial) - Weekly orchestration residuals

Task 56.4 completed 2026-09-11 (version `0.6.1`); task 56.2 completed 2026-09-21 (version
`0.13.0`); tasks 56.1, 56.3 stay in `TODO.md`.

### 56.2 - How postseason games enter evaluation, training and the rankings (version `0.13.0`)

Narrowed 2026-09-20 (`0.12.11`): the power-rankings through-week clamp landed (defaults to the
last regular-season week when the prediction week is postseason) and the README documented the
state at the time (code defaults exclude postseason; the shipped `config/weekly_run.yaml`
included it at weight `1.3`, so the weekly command and the documented defaults disagreed). The
decision on which side to align was left to the user.

Direction 2026-09-21: postseason matchups stay in the data and are kept separate from
regular-season matchups for training and for prediction. A model used for regular-season weeks
should not train on playoff games, and the prior-season blend should draw on the previous
regular season only. The playoff-week design itself (how a playoff-specific model or weighting
would work) stays deferred until the rest of the pipeline runs smoothly, wanted in time for the
2026 playoffs.

Decided 2026-09-21: set `include_postseason: false` and `wf_include_postseason: false` in
`config/weekly_run.yaml` (approved), so the weekly run matches the rule that a model used for
regular-season weeks never trains on playoff games. `postseason_weight` stays in the config but
inert. Verified facts behind the decision: season-to-date stats, adjusted strength, records and
the prior-season blend are regular-season only; Elo and QB Elo (`data/qb_elos.csv`) and the
TeamRankings playoff-week snapshots are the only features that carry playoff results, and they
enter as pre-game ratings, so there is no leakage. Landed 2026-09-21 in `0.13.0` on
`feat/m55-7-default-200`, together with the task 55.7 default-alignment chunk (`ARCHIVE.md`,
Milestone 55, "55.7").

### 56.4 - The in-season calibration window rolls back across the season boundary

- Cause: `ml_model_core._split_train_calibration_holdout` took its in-season calibration weeks only
  from the newest pool season and raised `Not enough weeks in season 2026 for calibration.` when
  that season had fewer than `calibration_weeks` (weekly default `4`). Every weekly run for weeks
  2-4 of a season failed at Stage 2; the weekly-run smoke test `models/smoke_20260911` found it
  once the rebuild added two completed 2026 games.
- Fix: the window is the newest `calibration_weeks` distinct `(season, week)` pairs across the pool
  in time order (`_latest_season_week_pairs`), training drops exactly those pairs
  (`_season_week_mask`), and whole-season calibration picks only seasons the window does not touch.
  The one remaining error is a pool with fewer weeks than requested. No flag was added and the
  guard in `scripts/weekly_run.py` is unchanged.
- Metadata: `splits.calibration_inseason` keeps `season` and `weeks` (the newest season in the
  window and its weeks) and adds `pairs` (`_inseason_calibration_pairs`), in both the margin/total
  and the blend reports. The return tuple of the split is unchanged, so no caller moved.
- Tests: `tests/test_ml_model_core_helpers.py` (the Week-2 rollback with training excluding exactly
  the window, the unchanged split when the newest season has enough weeks, whole-season calibration
  skipping touched seasons, the pairs helper, and the pool-too-small error replacing the test that
  pinned the old one), `tests/test_ml_model_training_report.py` (a two-season window recorded in
  metadata) and `tests/test_ml_model_training_score_blend.py`.
- Verification on the live dataset (regular season, `e388dc7a...`): the new split equals the
  previous code, frames included, in all 126 configurations the previous code accepted (holdout
  0-2, calibration seasons 0-2, weeks 0, 1, 2, 4 and 6, on the full build, the build cut at 2025,
  and one cut at 2025 week 8); the previous code raised in the other 9. `(0, 0, 4)` now calibrates
  on 2025 weeks 16-18 plus 2026 week 1 (50 games) and trains on 6904.
- Real pipeline: `weekly_run.py --skip-data-refresh --run-id smoke_20260911 --resume` with default
  training flags reused Stage 1, trained, and wrote predictions, confidence picks, the betting
  report, power rankings and projected standings; `metadata.json` lists the four pairs. The user's
  earlier run of the same id with the workaround flags (`--train-calibration-weeks 0
  --train-calibration-seasons 2`) is kept in `models/smoke_20260911_workaround/`.
- Observed, not changed: training early-stops on the calibration frame, so the 50-game window
  stopped the anchored margin head at iteration 1. That predates the fix and is an open follow-up
  in `TODO.md`.
- Review follow-up (2026-09-11, version `0.7.1`): the season-count guard predated the window and
  still demanded a spare pool season beyond the whole calibration seasons, so a three-season pool
  with `--train-calibration-seasons 1` raised `Not enough seasons` in weeks 2-4; it now raises
  only when no season is left to train on. The `Calibration weeks` log line prints the window's
  `[season, week]` pairs. The walk-forward's own calibration selection was not changed (open
  follow-up in `TODO.md`).

---

## Milestone 51 - Power rankings on the adjusted composite

Completed 2026-09-11 (formerly Milestone 43 phase 2). The 51.1 design fork was settled by the
user as option (c): the ETL writes the per-team weekly strength snapshot it already solves, so bye
teams are ranked exactly, offline, and from the same numbers the model sees.

### What landed

- `data/strength_snapshots.csv` from `nfl_predictor.data_collection`: one row per
  `(season, week, team)` for every team on the season's schedule, bye teams included, with
  `constants.ADJUSTED_STRENGTH_STATS` plus the home-field term `adj_hfa`
  (`constants.STRENGTH_SNAPSHOT_FILE_COLUMNS`). `process_week` records the same
  `build_strength_table` frame it joins onto the game rows, so the file equals the model's features
  by construction. `process_season` adds the week after the regular season when the playoff
  schedule is not published yet, so a ranking through the last regular-season week always works.
- `scripts/power_rankings.py --method composite`, now the default, ranks through week N on the
  week N+1 snapshot. The documented transform (`rank_teams_on_composite`):
  `points_vs_average = beta * (composite - mean)`, with `beta` the within-week OLS slope of
  `adj_srs` on the composite, then `p = Phi(points / SCORE_DIFF_STD_DEV)` and the existing
  `1 + 9p` and `10p` scales. Each row publishes the composite, `points_vs_average`, the five
  weighted components, `adj_srs`, `strength_games_played` and `snapshot_week`.
- `--method bradley_terry` keeps the previous default output, and `--legacy-franchise-fit` implies
  it. `compute_power_rankings` is shared with `scripts/weekly_run.py`, which exposes
  `--power-rankings-method`, `--power-rankings-strength-snapshots`, the four `--ratings-*` options
  and `--legacy-franchise-fit` (51.4). They are validated at parse time and included in the
  reports-stage reuse hash, and a missing snapshot week skips the rankings with a warning.
  Deviation from the plan: the weekly flag is `--power-rankings-method`, not `--method`, because
  the weekly runner has many stages and its other ranking flags carry the same prefix.
- 51.2: `scripts/golden_command.py` writes its per-model rating table as
  `model_rating_rankings.csv` and labels it a diagnostic; projected standings keep their method.
- 51.5: README, `--help` and `AGENTS.md` explain the current-season (composite) and franchise
  (legacy Bradley-Terry) views and the new data file.
- Tests: `tests/test_strength_snapshot_file.py` (bye-team rows; a week-N snapshot unchanged when
  week N onward is rewritten, with a counter-test that earlier weeks do move it; file equal to the
  game-row features; schema when nothing was solved; the full-season week; `main` writes the file),
  composite tests in `tests/test_power_rankings.py` (strongest first, monotone and bounded scales,
  average team at mid-scale, missing points scale, unrated team kept last, a breakout team first by
  week 16 but not in week 2), script tests (next-week snapshot, later weeks ignored, missing and
  duplicate snapshot errors, option resolution, a Bradley-Terry characterization pinned before the
  refactor), and `tests/test_weekly_run_power_rankings.py`.

### Found and fixed on the way

- Records and projected standings compared scores as text: the ETL writes the newest games first,
  so once unplayed 2026 games led `all_data.csv`, Polars inferred the score columns as strings.
  For 2024 through week 17, 26 of 32 records were wrong (DET 11-5 instead of 14-2, KC 14-2 instead
  of 15-1). Bradley-Terry ratings were unaffected because `outcome_to_home_prob` coerces.
- Projected standings were empty before a season's first game, because they were built on record
  rows that do not exist yet. That hit the live 2026 Week 1 weekly run.
- Not a defect: nine model features (QB Elo trends, `sos_played_raw`) are also inferred as strings
  when `_predict_future_games` reads `all_data_ml.csv`, but the model coerces them; predicted
  probabilities are identical to a full-file read for 2026 and 2024.

### Verification (2026-09-11 rebuild, about 10 minutes)

- `data/completed_games_ml.csv`: `7262` rows, `498` columns, fingerprint `e388dc7a...`. That is the
  previous build's `7261` rows plus SF at LAR (2026 week 1, 27-7), completed since. All `7261`
  shared rows match the previous build within `1e-9` on the `468` numeric columns outside
  schedule strength; the `sos_*` columns move by the known last-ULP drift only (max `5.6e-17`).
  In `all_data_ml.csv` only `263` future 2026 rows moved, from the new result. The previous build
  is in `data/backup_pre_m51/`.
- `data/strength_snapshots.csv`: `18818` rows, fingerprint `ff5f4823...`, 31-32 teams per week, no
  duplicate keys, 2026 weeks 1-19 with 32 teams each. Across all `7533` game rows, 0 of `165726`
  strength cells differ from the snapshot (null-safe, `1e-12`). Five rows have a null composite:
  teams with no prior season in the data and no game yet (BAL, LAC and LAR in 1999 weeks 2-3; HOU
  in 2002 week 1). The composite method ranks such a team last with a warning.
- `--method bradley_terry` on the real 2024 data through week 17 reproduces the pre-change ranks,
  power ratings and columns exactly (`rating_raw` within `2.2e-16`, the CSV round trip).
- Anchor, 2024 through week 17 (snapshot week 18). Composite top ten: BAL, DET, PHI, BUF, GB, KC,
  MIN, TB, DEN, WSH (BAL `8.07`, DET `7.80`). Bradley-Terry: DET, BAL, BUF, GB, PHI, KC, MIN, TB,
  LAC, DEN. Both top fives match the anchor.
- 2026 through week 0 (the Week 1 ranking): 32 teams, no nulls; LAR, SEA, NE, BUF and JAX lead and
  LV is last.
- No walk-forward: training rows did not change. The leakage audit was not rerun; its last run
  (2026-09-10 build, `463` features, `0` findings) predates one added game and no new feature.

---

## Milestone 43 phase 1 - Current-season Bradley-Terry power rankings

Completed 2026-09-09. Phase 2 continues as Milestone 51 in `TODO.md`.

The old `scripts/power_rankings.py` fit Bradley-Terry over every season since 1999 with equal
weights, fixed `0.97 / 0.03` targets, and future games filled with model probabilities. For 2024
through week 18 it ranked a 4-13 New England first.

What landed, in `scripts/power_rankings.py` and `nfl_predictor/reporting/power_rankings.py`:

- `--ratings-window-seasons` (default `2`) and `--ratings-prior-season-weight` (default `0.25`),
  implemented as per-game sample weights in `fit_bradley_terry_ratings`; uniform weights reproduce
  the unweighted fit exactly.
- Margin-based targets by default (`--ratings-target`), scoring completed games through the model's
  win-probability curve.
- Future model-probability rows excluded from the strength fit (`--ratings-include-future`).
- `--legacy-franchise-fit` reproduces the old output exactly, pinned by a test.

Evidence: the new default ranks DET, BAL, BUF, GB, PHI for 2024 through week 18, matching the
season's results and the schedule-adjusted snapshot. Not done in this phase: `scripts/weekly_run.py`
inherits the defaults but exposes none of the flags (task 51.4).

---

## Milestone 49 - Continuous early-season shrinkage

Completed 2026-09-10.

Season-to-date team stats (the nflreadpy families and the play-by-play counts alike) used to switch
from 100% regressed prior season in week 1 to a single unshrunk game in week 2. They now hand over
continuously: `w = games / (games + K)`, `K = constants.PRIOR_BLEND_GAMES = 4.0`,
`published = w * in_season_mean + (1 - w) * regressed_prior_mean`, with every derived rate
recomputed from the blended sums. A team with zero games has `w = 0`, which is the old Week-1
fallback. **Week 2 is no longer the weak week**, and nothing else got worse on the primary metrics.
Shipped **default-on**.

### What landed

- `polars_utils.blend_with_prior_stats` (in `utils/polars/teamrankings.py`): blends the per-game
  means, keeps `games_played` as the in-season count, publishes whichever side exists when only one
  does, then calls `recompute_derived_metrics`.
- `data_collection.build_prior_season_stats`: the regressed previous regular season, built once per
  season in `process_season` and passed to `process_week` as `prior_season_stats` (computed inside
  when `None`). It is both the Week-1 fallback and the blend's prior.
- CLI on `nfl_predictor.data_collection`: `--stat-prior-blend` / `--no-stat-prior-blend` (default
  on) and `--stat-prior-blend-games` (default `4`, must be positive). `PRIOR_BLEND_GAMES` moved from
  `strength_snapshot.py` to `constants.py`, shared by both blends.
- `tests/test_stat_prior_blend.py` (15 tests) plus two CLI tests: week-1 rows identical with the
  blend on and off; `0.2 * in_season + 0.8 * prior` for a one-game team; rates as ratios of blended
  sums, distinguished from a blend of rates (`0.2667` vs `0.30`); first season untouched; team with
  no prior season untouched; the prior built once per season.

### Build verification (2026-09-10 11:17 rebuild, 11.5 min)

- Week-1 stat columns are bit-identical to the pre-change backup (`data/backup_pre_m49/`). The only
  week-1 differences anywhere are the three `sos_remaining_adj` columns at most `6.9e-17` apart, the
  known Polars summation-order drift.
- 2024 week-2 `away_success_rate` std `0.0807` (range `0.250-0.569`) became `0.0316`
  (`0.366-0.485`), below the old week-16 spread of `0.0359`. Week 16 moved too (`0.0359` to
  `0.0306`), as designed: the prior never fully drops out.
- No strength, TeamRankings, Elo, trend, or record column changed; 291 stat columns did.
- Leakage audit OK: `463` features, `0` findings
  (`models/wf_shrink_2023_2025_on/leakage_audit.json`).

### Walk-forward (2023-2025, from week 1, 816 games, 54 folds)

Both arms ran on one code version (`91aaffc` plus this change) and one config. Off arm: the
pre-change build (`data/completed_games_ml.pre_m49.csv`, hash `5b6af6aa...`),
`models/wf_shrink_2023_2025_off/`. On arm: the blend build cut to seasons `<= 2025`
(`data/completed_games_ml.m49_on_through_2025.csv`, hash `5d67ddff...`) because the rebuild had
picked up the 2026 opener, `models/wf_shrink_2023_2025_on/`. Same 816 games on both.

| window | games | arm | Brier | log loss | pick acc | margin MAE | total MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| week 1 | 48 | off | `0.2119` | `0.6134` | `0.6667` | `9.3640` | `10.1288` |
| week 1 | 48 | on | `0.2097` | `0.6097` | `0.7083` | `9.1365` | `9.9426` |
| week 2 | 48 | off | `0.2434` | `0.6799` | `0.5417` | `8.7892` | `10.2705` |
| week 2 | 48 | on | **`0.2268`** | **`0.6452`** | **`0.6042`** | **`8.3401`** | `9.8727` |
| weeks 3-18 | 720 | off | `0.2293` | `0.7541` | `0.6847` | `9.8952` | `10.1536` |
| weeks 3-18 | 720 | on | `0.2284` | `0.7406` | `0.6847` | `9.9578` | `10.1257` |
| all weeks | 816 | off | `0.2291` | `0.7414` | `0.6752` | `9.7989` | `10.1590` |
| all weeks | 816 | on | `0.2272` | `0.7273` | `0.6814` | `9.8143` | `10.1000` |

Paired game-level differences (on minus off; 95% interval from 10,000 bootstrap resamples of
games; per-game predictions read from the two arms' fold checkpoints):

| window | Brier diff | log loss diff |
| --- | --- | --- |
| week 1 | `-0.0022` [`-0.0146`, `+0.0102`] | `-0.0037` [`-0.0296`, `+0.0216`] |
| week 2 | `-0.0166` [`-0.0332`, `-0.0007`] | `-0.0347` [`-0.0717`, `+0.0002`] |
| weeks 3-18 | `-0.0009` [`-0.0071`, `+0.0055`] | `-0.0134` [`-0.0396`, `+0.0130`] |
| all weeks | `-0.0019` [`-0.0075`, `+0.0038`] | `-0.0141` [`-0.0367`, `+0.0086`] |

Per season (Brier / log loss / pick accuracy / margin MAE, all weeks):

| season | off | on |
| --- | --- | --- |
| 2023 | 0.2427 / 0.7706 / 0.6654 / 10.0492 | 0.2353 / 0.7515 / 0.6765 / 10.1017 |
| 2024 | 0.1980 / 0.6817 / 0.7169 / 9.5774 | 0.1989 / 0.6735 / 0.7243 / 9.5480 |
| 2025 | 0.2468 / 0.7719 / 0.6434 / 9.7701 | 0.2475 / 0.7569 / 0.6434 / 9.7933 |

Reliability ECE over all weeks: `0.1073` off, `0.1081` on (flat).

**Did it work?** On the target, yes. Week-2 Brier falls to the weeks 3-18 level (`0.2268` against
`0.2284`), the one Brier interval that excludes zero, and pick accuracy gains 6.25 points. Week-2
log loss improves by about as much, with its interval just touching zero. Weeks 3-18 do not
regress: Brier and log loss both move the right way, within noise, and pick accuracy is identical.

**What did not improve.** Weeks 3-18 margin MAE is worse (`9.8952` to `9.9578`). Season Brier is
slightly worse in 2024 and 2025 (`+0.0009`, `+0.0007`) and better in 2023; log loss improves in all
three. With 48 games a week, the week-2 result is supported but not overwhelming.

### Corrections to the plan

- The acceptance criterion "week 1 metrics identical on both arms" rested on a wrong premise.
  Week-1 *features* are identical (verified on the 48 evaluated rows), but each week-1 model trains
  on every earlier season, whose week 2+ rows the blend changes (6041 rows before 2023;
  `away_pass_yards` moves up to 153 yards). Week 1 therefore moves within noise, as the table shows.
  The right check is that the evaluated rows are unchanged, and they are.
- The rebuild picked up the 2026 opener (NE 10, SEA 13; 7261 completed rows). Without the cut to
  `<= 2025`, `--eval-last-n-seasons 3` would have scored 2024-2026 on the on arm.

### Landed alongside (tooling)

- Walk-forward runs log one line per finished week with elapsed and remaining time, and are
  resumable: each finished week is checkpointed under `models/wf_checkpoints/<fingerprint>/`
  (data, config, modelling source, library versions), and re-running an identical command restores
  it. A test pins that a resumed run equals an uninterrupted one exactly. Wired into
  `walk_forward_backtest.py`, `wf_compare.py`, `golden_command.py`, `weekly_run.py`, and
  `betting_pipeline.py`. The on-arm relaunch in this milestone restored its first week from a
  checkpoint after a deliberate stop.
- Operational lessons, now in `AGENTS.md`: two concurrent from-week-1 walk-forwards each burned
  more than a whole solo run's CPU (42 CPU-hours) without finishing and were stopped; under
  unrelated load a week took `730s` with the default OpenMP wait policy and `185s` with
  `OMP_WAIT_POLICY=PASSIVE`; on an idle machine the default was faster (`75s` against `~142s`).

### `games_played` as evidence (2026-09-17, version `0.11.0`; follow-up closed)

The follow-up asked for an effective-games column (`games + K * (1 - WEEK1_REGRESSION_FACTOR)`)
or a `stat_prior_weight`, because `games_played` published `17` for a week-1 fallback row and `1`
for a blended week-2 row. Investigation replaced that plan:

- **Neither proposed column can change an XGBoost fit.** `games + K * (1 - f)` is `games + 2.667`,
  an affine shift; `K / (games + K)` is strictly decreasing in `games`. Both are monotone
  transforms of the count, and axis-aligned trees are invariant to monotone transforms.
- **The model already had a correct counter.** `away_/home_strength_games_played` (plus their
  diff) publish the team's completed games per side, `0` in week 1, no nulls, 1999-2026, and all
  three were already in the trained feature list.
- **The defect was the column itself.** `away_games_played` matched the strength counter in 6845
  of 6848 weeks-2+ rows overall, and in 759 of 759 rows of the 2023-2025 weeks-3-18 evaluation
  window. It differed only on prior-season fallback rows, where it published the previous
  season's total (`17`, or `8` / `16` for the postponed first games of JAX 2001, JAX 2002 and
  MIA 2017) beside the `wins = 0, losses = 0, ties = 0` on the same row.
- **Cause: a join name collision.** The season-to-date stat frame and the record features both
  produce `games_played`; `merge_schedule_with_team_stats` ran first, so the records join
  suffixed the record values to `*_right` and final column selection dropped them. The column
  declared in `constants.RECORD_FEATURE_COLUMNS` was never a record feature. The collision was
  confined to this one name: `wins`, `losses`, `ties` and `win_pct` all carried record values.
- **`home_games_played` was already pruned**, so only the away side reached the model.

Landed: the stat-frame copies are dropped before the records join (`_RECORD_OWNED_STAT_COLUMNS`),
so the published column is `wins + losses + ties`; and both sides joined
`constants.PRUNED_FEATURE_COLUMNS`, since the strength counter already carries the fact.

Build: cached ETL rebuild at 22:28 (`7278` rows, `519` columns, `8bacad41...`); against the
pre-change build (`0cecc2e3...`, kept in `data/backup_pre_m49_games_played/`) exactly two columns
moved, `away_games_played` and `home_games_played`, max difference `17.0`. The extra row is
2026 week 2 DET at BUF, which finished during the rebuild. Leakage audit `483` features, `0`
flags (`models/audit_m49_games_played/leakage_audit.json`).

Walk-forward: `models/wf_m49_gp_2023_2025_pruned/` (`482` features, checkpoints
`85760def37b42ddcd594`) on `data/completed_games_ml.m49_through_2025.csv` (`7261` rows,
`07971269...`), benchmark config from week 1. The before arm is Milestone 53's
`models/wf_qbsched_2023_2025_off/` (`483` features, checkpoints `02a3a730da026668dffe`): same
config, same 816 games, and its cut differs only in the six lens columns it had disabled plus the
two treatment columns. Windows are per-game from the fold checkpoints; the scorer reproduces the
53.6 "off" table exactly in all sixteen cells, once a tied game counts as an incorrect pick as
the pipeline counts it.

| window | games | Brier after / before | log loss after / before | pick acc after / before | margin MAE after / before |
| --- | --- | --- | --- | --- | --- |
| week 1 only | 48 | `0.2024` / `0.2051` | `0.5945` / `0.5996` | `0.7917` / `0.7708` | `9.0399` / `8.9283` |
| week 2 only | 48 | `0.2282` / `0.2275` | `0.6473` / `0.6457` | `0.6042` / `0.6667` | `8.4502` / `8.5291` |
| weeks 3-18 | 720 | `0.2324` / `0.2282` | `0.7612` / `0.7551` | `0.6778` / `0.6903` | `9.9647` / `9.9708` |
| all weeks | 816 | `0.2304` / `0.2268` | `0.7447` / `0.7395` | `0.6801` / `0.6936` | `9.8212` / `9.8246` |

Paired bootstrap over games (5000 resamples, seed 0), after minus before. Week 1: Brier `-0.0026`
`[-0.0108, +0.0053]`, log loss `-0.0052` `[-0.0220, +0.0111]`, pick accuracy `+0.0208`
`[-0.0625, +0.1042]`. Weeks 3-18: Brier `+0.0042` `[-0.0010, +0.0094]`, log loss `+0.0061`
`[-0.0180, +0.0303]`, pick accuracy `-0.0125` `[-0.0278, +0.0028]`, margin MAE `-0.0061`
`[-0.1081, +0.1012]`. Every interval covers zero.

Interpretation: the change helps in the one window where the column lied (week 1 improves on
Brier, log loss and pick accuracy) and is informationally neutral everywhere else, because in
weeks 3-18 the removed column was value-identical to a retained one in all 759 rows. Dropping a
duplicate cannot remove information, so the weeks-3-18 drift is a column-sampling artifact:
with `colsample_bytree = 0.6098`, one fact held in two columns reaches a given tree with
probability `1 - 0.39**2 = 0.85`, against `0.61` when it is held in one. The published dataset is
now internally consistent, which the metrics do not measure and which was the point of the fix.

#### Review note (2026-09-18 audit)

The fix above is correct: in the `0.11.0` build `away_/home_games_played` equals
`wins + losses + ties` in all `7278` rows and is `0` in every week-1 row. Two qualifications to
the record, from `models/feature_audit_2026_09_18/`:

- "Agreed in 6845 of 6848 weeks-2+ rows" compared the old stat-frame count with
  `strength_games_played`, and both are built on the nflverse team-stats skeleton, which is
  missing Jacksonville's eight home games in 2001 and 2002 (upstream gap, verified live). Against
  the corrected record count the strength counter disagrees on `58` away and `54` home rows,
  almost all JAX 2001-2002 (`7` at week 17 against `15`) plus one-game gaps for six 1999-2000
  team-seasons. The pruning decision stands, but `strength_games_played` is not a clean count for
  those rows; the skeleton fix is task 59.3 in `TODO.md`.
- The weeks-3-18 drift in the table above (Brier `+0.0042`) is Platt-calibration noise rather
  than a `colsample_bytree` effect: rescored through the deterministic map
  `Phi(margin / SCORE_DIFF_STD_DEV)` on the same checkpoints, after-minus-before is `+0.0003`
  `[-0.0022, +0.0028]` (`rescored_arms.json`). See Milestone 59 in `TODO.md`.

---

## Milestone 46 - Weekly schedule-adjusted team strength

Completed 2026-09-09.

Published a leakage-safe, pre-week schedule-adjusted offense/defense/special-teams strength per
team, plus schedule strength for games played and remaining in both the ridge form and the one-hop
head-to-head-excluded form. **This is the first family in this workstream to improve the primary
selection metric**: Brier and log loss both improve with the group on.

### What landed

- `nfl_predictor/utils/polars/adjusted_strength.py`: simultaneous ridge (`solve_team_ridge`) with
  one offense and one defense coefficient per team plus a shared home-field term, centered
  independently per side; `solve_srs`; an offline `tune_ridge_lambda`; `build_team_design_matrix`.
  Ported from the read-only `nfl-sos-ratings` reference and verified to reproduce it **exactly**
  (max absolute difference `0.0` on both rating blocks, identical home-field term, identical
  tuner output) on the same inputs. The port additionally drops null rows before solving, which
  the reference does not.
- `nfl_predictor/utils/polars/schedule_strength.py`: `sos_played_adj` / `sos_remaining_adj` from
  opponents' pre-week composite, and `sos_played_raw`, the one-hop companion that profiles each
  faced opponent from only its games against the rest of the league, excluding every head-to-head
  game with the subject. Equal weight per unique opponent.
- `nfl_predictor/utils/polars/strength_snapshot.py`: the weekly snapshot builder. Frozen
  `STRENGTH_RIDGE_LAMBDA = 10.0`, `PRIOR_BLEND_GAMES = 4.0`, composite weights taken from the
  `nfl-sos-ratings` published team composite.
- `is_home` on the play-by-play team-game frame from `posteam_type`, kept out of every count and
  stat list and added to `EXCLUDE_FROM_OPPONENT_STATS`. Verified against the 2024 schedule:
  544 of 544 team-games agree, and season-to-date aggregation drops it, so it never reaches the
  published schema.
- 33 published columns (11 stats x `away_`/`home_`/`_diff`), schema `465` -> `498`, ablatable as
  the `strength` feature group. `--no-strength-prior-blend` ablates the early-season prior at ETL
  time.

### Ridge penalty provenance

`tune_ridge_lambda` (deterministic 5-fold CV over `logspace(-6, 2, 17)`) was run on 128 real
pre-week snapshots: seasons 2005, 2010, 2015, 2019, 2021, 2022, 2023, 2024 at week cutoffs 3, 5, 7,
9, 12, 14, 16, 18. The median selected penalty is `10.0` at **every** cutoff, early weeks included.

Known property, recorded in the module: a penalty this size relative to per-snap EPA (~0.0x)
recovers roughly 30% of true coefficient magnitude at 4 games per team, rising to about 50% by 17.
Ordering is essentially unaffected, but the raw `adj_*` columns therefore drift in scale across a
season while `adj_strength_composite` (standardized within each snapshot) does not. The importance
diagnostic below is consistent with this.

### Walk-forward (2023-2025, 720 games, 16 weeks per season)

All four arms were run on one dataset build and one code version. Reports are on disk under
`models/wf_strength_2023_2025_{both_on,prior_off,strength_off,both_off}/`.

| metric | strength on, prior on | strength on, prior off | strength off | both off |
| --- | --- | --- | --- | --- |
| Brier | **0.2277** | **0.2277** | 0.2312 | 0.2320 |
| log loss | **0.7431** | 0.7492 | 0.7495 | 0.7493 |
| pick accuracy | **0.6958** | 0.6847 | 0.6819 | 0.6736 |
| margin MAE | 9.9006 | 9.8838 | **9.8698** | 9.9772 |
| total MAE | 10.1074 | 10.1043 | 10.1025 | **10.0823** |
| reliability ECE | 0.1321 | 0.1315 | 0.1430 | **0.1237** |

Per season (Brier / log loss / pick accuracy / margin MAE):

| season | strength on, prior on | strength on, prior off | strength off | both off |
| --- | --- | --- | --- | --- |
| 2023 | 0.2457 / 0.8173 / 0.6958 / 10.3012 | 0.2427 / 0.7984 / 0.6667 / 10.2293 | 0.2439 / 0.7973 / 0.6875 / 10.1259 | 0.2471 / 0.7950 / 0.6625 / 10.3106 |
| 2024 | 0.1901 / 0.6647 / 0.7375 / 9.3675 | 0.1902 / 0.6499 / 0.7417 / 9.3822 | 0.1925 / 0.6719 / 0.7333 / 9.3832 | 0.1960 / 0.6798 / 0.7208 / 9.5095 |
| 2025 | 0.2474 / 0.7473 / 0.6542 / 10.0332 | 0.2502 / 0.7992 / 0.6458 / 10.0398 | 0.2572 / 0.7794 / 0.6250 / 10.1004 | 0.2530 / 0.7732 / 0.6375 / 10.1116 |

**Did the hypothesis hold?** Separately for the two things the milestone set out to test:

- **The opponent adjustment: yes, on the primary metric.** Brier improves to `0.2277` from `0.2312`
  with the group off and `0.2320` with both groups off; log loss to `0.7431` from `0.7495` /
  `0.7493`; pick accuracy gains 2.2 points over the both-off baseline. Brier improves in 2 of 3
  seasons against the strength-off arm (2024 and 2025; 2023 is slightly worse). This is the first
  family in the workstream to move the primary metric in the right direction.
- **The prior-carrying early-season blend: not on Brier.** Prior on and prior off are identical to
  four decimals (`0.2277`). The blend earns its place only on log loss (`0.7431` vs `0.7492`) and
  pick accuracy (`0.6958` vs `0.6847`), and it is neutral on ECE. It is kept as the default on that
  basis, but it is the weakest-supported part of the milestone and the ablation switch stays.

**What did not improve.** Margin MAE is worse with the group on than with it off (`9.9006` vs
`9.8698`), and ECE is worse than the both-off arm (`0.1321` vs `0.1237`). The gain is in probability
*ranking*, not in sharper point estimates or better-calibrated probabilities.

**Where the gain comes from.** Gain-based importance over 533 model features puts
`adj_strength_composite_diff` **6th** and `adj_srs_diff` **7th**, behind only the three market
columns and the two Elo diffs. The pass/rush by offense/defense decomposition ranks far lower
(median 183). So the win comes from the aggregate adjusted rating, **not** from the decomposition
that was half the stated rationale for the milestone. `strength_games_played_diff` has a gain of
exactly `0.0` and is dead.

### Validation

- ETL rebuild `1999-2026`: `480s` (`strength_features` adds about `6s` per season). Dataset is
  `7260` rows x `498` columns covering `1999-2025`; `predict/week_01_games_to_predict.csv` is
  `16` rows.
- Leakage audit passed on the refreshed dataset: `463` features, `7260` rows, `0` failures,
  `0` warnings, `0` flagged columns.
- Gates green: `548 passed`, coverage `90.8%`, ruff format/check, pyright, ty, markdownlint and
  `uv lock --check` all clean.
- Null rates, all by design: the four `adj_*` plus `adj_srs` and `st_rating` are null on the same
  `4` of `7260` rows as the play-by-play family; `sos_played_adj` is null in week 1 only;
  `sos_played_raw` is null through **week 2**, because a week-2 opponent's only prior game is the
  one against the subject and the head-to-head exclusion removes it. `sos_remaining_adj` is null
  on playoff rows, which have no remaining regular-season games.
- Sanity check (2024, pre-week-18): top five by composite BAL, DET, PHI, BUF, GB; bottom five TEN,
  NYG, JAX, NE, CAR. Spearman against current-season point differential `0.966` for the composite
  and `0.987` for `adj_srs`, so the snapshot ranks on the current season rather than prior ones.
- `sos_played_adj` vs `sos_played_raw` at 2024 week 18: Spearman `0.894`, sharing four of the top
  five hardest schedules (SF, LAR, TB, BAL). Neither looks wrong; they are different lenses.
- Leakage tests cover all three branches (regular season, playoff, Week 1) for both the
  play-by-play and the strength families, and each was **mutation-verified**: breaking the
  matching cutoff makes the matching test fail.

### Defects found and fixed during the milestone

- **`sos_remaining_adj` leaked the postseason bracket into regular-season rows.** The remaining
  lens averaged the whole remaining schedule, so which playoff games a team would play - an
  outcome of the season being predicted - reached its week-`N` features. Reproduced: the same
  regular season with a weak vs a strong playoff opponent moved a week-2 value from `1.0` to
  `4.0`. Both lenses are now restricted to the regular season. Found by independent review; the
  original leakage tests missed it because they perturbed play data only, never schedule
  structure.
- **Pre-kickoff Week 1 published nothing.** The snapshot drew its team universe from games already
  played, so a new season with a published schedule and no games produced zero rows: all 33
  strength columns were null across the live 2026 Week-1 slate, the exact week the prior blend
  exists to serve, and a train/serve skew against every historical Week-1 training row. The
  universe now comes from the schedule. The module docstring had already claimed this behavior,
  so the code did not match its own contract.
- **`NaN` responses poisoned every team's rating.** `NaN` is not null, so `drop_nulls` let one bad
  cell reach the normal equations and return `NaN` for all 32 teams rather than for the offending
  row. Non-finite values are now filtered alongside nulls.
- `is_home` was missing from the null-fill used when no play-by-play exists at all, so the
  invariant-schema claim did not hold for that column.
- A weak playoff leakage test: the playoff games sat *after* the target week, so the cutoff never
  mattered and mutating it did not fail the test. Rewritten to place them before the target week.

---

## Milestone 45 - Play-by-play foundation + per-snap team EPA families

Completed 2026-09-09.

Brought nflreadpy play-by-play into the Polars ETL with per-season Parquet caching, published a
per-snap EPA / success / explosive / special-teams feature family for every matchup, and measured
it under walk-forward with a dedicated ablation switch.

### What landed

- Cached PBP loader (`loaders.load_pbp`): one season at a time, guarded selection of
  `constants.PBP_COLUMNS`, regular-season filter, team normalization, and a
  `pbp_<season>_<reg|all>.parquet` cache. Historical failures raise; current-season failures
  degrade to cache or continue.
- New `nfl_predictor/utils/polars/pbp.py`: `aggregate_pbp_team_game_stats` produces one row per
  `(season, week, team_abbr, opponent_abbr)` of counts and sums only, plus the situational counts
  formerly in the unused `loaders.aggregate_pbp_stats` (now removed).
- 25 published stats (`constants.PBP_STATS`) derived in `_compute_pbp_derived_metrics` as ratios
  of season-to-date sums. Final schema grew from 384 to 465 columns (75 play-by-play + 6 for the
  newly published `rushing_epa`).
- `--disable-feature-groups` on `walk_forward_backtest.py` and `wf_compare.py`, resolved through
  `constants.FEATURE_GROUP_COLUMN_MARKERS`; `WalkForwardConfig.disabled_feature_groups` is
  authoritative inside `run_walk_forward_backtest` itself.

### Validation

- ETL rebuild `1999-2026`: 1,225,182 regular-season plays, 13,928 team-game records,
  `collect_all_data` 316s (play-by-play load 0.5s warm / ~16s cold for 27 seasons, aggregation
  0.35s). `completed_games_ml.csv` = 7260 rows x 465 columns covering `1999-2025`;
  `predict/week_01_games_to_predict.csv` = 16 rows.
- Season 2026 play-by-play is not published pre-kickoff; the loader degraded with a warning and the
  run completed.
- Leakage audit passed: 430 features, 7260 rows, 0 failures, 0 warnings, 0 flagged columns.
- Null rate for the family is 0 for every season except 4 rows of 7260 (0.055%): the 2002 Texans'
  first game and three 1999 games where a team had no prior in-season game and no 1998 season is
  loaded. Those emit nulls by design.
- League means are era-appropriate: EPA per dropback +0.0017 (1999-2000) rising to +0.0495
  (2020-2025); success rate 0.389 rising to 0.438; early-down pass rate 0.511 rising to 0.543;
  ~62-65 offensive snaps per game throughout.
- Allowed columns equal the opponent's offensive columns exactly on all 544 real 2024 team-games.

### Walk-forward (2023-2025, 720 games, 16 weeks per season)

| metric | PBP on | PBP off | reference |
| --- | --- | --- | --- |
| Brier | 0.2317 | 0.2314 | 0.2312 |
| log loss | 0.7455 | 0.7440 | 0.7352 |
| pick accuracy | 0.6778 | 0.6708 | 0.6833 |
| margin MAE | 9.9178 | 9.9977 | 9.8954 |
| total MAE | 10.1295 | 10.1164 | 10.1021 |
| reliability ECE | 0.1244 | 0.1269 | 0.1308 |

The "off" arm dropped exactly 75 columns. The original reports were written to a temporary
directory and lost; a 2026-09-09 review re-ran both arms with the default config and reproduced
every number above to four decimals (per season, PBP on vs off: 2023 Brier `0.2431` vs `0.2470`,
2024 `0.1946` vs `0.1918`, 2025 `0.2572` vs `0.2554`; margin MAE `10.2188` vs `10.3494`, `9.4179`
vs `9.4183`, `10.1168` vs `10.2252`). Reports: `models/review_wf_2023_2025_pbp_off/` and
`models/review_wf_2023_2025_pbp_on/`. Per season, PBP-on has the better margin MAE in 3 of 3
seasons (-0.131, -0.000, -0.108) but the better Brier in only 1 of 3. **The family is not a win on
the primary selection metric**: Brier and log loss are marginally worse with it on. It improves
margin MAE consistently and calibration slightly.

Two controls were run to interpret the gap against the recorded reference:

- The reference default config with the group dropped reproduced the "off" arm exactly, so
  `--xgb-tree-method hist` accounts for none of the difference.
- **The recorded reference is not reproducible on this machine.** Re-running the default config
  against the untouched pre-change dataset (`7260` rows x `384` columns, backed up before the
  rebuild) gives Brier `0.2300`, log loss `0.7501`, pick accuracy `0.6833`, margin MAE `9.9705`,
  total MAE `10.1229`, ECE `0.1331` - a *larger* log-loss gap from the recorded `0.7352` than the
  rebuilt dataset produces. The recorded reference therefore came from a different configuration or
  environment, and this milestone's dataset changes did not regress it. Only the on/off comparison
  above, run on one dataset with one code version, is a valid comparison.

### Defects found and fixed during the milestone

- nflreadpy signals an unavailable current season with `ValueError`, not `ConnectionError`, so the
  pre-kickoff degrade path never fired and the ETL aborted. Both directions are now pinned by
  tests.
- nflverse 1999-2000 play-by-play uses an empty string rather than null for a missing possession
  team. Those rows formed phantom team-game groups, duplicating the `(season, week, team_abbr)`
  join key and multiplying `team_stats_df` (1999: 495 -> 526 rows; 2000: 492 -> 526). Snap volumes
  and `games_played` were understated by up to ~24% for those seasons (max `games_played` read 21
  in a 16-game season). Fixed at the source, plus a guard so the join can never multiply rows.
- Two-point conversion tries were counted as dropbacks and carries; the reference excludes them.
- Derived ratios were computed before the Week-1 regression rewrote their components, so the
  fallback published regressed counts alongside unregressed ratios. `recompute_derived_metrics`
  now runs after regression. This also changes the Week-1 values of pre-existing derived metrics
  (`yards_per_point`, `points_per_play`, `penalty_yards_per_penalty` and their variants).

---

## Milestone 44 - Preseason 2026 repo hardening and tooling alignment

Completed 2026-06-13.

- [x] Aligned repo instructions, README guidance, changelog workflow, and helper docs around the
      Ruff-only, `.venv`-explicit toolchain.
- [x] Reconciled `pyproject.toml` metadata for Python 3.14, kept dependency groups in
      `pyproject.toml`, and confirmed `uv.lock` as the environment source of truth.
- [x] Kept both Pyright and Ty as mandatory gates, with minimal checked-in `tool.ty` settings to pin
      the repo venv and validated source roots.
- [x] Enforced the preseason coverage floor at `90%` in `pyproject.toml` and raised the suite to
      `406 passed` / `90.01%` coverage.
- [x] Kept the top-level `README.md` as the canonical documentation surface; nested `ml` and
      `reporting` README files remain unnecessary until those subsystems outgrow it.
- [x] Validation and release workflows are both checked in, and the clean-checkout
      `scripts/betting_pipeline.py --dry-run` regression remains covered.
- [x] The repo is back to a season-ready baseline and roadmap work resumes at Milestone 39.

---

## June 2026 maintenance snapshot

- [x] Rebuilt the local `.venv` on Python 3.14.6 and bumped the project version to `0.2.0`.
- [x] Refreshed pinned dependencies, added a direct `pyyaml` dependency, and added
      `update_requirements.sh` for repeatable dependency refreshes.
- [x] Shifted the active toolchain baseline to Ruff, Pyright, and Ty.
- [x] Removed duplicate TODO entries that were already completed under Milestones 23.5 and 23.6.
- [x] Migrated dependency management to `pyproject.toml` plus `uv.lock` and removed the legacy
      requirements files.
- [x] Consolidated agent instructions into `AGENTS.md` and removed the duplicate
      `.github/copilot-instructions.md` file.

---

## Completed core milestones (0-11)

> These items were completed and verified in prior work. They are archived here to keep `TODO.md`
> focused on active work.

### Milestone 0 - Repo scan & plan

- [x] Identify ML entrypoints (train, predict, backtest scripts).
- [x] Identify where margin/total and win prob calibration live.
- [x] List current artifact outputs and missing metadata.
- [x] Confirm where ML datasets and week prediction inputs are produced.

### Milestone 1 - Canonical Margin/Total pipeline

- [x] Margin/total modeling is the primary path.
- [x] Score derivation is stable and unit-tested.
- [x] Outputs include margin/total and derived scores.

### Milestone 2 - Preprocessing cleanup

- [x] XGBoost path avoids scaling and avoids accidental densification.
- [x] Missing values are handled intentionally.

### Milestone 3 - Training improvements

- [x] Early stopping is enabled.
- [x] `eval_metric` aligns to the optimization target.
- [x] Parallelism is configurable.
- [x] Training config is serialized into metadata.

### Milestone 4 - Walk-forward evaluation

- [x] Walk-forward backtest exists and is time-aware.
- [x] Per-week and per-season metrics are emitted.

### Milestone 5 - Probability calibration + diagnostics

- [x] Calibration options are implemented.
- [x] Brier/log loss and reliability summaries are reported.

### Milestone 6 - Quantile intervals

- [x] Margin and total include p10/p50/p90 outputs.
- [x] Interval columns exist and are validated.

### Milestone 7 - Market transforms + anchoring

- [x] Market transforms are explicit and configurable.
- [x] Market anchoring is supported.
- [x] Market probability blending/clamping exists as configured.

### Milestone 8 - Leakage audit

- [x] Leakage audit mode exists and emits a JSON report.
- [x] Tests verify detection of leaked columns.

### Milestone 9 - Artifact contract + metadata

- [x] Run directories include model + metadata + metrics report.
- [x] Artifacts are loadable without hidden state.

### Milestone 10 - Golden command entrypoint

- [x] One command runs train + backtest + weekly predictions and writes artifacts.

### Milestone 11 - Dependency pinning + documentation

- [x] ML dependencies are pinned.
- [x] CPU-only path works and is documented.

---

## Completed milestones (12-19)

### Milestone 12 - Documentation + repository cleanup (Polars-only narrative)

- [x] Remove documentation references to deprecated data collection and utility modules.
- [x] Ensure all docs describe `nfl_predictor/data_collection.py` as the authoritative ETL
      entrypoint.
- [x] Add a short "Data sources + missing data" section describing fallbacks and season coverage
      limits.

Acceptance:

- [x] Docs reference only the Polars+nflreadpy pipeline and current ML entrypoints.

### Milestone 13 - constants.py cleanup and organization

- [x] Audit `nfl_predictor/constants.py` for unused constants and remove them.
- [x] Group constants into clear sections (paths, season/week rules, team mappings, feature names,
      defaults).
- [x] Ensure schema/feature lists are centralized and used everywhere (no hard-coded columns).

Tests:

- [x] Team alias mapping resolves to canonical abbreviations.
- [x] Schema lists contain no duplicates.
- [x] Required output columns exist in the ML datasets.

Acceptance:

- [x] `constants.py` is organized, minimal, and referenced consistently across ETL/ML/docs.

### Milestone 14 - Missing data handling across seasons

- [x] Inventory sources with limited historical coverage (injuries, markets, etc.).
- [x] Define a per-feature-group missing-data policy: null, default, or carry-forward.
- [x] Implement ETL fallbacks so output schema is invariant across seasons.
- [x] Ensure ML preprocessing handles nulls explicitly and logs fallback usage counts.

Tests:

- [x] ETL produces the same columns for a season with missing sources and one without.
- [x] Model train/predict completes when market fields are null.
- [x] Fallback counters appear in metrics/report outputs.

Acceptance:

- [x] Pipeline and ML runs succeed across the full historical range with consistent schema.

### Milestone 15 - Season-to-date record features (W-L-T, division, conference)

- [x] Implement record features for away and home teams (prefix columns `away_` and `home_`).
- [x] Record columns are defined in `constants.py` and included in the ML feature range.

Tests:

- [x] Computed season-to-date records match known records for a small fixture season/week range.
- [x] Divisional records reconcile with overall when a team's prior games are divisional.
- [x] Week 1 records are zero for all teams.

Acceptance:

- [x] Datasets include record features and they are available for training and prediction.

### Milestone 16 - Divisional rivalry feature

- [x] Add a `is_divisional_matchup` feature for each game.
- [x] Implement using a division mapping table in `constants.py`.
- [x] Ensure this applies to all seasons and teams.

Tests:

- [x] Known divisional pairings are flagged correctly.
- [x] Cross-division pairings are not flagged.

Acceptance:

- [x] All game rows contain the divisional indicator and it is stable across seasons.

### Milestone 17 - Lookahead / trap indicators

- [x] Build next-week opponent features using the schedule.
- [x] Add per-team lookahead features and join to games for away/home teams.

Tests:

- [x] Next-week opponent lookup is correct for a fixed season/week range.
- [x] Missing next-week opponent (end of season) yields null/default.

Acceptance:

- [x] Lookahead features exist in the ML dataset for all games with defined fallbacks.

### Milestone 18 - Motivational asymmetry features

- [x] Create a playoff-incentive feature set computed from standings and tiebreak proxies.
- [x] Integrate into ETL as season-to-date features available prior to each game.

Tests:

- [x] Incentive state features do not use future games.
- [x] Motivation feature join is schema-invariant when schedule scores are missing.

Acceptance:

- [x] Motivation/standings proxy features are available for all games without leakage.

---

## Completed optional enhancements

- [x] Realistic score post-processing for display outputs.
- [x] Market-only model removed when anchoring sufficed.
- [x] Blending weights constrained where applicable.
- [x] Interval coverage diagnostics implemented.

---

### Milestone 19 - Blocked/time-series cross-validation for tuning

- [x] Implement blocked CV at the season-week level for hyperparameter tuning and model selection.
- [x] Ensure folds are strictly time-ordered (train < validation).
- [x] Integrate CV into Optuna objectives so tuning does not overfit a single season holdout.
- [x] Report CV mean/std metrics in tuning CV summary (stored under `metrics_report.json`).

Tests:

- [x] CV fold generation is strictly time-ordered.
- [x] CV fold generation is deterministic.

Acceptance:

- [x] Optuna tuning evaluates parameters using time-series CV over season-week timepoints.

Primary files:

- [x] `nfl_predictor/ml/ml_model_core.py`
- [x] `tests/test_time_series_cv.py`

### Milestone 20 - Unit tests and code coverage hardening

- [x] `pytest-cov` is configured and coverage is reported by default.
- [x] Coverage threshold is enforced (current floor: 80%).
- [x] Tests exist across ETL joins and feature derivations introduced in prior milestones.

Acceptance:

- [x] `pytest --cov=nfl_predictor --cov-report=term-missing --cov-fail-under=80` passes.

Primary files:

- [x] `setup.cfg`
- [x] `tests/`

### Milestone 21 - Remove deprecated modules from the import surface

- [x] No code or docs reference deprecated modules.
- [x] Compatibility facades import cleanly.

Acceptance:

- [x] The package imports cleanly and no deprecated modules are referenced.

Primary files:

- [x] `tests/test_imports.py`

---

## Completed milestones (22-30)

### Milestone 22 - Repo/tooling alignment (blocking)

Completion note: Setup and tooling workflow verified; README and config alignment complete.

- [x] Update `README.md` setup instructions to use the pinned requirements workflow:
  - install from `requirements.txt` + `requirements-dev.txt`
  - use `--no-deps` for editable installs to avoid unpinned dependency drift
  - document how to regenerate pins (`uv pip compile`)
- [x] Confirm dev requirements include: `black`, `ruff`, `pytest`, `pytest-cov` (and any other test
      plugins required by `pyproject.toml` addopts).
- [x] Fix minor config gotchas:
  - Ruff isort config should not treat `__main__` as a third-party package.
  - Coverage `exclude_lines` should match `if __name__ == "__main__":` exactly.
- [x] Confirm `.gitignore` covers run artifacts (models, optuna db, caches) and that `data/` being
      ignored is intentional and documented.

Acceptance:

- [x] `ruff check .` and `black --check .` pass in a clean environment.
- [x] `python -m pytest` passes (and `python -m pytest --no-cov` works as documented).
- [x] README instructions work end-to-end on a clean machine.

### Milestone 23 - Canonical training + validation methodology (the "source of truth")

Completion note: Walk-forward evaluation protocol and metrics schema standardized.

- [x] Define the **primary evaluation lens** for the project:
  - Walk-forward (rolling-origin) over multiple seasons is authoritative.
  - Season-blocked CV exists mainly for hyperparameter tuning.
- [x] Standardize the **outer evaluation window**:
  - Default: last N seasons (configurable), regular season only by default.
  - Explicit handling for incomplete current season and postseason evaluation.
- [x] Standardize the **inner calibration window**:
  - Time-aware calibration weeks (e.g., last K weeks before prediction week) and/or calibration
    seasons.
  - Minimum sample size rules (see Milestone 24).
- [x] Decide (and document) the **selection hierarchy** for "best model":
  - Primary: probability quality (Brier, log loss, reliability).
  - Secondary: confidence pool expected points (and stability across weeks).
  - Tertiary: margin/total MAE (and market-relative residual MAE if anchoring).
- [x] Produce a single **metrics report schema** that always includes:
  - per-week, per-season, and overall aggregates
  - mean + variance across folds
  - market-relative metrics when market is present

Acceptance:

- [x] There is one "blessed" evaluation command (or script) that reproduces the reported metrics.
- [x] A config sweep (Milestones 24/25) can run under this protocol without ad hoc code.

### Milestone 23.5 - Reporting pipeline correctness + schema safety (blocking)

Completion note: Reporting scripts now fail fast on schema issues and apply calibration correctly.

#### Tasks (Milestone 23.5)

- [x] **Fix REG-only consistency in record computation**
  - Update `_load_current_records()` to explicitly filter `game_type == "REG"` before computing
    wins/losses/ties.
  - Add or update a unit test that includes both REG and POST games (same season/week) and verifies
    that only REG games affect the computed record.

  Acceptance:
  - [x] Given mixed REG/POST inputs, computed records exactly match REG-only results.

- [x] **Fail fast when required ML feature columns are missing**
  - Replace silent column-dropping logic with explicit validation:
    - Compute `missing_required = set(spec.feature_columns) - set(available_cols)`
    - If non-empty, raise a `ValueError` listing missing columns (truncate list if long).
  - Add a unit test that constructs a minimal ML dataset missing at least one required feature and
    asserts that a clear, informative error is raised.

  Acceptance:
  - [x] The script refuses to run when required feature columns are missing.
  - [x] Error messages name missing columns and indicate how many are missing.

- [x] **Apply win-prob calibration consistently for `ScoreModel`**
  - Update the `ScoreModel` path in `scripts/power_rankings.py` so that:
    - If a calibrator is present, win probabilities are produced via the calibrated path (e.g.,
      `predict_home_win_prob(margin, calibrator)`).
    - If no calibrator is intended, this behavior is explicit and documented in code.
  - Add a unit test that proves calibration is applied when a non-identity calibrator exists.

  Acceptance:
  - [x] `ScoreModel` probabilities change appropriately when a calibrator is attached.
  - [x] Behavior matches `margin_total` and `blended_margin_total` semantics.

- [x] **Add minimal runtime diagnostics**
  - Log (INFO-level, single-line):
    - number of past games used in ratings fit
    - number of future games used
    - effective `ratings_min_season` value
  - Ensure logs are stable and suitable for automation/CI logs.

  Acceptance:
  - [x] Running the script prints these diagnostics exactly once per invocation.

### Milestone 23.6 - Operational documentation: weekly pipeline + evaluation rule

Completion note: README documents the weekly workflow and authoritative evaluation rule.

#### Tasks (Milestone 23.6)

- [x] **Add an authoritative "Weekly pipeline" section to `README.md`**
  - Clearly document:
    - data refresh step
    - canonical training/validation step (from Milestone 23)
    - prediction + reporting steps (including power rankings and standings)
  - Specify:
    - where outputs land on disk
    - naming conventions for run folders and artifacts

  Acceptance:
  - [x] A new user can follow the README end-to-end and produce weekly outputs without guessing.

- [x] **Add a single canonical evaluation rule to `README.md`**
  - Explicitly state:
    > "Model selection is based on time-aware walk-forward evaluation; random CV is not
    > authoritative."
  - Reference Milestone 23 outputs as the source-of-truth evaluation.

  Acceptance:
  - [x] The evaluation rule is visible and unambiguous in the README.

### Milestone 24 - Win-prob calibration: choose (and/or auto-choose) the best method

Completion note: Calibration comparison harness added; platt chosen as default.

Options in code today: `none`, `platt`, `isotonic`, `elo`.

- [x] Add a **calibration comparison harness** that evaluates calibration choices under the
      canonical walk-forward protocol (Milestone 23).
  - At minimum: compare Brier, log loss, and reliability.
  - Include pool metrics as tie-breakers.
- [x] Implement **"auto" calibration** (optional but recommended):
  - Use isotonic only when calibration sample size is large enough.
  - Fall back to Platt when calibration data is small/noisy.
  - Always keep an explicit override.
- [x] Add CLI **compatibility alias**: accept `logistic` as a synonym for `platt`.
- [x] Validate that calibrators are trained only on time-appropriate rows.

Acceptance:

- [x] Walk-forward results clearly show which calibration choice is best (and how sensitive it is by
      season/week).
- [x] `--win-prob-calibration logistic` behaves identically to `--win-prob-calibration platt`.

### Milestone 25 - Market integration decisions + correct probability blending

Completion note: Market anchoring and blending validated under walk-forward.

Decide, then enforce, the objectively best usage of market inputs:

- Market as **features**
- Market as **anchoring** (residual modeling)
- Hybrid (anchor + selected transforms)

#### Tasks (Milestone 25)

- [x] Evaluate market as features vs anchoring under the canonical protocol.
- [x] Fix/confirm the market probability source used for blending/clamping:
  - Current: implied prob from moneyline (includes vig).
  - Add: **no-vig** implied probability (normalize home/away to sum to 1).
- [x] Implement/validate **market probability blending** "the right way":
  - Consider blending in **log-odds space** (more stable than linear prob blends).
  - Add clear configuration: source (`raw` vs `novig`), blend method (`prob` vs `logit`), weight,
    and clamp delta.
- [x] Add a small test suite around moneyline->prob and no-vig normalization.

Acceptance:

- [x] The selected market mode (features vs anchor vs hybrid) is chosen via walk-forward.
- [x] Market blending/clamping uses the intended probability definition (raw or no-vig) and is
      unit-tested.

### Milestone 26 - Continuous retraining + weekly orchestration (one command, resumable)

Completion note: Weekly orchestration script added with resumable artifacts.

Goal: a single script to run 1–2x per week that:

1. runs data refresh (`python -m nfl_predictor.data_collection`)
2. re-trains and time-validates the best-known model configuration
3. emits all weekly outputs in a consistent, predictable place

Outputs to include (as available):

- weekly predictions (`*_predictions.csv`)
- confidence pool picks (unique 1..N ranks)
- power rankings for the week
- betting report + optional Excel template
- (optional) projected standings / season win distributions (Milestone 28)

#### Tasks (Milestone 26)

- [x] Create `scripts/weekly_run.py` (or equivalent) that composes existing steps:
  - data collection
  - config selection (Milestones 23–25)
  - tuning (optional)
  - final train
  - prediction + reports
- [x] Make it resumable (like `scripts/betting_pipeline.py`): reuse prior artifacts when inputs
      match.
- [x] Add a config file option (YAML/JSON) to avoid 200-character CLI invocations.

Acceptance:

- [x] One command produces a complete weekly output package from scratch.
- [x] Re-running does not redo expensive work unless inputs or config changed.

### Milestone 27 - Use uncertainty estimates to improve probabilities + confidence ranking

Completion note: Uncertainty-aware win probabilities and ranking path implemented and evaluated.

The repo already produces quantile intervals for margin/total. Use them more directly.

- [x] Derive a per-game uncertainty estimate (e.g., infer σ from p10/p90 width).
- [x] Convert margin + σ into a win probability via a distributional mapping (e.g., normal CDF),
      then optionally calibrate.
- [x] Compare uncertainty-aware probabilities vs current approach via walk-forward.
- [x] Consider uncertainty-aware confidence ranks (e.g., prioritize higher expected points with
      lower upset risk).

Acceptance:

- [x] Walk-forward shows whether uncertainty-aware probabilities improve Brier/log loss and/or pool
      points.

### Milestone 28 - Metric strategy: decide what "better" means (and track it)

Completion note: Metrics hierarchy and diagnostics added to reports.

- [x] Decide which metrics are first-class for model iteration:
  - margin MAE, total MAE
  - Brier, log loss, reliability
  - confidence pool expected/actual points
  - market-relative residual metrics (when market is used)
- [x] Add optional season-level diagnostics:
  - predicted vs actual season win totals (requires projecting remaining games)
  - calibration drift by season/week

Acceptance:

- [x] Metrics are easy to compare across runs (stable JSON schema + summary table).

### Milestone 29 - Hyperparameter optimization (Optuna) hygiene

Completion note: Full Optuna sweep run; artifacts and guardrails captured.

- [x] Run a "full" Optuna sweep for the current best configuration (time-series CV objective).
- [x] Persist best params + study metadata into the run artifacts.
- [x] Add guardrails to prevent accidental tuning on holdout.

Acceptance:

- [x] Optuna results are reproducible and clearly tied to a dataset fingerprint + config.

### Milestone 30 - Feature importance + regularization

Completion note: Feature-importance reports and SHAP script added; pruning/regularization validated
via walk-forward with platt as the best calibration.

- [x] Add a feature-importance report (XGBoost gain/weight) for each trained run.
- [x] Add an optional SHAP analysis script for deeper inspection (keep it optional; do not require
      it for CI).
- [x] Use importance results to:
  - prune noisy/redundant features
  - tune regularization (L1/L2, depth, min_child_weight, etc.)

Acceptance:

- [x] Feature pruning decisions are validated via walk-forward (no "it looked right" commits).

### Milestone 31A - Data collection performance + caching hygiene (blocking)

Completion note: Added nflreadpy caching, profiling toggles, and cache visibility in logs.

Goal: shorten and stabilize data-collection runs while minimizing network calls.

#### Tasks (Milestone 31A)

- [x] Add opt-in timing/profiling logs for data collection (per major step) with a clear toggle.
- [x] Add targeted debug logs around schedule/TeamRankings/ELO/team-stats merges so slow steps are
      visible.
- [x] Audit TeamRankings caching behavior and document the cache hit/miss rules.
- [x] Implement caching for nflreadpy outputs (schedule + team stats) and a clear refresh toggle.
- [x] Document caching and expected run-time behavior in `README.md`.

Acceptance:

- [x] A debug/profiling run prints step timings and shows cache hits.
- [x] A second run reuses cached data without network calls (unless refresh is forced).

### Milestone 31 - Recency + trend features (non-linearity and drift)

Completion note: Trend features and recency weighting shipped with ablation tooling. Walk-forward
ablation shows trend features improve Brier/log loss and margin MAE, while recency weighting with
half-life seasons=2 worsens probability metrics despite a small total-MAE improvement.

Goal: add leakage-safe trend/recency signals plus optional time-weighted training.

#### Feature design + audit

- [x] Inventory existing recency signals (TR last_5/last_10 ratings, lookahead, motivation).
- [x] Finalize minimal trend feature set and confirm they are time-safe.

#### Trend features (time-safe)

- [x] Rating trend: `last_5_games_rating - last_10_games_rating` for away/home + diff.
- [x] Elo trend: `elo_pre - rolling_4wk_mean(elo_pre)` for away/home + diff.
- [x] QB Elo trend: `qb_elo_pre - rolling_4wk_mean(qb_elo_pre)` for away/home + diff.
- [x] Performance trend (select 1-2 stats): recent 4-week mean vs season-to-date mean (scoring
      margin and turnover margin) for away/home + diff.
- [x] Season-phase features: normalized `week_in_season` plus early/mid/late bucket flags.

#### ETL + schema

- [x] Implement rolling aggregates in Polars (per team, per season, prior weeks only).
- [x] Add derived columns to `constants.py` and enforce schema ordering.
- [x] Ensure missing-data policy is consistent for early weeks and short seasons.

#### Recency weighting (exponential half-life)

- [x] Add optional exponential half-life sample-weighting for training + calibration.
- [x] Add CLI/config flags for half-life (weeks or seasons) in training + walk-forward.
- [x] Keep default off and ensure weights are deterministic.

#### Tests

- [x] Unit tests verifying trend features only use prior weeks.
- [x] Unit tests for recency weights (monotonic decay, boundary cases).
- [x] Unit tests for season-phase buckets and normalization.

#### Evaluation

- [x] Walk-forward comparisons with/without trend features and with/without weights.
- [x] Track Brier/log loss first; pool points as tie-breakers; MAE third.

Acceptance:

- [x] New features are leakage-safe and schema-invariant.
- [x] Walk-forward results show a clear improvement or documented tradeoff.

### Milestone 32 - Weather + venue effects (consistent, non-leaky)

Completion note: Stadium metadata features were kept and expanded; weather fields were later removed
after confirming they update post-kickoff.

#### Tasks (Milestone 32)

- [x] Extend stadium metadata beyond city/state (type + altitude).
- [x] Pull historical weather fields from NFLverse schedule data (implemented, later removed).
- [x] Define missing-data policy and enforce invariant schema.
- [x] Add tests for missing-weather fallbacks and schema invariance.

Acceptance:

- [x] Stadium metadata features are maintained; weather features were removed due to leakage risk.

### Milestone 32B - Stadium metadata + venue features (non-leaky)

Completion note: Stadium metadata expanded and wired through ETL/tests with safe defaults.

#### Tasks (Milestone 32B)

- [x] Expand `STADIUMS` to include `name` and `elevation` (and keep city/state).
- [x] Update stadium feature derivation to use the new `STADIUMS` fields and drop any legacy
      altitude map if redundant.
- [x] Keep stadium type/surface features derived from NFLverse schedule fields.
- [x] Add/adjust tests for stadium metadata parsing and safe fallbacks.
- [x] Update README feature list to reflect stadium-only (no weather/ref).

Acceptance:

- [x] Stadium metadata features are present for all games with safe defaults.

### Milestone 33 - Head coach features (if data is robust)

Completion note: Added coach prior record features (career and team-specific) with time-safe
aggregation and leakage tests; walk-forward ablation completed and coach_on retained for
full-feature training.

#### Tasks (Milestone 33)

- [x] Confirm coach coverage via NFLverse schedule fields.
- [x] Add coach prior record features computed strictly to date.
- [x] Add tests that verify no leakage in coach-derived features.

Acceptance:

- [x] Coach features are leakage-safe; walk-forward ablation completed (coach_on retained).

### Milestone 34 - Referee features (if data is robust)

Completion note: Referee features were removed after confirming assignments update post-kickoff.

#### Tasks (Milestone 34)

- [x] Confirm referee coverage via NFLverse schedule fields.
- [x] Implemented referee features (later removed due to post-game updates).

Acceptance:

- [x] Referee features are removed; data is not reliable pre-kickoff.
- [x] Feature ordering/schema remains invariant after removal.

### Milestone 35 - Pandas to Polars audit/refactor

Completion note: Completed a pandas usage audit; ETL is Polars-first and pandas usage is confined to
ML, reporting, and orchestration layers. No safe non-ML/reporting refactors were identified.

#### Tasks (Milestone 35)

- [x] Inventory pandas usage across the repo and classify by module (ETL vs ML vs reporting).
- [x] Identify pandas usage that can move to Polars safely (none found outside ML/reporting).
- [x] Refactor candidate modules to Polars-first implementations (no safe candidates).
- [x] Document any pandas usage that must remain (e.g., sklearn pipelines, calibration).
- [x] Confirm existing tests remain sufficient since no refactor was required.

Acceptance:

- [x] ETL and feature engineering are fully Polars-first with minimal pandas use.
- [x] Remaining pandas usage is justified and documented.

### Milestone 36 - Data availability guards (nflreadpy + TeamRankings)

Completion note: Added guardrails for nflreadpy/TeamRankings availability with tests.

#### Tasks (Milestone 36)

- [x] Enforce nflreadpy availability (min season >= 1999) in data collection CLI.
- [x] Skip TeamRankings loads for seasons before 2003 and use week 2 as the earliest week in 2003.
- [x] Add unit tests for the guardrails.

Acceptance:

- [x] Data collection fails fast for pre-1999 seasons and skips TR pre-2003 without errors.

### Milestone 37 - Canonical training + validation methodology (the "source of truth")

Completion note: Standardized walk-forward evaluation defaults and reporting, added explicit
handling for incomplete seasons, aligned comparison defaults with training settings, and updated
docs/tests to codify the canonical evaluation protocol.

#### Tasks (Milestone 37)

- [x] Define the **primary evaluation lens** for the project:
  - Walk-forward (rolling-origin) over multiple seasons is authoritative.
  - Align WF early stopping with training defaults (e.g., 30–50 rounds) so evaluation doesn’t favor
    configs tuned under a weaker/faster regime.
  - Season-blocked CV exists mainly for hyperparameter tuning.
- [x] Standardize the **outer evaluation window**:
  - Default: last N seasons (configurable), regular season only by default.
  - Explicit handling for incomplete current season and postseason evaluation.
- [x] Standardize the **inner calibration window**:
  - Time-aware calibration weeks (e.g., last K weeks before prediction week) and/or calibration
    seasons.
  - Minimum sample size rules.
- [x] Decide (and document) the **selection hierarchy** for “best model”:
  - Primary: probability quality (Brier, log loss, reliability).
  - Secondary: confidence pool expected points (and stability across weeks).
  - Tertiary: margin/total MAE (and market-relative residual MAE if anchoring).
- [x] Produce a single **metrics report schema** that always includes:
  - per-week, per-season, and overall aggregates
  - mean + variance across folds
  - market-relative metrics when market is present
- [x] Add one clear rule to docs:
  - “We select models using time-aware walk-forward evaluation; random CV is not authoritative."

Acceptance:

- [x] There is one “blessed” evaluation command (script) that reproduces reported metrics.
- [x] A config sweep (Milestones 38/39) can run under this protocol without ad hoc code.

### Milestone 38 - Walk-forward comparison checkpointing + true resumability (blocking)

Completion note: Added per-candidate walk-forward checkpoints, resumable summary artifacts, and
optional per-fold progress logging, with tests and docs updated.

#### Tasks (Milestone 38)

- [x] Identify where WF candidates are enumerated (weekly_run Stage 1) and define stable candidate
      keys.
- [x] Add dataset + run fingerprint helpers for caching/resume decisions.
- [x] Write per-candidate artifacts atomically and skip valid candidates on resume.
- [x] Persist and atomically update a summary table after each candidate.
- [x] Add optional per-fold heartbeat checkpointing.
- [x] Wire checkpointing into Stage 1 with clear progress logging.
- [x] Add unit + integration-ish tests for resume behavior and corrupt handling.
- [x] Update README/AGENTS with resumable WF artifact guidance.

Acceptance:

- [x] Stage 1 WF writes per-candidate artifacts and an aggregated summary table.
- [x] `--resume` continues without recomputing completed candidates.
- [x] A forced kill/restart preserves completed work and finishes correctly after restart.
