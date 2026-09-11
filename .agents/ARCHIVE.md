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
