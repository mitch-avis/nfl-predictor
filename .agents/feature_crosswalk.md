# Cross-Repo Feature Crosswalk and Planning Review

Written 2026-09-09 from a read-only review of `nfl-predictor`, `../nfeloqb`, and
`../nfl-sos-ratings`. This is the launch document for the feature-engineering implementation
sessions. Update it when a shortlist item lands, is rejected, or changes shape.

Verified baseline at time of writing: `412 passed`, coverage `90.03%`, working tree clean,
`data/completed_games_ml.csv` covers `1999-2025` (`7260` rows, `384` columns),
`data/predict/week_01_games_to_predict.csv` holds `16` rows for 2026 Week 1, and
`data/qb_elos.csv` already carries 2026 Week 1 rows.

## 1. Where things live in nfl-predictor (verified from code)

| Layer | Owning module | Notes |
| --- | --- | --- |
| ETL orchestration | `nfl_predictor/data_collection.py` | `collect_all_data` -> `process_season` -> `process_week`; per-week joins |
| Source loaders | `nfl_predictor/utils/polars/loaders.py` | schedule + team stats cached per season as Parquet under `data/cache/nflreadpy`; Elo from `data/qb_elos.csv` |
| Season-to-date aggregation | `nfl_predictor/utils/polars/teamrankings.py` | `aggregate_team_stats_to_week` = mean of every numeric column for `week < target`; playoffs use the full regular season; `_compute_derived_metrics` adds ratios |
| Week-1 fallback | `data_collection.process_week` | previous-season full mean regressed toward league mean by `WEEK1_REGRESSION_FACTOR` (1/3) |
| Opponent mirror | `loaders.add_per_game_opponent_stats` | `opponent_<stat>` is the same-game opponent's stat, so its season-to-date mean is an **allowed** stat, not schedule strength |
| Context features | `nfl_predictor/utils/polars/features.py` | records, divisional, lookahead, motivation, season phase, Elo/QB/stat 4-week trends, coach priors |
| External ratings | `nfl_predictor/utils/polars/teamrankings.py` + `scraping_utils.py` | TeamRankings ratings (predictive, SOS, future SOS, last 5/10, luck) and situational stats; floor 2003 Week 2 |
| Final schema | `nfl_predictor/utils/polars/finalize.py` | metadata, `away_*`, `away_opponent_*`, `home_*`, `home_opponent_*`, `*_diff` (away minus home), lines, results |
| Modeling | `nfl_predictor/ml/` | XGBoost margin/total heads, feature range `away_rest`..`home_moneyline`, market transform/anchor, calibration, quantiles |
| Evaluation | `nfl_predictor/ml/walk_forward.py`, `scripts/walk_forward_backtest.py`, `scripts/wf_compare.py` | one fold per (season, week >= 3); train on everything strictly earlier across all seasons; calibrate on last 4 weeks |
| Power rankings | `nfl_predictor/reporting/power_rankings.py`, `scripts/power_rankings.py`, `scripts/weekly_run.py` | Bradley-Terry fit; see section 2 |

EPA footprint today: `passing_epa` and `passing_cpoe` per game (totals from
`nflreadpy.load_team_stats`), their `opponent_` allowed mirrors, and the four diffs. That is 8 of
384 columns. `rushing_epa`, `receiving_epa`, `passing_air_yards`, and `passing_yac` are loaded and
renamed by `NFLREADPY_TEAM_STATS_MAPPING` but are dropped at `select_final_columns` because they are
not in `constants.NFLREADPY_STATS`. There is no per-snap normalization and no opponent adjustment
of any stat; the only opponent-adjusted strength inputs are Elo (`nfeloqb`) and the TeamRankings
`predictive_rating`.

`loaders.load_pbp` and `loaders.aggregate_pbp_stats` already exist (third/fourth down, red zone,
two-point, total plays) but are not called anywhere in the ETL path. They are a starting point, not
a finished loader: no caching, no column selection, no current-season 404 handling.

## 2. Power rankings: verified behavior and verdict

`scripts/power_rankings.py::_build_games_for_ratings` fits `fit_bradley_terry_ratings` on:

- every completed game in `data/all_data.csv` from `ratings_min_season` onward, and the default is
  `None`, which means every season back to 1999, all weighted equally;
- completed games mapped to fixed targets `0.97 / 0.03` (ties `0.5`) regardless of margin;
- future current-season games mapped to the trained model's home win probability;
- ridge alpha `1.0`, one shared home-field term.

`scripts/weekly_run.py` uses `through_week = predicted_week - 1` and passes `ratings_min_season`
through unchanged, so the default weekly run inherits the all-history fit. For a Week 10 ranking
each team contributes roughly 9 current-season rows against roughly 450 historical rows. The user's
skepticism is correct: this is a franchise-history rating with a small current-season nudge, and it
also blends the model's own forward projections into a "strength now" number.

`scripts/golden_command.py::_build_pregame_power_rankings` is a second, unrelated display artifact
that ranks teams by the model's pregame/postgame ratings. It is not the weekly-run path.

Recommended redesign (details in section 6): make "strength now" a current-season, schedule-adjusted
EPA rating built from games strictly before the target week, reuse the exact same snapshot columns
as ETL features, and keep future-game projections only in the projected-standings tables.

## 3. Feature inventory crosswalk

Sources: `nfl-sos-ratings/nfl_sos_ratings/team_stats.py`, `team_stats_expanded.py`,
`qb_stats.py`, `simultaneous_adjustment.py`, `validation/snapshots.py`, and the published
`data/2025_team_game_logs.parquet` (244 columns) and `data/2025_qb_game_logs.parquet` (52 columns).

| Family | nfl-predictor today | nfl-sos-ratings | Verdict |
| --- | --- | --- | --- |
| Passing volume (att, comp, yds, TD, INT, sacks) | yes (per game) | yes | present |
| Rushing volume | yes (per game) | yes | present |
| Passing EPA total, CPOE | yes (per game) | yes | present, narrow |
| Rushing EPA total | loaded, dropped | yes | trivial add |
| Offensive/defensive snap counts from PBP | no | yes (`scrimmage_snap_expr`) | port (foundation) |
| Per-snap EPA: off pass, off rush, def pass allowed, def rush allowed | no | yes (`*_per_offensive_snap`, `*_allowed_per_defensive_snap`) | port, highest priority |
| EPA per dropback, EPA per carry (+ allowed) | no | yes | port with the above |
| Success rate (overall/pass/rush, + allowed), EPA margin per play | no | yes | port, high |
| Explosive pass/rush rate, stuffed run rate, deep attempt rate | no | yes | port, high |
| Early-down pass rate, pass rate over expected, shotgun/no-huddle | no | yes | port PROE only after floor check |
| Drive efficiency (points/drive, score %, three-and-out, start field position) | no | yes | port, medium |
| Third/fourth down, red zone, two-point (situational) | yes via TeamRankings scrape (2003+) | yes from PBP | replace scrape with PBP (robustness) |
| Turnovers, turnover margin, turnover/takeaway EPA | margin yes; EPA no | yes | keep margin; takeaway EPA optional |
| Penalties | counts + yards | counts, yards, rates, DPI | present, rates optional |
| Special teams | `special_teams_tds` only | ST EPA margin per play, `SaSTR` | port ST EPA margin, medium |
| Schedule-adjusted offense/defense (ridge, simultaneous, with HFA) | no | yes (`solve_team_stat_ridge`) | port as weekly snapshot, highest priority |
| Strength of schedule from EPA ratings (played, remaining) | TeamRankings SOS + future SOS only | `sos` (played-game mean opponent `SaCR`) | port once ridge snapshot exists |
| Opponent profiles excluding head-to-head, `diff_*` surfaces | no | yes (season-level, descriptive) | do not port as-is; redundant once ridge snapshot exists |
| SRS (point-margin simultaneous) | no | yes (`solve_srs`) | cheap companion to the ridge snapshot |
| Team Elo | yes (`elo_pre`, `qb_elo_pre` from `nfeloqb`) | external baseline only | present |
| QB per-dropback EPA, CPOE, sack rate, ANY/A, TD-INT rate | no (only `qb_value_pre`, `qb_elo_pre`) | yes (`qb_*` game logs) | port, high, needs identity bridge |
| QB opponent-adjusted EPA/dropback (`QSaOR`) and faced-defense lens (`QSoS`) | no | yes (dropback-weighted ridge) | phase 2 after raw QB families |
| QB designed rush / scramble split | no | yes | defer |
| QB outcome stats (wins, 4QC, GWD) | records only | descriptive only | reject |
| Composite weights (`SaCR`, `QSaCR` frozen blends) | no | yes | do not port as features; borrow weights for rankings only |
| Pooled `_alltime` companions | no | yes | reject |
| ESPN QBR, PFR, NGS aggregates | no | loaders exist, mostly unimplemented | defer (2006+/2016+ floors, external assets) |
| Receiving display mirrors, time of possession, tackle accounting | no | display only / deferred | reject |

## 4. Prioritized shortlist to port

Each item: what it is, where it comes from, why it might help, data floor, leakage risk,
implementation complexity, and owning layer.

### 4.1 PBP foundation: cached play-by-play loader and team-game aggregation

- What: per-season `nflreadpy.load_pbp` with column selection and a Parquet cache under
  `data/cache/nflreadpy/pbp_<season>.parquet`, then one row per team-game with snap counts and
  the sums needed by every family below.
- Source: `nfl-sos-ratings/pbp_expressions.py` (`scrimmage_snap_expr`, `value_expr`, `rate_expr`)
  and `team_stats.compute_team_game_stats_from_pbp`.
- Why: every high-priority family depends on it; nflreadpy keeps only an in-memory cache, so the
  repo must persist its own.
- Floor: 1999 (matches `MIN_SEASON`). Roughly 45-50k plays per season; select about 45 columns.
- Leakage: none by itself; the team-game rows flow through the existing
  `aggregate_team_stats_to_week` filter (`week < target`).
- Complexity: low-medium. Must mirror the current-season refresh and the non-fatal current-season
  404 behavior already used for team stats (pre-kickoff PBP does not exist yet).
- Layer: ETL (`loaders.py`, new `polars/pbp.py`).

### 4.2 Per-snap team EPA families

- What: `off_pass_epa_per_snap`, `off_rush_epa_per_snap`, `def_pass_epa_allowed_per_snap`,
  `def_rush_epa_allowed_per_snap`, `epa_per_dropback`, `epa_per_carry`, allowed variants, and
  `epa_margin_per_play`.
- Source: `team_stats.py` rate specs.
- Why: the user's stated priority; per-snap EPA is the backbone of the sos ratings and is
  pace-independent, unlike the current per-game EPA totals.
- Floor: 1999.
- Leakage: none when aggregated from prior weeks. Carry sums and snap counts through aggregation
  and compute rates in `_compute_derived_metrics` (ratio of sums), not mean-of-game-rates.
- Complexity: low once 4.1 exists.
- Layer: ETL, plus `constants.py` schema lists and `finalize.py` ordering.

### 4.3 Success, explosive, and early-down families

- What: success rate (all/pass/rush, + allowed), explosive pass/rush rates (+ allowed), stuffed run
  rate, early-down pass rate, deep attempt rate; pass rate over expected only if `xpass` is
  present for the season.
- Source: `team_stats_expanded._aggregate_play_stats` and `_add_offense_ratios`.
- Why: success rate is more stable week to week than EPA and complements it; explosive and stuff
  rates capture distribution shape that EPA averages hide.
- Floor: 1999 for `success`; verify `xpass` availability per season before publishing PROE.
- Leakage: none (prior weeks only).
- Complexity: low-medium.
- Layer: ETL.

### 4.4 Weekly schedule-adjusted team strength (ridge snapshot)

- What: for each (season, week), solve offense, defense, and home-field coefficients
  simultaneously on the per-snap pass and rush EPA responses using only games with `week < N`
  in that season; publish `adj_off_pass_epa_snap`, `adj_off_rush_epa_snap`,
  `adj_def_pass_epa_snap`, `adj_def_rush_epa_snap`, plus a composite and SRS companion.
- Source: `simultaneous_adjustment.solve_team_stat_ridge`, `compute_team_adjusted_stats`, and
  `validation/snapshots.build_team_adjusted_snapshot` (already the leakage-safe form).
- Why: this is the schedule-adjusted "true strength" input the user asked for, and it directly
  fixes the power-rankings design. In the sos walk-forward it beat raw EPA and SRS on next-week
  margin MAE.
- Floor: 1999.
- Leakage: safe if the solve is fed only prior-week rows. Early weeks are ill-posed; use a
  prior-season final snapshot regressed by `WEEK1_REGRESSION_FACTOR` as the Week 1 value and
  shrink toward it through about Week 4 (documented, tested). Playoff weeks use the full regular
  season, matching existing aggregation.
- Complexity: medium. About 500 small NumPy solves for 1999-2025 (65 coefficients, at most ~270
  rows each); a fixed ridge lambda is fine for v1 and avoids per-week tuning noise.
- Layer: ETL (features) and reporting (power rankings consume the same columns).

### 4.5 Schedule strength from adjusted ratings

- What: `sos_played` = mean of opponents' pre-week adjusted composite over games played;
  `sos_remaining` = the same over the remaining regular-season schedule.
- Source: `main._build_team_schedule_strength` in the sos repo (season-level); the weekly form is
  new but trivial once 4.4 exists.
- Why: replaces reliance on scraped TeamRankings SOS and future SOS with an internal, explainable
  measure; the model can weigh raw and adjusted stats together.
- Floor: 1999. Leakage: none if opponent ratings are the same pre-week snapshot.
- Complexity: low after 4.4. Layer: ETL.

### 4.6 QB per-dropback families for the expected starter

- What: season-to-date and rolling `qb_epa_per_dropback`, `qb_cpoe` (2006+), `qb_sack_rate`,
  `qb_any_a`, `qb_td_int_margin_rate`, `qb_dropbacks` for `away_qb` / `home_qb`.
- Source: `qb_stats.compute_qb_game_stats_from_pbp` (grouped by `passer_player_id`), rate
  formulas in the sos README.
- Why: the user wants QB EPA to matter more; today the only QB signal is the nfeloqb value/Elo pair
  and its 4-week trend.
- Floor: 1999 for EPA-based fields, 2006 for CPOE.
- Leakage: none if computed from prior weeks; the row must use the QB the ETL already assigns for
  that game (`away_qb`/`home_qb`, filled for future weeks by `fill_future_qb_data`).
- Complexity: medium-high. Identity bridge: nfeloqb names (`qb1`/`qb2`) to GSIS ids via
  `../nfeloqb/Other Data/meta_data.csv` (`name_id` -> `gsis_id`), then to PBP `passer_player_id`.
  Needs a documented fallback for new starters (league mean or rookie prior) and for QBs who changed
  teams (career-to-date window is safer than team-season).
- Layer: ETL; treat the nfeloqb metadata file as a read-only input copied into `data/`, like
  `qb_elos.csv`.

### 4.7 Special teams EPA margin

- What: `st_epa_margin_per_play` per team-game (kicks, punts, returns, field goals), aggregated
  season-to-date, plus an SRS-style adjusted `st_rating` once 4.4 exists.
- Source: `validation/snapshots.build_special_teams_game_frame_from_pbp` and
  `build_special_teams_rating_snapshot`.
- Why: small but real; in the sos walk-forward adding special teams improved overall MAE.
- Floor: 1999. Leakage: none. Complexity: low (needs the `special` or `special_teams_play` flag).
- Layer: ETL.

### 4.8 Situational stats from PBP replacing the TeamRankings stat scrape

- What: third/fourth down, red zone, two-point rates computed from PBP for both offense and
  defense.
- Source: existing unused `loaders.aggregate_pbp_stats` plus `team_stats_expanded`.
- Why: extends the floor from 2003 to 1999, removes a network dependency for eight columns, and
  makes the invariant schema fully reproducible offline. Keep the TeamRankings ratings scrape.
- Complexity: low-medium (schema swap must keep column names stable or be versioned explicitly).
- Layer: ETL.

## 5. Rejected or deferred

- QB outcome stats (wins, comebacks, game-winning drives): outcome-only, redundant with records.
- Pooled `_alltime` companions and within-season z-scores as model features: keep model inputs in
  raw adjusted units; standardization is for rankings display only.
- Frozen composite weights as features: let XGBoost weight components; reuse the sos weights only
  as the default display composite for power rankings.
- Season-level opponent profiles with head-to-head exclusion and `diff_*` surfaces: the weekly
  ridge snapshot supersedes them and is cheaper.
- ESPN QBR, PFR, NGS: later floors, external release assets, small expected gain.
- Designed-run/scramble QB split, time of possession, tackle accounting, receiving mirrors: display
  value only at this stage.
- Alternative model families and ensembles: explicitly parked by the user.

## 6. Power rankings redesign recommendation

1. Immediate low-risk defaults (small, tested change to `scripts/power_rankings.py` and
   `scripts/weekly_run.py`): default the fit window to the current season plus the previous
   season, weight previous-season rows down (about 0.25), use margin-based targets via
   `margin_to_home_win_prob` instead of `0.97 / 0.03`, and exclude future model-probability rows
   from the strength fit. Keep `--ratings-min-season` and a `--legacy-franchise-fit` flag for the
   old behavior.
2. Target design (after shortlist items 4.1-4.4 land): rank teams by the pre-week schedule-adjusted
   EPA composite from the ETL snapshot (offense, defense, special teams), standardized within the
   season and mapped onto the existing 1-10 and 0-10 scales. Publish the components alongside the
   rank so the ranking is explainable and identical to the model's inputs.
3. Keep projected standings as they are: current record plus model win probabilities for the
   remaining schedule. That table is the right home for forward-looking information.
4. Retire or clearly label `golden_command._build_pregame_power_rankings` so there is one
   canonical ranking artifact.

## 7. Roadmap order and the first implementation session

Order: Milestone 45 (PBP foundation + per-snap EPA/success/explosive/ST families), then 46 (weekly
ridge-adjusted strength + schedule strength), then 43 rewritten (power rankings on the new
snapshot), then 47 (QB per-dropback families), then 48 (PBP situational stats replacing the scrape),
then the deferred 39/40 sweep and 41 residuals. Milestone 42 stays parked.

The first implementation session should deliver Milestone 45 end to end: cached PBP loader, team-game
aggregation, schema constants, invariant-schema finalization, leakage audit on the rebuilt dataset,
and a 2023-2025 walk-forward comparison against the recorded baseline (Brier `0.2312`, log loss
`0.7352`, pick accuracy `0.6833`, margin MAE `9.8954`, total MAE `10.1021`, ECE `0.1308`).

## 8. Guardrails specific to the PBP work

- Select PBP columns explicitly; a first-cut list: `game_id`, `season`, `season_type`, `week`,
  `posteam`, `defteam`, `home_team`, `away_team`, `posteam_type`, `play_type`, `qb_dropback`,
  `qb_kneel`, `qb_spike`, `rush`, `pass`, `epa`, `qb_epa`, `success`, `yards_gained`,
  `air_yards`, `cpoe`, `xpass`, `pass_oe`, `down`, `ydstogo`, `yardline_100`,
  `third_down_converted`, `third_down_failed`, `fourth_down_converted`, `fourth_down_failed`,
  `two_point_attempt`, `two_point_conv_result`, `td_team`, `sack`, `interception`, `fumble_lost`,
  `complete_pass`, `pass_touchdown`, `rush_touchdown`, `first_down`, `special`,
  `special_teams_play`, `fixed_drive`, `fixed_drive_result`, `drive_start_yard_line`,
  `passer_player_id`, `passer_player_name`. Guard every column with an existence check.
- Normalize `posteam`, `defteam`, `home_team`, `away_team` through `normalize_team_column`.
- Regular season only for feature inputs; playoff rows keep receiving full-regular-season values.
- Name allowed metrics explicitly (`*_allowed_per_snap`) and add them to
  `EXCLUDE_FROM_OPPONENT_STATS` so the generic `opponent_` mirror does not duplicate them.
- Every new column goes through `constants.py` lists and `build_final_column_order`; ETL must
  still emit the invariant schema when PBP is missing for a season.
- Cite the formula in the docstring and test each self-computed metric against a hand-built
  fixture, following the sos repo rule.
- Run `scripts/leakage_audit.py` after every schema change; add a test that perturbing a future
  week's plays does not change an earlier week's features (the sos repo has the pattern in
  `tests/test_validation_walk_forward.py`).
