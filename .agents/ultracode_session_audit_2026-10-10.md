# Audit of the unattended step-6 session of 2026-10-10

Session `793d07d6-9657-4103-a488-1c830f81ef04`, 07:49Z to 20:57Z (01:49 to 14:57 MDT). Audited
read-only on 2026-10-11 against git, `models/` and the session digest
(`~/.claude/digests/793d07d6-9657-4103-a488-1c830f81ef04/`). The yardstick is roadmap step 6 in
`.agents/TODO.md` and the open items in `.agents/sos_port_source.md`, both as they stand on
`feat/step6-sos-port`.

Conventions:

- Times are UTC with a `Z`; MDT is UTC-6. The repo's docs and file mtimes use MDT.
- "Review" means an independent reviewer subagent; verdicts come from the timeline digest and the
  "Review:" lines in `.agents/sos_port/DECISIONS.md`. Most subagent reports in the digest are cut
  off mid-sentence, so they do not carry verdicts themselves.
- Gate counts are the session's own claims (chat and the handoff). Commit messages record no gate
  result, and this audit did not run `scripts/gate.sh`: it runs `uv sync`, which would modify a
  worktree's venv.
- The brief says `main` is at `1cf08d1` (`0.45.1`). It is at `583a768` (`0.45.3`), which contains
  `1cf08d1`.

## Summary

1. The session landed 15 versions on `feat/step6-sos-port` (`0.46.0`-`0.57.1`, tip `2efbd3e`). Each
   was merged after an independent review. Nothing reached `main`, nothing was pushed, and the main
   checkout's `data/` was not touched.
2. Finished: phase 1 (inventory and parity), phase 2 (sources) and phase 3 (the ratings port and
   its ETL integration). Also the own Elo, its quarterback-alternative test, the power-ranking
   methods and their measurement, the web pages and pool helpers, and three ETL fixes.
3. Phase 4 is half done. Decision 1 (ported ratings replace the strength snapshot) is reviewed in
   `models/step6/REVIEW.md`. The diagnostic that splits that gain between the ratings and the
   play-by-play corrections is unreviewed, and decision 2 (own Elo) is neither scored nor reviewed.
4. All ten walk-forward runs finished cleanly (`models/step6/driver.log`). Three of the own-Elo runs
   finished after the session ended.
5. Four `worktree-agent-*` branches hold finished, committed, unmerged work. Only the betting-edge
   split had a reviewer pass, and its last five fix commits were never re-checked.
6. Not started: the adoption chunk (rule 11) and the betting-label test, so the production rebuild
   and the merge cannot happen yet. Also not started: projected standings on the ratings, the
   source's reruns and data notes, the `../nfeloqb` fix and the final report.
7. No walk-forward number entered a doc without a `REVIEW.md`.
8. Breaches to review: your email address went out in an HTTP header, system `python3` was used
   twice, tasks 58.3 and 58.6 were closed without asking, the main checkout was briefly switched
   off `main`, and pre-registration files were edited and re-hashed. Details below.
9. The step-6 handoff (`2efbd3e`) is stale. It says the ranking-default branch "probably died with
   uncommitted edits", but all 9 of its commits are on the branch.

## Finished

Listed in the order the session started them, merge commits all on `feat/step6-sos-port`.

### New nflverse sources, `MODELS_PATH`, `compare` tie rule (`0.46.0`)

- Merge `5a00db1` (09:47Z). Implementer `agent-ac0e0051f63130c11`, reviewer
  `agent-a7f9285d03743b600`.
- Review: "merge after fixes". The P2 was the QBR import-isolation test missing 12 of 15 command
  modules. The reviewer's own full `--web` gate on `7b59e1d` passed (1684 tests, 93.39%). The fixes
  (`7b59e1d..385bf2c`) were checked by the main session, not re-reviewed. The session's release
  gate was 1,699 Python and 29 web tests.
- Decisions (`.agents/sos_port/decisions/sources.md`): keep `WSH`; port the quarterback identity
  rule and the Jacksonville 2001-2002 player-team repair; ESPN QBR is for validation only; an exact
  0.5 probability picks home.
- Side effect, disclosed in `.agents/step6_plan.md`: market-line pick accuracy in future reports
  moves from 0.6401 to 0.6579 (pick-time) and from 0.6648 to 0.6670 (closing). No model number
  changes.

### Own Elo (`0.47.0`, wired into the ETL in `0.54.0`)

- Merge `4ccd73d` (10:05Z). Implementer `agent-a0c9aeb9949d70658`, reviewer
  `agent-aaa6c3e19fc224aee`. Round 1 was "merge after small fixes", round 2 "merge as is". The
  session's gate claim was 1,808 tests.
- The framework (`src/nfl_predictor/utils/polars/elo_ratings.py:75-87`):
  - `p = 1/(10^(-diff/400)+1)`, where `diff = home - away + home edge + QB adjustments`.
  - Each update is `K·mult·(result-p)` with K 20; the update lands on the base rating, not the QB
    adjustment.
  - `mult = ln(max(margin,1)+1) · 2.2/(winner_diff·0.001+2.2)`.
  - Home field is 65 (0 at a neutral site), and each offseason pulls ratings one third back toward
    1505.
  - Expansion teams start at 1300, and every team starts at 1505 in 1999.
- The 2019 additions (`elo_game_context.py`):
  - Kept: travel (55 + 4 per 1,000 visitor miles) and rolling home field (25 Elo per point of mean
    home margin over the prior 10 seasons, net of travel).
  - Implemented but off: rest/bye (+25) and the playoff boost (×1.2).
- The quarterback adjustment (`elo_quarterbacks.py:80-102`):
  - It uses 538's VALUE formula; the defense adjustment rolls 0.95/0.05 and reverts one third each
    offseason.
  - The quarterback rating rolls 0.9/0.1 and the team rating 0.95/0.05. The adjustment is
    `3.3·(QB - team)`.
  - Designed quarterback runs count in VALUE since `0.54.0`.
- The win-total pull (`elo_win_totals.py`):
  - The preseason rating is `(1-w)·regressed + w·wt_rating_elo`, from nfelosrs `wt_ratings.csv`.
  - `w` is chosen from 0 to 1 in steps of 0.05, by Brier on earlier seasons only.
- Two presets in every dataset:
  - `own_elo` (`MARKET_FREE_SETTINGS`): `away_own_elo`, `home_own_elo`, `own_elo_diff`,
    `own_elo_home_win_prob`.
  - `own_elo_wt` (`WIN_TOTAL_SETTINGS`): the `own_wt_elo` counterparts.
  - `own_elo_wt` is marked market-informed (`constants.MARKET_INFORMED_FEATURE_GROUPS`).
- Tests:
  - `tests/test_elo_ratings.py::test_framework_reproduces_fivethirtyeight_published_ratings_and_probabilities`
    runs on the fixture `tests/fixtures/fivethirtyeight_elo/nfl_games_2017_2019.csv` (801 games).
    It checks `elo_prob1` within 1e-12 and the ratings within 1e-9.
  - `test_market_free_settings_are_the_components_the_evaluation_kept`,
    `test_win_total_settings_add_only_the_win_total_pull` and
    `test_fitted_win_total_weight_never_reads_its_own_seasons_results`.
  - `tests/test_elo_game_context.py` and `tests/test_elo_quarterbacks.py`.
- The component evaluation (`.agents/sos_port/ELO_EVAL.md`, not a walk-forward):
  - Setup: forward selection on the 4,943 regular-season games of 2007-2025, 5,000 game resamples,
    seed 42, under the keep rule in `decisions/elo.md` entry 1.
  - Kept: travel −0.00040 [−0.00064, −0.00016], rolling home field −0.00030 [−0.00059, −0.00001],
    QB VALUE −0.00260 [−0.00388, −0.00133], win-total pull −0.00107 [−0.00200, −0.00009].
  - Dropped: rest/bye and the playoff boost.
  - Final Brier: 0.21701 market-free and 0.21594 with win totals. The `qb_elos.csv` columns score
    `elo_prob1` 0.22001 and `qbelo_prob1` 0.21620.
  - The reviewer found that rolling home field and the win-total pull pass by a hair: they fail on
    40 and 17 of 100 other seeds.

### Pool page, margin-only switch, live-betting design (`0.48.0`)

- Merge `6e89d8d` (10:11Z). Implementer `agent-aaea46c8498d3f6e7`, reviewer
  `agent-a36af36b4f2694497` over three rounds. Gate claim: 1,945 Python and 77 web tests.
- Decision: an exact survivor planner replaces the beam search. It is a substitute method held for
  you (`decisions/web.md`, entry 9).
- Roadmap effect: task 58.1 stays open with a `Narrowed:` note. Tasks 58.3 and 58.6 were archived
  as done in `.agents/ARCHIVE.md` (see the guardrail breaches).

### Ratings port (`0.49.0`)

- Merge `984a7f7` (10:43Z). Implementer `agent-af021dc0836482a44`, reviewer
  `agent-a8fb3fb616abddb12`.
  - Round 1: "merge after fixes". The bit-identical tests passed only on this machine's AVX-512
    kernel, so the session decided to test floats to 1e-12 and report bit-identity as a count.
  - Round 2: "merge as is".
- The ridge model (`src/nfl_predictor/ratings/ridge.py`):
  - One row per offense-against-defense game:
    `response = intercept + offense[unit] - defense[opp] + home·sign`.
  - Weighted least squares with a ridge penalty on the offense and defense effects only, solved as
    `(X'WX+P)b = X'Wy + P·m` around a prior `m`.
  - The penalty grid is `logspace(-2, 5, 29)` with 5 game-grouped folds. Each season uses the
    previous season's penalty; 1999 cross-validates its own.
- Team ratings (`team_rating.py`):
  - The target is scrimmage EPA per play, weighted by plays.
  - Offense and defense ratings are effect × league plays per team-game. Special teams is a separate
    possession/coverage fit.
  - `team_rating` is the sum of the three, in points per game against an average team on a neutral
    field.
  - Neutral-site games still get the home term, as in the source.
- The preseason prior (`team_prior.py`):
  - `fade(g) = max(0, 1-g/9)` times a carryover slope through the origin, pooled over every earlier
    season pair and centered. It needs 3 pairs, so the first prior is in 2003.
- `sos` (`team_rating.py:447`): the mean opponent rating, where each team is refit without the rated
  team's games and gets its prior from a previous season refit without that team.
- The quarterback rating (`qb_rating.py`): EPA per dropback without spikes, weighted by dropbacks.
  `qb_faced_pass_defense` comes from defenses refit without the passer.
- Rank ranges (`rating_ranges.py`): a 1,000-resample game bootstrap with `default_rng(0)`.
- The leakage-safe entry point is `RatingsHistory.pre_week` (`pre_week.py`). It fits games with
  `week < w` only, with the previous season's penalties.
- Tests:
  - Parity: `tests/test_ratings_published_parity.py` (tolerance 1e-12; fixtures under
    `tests/fixtures/sos_ratings_parity/`).
  - Units: `test_ratings_ridge.py`, `test_ratings_team_rating.py`, `test_ratings_team_prior.py`,
    `test_ratings_qb_rating.py` and `test_ratings_pre_week.py`.
  - Leakage: `test_pre_week_*_never_see_the_week_or_later_games`.
- Numbers (`.agents/sos_port/RATINGS_PARITY.md`): 310 tables and 767,141 cells, of which 766,979 are
  bit-identical. The other 162 are all `qb_epa_per_dropback`, within 1.1e-16; none is outside the
  tolerance.

### Play-by-play cache columns and stadium table (`0.50.0`)

- Merge `5f024b3` (10:56Z). Implementer `agent-aee4f463ac1335152`, reviewer
  `agent-a78c236c0974e2870`.
- Review: "merge after fixes", for documentation only. The fixes (`fd6bf34..3870a39`) were checked
  by the main session. Gate claim: 2,205 tests.
- Changes:
  - The cache grows from 55 to 139 columns. A fresh download equals the cache in 27 of 27 seasons.
  - 14 venue ids are added, covering 163 games.
  - Decision: nflverse's dome flags for the 2026 Melbourne, Munich and Paris venues are kept.
    Recheck before the Paris game on 2026-10-25 (`decisions/pbp_columns.md`).

### Phase 1: inventory and play-by-play parity (`0.50.1`)

- Merge `86eee8b` (11:26Z). Implementer `agent-a2c3c6e91a78b8738`, reviewer
  `agent-a57a9fbf83a41ca05`.
  - Round 1 found 49 or more columns misfiled as missing, and that `qb_scramble` was available.
  - Round 2 found the work correct, apart from two mechanical fixes. The final fixes were rerun by
    the main session.
- Numbers:
  - `INVENTORY.md`: 1,736 columns (30 matched, 526 different, 1,180 missing); 12 sources; 45 web
    items.
  - `PARITY.md`: 1999-2025, 29 pairs, 403,912 values. 40,694 values mismatch, all explained.
- Decisions (`decisions/phase1_parity.md`):
  - Adopted: pre-2006 scrambles count as dropbacks, the Jacksonville repair, and the down flags read
    on every play.
  - Carries were first kept as they were, then corrected at 10:32Z to the source's rule.
- Refresh: interrupted, on `worktree-agent-ab92fdaea474da115` (see the four branches).

### Season statistics port (`0.51.0`)

- Merge `4e2ce33` (13:33Z). Implementer `agent-a5d833400b788544f`, reviewer
  `agent-aa1b2d7c1f9fc38b0`.
  - Round 2 returned "merge as is", with five P3 findings.
  - The reviewer's full gate on `b7e46a0` passed: 2,663 tests, 94.56%.
- The new package is `src/nfl_predictor/season_stats/`, a near-verbatim port that keeps the nflverse
  quirks.
- The audit is `.agents/sos_port/SEASON_STATS_PARITY.md`, reproduced at 1 and 24 BLAS threads.

### Leaving feature groups out of production (`0.52.0`)

- Merge `d48a471` (13:43Z). Implementer `agent-ae2b0e6be7b287a3b`, reviewer
  `agent-ada0714a2e67d98b1`.
- Review: "merge as is", plus three hardening items, which the main session checked. Gate claim:
  2,713 tests.
- Mechanism: `disable_feature_groups` in `config/weekly_run.yaml:65`, default `[]`, resolved by
  `ml/feature_spec.py` (`validate_feature_groups`, `resolve_feature_group_columns`).
- Tests: `tests/test_weekly_run_feature_groups.py::test_the_shipped_config_leaves_no_group_out` and
  `tests/test_final_fit_characterization.py`.

### Team and quarterback web pages (`0.53.0`, task 58.2 narrowed)

- Merge `0a1621a` (14:26Z). Implementer `agent-a38db4dacc254c962`, reviewer
  `agent-afdf2c88a8d667999`.
  - Round 2 returned "merge as is". The last round was checked by the main session.
  - The gate ran with `--web`.
- The pages: `/teams`, `/qbs`, `/schedules`, `/rank-ranges` and `/head-to-head`, with gzip at level
  6.
- 58.2 stays `[ ]` with a `Narrowed:` note. It still needs the production rebuild and a first look
  in a browser; neither the session nor any reviewer opened the pages.
- Four source features were not ported (`decisions/web_pages.md`, entries 13 and 19-21).

### Ratings in the ETL, play-by-play corrections, one starter, own Elo in datasets (`0.54.0`)

- Merge `b525065` (15:32Z). Two implementers: `agent-ab77a9529b8ed64b8` (the corrections, the
  starter and the Elo wiring) and `agent-a0778ee61e6367a09` (the ratings in the ETL).
- Reviewers: `agent-a5116aea67c53cfa6` and `agent-a194f6a56baa4b463`, two rounds each. The final
  combined fixes (`cdb1ed7..ba9b7c7`) were checked by the main session. The gate ran with `--web`.
- The ratings feature group: 21 columns from `constants.RATINGS_FEATURE_STATS`
  (`constants.py:1480`), each as `away_`/`home_`/`_diff`. The group is `ratings`; its stats are
  `ratings_offense`, `ratings_defense`, `ratings_special_teams`, `ratings_team`, `ratings_sos`,
  `ratings_qb` and `ratings_qb_dropbacks`.
  - In week 1, `ratings_qb` is null and `ratings_sos` is null; in week 2 `ratings_sos` is still
    null.
  - The ETL path is `utils/polars/ratings_build.py`, which writes
    `data/ratings/{season}_{kind}.parquet`.
- Corrections, each with its record in `decisions/integration_pbp_elo.md`:
  - Carries are `rush_attempt & !two_point & !scramble & !kneel` (`pbp.py:325`). They drop from
    357,227 to 348,419 and equal the source in 13,928 of 13,928 team-games.
  - Scrambles filed as runs count as dropbacks (`pbp.py:257`): 3,744 plays in 1999-2005.
  - One starter, from `qb_stats.expected_starter_ids` (`qb_stats.py:393`). It differs from the old
    name chain in 12 of 13,922 team-games.
- Adapter parity (`.agents/sos_port/ADAPTER_PARITY.md`): 305,830 input cells, and 741,295 of the
  table cells bit-identical.
- Tests:
  - `tests/test_ratings_feature_group.py`.
  - `tests/test_ratings_build.py::test_features_of_a_week_never_see_that_week_or_later` and
    `::test_two_runs_attach_identical_features`.
  - `tests/test_data_collection_incremental.py::test_a_weeks_rating_features_never_see_its_or_later_plays_through_the_etl`.
  - `tests/test_polars_pbp.py`: the carry and scramble tests.
  - `tests/test_qb_starters.py`.
- Cost: about 453 s for a full ETL and about 31 s for an incremental one (`ratings_etl_timing.py`).

### Red-zone trips (`0.55.0`)

- Merge `eaa5751` (15:57Z). The release commit `d6064eb` is the walk-forward build commit.
  Implementer `agent-ad305e6aaa9e860bd`, reviewer `agent-a5e4ce87dff1ededc`.
- Review: round 1 found 230 phantom trips; round 2 returned "merge as is".
- Rule: a drive nflverse flags `drive_inside20` counts as a trip only when the credited team ran an
  offensive snap on it (`pbp.py:374`). The snap condition is a logged departure from the source.
- Effect: the league red-zone touchdown rate moves from 0.5709 to 0.5395.
- Tests: four in `tests/test_polars_pbp.py`, starting at
  `test_red_zone_trips_follow_the_nflverse_drive_flag_not_the_yard_line` (line 1952).

### Four-week trends of upcoming games (`0.55.1`)

- Merge `d56e34c` (16:10Z). Implementer `agent-a83a6340d65fc8958`, reviewer
  `agent-a305d43ed2b15c177`. One round, "merge after fixes"; the main session checked the fixes.
  Gate claim: 3,078 tests.
- The defect predates step 6: an upcoming game's `scoring_margin_4wk_trend` and
  `turnover_margin_4wk_trend` were 0.0. The fix keys trends on the schedule
  (`features.py:1063`).
- Effect: no regular-season training row moves, all 309 playoff rows do, and weekly predictions
  change from week 6. This is held as a weekly-output change.
- Tests: `tests/test_trend_parity.py` and `tests/test_trend_features.py`.

### Season statistics in the ETL (`0.56.0`)

- Merge `03040dc` (17:56Z). Implementer `agent-ad5396b63790d07a8`, reviewer
  `agent-a8b91251ebec76b19`. Round 2 returned "merge as is".
- Parity (`.agents/sos_port/SEASON_STATS_ETL_PARITY.md`): 280 tables for 1999-2026, bit-identical,
  as the reviewer reproduced.

### Power-ranking methods and their measurement (`0.57.0`)

- Merge `8785e5e` (18:17Z). Implementer `agent-aa082302f4fc5ac12`, reviewer
  `agent-a56d42d2fd2fc3dc3`, two rounds. The second key is `models/step6/rankings/REVIEW.md`
  ("merge as is"). Gate claim: 3,232 tests.
- The rule was pre-registered in `models/step6/rankings/HYPOTHESIS.md`, sha256 `44e3a33c...`.
- The methods (`src/nfl_predictor/reporting/power_rankings.py`, `rating_sources.py`):
  - `ported`: `ratings_team`.
  - `model`: a ridge fit with λ 1e-6 of `margin = r[home] - r[away] + h·(1-neutral)` to the
    walk-forward fold predictions before the week.
  - `blend`: `w·ported + (1-w)·model`, where `w` comes from a 0.1 grid chosen on earlier seasons.
  - The branch default is still `composite`.
- Scoring (`.agents/sos_port/rankings_eval.py`): next-week margin MAE over 4,400 games (2008-2025,
  weeks 2 on), with a game bootstrap (10,000 resamples, seed 42) and a season-cluster bootstrap.
- Numbers (cited to `REVIEW.md` in `DECISIONS.md` and `.agents/step6_plan.md`):
  - The blend scores 10.497 against 10.643 for the ported rating: −0.146, with games
    [−0.204, −0.088] and seasons [−0.194, −0.102]. It is better in 18 of 18 seasons.
  - Composite (10.590) and model (10.678) tie the ported rating.
  - Two seeds: −0.151 [−0.208, −0.094].
  - The blend's weight on the ported rating is 0.7 in 2008-2009, 0.6 in 2010-2014 and 0.5 from 2015.
- Inputs: `models/step6/ratings_anchored_seed42` and `models/wf_checkpoints/d97fab54e7261f18e349`.
- Gap: the producer's `RANKINGS_EVAL.md` exists only in `/tmp` (for example
  `/tmp/rankings_eval_two_seeds/`). It is not tracked, but `decisions/power_rankings.md:13` points
  to it.
- Your decision (13:20 MDT, `.agents/step6_plan.md`, "User decisions during the run"): the weekly
  run publishes the blend at weight 0.5. That change lives on `worktree-agent-a2071e4c492dbbdda`.

### The ported quarterback rating as the own Elo's adjustment (`0.57.1`)

- Merge `cb5ec57` (20:05Z).
- Implementers: two attempts stopped on the 16,000-token output limit before writing anything (the
  first was `agent-a927f71b61ace395b`). The third, `agent-afa04ba86d07ac1b9`, had every design
  choice fixed in its brief and finished.
- Reviewer `agent-af2ac0a0474353ca0`: "merge after small fixes", which the main session checked.
  Gate claim: 3,259 tests.
- The alternative adjustment is `25·D(s)·(r(starter) - team)`, with the team rating rolling
  `0.95·team + 0.05·r`.
- Result (`.agents/sos_port/ELO_QB_ALT.md`, rule in `decisions/elo.md` entry 17): VALUE stays.
  - The alternative scores Brier +0.00524 [+0.00319, +0.00725] and log loss +0.01203
    [+0.00736, +0.01667].
  - At half scale, Brier +0.00221 [+0.00095, +0.00351]; at 1.5 times, +0.01094.
- Test: `tests/test_sos_port_elo_qb_alt.py`.
- Finding for you: the published quarterback week-by-week tables use the whole season's
  cross-validated penalty. That is a mild look-ahead in displayed histories only.

### Phase 4, decision 1: ported ratings as model features (reviewed)

- Build: `d6064eb` (`0.55.0`) built into `~/scratch/step6/data` through `models/step6/build.sh`.
  - Input `models/step6/inputs/candidate.csv`, sha256 `74b21113...`.
  - The leakage audit passed with 490 features (`models/step6/audit/candidate.json`).
  - Provenance is in `models/step6/launch_record.txt`.
- Pre-registration: `models/step6/HYPOTHESIS.md`, sha256 `e0277939...`. The original checksum
  `5b91d1d4...` is in `HYPOTHESIS.original.sha256`; the reviewer verified that the amendment only
  appended lines 85-104.
- Launch: `models/step6/launch.sh` → `driver.sh`, from `~/scratch/step6_runs` (detached at
  `d6064eb`), starting 16:08:50Z, one run at a time on the GPU. Each run is the reference command
  (2007-2025, 200 trees, `--calibration auto`, seeds 42 and 7).
- Runs (each `exit 0` in `driver.log`):
  - `models/step6/ratings_anchored_seed42`: 16:08Z to 17:28Z.
  - `ratings_anchored_seed7`: to 18:01Z.
  - `ratings_marketfree_seed42`: to 18:36Z.
  - `ratings_marketfree_seed7`: to 19:10Z.
- Second key: `models/step6/REVIEW.md` by `agent-af50a0acaea4ef357`, about 20:12Z, before any doc
  quoted these numbers. Governing window: all weeks, two seeds.
- Comparison 1, ported ratings anchored against `models/step4_reference/adopted_seed{42,7}`:
  - All weeks: Brier −0.00084 [−0.00144, −0.00024] (win), log loss −0.00191 [−0.00326, −0.00057]
    (win), pool +77.5 [−15.0, +171.0], margin MAE −0.0239 [−0.0483, +0.0003].
  - Weeks 3-18: Brier −0.00102 [−0.00169, −0.00037], pool +91.5 [+5.0, +177.0].
  - Seed 42 alone is a tie (−0.00035 [−0.00113, +0.00044]); seed 7 alone is −0.00133
    [−0.00215, −0.00052].
  - Against the market, the arm ties on Brier: +0.00021 [−0.00102, +0.00138].
- Comparison 2, ported ratings market-free against
  `models/step4_reference/marketfree_all_seed{42,7}`:
  - All weeks: Brier −0.00032 [−0.00114, +0.00049] (tie), log loss −0.00055 [−0.00235, +0.00121]
    (tie).
  - Week 1 is flagged as a loss: Brier +0.00321 [+0.00045, +0.00596], log loss +0.00676
    [+0.00067, +0.01281], pool −41.0 [−74.5, −8.5].
- Outcome: rule 1's first branch, so replace the strength snapshot. Conditional arm 5 is not
  triggered.
- The reviewer's P2: hold the decision until the diagnostic splits the corrections' share from the
  ratings' share (next section).

## Interrupted

### Phase 4 diagnostic: the new-build reference (runs done, unreviewed)

- Runs: `models/step6/newref_anchored_seed42` (19:10Z to 19:45Z) and `_seed7` (to 20:19Z), both
  `exit 0`.
- What exists: the session's own `compare` check-ins at 20:46Z. They are
  `models/step6/checkins/newref_vs_adopted_two_seeds.{md,json}` and
  `ratings_anchored_vs_newref_two_seeds.{md,json}`. The numbers went to you in chat, marked
  unreviewed, and into no doc. This audit does not repeat them (rule 3).
- Missing: the rule-3 rescore appended to `models/step6/REVIEW.md`. Decision 1 waits on it.

### Phase 4, decision 2: own Elo against the `qb_elos.csv` columns (runs done, unscored)

- Runs: `models/step6/elo_anchored_seed42` (20:19Z to 20:52Z), `elo_anchored_seed7` (to 21:23Z),
  `elo_marketfree_seed42` (to 21:54Z) and `elo_marketfree_seed7` (to 22:23Z).
  - All exited `0`; the driver finished at 22:23Z (`driver.log`, "failed runs: 0").
  - Three of them finished after the session ended.
- Checkpoints: by modification time, inferred rather than recorded, they are `4d3d7199...`,
  `85d4d2e7...`, `d8462a7d...` and `54ec3f11...` under `models/wf_checkpoints/`.
- Missing: the `compare` check-ins (comparison 3 against arm 1, comparison 4 against arm 2), the
  reviewer's rescore, and the outcome of rule 2 in `models/step6/HYPOTHESIS.md`. The `../nfeloqb`
  fix waits on that outcome.

### Branch work finished but not merged

These are the four branches, detailed below:

- The open rating-definition tests, which the source left open: done, unreviewed.
- The phase-1 audit refresh: done, unreviewed.
- The betting-edge split: done and reviewed in two rounds, but its last five fix commits were not
  re-checked.
- The blend ranking default and the "no game lines" wording: done, with the review stopped at its
  start.

### Roadmap tasks narrowed by the session (correctly left open)

- 58.1, the pool helpers: delivered with the exact survivor planner as a substitute method, which
  waits for you (`.agents/TODO.md` and `decisions/web.md`, entry 9).
- 58.2, the team and quarterback pages: these still need the production rebuild and a browser look.

### `DECISIONS.md` review fields are stale

- Per the method-facts read of `feat/step6-sos-port`, most per-topic decision files still say
  "Review: not yet reviewed". That covers `integration_ratings.md`, `feature_groups.md`, `elo.md`
  1-16 and `trend_parity.md`, among others.
- `decisions/power_rankings.md:13` still says its results await review.
- The index in `.agents/sos_port/DECISIONS.md` records every batch as reviewed.

## Not started

- The adoption chunk. Production must match the adopted arms (rule 11): `disable_feature_groups`
  and `floor_sigma_reference_runs` in `config/weekly_run.yaml` need setting. Until then, a rebuilt
  `data/` would train on 29 columns no arm measured (`models/step6/REVIEW.md`, finding 3).
- The betting-label test: your decision at the 13:20 MDT check-in, to be pre-registered on the 815
  real-moneyline games and run after the edge split merges.
- Projected standings on the ported ratings combined with the model (phase 5).
  - `git grep` on the branch finds only the standings readers already on `main`, and
    `decisions/power_rankings.md` never mentions standings.
- Open items from `.agents/sos_port_source.md`:
  - Reruns of the source's `check-team-prior` and `check-wp-filter` (they predate the pre-2006
    scramble fix).
  - The ask-first data notes.
  - Separate pass and run quarterback ratings, if "every quarterback play" is adopted.
  - Whether `qb_designed_carries` and `designed_carries` follow the botched-snap rule.
  - Implementing the approved botched-snap removal. It was only measured, on the rating-tests
    branch.
- A measurement of the "no game lines" model with the win-total Elo kept. Your 13:20 MDT definition
  implies it, but it is not in the ladder.
- A previous-season carryover for the week-1 quarterback rating. The reviewer suggested it after
  the market-free week-1 loss; it needs its own pre-registration.
- The ratings refit speed-up through numpy masks (`.agents/step6_plan.md`, "Follow-up").
- The docs close-out, one consolidated approvals list, and your final HTML report.
- Held for you by design: the merge to `main`, the `data/` rebuild, the `../nfeloqb` fix and the
  feature ideas (referee, weather, attendance, team health), which you moved to a later session.

## The four branches

None changes a file that `feat/step6-sos-port` changed after the fork, apart from
`tests/sos_port_scripts.py`. That file is the same blob (`995a6dbc`) on all five branches. The
blend-default branch and the betting-edge branch both edit `README.md` and
`src/nfl_predictor/weekly_run/pipeline.py`, so whichever merges second resolves those. None of the
four has a `CHANGELOG.md` entry or a version bump.

### Blend ranking default and "no game lines" (`worktree-agent-a2071e4c492dbbdda`, 9 commits)

- Forked from `6cf3f7f` (`0.57.0`); 15 step-6 commits have landed since.
- Implementer `agent-a2071e4c492dbbdda`, started 20:04Z. It died on the limit, was resumed, and
  finished at 20:51Z with a gate claim of 3,243 tests.
- Reviewer `agent-ae0af5f3e8fdcecaa` was stopped at its start at 20:57Z, so there is no verdict.
- Change 1, the rankings:
  - `4c21abd` converts the tests to Arrange-Act-Assert.
  - `ac4b8fe` reads a fold without a neutral column as home games.
  - `3e7b7e2` takes earlier seasons from the averaged floor-sigma reference runs.
  - `cc101ed` records stage 1's fold checkpoints as a prediction source.
  - `6c0c285` `feat(weekly)!` sets `power_rankings_method: blend` and
    `power_rankings_blend_weight: 0.5` in `config/weekly_run.yaml`, and changes the code defaults in
    `weekly_run/config.py` to match. It is BREAKING; `composite` restores the old ranking.
  - `5c2eec3` documents it.
- Change 2, "no game lines" wording: `5284db2` turns the kept win-total groups' warning into an
  info note in `ml/feature_spec.py`; `0fdd2d7` and `5aa5c74` update the help text, `README.md` and
  `.agents/modeling_spec.md`.
- Recommendation: run one reviewer pass, then merge into `feat/step6-sos-port` as `0.58.0` (minor,
  for the changed default). Merge it after the betting-edge branch and after decision 2 is scored.
  You already chose both changes at the 13:20 MDT check-in, and the merge to `main` stays yours.
- Risks:
  - `5284db2` edits `src/nfl_predictor/ml/`, which changes every walk-forward checkpoint
    fingerprint. Group it with the adoption chunk (rule 14).
  - The weekly ranking now depends on stage-1 fold checkpoints and the reference runs. A missing
    source skips the rankings with a warning.
  - The characterization snapshot `power_rankings.csv` changes.
- If you defer the default, cherry-pick everything except `6c0c285` and `5c2eec3`.

### Betting-edge split (`worktree-agent-aa56c16b91074022f`, 23 commits)

- Forked from `406618d` (`0.57.0` release); 17 step-6 commits have landed since.
- Rule pre-registered in `7eccd01`.
- Measurement: `.agents/sos_port/betting_edges.py`, `BETTING_EDGES.md` and
  `decisions/betting_edges.md`. The reviewer's scripts are under
  `.agents/sos_port/review/betting_edges/`.
- Production:
  - `src/nfl_predictor/reporting/betting_report.py` adds `market_home_prob_spread`,
    `edge_beyond_spread_home_prob` and `spread_moneyline_gap_home_prob`, and rejects a non-positive
    sigma (`8b29447`).
  - `weekly_run/pipeline.py` passes the production floor sigma (`d79328d`).
  - The snapshot `betting_report.csv` changes.
- Review by `agent-a8512322c21aed91f`:
  - Round 1 asked for documentation fixes.
  - Round 2 returned at 20:47Z and asked for last fixes. Those landed as `07bd2ca..e9c7a09`
    (20:48Z to 20:49Z) and were not re-checked.
  - The implementer reported "ready to merge" with a gate claim of 3,266 tests.
  - The step-6 handoff still names `00f2d02` as the tip.
- Recommendation: a short reviewer check of `07bd2ca..e9c7a09`, then merge as `0.57.2`.
- Risk: the weekly betting report gains three columns and a sigma check. That is an additive
  weekly-output change, so it goes on the approvals list. Labels are unchanged (the measured rule
  keeps them on the whole edge, per the implementer).

### Phase-1 audit refresh (`worktree-agent-ab92fdaea474da115`, 6 commits)

- Forked from `9bba627` (`0.56.0`); 46 step-6 commits have landed since.
- Contents: `.agents/sos_port/inventory.py` and `parity.py` take `--data-dir` (`786b95c`) and run on
  the integrated play-by-play path (`a9d4e6b`). The refresh also covers the judgments, causes and
  reports, `decisions/phase1_refresh.md` (11 entries) and `tests/test_sos_port_audit.py` (24 tests).
  Nothing under `src/` changes.
- Implementer's unreviewed numbers, as the handoff quotes them:
  - Inventory columns move from 30/526/1,180 to 1,236/500/0.
  - Parity mismatches drop from 40,694 to 25,564, with none unexplained.
- Reviewer `agent-a30a772332b01d950` hit the weekly limit, was resumed, and was stopped at 20:57Z
  with no verdict. The implementer claimed 3,156 tests passed.
- Recommendation: review, regenerate on the merge tip (the scripts reproduce byte for byte), then
  merge as a patch.
- Risk: low; it is audit tooling only. The reports are pinned to `0.56.0` code, which the later
  rankings and Elo-alternative merges do not touch.

### Open rating-definition tests (`worktree-agent-a5c5909054257aa73`, 19 commits)

- Forked from `a1987ba` (`0.55.1`); 76 step-6 commits have landed since.
- Pre-registration: four protocols, all committed before scoring, under
  `.agents/sos_port/protocols/` (`81e37b2`, `ac1e01a`, `9b7eaf9`, `e63b38e`). A correction to the
  1999 snapshot check (`6d61fef`) also came before scoring.
- Contents: the scripts (`rating_tests_common.py`, `team_plays_test.py`, `qb_plays_test.py`,
  `botched_snaps.py`), four generated reports, `decisions/open_rating_tests.md` and four test files.
  Nothing under `src/` changes.
- Implementer's outcomes, unreviewed (numbers in the reports):
  - Kneel-downs and spikes: keep today's rating.
  - Penalty-wiped plays: a four-way tie. The simplicity order picks "every wiped snap".
  - Quarterback all-plays: a tie. Your stated order picks "every quarterback play".
  - The approved botched-snap removal moves the ratings very little.
- Departures: the pre-registered ESPN QBR extras and the passer holdout were skipped. That is a
  narrowing, listed below.
- Reviewer `agent-ae042d17ed7fd35ba` was stopped at its start at 20:57Z. The implementer claimed
  3,187 tests passed.
- Recommendation: review, then merge as a patch, with an index entry in `DECISIONS.md`.
- Risk: none to production. Adopting any of its definitions would change the ratings and break
  parity with the published files, so it needs another build and walk-forward. That is your call,
  below.

### Also

- `feat/sos-port-inventory` (`88e28cd`) is fully contained in `feat/step6-sos-port`, with 0 commits
  ahead. It can be deleted once you agree.

## Guardrail breaches

No merge into `main`, push, tag, reset or rebase happened. Only `feat/step6-sos-port` received
merges, and `main..feat/step6-sos-port` is 520 commits that are not on `main`.

The main checkout's `data/` has no file with a 2026-10-10 mtime. Under `models/`, the session added
`models/step6/` (untracked) and ten directories under `models/wf_checkpoints/`. The candidate data
went to `~/scratch/step6/data`. All of that is within the ladder you authorized
(`.agents/step6_plan.md`, "Authority").

What follows needs review:

1. **Your email address left the machine.**
   - Who: the own-Elo reviewer (`agent-aaa6c3e19fc224aee`).
   - What: its first test request to Wikipedia's API put your email address in the User-Agent
     header; its later requests used a generic agent.
   - Rule: your global instructions forbid sending the address to an unrelated service.
   - Source: disclosed in `.agents/step6_plan.md`, "For the morning report". It cannot be reverted.
2. **System `python3` was used twice.**
   - A web implementer edited two web files with it.
   - The main session rewrote `.agents/sos_port/DECISIONS.md` with `python3 -I` at 10:44Z.
   - Both edits were harmless, and both are disclosed in the same section.
3. **Two roadmap tasks were closed without asking (rule 5).**
   - 58.3 (the live-betting design) and 58.6 (the margin-only switch) were moved to
     `.agents/ARCHIVE.md` as completed, on the branch, at 10:12Z.
   - They exist only on the branch, so you can accept or revert them at the merge.
4. **The main checkout left `main` for 12 minutes.**
   - What: `git switch -c feat/sos-port-inventory` at 07:53:59Z, back at 08:05:55Z.
   - Why it matters: your web server imports code from that checkout.
   - Impact: only `.agents/` files differed (`cacd301`, `88e28cd`), so no Python changed. It also
     breaks the session's own layout rule.
5. **A check failure was hidden by output truncation.**
   - At 11:50Z, a `| tail -1` hid a markdownlint failure. The session then amended its unpushed
     commit.
   - It was caught, but it is the pattern rule 1 exists to stop. The commit is not identified in
     the digest.
6. **Pre-registration files were edited after hashing, both before any scoring or launch.**
   - `models/step6/rankings/HYPOTHESIS.md`: its stated time was changed by `sed` at 10:46Z and the
     file re-checksummed.
   - `models/step6/HYPOTHESIS.md`: its amendment header was edited by `sed` at 15:35Z and the file
     re-checksummed.
   - The reviewer verified that the rules are byte-identical. Because `models/` is gitignored,
     though, the record is not tamper-evident (`REVIEW.md`, finding 2).
7. **A narrowing is not yet on your question list.**
   - The rating tests skipped the pre-registered QBR extras and the passer holdout. The departures
     are recorded only in `decisions/open_rating_tests.md` on the unmerged branch.
8. **Fix rounds were checked by the delegating session instead of a reviewer.**
   - Releases: `0.46.0`, `0.50.0`, `0.50.1`, `0.52.0`, `0.54.0` (final round), `0.55.1` and
     `0.57.1`.
   - The "merge after fixes" items were verified by the main session, not re-reviewed. That falls
     short of the three-roles rule.
9. **Unreviewed numbers appear in a doc (not walk-forward numbers).**
   - The handoff `2efbd3e` quotes the rating tests' and the audit refresh's results, labeled "the
     producer's, unreviewed".
   - `decisions/power_rankings.md:13` points to a report that lives only in `/tmp`.
10. **The walk-forward authority was inferred.**
    - Your 08:01Z grant ("continue working through step 6 on your own …") does not mention
      walk-forwards.
    - The session read it as approving its own 10-run ladder (plus 2 conditional runs). The ladder
      was pre-registered, capped and run one at a time.
    - This is for you to ratify, not necessarily a breach.

## Recommended next three sessions

1. **A review session, no implementers.**
   - Independent reviewers with turn caps rescore the diagnostic and decision 2 from the existing
     fold checkpoints and append both to `models/step6/REVIEW.md`. No new runs.
   - They then review the rating-tests branch, the audit-refresh branch, the blend-default branch
     and the last five commits of the betting-edge branch.
   - The main session only rewrites the handoff, correcting its stale branch states.
2. **An integration session on `feat/step6-sos-port`.**
   - Merge the reviewed branches in this order: the audit refresh, the rating tests, the betting
     edges (`0.57.2`), then the blend default (`0.58.0`, if you keep it).
   - Then the adoption chunk, per the outcomes of rules 1 and 2.
   - Pre-register and run the betting-label test.
   - Regenerate `RANKINGS_EVAL.md` under `models/step6/rankings/`, refresh the stale "Review:"
     fields, and commit the hypothesis checksums.
   - Run `scripts/gate.sh --web`.
3. **An approval and merge session with you, between game weeks.**
   - Go through the approvals list and the final HTML report.
   - Merge to `main` and rebuild `data/` with a backup, then run
     `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`.
   - Take a browser look at the pages, and recheck the 2026 roofs before the Paris game on
     2026-10-25.
   - Then decide the `../nfeloqb` fix, which depends on decision 2.

## Questions for me

Answered 2026-10-11: you accepted every recommendation below and the three-session plan above.
`feat/sos-port-inventory` is deleted; the four agent branches go once merged. The record is in
`.agents/next_agent_session_prompt.md`, "Decisions on record".

1. **The blend ranking default** (`6c0c285` on `worktree-agent-a2071e4c492dbbdda`).
   - Today the weekly run publishes the composite (`config/weekly_run.yaml` on
     `feat/step6-sos-port`). You chose the blend at 13:20 MDT.
   - Recommendation: review the branch, then merge it whole into the step-6 branch. It stays off
     `main` until your final approval.
   - Alternatives: cherry-pick everything but `6c0c285` and `5c2eec3`, or drop it.
   - Confidence: high that the code matches your decision. The main risk is the checkpoint
     fingerprint change from `5284db2`.
2. **When to act on rule 1.**
   - Replacing the strength snapshot with the ported ratings could wait for the reviewed
     diagnostic.
   - Recommendation: wait. It costs one reviewer rescore and no runs, and it tells you whether the
     gain is the ratings' or the corrections'.
   - Confidence: high. What would change my mind: you treat the rule as already final.
3. **Ratify the session's reading of your grant.**
   - The session read your grant as covering the 10-run ladder and the archiving of 58.3 and 58.6.
   - Recommendation: ratify the ladder (pre-registered, capped, clean). Keep 58.3 and 58.6 archived
     at the merge, since both were reviewed.
   - Confidence: medium. Revert the archive if you want closures to stay yours alone.
4. **The rating-definition outcomes.**
   - Candidates: penalty-wiped plays to "every wiped snap", quarterback all-plays to "every
     quarterback play", and the botched-snap removal.
   - Recommendation: adopt none before the merge. Log them as a post-merge measured change, since
     each breaks published parity and needs a new build and walk-forward.
   - Confidence: medium. A tie goes to the simpler setting and to you (rule 9).
5. **The skipped QBR extras and passer holdout.**
   - Recommendation: accept them as logged departures. They were descriptive extras that cannot
     change the decision.
   - Confidence: medium. Ask for them if you want the protocol run in full.
6. **The betting report's three new columns.**
   - Recommendation: approve them with the merge. They are additive and the labels stay as they
     are.
   - Confidence: high.
7. **Branch cleanup.**
   - Recommendation: delete `feat/sos-port-inventory` and the four `worktree-agent-*` branches
     once their work is merged.
   - Confidence: high. Deleting branches is irreversible only for work that was never merged.
