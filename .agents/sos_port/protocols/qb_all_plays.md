# Protocol: the quarterback rating from all of a quarterback's plays

Pre-registered before any candidate's error is computed, and committed on its own. It ports the
protocol `nfl-sos-ratings` registered and reviewed but never ran: its roadmap section "Q1. QB rating
from all of a quarterback's plays" at the frozen commit `9081bce` (`.agents/roadmap.md` there,
read from `~/scratch/sos_9081bce_season_stats/`). Everything below follows that text; the
departures, all forced by running it in this repository, are listed in "Departures from the
source protocol". The result is report-only: adopting a candidate changes the `ratings` feature
group and the published quarterback tables, so it is the user's decision after this test and needs
its own measurement.

## Background

The ported quarterback rating, `adj_qb_epa_per_dropback` (`src/nfl_predictor/ratings/qb_rating.py`,
rows from `ratings/pbp_inputs.py`), fits EPA per dropback, where a dropback is a play with a passer:
pass attempts and sacks. nflverse leaves the passer empty on scrambles, so they drop out, and
designed quarterback runs were never counted. The source's maintainer asked that the rating count
every aspect of quarterback play.

Candidates, in plain names (the letters are shorthand inside this file and the script only):

- A, "passes and sacks": the rating as ported, EPA per dropback (pass attempts and sacks) with
  spikes left out of its EPA.
- B, "plus scrambles": A with the quarterback's scrambles added.
- C, "every quarterback play": B with his designed runs and the botched snaps the official scorer
  charged to him added.

## Decisions on record that this protocol carries

- The order of simplicity, which decides between candidates the test cannot tell apart, is the
  user's (reported by the `nfl-sos-ratings` session as the maintainer's decision of 2026-10-09,
  `.agents/sos_port_source.md`, "Handoff after the freeze", item 1): **"every quarterback play",
  then "plus scrambles", then "passes and sacks"**. It replaces the order the source protocol
  left open (its question 4).
- The charge rule for botched snaps (same record, item 2): a botched snap counts against a
  quarterback only when the official scorer charged the fumble to him (`fumbled_1_player_id`),
  whoever ran with the ball afterwards. C counts exactly those.
- One unit for C (the source's question 5): one rating over all of C's plays.

## What each candidate counts

Every candidate keeps the ported estimator unchanged: one row per quarterback-game,
`EPA per play = intercept + passer - defense + home field`, weighted by the candidate's plays, the
ridge penalty on the passer and defense effects (`qb_rating.fit_qb_ratings`). Each candidate's play
count and EPA per play go in the `qb_dropbacks` and `qb_epa_per_dropback` columns. The rows are the
same for every candidate and season: the quarterback-game rows `quarterback_rows` returns for the
whole season with at least one dropback, keyed by `qb_id`.

- **A**: the rows as they are. Plays with a passer and nflverse's `qb_dropback` flag (pass attempts
  and sacks, two-point passes included); EPA is the weekly player stats' `passing_epa` less the
  passer's spike `qb_epa` (the adapter's `qb_dropback_epa`). Among the dropbacks are aborted snaps
  that went on as a pass or sack, 60 of them charged by the scorer to a lineman in 2006-2025 (the
  source's count); they stay in A as ported (see the departures).
- **B = A + scrambles**: plus every `play_type == "run"` play flagged `qb_scramble` whose
  `rusher_player_id` is the quarterback, two-point tries out.
- **C = B + every other quarterback play**: plus his designed runs (every other `play_type == "run"`
  play flagged `rush`, with him as `rusher_player_id`, not a scramble, not `aborted_play`, two-point
  tries out) and the botched snaps charged to him (`play_type == "run"`, `rush` and `aborted_play`
  set, `fumbled_1_player_id` is the quarterback, two-point tries out).
- "C without the botched snaps" is a descriptive extra.

Plays go to a quarterback by player id, never by name: runs by `rusher_player_id`, charged botched
snaps by `fumbled_1_player_id`, matched to the quarterback rows on `game_id`, `team` (the play's
`posteam`) and `qb_id`. Runs by players without a quarterback row in that game are not quarterback
plays. Scrambles and runs take the play-by-play `epa`, as the source's helper does.

Left out of every candidate, as in the source: kneel-downs, spikes (out of A's EPA too), two-point
tries in the added plays, plays a penalty wiped out (`play_type == "no_play"`), and run-coded plays
flagged neither `rush` nor `qb_scramble` (passes and fumbles coded as runs).

## The test

- **Hypotheses (falsifiable)**: with scrambles counted (B), the starting quarterback's rating adds
  more than A's to the team fit's prediction of the team's offensive EPA per play in a game neither
  has seen; with every quarterback play counted (C), more than A's and more than B's. Each is not
  supported unless its paired interval lies entirely below zero.
- **Candidates, fixed in advance**: A, B and C. Nothing else is a decision input.
- **Primary metric**: mean absolute error of a team's offensive EPA per scrimmage play in a game,
  `offensive_epa / offensive_snaps` of the ported team-game rows (`pbp_inputs.team_games_from_pbp`),
  predicted before the game from the team fit's prediction and the starting quarterback's rating.
- **Starter**: the quarterback on the team's first quarterback snap: the first first-quarter play,
  by `play_id`, whose passer or rusher has a fit row (a quarterback row with at least one dropback)
  for that team in that game. Without one the starter is unknown and every candidate predicts with
  a passer effect of 0. The schedule's starter fields are not used; the check prints the unknown
  starters and the disagreements with the schedule by season, as a cross-check.
- **Prediction**: for candidate X, season s and week w of 2 or later, fit X's rows of season s from
  weeks before w with `fit_qb_ratings(rows, ridge_lambda=lambda_X(s))`, where `lambda_X(s)` is the
  penalty `fit_qb_ratings` cross-validates on X's full season s-1 rows (2005's for 2006). For each
  team-game of week w with starter q and opponent o: `p_X = intercept + passer[q] - defense[o] +
  home_field * home_sign`, with 0 for a passer or defense without earlier rows or an unknown
  starter. The team term `t` is the ported team fit on the season's games before week w
  (`team_prior.snapshot_fit`: the previous season's penalties and the 9-game preseason prior, the
  fit behind the published rating history), as a predicted scrimmage EPA per play: that
  snapshot's scrimmage `intercept + offense[team] - defense[o] + home_field * home_sign`, the
  per-play effects behind `offense_rating` and `defense_rating` (a team without games in the
  snapshot takes its prior means, as `snapshot_fit` rates it). The calibrated prediction is
  `k1 * p_X + k2 * t + m`, unweighted least squares over X's earlier team-games (earlier weeks of
  s and every week from 2 of earlier seasons from 2006 on), with `k1 = k2 = m = 0` without rows.
- **Allowed information set**: season s's quarterback rows and play-by-play parts from weeks before
  w, season s-1's full-season rows (for the penalty), the team fit on the games before week w (which
  reads earlier seasons for its penalties and prior), the starter of week w's game (or none), and
  the target only for earlier weeks. Shared caveats, not leaks between candidates: nflverse's
  expected-points model is trained on many seasons, later ones included; the starter is known at the
  first snap, not before kickoff; who has a quarterback row comes from the players file (a player's
  latest position) and the season's weekly rosters; the target counts penalty-wiped runs but no
  penalty-wiped passes, as the ported team rating does.
- **Window**: prediction weeks 5 and later of the regular seasons 2007-2025 (19 seasons), on the
  team-games every candidate predicts (all of them). 2006 is a warm-up: predicted and used for
  calibration, never scored. Weeks 2-4 of every season are predicted only for calibration.
- **Inference**: per team-game paired differences `|error X| - |error Y|` for B - A, C - A and
  C - B; a bootstrap that resamples whole seasons (19 clusters) with replacement, 10,000
  resamples, seed 0 (`numpy.random.default_rng(0)`, one draw matrix), the same draws for every
  comparison and week band; the statistic is the mean over the drawn team-games; 98.33% percentile
  intervals (quantiles 1/120 and 119/120), 95% Bonferroni-adjusted for three comparisons. A
  t-interval on the 19 season means (18 degrees of freedom, same level) is a sensitivity extra.
- **Decision rule**: X beats Y when the overall interval of X - Y lies entirely below zero and
  neither of its week-band intervals (prediction weeks 5-8 and 9 on; same draws and level) lies
  entirely above zero. Y beats X is read from the same X - Y intervals with the signs flipped, the
  band guard included. **The recommendation is the first candidate in the order "every quarterback
  play", "plus scrambles", "passes and sacks" that no other candidate beats.** An inconclusive
  comparison is a tie. Every interval that excludes zero is reported, in either direction. The
  recommendation can tie the candidate it displaces; the report says so. The decision goes to the
  user either way; adoption is held.
- **Power**: printed with the results, seeded (seed 1), never changing the rule. A synthetic
  league reuses the window's structure (2005-2025: every team-game with its opponent, home side and
  scrimmage snaps; every quarterback row with a dropback and its real counts of dropbacks,
  scrambles, designed runs and charged botched snaps; the real starters). Each of 100 simulated
  windows draws, per season, team offense and defense effects (normal, mean 0, the standard
  deviations of the per-play scrimmage effects of the ported full-season team fits of 2007-2025),
  a pass effect per passer-season (the dropback-weighted standard deviation of A's full-season
  passer effects, 2007-2025), and a scramble effect and a designed-run effect per passer-season
  (standard deviations by the method of moments: the variance of passer-season EPA per scramble, or
  per designed run, over passer-seasons with at least 10 such plays, less the mean sampling
  variance, floored at 0). A play's EPA is its type's mean in the window plus the offense effect,
  less the defense effect, plus home field times the home sign, plus the passer's effect for his
  type (none for botched snaps and for the team's other scrimmage plays), plus normal noise at the
  per-play standard deviation the full-season team fits' residuals give. Each window runs through
  the same code as the real test (penalties, snapshots, calibration, bootstrap and rule), except
  that its team term is fit without the preseason prior: its seasons are drawn independently, so
  the prior's carryover would be about zero. It prints the share of windows in which B beats A, C
  beats A and C beats B.

## Integrity checks (run before reading any result; if one fails, stop and read nothing else)

1. Reproduction: A's full-season fit of every season 2005-2025, with its own cross-validated
   penalty, equals `adj_qb_epa_per_dropback` in the candidate build's
   `data/ratings/{season}_qb_ratings.parquet` to 1e-9.
2. Weekly reproduction: A's snapshot for week w, refit with the season fit's own penalty as the
   rating history uses it, equals the history's rows of the last week before w in
   `data/ratings/{season}_qb_ratings_by_week.parquet`, for every passer in both, to 1e-9,
   2006-2025.
3. Play parts: on every quarterback-game of 2005-2025 the play-by-play scramble and kneel counts
   equal `qb_scrambles` and `qb_kneels` of this repository's ported season statistics
   (`season_stats.game_logs.build_qb_game_stats`, an independent implementation of the source's
   game logs); designed runs plus aborted snaps by rusher equal `qb_designed_carries`, and their
   EPA `qb_designed_rush_epa`, to 1e-9; `qb_rushing_epa` equals the sum of the rushing parts and
   `qb_passing_epa` the passer's `qb_epa` over his passes and spikes, to 1e-6. Differences in
   1999-2004 are not checked (the window starts in 2005).
4. Plays within the target: in every team-game of the window, every play a candidate counts is a
   scrimmage snap of the target (the adapter's definition), and the target rebuilt from the
   play-by-play by an independent count equals `offensive_epa / offensive_snaps` to 1e-9.
5. Team term: for every snapshot, the per-play effects behind `t`, times the snapshot's plays per
   game, equal the `offense_rating` and `defense_rating` that `snapshot_fit` returns for it to
   1e-12, and its team ratings equal the published history's rows of the last week before w
   (`data/ratings/{season}_ratings_by_week.parquet`) for every team in both, to 1e-9.
6. Zero addition: with the added plays set to zero, B's and C's rows, snapshot fits and
   predictions equal A's to 1e-12.
7. Penalties: every snapshot of season s uses `lambda_X(s)`, recomputed separately from X's season
   s-1 rows.
8. Information set: replacing every row from week w on (plays and EPA, quarterback and team, and
   the later games' starters) with other values leaves the week-w predictions unchanged to 1e-12,
   on every snapshot; every calibration row is from an earlier week.
9. Rows: every candidate scores the same team-games; the check prints their count, the unknown
   starters and the starter disagreements with the schedule by season, and the quarterback runs on
   quarterback rows without a dropback by season.
10. Extras: A's year-over-year Pearson over 1999-2025, from the candidate build's
    `{season}_qb_ratings.parquet` (passers listed in both seasons), reproduces the source's
    validation figure, 0.455 over 601 pairs, to three decimals.
11. Inputs: the report records the SHA-256 of every file the decision reads and a fingerprint of
    the play-by-play parts and starters it builds (sorted, as CSV).

## Descriptive extras (never decision inputs)

MAE and RMSE overall and by band with the paired intervals; each candidate's calibration slopes
`k1` and `k2` by band; the one-term model (`k * p_X + m`); the team-only baseline (`k2 * t + m`);
the starter-change band (team-games whose starter differs from the team's previous game's that
season) with its paired intervals; the t-interval sensitivity; the same comparisons on points
scored (`points_for`, calibrated to points); the starter's own EPA per dropback (A's play set) and
per quarterback play (C's), play-weighted, each labeled with the candidate it favors; C without the
botched snaps, on every metric; year-over-year Pearson and Spearman of each candidate's
full-season rating over consecutive seasons 2006-2025, passers qualifying in both; each
candidate's 2025 `qb_faced_pass_defense` and its correlation with A's; and the 2025 ranks of Drake
Maye and Josh Allen under each candidate.

## Departures from the source protocol

1. **Data**: this repository's play-by-play and nflverse player sources, read through its loaders
   from the candidate build's cache (`~/scratch/step6/data/cache/nflreadpy/`, read-only, downloads
   blocked), and the ported adapter's rows, instead of the source's published game logs. The
   adapter reproduces those logs in every input cell (`ADAPTER_PARITY.md`).
2. **Reference files of the integrity checks**: this repository's own outputs of the candidate
   build (`~/scratch/step6/data/ratings/`) for checks 1, 2, 5 and 10, and its ported season
   statistics for check 3, because this task reads `nfl-sos-ratings` only at its frozen commit,
   whose `data/` is not in git. Check 4 recounts the target from the play-by-play instead of
   reading `epa_per_offensive_snap`, and check 5 compares the team term with the published rating
   history instead of the source's `validate` rows (the ported history comes from the same
   snapshot fit).
3. **The 60 botched snaps charged to a lineman stay in A** (and so in B and C) as the rating is
   ported today. Removing them was approved but is not implemented; the botched-snap measurement
   (`botched_snaps.md` beside this file) reports what it changes. They sit identically in all three
   candidates, so they cannot favor one.
4. **Extras not run**: the per-season correlations with ESPN QBR (no QBR file is cached in the
   inputs this task may read, and the script must not download), the QBR half of integrity check 10
   for the same reason, and the passer holdout (it needs postseason quarterback rows, which this
   repository's adapter does not build).
5. **Made concrete here**: the power estimate's generator (above), which the source left at
   "spreads the team fit and A's fit show"; and `t` for a team without games in a snapshot (its
   prior means, as the ported snapshot rates it).

## Command and outputs

```bash
VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/qb_plays_test.py \
    --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy \
    --ratings-dir /home/mitch/scratch/step6/data/ratings
```

It writes `.agents/sos_port/QB_PLAYS_TEST.md` and `.agents/sos_port/qb_plays_test.json` and nothing
else, and runs with one BLAS thread.
