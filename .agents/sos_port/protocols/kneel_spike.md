# Protocol: kneel-downs and spikes in the team rating

Pre-registered before any candidate's error is computed, and committed on its own. The decision on
record (reported by the `nfl-sos-ratings` session as the maintainer's of 2026-10-09,
`.agents/sos_port_source.md`, "Handoff after the freeze", item 3) is that kneel-downs and spikes
leave the team ratings only after a pre-registered test, modeled on that project's garbage-time
filter test (its roadmap section "WP4", at `9081bce`), which found that leaving out
low-win-probability plays did not improve predictions. This is that test, run on the ported team
rating. The result is report-only: adopting the candidate changes every team rating, so every
column of the `ratings` feature group, and it is the user's decision after this test.

## Background

The ported team rating (`src/nfl_predictor/ratings/team_rating.py`) fits each team-game's
scrimmage EPA per play: a scrimmage snap is a dropback, a run (`rush`), a kneel-down (`qb_kneel`)
or a spike (`qb_spike`), with a run play flagged `qb_scramble` counted as a dropback
(`ratings/pbp_inputs.team_games_from_pbp`). Kneel-downs are called to run out the clock once a game
is decided, and spikes to stop it; their EPA reflects the clock, not the offense or the defense.
The quarterback rating already leaves both out.

## The test

- **Hypothesis (falsifiable)**: rating teams on their scrimmage plays without kneel-downs and spikes
  predicts game margins out of sample better than rating them on every scrimmage play. It is
  refuted unless the candidate's paired interval lies entirely below zero.
- **Candidates, fixed in advance**: "every scrimmage snap" (today's rating) and "without kneel-downs
  and spikes": every snap flagged `qb_kneel` or `qb_spike` leaves its team-game's
  `offensive_snaps` and `offensive_epa`. Special-teams plays and EPA, the margins and the set of
  team-games stay as they are.
- **Estimator**: each candidate is the ported team fit run unchanged on its plays: each season's
  ridge penalties cross-validated on the candidate's own previous-season rows (1999 cross-validates
  its own), the preseason prior built from the candidate's own earlier seasons
  (`team_prior.PriorHistory` over its rows, the 9-game horizon), and its own per-game scale. Each
  candidate gets its own margin model, so a scale difference cannot help or hurt it.
- **Harness**: a port of the source's walk-forward validation
  (`nfl_sos_ratings/validation/walk_forward.py` at `9081bce`: `build_snapshot_feature_rows`,
  `build_team_rating_feature_rows`, `evaluate_feature_rows`). For every week, each candidate is
  rebuilt from the season's games before the week (`team_prior.snapshot_fit`); a week without
  earlier games gets a rating gap of 0. The margin model is `home_margin = k * rating_gap +
  home_edge`, least squares on all of the candidate's earlier rows (earlier weeks of the season and
  every week of earlier seasons from 1999).
- **Allowed information set**: to predict week w of season s, a candidate sees season s's
  team-game rows from weeks before w and earlier seasons' rows (for its penalties and prior); its
  margin model sees only its own earlier predictions. Shared caveat, not a leak between candidates:
  nflverse's expected-points model is trained on many seasons, later ones included.
- **Metric and window**: mean absolute error of the predicted home margin over prediction weeks 5
  and later of the regular seasons 1999-2025, the window of the source's validation
  (`docs/validation-report.md` at `9081bce`: 5,297 games), on the games every candidate predicts.
- **Inference**: per-game paired differences `|error without| - |error today|`; a bootstrap that
  resamples whole seasons (27 clusters) with replacement, keeping every game of a drawn season
  together; 10,000 resamples, seed 0 (`numpy.random.default_rng(0)`, one draw matrix), the same
  draws for every band; the statistic is the mean over the drawn games; 95% percentile intervals
  (one comparison).
- **Decision rule**: the candidate qualifies only if its overall interval lies entirely below zero
  and neither of its week-band intervals (prediction weeks 5-7 and 8 on, the source validation's
  early and late splits; same draws and level) lies entirely above zero. If it qualifies, the
  recommendation is to leave kneel-downs and spikes out of the team rating. Otherwise the result is
  a tie (or a loss) and the recommendation is today's rating, every scrimmage snap: the simpler
  option, which judges no play by its situation (as "no filter" was in the garbage-time test), and
  the one the decision on record keeps unless this test supports the change. Every interval that
  excludes zero is reported, in either direction. The decision goes to the user either way;
  adoption is held.

## Integrity checks (run before reading any result; if one fails, stop and read nothing else)

1. Today's rating is the production rating: for every season 1999-2025, today's full-season fit
   (the previous season's penalties and the prior, as published) equals the four rating columns of
   the candidate build's `data/ratings/{season}_ratings.parquet`, and for every season 2000-2025
   every snapshot's team ratings equal the published history's rows of the last week before it
   (`data/ratings/{season}_ratings_by_week.parquet`, teams in both), to 1e-9. 1999 has no earlier
   season, so the source's harness cross-validates each 1999 snapshot's own penalties while the
   published 1999 history reuses the full-season fit's; its snapshots are checked against a
   separate cross-validated fit of the same games instead. (Amended before any scoring: the first
   version asked the history check of 1999 too, which cannot hold.)
2. The harness reproduces the source's validation: today's predictions give 5,297 games and an
   overall MAE of 10.575, 10.567 over the 1,141 games of weeks 5-7 and 10.577 over the 4,156 of
   weeks 8 on, to three decimals.
3. Candidate rows: per team-game, the candidate's snaps and EPA equal today's less the kneel-downs
   and spikes, recounted independently from the play-by-play; the special-teams columns, margins
   and team-games are unchanged.
4. Zero removal: with no play removed, the candidate's rows, snapshots and predictions equal
   today's to 1e-12.
5. Penalties: each candidate's season penalties equal a separate cross-validation of its previous
   season's rows.
6. Information set: replacing every row from week w on with other values leaves each candidate's
   week-w snapshot unchanged, on every snapshot; every margin-model row is from an earlier week.
7. Inputs: the report records the SHA-256 of every file the decision reads.

## Descriptive extras (never decision inputs)

MAE and RMSE overall and by band, with the paired intervals; the same comparison from a bootstrap
that resamples single games (the source validation's kind); each candidate's fitted margin slope
`k` by band; the two halves alone, "without kneel-downs" and "without spikes"; the removed plays'
count and EPA by season; the team year-over-year Pearson of full-season `team_rating`; the 2025
ratings and ranks under each candidate; and the feature-value impact: per snapshot, how far each
team's pre-week `team_rating`, `offense_rating` and `defense_rating` move.

## Departures from the source's garbage-time test

- The bootstrap resamples whole seasons, as the source's later preseason-prior test did, instead of
  single games: games of one week share a rating snapshot and a season shares its penalties and
  prior. The single-game bootstrap is a descriptive extra.
- One comparison, so 95% intervals; the garbage-time test compared three thresholds at 98.33%.
- A week-band guard, as the source's preseason-prior and quarterback tests use.
- Inputs are this repository's play-by-play (the candidate build's cache, read through its loaders,
  downloads blocked) and the ported adapter's rows, which reproduce the source's game logs in every
  input cell (`ADAPTER_PARITY.md`).

## Command and outputs

```bash
VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/team_plays_test.py \
    kneel-spike --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy \
    --ratings-dir /home/mitch/scratch/step6/data/ratings
```

It writes `.agents/sos_port/KNEEL_SPIKE_TEST.md` and `.agents/sos_port/kneel_spike_test.json` and
nothing else, and runs with one BLAS thread.
