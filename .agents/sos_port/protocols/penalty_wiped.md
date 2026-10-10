# Protocol: penalty-wiped plays in the team rating

Pre-registered before any candidate's error is computed, and committed on its own. It uses the
design of the kneel-down and spike test beside this file (`kneel_spike.md`): the same harness,
window, bootstrap, band guard and integrity checks, with the changes stated here. The result is
report-only: adopting a variant changes every team rating, so every column of the `ratings` feature
group, and it is the user's decision after this test.

## Background

A play a penalty wiped out is recorded as `play_type == "no_play"`. The ported team rating's
scrimmage snap (`ratings/pbp_inputs.team_games_from_pbp`, the source's definition) counts a play by
its flags, and nflverse keeps the `rush` flag on a wiped-out run but sets no dropback flag on a
wiped-out pass. So today the rating keeps every penalty-wiped run (8,848 plays, -5,660 EPA,
1999-2025) and leaves out every penalty-wiped pass (27,906 plays, +11,142 EPA), counts taken by the
phase-1 parity check (`decisions/integration_pbp_elo.md`, entry 2.3). Nothing in the source chose
that asymmetry; it follows from the flags. The quarterback rating counts no penalty-wiped play,
and since `0.54.0` neither do this repository's carries.

## Variants, fixed in advance

Definitions, on regular-season plays with both teams known:

- a **penalty-wiped run** is a `no_play` play today's scrimmage-snap rule counts (in practice the
  `rush` flag; the script prints them by flag);
- a **penalty-wiped pass** is a `no_play` play with the `pass` flag that today's rule does not
  count.

The four variants, each the ported team fit run unchanged on its plays:

| variant | penalty-wiped runs | penalty-wiped passes |
| --- | --- | --- |
| today's rating | counted | left out |
| every wiped snap | counted | counted |
| no wiped snap | left out | left out |
| wiped passes only | left out | counted |

A counted play adds one snap and its EPA to its team-game's `offensive_snaps` and
`offensive_epa`; a play left out takes them away. Special-teams plays and EPA, the margins and the
set of team-games stay as they are.

## The test

- **Hypothesis (falsifiable)**: for each of the three variants, rating teams on its play set
  predicts game margins out of sample better than today's rating. It is refuted for a variant
  unless that variant's paired interval lies entirely below zero.
- **Estimator, harness, information set, metric and window**: as in `kneel_spike.md`: each
  variant's own penalties and preseason prior from its own rows, the source validation's
  walk-forward with its own margin model, MAE of the predicted home margin over prediction weeks 5
  and later of 1999-2025, on the games every variant predicts.
- **Inference**: per-game paired differences `|error variant| - |error today|` for the three
  variants; a season-cluster bootstrap (27 clusters), 10,000 resamples, seed 0, the same draws for
  every variant and band; 98.33% percentile intervals (quantiles 1/120 and 119/120), 95%
  Bonferroni-adjusted for three comparisons.
- **Decision rule**: a variant qualifies only if its overall interval lies entirely below zero and
  neither of its week-band intervals (prediction weeks 5-7 and 8 on; same draws and level) lies
  entirely above zero. The recommendation is the qualifying variant with the lowest overall MAE.
  If none qualifies, the test cannot tell the variants from today's, and the recommendation is the
  first definition in the order of simplicity below whose overall interval does not lie entirely
  above zero (today's always meets that). Every interval that excludes zero is reported, in either
  direction. The decision goes to the user either way; adoption is held.
- **Order of simplicity** (decides only when the test cannot tell the variants apart): every wiped
  snap, no wiped snap, today's rating, wiped passes only. Reasons: the two symmetric definitions
  treat a wiped run and a wiped pass alike, and today's asymmetry is an accident of nflverse's
  flags, not a choice; between them, "every wiped snap" counts every snap the offense ran, which is
  the rating's own definition (a scrimmage play, wiped or not), and the user's stated preference for
  the quarterback test was all available data when the test cannot decide. "Wiped passes only" is
  the reverse asymmetry and comes last.

## Integrity checks

Checks 1, 2, 5, 6 and 7 of `kneel_spike.md`, unchanged. Checks 3 and 4 become:

- Check 3, variant rows: per team-game, each variant's snaps and EPA equal today's plus the wiped
  passes it counts and less the wiped runs it leaves out, recounted independently from the
  play-by-play; the special-teams columns, margins and team-games are unchanged; the wiped-run and
  wiped-pass sets are disjoint, and the counts and EPA over 1999-2025 are printed beside the
  phase-1 figures (8,848 at -5,660.46 and 27,906 at +11,142.27; a difference is reported with its
  cause, since the phase-1 count took every `no_play` play, both teams known or not).
- Check 4, zero change: with no play added or removed, each variant's rows, snapshots and
  predictions equal today's to 1e-12.

## Descriptive extras (never decision inputs)

As in `kneel_spike.md` (MAE and RMSE by band with intervals, the single-game bootstrap, the margin
slope by band, the moved plays' count and EPA by season, team year-over-year Pearson, the 2025
ratings and ranks, the feature-value impact), for every variant; plus the moved plays by the
penalty's side (`penalty_team` the offense or the defense) with their EPA, to show what each set
holds.

## Command and outputs

```bash
VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/team_plays_test.py \
    penalty-wiped --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy \
    --ratings-dir /home/mitch/scratch/step6/data/ratings
```

It writes `.agents/sos_port/PENALTY_WIPED_TEST.md` and `.agents/sos_port/penalty_wiped_test.json`
and nothing else, and runs with one BLAS thread.
