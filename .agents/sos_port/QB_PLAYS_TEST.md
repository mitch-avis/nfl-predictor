# The quarterback rating from all of a quarterback's plays: test result

Pre-registered in `protocols/qb_all_plays.md` before any candidate was scored, a port of
`nfl-sos-ratings`' roadmap section Q1 at `9081bce`. Candidates: passes and sacks (the rating as
ported), plus scrambles, and every quarterback play.

## Headline

- MAE of a team's offensive EPA per play over 7,496 team-games (2007-2025, weeks 5 on): passes and
  sacks 0.146628; plus scrambles 0.146634; every quarterback play 0.146640.
- plus scrambles - passes and sacks: +6.59e-06 (-1.85e-05 to +3.07e-05).
- every quarterback play - passes and sacks: +1.25e-05 (-4.76e-05 to +7.87e-05).
- every quarterback play - plus scrambles: +5.87e-06 (-4.12e-05 to +6.70e-05).
- Rule as written: wins: none. The recommendation is **every quarterback play**, the first candidate
  in the user's order (every quarterback play, plus scrambles, passes and sacks) that no other
  candidate beats.

## Decision

- Intervals that exclude zero, either direction: none.
- The decision goes to the user; adoption is held (it changes the `ratings` feature group and the
  published quarterback tables).

## Paired differences (98.33% season-bootstrap intervals, 10,000 resamples, seed 0)

Mean of `|error X| - |error Y|` per team-game; negative favors X.

| pair | overall | weeks 5-8 | weeks 9+ |
| --- | --- | --- | --- |
| plus scrambles - passes and sacks | +6.59e-06 (-1.85e-05 to +3.07e-05) | +3.15e-05 (-5.79e-05 to +1.14e-04) | -3.44e-06 (-4.13e-05 to +3.72e-05) |
| every quarterback play - passes and sacks | +1.25e-05 (-4.76e-05 to +7.87e-05) | +8.38e-05 (-4.43e-05 to +2.29e-04) | -1.62e-05 (-8.16e-05 to +7.40e-05) |
| every quarterback play - plus scrambles | +5.87e-06 (-4.12e-05 to +6.70e-05) | +5.22e-05 (-2.71e-05 to +1.41e-04) | -1.28e-05 (-5.81e-05 to +5.11e-05) |

## Integrity checks (all passed before any result was read)

1. Passes and sacks' full-season fits of 2005-2025 equal the candidate build's
   `{season}_qb_ratings.parquet` (largest gap 0.0e+00).
2. Its snapshots refit with the season's own penalty equal the published history of the week before
   in 325 weeks of 2006-2025 (largest gap 0.0e+00). The history is read only here, to check the
   port: every prediction uses the previous season's penalty.
3. On 13,070 quarterback-games of 2005-2025 the play parts equal the ported season statistics: 0
   scramble, kneel or carry mismatches, 0 rushing-EPA or passing-EPA gaps.
4. Every added play is a scrimmage snap (0 outside), and the target recounted from the play-by-play
   equals the adapter's (largest gap 0.0e+00).
5. The team term's effects equal its snapshot's ratings (largest gap 0.0e+00), and 325 snapshots
   equal the published team history of the week before (largest gap 0.0e+00).
6. With the added plays zeroed, the rows equal passes and sacks' and the predictions differ by at
   most 0.0e+00.
7. 100 season penalties equal separate cross-validations of each candidate's previous season.
8. On 325 weekly snapshots, replacing every later game's values moves no prediction (largest gap
   0.0e+00); every snapshot equals `qb_rating.fit_qb_ratings` on the same rows (largest gap
   0.0e+00); the calibrations of 5,200 candidate-weeks, over every target, were each fit on exactly
   the rows of the earlier weeks.
9. Every candidate scores the same team-games; by season below.
10. Passes and sacks' year-over-year Pearson from the published tables is 0.4551 over 601 pairs (the
    source: 0.455 over 601). The QBR half is not run (no QBR file among the allowed inputs).
11. Every input's SHA-256 is recorded; the play parts and starters fingerprint is
    `3cdc9e7bae83e2a99b620723f782b171846cfc762027a723632f653ffdb69560`.

| season | team_games | unknown_starters | listed_without_qb_row | first_snap_differs | qb_rows_without_dropback | runs_on_those_rows |
| --- | --- | --- | --- | --- | --- | --- |
| 2006 | 512 | 0 | 0 | 4 | 0 | 0 |
| 2007 | 512 | 0 | 0 | 2 | 0 | 0 |
| 2008 | 512 | 1 | 0 | 2 | 0 | 0 |
| 2009 | 512 | 0 | 0 | 2 | 0 | 0 |
| 2010 | 512 | 0 | 0 | 0 | 0 | 0 |
| 2011 | 512 | 0 | 0 | 2 | 0 | 0 |
| 2012 | 512 | 0 | 0 | 1 | 0 | 0 |
| 2013 | 512 | 0 | 0 | 0 | 25 | 10 |
| 2014 | 512 | 0 | 0 | 1 | 29 | 5 |
| 2015 | 512 | 1 | 0 | 1 | 35 | 6 |
| 2016 | 512 | 1 | 0 | 3 | 22 | 2 |
| 2017 | 512 | 1 | 0 | 2 | 20 | 0 |
| 2018 | 512 | 0 | 0 | 1 | 45 | 30 |
| 2019 | 512 | 0 | 0 | 1 | 44 | 17 |
| 2020 | 512 | 0 | 0 | 1 | 47 | 46 |
| 2021 | 544 | 0 | 0 | 1 | 64 | 33 |
| 2022 | 542 | 0 | 4 | 8 | 34 | 19 |
| 2023 | 544 | 0 | 0 | 0 | 52 | 36 |
| 2024 | 544 | 0 | 31 | 37 | 48 | 18 |
| 2025 | 544 | 0 | 7 | 8 | 52 | 25 |

## Power (descriptive, never changes the rule)

Over 100 synthetic windows (seed 1), the share in which the rule finds each win: plus scrambles
beats passes and sacks 4%; every quarterback play beats passes and sacks 1%; every quarterback play
beats plus scrambles 0%.

Spreads per play, from 2007-2025: team offense 0.0613, defense 0.0463, home field -0.0040; passer
0.0919, scramble 0.1159, designed run 0.1508; play noise 1.3051. Means: dropback +0.035, scramble
+0.451, designed +0.216, botched -2.246, other -0.124.

## Descriptive extras (never decision inputs)

MAE and RMSE by band (every quarterback play without the botched snaps included):

| candidate | band | team-games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| passes and sacks | overall | 7,496 | 0.146628 | 0.185887 |
| passes and sacks | weeks 5-8 | 2,150 | 0.145539 | 0.184353 |
| passes and sacks | weeks 9+ | 5,346 | 0.147066 | 0.186500 |
| plus scrambles | overall | 7,496 | 0.146634 | 0.185891 |
| plus scrambles | weeks 5-8 | 2,150 | 0.145570 | 0.184363 |
| plus scrambles | weeks 9+ | 5,346 | 0.147062 | 0.186502 |
| every quarterback play | overall | 7,496 | 0.146640 | 0.185886 |
| every quarterback play | weeks 5-8 | 2,150 | 0.145622 | 0.184409 |
| every quarterback play | weeks 9+ | 5,346 | 0.147050 | 0.186476 |
| every quarterback play without the botched snaps | overall | 7,496 | 0.146687 | 0.185947 |
| every quarterback play without the botched snaps | weeks 5-8 | 2,150 | 0.145566 | 0.184404 |
| every quarterback play without the botched snaps | weeks 9+ | 5,346 | 0.147138 | 0.186565 |

Mean fitted calibration slopes, `k1` on the quarterback term and `k2` on the team term:

| candidate | band | k1 | k2 |
| --- | --- | --- | --- |
| passes and sacks | overall | -0.071 | 0.996 |
| passes and sacks | weeks 5-8 | -0.078 | 1.001 |
| passes and sacks | weeks 9+ | -0.068 | 0.995 |
| plus scrambles | overall | -0.063 | 0.986 |
| plus scrambles | weeks 5-8 | -0.070 | 0.991 |
| plus scrambles | weeks 9+ | -0.060 | 0.984 |
| every quarterback play | overall | -0.048 | 0.970 |
| every quarterback play | weeks 5-8 | -0.057 | 0.977 |
| every quarterback play | weeks 9+ | -0.045 | 0.967 |
| every quarterback play without the botched snaps | overall | -0.073 | 0.990 |
| every quarterback play without the botched snaps | weeks 5-8 | -0.082 | 0.997 |
| every quarterback play without the botched snaps | weeks 9+ | -0.069 | 0.988 |

One-term model (`k * p_X + m`, the source draft's primary):

| candidate | band | team-games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| passes and sacks | overall | 7,496 | 0.149285 | 0.188663 |
| passes and sacks | weeks 5-8 | 2,150 | 0.150525 | 0.189299 |
| passes and sacks | weeks 9+ | 5,346 | 0.148787 | 0.188406 |
| plus scrambles | overall | 7,496 | 0.148907 | 0.188192 |
| plus scrambles | weeks 5-8 | 2,150 | 0.150316 | 0.189130 |
| plus scrambles | weeks 9+ | 5,346 | 0.148340 | 0.187813 |
| every quarterback play | overall | 7,496 | 0.148437 | 0.187724 |
| every quarterback play | weeks 5-8 | 2,150 | 0.149958 | 0.188984 |
| every quarterback play | weeks 9+ | 5,346 | 0.147826 | 0.187215 |
| every quarterback play without the botched snaps | overall | 7,496 | 0.148651 | 0.187932 |
| every quarterback play without the botched snaps | weeks 5-8 | 2,150 | 0.150132 | 0.188933 |
| every quarterback play without the botched snaps | weeks 9+ | 5,346 | 0.148055 | 0.187527 |

| pair | overall | weeks 5-8 | weeks 9+ |
| --- | --- | --- | --- |
| plus scrambles - passes and sacks | -3.78e-04 (-7.90e-04 to -3.09e-05) | -2.09e-04 (-9.33e-04 to +6.10e-04) | -4.46e-04 (-8.52e-04 to -7.30e-05) |
| every quarterback play - passes and sacks | -8.48e-04 (-1.38e-03 to -3.55e-04) | -5.66e-04 (-1.57e-03 to +4.95e-04) | -9.61e-04 (-1.49e-03 to -3.86e-04) |
| every quarterback play - plus scrambles | -4.70e-04 (-7.59e-04 to -2.09e-04) | -3.58e-04 (-7.26e-04 to -1.82e-05) | -5.15e-04 (-9.13e-04 to -1.20e-04) |

Team-only baseline (`k2 * t + m`): MAE 0.146512 over 7,496 team-games.

Starter-change band (the starter differs from the team's previous game's), season bootstrap on the
seasons it holds:

- plus scrambles - passes and sacks: -8.47e-05 (-1.80e-04 to +1.03e-05).
- every quarterback play - passes and sacks: -7.23e-05 (-2.62e-04 to +1.25e-04).
- every quarterback play - plus scrambles: +1.25e-05 (-1.39e-04 to +1.95e-04).

t-interval sensitivity on the 19 season means (same level):

- plus scrambles - passes and sacks: +6.46e-06 (-2.21e-05 to +3.50e-05).
- every quarterback play - passes and sacks: +1.26e-05 (-5.97e-05 to +8.49e-05).
- every quarterback play - plus scrambles: +6.15e-06 (-5.75e-05 to +6.98e-05).

Points scored (`points_for`, calibrated to points):

| candidate | band | team-games | MAE | RMSE |
| --- | --- | --- | --- | --- |
| passes and sacks | overall | 7,496 | 7.824987 | 9.821158 |
| passes and sacks | weeks 5-8 | 2,150 | 7.750440 | 9.737885 |
| passes and sacks | weeks 9+ | 5,346 | 7.854968 | 9.854449 |
| plus scrambles | overall | 7,496 | 7.824378 | 9.819756 |
| plus scrambles | weeks 5-8 | 2,150 | 7.753931 | 9.740984 |
| plus scrambles | weeks 9+ | 5,346 | 7.852710 | 9.851258 |
| every quarterback play | overall | 7,496 | 7.822404 | 9.817310 |
| every quarterback play | weeks 5-8 | 2,150 | 7.749745 | 9.738204 |
| every quarterback play | weeks 9+ | 5,346 | 7.851625 | 9.848945 |
| every quarterback play without the botched snaps | overall | 7,496 | 7.823309 | 9.818754 |
| every quarterback play without the botched snaps | weeks 5-8 | 2,150 | 7.746750 | 9.734799 |
| every quarterback play without the botched snaps | weeks 9+ | 5,346 | 7.854098 | 9.852316 |

| pair | overall | weeks 5-8 | weeks 9+ |
| --- | --- | --- | --- |
| plus scrambles - passes and sacks | -6.09e-04 (-3.60e-03 to +2.19e-03) | +3.49e-03 (-2.21e-03 to +1.14e-02) | -2.26e-03 (-5.47e-03 to +6.36e-04) |
| every quarterback play - passes and sacks | -2.58e-03 (-6.89e-03 to +1.76e-03) | -6.95e-04 (-9.75e-03 to +1.07e-02) | -3.34e-03 (-9.08e-03 to +2.73e-03) |
| every quarterback play - plus scrambles | -1.97e-03 (-5.01e-03 to +1.63e-03) | -4.19e-03 (-1.04e-02 to +1.31e-03) | -1.08e-03 (-5.54e-03 to +3.70e-03) |

The starter's own EPA per play in the game, play-weighted MAE (each target is one candidate's play
set, so it favors that candidate):

- per dropback, which favors passes and sacks: passes and sacks 0.233069; plus scrambles 0.233101;
  every quarterback play 0.233093; every quarterback play without the botched snaps 0.233218.
- per quarterback play, which favors every quarterback play: passes and sacks 0.221739; plus
  scrambles 0.221737; every quarterback play 0.221679; every quarterback play without the botched
  snaps 0.221780.

Every quarterback play without the botched snaps, same draws and level:

| pair | overall | weeks 5-8 | weeks 9+ |
| --- | --- | --- | --- |
| every quarterback play without the botched snaps - passes and sacks | +5.93e-05 (-9.47e-06 to +1.94e-04) | +2.76e-05 (-5.78e-05 to +1.10e-04) | +7.20e-05 (-2.00e-05 to +2.53e-04) |
| every quarterback play - every quarterback play without the botched snaps | -4.68e-05 (-1.38e-04 to +2.63e-05) | +5.62e-05 (-6.83e-05 to +2.40e-04) | -8.82e-05 (-1.87e-04 to -3.12e-05) |

Year-over-year stability of each full-season rating, 2006-2025, passers qualifying in both seasons:

- passes and sacks: Pearson 0.459, Spearman 0.439 (446 pairs).
- plus scrambles: Pearson 0.445, Spearman 0.424 (446 pairs).
- every quarterback play: Pearson 0.454, Spearman 0.431 (446 pairs).
- every quarterback play without the botched snaps: Pearson 0.449, Spearman 0.427 (446 pairs).

2025: faced pass defense against passes and sacks' (Pearson over qualified passers), and two
passers' ranks and ratings:

- passes and sacks: faced-defense correlation 1.000; Drake Maye 1 (+0.209); Josh Allen 11 (+0.102).
- plus scrambles: faced-defense correlation 0.995; Drake Maye 1 (+0.254); Josh Allen 8 (+0.149).
- every quarterback play: faced-defense correlation 0.988; Drake Maye 1 (+0.243); Josh Allen 6
  (+0.170).
- every quarterback play without the botched snaps: faced-defense correlation 0.987; Drake Maye 1
  (+0.247); Josh Allen 5 (+0.186).

Feature-value impact: each candidate's pre-week rating of the starter against passes and sacks' (the
fit behind the `ratings_qb` feature, with the season's identity):

- plus scrambles: 9,599 team-games, mean absolute change 0.0165, largest 0.0845 EPA per play, 2,850
  beyond 0.02.
- every quarterback play: 9,599 team-games, mean absolute change 0.0154, largest 0.1042 EPA per
  play, 2,587 beyond 0.02.
- every quarterback play without the botched snaps: 9,599 team-games, mean absolute change 0.0204,
  largest 0.1134 EPA per play, 3,997 beyond 0.02.

## Not run

- The per-season correlations with ESPN QBR and the QBR half of check 10: no QBR file is cached
  among the inputs this test may read, and the script does not download.
- The passer holdout: it needs postseason quarterback rows, which this repository's adapter does not
  build.

## Provenance

- nfl-predictor 0.57.1 at commit `0fdf79cee44dbaec2554315706071ea1bd88d855`; code SHA-256:
  `qb_plays_test.py` 070baac272c0, `rating_tests_common.py` 50da9bf186e7.
- Inputs: 179 files, fingerprint `dd46eecd2b1010d053ad8a520ea2e3ede9d3428b914180ce64173b8316871a55`
  (each file's SHA-256 is in `qb_plays_test.json`), read only, downloads blocked.
- Command: `VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/qb_plays_test.py
  --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy --ratings-dir
  /home/mitch/scratch/step6/data/ratings --schedules-file
  /home/mitch/scratch/nflreadpy_cache_snapshot_20261008/a7d0d7a2405b3d35047e8de4a40035e6.parquet`;
  1237 seconds.
