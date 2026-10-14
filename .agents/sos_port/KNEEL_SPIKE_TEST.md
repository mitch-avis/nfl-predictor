# Kneel-downs and spikes in the team rating: test result

Pre-registered in `protocols/kneel_spike.md` before any variant was scored. Next-game margin MAE,
prediction weeks 5 and later of 1999-2025, the ported team rating rebuilt each week from the
season's earlier games for every variant.

## Headline

- without kneel-downs and spikes: MAE 10.554 against today's 10.575 over 5,297 games; paired
  difference -0.021 (-0.039 to +0.0005).

## Decision

Rule as written (95.00% season-bootstrap intervals): no variant qualified, so the test cannot tell
the variants from today's rating; the recommendation is **today's rating (every scrimmage snap)**,
the first definition in the order of simplicity that is not significantly worse. The order: today's
rating (every scrimmage snap); without kneel-downs and spikes.

- Intervals that exclude zero, either direction: without kneel-downs and spikes, weeks 8+: better.
- Overall below zero but disqualified by a week band: none.
- The decision goes to the user; adoption is held (it changes every `ratings` column).

## Paired differences from today's rating

MAE of the variant minus today's (negative favors the variant); 95.00% percentile intervals from
10,000 resamples of whole seasons (27 clusters), seed 0, the same draws for every variant and band.

| variant | overall | weeks 5-7 | weeks 8+ |
| --- | --- | --- | --- |
| without kneel-downs and spikes | -0.021 (-0.039 to +0.0005) | -0.027 (-0.083 to +0.022) | -0.019 (-0.035 to -0.001) |

## Integrity checks (all passed before any result was read)

1. Today's rating is the production rating: the full-season ratings of 1999-2025 equal the candidate
   build's `{season}_ratings.parquet` (largest gap 0.0e+00); 421 snapshots of 2000-2025 equal the
   published history of the week before (largest gap 0.0e+00); the 1999 snapshots equal separate
   cross-validated fits of the same games (largest gap 0.0e+00).
2. The ported harness reproduces the source's validation: overall 5,297 games, MAE 10.575; weeks 5-7
   1,141 games, MAE 10.567; weeks 8+ 4,156 games, MAE 10.577 (the source: 5,297 at 10.575, 1,141 at
   10.567, 4,156 at 10.577).
3. Every variant's rows are today's with its plays moved, recounted from the play-by-play (largest
   EPA gap 7.1e-15); special-teams columns, margins and team-games unchanged; no play is both a
   wiped run and a wiped pass. Moved: without kneel-downs and spikes: 11,282 plays out (-5,516.2
   EPA), 0 in (+0.0 EPA); without kneel-downs: 9,404 plays out (-5,266.9 EPA), 0 in (+0.0 EPA);
   without spikes: 1,878 plays out (-249.3 EPA), 0 in (+0.0 EPA).
4. With no play moved, the rows equal the adapter's exactly and the features and predictions equal
   today's (largest gap 0.0e+00).
5. 104 season penalties equal separate cross-validations of each variant's previous season.
6. Snapshots unchanged when every later game is replaced: today's rating (every scrimmage snap) 437,
   without kneel-downs and spikes 437, without kneel-downs 437, without spikes 437; all 26,485
   predictions trained on earlier rows only.
7. Every input file's SHA-256 is recorded (Provenance).

## Descriptive extras (not decision inputs)

MAE, RMSE and the mean fitted margin slope `k`:

| variant | band | games | MAE | RMSE | mean k |
| --- | --- | --- | --- | --- | --- |
| today's rating (every scrimmage snap) | overall | 5,297 | 10.575 | 13.562 | 0.903 |
| today's rating (every scrimmage snap) | weeks 5-7 | 1,141 | 10.567 | 13.691 | 0.911 |
| today's rating (every scrimmage snap) | weeks 8+ | 4,156 | 10.577 | 13.526 | 0.900 |
| without kneel-downs and spikes | overall | 5,297 | 10.554 | 13.533 | 0.858 |
| without kneel-downs and spikes | weeks 5-7 | 1,141 | 10.540 | 13.658 | 0.867 |
| without kneel-downs and spikes | weeks 8+ | 4,156 | 10.558 | 13.499 | 0.856 |
| without kneel-downs | overall | 5,297 | 10.555 | 13.534 | 0.857 |
| without kneel-downs | weeks 5-7 | 1,141 | 10.541 | 13.660 | 0.866 |
| without kneel-downs | weeks 8+ | 4,156 | 10.558 | 13.500 | 0.854 |
| without spikes | overall | 5,297 | 10.574 | 13.561 | 0.904 |
| without spikes | weeks 5-7 | 1,141 | 10.566 | 13.689 | 0.913 |
| without spikes | weeks 8+ | 4,156 | 10.577 | 13.525 | 0.902 |

Single-game bootstrap of the same differences (95.00%):

| variant | overall | weeks 5-7 | weeks 8+ |
| --- | --- | --- | --- |
| without kneel-downs and spikes | -0.021 (-0.035 to -0.007) | -0.027 (-0.064 to +0.009) | -0.019 (-0.034 to -0.004) |
| without kneel-downs | -0.020 (-0.034 to -0.007) | -0.026 (-0.062 to +0.011) | -0.019 (-0.034 to -0.004) |
| without spikes | -0.001 (-0.002 to +0.001) | -0.001 (-0.004 to +0.002) | -0.001 (-0.002 to +0.001) |

Halves of the decision variant, season bootstrap at the same level:

| variant | overall | weeks 5-7 | weeks 8+ |
| --- | --- | --- | --- |
| without kneel-downs | -0.020 (-0.038 to +0.001) | -0.026 (-0.078 to +0.023) | -0.019 (-0.034 to -0.0003) |
| without spikes | -0.001 (-0.002 to +0.001) | -0.001 (-0.004 to +0.001) | -0.001 (-0.002 to +0.001) |

Plays moved, by season (count and EPA):

without kneel-downs and spikes:

| season | removed | removed_epa | added | added_epa |
| --- | --- | --- | --- | --- |
| 1999 | 344 | -155.839 | 0 | 0.000 |
| 2000 | 38 | -14.120 | 0 | 0.000 |
| 2001 | 43 | -17.087 | 0 | 0.000 |
| 2002 | 245 | -127.104 | 0 | 0.000 |
| 2003 | 363 | -152.509 | 0 | 0.000 |
| 2004 | 434 | -239.948 | 0 | 0.000 |
| 2005 | 430 | -207.079 | 0 | 0.000 |
| 2006 | 484 | -262.193 | 0 | 0.000 |
| 2007 | 465 | -171.133 | 0 | 0.000 |
| 2008 | 478 | -238.011 | 0 | 0.000 |
| 2009 | 463 | -229.603 | 0 | 0.000 |
| 2010 | 426 | -156.930 | 0 | 0.000 |
| 2011 | 444 | -191.161 | 0 | 0.000 |
| 2012 | 449 | -235.864 | 0 | 0.000 |
| 2013 | 452 | -210.314 | 0 | 0.000 |
| 2014 | 457 | -236.221 | 0 | 0.000 |
| 2015 | 474 | -234.880 | 0 | 0.000 |
| 2016 | 445 | -219.333 | 0 | 0.000 |
| 2017 | 455 | -204.965 | 0 | 0.000 |
| 2018 | 473 | -231.282 | 0 | 0.000 |
| 2019 | 473 | -225.339 | 0 | 0.000 |
| 2020 | 504 | -255.819 | 0 | 0.000 |
| 2021 | 466 | -242.542 | 0 | 0.000 |
| 2022 | 485 | -290.802 | 0 | 0.000 |
| 2023 | 497 | -294.747 | 0 | 0.000 |
| 2024 | 480 | -230.687 | 0 | 0.000 |
| 2025 | 515 | -240.737 | 0 | 0.000 |

without kneel-downs:

| season | removed | removed_epa | added | added_epa |
| --- | --- | --- | --- | --- |
| 1999 | 292 | -144.656 | 0 | 0.000 |
| 2000 | 26 | -7.441 | 0 | 0.000 |
| 2001 | 26 | -14.279 | 0 | 0.000 |
| 2002 | 232 | -125.536 | 0 | 0.000 |
| 2003 | 277 | -135.202 | 0 | 0.000 |
| 2004 | 352 | -220.830 | 0 | 0.000 |
| 2005 | 359 | -191.561 | 0 | 0.000 |
| 2006 | 387 | -238.285 | 0 | 0.000 |
| 2007 | 353 | -149.756 | 0 | 0.000 |
| 2008 | 380 | -220.963 | 0 | 0.000 |
| 2009 | 371 | -211.535 | 0 | 0.000 |
| 2010 | 350 | -147.380 | 0 | 0.000 |
| 2011 | 361 | -176.906 | 0 | 0.000 |
| 2012 | 368 | -223.606 | 0 | 0.000 |
| 2013 | 393 | -203.868 | 0 | 0.000 |
| 2014 | 383 | -231.832 | 0 | 0.000 |
| 2015 | 422 | -229.553 | 0 | 0.000 |
| 2016 | 374 | -215.969 | 0 | 0.000 |
| 2017 | 398 | -201.681 | 0 | 0.000 |
| 2018 | 395 | -226.809 | 0 | 0.000 |
| 2019 | 398 | -219.140 | 0 | 0.000 |
| 2020 | 419 | -255.068 | 0 | 0.000 |
| 2021 | 396 | -239.480 | 0 | 0.000 |
| 2022 | 424 | -285.861 | 0 | 0.000 |
| 2023 | 429 | -289.618 | 0 | 0.000 |
| 2024 | 405 | -221.812 | 0 | 0.000 |
| 2025 | 434 | -238.287 | 0 | 0.000 |

without spikes:

| season | removed | removed_epa | added | added_epa |
| --- | --- | --- | --- | --- |
| 1999 | 52 | -11.183 | 0 | 0.000 |
| 2000 | 12 | -6.679 | 0 | 0.000 |
| 2001 | 17 | -2.807 | 0 | 0.000 |
| 2002 | 13 | -1.568 | 0 | 0.000 |
| 2003 | 86 | -17.307 | 0 | 0.000 |
| 2004 | 82 | -19.117 | 0 | 0.000 |
| 2005 | 71 | -15.517 | 0 | 0.000 |
| 2006 | 97 | -23.908 | 0 | 0.000 |
| 2007 | 112 | -21.377 | 0 | 0.000 |
| 2008 | 98 | -17.047 | 0 | 0.000 |
| 2009 | 92 | -18.068 | 0 | 0.000 |
| 2010 | 76 | -9.551 | 0 | 0.000 |
| 2011 | 83 | -14.255 | 0 | 0.000 |
| 2012 | 81 | -12.258 | 0 | 0.000 |
| 2013 | 59 | -6.446 | 0 | 0.000 |
| 2014 | 74 | -4.389 | 0 | 0.000 |
| 2015 | 52 | -5.328 | 0 | 0.000 |
| 2016 | 71 | -3.364 | 0 | 0.000 |
| 2017 | 57 | -3.284 | 0 | 0.000 |
| 2018 | 78 | -4.473 | 0 | 0.000 |
| 2019 | 75 | -6.199 | 0 | 0.000 |
| 2020 | 85 | -0.751 | 0 | 0.000 |
| 2021 | 70 | -3.062 | 0 | 0.000 |
| 2022 | 61 | -4.941 | 0 | 0.000 |
| 2023 | 68 | -5.129 | 0 | 0.000 |
| 2024 | 75 | -8.876 | 0 | 0.000 |
| 2025 | 81 | -2.449 | 0 | 0.000 |

Team year-over-year Pearson of full-season `team_rating` (consecutive seasons, by team): today's
rating (every scrimmage snap) 0.432 (829 pairs); without kneel-downs and spikes 0.445 (829 pairs);
without kneel-downs 0.445 (829 pairs); without spikes 0.433 (829 pairs).

Feature-value impact against today's, over every pre-week snapshot (team-weeks moved beyond 1e-9,
largest and mean absolute change, points per game):

- without kneel-downs and spikes: `team_rating` 13,930 of 13,930, largest 4.321, mean 0.310;
  `offense_rating` 13,930 of 13,930, largest 2.649, mean 0.208; `defense_rating` 13,930 of 13,930,
  largest 2.596, mean 0.199.
- without kneel-downs: `team_rating` 13,900 of 13,930, largest 4.371, mean 0.310; `offense_rating`
  13,900 of 13,930, largest 2.660, mean 0.208; `defense_rating` 13,900 of 13,930, largest 2.630,
  mean 0.198.
- without spikes: `team_rating` 13,868 of 13,930, largest 0.305, mean 0.024; `offense_rating` 13,868
  of 13,930, largest 0.297, mean 0.017; `defense_rating` 13,868 of 13,930, largest 0.289, mean
  0.016.

2025 full-season ranks that differ from today's:

- without kneel-downs and spikes: BUF 10->8, DET 8->9, IND 9->10, MIN 18->17, PIT 17->18, CAR
  25->24, ARI 24->25.
- without kneel-downs: BUF 10->8, DET 8->9, IND 9->10, CAR 25->24, ARI 24->25.
- without spikes: MIN 18->17, PIT 17->18.

## Provenance

- nfl-predictor 0.57.1 at commit `0fdf79cee44dbaec2554315706071ea1bd88d855`; code SHA-256:
  `team_plays_test.py` 576a3b3d5bb0, `rating_tests_common.py` 50da9bf186e7.
- Inputs: 108 files, fingerprint `85783d39e38c710f8f901fa423d2f46713429ab6f12e91a42c64150e7fd35a7a`
  (every file's SHA-256 is in `kneel_spike_test.json`), from
  `/home/mitch/scratch/step6/data/cache/nflreadpy` and `/home/mitch/scratch/step6/data/ratings`,
  read only, downloads blocked.
- Command: `VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/team_plays_test.py
  kneel-spike --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy --ratings-dir
  /home/mitch/scratch/step6/data/ratings`; 117 seconds in all; seconds per variant: today's rating
  (every scrimmage snap) 22, without kneel-downs and spikes 22, without kneel-downs 19, without
  spikes 17, no play moved (an integrity check) 8.
