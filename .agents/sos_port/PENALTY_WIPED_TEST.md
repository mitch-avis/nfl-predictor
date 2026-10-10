# Penalty-wiped plays in the team rating: test result

Pre-registered in `protocols/penalty_wiped.md` before any variant was scored. Next-game margin MAE,
prediction weeks 5 and later of 1999-2025, the ported team rating rebuilt each week from the
season's earlier games for every variant.

## Headline

- every wiped snap: MAE 10.576 against today's 10.575 over 5,297 games; paired difference +0.000
  (-0.020 to +0.020).
- no wiped snap: MAE 10.574 against today's 10.575 over 5,297 games; paired difference -0.001
  (-0.019 to +0.016).
- wiped passes only: MAE 10.579 against today's 10.575 over 5,297 games; paired difference +0.004
  (-0.020 to +0.027).

## Decision

Rule as written (98.33% season-bootstrap intervals): no variant qualified, so the test cannot tell
the variants from today's rating; the recommendation is **every wiped snap**, the first definition
in the order of simplicity that is not significantly worse. The order: every wiped snap; no wiped
snap; today's rating (every scrimmage snap); wiped passes only.

- Intervals that exclude zero, either direction: none.
- Overall below zero but disqualified by a week band: none.
- The decision goes to the user; adoption is held (it changes every `ratings` column).

## Paired differences from today's rating

MAE of the variant minus today's (negative favors the variant); 98.33% percentile intervals from
10,000 resamples of whole seasons (27 clusters), seed 0, the same draws for every variant and band.

| variant | overall | weeks 5-7 | weeks 8+ |
| --- | --- | --- | --- |
| every wiped snap | +0.000 (-0.020 to +0.020) | -0.019 (-0.064 to +0.027) | +0.006 (-0.019 to +0.029) |
| no wiped snap | -0.001 (-0.019 to +0.016) | -0.024 (-0.082 to +0.015) | +0.005 (-0.008 to +0.020) |
| wiped passes only | +0.004 (-0.020 to +0.027) | -0.033 (-0.081 to +0.016) | +0.014 (-0.014 to +0.042) |

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
   wiped run and a wiped pass. Moved: every wiped snap: 0 plays out (+0.0 EPA), 27,904 in (+11,141.3
   EPA); no wiped snap: 8,953 plays out (-5,652.6 EPA), 0 in (+0.0 EPA); wiped passes only: 8,953
   plays out (-5,652.6 EPA), 27,904 in (+11,141.3 EPA).
4. With no play moved, the rows equal the adapter's exactly and the features and predictions equal
   today's (largest gap 0.0e+00).
5. 104 season penalties equal separate cross-validations of each variant's previous season.
6. Snapshots unchanged when every later game is replaced: today's rating (every scrimmage snap) 437,
   every wiped snap 437, no wiped snap 437, wiped passes only 437; all 26,485 predictions trained on
   earlier rows only.
7. Every input file's SHA-256 is recorded (Provenance).

## Descriptive extras (not decision inputs)

MAE, RMSE and the mean fitted margin slope `k`:

| variant | band | games | MAE | RMSE | mean k |
| --- | --- | --- | --- | --- | --- |
| today's rating (every scrimmage snap) | overall | 5,297 | 10.575 | 13.562 | 0.903 |
| today's rating (every scrimmage snap) | weeks 5-7 | 1,141 | 10.567 | 13.691 | 0.911 |
| today's rating (every scrimmage snap) | weeks 8+ | 4,156 | 10.577 | 13.526 | 0.900 |
| every wiped snap | overall | 5,297 | 10.576 | 13.570 | 0.902 |
| every wiped snap | weeks 5-7 | 1,141 | 10.548 | 13.676 | 0.912 |
| every wiped snap | weeks 8+ | 4,156 | 10.583 | 13.541 | 0.899 |
| no wiped snap | overall | 5,297 | 10.574 | 13.559 | 0.885 |
| no wiped snap | weeks 5-7 | 1,141 | 10.543 | 13.673 | 0.895 |
| no wiped snap | weeks 8+ | 4,156 | 10.582 | 13.528 | 0.882 |
| wiped passes only | overall | 5,297 | 10.579 | 13.573 | 0.895 |
| wiped passes only | weeks 5-7 | 1,141 | 10.534 | 13.667 | 0.905 |
| wiped passes only | weeks 8+ | 4,156 | 10.591 | 13.547 | 0.892 |

Single-game bootstrap of the same differences (98.33%):

| variant | overall | weeks 5-7 | weeks 8+ |
| --- | --- | --- | --- |
| every wiped snap | +0.000 (-0.023 to +0.024) | -0.019 (-0.072 to +0.034) | +0.006 (-0.020 to +0.033) |
| no wiped snap | -0.001 (-0.014 to +0.011) | -0.024 (-0.055 to +0.005) | +0.005 (-0.009 to +0.019) |
| wiped passes only | +0.004 (-0.022 to +0.029) | -0.033 (-0.087 to +0.023) | +0.014 (-0.014 to +0.043) |

Plays moved, by season (count and EPA):

every wiped snap:

| season | removed | removed_epa | added | added_epa |
| --- | --- | --- | --- | --- |
| 1999 | 0 | 0.000 | 1,049 | 458.674 |
| 2000 | 0 | 0.000 | 949 | 408.462 |
| 2001 | 0 | 0.000 | 834 | 360.203 |
| 2002 | 0 | 0.000 | 972 | 383.351 |
| 2003 | 0 | 0.000 | 937 | 362.695 |
| 2004 | 0 | 0.000 | 1,016 | 421.065 |
| 2005 | 0 | 0.000 | 1,057 | 440.963 |
| 2006 | 0 | 0.000 | 783 | 281.497 |
| 2007 | 0 | 0.000 | 726 | 230.950 |
| 2008 | 0 | 0.000 | 772 | 235.761 |
| 2009 | 0 | 0.000 | 874 | 297.353 |
| 2010 | 0 | 0.000 | 906 | 336.407 |
| 2011 | 0 | 0.000 | 998 | 330.819 |
| 2012 | 0 | 0.000 | 1,026 | 471.639 |
| 2013 | 0 | 0.000 | 989 | 406.131 |
| 2014 | 0 | 0.000 | 1,195 | 442.307 |
| 2015 | 0 | 0.000 | 1,239 | 411.545 |
| 2016 | 0 | 0.000 | 1,196 | 444.852 |
| 2017 | 0 | 0.000 | 1,171 | 507.397 |
| 2018 | 0 | 0.000 | 1,161 | 462.856 |
| 2019 | 0 | 0.000 | 1,267 | 476.087 |
| 2020 | 0 | 0.000 | 1,040 | 645.013 |
| 2021 | 0 | 0.000 | 1,164 | 499.783 |
| 2022 | 0 | 0.000 | 1,075 | 403.187 |
| 2023 | 0 | 0.000 | 1,113 | 515.656 |
| 2024 | 0 | 0.000 | 1,239 | 429.615 |
| 2025 | 0 | 0.000 | 1,156 | 477.027 |

no wiped snap:

| season | removed | removed_epa | added | added_epa |
| --- | --- | --- | --- | --- |
| 1999 | 370 | -200.187 | 0 | 0.000 |
| 2000 | 350 | -204.216 | 0 | 0.000 |
| 2001 | 274 | -118.926 | 0 | 0.000 |
| 2002 | 293 | -147.941 | 0 | 0.000 |
| 2003 | 343 | -191.712 | 0 | 0.000 |
| 2004 | 342 | -160.849 | 0 | 0.000 |
| 2005 | 441 | -247.916 | 0 | 0.000 |
| 2006 | 288 | -140.050 | 0 | 0.000 |
| 2007 | 260 | -126.720 | 0 | 0.000 |
| 2008 | 287 | -146.096 | 0 | 0.000 |
| 2009 | 285 | -163.778 | 0 | 0.000 |
| 2010 | 337 | -231.580 | 0 | 0.000 |
| 2011 | 331 | -217.301 | 0 | 0.000 |
| 2012 | 360 | -239.595 | 0 | 0.000 |
| 2013 | 289 | -221.976 | 0 | 0.000 |
| 2014 | 327 | -234.835 | 0 | 0.000 |
| 2015 | 385 | -291.233 | 0 | 0.000 |
| 2016 | 358 | -240.676 | 0 | 0.000 |
| 2017 | 344 | -225.372 | 0 | 0.000 |
| 2018 | 374 | -294.330 | 0 | 0.000 |
| 2019 | 379 | -269.515 | 0 | 0.000 |
| 2020 | 280 | -215.602 | 0 | 0.000 |
| 2021 | 338 | -267.125 | 0 | 0.000 |
| 2022 | 320 | -218.971 | 0 | 0.000 |
| 2023 | 286 | -186.501 | 0 | 0.000 |
| 2024 | 377 | -250.214 | 0 | 0.000 |
| 2025 | 335 | -199.339 | 0 | 0.000 |

wiped passes only:

| season | removed | removed_epa | added | added_epa |
| --- | --- | --- | --- | --- |
| 1999 | 370 | -200.187 | 1,049 | 458.674 |
| 2000 | 350 | -204.216 | 949 | 408.462 |
| 2001 | 274 | -118.926 | 834 | 360.203 |
| 2002 | 293 | -147.941 | 972 | 383.351 |
| 2003 | 343 | -191.712 | 937 | 362.695 |
| 2004 | 342 | -160.849 | 1,016 | 421.065 |
| 2005 | 441 | -247.916 | 1,057 | 440.963 |
| 2006 | 288 | -140.050 | 783 | 281.497 |
| 2007 | 260 | -126.720 | 726 | 230.950 |
| 2008 | 287 | -146.096 | 772 | 235.761 |
| 2009 | 285 | -163.778 | 874 | 297.353 |
| 2010 | 337 | -231.580 | 906 | 336.407 |
| 2011 | 331 | -217.301 | 998 | 330.819 |
| 2012 | 360 | -239.595 | 1,026 | 471.639 |
| 2013 | 289 | -221.976 | 989 | 406.131 |
| 2014 | 327 | -234.835 | 1,195 | 442.307 |
| 2015 | 385 | -291.233 | 1,239 | 411.545 |
| 2016 | 358 | -240.676 | 1,196 | 444.852 |
| 2017 | 344 | -225.372 | 1,171 | 507.397 |
| 2018 | 374 | -294.330 | 1,161 | 462.856 |
| 2019 | 379 | -269.515 | 1,267 | 476.087 |
| 2020 | 280 | -215.602 | 1,040 | 645.013 |
| 2021 | 338 | -267.125 | 1,164 | 499.783 |
| 2022 | 320 | -218.971 | 1,075 | 403.187 |
| 2023 | 286 | -186.501 | 1,113 | 515.656 |
| 2024 | 377 | -250.214 | 1,239 | 429.615 |
| 2025 | 335 | -199.339 | 1,156 | 477.027 |

Penalty-wiped plays by set, 1999-2025, beside the phase-1 parity check's figures (every `no_play`
play, both teams known or not: 8,848 with the rush flag at -5,660.46 EPA and 27,906 with the pass
flag at +11,142.27). A wiped run is a `no_play` play today's snap rule counts, whichever flag makes
it one (a play can carry several):

| set | plays | epa |
| --- | --- | --- |
| every penalty-wiped play | 110,841 | -2,776.958 |
| with the rush flag | 8,848 | -5,660.460 |
| with the pass flag | 27,907 | 11,141.039 |
| both teams known | 65,122 | -2,770.624 |
| wiped runs (today's snaps) | 8,953 | -5,652.558 |
| wiped runs (today's snaps), rush flag | 8,848 | -5,660.460 |
| wiped runs (today's snaps), qb_kneel flag | 36 | 3.797 |
| wiped runs (today's snaps), qb_spike flag | 69 | 4.105 |
| wiped runs (today's snaps), qb_dropback flag | 0 | 0.000 |
| wiped passes | 27,904 | 11,141.297 |

Penalty-wiped plays by the penalized side, 1999-2025:

| kind | side | plays | epa |
| --- | --- | --- | --- |
| wiped_run | defense | 2,055 | 1,238.079 |
| wiped_run | neither recorded | 154 | -1.604 |
| wiped_run | offense | 6,744 | -6,889.033 |
| wiped_pass | defense | 16,946 | 20,528.216 |
| wiped_pass | neither recorded | 870 | -10.792 |
| wiped_pass | offense | 10,088 | -9,376.127 |

Team year-over-year Pearson of full-season `team_rating` (consecutive seasons, by team): today's
rating (every scrimmage snap) 0.432 (829 pairs); every wiped snap 0.432 (829 pairs); no wiped snap
0.429 (829 pairs); wiped passes only 0.427 (829 pairs).

Feature-value impact against today's, over every pre-week snapshot (team-weeks moved beyond 1e-9,
largest and mean absolute change, points per game):

- every wiped snap: `team_rating` 13,930 of 13,930, largest 3.045, mean 0.421; `offense_rating`
  13,930 of 13,930, largest 2.197, mean 0.294; `defense_rating` 13,930 of 13,930, largest 1.974,
  mean 0.291.
- no wiped snap: `team_rating` 13,930 of 13,930, largest 2.783, mean 0.199; `offense_rating` 13,930
  of 13,930, largest 2.599, mean 0.145; `defense_rating` 13,930 of 13,930, largest 2.084, mean
  0.136.
- wiped passes only: `team_rating` 13,930 of 13,930, largest 2.833, mean 0.455; `offense_rating`
  13,930 of 13,930, largest 3.583, mean 0.321; `defense_rating` 13,930 of 13,930, largest 1.890,
  mean 0.317.

2025 full-season ranks that differ from today's:

- every wiped snap: BUF 10->6, IND 9->7, PHI 7->8, SF 11->9, DET 8->10, DEN 6->11, KC 14->13, CHI
  13->14, MIN 18->17, PIT 17->18, CAR 25->24, ARI 24->25, WSH 27->26, CIN 28->27, MIA 26->28.
- no wiped snap: DET 8->7, PHI 7->8, BUF 10->9, IND 9->10, KC 14->12, GB 12->14.
- wiped passes only: BUF 10->6, IND 9->7, PHI 7->9, DEN 6->10, KC 14->13, LAC 15->14, CHI 13->15,
  MIN 18->17, PIT 17->18, DAL 23->22, NO 22->23, CAR 25->24, ARI 24->25, WSH 27->26, CIN 28->27, MIA
  26->28.

## Provenance

- nfl-predictor 0.55.1; code SHA-256: `team_plays_test.py` b453010657c5, `rating_tests_common.py`
  e92451402822.
- Inputs: 108 files, fingerprint `85783d39e38c710f8f901fa423d2f46713429ab6f12e91a42c64150e7fd35a7a`
  (every file's SHA-256 is in `penalty_wiped_test.json`), from
  `/home/mitch/scratch/step6/data/cache/nflreadpy` and `/home/mitch/scratch/step6/data/ratings`,
  read only, downloads blocked.
- Command: `VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/team_plays_test.py
  penalty-wiped --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy --ratings-dir
  /home/mitch/scratch/step6/data/ratings`; seconds per variant: today's rating (every scrimmage
  snap) 36, every wiped snap 31, no wiped snap 31, wiped passes only 34, no play moved (an integrity
  check) 16.
