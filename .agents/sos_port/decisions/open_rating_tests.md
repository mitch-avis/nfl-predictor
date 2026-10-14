# The rating questions `nfl-sos-ratings` left open: tests, results and recommendations

Roadmap step 6 absorbs `nfl-sos-ratings`, frozen at `9081bce`; the rating questions still open
there at the freeze are this repository's work (`../../sos_port_source.md`, "Open or fragile at
the frozen commit" and "Handoff after the freeze"). Each was measured, report-only, on the ported
ratings, under a protocol committed before anything was scored (`../protocols/`). Nothing here
changes a rating: adopting any result changes the `ratings` feature group the phase-4
walk-forwards are measuring, so adoption is the user's decision and needs its own measurement.

The decisions on record these tests carry (reported by the `nfl-sos-ratings` session as its
maintainer's, 2026-10-09): the quarterback test's tie-break order (every quarterback play, then
plus scrambles, then passes and sacks); the botched-snap charge rule (a botched snap counts
against a quarterback only when the official scorer charged the fumble to him; remove the 60
snaps charged to a lineman); and a pre-registered test before kneel-downs and spikes leave the
team ratings. The user asked on 2026-10-10 for step 6 to run autonomously with final approvals
batched; judgment calls inside these protocols are the session's, logged below.

Every run reads the candidate build's play-by-play and player-source cache
(`~/scratch/step6/data/cache/nflreadpy/`) through this repository's loaders with downloads
blocked, and its published ratings tables (`~/scratch/step6/data/ratings/`), read only. Each
report records the SHA-256 of its code and of every input file. Commands are in each entry and in
`../README.md`, "The rating-definition tests".

## 1. Kneel-downs and spikes in the team rating

- **What differs**: the ported team rating's scrimmage EPA counts every scrimmage snap, kneel-downs
  and spikes included (`ratings/pbp_inputs.team_games_from_pbp`, the source's definition); the
  quarterback rating leaves both out. The source's maintainer wanted a pre-registered test before
  they leave the team ratings.
- **Options**: keep every scrimmage snap (today); leave kneel-downs and spikes out of the team
  rating's snaps and EPA. Descriptive halves: kneel-downs only, spikes only.
- **Decision**: the rule as written keeps today's rating. Leaving them out lowers the next-game
  margin MAE from 10.575 to 10.554 over 5,297 games, a paired difference of -0.021, but the 95%
  season-bootstrap interval runs from -0.039 to +0.0005, so the overall interval does not lie
  below zero and the candidate does not qualify; on a tie the protocol recommends today's
  definition (no play judged by its situation, as in the garbage-time test). **Recommendation:
  keep today's rating, as the rule says**, and bring the near-miss to the user as a question:
  every window and the stability extra lean toward removal, so adopting it on principle (the
  quarterback rating already leaves both out) would be defensible but is not what the measurement
  supports. Moderately sure; what would change it: the same direction with an interval below zero
  in a measurement on later seasons, or in the phase-4 feature walk-forward.
- **Reasoning**: the protocol fixed the rule before scoring (`../protocols/kneel_spike.md`): the
  candidate qualifies only when its overall interval lies below zero and no week band lies above
  zero. The overall upper bound is +0.00047, so it fails by a hair; the weeks-8-on band
  (-0.019, -0.035 to -0.001) excludes zero in the candidate's favor and weeks 5-7 (-0.027, -0.083
  to +0.022) do not. Guardrail 12 applies the rule as written: the result is not between its
  branches, so the outcome is today's rating.
- **Data**: `../KNEEL_SPIKE_TEST.md` and `../kneel_spike_test.json`, from
  `VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/team_plays_test.py
  kneel-spike --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy --ratings-dir
  /home/mitch/scratch/step6/data/ratings` at commit `983bdb2` (three runs, identical numbers).
  Integrity checks all held before any result was read: today's rows and full-season ratings equal
  the candidate build's published files (gap 0.0), 421 snapshots of 2000-2025 equal the published
  history (gap 0.0), and the ported harness reproduces the source's validation exactly (5,297
  games at 10.575; 1,141 at 10.567; 4,156 at 10.577). The plays removed, 1999-2025: 11,282
  (-5,516.2 EPA), 9,404 kneel-downs (-5,266.9) and 1,878 spikes (-249.3). Descriptive: the
  single-game bootstrap interval excludes zero (-0.035 to -0.007); the kneel-downs carry the
  effect (-0.020, -0.038 to +0.001) and spikes add almost nothing (-0.001, -0.002 to +0.001);
  team year-over-year Pearson 0.432 today, 0.445 without; the fitted margin slope falls from 0.903
  to 0.858 (removal shrinks the rating's scale, which each variant's own margin model absorbs).
- **Feature-value impact**: if adopted, every team rating in every pre-week snapshot moves: over
  13,930 team-weeks, `team_rating` by 0.310 points per game on average (largest 4.321),
  `offense_rating` 0.208 and `defense_rating` 0.199 on average, so every column of the `ratings`
  feature group (and `ratings_sos`) changes in every game from 2000 on; 2025's full-season ranks
  swap in seven places (for example BUF 10th to 8th).
- **How to reverse it**: nothing to reverse (no change made). Adoption would drop the
  `qb_kneel` and `qb_spike` terms from the team snap in `ratings/pbp_inputs._scrimmage_snap` (the
  team-game rows only, the quarterback rows unchanged), then an ETL rebuild and a new reference.

## 2. Penalty-wiped plays in the team rating

- **What differs**: a play a penalty wiped out (`play_type == "no_play"`) counts as a team
  scrimmage snap when it keeps a snap flag. nflverse keeps `rush` on wiped-out runs and sets no
  dropback flag on wiped-out passes, so the ported rating (as the source) keeps 8,953 wiped runs
  (-5,652.6 EPA, 1999-2025) and leaves out 27,904 wiped passes (+11,141.3 EPA). Nothing chose
  this asymmetry. Carries since `0.54.0` and the quarterback rating count no wiped play.
- **Options**: today's (runs in, passes out); every wiped snap (both in); no wiped snap (both
  out); wiped passes only (runs out, passes in).
- **Decision**: no variant qualifies; no interval excludes zero in any window. The rule's
  recommendation is the first definition in the pre-registered order of simplicity that is not
  significantly worse: **every wiped snap**. **Recommendation: report the tie to the user with
  the rule's answer (every wiped snap) and low confidence in the order between the two symmetric
  definitions**; the data cannot tell any of the four apart, so the choice is definitional, and
  any change should wait for the batched rebuild with the other definition changes.
- **Reasoning**: the paired differences from today's, 98.33% season-bootstrap intervals: every
  wiped snap +0.000 (-0.020 to +0.020), no wiped snap -0.001 (-0.019 to +0.016), wiped passes
  only +0.004 (-0.020 to +0.027); the bands are as flat. The order of simplicity was fixed before
  scoring (`../protocols/penalty_wiped.md`): the two symmetric definitions first, since today's
  asymmetry is an accident of the flags, and "every wiped snap" ahead of "no wiped snap" because
  it counts every snap the offense ran (the rating's own definition) and the user's stated
  preference in the quarterback test was all available data on a tie. The case for "no wiped
  snap" is consistency with the carries and the quarterback rating, which count no wiped play;
  the session judged the rating's own snap definition the stronger anchor, a judgment the user
  may reverse at no measured cost.
- **Data**: `../PENALTY_WIPED_TEST.md` and `../penalty_wiped_test.json`, from
  `VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/team_plays_test.py
  penalty-wiped --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy --ratings-dir
  /home/mitch/scratch/step6/data/ratings` at commit `983bdb2` (three runs, identical numbers), with
  the same integrity checks as entry 1, all held. MAE: today 10.575, every wiped snap 10.576, no
  wiped snap 10.574, wiped passes only 10.579. The counts reconcile with the phase-1 figures
  exactly once their filter is known: the phase-1 total of 65,122 penalty-wiped plays is the count
  with both teams known (not every `no_play` row, as the protocol supposed; 110,841 rows carry
  `no_play` before that filter). With both teams known, 8,848 carry the rush flag (-5,660.46) and
  27,906 the pass flag (+11,142.27), the phase-1 figures; the 8,953 wiped runs are those 8,848
  plus 36 kneel-downs and 69 spikes a penalty wiped out, and 2 of the pass-flagged plays (one in
  2001, one in 2002, +0.97 EPA) also carry a snap flag, so today's rule counts them as runs,
  leaving 27,904 wiped passes. Team year-over-year Pearson: 0.432 today and with every wiped snap,
  0.429 with none, 0.427 with passes only.
- **Feature-value impact**: if adopted, every pre-week team rating moves: `team_rating` by 0.421
  points per game on average for every wiped snap (largest 3.045), 0.199 for no wiped snap; every
  `ratings` column changes in every game from 2000 on, and 2025's full-season ranks move (every
  wiped snap: 15 teams, for example DEN 6th to 11th; no wiped snap: 6 teams).
- **How to reverse it**: nothing to reverse (no change made). Adoption would add
  `(play_type == "no_play") & pass` to the team snap in `ratings/pbp_inputs._scrimmage_snap` for
  every wiped snap, or exclude `no_play` from it for no wiped snap, then an ETL rebuild and a new
  reference.

## 3. The quarterback rating from all of a quarterback's plays

- **What differs**: the ported quarterback rating (`ratings/qb_rating.py`, as the source
  published it) fits EPA per dropback, pass attempts and sacks only; a quarterback's scrambles
  (nflverse leaves their passer empty), designed runs and the botched snaps charged to him are not
  counted. The source's maintainer asked that the rating count every aspect of quarterback play;
  its protocol (roadmap section Q1) was registered and reviewed there but never run.
- **Options**: passes and sacks (today); plus scrambles; every quarterback play (plus designed
  runs and the botched snaps the scorer charged to him). Descriptive extra: every quarterback
  play without the botched snaps.
- **Decision**: no candidate beats another; no interval excludes zero in any window. The rule
  recommends the first candidate in the user's order that no other beats, **every quarterback
  play**, which ties the rating it would replace: it comes without any measured gain.
  **Recommendation: report the rule's answer (every quarterback play) and that it rests on the
  user's order alone**; adopting it is a definitional choice (the user's stated preference for all
  of a quarterback's plays), not an improvement this test found, so it should land, if at all,
  with the other definition changes in one rebuild and be measured in the feature walk-forward.
  Fairly sure the test cannot separate the candidates; what would change it: a target the
  quarterback term actually informs (see Reasoning) or many more starter changes.
- **Reasoning**: the protocol (`../protocols/qb_all_plays.md`, the source's Q1 with the user's
  order of simplicity) decides by paired differences in the MAE of a team's offensive EPA per play,
  98.33% season-bootstrap intervals over 19 seasons, a candidate beating another only when its
  overall interval lies below zero and no band lies above. The differences, in EPA per play: plus
  scrambles - passes and sacks +6.6e-06 (-1.9e-05 to +3.1e-05); every quarterback play - passes
  and sacks +1.2e-05 (-4.8e-05 to +7.9e-05); every quarterback play - plus scrambles +5.9e-06
  (-4.1e-05 to +6.7e-05). Two descriptive results explain the tie. First, the quarterback term adds
  nothing beyond the team fit's own prediction for any candidate: the calibration puts a slightly
  negative weight on it (`k1` -0.071, -0.063, -0.048) and nearly all on the team term (`k2` about
  0.97 to 1.00), and the team-only baseline's MAE (0.146512) is a hair below all three
  (0.146628, 0.146635, 0.146640). Second, the seeded power estimate on a synthetic league with the
  real schedules, play mix and starters finds a win in 4% (plus scrambles over passes and sacks),
  1% and 0% of 100 windows: at the spreads the data show, starter changes are too rare and a
  quarterback's runs too small a share of a team's plays for this target to show a difference. When
  the quarterback rating predicts alone (the source draft's one-term model, descriptive), counting
  more plays helps: every quarterback play - passes and sacks -8.5e-04 (-1.4e-03 to -3.6e-04),
  every quarterback play - plus scrambles -4.7e-04 (-7.6e-04 to -2.1e-04), plus scrambles - passes
  and sacks -3.8e-04 (-7.9e-04 to -3.1e-05), the ordering the source expected; the protocol
  rejected that model as its decision because it also rewards lasting team habits in the extra
  plays, which the team term absorbs.
- **Data**: `../QB_PLAYS_TEST.md` and `../qb_plays_test.json`, from `VIRTUAL_ENV="$PWD/.venv"
  nice -n 10 .venv/bin/python .agents/sos_port/qb_plays_test.py --cache-dir
  /home/mitch/scratch/step6/data/cache/nflreadpy --ratings-dir
  /home/mitch/scratch/step6/data/ratings` (the schedule file for check 9 defaults to the snapshot's
  `a7d0d7a2...parquet`), at commit `a0f534e`; two runs gave identical numbers (two earlier runs,
  before the pairs were sorted, differed from them only below 1e-18). All eleven integrity
  checks held before any result was read: passes and sacks reproduces the published quarterback
  ratings of 2005-2025 and their weekly history (gap 0.0); the play parts equal the ported season
  statistics on 13,070 quarterback-games; the team term equals its snapshot's ratings and the
  published team history (gap 0.0); zeroed additions give passes and sacks exactly; 325 weekly
  snapshots ignore every later game. The port reproduces the source's own counts exactly: 42
  schedule-listed starters without a quarterback row in 2006-2025 (31 in 2024), 77 first-snap
  starters that differ from the schedule's, 4 unknown starters (one each in 2008, 2015, 2016 and
  2017), 517 snap-count rows without a dropback from 2013, the year-over-year Pearson of 0.455 over
  601 pairs, and the 2025 refits (Drake Maye first under all three at 0.209, 0.254 and 0.243; Josh
  Allen 11th, 8th and 6th). Not run, as the protocol's departures say: the ESPN QBR correlations
  and the QBR half of check 10 (no QBR file among the allowed inputs) and the passer holdout (no
  postseason quarterback rows). The predictions use the previous season's penalty throughout;
  the published quarterback weekly history, which uses the season's own penalty, is read only by
  check 2 to confirm the port, never by a score.
- **Feature-value impact**: if adopted, the starter's pre-week rating behind `ratings_qb` moves in
  most games: over 9,599 team-games of 2006-2025 with a rated starter, by 0.015 EPA per play on
  average for every quarterback play (largest 0.104; 2,587 beyond 0.02) and 0.017 for plus
  scrambles; `ratings_qb_dropbacks` would count every quarterback play; and every published
  quarterback table (ratings, history, ranges, pairs, faced defense) changes, its
  `adj_qb_epa_per_dropback` and `qb_faced_pass_defense` names then inexact (a rename at adoption).
  Year-over-year Pearson: 0.459 today, 0.445 plus scrambles, 0.454 every quarterback play.
- **How to reverse it**: nothing to reverse (no change made). Adoption would add each candidate's
  parts (the script's `qb_play_parts`) to `qb_dropbacks` and `qb_dropback_epa` in
  `ratings/pbp_inputs.quarterback_rows`, then the renames, an ETL rebuild and a new reference.

## 4. The botched-snap charge rule

- **What differs**: the ported rating counts as the passer's dropbacks the aborted snaps the
  official scorer charged to a center or other lineman that went on as a pass or sack. The rule on
  record counts a botched snap against a quarterback only when the scorer charged the fumble to
  him, and the approved change removes those snaps (60 in 2006-2025 by the source's count).
- **Options**: keep them (today); remove them as approved.
- **Decision**: a measurement, no decision rule. The approved change removes 60 plays of
  2006-2025 at -21.3 EPA, exactly the source's count, and moves the rating very little.
  **Recommendation: adopt the approved removal with the next ratings rebuild, applying the rule as
  worded** (also to the 14 such plays of 2000-2005 and to the 14 pass-or-sack botched snaps of
  2006-2025 charged to another player or to no one, which the rule's wording also excludes; a
  question for the user, since the approved change names only the 60). Fairly sure: the effect is
  negligible and the rule is the user's.
- **Reasoning**: the rule is already decided; the plan (`../protocols/botched_snaps.md`) only
  measures it. Full-season ratings (each season's own penalty, as published): 1,090
  passer-seasons move by more than 1e-9, by at most 0.0080 EPA per dropback, because the
  simultaneous fit spreads one passer's change over the intercept and the defenses; 18 ranks among
  qualified passers change, each by one place; no season's cross-validated penalty changes. The
  published weekly history moves by at most 0.0112.
- **Data**: `../BOTCHED_SNAPS.md` and `../botched_snaps.json`, from `VIRTUAL_ENV="$PWD/.venv"
  nice -n 10 .venv/bin/python .agents/sos_port/botched_snaps.py --cache-dir
  /home/mitch/scratch/step6/data/cache/nflreadpy` at commit `fe36d58` (the first run, before a
  fix to the report's table format, gave the same headline numbers), with every removed play
  listed with its text. The classification is the source
  helper's (`aborted_snaps.py` at `9081bce`): the charged player is a quarterback when he has a
  quarterback row in the game or the players file lists him at QB, a lineman when the players
  file lists him at C, G, T, OL, OT or OG; the play counts when no one ran with the ball and it is
  one of the rating's dropbacks.
- **Feature-value impact**: if adopted, the pre-week ratings behind `ratings_qb` move in 11,742
  of 19,881 passer-weeks of 2006-2026 (2026 through the 2025 penalty), by at most 0.0112 EPA per
  dropback, and `ratings_qb_dropbacks` drops by one in 504 passer-weeks; the published
  quarterback tables move as above.
- **How to reverse it**: nothing to reverse (no change made). Adoption would subtract these plays'
  dropbacks and `qb_epa` in `ratings/pbp_inputs.quarterback_rows` (the cache carries
  `aborted_play`, `fumbled_1_player_id` and `rusher_player_id` since `0.50.0`; the positions come
  from the players file), then an ETL rebuild and a new reference.

## Review record

- 2026-10-12, independent review (approve with findings,
  [review_2026-10-12.md](../review/rating_tests/review_2026-10-12.md)): every protocol precedes
  its script and report, and all four reports reproduce bit for bit. Fixed after it: the
  quarterback test checks the line it printed unchecked, that each calibration used earlier weeks
  only; a cycle of wins goes to the user instead of falling back to the first in the order; a
  charged botched scramble counts once in a quarterback's plays; the information-set checks change
  every later value; the season t interval names a season without rows; tests pin the reports'
  tables and intervals, one helper writes every JSON record, and every report records the git
  commit.
- 2026-10-14, independent re-check of those fixes (approve with findings, no P0-P2,
  [recheck_2026-10-14.md](../review/rating_tests/recheck_2026-10-14.md)): with its tests in place,
  each reverted fix fails them, except the call to the calibration check; all four reports rerun
  to the same decisions, comparisons, scores and integrity checks (descriptive floats differ in
  the last bit, at most 4.5e-16 relative). Open P3: the calibration check reads the recorded row
  count, not the fit, and no test shows `run_integrity_checks` calls it; the information-set
  change leaves 1.75 and nulls as they were, and the team check does not flip later sites; the
  recorded commit does not mark uncommitted edits (the code's SHA-256 does); `season_t_interval`
  returns NaN bounds for a single season.
- The user's decision (2026-10-11): no outcome here (every wiped snap, every quarterback play, the
  botched-snap removal) is adopted before step 6 merges; each is a measured change after it.
