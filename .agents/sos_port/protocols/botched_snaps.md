# Measurement plan: the botched-snap charge rule in the quarterback rating

Written before measuring, and committed on its own. This is a measurement, not a test: the rule
is already decided, and the question is only what it would change. Nothing is adopted here; the
ported rating keeps today's rows until the user approves the change (it changes the `ratings`
feature group, which the phase-4 walk-forwards are measuring).

## The rule and the plays it removes

The rule (reported by the `nfl-sos-ratings` session as the maintainer's decision of 2026-10-09,
`.agents/sos_port_source.md`, "Handoff after the freeze", item 2): a botched snap counts against a
quarterback only when the official scorer charged the fumble to him (`fumbled_1_player_id`),
whoever ran with the ball afterwards. The approved change: remove from the quarterback rating the
aborted snaps of 2006-2025 that the scorer charged to a center or other lineman and that went on as
a pass or sack, 60 plays at -21.3 EPA by the source's count (its
`.agents/findings_2026_10_08/aborted_snaps.py 2006 2025`). The rating counts them today as
dropbacks of the passer.

The plays, classified as the source's helper does, on this repository's play-by-play (the
candidate build's cache, read through its loaders, downloads blocked):

- an aborted play: `aborted_play` set, not a two-point try, `play_type` not `no_play`, `posteam`
  known, a regular-season week;
- charged to a lineman: `fumbled_1_player_id` is not a quarterback (no quarterback row for that
  team in that game, and not listed at QB in the players file) and the players file lists him at
  C, G, T, OL, OT or OG;
- went on as a pass or sack: no `rusher_player_id`, and the play is one of the rating's dropbacks
  (`qb_dropback` set, `passer_player_id` the `qb_id` of a quarterback row of that team and game with
  at least one dropback).

The script prints the count and the `epa` and `qb_epa` sums by season, and every play with its
text. If the count differs from 60 or the `epa` sum from -21.3, it says so and lists the
difference before measuring; the measurement then uses this repository's plays.

## What is measured

Removing a play takes one dropback and the play's `qb_epa` (the rating's EPA is the official
passing EPA, which sums `qb_epa` over the passer's passes and sacks) from the passer's row of
that game. For every season 2006-2025, with and without the plays:

1. The full-season rating as published (`fit_qb_ratings`, the season's own cross-validated
   penalty): the penalty, every passer's `adj_qb_epa_per_dropback`, and among the passers the
   rankings qualify (14 attempts per team game), the rank changes. Headline: how many passers
   move by more than 1e-9 and by how much at most; how many qualified ranks change and by how much
   at most; the affected passers' own changes.
2. The published weekly history (`fit_qb_ratings_by_week` with the season fit's penalty): the
   largest change.
3. The pre-week ratings the model reads (`ratings.pre_week.RatingsHistory.pre_week_quarterbacks`,
   the `ratings_qb` and `ratings_qb_dropbacks` features): for every week of 2006-2025 and the
   following season's weeks (a season's rows set the next season's penalty), the rows that change
   by more than 1e-12 and the largest change.

Descriptive, outside the approved change: the same plays in 1999-2005 and 2026 (the rule as
written reaches every season), and the aborted snaps that went on as a pass or sack charged to
another player or to no one (the rule's wording removes those too; the approved change names only
the lineman's).

## Command and outputs

```bash
VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/botched_snaps.py \
    --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy
```

It writes `.agents/sos_port/BOTCHED_SNAPS.md` and `.agents/sos_port/botched_snaps.json` and nothing
else, and runs with one BLAS thread.
