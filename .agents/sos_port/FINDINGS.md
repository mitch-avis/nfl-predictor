# Findings: what the phase-1 reports changed

Phase 1 compared this repository with `nfl-sos-ratings` at its frozen commit `9081bce`: a scripted
inventory of its sources, output columns, metric registry, ratings and web pages
([INVENTORY.md](INVENTORY.md)), and a play-by-play parity check of this repository's per-team-game
aggregates against its game logs for 1999-2025 ([PARITY.md](PARITY.md)). Every mismatching
team-game is explained by a named cause; each difference, its options, data and the decision
taken are in [decisions/phase1_parity.md](decisions/phase1_parity.md), and the one-line index is
[DECISIONS.md](DECISIONS.md). The user deferred these decisions to the session on 2026-10-10;
the adopted changes land on the step-6 integration branch and reach production only with the
user's final approval of the rebuild and merge.

## Adopted (this repository's play-by-play features change)

1. **Pre-2006 scrambles count as dropbacks**: nflverse leaves `qb_dropback` 0 on most scrambles
   before 2006, so a run play flagged `qb_scramble` counts as a dropback, as in the source; this
   also covers the five scrambles nflverse codes as runs and the one play flagged as both a
   dropback and a run, which now counts once.
2. **Carries follow the source's designed-carry rule** (`rush_attempt` and not a two-point try,
   scramble or kneel-down): penalty-wiped runs leave the carry counts and sums, so penalty-wiped
   plays are in neither the dropback nor the carry family. This corrects the session's first
   decision, which rested on a wrong premise (no penalty-wiped pass keeps `qb_dropback`).
3. **Situational flags on every play** (third and fourth downs, two-point tries), mostly 1999-2000
   plays with no play type, and **a down flagged both converted and failed counts once**.
4. **Jacksonville's 2001-2002 home games** get the source's player-team repair (penalties and
   red-zone touchdowns credited to the right team).
5. **The play-by-play cache carries every column the port reads**, including `qb_scramble` and
   `play_id` (`0.50.0`).

## Kept as they are (both definitions defensible; the ported ratings use the source's internally)

Two-point tries stay out of dropbacks; first downs stay passing plus rushing; fumbles stay on
plays that are not special teams; play-by-play stays ahead of nflverse's team stats; both teams'
special-teams plays stay counted; the explosive and stuffed definitions and Washington's `WSH`
stay. Kneel-downs and spikes stay out of this repository's dropbacks and carries; whether they
should leave the ported team rating's scrimmage EPA is a pre-registered test for later, as is the
team rating's handling of penalty-wiped plays.

## Checked and equal

Special-teams EPA, pass attempts, completions, rush attempts, two-point conversions, offensive
touchdowns, sacks, interceptions and passing touchdowns match in every team-game; both
repositories read the Rams as `LAR` and lack the same three games nflverse has no play-by-play
for; the snapshot of the play-by-play `nfl-sos-ratings` read equals this repository's cache row
by row in all 27 seasons, so no difference comes from an nflverse release.
