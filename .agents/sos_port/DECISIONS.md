# Step 6 decision log: differences and judgment calls

The user deferred every difference found while absorbing `nfl-sos-ratings` (frozen at `9081bce`)
to the session's judgment on 2026-10-10, on the condition that each one is logged here with the
decision, the exact reasoning and objective data, so it can be reviewed and reversed. Each entry
below links to the full record in `decisions/`, which gives:

- **What differs**: this repo against `nfl-sos-ratings` at `9081bce`, or the plan's text.
- **Options**: the choices considered.
- **Decision** and **reasoning**.
- **Data**: the numbers behind it, each with the command or script and the artifact that
  reproduces it.
- **How to reverse it**: the flag, constant or commit.
- **Review**: who re-ran the data (a reviewer that did not produce it), or "not yet reviewed".

## Index

### New nflverse sources and two small fixes (`0.46.0`)

Full records: [decisions/sources.md](decisions/sources.md). Review: an independent reviewer
re-derived the numbers from live data and the reviewed runs (verdict "merge after fixes"); the
fixes were checked by the delegating session before the merge.

- **Team codes**: this repository's canonical codes stay (Washington `WSH`, not the source's
  `WAS`); the alias table already maps `WAS`, so nothing downstream needs a new mapping.
- **League roster codes** (`ARZ`, `BLT`, `CLV`, `HST`, `SL`, found in every roster 2002-2015): a
  roster-only map, kept out of the global alias table.
- **Jacksonville 2001-2002 player teams**: repaired as the source does, reading the two teams from
  the game id in canonical form (2 rows left unplaced instead of 36). Only weekly player stats are
  repaired now; the play-by-play repair is ported and tested but waits for the feature-value
  batch, because it changes feature values.
- **Stats nflverse never charted** (tackles for loss 2003-2011, QB hits 2003-2005): null, not zero.
- **Quarterback identity**: exactly the source's rule (players file or any roster week of the
  season); the season roster and the player stats' own position add nobody in the seasons checked.
- **ESPN QBR**: validation only, guarded by a test over every command, the weekly run and the web
  app; a season the download lacks is an error, never an empty cached file.
- **Models directory**: six places now read `constants.MODELS_PATH` at call time; the floor-sigma
  reference runs and the config file stay on the repository root, as their config documents.
- **`compare` coin flips**: an exact 0.5 picks home, as the weekly picks do. No model number
  changes; the betting lines' pick accuracy does (170 pick-time and 25 closing-line coin flips).
- **Pool-point intervals** name their resampling unit (week blocks) in `compare`'s report.
