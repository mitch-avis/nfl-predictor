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

### This repository's own Elo (`0.47.0`)

Full records: [decisions/elo.md](decisions/elo.md) (16 entries) and the generated evaluation
[ELO_EVAL.md](ELO_EVAL.md). Review: two independent rounds (verdict "merge as is"); the reviewer
reproduced the evaluation byte for byte and ran its own leakage perturbations (0 of 4,380
pre-cutoff games moved).

- **Component rule, written before scoring**: a component stays only if its paired Brier interval
  (5,000 game resamples, seed 42, 2007-2025) lies below zero and log loss does not get worse.
  Kept: travel, rolling home field, the quarterback adjustment, the win-total pull. Dropped:
  rest and byes, the playoff boost. Two kept components passed by a hair (rolling home field and
  the win-total pull; the reviewer found they would fail on 40 and 17 of 100 other bootstrap
  seeds); the rule's outcome stands as written, and the phase-4 walk-forward decides whether the
  own Elo is used at all.
- **Seeding**: every team starts at 1505 in 1999 with a burn-in; 538's published 1999 ratings
  changed nothing measurable (Brier difference about 4e-09).
- **Home sides and neutral sites follow nflverse**, not 538 (30 swapped games: Super Bowls,
  international and relocated games, and four Saints games of 2005).
- **Quarterback adjustment from this repository's play-by-play**; 538's draft-based rookie
  ratings are replaced by the mean of earlier first starts (no draft data here); designed
  quarterback runs were missing because the play-by-play cache lacked the rusher's id, which it
  carries since `0.50.0`, so the integration adds them and reruns the evaluation under the same
  rule.
- **Starters**: a completed game uses its actual starter, an upcoming game the projected one, as
  the `qb_elos.csv` columns do; the diagnostic cost of knowing only the previous starter is in
  the evaluation report and on the user's list (`.agents/step6_plan.md`).
- **Two groups**: `own_elo` (no market information) and `own_elo_wt` (with the win-total pull),
  named so the substring-based group switches stay disjoint.

### The web app's Pool page and margin-only switch (`0.48.0`)

Full records: [decisions/web.md](decisions/web.md) (15 entries). Review: three independent rounds;
the last found every code fix correct (the exact survivor planner checked on 3,840 plans).

- **For your approval (a substitute method)**: the survivor planner solves each plan exactly (an
  assignment of weeks to teams, about 2 ms a season) instead of the plan's beam search. Both
  reviews' simulations found the beam below the best achievable plan in many seasons (in the
  kept script, 9 of 30 at width 64, up to 13.8% below; 17 of 30 at width 16, with a wrong first
  pick in 4). Task 58.1 stays open until you accept it.
- **The confidence sheet** reads the existing ranks and never re-ranks; after the lock it is
  scored from the picks file written before the lock, with newer ranks as notes. Known limit: a
  re-predict before the lock and another after it fall back to the weekly run's picks file.
- **The margin-only ranking** is computed on the server over every feature (5 of the margin
  head's top 25 were outside the combined top 40 on the latest weekly run).
- **Run and week resolution**: a run keeps its own week when later weeks are predicted into it;
  a newer run's own week outranks a file predicted ahead into an older run; the selector and the
  routes agree on every week.
- **Corrections without asking**: the week selector's one-update fix and the removal of the
  frontend's plan labels (both a stated rule with one plausible fix).
- **Tiebreakers**: whole-point entries; the week's highest and lowest team scores come from a
  distribution and run about 2 points low on top score against 2015-2025 (calibrating it is on
  your list).
- **Live betting (design only)**: the in-game probability updates the pregame prediction with the
  score and clock, the market is shown beside it and never blended, providers are off by
  default, and odds poll on a monthly credit budget. Three provider choices are on your list.

### The ported ratings (`0.49.0`)

Full records: [decisions/ratings.md](decisions/ratings.md) (13 entries) and the generated audit
[RATINGS_PARITY.md](RATINGS_PARITY.md). Review: two independent rounds (verdict "merge as is");
the reviewer reproduced the audit on three CPU math kernels and caught every planted leak.

- **Parity**: rebuilt over 1999-2026, the port reproduces every ratings file `nfl-sos-ratings`
  published (767,141 cells; 766,979 bit-identical, the rest within 1.1e-16 on this machine; all
  114 rank-range and head-to-head tables bit-identical). Tests compare floats to 1e-12 rather than
  bit for bit, because last bits depend on the CPU's math kernel.
- **Pre-week ratings for features**: fitted only on earlier weeks, with the previous season's
  cross-validated penalty for both teams and quarterbacks (the source's quarterback history uses
  the season's own penalty, which is not known before the week; the two agree in 7 of 27
  seasons, and elsewhere differ by 0.021 EPA per dropback on average).
- **Week 1** returns the preseason prior only, `sos` is null in weeks 1-2 (no opponent has an
  earlier game to rate it by), and the first season (1999) is null throughout.
- **Determinism**: the fits keep the caller's row order and `sos` sums opponents in a fixed order;
  the ETL must pin one math-library thread (an integration note).
- **SRS** comes from this repository's existing solver (bit-identical to the source's).

### The play-by-play columns and the stadium table (`0.50.0`)

Full records: [decisions/pbp_columns.md](decisions/pbp_columns.md) (15 entries and a note on the
first ETL after the merge). Review: one independent round plus a verified documentation round;
the reviewer reproduced every number on real data for all 27 seasons and found no existing value,
type or aggregation changed.

- **One refresh for the whole port**: the cache takes every play-by-play column `nfl-sos-ratings`
  reads (134, derived by script from its code; 84 added), plus the play text its repair parses.
- **Release check**: a fresh nflverse download equals the cache the 2026-10-09 reference build
  used, on every shared column, in 27 of 27 seasons, so the candidate build is not confounded by
  a newer release.
- **Team codes**: all 23 team-code columns are normalized at load; yard-line text stays as
  published (normalizing it would change an already-cached column).
- **Stadiums**: 14 venue ids the table lacked are added (9 used in 1999-2002, 5 in 2026), with
  elevations measured from terrain models at each venue's coordinates; 163 games move from 0 ft.
  Mile High Stadium gets its measured 5,210 ft while Empower Field keeps its nominal 5,280 ft.
- **2026 roofs**: nflverse lists three 2026 international venues as domes; no override now (the
  current season's schedule re-downloads every run); recheck before the Paris game on Oct 25.

### Phase 1: the inventory and the play-by-play parity check

Full records: [decisions/phase1_parity.md](decisions/phase1_parity.md) (18 entries), the generated
[INVENTORY.md](INVENTORY.md) and [PARITY.md](PARITY.md), and the summary
[FINDINGS.md](FINDINGS.md). Review: two independent rounds; the final round's fixes were checked
by re-running both scripts on the integration branch.

- **Adopted**: pre-2006 scrambles as dropbacks; situational flags read on every play; a down
  flagged both ways counted once; the Jacksonville 2001-2002 player-team repair; the play-by-play
  cache extension; carries on the source's designed-carry rule.
- **Corrected by the session**: carries were first kept on this repository's `rush` flag, on the
  premise that penalty-wiped passes stay dropbacks. The data showed none does (of 65,122
  penalty-wiped plays, 8,848 keep the run flag at -5,660 EPA, 27,906 the pass flag at +11,142 EPA,
  0 the dropback flag), so the old rule put penalty-wiped runs' EPA into rushing while
  penalty-wiped passes counted nowhere. Carries now follow the source (`rush_attempt` and not a
  two-point try, scramble or kneel-down).
- **Kept** (definitional): two-point tries out of dropbacks, first downs as passing plus rushing,
  fumbles on plays that are not special teams, play-by-play ahead of nflverse team stats,
  both-sides special-teams counts, the explosive and stuffed definitions, Washington as `WSH`.
- **Inventory**: 1,736 output columns, each resolved to one rule (matched 30, different 526,
  missing 1,180 here), with guards that fail on an unclassified column, a same-name counterpart
  filed as missing, or a column missing in one file kind but classified in another.
- **Parity**: every mismatching team-game explained by a named cause (40,694 values); the
  snapshot of the play-by-play the source read equals this repository's cache row by row.

### The season statistics (`0.51.0`)

Full records: [decisions/season_stats.md](decisions/season_stats.md) (15 entries) and the
generated audit [SEASON_STATS_PARITY.md](SEASON_STATS_PARITY.md). Review: two independent rounds
(verdict "merge as is"); the reviewer reproduced the audit at 1 and 24 threads and with this
repository's own play-by-play column list.

- **Parity**: rebuilt from the play-by-play `nfl-sos-ratings` used, read through this
  repository's loaders, every published season table of 1999-2026 comes out bit for bit (280
  tables; the rating columns of two tables are taken from the ported ratings by design).
- **Inputs** come through this repository's loaders, plus the source's own preparation steps
  (canonical codes, the never-charted fields, the Jacksonville repair); nflverse's official weekly
  team stats get their own cached source, because the existing team-stats loader sums away
  columns the source's overrides read.
- **Washington is `WSH`** and whole numbers are 64-bit (the published files hold some as 32-bit;
  values are equal).
- **Not ported**: the garbage-time filter view (no published table depends on it), the source's
  SRS (the ratings package covers it), its command-line parts and file writing.
- **Display data only**: these tables are whole-season values, not pre-week values, and feed the
  web pages, never the model.

### Leaving feature groups out of production (`0.52.0`)

Full records: [decisions/feature_groups.md](decisions/feature_groups.md) (10 entries). Review: one
independent round (verdict "merge as is") plus three hardening items checked by the session.

- **One resolver** decides which columns a set of disabled groups removes, for the backtest,
  `nfl-predictor train` and both weekly stages, so an adopted walk-forward arm and production
  train on exactly the same columns (checked for the four combinations the step-6 arms use).
- **The setting ships empty** (`disable_feature_groups: []` in `config/weekly_run.yaml`), so no
  weekly output changes; setting a value after the measurement is a default change held for the
  user.
- **A saved model records** the groups and the columns it left out in `metadata.json`; the pickled
  model needs nothing new, because its feature list already says what it was trained on.
- **A group that matches no column warns** once per command, against the data it loaded (so an
  older dataset build without a group cannot silently train differently from the adopted arm).
