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

### The team and quarterback web pages (`0.53.0`)

Full records: [decisions/web_pages.md](decisions/web_pages.md) (21 entries). Review: two
independent rounds (verdict "merge as is"; 5,603 payloads matched the source app's, every table in
the app rendered identically before and after the shared table change) plus a last small round
checked by the session. Not yet checked in a browser: look at the pages once.

- **For your review**: the sidebar is regrouped into four sections (Betting moves up under "This
  week", Model sits above Data); the "Opponents faced" ledger is simpler than the source's
  (games, results, average score, the opponent's season rating and a Tougher/Middle/Softer tier on
  Team Rating everywhere, where the source switched the metric by stat group).
- **Not ported** (logged, additive later if wanted): the per-row rank-range column and the rank
  ranges in the comparison, the "n selected" bar, search and sort kept across navigation, the
  refresh button, team palettes, CSV export and outside links, and the garbage-time filter view
  (its 8 glossary metrics are hidden).
- **Gzip at level 6** app-wide (the job log stream stays uncompressed): the team table goes from
  about 0.96 MB to 0.18 MB; level 9 saved only 1-9% more bytes for up to 10 ms more CPU.
- **Robustness**: a table a page needs that cannot be read answers 503 naming the file; one it can
  do without is left out with a warning; the API reads `data/ratings/` and the old
  `NFLP_SOS_DATA_DIR` setting is gone.
- **Waiting on**: the ETL writes every table the pages read since `0.56.0`; the pages fill once
  `data/` is rebuilt, which is held for your approval.

### The ratings in the ETL, the play-by-play corrections and the own Elo in the datasets (`0.54.0`)

Full records: [decisions/integration_ratings.md](decisions/integration_ratings.md) (19 entries),
[decisions/integration_pbp_elo.md](decisions/integration_pbp_elo.md) (entries 1.1-4.2), the
generated [ADAPTER_PARITY.md](ADAPTER_PARITY.md), and the measured feature changes in
`decisions/integration_feature_diffs/`. Review: two independent reviews, each in two rounds
(final fixes checked); the reviewers reproduced every quoted number from their own rebuilds.

- **The ratings in the ETL**: the adapter from this repository's play-by-play reproduces the
  source's game logs in all 305,830 input cells and its published ratings tables (741,295 cells
  bit-identical, the rest within 1.1e-16). The ETL writes the tables to `data/ratings/` and adds a
  `ratings` feature group (21 columns); the strength snapshot stays beside it until the
  measurement decides. A full rebuild takes about 7.5 minutes longer; an incremental run about 30
  seconds (about a minute late in the season).
- **Leakage**: every pre-week rating uses games before the week and roster weeks before the week;
  the reviewers' perturbations of every later input moved nothing; a starter's name is matched only
  against earlier passers (a later rookie can no longer change an earlier season's feature).
- **Play-by-play corrections adopted** (each measured, see the diffs): scrambles filed as runs are
  dropbacks (3,744 more dropbacks in 1999-2005); carries follow the source's designed-carry rule
  (348,419 instead of 357,227; penalty-wiped runs leave the rush sums but stay snaps), checked
  equal to the source in all 13,928 team-games; flags read on every play; a down flagged both ways
  counted once; the Jacksonville repair.
- **One starter**: a completed game's starter comes from its play-by-play (the source's rule),
  agreeing with the old name chain in all but 12 of 13,922 team-games (net better: the reviewer's
  hand-check found the chain wrong in 8 of the 12); upcoming games use the projected starter.
- **Own Elo in the datasets**: both presets in every build; designed quarterback runs now in
  VALUE; the evaluation rerun under its unchanged rule kept the same components (Brier 0.21701
  market-free, 0.21594 with win totals).
- **Market-free runs** keep the step-5 design (line columns only, groups left out by flag) and now
  warn when a market-informed group stays; making it automatic is on your list.

### Red-zone trips (`0.55.0`)

Full record: [decisions/integration_pbp_elo.md](decisions/integration_pbp_elo.md), entries 2.8 and
2.9, with `integration_red_zone_parity.py`, `integration_red_zone_movement.py` and their outputs.
Review: two independent rounds (verdict "merge as is"; every figure reproduced byte for byte).

- **The bug**: this repository counted a red-zone trip for any drive with a play at or inside the
  20, so 7,952 touchdown drives became converted trips through their extra point or two-point try
  alone, and 2,282 one-play "drives" holding only a try after a defensive or return touchdown
  became trips; the league red-zone touchdown rate read 0.5709.
- **The fix**: a trip is a drive nflverse flags inside the 20 (`drive_inside20`, the definition
  `nfl-sos-ratings` uses) in which the team ran an offensive snap, a two-point try not counting.
  The snap condition is a departure from the source: it removes 202 phantom trips the source keeps
  (128 kickoffs the kicking team recovered, 75 of them counted as the receiving team's red-zone
  touchdowns; extra points and tries after return touchdowns) and 9 double counts, at the cost of
  about 16 genuine field-goal-only possessions (0.04% of trips, all non-touchdowns). Every
  remaining difference from the source is attributed (222 team-games, 0 unexplained). The league
  rate now reads 0.5395; only the two red-zone rate columns move.
- **Known leftover**: 44 trips without a recorded drive result, mostly the first half of a
  1999-2000 drive nflverse split at a timeout, stay in the rate's denominator; joining split
  pieces is a later fix (0.1% of trips).

### The four-week trends of upcoming games (`0.55.1`)

Full record: [decisions/trend_parity.md](decisions/trend_parity.md), with its scripts and outputs.
Review: one independent round (merge after fixes; fixes checked); the reviewer recomputed every
scheduled team-week by brute force.

- **The defect** (found by a reviewer, predating step 6): an upcoming game's
  `scoring_margin_4wk_trend` and `turnover_margin_4wk_trend` were filled with 0.0, and so was every
  playoff game's, while completed rows carry the real pre-game trend. Not leakage: completed rows
  use earlier games only, so no training row or walk-forward result changes (0 of 6,952 completed
  regular-season rows move). Production predicted with zeros the model never trained on.
- **The fix**: trends are keyed on the season's schedule, so each week takes the team's trend
  after its last game before the week; playoff games take the full regular season's. Weekly
  predictions change from week 6 on (held with step 6 for your approval).
- **Also added**: a warning when `qb_elos.csv` lacks the week being predicted (a week-6 run with
  today's copy would give every game a 0.0 Elo trend and no quarterback trends, values never seen
  in training from week 6 on; refresh the file before each weekly run, as you do).
- **Known leftover, unchanged**: 11 turnover-margin sides in 2000-2002 publish 0.0 after a game
  nflverse has no team stats for; changing it would change completed training rows.

### The season statistics in the ETL (`0.56.0`)

Full record: [decisions/season_stats_etl.md](decisions/season_stats_etl.md) (11 entries) and the
generated [SEASON_STATS_ETL_PARITY.md](SEASON_STATS_ETL_PARITY.md). Review: two independent rounds
(verdict "merge as is"); the reviewer reproduced parity and a real-data cache round trip.

- **Parity through the ETL path**: built and written by the ETL with the ratings fitted in the
  same run, all 280 season tables of 1999-2026 match `nfl-sos-ratings`' published files bit for
  bit, the rating columns included; reloading every season from the cache rewrites all 534 tables
  byte for byte.
- **One schema in every season** (the repository's missing-data rule): a season built without a
  source nflverse has not published yet keeps every column as typed nulls, conformed to a
  reference schema generated from full builds; the two opponent-profile tables appear from week
  2, when a profile exists, as the source's own build does.
- **Sources**: the frames the run already holds are reused; nflverse's official weekly team stats
  are the one new cached source, loaded once a run, with the current season's download falling
  back to the cache.
- **Cost**: about a minute more on a full rebuild and about half a minute on an incremental run
  (measured under the walk-forwards' load, so high).

### Power rankings on the ported ratings and the model (`0.57.0`)

Full record: [decisions/power_rankings.md](decisions/power_rankings.md) (11 entries); the score's
rule is `models/step6/rankings/HYPOTHESIS.md` (main checkout, written before any candidate was
scored) and its second key `models/step6/rankings/REVIEW.md` (two rounds, verdict "merge as is";
the reviewer reproduced every number with an independent script).

- **What was measured**: four candidate rankings, each a rating in points known before the week,
  scored on next-week margins (MAE, 2008-2025 from week 2, 4,400 games, game and season
  bootstraps): the ported team rating, the composite production ranks on today, a model rating
  (a ridge fit of team strengths to the anchored ported-ratings walk-forward's pre-game predicted
  margins), and a blend of the ported and model ratings with its weight chosen on earlier seasons.
- **Outcome, applied as written** (seed 42, the rule's input): the blend replaces the ported
  rating (10.497 against 10.643, -0.146, games [-0.204, -0.088], seasons [-0.194, -0.102], better
  in 18 of 18 seasons); the composite (10.590) and the model rating (10.678) tie the ported rating.
  Both seeds averaged give the same outcome (blend -0.151 [-0.208, -0.094]).
- **Decided here**: the model rating's ridge penalty and week 1 (the previous regular season's
  fit carried forward by the ported prior's factor), the league home term, the blend's weight
  grid, the two bootstraps, and the three methods behind the existing setting with the composite
  still the default (one implementation serves the measurement and production, rule 11).
- **Seeds**: the two-seed supplement averages each game's predicted margin before the fit, where
  rule 13 averages per-game loss differences; the reviewer computed both, with the same outcome.
- **For the user**: whether production switches to the blend (`config/weekly_run.yaml`), and if
  so where a weekly run reads the model's predictions (the implementer recommends the run's own
  stage-1 folds plus the floor-sigma reference runs for week 1). The blend's model half comes
  from the market-anchored model, so a published blend partly carries the betting line.

### The ported quarterback rating as the own Elo's adjustment (`0.57.1`)

Full record: [decisions/elo.md](decisions/elo.md), entries 17 (the rule, committed before any
score) and 18 (the result), and the generated [ELO_QB_ALT.md](ELO_QB_ALT.md). Review: one
independent round (verdict "merge after small fixes", fixes in); the reviewer reimplemented the
method from the brief, matched every game's adjustment to 1e-13 Elo points and reproduced every
interval.

- **What was measured**: the own Elo's market-free recommended setting with its quarterback
  adjustment built from the ported rating (each starter's pre-week opponent-adjusted EPA per
  dropback, else his latest earlier season, else a replacement level; a slow team baseline as in
  538's model; 25 Elo points per point times the dropbacks per team-game of earlier seasons)
  against the same setting with 538's VALUE, on 4,943 regular-season games of 2007-2025.
- **Outcome, applied as written**: VALUE stays. The ported rating's Brier score is worse by
  +0.00524 [+0.00319, +0.00725] and its log loss by +0.01203 [+0.00736, +0.01667]; half and one
  and a half times the scale lose too, and VALUE ranks first in every week window.
- **Changed before any score**: each game reads the leakage-safe pre-week rating the ETL attaches,
  not the published week-by-week table, which uses the penalty cross-validation chose on the whole
  season (`src/nfl_predictor/ratings/pre_week.py`). The reviewer confirmed the cause by refitting
  2010 and 2013 with both penalties; with the table instead, VALUE still stays (+0.00537).
- **Finding for the user**: the published quarterback week-by-week tables (shown on the web pages,
  as in `nfl-sos-ratings`' own files) carry that whole-season penalty, a mild look-ahead for past
  seasons; no feature or evaluation reads them.

### The betting report's moneyline edge split at the spread (`0.57.2`)

Full record: [decisions/betting_edges.md](decisions/betting_edges.md), entries 1 (the rule,
committed before any game was scored) and 2 (the outcome), and the generated
[BETTING_EDGES.md](BETTING_EDGES.md). Review: two independent rounds (2026-10-10; "merge after
fixes", then "merge after one doc fix") and two re-checks of the fix commits (2026-10-12 and
2026-10-14, both "approve with findings"); the reviewers' scripts reproduce every number from the
fold checkpoints, and the scripts and re-check reports are in
[review/betting_edges/](review/betting_edges/).

- **What was measured**: each game's moneyline edge (the floor's home win probability minus the
  no-vig moneyline probability) split into what the model adds beyond the spread and the
  spread-moneyline gap, on the 4,943 regular-season games of 2007-2025 in the adopted-settings
  reference, seed 42.
- **Outcome, applied as written**: the three columns are added and the labels stay on the whole
  edge. The gap outweighs the model's part in 14.3% of the largest tenth of edges (495 games),
  against the rule's more than half. On the 815 games with real moneyline pairs it does in 18.3%,
  and on 2025 alone in 32.1% (28 games); these groups decide nothing.
- **Decided here**: the weekly run passes the report the single `floor_sigma` behind its
  predictions, and none when the column is missing, holds several values, or is not finite and
  positive; the report itself rejects such a sigma. Most seasons' moneylines are derived from the
  spread through the fitted map, so on those games the gap is a function of the spread and sigma.
- **User decision** (2026-10-11): the three columns and the sigma check are approved, the labels
  unchanged. Still open: a label test on the 815 real-moneyline games, written before it is
  scored, in step 6's integration; a tie keeps today's labels, and any change goes to the user.

### The phase-1 audit refreshed on the integrated code (`0.57.3`)

Full record: [decisions/phase1_refresh.md](decisions/phase1_refresh.md) (11 entries) and the
regenerated [INVENTORY.md](INVENTORY.md) and [PARITY.md](PARITY.md). Review: one independent
round (2026-10-12) and two re-checks of the fix rounds (2026-10-14), all "approve with findings";
the reports are in [review/audit_refresh/](review/audit_refresh/).

- **Read on**: the candidate build, through `--data-dir` (the main checkout's `data/` predates the
  integration), with every input file's sha256 recorded and downloads blocked.
- **Inventory**: 1,736 output columns, matched 1,236 (the 1,180 missing ones on the ported tables,
  26 the integration made equal), different 500 (kept on purpose, notes refreshed), missing 0;
  data sources matched 10, different 1, missing 1 (team colors); web items matched 11, different
  30, missing 4 (the garbage-time filter).
- **Parity**: 35 judged pairs; 25,564 mismatching pair values, all explained; two new
  scrimmage-EPA causes from the designed-carry rule, and nine causes the integration closed
  removed.
- **Open**: no test covers the run loop's call to `check_opponent_rows` (P3).

### The rating questions the source left open (`0.57.4`)

Full record: [decisions/open_rating_tests.md](decisions/open_rating_tests.md) (4 entries), each
test under a protocol in [protocols/](protocols/) committed before anything was scored, and the
generated [KNEEL_SPIKE_TEST.md](KNEEL_SPIKE_TEST.md),
[PENALTY_WIPED_TEST.md](PENALTY_WIPED_TEST.md), [QB_PLAYS_TEST.md](QB_PLAYS_TEST.md) and
[BOTCHED_SNAPS.md](BOTCHED_SNAPS.md). Review: one
independent round (2026-10-12) and a re-check of the fixes (2026-10-14), both "approve with
findings"; the reports are in [review/rating_tests/](review/rating_tests/). Report-only: no
rating changes.

- **Kneel-downs and spikes in the team rating**: leaving them out lowers the next-game margin MAE
  by 0.021 (10.575 to 10.554, 5,297 games), but the interval reaches +0.0005, so the rule keeps
  today's rating; a near-miss for the user.
- **Penalty-wiped plays in the team rating**: no definition differs from another; the rule's
  answer, every wiped snap, rests on the order fixed before scoring.
- **The quarterback rating from all of a quarterback's plays**: no candidate beats another; the
  rule's answer, every quarterback play, rests on the user's order alone.
- **The botched-snap charge rule**: removing the 60 snaps of 2006-2025 charged to a lineman moves
  the published quarterback ratings by at most 0.0112 EPA per dropback.
- **User decision** (2026-10-11): none is adopted before step 6 merges; each is a measured change
  after it.
