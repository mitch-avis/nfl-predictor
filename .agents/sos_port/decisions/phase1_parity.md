# Step 6, phase 1: decisions on the differences with nfl-sos-ratings

One entry per difference the inventory and the play-by-play parity check found between this
repository and `nfl-sos-ratings` at its frozen commit `9081bce`. The user deferred these to the
delegating session's judgment (2026-10-10) provided each is logged with its reasoning and data; the
recommendation in each entry is the implementer's, and the delegating session takes the decision
when it implements it. Phase 1 changes nothing: every change below lands in a later step-6 phase.

Every number is quoted from a generated report. Reproduce them with:

```bash
.venv/bin/python .agents/sos_port/parity.py     # PARITY.md, parity.json
.venv/bin/python .agents/sos_port/inventory.py  # INVENTORY.md, inventory.json
```

"Here minus there" is this repository's value minus the other repository's, before any adjustment.
A cause's counts are the team-games it makes match, alone or with the fewest other causes
(`parity.json`, `causes.<cause>.columns.<pair>`).

## 1. Pre-2006 scrambles are dropbacks there and in no count here

- **What differs.** There, `pbp_expressions.dropback_expr` counts a run play flagged `qb_scramble`
  as a dropback, so it is also a scrimmage snap (`scrimmage_snap_expr`); nflverse leaves
  `qb_dropback` at 0 on most scrambles before 2006. Here, `utils/polars/pbp.py`
  (`_play_conditions`, `_scrimmage_snap_expr`) needs `qb_dropback`, `rush`, `qb_kneel` or
  `qb_spike`, none of which such a scramble carries, so it is in no play-by-play count: not a snap,
  not a dropback, its EPA in no sum. This repository also cannot apply the rule today: its cached
  columns (`constants.PBP_COLUMNS`) leave out `qb_scramble`.
- **Options.**
  1. Cache `qb_scramble` and count a run play flagged `qb_scramble` as a dropback, as there.
  2. Count nflverse's scramble coding (a rush attempt coded as a pass play: `pass` 1, `rush` 0,
     `rush_attempt` 1) without the flag, the parity check's stand-in.
  3. Keep today's rule.
- **Recommended decision: option 1**, in phase 3's batch of feature-value changes, after first
  caching `qb_scramble` (entry 15) and rerunning the parity check with the flag, so its provisional
  attributions are confirmed or overturned before the rule ships. Reasoning: before 2006 this
  repository's dropback rates, snap counts and quarterback rates leave out most scrambles, which
  are among the best plays a quarterback makes, and every season's strength and quarterback
  features inherit those years through the early-season priors and career windows. Option 2 is not
  safe: the stand-in also takes plays coded like scrambles that nflverse does not flag as
  scrambles, and nearly every unexplained team-game of the check has such a play (item 3 of the
  data).
- **Data.**
  - `parity.json`, `causes.scramble_dropbacks`: 2,525 team-games in 1,266 games; every attribution
    needs the stand-in (`explained` 0 in each column).
  - Same key, `columns.offensive_snaps`: 1,810 team-games, 1999-2005, here minus there -3,726 snaps,
    largest 13 in one team-game; `columns.dropbacks` 1,813 team-games, -3,917 dropbacks;
    `columns.pass_epa_sum` 1,808 team-games, -1,152.53 EPA net and 2,325.77 absolute;
    `columns.scrimmage_epa_sum` 1,809 team-games, 2,537.00 EPA absolute.
  - `PARITY.md`, Summary: 89 unexplained pair values, 84 of them in team-games with plays only the
    `qb_scramble` flag can classify; 11,714 provisional.
- **Feature-value impact.** Yes; needs a cache re-download and an ETL rebuild. Seasons 1999-2005
  directly: `offensive_snaps`, `defensive_snaps`, `off_pass_epa_per_snap`, `off_rush_epa_per_snap`
  (its snap denominator), `epa_per_dropback`, `pass_success_rate`, `success_rate`,
  `early_down_pass_rate`, `epa_margin_per_play` and their allowed and opponent mirrors; the
  strength snapshot (`adj_*`, `sos_*`); and the quarterback features (`qb_dropback_epa`,
  `qb_dropback_epa_recent`, `qb_sack_rate`, `qb_history_dropbacks`), whose career windows carry
  those seasons into later ones.
- **How to reverse it.** Drop the scramble term from the dropback rule and rebuild; the cached
  column can stay.

## 2. Two-point tries count as dropbacks there

- **What differs.** There, `dropback_expr` has no two-point exclusion, so a two-point pass is a
  dropback and its EPA enters passing and scrimmage EPA. Here, `_play_conditions` leaves two-point
  tries out of dropbacks and carries, by design ("untimed conversion attempts rather than scrimmage
  downs"). Both count them as scrimmage snaps.
- **Options.** Keep this repository's exclusion for its features and reproduce the other rule
  inside the ported ratings; or adopt the other rule everywhere.
- **Recommended decision: keep the exclusion here**; phase 3's ported team rating uses its own
  rule internally so its tests match its published outputs. Reasoning: the exclusion is a stated,
  reasoned choice here, and a conversion try is not a down.
- **Data.** `parity.json`, `causes.two_point_tries`: `columns.dropbacks` 1,506 team-games
  (1,336 without the stand-in), 1999-2025, -2,034 dropbacks; `columns.pass_epa_sum` +36.36 EPA net,
  1,629.45 absolute; `columns.scrimmage_epa_sum` 2,009 team-games, 2,529.94 EPA absolute.
- **Feature-value impact.** None if kept.
- **How to reverse it.** Not applicable while kept; adopting it means removing `~is_two_point` from
  the dropback condition and rebuilding.

## 3. Kneel-downs and spikes enter scrimmage EPA there

- **What differs.** There, `offensive_epa` (the team rating's input) sums EPA over every scrimmage
  snap, kneel-downs and spikes included; whether they should leave the team ratings is an open
  question there with a pre-registered test unbuilt. Here, `pass_epa_sum` and `rush_epa_sum` leave
  them out while `offensive_snaps` counts them, so this repository's per-snap rates divide EPA
  without kneels, spikes and two-point tries by snaps with them.
- **Options.** Port its definition with the ratings and settle the kneel question by its
  pre-registered test; or remove kneels and spikes from the ported rating now.
- **Recommended decision: port its definition unchanged** and carry the kneel-and-spike test into
  phase 4 as a measured arm. Reasoning: the port should first reproduce its published outputs; the
  open question is a modeling choice that deserves its own measurement.
- **Data.** `parity.json`, `causes.kneels_and_spikes`, `columns.scrimmage_epa_sum`: 3,718
  team-games, 1999-2025, here minus there +5,225.00 EPA net, 6,089.19 absolute, largest 7.70.
- **Feature-value impact.** None in phase 1-3 features; the ported ratings (phase 3) use its
  definition.
- **How to reverse it.** Not applicable while unchanged.

## 4. Runs a penalty wiped out count as carries here

- **What differs.** There, designed carries and their rates start from `rush_attempt`, because
  "nflverse keeps the scramble and run flags on plays a penalty wiped out". Here, carries use the
  `rush` flag, so a run a penalty wiped out (`rush` 1, `rush_attempt` 0) is a carry, with its EPA,
  success, explosive and stuffed flags. Both repositories count such a play as a scrimmage snap
  (the `rush` flag is in both snap definitions).
- **Options.** Base carries on rush attempts here; or keep.
- **Recommended decision: base carries on rush attempts**, in phase 3's feature-value batch.
  Reasoning: the run did not happen; its EPA is the penalty's, already in the down's other values.
- **Data.** `parity.json`, `causes.wiped_out_runs`: `columns.designed_carries` 6,388 team-games, all
  without the stand-in, 1999-2025, +8,830 carries, largest 6 in one team-game;
  `columns.rush_success_count` 1,784 team-games, +2,001 successes.
- **Feature-value impact.** Yes, every season: `epa_per_carry`, `rush_success_rate`,
  `explosive_rush_rate`, `stuffed_rush_rate`, `success_rate`, `off_rush_epa_per_snap` and their
  allowed and opponent mirrors, and the strength snapshot's rush coefficients. Needs an ETL
  rebuild.
- **How to reverse it.** Restore the `rush` flag as the carry basis and rebuild.

## 5. Rush attempts nflverse does not flag as runs

- **What differs.** There, a rush attempt that is neither a scramble nor a kneel-down is a designed
  run whatever its `rush` flag. Here, a carry needs `rush` 1, so a rush attempt nflverse codes with
  `rush` 0 that is not a scramble is left out.
- **Options.** Count them as carries with entry 4's change; or keep.
- **Recommended decision: count them with entry 4's change**, which needs the `qb_scramble` flag to
  tell them from scrambles. Reasoning: one carry basis, the rush attempt, for both repositories; the
  aborted-snap rule there (its open question on botched snaps charged to linemen) should be decided
  with the quarterback rating in phase 3.
- **Data.** `parity.json`, `causes.unflagged_rush_attempts`: 5 team-games, 2001-2012 (2 of the
  `designed_carries` ones need the stand-in); `PARITY.md`, Unexplained team-games: the
  `designed_carries` rows are in team-games with a rush attempt coded as a pass play without a
  dropback flag, the plays only `qb_scramble` can classify.
- **Feature-value impact.** Yes, with entry 4: a handful of carries a season.
- **How to reverse it.** As entry 4.

## 6. Third-down, fourth-down and two-point counts read every play there

- **What differs.** There, the conversion, failure and two-point flags count on every play. Here,
  `_situational_aggregations` counts them only on run, pass, kneel and spike plays
  (`SITUATIONAL_PLAY_TYPES`), so flagged plays of another type or none are left out: the 1999-2000
  plays nflverse publishes without a play type, penalty plays (`no_play`), and one punt.
- **Options.**
  1. Count the flags on every play, as there.
  2. Count them on plays without a play type only (the 1999-2000 gap), keeping penalty plays and
     kicks out.
  3. Keep the filter.
- **Recommended decision: option 1**, in phase 3's batch. Reasoning: the flags are nflverse's own
  record of the down; whether a third down was kept alive by a play or by the defense's penalty,
  the drive went on, and one definition keeps this repository's counts equal to the ported code's.
  Option 2 is the narrower fix if penalty conversions should stay out.
- **Data.** `parity.json`, `causes.every_play_type`: 182 team-games in 173 games, 1999-2024;
  `columns.third_down_attempts` 133 team-games (1999-2018, -142 attempts net),
  `columns.third_down_conversions` 49 (1999-2000), `columns.fourth_down_attempts` 13,
  `columns.two_point_attempts` 39, `columns.two_point_successes` 33. Same key,
  `flagged_plays_left_out_here`: 160 plays without a play type (1999-2000), 47 penalty plays
  (2003-2024), 1 punt (2013).
- **Feature-value impact.** Yes: `third_down_pct`, `fourth_down_pct`, `two_point_conversion_pct`
  and their opponent mirrors, mostly 1999-2000. Needs an ETL rebuild.
- **How to reverse it.** Restore the play-type filter and rebuild.

## 7. A down flagged both converted and failed counts twice here

- **What differs.** There, an attempt is a play with either flag, counted once. Here,
  `third_down_attempts` is conversions plus failures, so a play nflverse flags both ways counts
  twice; one game (`2018_01_ATL_PHI`) has twelve such plays.
- **Options.** Count each attempt once; or keep.
- **Recommended decision: count each attempt once**, in phase 3's batch. Reasoning: it is a
  double count, not a definition.
- **Data.** `parity.json`, `causes.both_conversion_flags`, `columns.third_down_attempts`: 2
  team-games in 1 game, 2018, +7 attempts, largest 5.
- **Feature-value impact.** Yes, but only `third_down_pct` of the two teams after that game in
  2018. Needs an ETL rebuild (batched).
- **How to reverse it.** Restore the sum and rebuild.

## 8. First downs by penalty count there

- **What differs.** There, `first_downs` counts every play with `first_down` set. Here, the box
  score's `first_downs` is passing plus rushing first downs, nflverse's team-stat definition.
- **Options.** Keep this repository's; or add penalty first downs.
- **Recommended decision: keep**; phase 6 shows its definition on its pages under its own name.
  Reasoning: this repository's value matches the nflverse team stat it falls back to.
- **Data.** `parity.json`, `causes.penalty_first_downs`, `columns.first_downs`: 10,587 team-games,
  1999-2025, -21,028 first downs, largest 10.
- **Feature-value impact.** None if kept.
- **How to reverse it.** Not applicable.

## 9. Fumbles: scrimmage snaps there, plays that are not special teams here

- **What differs.** There, `fumbles` counts fumbles on scrimmage snaps. Here, the box score counts
  fumbles on every play that is not a special-teams play, mirroring nflverse's offensive fumbles,
  so a fumble on a play with no play type counts here and one on a fake kick does not. Lost fumbles
  differ more: there charged to the team that fumbled (`fumbled_1_team`), here to the team with the
  ball; the cache lacks `fumbled_1_team`, so the check does not judge them.
- **Options.** Keep; or adopt its scrimmage-snap and fumbling-team rules.
- **Recommended decision: keep the count; decide the lost-fumble rule in phase 3**, once the cache
  has `fumbled_1_team`. Reasoning: the count difference is tiny; the lost-fumble rule changes
  turnover features and deserves its own look.
- **Data.** `parity.json`, `causes.fumbles_on_scrimmage_snaps`, `columns.fumbles`: 29 team-games,
  1999-2018; `PARITY.md`, Unexplained team-games: 8 `fumbles` rows, 2008-2023, each in a team-game
  with a rush attempt coded as a pass play without a dropback flag, which the stand-in calls a
  scramble.
- **Feature-value impact.** None if kept.
- **How to reverse it.** Not applicable.

## 10. Jacksonville's 2001-2002 home games: player teams repaired there

- **What differs.** In those games nflverse names the visitor for every player team value. There,
  `player_team_repair` rebuilds `penalty_team`, `td_team` and the other player team columns from
  the season roster and the play text. Here, the cached `penalty_team` and `td_team` stay as
  nflverse wrote them, so every penalty and touchdown in those games is credited to the visitor.
- **Options.** Port the repair (it needs rosters and player ids); or keep.
- **Recommended decision: port the repair** with phase 2's roster source and phase 3's batch.
  Reasoning: the values are wrong as cached; the repair is already written and tested there.
- **Data.** `parity.json`, `causes.jacksonville_player_teams`: 30 team-games in 15 games,
  2001-2002; `columns.penalties` 178 penalties moved in total, largest 13 in one team-game;
  `columns.penalty_yards` 1,354 yards, largest 111.
- **Feature-value impact.** Yes, 2001-2002: `penalties`, `penalty_yards`,
  `penalty_yards_per_penalty`, `red_zone_td_pct` (touchdowns credited through `td_team`) and their
  opponent mirrors. Needs an ETL rebuild.
- **How to reverse it.** Drop the repair and rebuild.

## 11. Official team stats or play-by-play first

- **What differs.** There, 24 team game-log columns (passing and rushing yards, touchdowns, first
  downs, EPA, CPOE, sacks, interceptions, lost fumbles and their allowed mirrors) take nflverse's
  official weekly team stat, play-by-play only for a game with one team's row. Here, the play-by-play
  box score overrides nflverse's team stats by default (`team_stats_source` "pbp", since
  2026-09-21).
- **Options.** Keep this repository's precedence; or switch to official first.
- **Recommended decision: keep**; its ratings do not read these columns (they use scrimmage EPA and
  snaps from play-by-play), and phase 6 can show either.
- **Data.** `parity.json`, `game_log_source_facts.official_overrides` (the 24 columns, read from its
  code); `inventory.json`, `rules`, `gl-official-box` and `gl-official-allowed`.
- **Feature-value impact.** None if kept.
- **How to reverse it.** Not applicable.

## 12. Washington's team code

- **What differs.** Washington is `WAS` there and `WSH` here; every other code agrees.
- **Recommended decision: the port maps its codes through `constants.normalize_team_abbr`**, as both
  scripts do.
- **Data.** `parity.json`, `team_code_map`.
- **Feature-value impact.** None.
- **How to reverse it.** Not applicable.

## 13. Special-teams plays: one side there, both sides here

- **What differs.** There, `st_plays` is the team's own special-teams plays, the weight of its
  special-teams rating. Here, `st_plays` counts both teams' special-teams plays, the denominator of
  `st_epa_margin_per_play`. Special-teams EPA itself matches in every team-game.
- **Recommended decision: the ported special-teams rating uses its one-side count**; this
  repository's `st_plays` stays for its margin feature.
- **Data.** `parity.json`, `pairs` (`st_epa`: 0 mismatches of 13,928); `inventory.json`, `rules`,
  `gl-st-plays`.
- **Feature-value impact.** None.
- **How to reverse it.** Not applicable.

## 14. Explosive and stuffed plays

- **What differs.** There, an explosive pass is a completion of 20 or more yards, an explosive run
  10 or more yards, and a stuff a carry without kneel-downs for no gain, all over every carry. Here,
  an explosive pass is a dropback of 20 or more, an explosive run 12 or more, and a stuff a rush-flag
  carry for no gain; `utils/polars/pbp.py` documents the first two as deliberate.
- **Recommended decision: keep this repository's**; phase 6 computes its definitions for its pages.
- **Data.** `inventory.json`, `rules`, `gl-carry-rates` and `gl-explosive-pass`.
- **Feature-value impact.** None if kept.
- **How to reverse it.** Not applicable.

## 15. The play-by-play cache lacks columns the port needs

- **What differs.** There, the full play-by-play frame is read. Here, `constants.PBP_COLUMNS` caches
  a selection without `qb_scramble`, `play_id`, the play text, player ids and the player team
  columns (`fumbled_1_team` and others), which entries 1, 5, 9 and 10 and the quarterback rating
  need.
- **Options.** Extend `constants.PBP_COLUMNS` in phase 2 and re-download the cache once; or read
  the missing columns from another source.
- **Recommended decision: extend `constants.PBP_COLUMNS` in phase 2**, batched with the new nflverse
  sources, then rerun the parity check with the flag. A re-download writes under `data/` and the
  change touches `constants.py` (part of the walk-forward checkpoint fingerprint), so both need the
  user's go-ahead first.
- **Data.** `parity.json`, `scramble_flag` (`stand-in`); `inventory.json`, `sources.classified`,
  rule `pbp`.
- **Feature-value impact.** None by itself (columns are only added to the cache).
- **How to reverse it.** Remove the columns from the list; the cache keeps working.
