# Step 6, phase 1: decisions on the differences with nfl-sos-ratings

One entry per difference the inventory and the play-by-play parity check found between this
repository and `nfl-sos-ratings` at its frozen commit `9081bce`. On 2026-10-10 the user deferred
every difference between the two repositories to the delegating session (recorded in
`.agents/step6_plan.md`, "Authority", and the handoff prompt; the delegating session records it in
`TODO.md`). Adopted changes are implemented on the step-6 integration branch; the production data
rebuild and the merge stay held for the user's final approval. Each entry's "Recommended decision"
is the implementer's; the delegating session records the decision taken in
`.agents/sos_port/DECISIONS.md`.

Every number is quoted from a generated report. Reproduce them with:

```bash
.venv/bin/python .agents/sos_port/parity.py     # PARITY.md, parity.json, parity_attributions.parquet
.venv/bin/python .agents/sos_port/inventory.py  # INVENTORY.md, inventory.json
```

Two kinds of count appear. **Explained** is the team-games a cause makes match, alone or with the
fewest other causes (`parity.json`, `causes.<cause>.columns.<pair>`). **Effect** is the cause's
own adjustment applied alone to every team-game, adjusted value minus unadjusted
(`causes.<cause>.effect.<value>`), on this repository's side unless the entry says otherwise. The
check reads `qb_scramble` and `play_id` from the snapshot of the play-by-play the other repository
read, which equals this repository's cache row by row in all 27 seasons (`PARITY.md`, Summary), and
it now explains every mismatching team-game (`parity.json`, `failures`: none).

## 1. Pre-2006 scrambles are dropbacks there and in no snap or dropback count here

- **What differs.** There, `pbp_expressions.dropback_expr` counts a run play flagged `qb_scramble`
  as a dropback, so it is also a scrimmage snap (`scrimmage_snap_expr`); nflverse leaves
  `qb_dropback` at 0 on most scrambles before 2006. Here, `utils/polars/pbp.py`
  (`_play_conditions`, `_scrimmage_snap_expr`) needs `qb_dropback`, `rush`, `qb_kneel` or
  `qb_spike`, none of which such a scramble carries, so it is not a snap or a dropback and its EPA
  is in neither `pass_epa_sum` nor `rush_epa_sum`. The box score does count it: its
  `rush_attempts`, `rush_yards` and `rushing_epa` read the `rush_attempt` flag, which a scramble
  carries. This repository cannot apply the rule from its own cache, which leaves out
  `qb_scramble` (`constants.PBP_COLUMNS`).
- **Options.**
  1. Cache `qb_scramble` and count a run play flagged `qb_scramble` as a dropback, as there.
  2. Keep today's rule.
- **Recommended decision: option 1**, in phase 3's batch of feature-value changes, once the cache
  carries `qb_scramble` (entry 15). Reasoning: before 2006 this repository's snap counts, dropback
  rates and quarterback rates leave out most scrambles, and every later season inherits those years
  through the early-season priors and the quarterback career windows. With the flag, the rule
  explains every pre-2006 snap and dropback difference by itself.
- **Data.** `parity.json`, `causes.scramble_dropbacks`: explains 2,527 team-games in 1,264 games,
  1999-2005; alone in all 1,816 `offensive_snaps` and all 1,816 `defensive_snaps` team-games.
  Effect: `dropbacks (here)` +3,744 in 1,817 team-games; `offensive_snaps (here)` and
  `defensive_snaps (here)` +3,740 each; `pass_epa_sum (here)` +1,158.55 EPA net, 2,286.20
  absolute.
- **Feature-value impact.** Yes; needs the cache extension and an ETL rebuild. Seasons 1999-2005
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
- **Data.** `parity.json`, `causes.two_point_tries`: explains 2,010 team-games, 1999-2025. Effect:
  `dropbacks (here)` +1,671 (the two-point passes) in 1,507 team-games; `pass_epa_sum (here)`
  -140.44 EPA net (their EPA), 1,543.97 absolute.
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
- **Data.** `parity.json`, `causes.kneels_and_spikes`: explains 3,722 team-games, 1999-2025.
  Effect: `scrimmage_epa_sum (here)` -5,516.25 EPA net (their EPA), 5,799.96 absolute; of it
  `rush_epa_sum` -5,266.91 (kneels) and `pass_epa_sum` -249.33 (spikes).
- **Feature-value impact.** None in this repository's features; the ported ratings (phase 3) use
  its definition.
- **How to reverse it.** Not applicable while unchanged.

## 4. Runs a penalty wiped out count as carries here

- **What differs.** There, designed carries and their rates start from `rush_attempt`, because
  "nflverse keeps the scramble and run flags on plays a penalty wiped out". Here, carries use the
  `rush` flag, so a run a penalty wiped out (`rush` 1, `rush_attempt` 0) is a carry, with its EPA,
  success, explosive and stuffed flags. nflverse keeps one row per play, so that EPA (the penalty's
  effect on the down) is in no other row: dropping the play from carries would drop its EPA from
  every EPA sum here, while `offensive_snaps` still counts it (the `rush` flag is in both
  repositories' snap definitions). Passes a penalty wiped out are different: nflverse leaves their
  `pass` flag set but their `qb_dropback` at 0, so they are dropbacks in neither repository, and
  their EPA is in no EPA sum here.
- **Options.**
  1. Base carries on rush attempts, as there: wiped-out runs leave carries and the EPA sums.
  2. Keep the `rush` flag for carries (today).
  3. Keep the `rush` flag and also count passes a penalty wiped out (the `pass` flag) as dropbacks,
     so both kinds of penalty play count, as the usual nflverse `pass == 1 | rush == 1` analyses do.
- **Recommended decision: option 1, or option 3 if penalty plays should count.** Reasoning: today's
  rule is asymmetric: it puts the wiped-out runs' EPA (a net loss) in the run game and leaves out the
  wiped-out passes' EPA (a net gain), so it understates rushing EPA against passing. Either option
  makes the two sides agree; option 2 keeps the asymmetry. The ported ratings keep the source's
  definitions internally either way.
- **Data.** `parity.json`, `causes.wiped_out_runs`: explains 6,398 team-games, 1999-2025. Effect of
  option 1: `designed_carries (here)` -8,862 carries; `rush_epa_sum` +5,687.69 (the dropped plays'
  EPA is -5,687.69). Same key, `wiped_out_plays`: 65,122 plays a penalty wiped out; 8,848 keep the
  `rush` flag (-5,660.46 EPA), 27,906 keep the `pass` flag (+11,142.27 EPA), 0 keep `qb_dropback`.
- **Feature-value impact.** Options 1 and 3, every season: `epa_per_carry`, `rush_success_rate`,
  `explosive_rush_rate`, `stuffed_rush_rate`, `success_rate`, `off_rush_epa_per_snap` (and with
  option 3 the dropback-based features) with their allowed and opponent mirrors, and the strength
  snapshot. Needs an ETL rebuild. Option 2: none.
- **How to reverse it.** Restore the carry and dropback conditions and rebuild.

## 5. Rush attempts nflverse does not flag as runs

- **What differs.** There, a rush attempt that is neither a scramble nor a kneel-down is a designed
  run whatever its `rush` flag. Here, a carry needs `rush` 1, so a rush attempt nflverse codes with
  `rush` 0 that is not a scramble is left out.
- **Options.** Count them as carries with entry 4's option 1; or keep.
- **Recommended decision: count them with entry 4's option 1** (one carry basis, the rush attempt,
  in both repositories). The aborted-snap rule there (its open question on botched snaps charged to
  linemen) should be decided with the quarterback rating in phase 3.
- **Data.** `parity.json`, `causes.unflagged_rush_attempts`: explains 37 team-games, 2001-2025.
  Effect: `designed_carries (here)` +53 carries in 53 team-games.
- **Feature-value impact.** Yes, with entry 4: a few carries a season.
- **How to reverse it.** As entry 4.

## 6. Third-down, fourth-down and two-point counts read every play there

- **What differs.** There, the conversion, failure and two-point flags count on every play. Here,
  `_situational_aggregations` counts them only on run, pass, kneel and spike plays
  (`SITUATIONAL_PLAY_TYPES`), so flagged plays of another type or none are left out: the 1999-2000
  plays nflverse publishes without a play type, penalty plays (`no_play`), and one punt. Its
  `turnovers_on_downs` is this repository's `fourth_down_fails` under the same rule.
- **Options.**
  1. Count the flags on every play, as there.
  2. Count them on plays without a play type only (the 1999-2000 gap), keeping penalty plays and
     kicks out.
  3. Keep the filter.
- **Recommended decision: option 1**, in phase 3's batch. Reasoning: the flags are nflverse's own
  record of the down; whether a third down was kept alive by a play or by the defense's penalty,
  the drive went on, and one definition keeps this repository's counts equal to the ported code's.
  Option 2 is the narrower fix if penalty conversions should stay out.
- **Data.** `parity.json`, `causes.every_play_type`: explains 182 team-games in 173 games;
  `third_down_attempts` 133 team-games in 1999-2000 and 2018 (effect +154 attempts),
  `third_down_conversions` 49 in 1999-2000 (+50), `fourth_down_attempts` 13 (+14),
  `turnovers_on_downs` 7 (+8), `two_point_attempts` 39 (+40), `two_point_successes` 33 (+33). Same
  key, `flagged_plays_left_out_here`: 160 plays without a play type (1999-2000), 47 penalty plays
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
- **Data.** `parity.json`, `causes.both_conversion_flags`: explains 2 team-games in 1 game (2018),
  with the play-type rule; effect `third_down_attempts (here)` -12 attempts in those 2 team-games.
- **Feature-value impact.** Yes, but only the 2018 `third_down_pct` of the two teams after that
  game. Rides the same ETL rebuild.
- **How to reverse it.** Restore the sum and rebuild.

## 8. First downs by penalty count there

- **What differs.** There, `first_downs` counts every play with `first_down` set. Here, the box
  score's `first_downs` is passing plus rushing first downs, nflverse's team-stat definition.
- **Options.** Keep this repository's; or add penalty first downs.
- **Recommended decision: keep**; phase 6 shows its definition on its pages under its own name.
  Reasoning: this repository's value matches the nflverse team stat it falls back to.
- **Data.** `parity.json`, `causes.penalty_first_downs`: explains 10,587 team-games, 1999-2025;
  effect `first_downs (here)` +21,028 first downs.
- **Feature-value impact.** None if kept.
- **How to reverse it.** Not applicable.

## 9. Fumbles: scrimmage snaps there, plays that are not special teams here

- **What differs.** There, `fumbles` counts fumbles on scrimmage snaps (run plays flagged
  `qb_scramble` included). Here, the box score counts fumbles on every play that is not a
  special-teams play, mirroring nflverse's offensive fumbles, so a fumble on a play that is neither
  (no play type, an aborted snap) counts here and one on a fake kick does not. Lost fumbles differ
  more: there charged to the team that fumbled (`fumbled_1_team`), here to the team with the ball;
  the cache lacks `fumbled_1_team`, so the check does not judge them.
- **Options.** Keep; or adopt its scrimmage-snap and fumbling-team rules.
- **Recommended decision: keep the count; decide the lost-fumble rule in phase 3**, once the cache
  has `fumbled_1_team`. Reasoning: the count difference is small; the lost-fumble rule changes
  turnover features and deserves its own look.
- **Data.** `parity.json`, `causes.fumbles_on_scrimmage_snaps`: explains 37 team-games, 1999-2023;
  effect `fumbles (here)` +3 net, 37 absolute.
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
- **Data.** `parity.json`, `causes.jacksonville_player_teams`: explains 30 team-games in 15 games,
  2001-2002. Effect on its side (`penalties (there)`, `penalty_yards (there)`): 178 penalties and
  1,354 yards absolute, each moved penalty counted on both teams' rows, so 89 penalties and 677
  yards moved.
- **Feature-value impact.** Yes, 2001-2002: `penalties`, `penalty_yards`,
  `penalty_yards_per_penalty`, `red_zone_td_pct` (touchdowns credited through `td_team`) and their
  opponent mirrors. Needs an ETL rebuild.
- **How to reverse it.** Drop the repair and rebuild.

## 11. Official team stats or play-by-play first

- **What differs.** There, 24 team game-log columns (passing and rushing yards, touchdowns, first
  downs, EPA, CPOE, sacks, interceptions, lost fumbles and their allowed mirrors) take nflverse's
  official weekly team stat, play-by-play only for a game with one team's row. Here, the
  play-by-play box score overrides nflverse's team stats by default (`team_stats_source` "pbp",
  since 2026-09-21).
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
- **Data.** `parity.json`, `pairs` (`st_epa`: no mismatch in 13,928 team-games); `inventory.json`,
  `rules`, `gl-st-plays`.
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
  a selection: it keeps `passer_player_id` and `passer_player_name` but leaves out `qb_scramble`,
  `play_id`, the play text, the other player ids and the player team columns (`fumbled_1_team` and
  others), which entries 1, 5, 9 and 10 and the quarterback rating need.
- **Options.** Extend `constants.PBP_COLUMNS` and re-download the cache once; or read the missing
  columns from another source.
- **Recommended decision: extend `constants.PBP_COLUMNS`**, which the port needs and which is being
  done separately. For phase 1 the snapshot of the play-by-play the other repository read is the
  source of `qb_scramble` and `play_id`: the parity check attaches them only after the snapshot
  equals this repository's cache row by row. A re-download writes under `data/` and the change
  touches `constants.py` (part of the walk-forward checkpoint fingerprint).
- **Data.** `PARITY.md`, Summary (27 of 27 seasons equal row by row); `parity.json`, `snapshot`
  and `causes.release_difference`; `inventory.json`, `sources.classified`, rule `pbp`.
- **Feature-value impact.** None by itself (columns are only added to the cache).
- **How to reverse it.** Remove the columns from the list; the cache keeps working.

## 16. Scrambles nflverse codes as runs

- **What differs.** nflverse flags five run plays both `qb_scramble` and `rush` (1999-2004). There,
  such a play is a dropback (`dropback_expr`) and a scramble, so not a designed run. Here, the
  `rush` flag makes it a carry, its EPA in `rush_epa_sum`.
- **Options.** With entry 1's rule, take the `rush` flag off plays flagged `qb_scramble`; or leave
  them carries.
- **Recommended decision: take the `rush` flag off them with entry 1's change.** Reasoning: once a
  flagged scramble is a dropback here, leaving it a carry too would count its EPA twice.
- **Data.** `parity.json`, `causes.scrambles_coded_as_runs`: explains 4 team-games in 4 games,
  1999-2004; effect `designed_carries (here)` -5 carries, `rush_epa_sum` -2.59 EPA net.
- **Feature-value impact.** Yes, with entry 1: five plays.
- **How to reverse it.** As entry 1.

## 17. A play flagged both a dropback and a run counts in both here

- **What differs.** Here, the dropback and carry conditions are independent, so a play with
  `qb_dropback` and `rush` both set is a dropback and a carry, its EPA in both `pass_epa_sum` and
  `rush_epa_sum`; there, scrimmage EPA counts every snap once. One play in the cache is flagged both
  ways (`2004_05_BUF_NYJ`, a scramble coded as a run).
- **Options.** Count such a play once, as a dropback; or keep.
- **Recommended decision: count it once, as a dropback**, in phase 3's batch. Reasoning: a double
  count, not a definition.
- **Data.** `parity.json`, `causes.dropback_and_carry`: explains 1 team-game (2004), where entry 16
  matches too; effect `rush_epa_sum` -0.86 EPA.
- **Feature-value impact.** Yes, one team-game in 2004.
- **How to reverse it.** Restore the independent conditions and rebuild.

## 18. A spike also recorded as a rush attempt

- **What differs.** One spike (`2001_05_SD_NE`, a reversed spike nflverse also records as an
  aborted rush attempt) is a designed run there (a rush attempt that is neither a scramble nor a
  kneel-down). Here, a spike is never a carry.
- **Options.** Count it as a carry with entry 4's option 1 (rush attempts); or keep.
- **Recommended decision: follow entry 4's decision.** One play does not justify a rule of its own.
- **Data.** `parity.json`, `causes.spike_rush_attempts`: explains 1 team-game (2001); effect
  `designed_carries (here)` +1.
- **Feature-value impact.** Negligible.
- **How to reverse it.** As entry 4.
