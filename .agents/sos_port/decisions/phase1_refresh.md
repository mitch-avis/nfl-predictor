# Step 6, phase 1 refresh: the inventory and the parity check on the integrated code

The phase-1 audit scripts (`inventory.py`, `parity.py`) first compared `nfl-sos-ratings` at its
frozen commit `9081bce` with this repository's code on `main` at `ad07198`. This refresh points
them at the step-6 branch (`feat/step6-sos-port` at `9bba627`, `0.56.0`) and at the candidate
build `~/scratch/step6/data/` (built at `0.55.0`), and refreshes their judgments
(`mapping.yaml`, `parity.yaml`) for the ported and integrated code. One entry per group of
reclassifications, each with what differs, the options, the decision, the reasoning, the data and
how to reverse it. No finding: every mismatch the integration set out to close is closed, and every
difference left is one the decision log keeps or a consequence of a rule it adopted (entries 5 and
6).

Every number is quoted from a generated report or from a script checked in beside this file.
Reproduce them from the repository root with:

```bash
# INVENTORY.md, inventory.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/inventory.py --data-dir ~/scratch/step6/data
# PARITY.md, parity.json, parity_attributions.parquet
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/sos_port/parity.py --data-dir ~/scratch/step6/data
# Class moves against the reports as checked in at 9bba627 (phase1_refresh_transitions_output.txt)
.venv/bin/python .agents/sos_port/decisions/phase1_refresh_transitions.py
# Cause counts of a parity run (phase1_refresh_causes_output.txt)
.venv/bin/python .agents/sos_port/decisions/phase1_refresh_causes.py .agents/sos_port/parity.json
```

The causes as they stood before the refresh were run once on the candidate build at commit
`a9d4e6b` (the refreshed data path with the old `parity.yaml`); the first part of
`phase1_refresh_causes_output.txt` is that run.

Summary of the moves (`phase1_refresh_transitions_output.txt`):

| item | before | after |
| --- | --- | --- |
| output columns (1,736) | matched 30, different 526, missing 1,180 | matched 1,236, different 500, missing 0 |
| data sources (12) | matched 1, different 2, missing 9 | matched 10, different 1, missing 1 |
| web items (45) | matched 1, different 9, missing 35 | matched 11, different 30, missing 4 |
| parity: mismatching pair values | 40,694, all explained (main's data) | 25,564, all explained (candidate build) |

## 1. The reports read a candidate build, through `--data-dir`

- **What differs.** Both scripts read this repository's data from the main checkout's `data/`,
  which predates the integration: its play-by-play cache lacks the columns the code now reads and
  it has no `data/ratings/`.
- **Options.** (a) Wait for the user's approved rebuild of `data/`; (b) add `--data-dir` and run on
  the candidate build `~/scratch/step6/data/`; (c) build a new cache from the snapshot.
- **Decision.** (b). The default stays the main checkout's `data/`, so the commands keep working
  once `data/` is rebuilt. Each report records the directory (`data_dir`) and the sha256 of every
  file read there (`inputs`). The candidate was built at `0.55.0`, before the ETL wrote the
  season-statistics tables (`0.56.0`), so the inventory reads those kinds' columns from
  `SEASON_STATS_SCHEMAS`, the schema the ETL conforms each table to, and the ratings kinds' from
  the build's files of 2025 (and of 2026 for the two rank-range-by-week kinds). The parity check
  blocks every nflreadpy download, so a cache miss fails instead of writing to the build.
- **Reasoning.** The candidate build is the one the step-6 measurement runs on, and it was made
  by the integrated code; (c) would test a cache nothing uses. Reading the season-statistics
  columns from the code is what the ETL writes, and the ETL parity report shows those tables
  equal the published files (`SEASON_STATS_ETL_PARITY.md`).
- **Data.** Inventory: 21 file kinds, 1,736 of 1,736 columns written by the ported tables under
  their kind and name; 0 failures. Parity: 27 of 27 seasons of the snapshot equal to the cache on
  all 139 cached columns (1,225,182 plays); 111 input files recorded; 0 failures.
- **How to reverse it.** Run without `--data-dir` (it reads the main checkout's `data/`);
  `port_audit.parse_args`.
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 2. A column with two counterparts is matched only when both share its definition

- **What differs.** Every output column now has two kinds of counterpart here: the same column of
  the ported table (`ratings:<column>`, equal to the published file) and, for 553 of them, a model
  input of this repository's own (a play-by-play aggregate, a team stat, a dataset feature, the
  strength snapshot), which may define it differently. The other 1,183 have only the port.
- **Options.** (a) Class each column on its closest counterpart: nearly every column `matched`,
  since the port reproduces them all; (b) class each column on every counterpart named: `matched`
  only when the model input agrees too, else `different` with the note saying which; (c) add a
  second class field, one for the port and one for the model.
- **Decision.** (b). The inventory names the ported column first on every row it writes
  (automatically, checked per kind); the rules name the model inputs and decide the class. A
  separate summary line and a per-kind column count the columns the port writes.
- **Reasoning.** (b) keeps the classification standard (`matched` means the same definition
  here) and keeps visible the differences the decision log keeps on purpose (two-point tries,
  first downs, the red-zone snap rule, Washington as `WSH`), which (a) would hide behind the port.
  The port's progress is still one number: columns written here, 1,736 of 1,736. (c) would change
  the report's design for what a count already shows.
- **Data.** `INVENTORY.md`, Summary and Output files: of the 553 columns with a model input, 53
  are matched and 500 different; the 1,183 with only the port are all matched.
- **How to reverse it.** For (a), class every ported column `matched` in `mapping.yaml`;
  `inventory.add_port_counterparts` names the port.
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 3. Columns the integration made equal are matched (26; turnovers on downs, 3, on the port only)

- **What differs.** Before the refresh these game-log columns were `different` from this
  repository's play-by-play aggregates; the integration (`0.54.0`) adopted the source's rules for
  them: scrimmage snaps with scrambles filed as runs (`gl-snaps`, 2), designed carries and their
  success rate (`gl-designed-runs`, 2), the third-down, fourth-down and two-point flags read on
  every play with a down flagged both ways counted once (`gl-downs`, 8), the Jacksonville
  2001-2002 player-team repair (`gl-penalties`, 2), and through them the opponent's carries
  (`gl-carries-faced`, 1) and four allowed rates (`gl-defense-rates-same`: rush success, third
  down, fourth down, two-point conversion). The weekly rating history (`ratings-history`, 4) now
  has the pre-week rating features as its model counterpart. Turnovers on downs (game logs and
  the two season kinds, 3) have no model counterpart any more: the aggregates stopped counting
  failed downs, so only the port writes them.
- **Options.** Match them on the current columns, or keep them `different` until a second build.
- **Decision.** Matched, each `here` pointing at the current column. The stale notes
  (`gl-designed-runs`, `gl-red-zone`, `gl-carries-faced` and every other game-log rule) are
  rewritten for the current code. `gl-carries-faced` now names the opponent's box-score carries
  (`box_score[opponent]:rush_attempts`), not its designed carries.
- **Reasoning.** The parity check judges each game-log column of these on every team-game of
  1999-2025 and finds no mismatch, the allowed rates and the carries faced (on the opponent's row)
  included; the pre-week features equal the published week-by-week history of the week before
  (`tests/test_ratings_pre_week.py`,
  `test_pre_week_team_ratings_are_the_history_of_the_week_before`), so a published value after
  week N is the feature of its team's game in week N + 1.
- **Data.** `PARITY.md`, Judged pairs: `offensive_snaps`, `defensive_snaps`, `designed_carries`,
  `rush_success_count`, the six third-down, fourth-down and two-point pairs, `penalties`,
  `penalty_yards` and `rush_attempts` each 0 mismatched of 13,928 team-games; the four
  allowed-rate pairs and `carries_faced` are in the same table.
  `phase1_refresh_transitions_output.txt`: different to matched, 26.
- **How to reverse it.** Set the rules' class back to `different` in `mapping.yaml`.
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 4. Columns that were missing are matched on the ported tables (1,180)

- **What differs.** 1,180 columns had no counterpart here: the descriptive game-log, season and
  quarterback stats, the opponent profiles, the rank ranges and pairs, the team prior, the
  quarterback faced-defense and games columns, the win-probability bins and `games`. The season
  statistics (`0.51.0`, in the ETL since `0.56.0`) and the ratings (`0.49.0`, in the ETL since
  `0.54.0`) now write every one under its kind and name.
- **Options.** Matched, naming the ported column; or `different` where only a page shows them.
- **Decision.** Matched, `here: ratings:{column}`; the guard for missing columns now fails any
  column the port writes.
- **Reasoning.** Same kind, same name, values equal to the published files cell for cell
  (`SEASON_STATS_ETL_PARITY.md`, `RATINGS_PARITY.md`); the model reads none of them, so the port
  is their one counterpart.
- **Data.** `phase1_refresh_transitions_output.txt`: missing to matched, 1,180 (by rule:
  `gl-rest` 111, `season-rest` 222, `opponent-rest` 239, `opponent-prefix-rest` 239,
  `qb-game-rest` 26, `qb-season-rest` 112, `qb-opponents` 102, `rank-ranges` 115, `wp-bins` 7,
  `team-prior` 2, `qb-faced-defense` 2, `qb-ratings-volume` 2, `gl-games` 1).
- **How to reverse it.** Set the rules back to `missing` with a phase (the port guard then fails
  the run, as it should while the port writes them).
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 5. Differences kept on purpose stay different, with their notes refreshed (500)

- **What differs.** 500 columns keep a model input that defines them otherwise: two-point passes
  out of dropbacks (`gl-dropbacks`, `gl-sack-rate`, `gl-early-down` with null downs), first downs
  without penalties, fumbles off special teams, the explosive and stuffed definitions, success over
  dropbacks and carries, both teams' special-teams plays, play-by-play ahead of nflverse's team
  stats and nflverse's team stats where the source reads play-by-play or player stats, the
  red-zone snap rule (`0.55.0`), the season-to-date and pre-game features against season values,
  and Washington as `WSH` (in the ported tables too).
- **Options.** Keep them `different`, or class them on the port (entry 2).
- **Decision.** Kept `different`; each note now describes the current code and names the cause or
  decision that keeps it (for example `gl-red-zone`: both count the drives nflverse flags inside
  the 20, here only those with an offensive snap other than a two-point try).
- **Reasoning.** The decision log keeps each on purpose (DECISIONS.md, phase 1, red-zone trips and
  the season statistics' team codes); the brief keeps them `different` with their cause.
- **Data.** `phase1_refresh_transitions_output.txt`: different to different, 500. `PARITY.md`:
  `dropbacks`, `pass_epa_sum` and `pass_success_count` explained by `two_point_tries` alone;
  `early_down_pass_rate` by `null_downs` with `two_point_tries`; `first_downs` by
  `penalty_first_downs`; `fumbles` by `fumbles_on_scrimmage_snaps`.
- **How to reverse it.** Change a rule's class in `mapping.yaml` once its model input changes.
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 6. Scrimmage EPA: two new causes from the designed-carry rule, not a finding

- **What differs.** Scrimmage EPA (`offensive_epa`, here `pass_epa_sum + rush_epa_sum`) mismatched
  in 9,220 team-games on the candidate build. With the causes as they stood, 6,447 stayed
  unexplained. The designed-carry rule (`0.54.0`: carries are rush attempts, whatever the run
  flag) moved two kinds of plays: a run a penalty wiped out keeps the run flag but no rush attempt,
  so it is a scrimmage snap in both repositories, in `offensive_epa` there, in neither EPA sum here;
  and a rush attempt nflverse files without the run flag is a designed run in both, but no
  scrimmage snap there, so its EPA is in `rush_epa_sum` here and not in `offensive_epa` there.
- **Options.** (a) Report the 6,447 as a finding; (b) name the two causes, each with an adjustment
  that undoes it, and keep the pair; (c) drop the pair.
- **Decision.** (b): `wiped_out_run_epa` (count such a run as a rush attempt) and
  `unflagged_rush_attempt_epa` (take such a carry out of carries), each with evidence patterns on
  both repositories' code. The second applies after the two-point rule in a combination, since a
  two-point try nflverse files as a rush attempt without the run flag becomes a carry once the
  two-point rule clears its flag (Seattle's team-games `2000_16_OAK_SEA` and `2025_16_LA_SEA`
  stayed unexplained with the opposite order). `kneels_and_spikes` now gives a kneel-down the
  rush-attempt flag as well, so it becomes a carry under the new rule.
- **Reasoning.** Not a finding: the integration never set out to make scrimmage EPA equal, and
  both differences follow from the adopted carry rule. The first is kept on purpose ("penalty-wiped
  runs leave the rush sums but stay snaps", DECISIONS.md, `0.54.0`). In the second the source's own
  rush EPA (`aux_rush_epa`, over every rush attempt but a two-point try) counts the play too; only
  its scrimmage total leaves it out. The ported ratings build their inputs the source's way
  (`ADAPTER_PARITY.md`), so the team ratings are not affected.
- **Data.** `phase1_refresh_causes_output.txt`: before, `scrimmage_epa_sum` 9,220 mismatched, 2,773
  explained, 6,447 unexplained; after, 9,220 explained. Team-games explained: `wiped_out_run_epa`
  6,397, `unflagged_rush_attempt_epa` 55, `kneels_and_spikes` 3,721, `two_point_tries` 2,010.
  Tests: `test_wiped_out_run_epa_puts_a_penalty_wiped_run_into_rush_epa`,
  `test_unflagged_rush_attempt_epa_takes_an_unflagged_carry_out_of_rush_epa`,
  `test_unflagged_rush_attempt_epa_keeps_a_spike_filed_as_a_rush_attempt`,
  `test_adjust_takes_an_unflagged_two_point_run_out_after_two_point_tries`,
  `test_kneels_and_spikes_puts_every_kneel_down_into_rush_epa`.
- **How to reverse it.** Remove the two causes from `parity.yaml` and `parity_rules.CAUSES`; the
  run then fails on the unexplained team-games.
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 7. The ratings name the pre-week rating features; the season values stay different

- **What differs.** The source's ratings (`season-ratings`: offense, defense, special teams, team
  rating, `sos`, SRS in `ratings` and `combined`; `qb-ratings`: adjusted EPA per dropback) were
  compared with the strength snapshot, "Phase 3 replaces these". The ratings are ported and the ETL
  adds them as pre-week features (`ratings_*`, `0.54.0`).
- **Options.** (a) Name the pre-week features, `different` (fitted before each week); (b) name the
  port only, `matched`; (c) keep the strength snapshot.
- **Decision.** (a), and SRS keeps the strength snapshot's (`adj_srs`, the one SRS the dataset
  has). The weekly history is `matched` (entry 3). The notes say the strength snapshot's own
  ratings stay in the dataset beside the new ones.
- **Reasoning.** The model's features are the same fits, but on the games before each week with
  the previous season's penalties; only a postseason row carries the season's fit. That is the
  same distinction as the season-to-date stat features, which stay `different` too. For the
  quarterback rating, the module says the two differ slightly (the published history uses the
  season's own penalty).
- **Data.** `INVENTORY.md`, Ratings: `ratings` 8 columns, all different (the team code, the six
  ratings, `games_played`); `ratings_by_week` 5 matched, 2 different.
- **How to reverse it.** Point `season-ratings` and `qb-ratings` back at `strength:` in
  `mapping.yaml`.
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 8. Data sources: nine reclassified, team colors still missing

- **What differs.** Weekly player stats, snap counts, players, season and weekly rosters and the
  ESPN QBR files (urls and the fetch) were missing; this repository loads them since `0.46.0`. The
  play-by-play was `different` because the cache left out columns it reads; since `0.50.0` it
  carries every one. `nflreadpy.load_teams` (team colors for its pages) has no counterpart.
- **Options.** Match each on this repository's load of the same key, or keep them missing.
- **Decision.** Matched (rules name `load:{key}`, the item's own key); `load_teams` stays missing,
  "none planned", since the pages do not port the palettes (`decisions/web_pages.md` entry 13).
  The weekly team stats stay `different`: the season statistics read them as the source does, the
  model's team-game frame lets play-by-play override them. A new guard fails a source or web item
  left missing whose key this repository has.
- **Reasoning.** Same nflverse loads with the source's cache rules (`decisions/sources.md`).
- **Data.** `phase1_refresh_transitions_output.txt`, data sources: missing to matched 8,
  different to matched 1, missing to missing 1.
- **How to reverse it.** `mapping.yaml`, `sources`.
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 9. Web items: 31 of 35 missing ones now name the ported pages

- **What differs.** The team and quarterback pages were ported in `0.53.0` under one route family
  (`/api/ratings/{season}/{kind}/...`) and the components of `web/src/components/ratings/`.
- **Options.** Class each item on its counterpart, from the web-pages decision log; or keep the
  web items out of the refresh.
- **Decision.** Routes, pages and components with a logged difference are `different`, each note
  citing its `decisions/web_pages.md` entry (route family, opponents ledger, no rank-range column,
  no selection bar, the app's palette); nine components ported under their names and the renamed
  trend chart are `matched`; the garbage-time filter's two routes and two components stay missing
  ("none planned", entry 9). The inventory now also lists this app's ratings components and its
  shared table, so they can be named.
- **Reasoning.** The classes rest on the reviewed web-pages decision log, not on a component by
  component review; a component that log does not mention was ported as it was.
- **Data.** `phase1_refresh_transitions_output.txt`, web items: missing to different 21, missing
  to matched 10, missing to missing 4.
- **How to reverse it.** `mapping.yaml`, `web`; `inventory.HERE_COMPONENTS`.
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 10. Parity causes the integration closed are removed (9)

- **What differs.** Nine causes explained no team-game on the candidate build, and their
  adjustments alone changed no value they are judged by: `scramble_dropbacks`,
  `scrambles_coded_as_runs`, `dropback_and_carry`, `spike_rush_attempts`, `wiped_out_runs`,
  `unflagged_rush_attempts`, `every_play_type`, `both_conversion_flags`,
  `jacksonville_player_teams`.
- **Options.** Keep them as zero lines in the report, or remove them with their diagnostics.
- **Decision.** Removed, with the play-type and carry-decision diagnostics they carried; the
  `no_play` counts stay beside `wiped_out_run_epa`. Kept although they explain no team-game:
  `rams_written_la`, `release_difference` and `games_without_play_by_play`, the report's checks of
  team codes, the snapshot and coverage.
- **Reasoning.** The integration adopted each rule (DECISIONS.md, `0.54.0`); a cause that cannot
  change a value is no longer a candidate explanation, and the report's design needs only its
  checks.
- **Data.** `phase1_refresh_causes_output.txt`, first part (commit `a9d4e6b`): each of the nine
  explains 0 team-games and changes 0 team-games on every value it is judged by.
- **How to reverse it.** Restore them from `a9d4e6b` (`parity.yaml`, `parity_rules.py`).
- **Review.** 2026-10-12, approved with findings (Review record, below).

## 11. The parity check reads the play-by-play as the ETL does

- **What differs.** The check read the cache file directly and attached `play_id`, `qb_scramble`
  and the play text from the snapshot by row position; the cache now carries them, and the ETL
  repairs Jacksonville's 2001-2002 player teams before it aggregates. The aggregates also no longer
  count failed fourth downs, which the `turnovers_on_downs` pair read.
- **Options.** Keep reading the file as it is, or read it through the loader and the repair.
- **Decision.** Through `loaders.load_pbp` and `nflverse_sources.repair_one_team_pbp_games`, with
  downloads blocked; the snapshot must equal the cache on every cached column; nothing is attached,
  so the row-ambiguity checks go. The `turnovers_on_downs` pair is dropped (the column has only the
  port as a counterpart, entry 3).
- **Reasoning.** The parity check judges the model's play-by-play path, which includes the repair.
- **Data.** `PARITY.md`, Summary: 27 of 27 seasons equal on all 139 cached columns; 16 games
  repaired, all Jacksonville home games of 2001-2002; judged pairs 28 (29 before).
- **How to reverse it.** `parity.season_pbp` as it was at `9bba627` read the file directly.
- **Review.** 2026-10-12, approved with findings (Review record, below).

## Review record

- 2026-10-12, independent review of entries 1-11 (approve with findings,
  `~/scratch/step6_review/audit_refresh_review.md`). Fixed after it: the parity check judges the
  four allowed rates and `carries_faced` (a pair on the opponent's row, `row: opponent`), and
  `gl-carries-faced` names `box_score[opponent]:rush_attempts`; the ten web items matched by
  `decisions/web_pages.md` say so in their notes; an empty cached schedule fails the run instead
  of raising; the `no_play` EPA sums are rounded to 9 decimals, so `parity.json` is the same at any
  thread count; entry 3's heading and the `ratings-history` note's citation are corrected.
