# Decisions: play-by-play corrections, one quarterback starter, the own Elo in the ETL

Roadmap step 6's integration of this repository's own play-by-play feature family, its
quarterback family and its own Elo, on branch `worktree-agent-ab77a9529b8ed64b8` (off `main`,
fast-forwarded to `feat/step6-sos-port`, with the reviewed step-6 branch merged at `a698aff` and
the ratings integration at `0cb7c00`). The scope was decided by the delegating session
(`.agents/step6_plan.md`, "Integration design"); each judgment call inside it is logged here
with the parts `.agents/sos_port/DECISIONS.md` asks for. These changes alter feature values on
purpose; the production rebuild stays held for the user.

## How the numbers were measured

Every number below was measured on real data, read-only, by scripts beside this file; their
output is kept as `<script>_output.txt` where the docs quote it.

- `integration_cache.py` builds a scratch cache in this repository's format: each 1999-2025
  play-by-play season from the saved nflverse downloads of 2026-10-08
  (`/home/mitch/scratch/nflreadpy_cache_snapshot_20261008/`, prepared with the loader's own
  `_prepare_pbp`, so it holds all 139 cached columns), the players file and the season and
  weekly rosters from the same downloads, and byte copies of the main checkout's cached
  schedules and team stats. Its manifest (`integration_cache_output.txt`) lists every file with
  its SHA-256 and source. The play-by-play has 1,225,182 regular-season plays, the same count
  the main checkout's cache holds.
- `integration_play_facts.py` counts, per season, the plays each correction touches.
- `integration_feature_values.py` runs the production code at one commit over 1999-2025 from
  that cache (`process_season` and `_finish_games`: the published schema, the `qb_elos.csv`
  columns, the quarterback family and, once wired, the own Elo), offline;
  `integration_feature_diff.py` compares two runs column by column, by season. A commit before
  the current one runs from a `git archive` export with `PYTHONPATH`. Every comparison the
  entries quote is kept in `integration_feature_diffs/<before>_<after>.txt`, named by the two
  commits: from `f601835` (the branch before the first correction) through `06e9863`, after which
  no change on this integration's branch touches the columns these runs build, and
  `7ad10ec_167f4fb.txt`, `167f4fb_d308db6.txt` and `7ad10ec_d308db6.txt` for the red-zone change
  of entry 2.9; each run used `POLARS_MAX_THREADS=8`. `integration_red_zone_movement.py` splits
  a comparison's published red-zone columns by column and season, with the sign.
- `integration_carry_parity.py`, `integration_starters.py` and `integration_red_zone_parity.py`
  compare with `nfl-sos-ratings`' published files (`/home/mitch/workspace/nfl-sos-ratings/data/`,
  read-only).

Reproduce from the repository root (the cache once, then any script):

```bash
.venv/bin/python -I .agents/sos_port/decisions/integration_cache.py \
    --snapshot /home/mitch/scratch/nflreadpy_cache_snapshot_20261008 \
    --main-cache /home/mitch/workspace/nfl-predictor/data/cache/nflreadpy \
    --out-dir /tmp/step6_int/cache --first-season 1999 --last-season 2025
.venv/bin/python -I .agents/sos_port/decisions/integration_play_facts.py \
    --cache-dir /tmp/step6_int/cache
```

## 1. One starter definition for every quarterback feature

### 1.1 The play-by-play starter rule, as ported

- **What differs**: the quarterback family keyed every game on the starter `qb_elos.csv` names
  (the identity chain: the Elo name, then `meta_data.csv`, then the play-by-play passer name),
  and the own Elo on the passer of a team's first dropback. The plan asks for one definition:
  the source's rule for a completed game, the projected starter for an upcoming one, and never
  the schedule's `away_qb_id`/`home_qb_id`.
- **Options**: the source's rule word for word ("the first first-quarter play, by `play_id`,
  whose passer or rusher has a fit row", a fit row being one of its quarterback rows with at
  least one dropback); or the brief's shorter wording, "a passer or rusher who is a quarterback
  that season", which drops the fit-row condition.
- **Decision**: the source's rule (`utils/polars/qb_starters.py`). A candidate is the passer or
  the rusher of a first-quarter regular-season play with both teams known; he counts when he
  has a row in this repository's quarterback games (`qb_stats.aggregate_qb_game_stats`: at
  least one dropback credited to him for that team in that game, as the passer or the
  scrambler) and the season identity does not list him at another position (a player no
  identity source lists can start, as the source's rows keep him); the first by `play_id`
  wins, the passer before the rusher on one play.
- **Reasoning**: without the fit-row condition a gadget runner the rosters list at quarterback
  (Taysom Hill in his tight-end seasons) would start every game whose first snap was his carry.
  The source's rule is what the plan names, and it reproduces the source exactly (below).
- **Data**: `integration_starters.py`, output in `integration_starters_output.txt` (run at
  `e865810`). Over the 13,928 regular-season team-games of 1999-2025 with play-by-play, the port
  names 13,922 starters and the source's own rule, computed from its published quarterback rows
  (`{season}_qb_game_logs.parquet`, a row with a dropback) on the same plays, names the same
  13,922 with **0** differences. In 2006-2025 both name 10,394 of 10,398, the four without a
  first-quarter quarterback snap being the source's own count (one each in 2008, 2015, 2016 and
  2017).

  Agreement, team-games where both sources name a starter and how many of those differ:

  | sources | 1999-2025 both | differ | 2006-2025 both | differ |
  | --- | --- | --- | --- | --- |
  | rule and source rule | 13,922 | 0 | 10,394 | 0 |
  | rule and identity chain | 13,922 | 12 | 10,394 | 11 |
  | rule and schedule ids | 13,922 | 98 | 10,394 | 77 |
  | identity chain and schedule ids | 13,928 | 92 | 10,398 | 72 |

  The schedule ids name a player without a quarterback-game row in that game in 44 team-games
  of 2006-2025 (4 in 2022, 33 in 2024, 7 in 2025); the source counted 42 against its own rows,
  which also include snap-count rows without a dropback. The 77 disagreements with the schedule
  in 2006-2025 match the source's count. The 12 disagreements with the identity chain, the rule's
  quarterback first and the chain's second, named from the players file: Kordell Stewart and
  Kent Graham (`2000_03_PIT_CLE`), Andrew Walter and Aaron Brooks (`2006_02_OAK_BAL`), Seneca
  Wallace and Matt Hasselbeck (`2009_11_SEA_MIN`), Tyrod Taylor and Matt Cassel
  (`2015_01_IND_BUF`), Robert Griffin III and Trace McSorley (`2019_17_PIT_BAL`), Taysom Hill and
  Andy Dalton (`2022_10_NO_PIT`), Jeff Driskel and Davis Mills (`2022_15_KC_HOU`), Tyler Huntley
  and Tim Boyle (`2024_07_MIA_IND`), Drake Maye and Joe Milton III (`2024_18_BUF_NE`), C.J.
  Stroud and Davis Mills and Will Levis and Mason Rudolph (`2024_18_HOU_TEN`), Jayden Daniels
  and Marcus Mariota (`2025_14_WAS_MIN`). In each the rule's quarterback has his team's first
  first-quarter quarterback snap and the source's rule names him too. The independent review
  checked the 12 by hand: in about 4 the rule names a package or gadget quarterback who took
  that first snap (Stewart 1 dropback against Graham's 30, Wallace 4 against Hasselbeck's 29,
  Hill 1 against Dalton's 30, Driskel 5 against Mills's 27), and in the other 8 the identity
  chain is wrong (in `2006_02_OAK_BAL` and `2015_01_IND_BUF` the chain's quarterback has no
  dropback at all). The switch is net positive, and the package cases are the rule's blind spot
  (a second quarterback whose play came before the starter's first), which the source accepts.
- **Feature-value impact**: measured with the switch, entry 1.3.
- **How to reverse it**: `qb_starters.play_by_play_starters`; pass `starters=None` to
  `qb_stats.attach_qb_features` to key every game on its named starter again.

### 1.2 A game the play-by-play names no starter for

- **What differs**: the rule names no starter for six regular-season team-games of 1999-2025
  (no first-quarter snap by a quarterback with a dropback), for every postseason game (the cache
  holds the regular season only) and for an upcoming game. The source rates the first with a
  passer effect of 0; the plan covers only upcoming games.
- **Options**: no starter (null quarterback features); the team's previous starter; the starter
  the ETL names (`away_qb`/`home_qb`: the projected starter of an upcoming game, the starter
  `qb_elos.csv` records for a completed one), mapped to an id through the identity chain.
- **Decision**: the named starter, for all three.
- **Reasoning**: for an upcoming game it is the plan's rule. For a completed game it is the
  game's recorded starter, which is what the rule would have named; the identity chain and the
  schedule agree on all six regular-season team-games (`integration_starters_output.txt`,
  "Team-games the rule names no starter for"), and the postseason rows keep the starter they
  carry today.
- **Data**: the six team-games are `2003_10_NYJ_OAK` (LV), `2005_10_SF_CHI` (SF),
  `2008_17_NE_BUF` (BUF), `2015_12_MIN_ATL` (ATL), `2016_05_TB_CAR` (CAR) and
  `2017_14_IND_BUF` (IND).
- **Feature-value impact**: none beyond entry 1.3: those rows keep the starter they had.
- **How to reverse it**: drop the `starters` fallback in
  `qb_stats._with_play_by_play_starters`.

### 1.3 The quarterback family switched to the one starter

- **What differs**: `_attach_qb_features` keyed every game on the starter `qb_elos.csv` names,
  through the identity chain.
- **Decision**: the play-by-play starter of a completed regular-season game (entry 1.1), else
  the named starter (entry 1.2); the schedule's quarterback ids are never read. The agreement was
  measured at `e865810` and logged in `146aa97` before the switch landed in `aa520c0`.
- **Data**: the measurement run at `aa520c0` logs, for the away and home sides, 6,948 and 6,946
  rows keyed on the play-by-play starter and 313 and 315 on a named starter (the 618 postseason
  team-games, the six of entry 1.2, and the four of the two published games without
  play-by-play, `2000_03_SD_KC` and `2000_06_BUF_MIA`), and no named starter unmatched.
- **Feature-value impact** (`18b82d2_aa520c0.txt`): the quarterback family moves in 11 games,
  the 12 team-games where the rule and the identity chain disagree (entry 1.1;
  `2024_18_HOU_TEN` has two): `qb_dropback_epa` by at most 0.217, `qb_history_dropbacks` by up to
  5,380 dropbacks, `qb_cpoe` by up to 1.92 (10 of the games, from 2006). Every other published
  column is unchanged.
- **How to reverse it**: as entry 1.1.

### 1.4 A scramble is credited to the scrambler

- **What differs**: nflverse leaves the passer id empty on a scramble. The quarterback family
  credited it to the team-game's busiest passer, so a starter who left early had his scrambles
  booked to his backup; the source credits it to the rusher.
- **Decision**: the rusher (`rusher_player_id`), named by his rusher name when he has no passer
  row in the game; a dropback with neither id is no quarterback's
  (`qb_stats.aggregate_qb_game_stats`).
- **Data**: two dropbacks in 1999-2025 have neither id (`integration_play_facts_output.txt`,
  `dropbacks_without_ids`, both in 2001).
- **Feature-value impact** (`03917c1_4e2d9cf.txt`; the pair spans the merge `a698aff`, whose only
  change to the ETL is a log warning, and only the quarterback family moved): quarterback games
  rise from 17,142 to 17,332 (190 games in which a quarterback only scrambled). Published, at
  most by: `qb_dropback_epa` 0.022 (7,219 games), `qb_dropback_epa_recent` 0.042 (7,219),
  `qb_sack_rate` 0.0023 (7,218), `qb_any_a` 0.0054 (6,555), `qb_any_a_recent` 0.38 (6,556) and
  `qb_history_dropbacks` 31 (6,685).
- **How to reverse it**: the `qb_id` coalesce in `qb_stats.aggregate_qb_game_stats`.

### 1.5 The quarterback running sums no rate reads

- **What differs**: the career, recent and league running sums carried completions, which no
  published rate reads, and the recent window carried the CPOE sum and count, though it publishes
  only EPA per dropback and ANY/A.
- **Decision**: drop them from the running sums; completions stay in the per-game rows, which
  VALUE reads.
- **Feature-value impact** (`4e2d9cf_18b82d2.txt`): none; every published column is identical.
- **How to reverse it**: `qb_stats._CAREER_SUM_COLUMNS` and `_RECENT_SUM_COLUMNS`.

### 1.6 The current season's play-by-play is requested once

- **What differs**: before kickoff the current season has no play-by-play, so the quarterback
  history found it missing from the loaded plays and asked nflverse for it a second time, with a
  second warning.
- **Decision**: `_EtlSources.pbp_seasons` records the seasons the run requested, and
  `_quarterback_history` loads only the others.
- **Feature-value impact**: none; the measurement runs read every season from the cache, and the
  comparison across this commit (`aa520c0_ec315e6.txt`) adds the own Elo's columns and moves
  nothing else.
- **How to reverse it**: drop `requested_seasons` from `_quarterback_history`.

### 1.7 Every quarterback feature reads one starter function

- **What differs**: the ratings integration, merged at `0cb7c00`, routes every expected-starter
  read through `qb_stats.expected_starter_ids` (the quarterback family and the ratings'
  `ratings_qb`), which mapped the named starter only; this integration had applied the
  play-by-play rule inside the quarterback family alone.
- **Decision**: the rule lives in `expected_starter_ids`, which takes the run's play-by-play
  starters (`2c5d028`): the quarterback family, `RatingsBuild.attach` (handed the run's starters
  by `_finish_games`) and the own Elo's named starters all read their starter from it.
- **Reasoning**: the plan asks for one starter definition; the delegating session asked for it
  in the one function every reader goes through, so a later reader cannot miss it.
- **Feature-value impact** (`25cc5d0_2c5d028.txt`, across the merge and this change): the 21
  ratings columns are added and the other 468 published numeric columns are identical, the
  quarterback family and the own Elo included. `ratings_qb` and `ratings_qb_dropbacks` follow
  the play-by-play starter in the 12 team-games of entry 1.1 where it differs from the named
  one; that is not measured here, because the measurement runs do not build the ratings (their
  columns are null in both runs).
- **How to reverse it**: call `expected_starter_ids` without `starters`.

### 1.8 The ratings' play-by-play history asks for each season once

- **What differs**: the ratings loader asked again for every season it found without plays, so
  before kickoff the current season was requested twice, the defect entry 1.6 fixed for the
  quarterback history.
- **Options**: a `requested_seasons` argument, as the quarterback history takes, which would
  take the loader past the argument limit and change the signature of a branch still in review;
  or asking only for the seasons before the first one the ETL loaded.
- **Decision**: the second (`ratings_build._with_earlier_seasons`, in the merge `0cb7c00`). The
  ETL's play-by-play runs from its first season through the last built one, so a season after
  the first that has no plays was asked for already.
- **Feature-value impact**: none.
- **How to reverse it**: ask for every season `_with_earlier_seasons` finds without rows.

### 1.9 The passer-name fallback reads only earlier games

- **What differs**: a named starter the identity file lacks falls back to the play-by-play
  passer name (`F.Last`) when one passer has it, and that map was built from every season's
  quarterback games, so a later passer could decide a row's id. A 2026 rookie sharing an
  abbreviation (`J.Daniels`) would leave an earlier starter of that name unidentified, and a
  debutant's own debut could name him in training rows while the same name stays unknown before
  kickoff (the independent review's finding).
- **Decision**: the fallback reads only the games strictly before the row's season and week
  (`06e9863`, inside `qb_stats.expected_starter_ids`, so every quarterback feature follows it).
- **Data**: `test_a_later_season_passer_with_the_same_abbreviation_leaves_the_name_resolved` and
  `test_a_passer_first_seen_in_the_rows_week_or_later_does_not_name_it`; the measurement run at
  `06e9863` against `2c5d028` (`2c5d028_06e9863.txt`, which also spans the step-6 merge `dea8417`
  and the warning of `8fcbb8a`) leaves every published column identical.
- **Remaining**: a completed game's starter comes from its play-by-play, by id, so a debutant
  the identity file lacks is still identified in his training rows while before kickoff his
  name has no earlier passer to match; that is the play-by-play starter convention of entries
  1.1 and 1.3, the same one the `qb_elos.csv` columns follow.
- **How to reverse it**: build the fallback from every game again (`qb_stats._resolve_ids`).

### 1.10 The players file and the weekly rosters load once

- **What differs**: a current-season run downloaded the players file and the current weekly
  roster twice, once for the ratings' sources and again for every season's quarterback identity,
  which loaded both per season (the independent review's finding).
- **Decision**: the ETL keeps the frames the ratings loaded
  (`nflverse_sources.QbIdentitySources`) and reads each season's identity from them (`d9a2c6b`);
  without them it still loads the identity per season.
- **Data**: on the measurement cache every season's identity read this way equals the per-season
  load's in all 27 seasons, 1999-2025 (`integration_identity_check.py`, output
  `integration_identity_check_output.txt`);
  `test_shared_identity_sources_give_each_season_the_loaders_identity`,
  `test_quarterback_history_reads_the_shared_identity_sources_without_loading_them`.
- **Feature-value impact**: none.
- **How to reverse it**: drop `identity_sources` from `_quarterback_history`.

## 2. Play-by-play corrections

The plan adopted these from phase 1's differences (`.agents/step6_plan.md`, "Integration
design"). Each landed as its own commit; its effect was measured by running
`integration_feature_values.py` at the commit before and at the commit itself and comparing the
two with `integration_feature_diff.py`. The full comparisons, by column and season, are in
`integration_feature_diffs/<before>_<after>.txt`; the entries quote their headlines. "Games" are
the 7,261 published 1999-2025 game rows. A change in one season reaches the next season's early
weeks through the week-1 prior and the in-season blend.

### 2.1 Scrambles filed as run plays are dropbacks

- **What differs**: before 2006 nflverse files most scrambles as run plays (`play_type` `run`,
  `qb_scramble` 1) without the `qb_dropback` flag. This repository counted them as carries when
  they carried the `rush` flag and as no snap otherwise; the source counts every scramble as a
  dropback.
- **Options**: keep the flags as published; count a run play flagged `qb_scramble` as a dropback.
- **Decision**: a dropback (`pbp._dropback_flag_expr`), hence a scrimmage snap and never a carry;
  the quarterback family credits it to the scrambler (entry 1.4).
- **Reasoning**: a scramble is a called pass play; the missing flag is a change in nflverse's
  charting in 2006, not in football, so without the rule the 1999-2005 dropback rates measure
  something different from the later ones.
- **Data**: `integration_play_facts_output.txt`: 3,744 run-play scrambles without the dropback
  flag, all in 1999-2005 (821, 440, 311, 270, 659, 711 and 532 a season), none from 2006 on. A
  scramble a penalty wiped out (`no_play`) is not one, as before.
- **Feature-value impact** (`f601835_6ed756d.txt`): dropbacks rise by those counts and offensive
  snaps by 818, 440, 311, 270, 658, 711 and 532. Five scrambles leave carries: four carried the
  `rush` flag without the dropback flag and were snaps and carries already (3 in 1999, 1 in
  2003), and the fifth, `2004_05_BUF_NYJ` play 2555, carried both flags (entry 2.2). Published: the
  per-snap, per-dropback, early-down and strength columns move in 2,092-2,097 games of 1999-2006
  (the 2006 rows read the 2005 prior; `epa_per_dropback` by at most 0.170, mean per season
  0.006-0.030), the carry rates in 830 games of 1999-2005, and the quarterback family in all
  7,261 games (`qb_dropback_epa` by at most 0.057, mean 0.016 in 1999 falling to 0.0006 in 2025,
  because its career and league sums run from 1999).
- **How to reverse it**: drop the scramble term of `pbp._dropback_flag_expr`.

### 2.2 A play flagged both a dropback and a rush counts once, as a dropback

- **What differs**: a play with both `qb_dropback` and `rush` set counted as a dropback and as a
  carry.
- **Decision**: a dropback only (the carry rule excludes the dropback flag).
- **Data**: one such play in 1999-2025, `2004_05_BUF_NYJ` play 2555, a scramble filed as a run
  play (`qb_dropback`, `rush`, `rush_attempt` and `qb_scramble` all 1), which entry 2.1 already
  counts as a dropback only.
- **Feature-value impact** (`6ed756d_0a32f48.txt`): none in 1999-2025; the rule keeps a later
  play flagged both from counting twice.
- **How to reverse it**: drop `~dropback_flag` from the carry term of `pbp._play_conditions`.

### 2.3 Carries are designed runs, the source's rule

- **What differs**: carries counted the `rush` flag (kneel-downs and two-point tries out); the
  source counts designed carries, `rush_attempt == 1 & two_point_attempt == 0 & qb_scramble == 0
  & qb_kneel == 0` (nulls as 0). The two differ on runs a penalty wiped out, which keep `rush`
  but lose `rush_attempt`, and on a few rush attempts without the `rush` flag.
- **Options**: keep the `rush` flag (this session's first decision, on the premise that
  penalty-wiped passes stay dropbacks, so wiped runs should stay carries to match); the source's
  rule.
- **Decision**: the source's rule, exactly, decided by the delegating session after the phase-1
  review (`pbp._play_conditions`). `offensive_snaps` keeps its definition.
- **Reasoning**: the premise was wrong. Of 65,122 penalty-wiped plays in 1999-2025, 8,848 keep
  the `rush` flag (-5,660.46 EPA), 27,906 keep the `pass` flag (+11,142.27 EPA) and **0** keep
  `qb_dropback` (`.agents/sos_port/parity.json` on branch `worktree-agent-a2c3c6e91a78b8738`,
  `causes.wiped_out_runs.wiped_out_plays`). Penalty-wiped passes were never dropbacks, so
  counting wiped runs as carries put their EPA into the rushing rates alone.
- **Data**: the acceptance check, `integration_carry_parity.py` (output
  `integration_carry_parity_output.txt`, on the counts of the run at `2c5d028`, the last code
  change; the runs at `1a86e18` and `25cc5d0` gave the same): over the 13,928 regular-season
  team-games of 1999-2025, this repository's carries and rush successes equal the source's
  `designed_carries` and rush successes (`rush_success_rate * designed_carries`, rounded) in
  **every** team-game: 0 differences in each, so no cause had to be named. The source's files are
  its published `{season}_team_game_logs.parquet` (`/home/mitch/workspace/nfl-sos-ratings/data/`,
  read-only; one SHA-256 over the 27 files in the output). The plays that move, counted on the
  same cache:
  8,862 leave carries (8,841 penalty-wiped `no_play` runs, 20 `punt` plays and one `kickoff`
  flagged `rush` without `rush_attempt`; -5,687.69 EPA, 2,003 successes) and 54 join (53 run
  plays and one spike with `rush_attempt` but not `rush`; -78.72 EPA, 5 successes).
- **Snaps**: `offensive_snaps` and `defensive_snaps` keep the penalty-wiped runs, as the source's
  scrimmage snap does: the per-snap EPA denominators (`off_rush_epa_per_snap`,
  `off_pass_epa_per_snap`, their allowed pair, `epa_margin_per_play` and the strength solve's
  per-snap responses) count those 8,848 plays while their EPA leaves the rushing sums. The 53 run
  plays that join carries are not scrimmage snaps (no `rush`, dropback, kneel or spike flag), so
  their EPA enters the rushing sums without a snap; the spike that joins is a snap already.
- **Feature-value impact** (`0a32f48_1a86e18.txt`): carries fall from 357,227 to 348,419 over
  1999-2025 (fewer in 6,395 team-games), the rushing EPA sum rises from -32,325.23 to -26,716.26,
  rush successes fall from 137,168 to 135,170, and snaps are unchanged. Published, every
  carry-based column moves in nearly every game of 1999-2025, at most by: `epa_per_carry` 0.110
  (7,259 games), `rush_success_rate` 0.034 (7,259), `explosive_rush_rate` 0.011 (7,257),
  `stuffed_rush_rate` 0.069 (7,259), `success_rate` 0.013 (7,259), `off_rush_epa_per_snap` 0.040
  (7,258), each allowed mirror likewise; the strength solve's rush coefficients
  `adj_off_rush_epa_snap` 0.0095 and `adj_def_rush_epa_snap` 0.0102 (all 7,261 games),
  `adj_strength_composite` 0.21, and the schedule-strength columns. The by-season tables are in
  the file.
- **How to reverse it**: the carry term of `pbp._play_conditions`.

### 2.4 Jacksonville's 2001-2002 home games: player teams repaired in the play-by-play

- **What differs**: nflverse credits every player-team column of Jacksonville's 2001-2002 home
  games to the visitors (nflverse-pbp issue 92). The repair was ported in `0.46.0` but applied to
  the weekly player stats only.
- **Options**: repair the plays when the cache is written; repair them where the ETL loads them.
- **Decision**: where the ETL loads them: `data_collection._load_team_game_frame` calls
  `nflverse_sources.repair_one_team_pbp_games` after `load_pbp`. It finds the games whose player
  teams all name one team, logs them, and rebuilds each player's team from those seasons' rosters
  (2001 and 2002 today, cache-then-degrade); a player the rosters do not place gets a null team,
  and a penalty without one takes the team its play text names.
- **Reasoning**: the cache keeps what nflverse publishes (`pbp_columns.md`, entries 4, 5 and 14),
  and the ETL's one play-by-play load feeds every family, so one call covers them all.
- **Data**: `integration_play_facts_output.txt`, "Games whose player team columns name one team":
  the 16 games (8 a season).
- **Feature-value impact** (`1a86e18_4a45570.txt`): penalties and penalty yards move between the
  two teams in 16 team-games of 2001 and 14 of 2002 (season totals unchanged); red-zone
  touchdowns rise by 14 and 15, a count no longer carried (entry 4.2). Published: `penalties`,
  `penalty_yards`, `penalty_yards_per_penalty` and their `opponent_` mirrors in 346 games of
  2001-2003, by at most 3.5 penalties and 25.8 yards a game. Every other column is unchanged.
- **How to reverse it**: remove the call from `_load_team_game_frame`.

### 2.5 Third-down, fourth-down and two-point flags read on every play

- **What differs**: the situational counts kept to run, pass, kneel and spike plays; 1999-2000
  plays often have no `play_type`, and the flags also sit on penalty-wiped plays and a punt. The
  source reads them on every play.
- **Decision**: every play (`pbp._situational_aggregations`).
- **Data**: `integration_play_facts_output.txt`, "flags outside run, pass, kneel and spike plays":
  third-down flags on 73 (1999), 76 (2000) and 5 (2018) such plays, fourth-down flags on 4, 5, 1
  (a 2013 punt), 3 (2018) and 1 (2024), and 40 two-point tries (2 in 2000 without a play type,
  38 on penalty-wiped plays in 2003-2023), which now count as attempts.
- **Feature-value impact** (`4a45570_46aedd4.txt`): `third_down_pct` in 1,009 games of 1999-2019
  (at most 6.9 points), `fourth_down_pct` in 1,419 games (at most 50 points, an early-season rate
  on one or two attempts), `two_point_conversion_pct` in 4,139 games of 2000-2024 (at most 51
  points; mean per season 0.6-30 points), and their `opponent_` mirrors.
- **How to reverse it**: restore the play-type filter in `pbp._situational_aggregations`.

### 2.6 A down flagged both converted and failed counts once

- **What differs**: such a down counted as a conversion and as a failure, so twice in the
  attempts.
- **Decision**: one attempt, a conversion (attempts are plays with either flag).
- **Data**: 12 third downs flagged both, all in 2018; no fourth down
  (`integration_play_facts_output.txt`, `third_both` and `fourth_both`).
- **Feature-value impact** (`46aedd4_03917c1.txt`): third-down attempts fall by 12 in two 2018
  team-games; `third_down_pct` and its mirror move in 299 games of 2018-2019 (at most 4.1
  points).
- **How to reverse it**: count the failed flag separately again in
  `pbp._situational_aggregations`.

### 2.7 The source loader's two repairs

`pbp_columns.md` entries 14 and 15 leave two of the source loader's repairs to the code that
reads the columns. Every reader this integration adds or changes (the team-game aggregation, the
quarterback games, the starter rule and VALUE's rushing) takes its plays from
`pbp.regular_season_plays`, which drops a play whose `posteam` or `defteam` is empty, the
source's treatment of an empty team code (entry 14). None reads a field entry 15 blanks.

### 2.8 Red-zone trips: investigated, not changed

- **What differs**: `red_zone_trips` differs from the source's published game logs in 8,238 of
  13,928 team-games, and the plan named it neither a correction nor a kept difference.
- **Cause**: this repository counts a team's drives (`fixed_drive`) with any of its plays at
  `yardline_100 <= 20` (`pbp._red_zone_drive_aggregations`). The source counts the drives
  nflverse flags inside the 20 (`drive_inside20`, its `team_stats_expanded`
  `_aggregate_drive_stats`, read from its working tree; the phase-1 mapping at `9081bce`,
  `mapping.yaml` `gl-red-zone`, records the same definition).
- **Data**: `integration_red_zone.py` (output `integration_red_zone_output.txt`, read-only on the
  measurement cache and the source's published logs). Counting `drive_inside20` on this
  repository's plays reproduces the source's `red_zone_trips` in all 13,928 team-games (0
  differences). Over 1999-2025 this repository counts 56,127 trips against the source's 43,771.
  The drives the two disagree on:
  - 10,234 whose only plays inside the 20 belong to a try (the extra point is snapped from the 15
    since 2015 and the 2 before, the two-point try from the 2, plus penalties on the try), of two
    kinds. 7,952 are touchdown drives that never snapped inside the 20 before scoring. The other
    2,282 hold the try alone after a defensive or return touchdown, which nflverse files under
    the drive that lost the ball with the scoring team as `posteam`, so grouping by team and
    drive makes it a drive of its own: their `fixed_drive_result` is "Opp touchdown" in 2,276,
    none in 5 and "Punt" in 1. 2,230 of them are a single play; 2,259 come right after a
    touchdown by the try's team on a play where it did not have the ball, and in 2,257 of those
    the try has that play's `fixed_drive`. The play before each of the other 23 is a field goal
    (6), a pass without a touchdown (5), a touchdown credited to the other team (3), an extra
    point (2), a kneel-down (2), a penalty (1) or a row without a play type (1), and 3 have no
    play before them.
  - 2,132 whose only other plays inside are at exactly the 20 (nflverse's flag means inside it).
  - 82 whose only other plays inside were wiped out by a penalty.
  - 132 with a live play inside the 20 that nflverse does not flag.
  - 224 that nflverse flags without a play of the drive inside the 20.
- **Assessment**: the 10,234 try-only drives are a bug against this repository's own definition
  (the docstring and the README say "drives that reached the red zone", and the try is snapped
  there by rule after the touchdown). The 7,952 touchdown drives make every long touchdown drive
  a converted red-zone trip, adding to both the trips and the touchdown drives; the 2,282 tries
  after a defensive or return touchdown add trips only, since their drive result is not a
  touchdown. Together the league-wide touchdown rate per trip is 0.5709 by this repository's
  count against 0.5385 by nflverse's flag. The other cases are definitional (a snap at the 20 or
  closer against nflverse's inside-the-20 flag).
- **Options**: (a) keep the count; (b) leave the try (the extra point, the two-point try and the
  penalties on them) out of the snap test, keeping this repository's definition, which still
  differs from the source in about 2,600 drives; (c) count nflverse's `drive_inside20`, the
  source's definition, which matches its logs in every team-game.
- **Decision**: kept unchanged in this integration. Either fix moves the published
  `red_zone_td_pct` and `opponent_red_zone_td_pct` (read by default, `--tr-stats-source pbp`) in
  most games and there are two plausible fixes, so the choice goes to the user, with (c)
  recommended: exact parity with the source and with nflverse's own drive flag, the try out by
  construction, and one rebuild with the other step-6 changes. Its cost is that both rates move
  in most games (not measured here).
- **How to reverse it**: not applicable.

### 2.9 Red-zone trips: the drives nflverse flags inside the 20, with a snap

- **What differs**: entry 2.8. A red-zone trip was any drive with a play at `yardline_100 <= 20`,
  so a drive whose only plays that deep belonged to the try after a touchdown was a trip, and a
  converted one when it was the scoring drive. The source counts every drive group nflverse flags
  inside the 20 (`drive_inside20`), a group being the plays of one team's `fixed_drive` in one
  game.
- **Options**: entry 2.8's three: keep the count; leave the try out of the yard-line test; count
  `drive_inside20`. Then, for a flagged group in which the team it is filed under ran no snap:
  count it, as the source does, or not.
- **Decision**: count `drive_inside20` (entry 2.8's recommendation, decided by the delegating
  session's brief of 2026-10-10, `167f4fb`), and only in a group where the team ran an offensive
  snap (decided after the independent review of that change, `d308db6`), on branch
  `worktree-agent-ad305e6aaa9e860bd` (`feat/step6-sos-port` with this integration merged at
  `7ad10ec`). A group is the team's trip when any of its plays carries the flag and any is an
  offensive scrimmage snap as `offensive_snaps` counts them, other than a two-point try
  (`pbp._with_red_zone_trip_flag`); the allowed mirror follows from the same plays. Its touchdowns
  stay the trips whose `fixed_drive_result` is `Touchdown`, the possession team's.
  - A two-point try is not the snap: it follows the score, like the extra point, which is never
    one. Three flagged groups have a two-point try as their only scrimmage play, each ending in
    the opponent's touchdown, that is, the try after a return or defensive touchdown.
  - A scramble filed as a run play before 2006 is the snap, as `offensive_snaps` counts it. Three
    flagged groups have no other: `2001_07_BUF_SD` (LAC), `2003_13_MIN_STL` (LAR) and
    `2004_08_DET_DAL` (DAL), each a touchdown drive. A snap test without that term would also drop
    them, for 230 groups instead of 227.
  - The yard line is no longer read (`pbp.RED_ZONE_YARDLINE` is removed). A frame without
    `drive_inside20` counts no trip, as one without `fixed_drive` already did; when a failed
    current-season download falls back to a cache file written before the column was requested,
    `loaders.load_pbp` now names every requested column the file lacks (`c0a1215`).
    `red_zone_td_pct` and `opponent_red_zone_td_pct` divide the same counts as before.
- **Reasoning**: nflverse's own drive flag, which with rare exceptions leaves out the try, rather
  than a second yard-line rule that would still differ from the source in about 2,600 drives. The
  rule reads any play of the group, as the source reads the flag's maximum over it, because
  nflverse leaves the flag null or 0 on some plays of a flagged drive. The snap is a **departure
  from the source**: these are this repository's own model features, and its definition credits a
  trip to the team in possession. nflverse files a kickoff the kicking team recovers under the
  receiving team, with the kicking team's next drive number, result and flag (`2002_03_NO_CHI`:
  Chicago muffed New Orleans's kickoff and New Orleans recovered at the Chicago 8, so the kickoff
  carries New Orleans's next drive, a touchdown), and the try after a return or defensive
  touchdown under the scorer, with the number of the drive the scorer stopped (`2024_01_DAL_CLE`:
  Dallas returned a punt for a touchdown, and its extra point carries Cleveland's drive number);
  neither group holds a snap of the team it is filed under. The ported season statistics keep the
  source's definition for display (`season_stats/team_stats_expanded.py`).
- **Data**: the acceptance check, `integration_red_zone_parity.py` (output
  `integration_red_zone_parity_output.txt`, on the counts of the run at `d308db6`). Over the 13,928
  regular-season team-games of 1999-2025:
  - 227 of the 43,771 flagged groups hold no snap of their team, 78 of them ending in a touchdown:
    128 a kickoff alone (75 touchdowns), 62 an extra point alone, 3 a two-point try alone, 4 an
    extra point and a kickoff, 23 a field goal alone, 1 a field goal and a kickoff, 1 a field goal
    and a penalty, 4 a penalty alone and 1 without a play type. They end in a touchdown (78), the
    opponent's touchdown (67), a field goal (58), a missed field goal (8), a turnover (5), a
    turnover on downs (5), no recorded result (4) or the end of the half (2); 33 are in 1999, 29 in
    2000 and 1-22 in each later season. The output lists every one.
  - `red_zone_trips` differs from the source's published `red_zone_trips` in 222 team-games, and
    the team's groups without a snap explain every one exactly (**0** unexplained); trips are
    43,544 against the source's 43,771, 227 fewer. `red_zone_trips_allowed` against the opponent's
    trips: the same 222, all explained.
  - Touchdown drives differ in 77 team-games, all explained exactly by the groups without a snap
    that ended in a touchdown (**0** unexplained; 23,492 against 23,570, 78 fewer), and the allowed
    mirror likewise, with the source's `red_zone_td_pct` read over its own denominator. That rate
    is the mean of `fixed_drive_result == "Touchdown"` over its trips, so a trip whose drive has no
    recorded result drops out of it. There are 44 such flagged groups: 18 in 1999, 21 in 2000, 2
    in 2002 and 1 each in 2001, 2005 and 2018. Read over all trips instead, 14 team-games are left
    unexplained.
  - This repository's per-game rate (touchdown drives over trips) differs from the source's
    game-log `red_zone_td_pct` in 209 team-games, and the allowed rate in 209; every one holds a
    group without a snap or a trip without a recorded result, which this repository counts as a
    trip without a touchdown.
  - The league touchdown rate per trip is 0.5395: 0.5709 under the yard-line rule and 0.5385 by
    the flag alone (`167f4fb`, in the output as committed at `64003d7`).
  - The flag on the plays: null on 3,999 of the 1,164,547 regular-season plays and on every play
    of 2,664 groups, and set on some plays but not others in 627 groups; a rule reading only a
    group's first play would lose 295 of the 43,771 flagged groups.
  - The try under the yard-line rule: 10,305 drives have no play at or inside the 20 but the
    try's. The 10,234 that nflverse does not flag are entry 2.8's try-only drives: 7,952 touchdown
    drives, which the yard-line rule counted as converted trips, 2,276 drives ending in the
    opponent's touchdown, which it counted as trips without a touchdown, 5 without a result and 1
    punt. nflverse flags the other 71: 4 touchdown drives (2 of them without a snap) and 67 groups
    ending in the opponent's touchdown, which are the 67 such groups without a snap above, so none
    of them is a trip here. Of the 2,343 try-only groups ending in the opponent's touchdown, 2,316
    hold nothing but the try after a return or defensive touchdown.
- **Feature-value impact** (both runs read a cache rebuilt with `integration_cache.py` whose files
  match `integration_cache_output.txt`'s SHA-256s one for one). The whole change against the
  yard-line rule, `7ad10ec_d308db6.txt`, by column and with the sign in
  `integration_red_zone_movement_output.txt`: in the team-game frame only the four red-zone counts
  move. Trips fall from 56,127 to 43,544 (in 281-326 team-games a season) and touchdown drives
  from 32,045 to 23,492 (in 208-258 a season), each allowed mirror likewise; the box scores and the
  quarterback games are identical. Published, only `red_zone_td_pct` and
  `opponent_red_zone_td_pct` move, each as `away_`, `home_` and `_diff`:
  - the four per-team columns change in 7,244-7,246 of the 7,261 games and fall in every season,
    by 2.0 (2013) to 4.7 (2010) points on average;
  - the two `_diff` columns change in 7,253 and 7,256 games, but their per-season mean moves by at
    most 0.3 points, because the drop is shared by both teams;
  - the largest change is 66.7 points, in week 2 of 1999 (one game of history and no earlier
    season), and 50.0 after 1999 (week 2 of 2002); no value turns null or back (the 2-5 nulls per
    column stay);
  - the league rate per trip falls in every season, by 1.9 (2008) to 4.4 (1999) points.

  The other 462 published numeric columns are identical. The flag alone moved the same columns
  (`7ad10ec_167f4fb.txt`); the snap alone moves the counts in 222 and 77 team-games and the two
  published rates in 7,163 and 7,151 games, by at most 16.7 points (`167f4fb_d308db6.txt`).
- **How to reverse it**: the snap term of `pbp._with_red_zone_trip_flag` (without it every flagged
  group counts, the source's definition); the flag itself, by a test on `yardline_100` again.

## 3. The own Elo in the ETL

The own Elo's own decisions are in `elo.md`; these entries cover how the integration feeds and
publishes it.

### 3.1 The quarterback inputs, and the evaluation rerun under its rule

- **What differs**: VALUE's box scores read only the passer's dropbacks, because the cache kept
  no rusher id when `elo.md` entry 10 was decided, and a completed game's starter was the passer
  of the team's first dropback. The cache now keeps the rusher and play ids (`0.50.0`), and the
  plan asks for the designed quarterback runs in VALUE and for the one starter.
- **Decision**: VALUE's rushing is every rush attempt credited to a quarterback's rusher id
  (designed runs, scrambles and kneel-downs, the box score's carries; two-point tries out; a
  player the identity lists at another position that season is left out); a completed game's
  starter is the shared play-by-play starter (`qb_starters`), and a game without one (every
  playoff game) takes the starter the ETL names, mapped as the quarterback family maps it
  (`qb_stats.named_starters`). `elo_eval.py` was rerun on these inputs under `elo.md` entry 1,
  unchanged.
- **Data**: `.agents/sos_port/ELO_EVAL.md`, generated at `4268dca` and committed in `8c3e4dc`
  (reproduction command in its header). The forward selection keeps the same components, so the
  presets do not change: travel (Brier -0.00040, interval [-0.00064, -0.00016]), the rolling home
  field (-0.00030, [-0.00059, -0.00001]), the quarterback adjustment (-0.00260, [-0.00388,
  -0.00133]) and the win-total pull (-0.00107, [-0.00200, -0.00009]) stay; rest and byes
  (-0.00019, [-0.00047, +0.00009]) and the playoff boost (+0.00001, [-0.00002, +0.00004]) stay
  out. On the 4,943 regular-season games of 2007-2025 the market-free preset scores Brier 0.21701
  and the win-total preset 0.21594, against 0.21620 for `qb_elos.csv`'s `qbelo_prob1` and
  0.22001 for its `elo_prob1`. The rolling home field again passes by a hair; the rule's outcome
  stands as written, and the phase-4 walk-forward decides whether the own Elo is used at all.
- **Feature-value impact**: the own Elo's columns, which no build carried before entry 3.2.
- **How to reverse it**: `elo_quarterbacks._rushing` (the rushing lines) and the `starters` of
  `qb_starters.QuarterbackHistory`.

### 3.2 Both presets in every build

- **What differs**: `build_own_elo_features` had no caller; the plan asks the ETL to build both
  presets.
- **Options**: for the two home win probabilities, which have no away and home pair: after the
  difference columns, among them, or with the lines.
- **Decision**: `own_elo_build.own_elo_columns` builds both presets from one set of quarterback
  adjustments, and `data_collection._attach_own_elo` runs it after the quarterback features over
  every game from 1999 on, upcoming games included, joining the eight columns on `game_id`. A
  completed regular-season game is rated for its play-by-play starter and every other game for
  the starter the ETL names (`away_qb`/`home_qb`, the projected starter of an upcoming game),
  else the team's last one. A run that builds fewer seasons reads the earlier schedules from the
  cache and names their starters from `qb_elos.csv`, so it publishes the same values as a full
  build (`tests/test_data_collection_own_elo.py`). The per-team columns sit with the other team
  stats (`away_`, `home_`, `_diff`, alphabetical) and the two probabilities right after the
  difference columns, inside the default feature range and apart from the market columns. A run
  without a quarterback history, or with a schedule that lacks a column the ratings read, logs a
  warning and publishes the columns null.
- **Reasoning**: one set of adjustments keeps the two presets apart only by the win-total pull,
  as in the evaluation; rating every game from 1999 is what the seeding and the rolling home
  field need (`elo.md`); the lines are market information, so a probability built without them
  stays out of that block.
- **Data**: the measurement runs at `ec315e6`, `25cc5d0` and `2c5d028` publish the eight columns
  on all 7,261 games with no null, and on the evaluation's window (4,943 games, 2,746 home wins)
  the two probabilities score Brier 0.21701 and 0.21594, the evaluation's own numbers to five
  places (`integration_own_elo_check.py`; output `integration_own_elo_check_output.txt`, from the
  run at `2c5d028`).
- **Feature-value impact** (`aa520c0_ec315e6.txt`): the eight columns are added; the other 461
  published numeric columns are identical. They sit inside the feature range the model reads,
  so a backtest or weekly run on a rebuilt dataset trains on both presets unless
  `--disable-feature-groups own_elo,own_elo_wt` leaves them out. Leaving the market columns out
  (`--exclude-market`, the weekly run's `--wf-market-mode none`) does not leave out the
  market-informed `own_elo_wt` (or `elo`): a run with no market information at all disables them
  itself (`--disable-feature-groups elo,own_elo_wt`, `disable_feature_groups` in the weekly
  config), and since `8fcbb8a` a run that leaves the market columns out but keeps either logs a
  warning naming it (`.agents/modeling_spec.md`); since `f8336b1` the warning names only a group
  with a column left in the spec's final features, so a group whose columns are all null or
  constant in the training rows goes unnamed. Whether the exclusion should be automatic is a
  question for the user.
- **How to reverse it**: drop the `_attach_own_elo` step from `_finish_games` and the own Elo's
  entries from `finalize.build_final_column_order`.

### 3.3 The win totals: a build without them stops, and every build records them

- **What differs**: the win-total loader falls back to the cache and then to no rows, after
  which the win-total preset would publish ratings without their pull under the same names.
- **Decision**: `own_elo_build.require_win_totals` stops the build when the win totals cover none
  of its seasons, and logs a built season after their first one that they lack (the current
  season, until nfelosrs publishes it, gets no pull). Every build writes
  `own_elo_metadata.json` beside the datasets (`constants.OWN_ELO_METADATA_NAME`): both presets'
  switches and fitted win-total weights, the first and last built season, and the win-total
  file's URL, origin (download or cache) and SHA-256; a build without the own Elo writes `{}`, so
  a record never outlives its build.
- **Options**: a key in `market_lines_metadata.json` or a file of its own.
- **Reasoning**: the line record is what `nfl-predictor lines` reads to pick its refresh path, so
  it stays the line source's alone; one writer serves both files.
- **Data**: the record the run at `2c5d028` wrote, as `integration_own_elo_check_output.txt`
  prints it: the cached file (origin `cache`, 768 rows, SHA-256 `1e6fac80...`, the copy
  `ELO_EVAL.md` read) and weights of 0.0 for 2003 (no earlier season to fit on), 1.0 for
  2004-2006 and 0.6-0.95 from 2007.
- **Feature-value impact**: none; a build that would have published the win-total preset without
  its pull stops instead.
- **How to reverse it**: `own_elo_build.require_win_totals` and
  `data_collection._save_build_record`.

### 3.4 The collection's records travel in one object

- **What differs**: `collect_all_data` handed its records back through one argument each; this
  integration added the own Elo's and the ratings integration the ratings tables, which took it
  past the argument limit at the merge.
- **Decision**: `data_collection.CollectionRecords` (the strength snapshots, the line record,
  the own Elo's record and the ratings tables), filled in by the run and written by `main`, in
  the merge `0cb7c00`.
- **Feature-value impact**: none; every file `main` writes is unchanged.
- **How to reverse it**: one argument per record again.

## 4. Columns no longer carried

### 4.1 The strength game count's difference

- **What differs**: `strength_games_played_diff` is a constant zero outside bye weeks (the two
  teams have almost always played the same number of games) and had a gain importance of exactly
  0 (`.agents/TODO.md`, the Milestone 46 follow-ups); it was already on the pruning list.
- **Decision**: the week rows no longer compute it and the schema no longer lists it
  (`constants.STATS_WITHOUT_DIFF`); the per-side columns stay, and the difference stays on
  `PRUNED_FEATURE_COLUMNS`, so a dataset that still carries it trains the same way.
- **Feature-value impact** (`ec315e6_5d70bdd.txt`): the one column is removed; every other
  published column is identical.
- **How to reverse it**: empty `constants.STATS_WITHOUT_DIFF`.

### 4.2 Play-by-play counts no published rate reads

- **What differs**: nine team-game counts were joined, aggregated and blended without any
  published rate reading them: the third- and fourth-down failures and their allowed mirrors
  (attempts already count either flag), the red-zone play and touchdown counts and their mirrors
  (`red_zone_td_pct` divides touchdown drives by trips), and the total play count.
- **Options**: publish them as raw counts, or stop carrying them.
- **Decision**: stop carrying them (the task asks for no new raw counts); a test perturbs every
  carried count and requires a published rate to move.
- **Feature-value impact** (`5d70bdd_25cc5d0.txt`): the team-game frame loses the nine columns;
  every published column is identical.
- **How to reverse it**: restore them in `pbp._situational_aggregations`, the team-game schema and
  `constants.PBP_COUNT_COLUMNS`.
