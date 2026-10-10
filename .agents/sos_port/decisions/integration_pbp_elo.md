# Decisions: play-by-play corrections, one quarterback starter, the own Elo in the ETL

Roadmap step 6's integration of this repository's own play-by-play feature family, its
quarterback family and its own Elo, on branch `worktree-agent-ab77a9529b8ed64b8` (off `main`,
fast-forwarded to `feat/step6-sos-port`). The scope was decided by the delegating session
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
  the current one runs from a `git archive` export with `PYTHONPATH`.
- `integration_carry_parity.py` and `integration_starters.py` compare with `nfl-sos-ratings`'
  published files (`/home/mitch/workspace/nfl-sos-ratings/data/`, read-only).

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
  first-quarter quarterback snap and the source's rule names him too; which of the two started
  each game is not checked against another record here. Where the rule's quarterback did not
  start (a second quarterback with a dropback in the game whose play came before the starter's
  first), it is the rule's blind spot, which the source accepts.
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
