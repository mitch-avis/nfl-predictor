# Absorbing `nfl-sos-ratings`: the frozen source (roadmap step 6)

The port of `nfl-sos-ratings` into this repo starts from one frozen commit of that repo, so the
inventory, the parity tests and the ported code all describe the same thing. This file records
that commit and what its maintainer session (`nfl-sos-ratings-dc`) reported about it on
2026-10-09. That repo stays read-only from here; read it at the frozen commit
(`git -C ../nfl-sos-ratings show 9081bce:<path>`), not at whatever its checkout holds.

## The frozen commit

- `origin/main` `9081bce` (2026-10-09). Its `data/` was rebuilt from `4d23890`, which is
  code-identical to `9081bce` for every output (later merges are docs only), and
  `docs/validation-report.md` was regenerated on that data.
- The repo is frozen at this commit and that session is archived (2026-10-09): nothing will change
  there. Its open items (below, and the newer record in "Handoff after the freeze") are now this
  repo's work, done on this repo's data after the port.

## What landed there after this repo's plan was written

- **Preseason prior inside the team fit**, fading out by 9 games (`team_prior.py`, wired in
  `main.py`). The head-to-head-excluded `sos` takes its opponents' prior means from a
  previous-season refit without the evaluated team. New per-season output `{season}_team_prior`.
  It replaces this repo's after-the-fit strength blend (`strength_snapshot._blend_prior`).
- **Pre-2006 scrambles count as dropbacks** in the team stats: nflverse leaves `qb_dropback = 0`
  on most scrambles before 2006, so `pbp_expressions.dropback_expr` counts a run play flagged
  `qb_scramble` as a dropback. It changed the 1999-2005 team ratings. This repo's play-by-play
  aggregates need the same rule, or they will not match its game logs before 2006.
- **A player any source lists at quarterback in a season is a quarterback that season** (Taysom
  Hill, Terrelle Pryor and others), in `data_loader.load_qb_identity_crosswalk` and
  `_load_official_weekly_qb_stats`.
- **Quarterback EPA per dropback leaves out spikes** (new column `qb_spike_epa`;
  `qb_passing_epa` stays nflverse's official total).
- The quarterback fit is `qb_rating.fit_qb_ratings` on the shared `ridge.py` solver;
  `simultaneous_adjustment.py` no longer exists there (removed in `17070aa`).

## Open or fragile at the frozen commit (items 1-2 settled by the handoff below)

1. The quarterback all-plays test (its roadmap section "Q1") is pre-registered, not built or run.
   Two of its maintainer's questions are open: the order of simplicity, and 60 center-charged
   botched snaps still counted in the published quarterback rating.
2. Open question there: whether kneel-downs and spikes should also leave the team ratings (they
   are still scrimmage plays).
3. Ask-first data notes there (its roadmap "Data notes"): the extra-point drive after a return
   touchdown, near-duplicate yards-per-snap columns, `opp_longest_*` averaging per-game maxima,
   `fourth_down_aggressiveness` at 2.0 in two 2000 games, tackles for loss before 2012, and
   kneels under-recorded in 2000-2001.
4. nflverse quirks its loaders repair, which a port must carry:
   - Jacksonville's 2001-2002 home games credit every player to the visiting team
     (nflverse-pbp issue 92); `player_team_repair.py` rebuilds the codes from rosters.
   - The Rams appear as `LA` in every play-by-play team column.
   - The official weekly stats give a player's latest position, not the one at the time.
5. Its `check-team-prior` and `check-wp-filter` results predate the pre-2006 scramble fix; they
   have not been rerun.

## Handoff after the freeze (2026-10-09, the newer record)

The `nfl-sos-ratings` session reported on 2026-10-09 that its maintainer declared the repo frozen
at `9081bce` and that every remaining task and decision there now belongs to this repo. Nothing
will change there (no open pull requests, no branches but `main`, no worktrees), and that session
is archived. Its `.agents/current-status.md` still lists the decisions below as open because the
freeze came first; this section supersedes it. Items 1-4 are its report of the maintainer's
decisions, to confirm with the user before acting on them.

### Decisions reported as made, not implemented there

1. **Quarterback all-plays test, tie-break order** (its roadmap section "Q1. QB rating from all
   of a quarterback's plays", pre-registered, nothing built). When the test cannot tell the
   candidates apart, recommend in this order: "every quarterback play" (dropbacks, scrambles,
   designed runs, and the botched snaps the official scorer charged to him), then "plus
   scrambles", then "passes and sacks" (today's rating). This replaces the order in the protocol
   (plus scrambles, passes and sacks, every quarterback play) and is written into the protocol
   before any test code. The maintainer expects "every quarterback play" to win and prefers all
   available data on a tie. Decided earlier: if it is adopted, separate pass and run ratings are
   published beside the one-unit rating, outside the main ratings view; whether each is its own
   adjusted fit or a descriptive rate is open (a defense faces only 6 to 35 designed quarterback
   runs a season).
2. **Botched snaps in today's quarterback rating.** Approved: remove the 60 aborted snaps
   (2006-2025, -21.3 EPA) that the scorer charged to a center or other lineman
   (`fumbled_1_player_id`) and that went on as a pass or sack; the rating still counts them as
   dropbacks. The rule for every quarterback rating: a botched snap counts against a
   quarterback only when the official scorer charged the fumble to him, whoever ran with the ball
   afterwards ("9-J.Burrow FUMBLES (Aborted)" charges the quarterback; "18-K.Cousins Aborted.
   67-D.Dalman FUMBLES" charges the center). Aborted plays 2006-2025 by charged player
   (`.agents/findings_2026_10_08/aborted_snaps.py 2006 2025` there): charged to the quarterback,
   1,520 he ran with, 111 that went on as a pass or sack, 17 another player ran with; charged to
   a lineman, 255 the quarterback ran with (-585.2 EPA), 60 pass or sack, 11 another player ran
   with; charged to another player (punters, direct-snap running backs) 165; to no one (mostly
   kicks) 27. Open: whether `qb_designed_carries`, `qb_designed_rush_epa` and the team
   `designed_carries` (and so `rush_success_rate`), which count aborted snaps by rusher, follow
   the rule.
3. **Kneel-downs and spikes in the team ratings.** `pbp_expressions.scrimmage_snap_expr` counts
   them as team scrimmage plays, so they enter the offense and defense ratings. The maintainer
   wants a pre-registered test before deciding, modeled on the garbage-time filter test (which
   found that removing low-win-probability plays did not improve predictions). The quarterback
   rating already leaves both out.
4. **Plain names.** The user's global `~/.agents/AGENTS.md` ("Name things in plain words") now
   carries the rule this repo's guardrail 16 states (checked 2026-10-09).

### The quarterback all-plays test, as pre-registered there

- Candidates: "passes and sacks" (today, EPA without spikes); "plus scrambles"; "every
  quarterback play" (plus designed runs with the rush flag, and rush-flagged botched snaps
  charged to the quarterback); an extra, "every quarterback play" without the botched snaps.
- Primary metric: walk-forward mean absolute error of a team's offensive EPA per scrimmage play,
  predicted from `k1 * starting quarterback's rating + k2 * the team fit's per-play matchup
  prediction + m`.
- Starter: the first first-quarter snap by a passer or rusher with a quarterback row. The
  schedule's `home_qb_id`/`away_qb_id` are wrong in 42 team-games of 2006-2025, 31 of them in
  2024.
- Window: 2006 warms up; scoring covers 2007-2025, weeks 5 on. Inference: a season-cluster
  bootstrap (19 clusters), 98.33% intervals over three comparisons, a week-band guard, a
  t-interval sensitivity extra, and a seeded power estimate. Integrity checks 1-11, tests,
  extras and the command (`check-qb-plays --data-dir data --start-season 2007 --end-season 2025
  --start-week 5`) are specified there; two independent reviews were resolved in the protocol.
- Helpers there, in `.agents/findings_2026_10_08/`: `qb_play_types.py`, `qb_protocol_checks.py`,
  `aborted_snaps.py`, `scramble_coding.py`, `qb_epa_sources.py`.
- Window totals 2006-2025 on quarterback rows: 376,653 dropbacks, 12,422.6 EPA (+0.033 per
  play); 15,096 scrambles (+0.450); 10,734 designed runs (+0.214); 1,518 charged botched snaps
  (-2.237); left out, 7,641 kneels and 1,487 spikes. 2025 refits: "plus scrambles" moves 23
  ranks (Spearman 0.985 with today's), "every quarterback play" 26 (0.963).

### Reruns that predate the pre-2006 scramble fix (ask first)

Its preseason-prior check (the adopted 9-game horizon) and its garbage-time filter test ran on
data in which 2003-2005 (prior check) and 1999-2005 (filter test) lacked most scrambles. Neither
has been rerun.

### Open data notes (each an ask-first data change; details in its roadmap "Data notes")

- Three regular-season games have no nflverse play-by-play rows (`1999_01_BAL_STL`,
  `2000_03_SD_KC`, `2000_06_BUF_MIA`), so BAL and LAR in 1999, and KC, LAC, BUF and MIA in
  2000, are rated on 15 games (`missing_pbp_games.py`).
- The extra-point drive after a return touchdown.
- Near-duplicate per-snap yard columns (`yards_per_offensive_snap` against
  `total_yards_per_offensive_snap`, and the defensive pair) differ in 23 of 28 seasons; whether a
  per-play view keeps both is open.
- `opp_longest_pass` and `opp_longest_rush` average per-game maxima instead of taking the season
  maximum.
- `fourth_down_aggressiveness` is 2.0 in two 2000 games (`2000_04_CIN_BAL` CIN, `2000_10_SF_NO`
  SF).
- Play-by-play as a 2003-2011 source for tackles for loss.
- Kneels are under-recorded in 2000-2001 (24 and 25 on quarterback rows), so those kneels count
  as designed runs.
- In 1999-2000 nflverse flags scrambles on running backs' carries (C. Dillon, 1999); the team
  scrambles count takes them as they are.

### nflverse quirks its loaders handle (a port keeps them)

- Scrambles are coded as pass plays (`rush` 0, `pass` 1, `rush_attempt` 1) in every season, but
  `qb_dropback` is 0 on most of them before 2006 (818 of 882 in 1999, nearly all in 2003-2005).
  `dropback_expr` counts a run play flagged `qb_scramble` as a dropback; adding the pass flag
  instead would also add 783-1,156 penalty-wiped passes a season.
- Jacksonville's 2001-2002 home games credit every player to the visiting team in play-by-play
  player-team columns, weekly player stats and weekly team stats (nflverse-pbp issue 92);
  `player_team_repair` rebuilds the codes from rosters, and team stats drop one-team games.
- Play-by-play writes the Rams as `LA` in every team column and yard-line text; normalize before
  joins.
- The players file and weekly player stats give a player's latest position; quarterback identity
  uses any quarterback listing in the season (players file or weekly roster).
- Lost fumbles go to the fumbling team (`fumbled_1_team`), including on punts and after
  interceptions.
- A player's plays are grouped by id, never by name ("T.Pike (3rd QB)").
- Official `passing_epa` includes spikes; `qb_spike_epa` is the play-by-play sum the rating
  subtracts.

### Lessons from its last fixes

- `data_loader.load_qb_stats` ends with a fixed column list. A new column left off it made the
  season `qb_epa_per_dropback` silently fall back to the mean of game rates; a guard test now
  checks that every pooled quarterback rate keeps its inputs. After any rebuild, read the data
  diff for removed columns.
- The preseason prior's carryover slopes pool every earlier season pair, so any 1999-2005 change
  ripples into the early-week histories of every later season while full-season ratings stay put.

### Its data and app at the freeze

- Its `data/` was rebuilt from `4d23890` on 2026-10-09; 2026 runs through week 4. There is no
  scheduled refresh (the maintainer chose the app's refresh button), so its 2026 data goes stale.
  Season rollover is `END_YEAR` and `SEASON` in its `config.py`.
- Its validation at `9081bce`: the team rating's overall mean absolute error on next-game margins
  is 10.575, better than SRS (-0.083, 95% interval -0.155 to -0.015) and raw EPA (-0.121), and
  tied with Elo; quarterback year-over-year stability 0.455; correlation with ESPN QBR 0.892
  (Pearson) and 0.874 (Spearman). (Its numbers, from its `docs/validation-report.md`; this repo
  re-measures anything it relies on.)
- Frontend follow-ups left open there: weekly-log detail pages; grouped opponent ledgers; parked
  ideas (rank ranges at a garbage-time threshold, a head-to-head heatmap, season-over-season rank
  change, a per-commit CI gate). Its latest UI change shades context columns (schedule strength,
  opponents faced, the `opp_` and `qopp_` columns) in the active palette's primary color.
