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
- If a published rating changes before step 6's inventory is frozen (the open quarterback items
  below), that session messages a new commit; record it here and say which items it changed.

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

## Open or fragile at the frozen commit

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
