# This repository's own Elo: decision log

Decisions taken while building the own Elo, the step-6 phase-4 candidate that would replace the
columns read from `data/qb_elos.csv` (`.agents/TODO.md`, "Roadmap Status", step 6, item 4). The
code is `src/nfl_predictor/utils/polars/elo_ratings.py` (the ratings and the feature builder),
`elo_game_context.py` (byes, travel, rolling home field), `elo_quarterbacks.py` (the VALUE
quarterback adjustment) and `elo_win_totals.py` (the win-total source).

Each entry gives what differs from FiveThirtyEight's (538's) published method or from the plan,
the options, the decision and its reasoning, the data behind it with the command that reproduces
it, the effect on feature values, how to reverse it, and who reviewed it. Sources: 538's "How Our
NFL Predictions Work" (saved as `~/scratch/elo_research/538_method.html`), its
`fivethirtyeight/nfl-elo-game` repository at `fbec1af` (`forecast.py`, `data/nfl_games.csv`), and
538's published game file as copied into `data/qb_elos.csv` (538's own numbers through the 2022
season, `../nfeloqb`'s from 2023).

Every number below the rule is quoted from `ELO_EVAL.md` (or its JSON twin), generated at commit
`e0c0663` from the main checkout's data, read-only (the numbers of entries 2-14 are unchanged
since `ff14dec`); that report is in the history at `e0c0663`. It was reproduced from the
repository root with:

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python .agents/sos_port/elo_eval.py
```

The evaluation was rerun under the same rule at `4268dca` on the integration's quarterback
inputs: the shared play-by-play starter, every quarterback rush in VALUE, and the starter
`qb_elos.csv` names for playoff games (`integration_pbp_elo.md`, section 3). It kept the same
components, so the presets did not change; the current `ELO_EVAL.md` holds the rerun's numbers,
and the entries below keep the numbers they were decided on.

The fixture test that pins the framework runs with:

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python -m pytest tests/test_elo_ratings.py -k fivethirtyeight -n 0
```

No entry changes any feature value today: `build_own_elo_features` has no caller, so the ETL
writes exactly what it wrote before. Each "Feature-value impact" therefore says what the decision
does to the own Elo's columns once the integration stream wires them in.

## 1. The rule for keeping components

Written and committed before any variant was scored.

**What differs**: the plan lists four component families after the framework (538's 2019
additions, the quarterback adjustment, the win-total pull) and says the own Elo replaces the
`qb_elos.csv` columns only if it measures at least as well. Which components the recommended
setting keeps is decided here, without walk-forward runs, by scoring each variant's own pre-game
win probability.

**Options**: keep every 538 component because 538 kept it; keep a component only when it
improves this repository's games by more than chance; pick the best-scoring combination over all
subsets (selection bias over 64 combinations, and no stated order).

**Decision**: forward selection in the build order, one component at a time, on one fixed window,
with a stated interval test:

- Window: every regular-season game of 2007-2025 with a final score, from this repository's
  schedule frame (the cached nflverse schedules). The outcome is a home win; a tie counts as not
  one, as in the walk-forward.
- Score: each variant's own pre-game home win probability. Brier score governs; log loss guards.
- Paired difference: candidate minus current, per game, averaged; its 95% interval is the 2.5th
  and 97.5th percentiles of 5,000 resamples of the games with replacement (`numpy` generator seed
  `42`), the walk-forward's bootstrap.
- Seeding first: on the framework, 538's 1999 preseason ratings against every team at 1505 with
  a burn-in. Every team at 1505 is the simpler variant (no outside numbers); the seeds are kept
  only if they pass the test below.
- Then, starting from the framework with the chosen seeding, in this order: rest and bye, travel
  (with its 55-point base), playoff boost, rolling home field, quarterback adjustment, win-total
  pull. Each candidate is the components kept so far plus the next one.
- A candidate is kept if and only if its Brier difference interval lies entirely below zero and
  its log-loss difference interval does not lie entirely above zero. Otherwise it is dropped, the
  simpler variant stays, and the next component is tested against the components kept so far. An
  interval that touches zero is not below it.
- The playoff boost changes playoff games directly and regular-season games only through the
  ratings it carries into the next season. It is judged by the same rule on the same
  regular-season window; its effect on playoff games is reported beside it and governs nothing.
- The win-total pull is tested last. The market-free recommendation is the kept set before it,
  and the anchored recommendation adds it only if it passes.
- The comparisons with the `qb_elos.csv` probabilities (538's classic `elo_prob1` and its
  quarterback model's `qbelo_prob1` on the same games), the per-season and per-week breakdowns
  and the checks against 538's published numbers are reported beside the steps and decide
  nothing here: whether the own Elo replaces the `qb_elos.csv` columns is phase 4's walk-forward
  question.

**Reasoning**: the build order puts the components 538 found most robust first and the market one
last, so each candidate answers one question with the earlier answers fixed. A component that
cannot show an improvement larger than resampling noise on 4,943 games is not kept on 538's word
alone (`AGENTS.md` rule 9: when a measurement ties, the simpler setting is the recommendation).
Log loss guards against a component that sharpens probabilities into overconfidence.

**Data**: none; this entry precedes every number.

**Feature-value impact**: none by itself. The kept set becomes the recommended setting of
`build_own_elo_features`, which no ETL step calls yet.

**How to reverse it**: change the rule here before rerunning `.agents/sos_port/elo_eval.py`; the
script prints every step and every interval, so another rule can be applied to the same numbers.

**Review**: not yet reviewed.

## 2. Seeding the first season

**What differs**: nflverse schedules start in 1999, 538's ratings in 1920. The ratings need a
1999 starting point.

**Options**: 538's 1999 preseason ratings (each team's published rating before its first 1999
game: its end-of-1998 rating reverted a third toward 1505, and Cleveland at 1300 as a new team);
or every team at 1505 with an eight-season burn-in before the 2007 evaluation start.

**Decision**: every team at 1505 (`EloInputs.initial_ratings` left empty). A team first seen
after the first season is an expansion team at 1300 (Houston in 2002); Cleveland, new in 1999,
starts at 1505 because 1999 is the first season.

**Reasoning**: the rule (entry 1) keeps the seeds only if they improve the window's Brier score
beyond resampling noise. They change nothing measurable: by 2007 the offseason regressions and
eight seasons of games have erased the starting points.

**Data**: seeds minus 1505 on the 4,943 window games: Brier `-3.57e-09`, interval `[-3.66e-08,
+2.90e-08]`; log loss `-4.14e-09`, interval `[-7.60e-08, +6.56e-08]`; the largest change in any
window probability is `0.000021` (`ELO_EVAL.md`, "Seeding").

**Feature-value impact**: own Elo values in 1999-2006 differ from a 538-seeded run; from 2007 on
the difference is at most `0.000021` in probability.

**How to reverse it**: pass `EloInputs(initial_ratings=...)` with the table in `ELO_EVAL.md`,
"538's 1999 preseason ratings".

**Review**: not yet reviewed.

## 3. Pinning the framework to 538's published numbers

**What differs**: the plan asks for a test that reproduces 538's published ratings and
probabilities to floating precision. 538 stored its ratings to three decimals through 2014 and to
full precision from 2015, so a replay that starts before 2015 can match only to about `2e-6` in
probability.

**Options**: a fixture from before 2015 with a looser tolerance; a fixture from 2015 on, seeded
with each team's first published rating in it, at floating precision; or the whole 1999-2020
replay on this repository's schedules.

**Decision**: the fixture is every 2017-2019 row of `nfl-elo-game`'s `data/nfl_games.csv` (801
games, MIT License, notice kept beside it in `tests/fixtures/fivethirtyeight_elo/`), and
`tests/test_elo_ratings.py` requires every pre-game rating within `1e-9` and every probability
within `1e-12`. The long replay is a check in the evaluation, not a test, because it needs the
cached schedules.

**Reasoning**: three seasons hold two offseason reversions, three ties, London and Mexico City
games and three Super Bowls, which exercise every branch of the framework, and they need no data
outside the repository.

**Data**: the fixture test passes. On this repository's schedules (`ELO_EVAL.md`, "The framework
on this repository's schedules"), seeded with 538's 1999 ratings, 5,822 of 5,852 published
1999-2020 games match on season, date and teams, with a median probability gap of `1.82e-06`
(1999-2014) and `4.60e-06` (2015-2020) but a maximum of `0.093`. Run again with 538's home side
and neutral flag on every game 538 lists, all 5,852 match, with a largest probability gap of
`1.82e-06` (1999-2014) and `1.56e-06` (2015-2020) and none over `1e-4`: the remaining gaps are
538's three-decimal rounding and its decay.

**Feature-value impact**: none beyond entry 4.

**How to reverse it**: the fixture and its tolerances are in the test; nothing to reverse.

**Review**: not yet reviewed.

## 4. Home sides and neutral sites as this repository's schedule records them

**What differs**: nflverse and 538 record 30 games of 1999-2020 with home and away swapped: 12
Super Bowls, 12 international games, 2 relocated games (`2003_08_MIA_SD` in Tempe after the San
Diego fires, `2010_14_NYG_MIN` at Ford Field after the Metrodome roof fell in), all neutral-site
in both sources, and 4 of the 2005 Saints' displaced games. After those swaps, 9 carry a
different neutral-site flag, in two directions. nflverse calls three Saints games at the
Alamodome Saints home games where 538 called them neutral (`2005_04_BUF_NO`, `2005_06_ATL_NO`,
`2005_16_DET_NO`). nflverse calls neutral six games where 538 gave the listed home team its home
field: two of the Bills' Toronto games and the 2014 Detroit snow game (`2008_14_MIA_BUF`,
`2013_13_ATL_BUF`, `2014_12_NYJ_BUF`) and the 49ers' three 2020 Arizona games
(`2020_13_BUF_SF`, `2020_14_WAS_SF`, `2020_17_SEA_SF`). The fourth Saints game,
`2005_02_NYG_NO` at Giants Stadium, is non-neutral in both, with the home field given to the
Saints by nflverse and to the Giants by 538.

**Options**: follow nflverse (this repository's schedule frame); patch the flags to 538's.

**Decision**: follow the schedule frame as it is.

**Reasoning**: the own Elo rates this repository's schedule, which every other feature reads.
Where the flags differ, nflverse is the more accurate for the Bills' and 49ers' games (a team
playing in Toronto or Arizona is not at home), arguable for the Alamodome games (the Saints'
temporary home) and the less accurate for the Saints' "home" game at Giants Stadium. A swap at a
neutral site changes nothing in the framework, since the probability and the update are
symmetric there.

**Data**: entry 3; the game lists are in `ELO_EVAL.md`.

**Feature-value impact**: on the six games nflverse calls neutral, the own Elo gives the listed
home team no home field where 538 gave it 65 points; on the three Alamodome games it gives the
Saints 65 points where 538 gave none; at Giants Stadium it gives the Saints the 65 points 538
gave the Giants. The probability gap is up to `0.093`, and the ratings these games move differ
by up to 16 points for a few weeks afterwards.

**How to reverse it**: patch `neutral` on those games in the schedule frame before calling
`build_own_elo_features`.

**Review**: not yet reviewed.

## 5. Game order and the week of league-level values

**What differs**: 538's code rates games in its file's order; the own Elo rates this repository's
schedule frame, and two inputs are league-wide (the rolling home field and the quarterback
defense baseline).

**Options**: rate by date; rate by season, week, date and `game_id`; compute league-wide values
per game or per week.

**Decision**: games are rated in season, week, date and `game_id` order; every league-wide value
for a game uses only games of earlier weeks, never another game of the same week, even one played
days before it.

**Reasoning**: a team plays once per week, so the order inside a week never changes its ratings,
and picks are made once, before the week's first game (`.agents/modeling_spec.md`): a Sunday
game's league-wide inputs must not include that Thursday's result.

**Data**: the order tests in `tests/test_elo_ratings.py`; the rolling-window tests in
`tests/test_elo_game_context.py`.

**Feature-value impact**: none measurable; it fixes what the values may depend on.

**How to reverse it**: the sort keys in `elo_ratings._ordered`, the window bounds in
`elo_game_context.rolling_home_field`.

**Review**: not yet reviewed.

## 6. Rest and bye

**What differs**: 538 gives 25 points to "a team coming off of a bye week (including when
top-seeded teams don't play during the opening week of the playoffs)" without defining the bye.
`../nfeloqb`'s version counts any rest of 10 days or more, which includes a Thursday-to-Sunday
gap with no bye week.

**Options**: a bye is a week without a game (the team's previous game of the season was two or
more schedule weeks earlier); or a rest of at least 10 or 13 days (nflverse `home_rest`,
`away_rest`).

**Decision**: the week-based definition, `elo_game_context.bye_weeks`. Not kept: the
recommended settings leave rest off (entry 14).

**Reasoning**: rest days cannot separate a bye from a mini-bye (a Monday game followed by a
Thursday game two weeks later is 10 days after a bye; a Thursday game followed by a Sunday one
is 10 days without one); week numbers can. 538's own live numbers fit the week-based term.

**Data**: 538's game-level terms (`ELO_EVAL.md`, "538's game-level terms"): with 55 points at
home, 4 per 1,000 extra visitor miles and 25 per bye, the 2021 live season is reproduced within
0.5 Elo points for `0.996` of 281 non-neutral games (median gap `0.004`), and 2019 for `0.858` of
261 (whose least-squares bye term is 16.89 points, so 538 left the bye out of some live 2019
games). In the forward selection, rest
added to the framework moves Brier by `-0.00019`, interval `[-0.00047, +0.00009]`: not kept.

**Feature-value impact**: none in the recommended settings; with `rest=True`, 25 points to the
side off a bye.

**How to reverse it**: `EloSettings(rest=True)`.

**Review**: not yet reviewed.

## 7. Travel

**What differs**: 538 gives "4 points of Elo for every 1,000 miles traveled" on top of a base of
55 (2019-2021), and none at neutral sites except the Super Bowl. It does not say from where to
where it measures.

**Options**: the base stays 65 with travel added (the home edge would grow by the average trip);
the base drops to 538's 55. Distances between team cities or between stadiums. Travel at every
neutral site or only at the Super Bowl.

**Decision**: with travel on, home field is 55 plus `0.004 * (away_miles - home_miles)`, where a
team's miles run from its home stadium (the venue of most of its non-neutral home games that
season) to the game's stadium, by the great-circle formula over
`constants.STADIUM_COORDINATES` (approximate coordinates for all 67 venues on the 1999-2026
schedules, written for this table). At a neutral site the term is 0 except in the Super Bowl. A
venue without coordinates gets no term and a warning.

**Reasoning**: this is 538's 2019 split of the home edge, and its live numbers confirm it to a
few thousandths of a point with these distances, so the coordinates are accurate enough. The
`home_miles` term covers a home team away from its own stadium. nflverse records the 2025
international games at the listed home team's usual stadium (São Paulo as `LAX01`, Dublin as
`PIT00`, London as `CLE00`, `NYC01` and `JAX00`, Berlin as `IND00`, Madrid as `MIA00`), with the
neutral flag set; that is harmless, because a neutral game other than the Super Bowl gets no
travel term and neutral games never count toward a team's home stadium.

**Data**: entry 6's fit: within 0.5 points for `0.996` of 2021's non-neutral games and `0.858`
of 2019's, median gap `0.004` and `0.004`; neutral games within 0.5 points with no travel term:
4 of 5 (2019) and 3 of 3 (2021). Forward selection: travel on the framework moves Brier by
`-0.00040`, interval `[-0.00064, -0.00016]`, log loss `-0.00087`, interval `[-0.00141,
-0.00033]`: kept.

**Feature-value impact**: about 4 points per 1,000 miles of the visitor's trip, and a home edge
of 55 instead of 65.

**How to reverse it**: `EloSettings(travel=False)`.

**Review**: not yet reviewed.

## 8. Playoff boost

**What differs**: 538 multiplies "EloDiff by 1.2 before computing the expected win probabilities
and point spreads for playoff games" but does not say which difference its margin multiplier
damps afterwards.

**Options**: damp with the boosted difference or with the unboosted one.

**Decision**: the whole difference (ratings, quarterback adjustments, home field) times 1.2,
and the margin multiplier damps that same boosted difference, one difference per game. Not kept.

**Reasoning**: one difference keeps the update consistent with the probability it corrects. The
boost touches the window only through the ratings it carries into the next season.

**Data**: forward selection: Brier `+0.00001`, interval `[-0.00002, +0.00004]`: not kept. On the
221 playoff games of 2007-2025 (a diagnostic), the anchored recommendation scores Brier `0.23158`
without the boost and `0.23328` with it (`ELO_EVAL.md`, "Playoff games").

**Feature-value impact**: none in the recommended settings; with the boost, playoff
probabilities move away from 0.5.

**How to reverse it**: `EloSettings(playoff_boost=True)`.

**Review**: not yet reviewed.

## 9. Rolling home field

**What differs**: 538's 2022 version uses "a rolling 10-year average of home-field advantage"
that "changes during the season", and 33 points instead of 55 for games without fans (2020). It
publishes neither the averaging window nor how travel enters.

**Options**: one value per season (the previous ten seasons, as `../nfeloqb` does) or per week;
all games or regular-season non-neutral ones; with or without the no-fans rule.

**Decision**: per week, 25 Elo points per point of mean home margin over completed non-neutral
regular-season games from the same week ten seasons earlier up to the week before, less the mean
travel term of those games when travel is on (so that base plus the average trip still equals the
observed margin). The frame's first season uses the fixed value. No no-fans rule.

**Reasoning**: a weekly window follows 538's "changes during the season" without reading the
predicted week; playoff home teams are the better teams, so their margins would inflate the
estimate; neutral sites have no home team. A no-fans rule would be a hand-set constant for one
season, which the rolling window partly absorbs.

**Data**: the rolling value at week 1 falls from about 61 points (2007-2015) to `52.60` (2020),
`48.01` (2021) and `39.18` (2025) (`ELO_EVAL.md`, "Rolling home field at week 1"). Forward
selection, after travel: Brier `-0.00030`, interval `[-0.00059, -0.0000058]`, log loss
`-0.00057`, interval `[-0.00123, +0.00009]`: kept, by a margin of six millionths at the
interval's upper end. Alone on the framework it is larger (`-0.00072`, `[-0.00110, -0.00035]`),
because the fixed 55 or 65 points overstate home field since 2020.

**Feature-value impact**: the home edge follows the decline in home margins, about 39 points in
2025 instead of 55.

**How to reverse it**: `EloSettings(rolling_home_field=False)`.

**Review**: not yet reviewed.

## 10. Quarterback box scores and starters

Superseded in part by the integration (`integration_pbp_elo.md`, section 3): the play-by-play
cache now keeps the rusher and play ids, VALUE's rushing is every rush attempt the quarterback
made as the rusher (designed runs, scrambles and kneel-downs), and a completed game's starter is
the shared play-by-play starter (`qb_starters`: the quarterback on the team's first
first-quarter snap). The entry below records the decision as it was taken.

**What differs**: 538's VALUE reads box scores, rushing included. This repository's
play-by-play cache kept the passer id but not the rusher id when this was decided, and holds
regular-season plays only. The plan asks for the box stats from this repository's play-by-play
or another source already here, and for the source and the reason to be logged.

**Options**: the cached play-by-play (designed quarterback runs missing); nflverse team stats
(team totals, so a running back's carries would count as the quarterback's); `qb_elos.csv`'s
published game values (the file the own Elo is meant to replace); nflverse weekly player stats
(not in this repository yet; the second stream is adding them). For the starter: the passer
with the most dropbacks, or the passer of the team's first dropback.

**Decision**: the cached play-by-play, through `elo_quarterbacks.quarterback_box_scores`:
passing lines from `qb_stats.aggregate_qb_game_stats` (two-point tries, kneels and spikes left
out of attempts), plus scrambles and kneels as rushes credited to the primary passer. A completed
game's starter is the passer of the team's first dropback with a passer id, in the cache's play
order: the game's actual starter, known only at kickoff, not when picks are made. An unplayed
game's starter is the one `projected_starters` names, else the team's last starter. This is the
convention of the `qb_elos.csv` columns, whose completed games carry the actual starter, and it
stays (entry 15 measures what knowing the starter before kickoff would cost).

**Reasoning**: it is the only per-quarterback source in the repository, and the
`value_qb_adjustments` seam takes any per-game box scores, so weekly player stats can replace it
when they land. The first passer is the player who took the first snap, which is what 538 and
`qb_elos.csv` call the starter; the most-dropbacks passer is wrong when a starter leaves early.
The starter's identity is the one input a completed game takes from itself; every rating and
value the adjustment reads comes from earlier games.

**Data**: the first passer agrees with `qb_elos.csv`'s starter (names mapped to GSIS ids through
`meta_data.csv`) on `0.9981` of 13,921 team-games of 1999-2025 and `0.9979` of 9,879 of
2007-2025; the most-dropbacks passer on `0.9666` and `0.9699` (`ELO_EVAL.md`, "Starters"). The
resulting adjustments are compared with 538's in entry 11.

**Feature-value impact**: a mobile quarterback's VALUE omits his designed runs, so it is lower
than 538's; the adjustment compares him with his own team's rating, which omits the same runs.

**How to reverse it**: pass other box scores (same columns, `elo_quarterbacks.BOX_SCORE_SCHEMA`)
to `value_qb_adjustments`, or other per-game adjustments to `EloInputs.qb_adjustments`.

**Review**: not yet reviewed.

## 11. The quarterback model

**What differs**: 538 states the rolling rates (a quarterback "every 10 games", a team "every 20
games"), the defense adjustment (subtract what an opponent "usually gives up" above the league),
the 3.3 Elo points per point of VALUE, the veteran reversion (a quarter toward the average start
for 10-100 career starts) and the team preseason blend (a third the old rating, two thirds the
projected starter's). It does not state the defense's rolling rate or offseason step, which
value a team's rating rolls in, or how it rates a rookie beyond "draft position (0 undrafted,
113 Elo points for the first pick)".

**Options**: for the team, the starter's game value or all its quarterbacks' value; for the
defense, the team rate (20 games) or the quarterback rate (10); for a first start, 0 for
everyone, a draft-based rating (no draft data here), or the mean of earlier first starts.

**Decision**: the team's rating rolls in the starter's defense-adjusted game value at rate 0.05,
the quarterback's at 0.1; each defense keeps a rolling rating of the quarterback VALUE it allows
above the week's league mean, at rate 0.05, reverting a third toward 0 each offseason; the
veteran reversion targets the previous season's mean adjusted start; the team preseason rating
follows 538's stated blend; a quarterback's first start in the data is rated at the mean
adjusted value of every first start in earlier seasons (the first season excluded), 0 before
any. Ratings update on regular-season starts only. A completed regular-season game is rated
for its actual starter (entry 10), known only at kickoff; an upcoming or playoff game for the one
`projected_starters` names, else the team's last starter. `value_qb_adjustments(...,
pre_game_starter="last_starter")` rates every game the second way, as known before kickoff,
while the ratings still roll in the actual starter's game (entry 15).

**Reasoning**: 538's published values fix the two rolling rates and show that the team rolls in
the starter's value; the defense rate follows the team one because 538 describes both as team
ratings; the first-start mean is the draft-free estimate closest to 538's intent of rating a
debut below an average starter. 538's published season-start team values do not follow its
stated blend, and no other rule is published, so the stated one is used.

**Data**: 538's 2007-2022 values (`ELO_EVAL.md`, "538's rolling updates"): a quarterback's next
value equals `0.9 * before + 0.1 * game` within 0.01 points for `1.000` of 8,618 starts (rate
0.05: `0.001`); a team's next value equals `0.95 * before + 0.05 * game` with median gap
`0.0007` (`0.551` within 0.01; rate 0.1: median gap `4.4366`); the stated season-start blend
misses the published team value by a median `4.52` points over 480 season starts. Against the
published adjustments (`ELO_EVAL.md`, "Quarterback adjustments"), regular season: correlation
`0.865` with 538's (2007-2022, 8,254 team-games), mean `-2.84` against `-2.59`, standard
deviation `31.04` against `31.20`, mean absolute gap `9.30`; on the 204 team-games 538 adjusted
below -100 points, `-111.2` against `-132.0` (538's draft-based rookie ratings are lower); with
nfeloqb's (2023-2025) correlation `0.899`. Forward selection, after travel and the rolling home
field: Brier `-0.00249`, interval `[-0.00381, -0.00119]`, log loss `-0.00621`, interval
`[-0.00929, -0.00326]`: kept.

**Feature-value impact**: each side's own Elo carries `3.3 * (starter - team)`, about 31 points
standard deviation, and more than 100 points down for a backup.

**How to reverse it**: `EloSettings(quarterback=False)`; the constants are in
`elo_quarterbacks`.

**Review**: not yet reviewed.

## 12. The win-total pull

**What differs**: 538 blends a preseason rating as one third regressed Elo and two thirds a
rating from Vegas win totals (since 1990). The plan asks for nfelosrs's `wt_ratings.csv` (2003
on, plain regression before), cached with the cache-then-degrade pattern, its weight fitted on
earlier seasons only, and never used by a market-free option.

**Options**: 538's two thirds (fitted by 538 on seasons that overlap the evaluation); one weight
fitted once on all seasons (leaks later seasons into earlier ones); a weight per season fitted on
the earlier seasons.

**Decision**: `elo_win_totals.load_win_totals` reads
`https://raw.githubusercontent.com/greerreNFL/nfelosrs/main/wt_ratings.csv` (`wt_rating_elo`,
2003-2026), keeps the last good copy and every copy a run used under the nfelo cache's
`win_totals/` directory, and degrades to no rows. For each season the weight is the one of 0,
0.05, ..., 1 whose fixed-weight run scores the lowest Brier on the completed regular-season games
of the earlier win-total seasons (ties to the smaller weight); 2003, with nothing earlier, gets
none. A setting without the switch refuses the win totals, and `MARKET_FREE_SETTINGS` never has
it.

**Reasoning**: the per-season fit is the only option that both measures the weight on this
repository's games and never reads a season's own results.

**Data**: the fitted weights (`ELO_EVAL.md`, "Win-total weights", for the pull on the market-free
recommendation): `1.00` for 2004-2006 (fitted on one to three seasons), then `0.95` (2007),
`0.90` (2008) and `0.55` to `0.75` from 2009, `0.65` in 2025, close to 538's two thirds. Forward
selection, last: Brier `-0.00096`, interval `[-0.00185, -0.0000074]`, log loss `-0.00229`,
interval `[-0.00435, -0.00017]`: kept, by a margin of seven millionths at the Brier interval's
upper end. The evaluation read a downloaded copy of the file (its SHA-256 is in `ELO_EVAL.md`).

**Feature-value impact**: none in the market-free settings; with `WIN_TOTAL_SETTINGS`, each
preseason rating moves about two thirds of the way to the win-total rating.

**How to reverse it**: use `MARKET_FREE_SETTINGS`, or `EloInputs(win_total_weights=...)` for
fixed weights.

**Review**: not yet reviewed.

## 13. The published columns and their feature group

**What differs**: the plan asks for "pre-game Elo for both teams, their difference and the Elo
win probability, as a new feature group with its own name and ablation switch, distinct from the
existing `qb_elos.csv` columns", and a function from this repository's schedule frame to the
per-game columns, not wired into the ETL.

**Options**: each side's base rating, or its rating with the quarterback adjustment, or both; the
difference home minus away or away minus home.

**Decision**: `build_own_elo_features(schedule, variant=..., inputs=...)` publishes one of two
families and returns it with metadata (the variant, its switches, its columns, the first season,
whether seeds were given, the win-total weights used). The `market_free` variant
(`MARKET_FREE_SETTINGS`) returns `game_id` and `constants.OWN_ELO_COLUMNS`: `away_own_elo` and
`home_own_elo` (the side's pre-game rating plus its quarterback adjustment, 538's "effective" Elo;
home field, travel and the boost are game-level and only in the probability), `own_elo_diff`
(away minus home, the repository's `_diff` convention) and `own_elo_home_win_prob`, the `own_elo`
group of `--disable-feature-groups`. The `win_totals` variant (`WIN_TOTAL_SETTINGS`) returns the
same four under the stat `own_wt_elo` (`constants.OWN_WT_ELO_COLUMNS`), the `own_elo_wt` group,
which is market-informed: a run with no market information disables it with `elo`. A family's
names always come with its preset, so the market-free columns can never carry the win totals.
The stat is `own_wt_elo` rather than `own_elo_wt` because group markers match substrings and
`away_own_elo` is a substring of `away_own_elo_wt`. No name contains another group's markers
(`elo_pre`, `qb_elo_pre`, `qb_value_pre` and the trends), so the three Elo groups are disjoint
and one build can carry all of them.

**Reasoning**: the effective rating is what the probability uses for each side; the base rating
and the adjustment stay available from `compute_elo_ratings` if a later measurement wants them
apart.

**Data**: `tests/test_own_elo_feature_group.py` (each group resolves to exactly its four columns
on the published schema plus both families, each is disjoint from every other group, and
`elo,own_elo_wt` leaves the market-free columns in); `tests/test_elo_ratings.py` (each variant
publishes only its own family, the market-free one refuses win totals, the metadata records the
preset).

**Feature-value impact**: none until the integration stream adds the columns to a build.

**How to reverse it**: `constants.OWN_ELO_TEAM_STATS`, `OWN_WT_ELO_TEAM_STATS` and their game
columns, and `elo_ratings.OWN_ELO_VARIANTS`.

**Review**: not yet reviewed.

## 14. The rule applied

**What differs**: nothing; this entry applies entry 1 as written to the numbers of `ELO_EVAL.md`
(commit `ff14dec`, unchanged at `e0c0663`).

**Options**: none; the rule decides.

**Decision**: seeding every team at 1505; components kept, in order: travel, the rolling home
field, the quarterback adjustment and the win-total pull; not kept: rest and the playoff boost.
The recommendations are `elo_ratings.MARKET_FREE_SETTINGS` (travel, rolling home field,
quarterback) and `elo_ratings.WIN_TOTAL_SETTINGS` (the same plus the win-total pull), published
by `build_own_elo_features` as its `market_free` (default) and `win_totals` variants.

**Reasoning**: entry 1. Two kept components pass by a hair: the rolling home field's Brier
interval ends at `-0.0000058` and the win-total pull's at `-0.0000074`. The rule counts both as
below zero, and the bootstrap seed was fixed before the run; a different seed could move either
end across zero.

**Data**: the forward selection, each candidate minus the components kept so far, on 4,943 games:

| component | Brier diff | 95% interval | log-loss diff | 95% interval | kept |
| --- | --- | --- | --- | --- | --- |
| rest and bye | -0.00019 | [-0.00047, +0.00009] | -0.00051 | [-0.00113, +0.00012] | no |
| travel | -0.00040 | [-0.00064, -0.00016] | -0.00087 | [-0.00141, -0.00033] | yes |
| playoff boost | +0.00001 | [-0.00002, +0.00004] | +0.00000 | [-0.00006, +0.00007] | no |
| rolling home field | -0.00030 | [-0.00059, -0.00001] | -0.00057 | [-0.00123, +0.00009] | yes |
| quarterback adjustment | -0.00249 | [-0.00381, -0.00119] | -0.00621 | [-0.00929, -0.00326] | yes |
| win-total pull | -0.00096 | [-0.00185, -0.00001] | -0.00229 | [-0.00435, -0.00017] | yes |

Beside the rule, on the same games (diagnostics; they decide nothing here): the market-free
recommendation scores Brier `0.21712` and log loss `0.62268`, the win-total one `0.21616` and
`0.62038`; `qb_elos.csv`'s classic `elo_prob1` scores `0.22001` and `0.62971`, its quarterback
model `qbelo_prob1` `0.21620` and `0.62063`. Market-free minus classic: Brier `-0.00289`,
interval `[-0.00429, -0.00152]`; minus the quarterback model: `+0.00091`, interval `[-0.00064,
+0.00246]`. Win-total minus the quarterback model: `-0.00004`, interval `[-0.00147, +0.00136]`.
By week window (no intervals computed) the market-free recommendation scores `0.22058`,
`0.22719` and `0.21617` in weeks 1, 2 and 3-18, the quarterback model `0.21904`, `0.22477`
and `0.21541`.

**Feature-value impact**: the presets define the columns the integration stream would publish.

**How to reverse it**: rerun `.agents/sos_port/elo_eval.py` under another rule, or pass other
`EloSettings`.

**Review**: not yet reviewed.

## 15. The starter is known at kickoff

**What differs**: a completed game's quarterback adjustment is computed for that game's actual
starter, the passer on its first dropback, which is known only at kickoff, not when picks are
made before the week's first game. The rest of the own Elo reads only earlier games. The
independent review flagged that the docs overclaimed "every value from earlier games".

**Options**: keep the actual starter, as the `qb_elos.csv` columns do for completed games; or
rate every game for the starter known before kickoff (the projected one, else the team's last).

**Decision**: keep the actual starter (decided by the delegating session); state the convention
exactly in the code, the README and entries 10 and 11; report the before-kickoff alternative as a
measured diagnostic that does not feed the rule.

**Reasoning**: it is the convention of the columns the own Elo would replace, so a comparison
between the two stays like for like; a pick-time starter needs a projected-starter source, which
the integration stream can pass through `projected_starters`.

**Data**: `ELO_EVAL.md`, "Starters known before kickoff", every game rated for the team's previous
starter while the ratings still roll in the actual starter's game (`pre_game_starter=
"last_starter"`), previous minus actual on the 4,943 window games:

| preset | actual Brier | previous Brier | Brier diff | 95% interval | log-loss diff | 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| market-free | 0.21712 | 0.21813 | +0.00101 | [-0.00027, +0.00232] | +0.00331 | [+0.00035, +0.00631] |
| win-total | 0.21616 | 0.21744 | +0.00128 | [-0.00002, +0.00261] | +0.00365 | [+0.00077, +0.00665] |

The quarterback step (travel and the rolling home field, with the adjustment against without
it) gains Brier `-0.00249`, interval `[-0.00381, -0.00119]`, with actual starters and
`-0.00148`, interval `[-0.00294, +0.00002]`, with previous starters; its log loss moves
`-0.00290`, interval `[-0.00632, +0.00049]`, with previous starters. These reproduce the
reviewer's measurement.

**Feature-value impact**: none today; with a projected-starter source the integration stream
can rate upcoming games for the expected starter, and historical games would keep the actual one.

**How to reverse it**: `value_qb_adjustments(..., pre_game_starter="last_starter")` rates every
game for the starter known before kickoff.

**Review**: not yet reviewed (the numbers reproduce the independent review's).

## 16. Where the remaining quarterback alternative lives

**What differs**: the plan names two quarterback adjustments, 538's VALUE first and the ported
`nfl-sos-ratings` quarterback rating as the alternative. Only VALUE is built here.

**Options**: port the rating now (it belongs to the ratings-port stream); measure it later
through the same seam.

**Decision**: the ported rating is measured with `.agents/sos_port/elo_eval.py` after the ratings
port and the play-by-play column extension merge into the integration branch; the delegating
session schedules it. Its entry point is `EloInputs.qb_adjustments`, which takes per-game
adjustments in Elo points (`game_id`, `home_qb_adj`, `away_qb_adj`) from any source, so the
rating replaces VALUE without touching the engine.

**Reasoning**: the rating is another stream's work, and the seam already exists.

**Data**: none yet.

**Feature-value impact**: none until it is measured and chosen.

**How to reverse it**: nothing to reverse.

**Review**: not yet reviewed.
