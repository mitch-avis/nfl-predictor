# Pick-time checks for four feature ideas

Research notes for the roadmap's feature ideas (step 8 in `.agents/TODO.md`): the referee crew,
game-day weather, attendance and team health. Written 2026-10-16. No code, no runs and no
writes under `data/` or `models/`; the user approved notes only. Each idea gets its source, a
pick-time check, the leakage traps, a possible feature form and a verdict. The notes give the
groundwork for a later hypothesis but do not state one.

How to read the evidence:

- **Verified** facts name the command or the URL. The scripts are outside the repo in
  `~/scratch/step8_research/scripts/` and run as
  `/home/mitch/workspace/nfl-predictor/.venv/bin/python -I <script> <args>`, with nflreadpy
  0.1.5 from the main checkout's virtualenv. Downloads sit in
  `~/scratch/step8_research/downloads/`, fetched 2026-10-16 at about 08:52 UTC. The repo's
  cache `data/cache/nflreadpy/` and `data/all_data_ml.csv` were read, never written.
- **Unverified** marks anything not checked against a source. A search that failed is reported
  as failed.
- The 2026 calendar, from the downloaded schedule (`week_dates.py`): week 6 is in progress. Its
  Thursday game was played 2026-10-15, and the next week to predict is week 7, which starts
  2026-10-22.

## The pick-time line

- The pool rule: "Picks are submitted before the first game of the week (single-shot; no in-week
  updates in backtests)" (`.agents/modeling_spec.md`, pool section). The roadmap frames the
  check as "what is known before Thursday for an upcoming game". The handoff
  (`.agents/next_agent_session_prompt.md`) has the user's weekly run due before Thursday, so it
  runs on a Tuesday or a Wednesday.
- These notes take Tuesday or Wednesday of the predicted week as the line. Anything published
  later fails, even when it comes out before Thursday night's kickoff. Whether the line is the
  weekly run or the first kickoff matters for team health. That is the fifth question at the end.

## What the leakage audit can and cannot catch

`nfl-predictor leakage-audit` (`src/nfl_predictor/ml/leakage_audit.py`, `run_leakage_audit`)
runs four checks on completed games only. Verified by reading the code:

- a target column (margin, total, scores) among the features fails the audit;
- a feature named with `winner`, `result`, `final`, `post` or `boxscore` gets a warning;
- an absolute correlation with margin or total of at least `0.995` is flagged, and at least
  `0.999999` fails;
- a feature equal to a score column in at least 98% of rows is flagged, and in 99.9999% fails.

It never compares what a training row knows with what a prediction row knows. Every trap below
has the same shape: training rows (completed games) carry a value recorded at or after kickoff,
while prediction rows carry a null, a stale value or a different category. None of these values
is close to the score, so **the audit passes all of them**. The spec's audit section also says
the audit "validates season-to-date features exclude the current game row". No such check is
visible in `leakage_audit.py`. That is a gap between the spec and the code, and other modules
were not searched.

What would catch these traps is a parity check between training rows and prediction rows: for
each column, compare the null share and the set of categories in the last completed week with
the upcoming week. The incidental finding near the end shows the existing `stadium_type` column
failing exactly that kind of check today.

## Earlier attempts in this repo

- The archive's weather milestone (Milestone 32, `.agents/ARCHIVE.md`): "weather fields were
  later removed after confirming they update post-kickoff".
- The archive's referee milestone (Milestone 34): "Referee features were removed after confirming
  assignments update post-kickoff".
- `CHANGELOG.md` `0.2.0` (2026-06-12), Removed: "Remove weather and referee features after
  confirming the data is not pre-kickoff safe".
- The repo already knows the pattern behind the health trap. Completed games carry the actual
  starting quarterback, known only at kickoff, while upcoming games carry the projected one
  (`.agents/step6_plan.md`, the quarterback section).

## The referee crew

### Source

- **Schedule referee.** `nflreadpy.load_schedules()`, column `referee` (head referee's name).
  Verified (`download.py`, `analyze2.py`): 7,548 rows and 46 columns, 1999-2026. The column is
  filled for every played game 1999-2025 except one in 2021 (284 of 285). The repo's schedule
  cache does not hold it: `constants.NFLREADPY_SCHEDULE_COLUMNS` lists no `referee`, `temp` or
  `wind`. Commit `669e919` refetches schedule caches when that list grows, and its test uses
  `referee` as the example.
- **Full crews.** `nflreadpy.load_officials()`. Its docstring says 2015 onward. Columns: `game_id`
  (the old 10-digit id, which joins to the schedule's `old_game_id`), `game_key`,
  `official_name`, `position`, `jersey_number`, `official_id`, `season`, `season_type` and `week`.
  Verified (`analyze2.py`):
  - full seasons 2015-2025, mostly seven officials per game;
  - 3 played 2021 games have no rows;
  - 17 referees a season, each with 14-17 regular-season games in 2024 and 2025;
  - in 2025 each referee kept a stable crew: all six other officials matched his usual crew for
    16 referees, and five did for one.
- **Current season.** The downloaded schedule has `referee` for weeks 1-4 (16 of 16 each) and for
  1 of 15 played week-5 games. Upcoming games have none. The officials file has 2026 week 1 only
  (16 games).
- **How the officials file is built.** Verified with `gh api` on `nflverse/nflverse-rosters`: the
  builder is `exec/update_officials.R`, which reads the league's Next Gen Stats feed for every
  game not yet in the file. The release asset was last updated 2026-09-02 11:48 UTC. The
  workflow `update_officials.yaml` failed every day from 2026-09-03 to 2026-09-14 (`gh run
  list`). Commit `ecc6ef18` on 2026-09-15 ("Disable schedule because the workflow keeps
  failing") turned off its schedule, so nflverse is not updating the officials file in-season.
- **How the schedule referee is filled.** The schedule release is rebuilt whenever
  `nflverse/nfldata`'s games file changes (`release_games.yml`, read with `gh api`). Ten
  snapshots of `nfldata/data/games.csv`, taken between 2026-09-09 and 2026-10-14
  (`gh api repos/nflverse/nfldata/commits?path=data/games.csv&until=...` and a raw download;
  `games_history2.py`), show the pattern:
  - weeks 1-4: the referee appears in the same update as the result;
  - week 5: the result came first, and on 2026-10-14 only 1 of 15 games had a referee;
  - no snapshot ever shows a referee on an unplayed game.

  The nflreadr dictionary (`data-raw/dictionary_pbp.csv`) credits the schedule's stadium and
  weather columns to Pro-Football-Reference. That the referee comes from there too is
  unverified.

### Pick-time check

- **The league tells each official his own assignment.** Football Zebras, "Referee assignments,
  preseason week 2", 2023-08-17 (<https://footballzebras.com/?p=39395>): "The league office does
  not distribute the full list of assignments to officials, other than their own assignments."
- **Unofficial Tuesday posts.** "NFL referee assignments drop on Tuesdays during the season"
  and "There's no press release" (SportsBettingDime referee tracker,
  <https://www.sportsbettingdime.com/?p=804427>). Football Zebras posted:
  - 2025 week 7 on Tuesday 2025-10-14, two days before the Thursday game
    (<https://www.footballzebras.com/2025/10/week-7-referee-assignments-2025>);
  - 2025 week 16 on Tuesday 2025-12-16, with the first game Thursday 2025-12-18
    (<https://www.footballzebras.com/2025/12/week-16-referee-assignments-2025>).

  Both carry the note "there may be substitutions from the regular crew personnel each week".
- **Changes after the first post.** The 2018 week 5 post was updated: "The league made changes to
  the assignments, which it confirmed Tuesday afternoon"
  (<https://footballzebras.com/2018/10/week-5-referee-assignments-2018>).
- **Crews are set in summer.** The 2025 crews for all 17 referees were published 2025-07-09
  (<https://www.footballzebras.com/2025/07/officiating-crews-for-the-2025-season/>). Once the
  referee is known, the rest of the crew is mostly known too.
- **The Next Gen Stats feed had crews early once.** The officials file written 2026-09-02 already
  held the 2026 week-1 crews, seven days before the 2026-09-09 opener. Its week-1 referees match
  the schedule's after-the-game referee in 16 of 16 games, counting one spelling variant
  (`officials_week1.py`). Whether the feed carries crews that early every week is unverified.
- **Conclusion.** The head referee is public on Tuesday, so the timing passes. But no allowed
  source has it at pick time. The schedule fills it after the game, the officials feed is off,
  and scraping Football Zebras or a betting site would be a new scraper, which AGENTS.md rules
  out ("No unrelated features, scrapers or external services").

### Leakage traps

- **The schedule referee is recorded after the game.** It usually equals the Tuesday assignment,
  but a late substitution would put the actual referee in a training row that a Tuesday
  prediction could not have had. From 2015 to 2025, 3,041 games join between the two nflverse
  sources. Beyond three spelling variants (Ron and Ronald Torbert in 151 games, Perry and Parry,
  Hall and Hill), the referee name differs in 27 games (`officials_week1.py`). Whether those are
  late swaps or errors in either source is unverified. The audit would not catch this: a name is
  not close to the score.
- **Train/serve skew.** Prediction rows would hold a null referee today, while every training
  row has one. With `OneHotEncoder(handle_unknown="ignore")` (`ml/feature_spec.py`), a null
  becomes the most frequent referee (most-frequent imputer), and a new referee becomes an
  all-zero row. The audit does not compare training rows with prediction rows, so it would not
  catch this.
- **Same-game crew statistics.** Penalties or totals from the game being predicted must never
  enter that game's crew features. The audit would catch this only if a statistic nearly
  reproduced the score, which a penalty count does not. The guard is the existing rule that
  values for week N come from games strictly before week N.

### Possible feature form

- If an allowed pick-time source returns, add a new `referee` feature group: one marker entry in
  `constants.FEATURE_GROUP_COLUMN_MARKERS`, so it can be left out with
  `--disable-feature-groups`. Do not one-hot the name: 17 referees and yearly turnover make the
  one-hot sparse and unstable. Use crew tendencies from games strictly before the predicted week,
  shrunk toward the league mean, built like the coach features (`COACH_FEATURE_COLUMNS`,
  `*_prior` columns):
  - the crew's penalties per game;
  - its average total minus the closing total;
  - home win rate and home penalty share;
  - games of history.
- Expected effect on win probability: small (unverified). A referee is not a home or away side,
  so the crew features enter the margin only through home-side tendencies. The total, which is
  display-only, is where officiating shows up (`.agents/ARCHIVE.md`, the total-head discussion
  names "officiating" with weather and pace).
- **A measured arm:** two dataset builds behind an ETL flag (back up `data/*.csv` first). The
  reference runs on the build without the group. The candidate runs on the build with it,
  checked to leave every other column unchanged. Two seeds (`42` and `7`), averaged per game. The
  command shape follows `.agents/benchmarks.md` ("`qb_elos.csv` measurement"):
  `nfl-predictor backtest --calibration auto --eval-seasons 2007..2025 --wf-start-week 1
  --random-seed {42,7} --market-anchor --market-transform`. The data allows 2015 and later only
  (full crews) or 1999 and later (head referee only), so the governing window has to be written
  into `HYPOTHESIS.md` before the builds.

### Verdict

**Fails** for now. The crew is public on Tuesday, but no allowed source has it at pick time.
Recommendation: park it, and revisit only if nflverse turns the officials feed back on and it
fills upcoming weeks. How sure: high on the timing and the source state, low on the effect size.

## Game-day weather

### Source

- `nflreadpy.load_schedules()`, columns `temp` (°F) and `wind` (mph), plus `roof` and `surface`.
  The nflreadr dictionary (`data-raw/dictionary_pbp.csv`, downloaded to
  `~/scratch/step8_research/dictionaries/`) says they come from Pro-Football-Reference, and that
  `temp` and `wind` exist only when `roof` is `outdoors` or `open`.
- **Coverage on outdoor or open-roof regular-season games** (verified, `analyze2.py`): 95-100%
  in most seasons since 2015, but 2021 at 94.5%, 2022 at 107 of 198 (54%) and 2023 at 158 of 199
  (79%). From 1999 to 2025 there are 5,205 outdoor or open games with both values: wind 15 mph or
  more in 684 of them, 32°F or colder in 384.
- **Current season:** weeks 1-4 filled (10, 10, 13 and 12 games with `temp`), week 5 none,
  upcoming none (`week_dates.py`).
- **The repo loads `roof` and `surface` only** (`constants.NFLREADPY_SCHEDULE_COLUMNS`, renamed
  `stadium_roof` and `stadium_surface`). The local play-by-play cache has no weather columns.
- **Roof values.** `roof` takes `dome`, `outdoors`, `closed` or `open`. A retractable roof is
  recorded per game as `open` or `closed` after the game and is empty beforehand. For example,
  Baltimore at Atlanta in week 5 (`2026_05_BAL_ATL`) was empty in the 2026-10-09 snapshot and
  `open` in the 2026-10-12 one (`games_history.py`).
- **The 2026 international venues as published:**

  | Venue        | Code  | Roof as published |
  | ------------ | ----- | ----------------- |
  | Melbourne    | MEL00 | `dome`            |
  | Rio          | RIO00 | `open`            |
  | London       | LON00 | `outdoors`        |
  | London       | LON02 | `outdoors`        |
  | Paris        | PAR00 | `dome`            |
  | Madrid       | MAD01 | empty             |
  | Munich       | MUN01 | `dome`            |

  `.agents/sos_port/decisions/pbp_columns.md` has the details. The task brief says Paris and
  Munich get an `outdoors` override on a branch landing tonight. That was not checked here.

### Pick-time check

- **Actual weather is known only on game day.** The schedule fills `temp` and `wind` after the
  game (the snapshot pattern under the referee source applies here too). The repo removed weather
  features for this reason (`CHANGELOG.md` `0.2.0`).
- **A forecast is the only pick-time weather.** NOAA NESDIS, "How Reliable Are Weather
  Forecasts?"
  (<https://www.nesdis.noaa.gov/about/k-12-education/weather-forecasting/how-reliable-are-weather-forecasts>):
  "A seven-day forecast can accurately predict the weather about 80 percent of the time and a
  five-day forecast ... approximately 90 percent of the time", and a 10-day forecast "is only
  right about half the time". From a Tuesday run, a Sunday game is five days out.
- **No allowed source has archived forecasts.** nflverse carries none (dictionary search,
  verified). Open-Meteo's Historical Forecast API keeps only "Each run's first few hours", from
  2022. Its Previous Runs API gives fixed 1-7 day lead times "from January 2024 (GFS from March
  2021, JMA from 2018)" (<https://open-meteo.com/en/docs/historical-forecast-api>). Either would
  be an external service, which AGENTS.md rules out, and neither covers a 2007-2025
  walk-forward.
- **Retractable roofs are decided on game day.** "Up to 90 minutes prior to kickoff, the home team
  must notify the referee" (Football Zebras, 2013-05-01,
  <https://footballzebras.com/2013/05/nfl-retractable-roof-policies>). Atlanta "has until 90
  minutes prior to kickoff to inform the referee or highest-ranking league official"
  (Atlanta Journal-Constitution, 2018-09-06,
  <https://www.ajc.com/sports/finished-roof-opens-new-option-for-falcons-home-games/iQVQA7iiItfuFNPUcI218L/>).
  Whether these policies still hold in 2026 is unverified.

### Leakage traps

- **Actual `temp` and `wind` in training rows.** These are game-day values, missing for every
  upcoming game. The audit would not catch them: temperature is not close to the score, and the
  audit never looks at prediction rows. This is the trap the repo already fell into once.
- **The per-game roof state (`open` or `closed`).** It is a game-day decision driven by the
  weather, and it already enters the model through `stadium_type`. See the incidental finding
  below. The audit does not catch it.
- **Nulls that depend on when the data was recorded.** `temp` is missing in 46% of 2022's
  outdoor games and 21% of 2023's. Imputing the median would give those seasons a different mix
  from the rest. Not leakage, but a reason to build any weather proxy from a source with even
  coverage.
- **The descriptive check below uses after-the-fact values.** Outdoor and open games 2006-2025,
  actual wind against the stored total line (`analyze2.py`), average actual total minus the line:

  | Wind      | Games | Total minus line |
  | --------- | ----- | ---------------- |
  | 0-9 mph   | 2,415 | `+1.10`          |
  | 10-14 mph | 870   | `-0.45`          |
  | 15-19 mph | 348   | `-1.81`          |
  | 20+ mph   | 118   | `-0.65`          |

  This uses the actual wind, which is not known at pick time, and it is not a measurement. Treat
  it as an upper bound on what a pick-time proxy could carry for totals.

### Possible feature form

- **Venue climatology as the pick-time proxy.** For each home venue and calendar week (or
  month), take the average temperature and wind from that venue's outdoor and open games in
  earlier seasons only. For retractable venues, add the share of earlier-season games played
  open. Domes get no weather value: either null with an indicator, or a fixed indoor value. That
  is a design choice for the hypothesis. All of it is known before the season starts.
- **Where it goes.** The stadium columns (`STADIUM_FEATURE_COLUMNS`) are not an ablatable group
  today, so the climatology needs its own `climate` entry in
  `constants.FEATURE_GROUP_COLUMN_MARKERS`.
- **Expected effect.** Mostly on the total, which is display-only (AGENTS.md: score outputs are
  secondary). Weather affects both teams, so any effect on win probability would come through
  interactions, such as a dome team playing outdoors in the cold. That is unverified. The closing
  total already prices the forecast, so the arm with game lines is the harder test.
- **A measured arm:** two dataset builds (the flag adds the group, and every other column stays
  unchanged), seeds `42` and `7`, 2007-2025 from week 1, with game lines and without them. The
  2022 and 2023 coverage holes argue for a climatology window that skips missing values rather
  than imputing them.

### Verdict

**Passes only with a pick-time proxy**: venue climatology from earlier seasons' nflverse
observations. Actual game-day weather fails, and a forecast needs an external service.
Recommendation: second in line and low priority, because the main effect falls on the
display-only total. How sure: high on the timing, medium that the proxy carries any margin
signal.

## Attendance

### Source

- **No nflverse source found.** Verified by search:
  - `grep -ci attendance` returned 0 on the nflreadr dictionary CSVs for schedules,
    play-by-play and injuries (`~/scratch/step8_research/dictionaries/`);
  - a search of the installed nflreadpy 0.1.5 package source found no string "attendance".

  The failed search is reported as such. No other nflverse release was checked.
- Box-score attendance is published by Pro-Football-Reference and ESPN (unverified; not fetched
  for this note). Both would be external sources. AGENTS.md: "No unrelated features, scrapers or
  external services".

### Pick-time check

- **Attendance is a count of who showed up,** so it is known only after the game. A pick-time
  stand-in would be stadium capacity, a fixed property of the venue that adds nothing beyond the
  venue itself, or the 2020 crowd limits.
- **The 2020 crowd limits are the one case with a real pick-time signal.** They were announced
  before games (unverified; no source was fetched). They would need a hand-built table, which no
  nflverse source provides.

### Leakage traps

- **Actual attendance in training rows.** Recorded after the game; a crowd announced as smaller
  than capacity can reflect a team's record or the weather that day. Prediction rows would have
  none. The audit would not catch it: a crowd count is not close to the score.

### Possible feature form

None that passes from an allowed source. The only variant known at pick time is a 2020 no-fans
or limited-fans indicator from a hand-built table, and a measured arm would rest on one season of
signal.

### Verdict

**Fails.** It is recorded after the game, and nflverse has no source. Recommendation: drop it
from the roadmap (retiring an item is the user's call). How sure: high.

## Team health

### Source

- **Injury reports.** `nflreadpy.load_injuries(seasons)`, 2009 onward. Columns: `season`,
  `game_type`, `team`, `week`, `gsis_id`, `position`, the name fields,
  `report_primary_injury`, `report_secondary_injury`, `report_status`,
  `practice_primary_injury`, `practice_secondary_injury`, `practice_status`, `date_modified` and
  `season_type`. Verified (`download.py`, `analyze2.py`) on the files fetched 2026-10-16:
  - **One row per player and week.** 92,515 of 92,517 player-weeks have exactly one row, so the
    history keeps only the final state of each week's report, not the Wednesday, Thursday and
    Friday versions.
  - **`report_status` values.** `Out` 16,421, `Doubtful` 3,405, `Questionable` 23,884,
    `Probable` 17,400 (2015 and earlier, before the league dropped it), `Note` 6, and null 31,403
    (practice-only rows).
  - **`date_modified`.** Populated 2010-2024. Mostly null in 2009, and null in every row of 2025
    and 2026.
- **How the injury file is built.** Verified with `gh api` and `gh run list` on
  `nflverse/nflverse-rosters`: the builder `exec/update-injuries.R` calls
  `nflapi::nflapi_injuries` and overwrites the current season. Its workflow ran every day in
  October 2026, all successful: a dispatch at 06:00 UTC and a scheduled run between about 12:00
  and 17:00 UTC. The `injuries_2026` asset was updated 2026-10-16 06:01 UTC.
- **The current season shows the overwrite.** In the 2026 file fetched Friday 2026-10-16, the
  week-6 Sunday teams have practice statuses but no game status. The Thursday-game teams
  (Seattle and Denver) have `Out` or `Questionable`. Each week's row therefore changes in place
  until the final report, and earlier states of past weeks cannot be recovered.
  `nflverse/nflverse-data-archives` is "Updated quarterly with the RDS files" (README, read with
  `gh api`), so it holds no daily snapshots either.
- **Snap counts (already cached).** `data/cache/nflreadpy/snap_counts_{season}_reg.parquet`
  (verified, `cache_peek.py`) holds `game_id`, `season`, `game_type`, `week`, `player`,
  `pfr_player_id`, `position`, `team`, `opponent`, `offense_snaps` and `offense_pct`. That is
  offense only: the cache keeps no defense columns, and whether nflreadpy offers them was not
  checked. It covers 2013-2025, plus 2026 weeks 1-5; the 2012 file is empty.
- **Depth charts.** `nflreadpy.load_depth_charts()` (verified, `depth_charts.py`): 2025 and
  later are daily snapshots with a `dt` timestamp (2025 from 2025-08-03 to 2026-03-14, 2026 from
  2026-03-22 to 2026-10-16). 2024 and earlier are weekly with no timestamp. They give a dated
  pick-time view of the roster, but only from 2025.
- **Weekly rosters.** The cached `rosters_weekly_*` files have no status column (`cache_peek.py`),
  so they carry no game-day inactive list into the build today.

### Pick-time check

- **For a Sunday game, practice reports are Wednesday, Thursday and Friday, with the game status
  on Friday.** This is from the 2015 league policy: "for a Sunday game, he must be listed on the
  practice report on Wednesday, Thursday, and Friday, and the game status report on Friday". The
  quote comes from a search excerpt of
  <https://operations.nfl.com/media/1818/2015-injury-report-policy.pdf>; the PDF itself returned
  404, as did the 2016 policy page on operations.nfl.com.
- **The game status comes two days before kickoff.** ESPN, 2016-08-21: "Two days before kickoff,
  teams must list injured players as 'questionable,' 'doubtful' or 'out'", and the league
  dropped "probable"
  (<https://africa.espn.com/nfl/story/_/id/17361111/nfl-streamlines-injured-players-game-status-removes-probable-designation>).
- **Thursday teams report Monday to Wednesday.** "Teams report Wednesday, Thursday and Friday
  (Monday through Wednesday for the Thursday night teams)", with game statuses "with the final
  report" (Legion Report, undated, <https://legionreport.com/nfl-week-3-injury-report/>). A web
  search summary (TSN and AOL, 2026) said Seattle's Monday and Tuesday reports before its
  Thursday game were estimates. Those pages were not fetched.
- **Clock times are unverified.** The excerpt mentions a 4 p.m. ET deadline the day before the
  next game, but no fetched source confirms clock times.
- **Conclusion.** At a Tuesday or Wednesday line, a Sunday game's final status (Friday) is not
  known. At most Wednesday's practice report is. The final report that the history keeps
  therefore fails, and the in-week state that would pass cannot be rebuilt for 2009-2025.

### Leakage traps

- **The week's own `report_status` in training rows.** It is Friday information (or later, since
  the overwrite keeps the final state), set after the deadline for every Sunday game.
  Prediction rows would hold whatever the file has on Tuesday: nothing, or estimates. The audit
  would not catch this. An injury count is not close to the score, and the audit never compares
  the two kinds of row.
- **Same-game snap counts.** Whether a player took a snap in the predicted game is known only
  after kickoff. Snap features for week N must come from games strictly before week N. The audit
  catches this only if a snap count nearly reproduced the score, which it would not.
- **Inactive lists.** Game-day inactives are set about 90 minutes before kickoff (unverified;
  not fetched), so any roster status taken from the game itself is a trap. The cached weekly
  rosters carry no status column today.
- **The quarterback.** Completed games carry the actual starter, upcoming games the projected one
  (`.agents/step6_plan.md`). A health group must not double-count the quarterback or reopen that
  skew.
- **Market overlap.** The pick-time spread already prices injuries known by Tuesday. With game
  lines, a health feature can add only what the line lacks. This is not leakage, but it is why
  the arm runs both with and without game lines.

### Possible feature form

- **The pick-time proxy uses only last week's information,** known by the end of week N-1, before
  any Tuesday line:
  - players listed `Out` or `Doubtful` (or not on the report but absent) in week N-1, counted by
    position group, with the quarterback left out;
  - starters absent last game: players with a high season-to-date snap share before week N-1 who
    took zero snaps in week N-1, weighted by that share. Built from the cached snap counts
    (2013+). Offense only unless the snap-count source is widened.
- **Where it goes.** A new `health` entry in `constants.FEATURE_GROUP_COLUMN_MARKERS`, with
  `away_`, `home_` and `_diff` columns like the other groups. An injury-report version needs a
  new nflreadpy source that caches per season under `data/cache/nflreadpy/` and degrades
  non-fatally when the current season is missing (AGENTS.md). A version from snap counts alone
  needs no new source.
- **Coverage.** Injuries start in 2009 and snap counts in 2013. Before then the columns are null,
  and the numeric imputer (`SimpleImputer(strategy="median")`, `ml/feature_spec.py`) would fill
  them with medians. The hypothesis should name a governing window inside the covered seasons,
  for example 2014-2025, and report weeks 1, 2 and 3-18 separately (`.agents/modeling_spec.md`,
  feature rules).
- **A measured arm:** two dataset builds behind an ETL flag (back up `data/*.csv`; rebuilding
  `data/` is must-ask). The reference runs on the build without the group. The candidate runs on
  the build with it, checked so that every other column is identical and so that the build with
  the group disabled reproduces the reference bit for bit, as the `qb_elos.csv` measurement did.
  Seeds `42` and `7`, averaged per game, candidate minus reference, 95% paired bootstrap
  intervals. Arms with game lines (`--market-anchor --market-transform`, production's disabled
  groups) and without (`--exclude-market --no-market-anchor --no-market-transform`, the
  no-game-lines default groups). The handoff's stats-only note asks for both. `HYPOTHESIS.md` is
  checksummed before the builds, and an independent reviewer writes `REVIEW.md`. The selection
  order is from `.agents/benchmarks.md`: Brier selects, log loss must not lose, then pool points
  and margin MAE. The six-season fit-noise floors are Brier `0.0013`, pick accuracy `0.008`, `75`
  pool points and margin MAE `0.03`.
- **A later serving check.** From 2025, the daily depth charts and the `injuries_2026` file as of
  Tuesday could test whether the lagged proxy matches what was known on Tuesday. That is a check
  on the proxy, not a feature.

### Verdict

**Passes only with a pick-time proxy**: last week's final report and starters absent in last
week's game, from snap counts. The current week's report fails, because the history keeps only
the final Friday state. Recommendation: measure this first. How sure: high on the timing and the
overwrite, medium that the lagged proxy beats the noise floor (the spread already prices
Tuesday's news).

## Incidental finding: retractable roofs differ between training and prediction rows

Found while checking the weather roof states. Not a change, only a question for the user:
changing feature values is must-ask.

- **What the code does today.** `src/nfl_predictor/utils/polars/loaders.py` (around lines
  300-349) maps the schedule's `roof` to `stadium_type`:
  - text containing "retract" becomes `retractable`;
  - "dome", "indoor" or "closed" becomes `dome`;
  - "outdoor" or "open" becomes `open`;
  - null becomes `unknown`.

  nflreadpy never writes "retractable". A retractable venue is `closed` or `open` after the game
  and null before it, so the `retractable` branch never fires.
- **Training rows never see `unknown`.** In the main checkout's `data/all_data_ml.csv` (built
  2026-10-16 02:00, read only; `stadium_type_check.py`), all 7,341 played rows are `dome` or
  `open`. All 30 unplayed 2026 games at the five retractable venues are `unknown`, and so is
  the Madrid game. From 2016 on, played games at those venues are 379 `dome` (roof closed) and
  50 `open`.
- **The model has no column for `unknown`.** The latest weekly run
  (`models/weekly_20261015_022416_9a4c092c/feature_importance.json`, read only) has only
  `cat__stadium_type_dome` and `cat__stadium_type_open`. The one-hot encoder ignores unknown
  categories, so these games get all zeros, a combination no training row has.
- **Two problems.**
  - A small game-day weather leak: training rows know whether the roof was opened, a choice made
    90 minutes before kickoff.
  - A difference between the benchmark and production (AGENTS.md rule 11): walk-forward folds
    never see `unknown`, so the benchmark does not measure what the weekly run does for these
    games.

  Week 7 has three of them: San Francisco at Atlanta, the Giants at Houston and Denver at
  Arizona.
- **A possible fix, not proposed as a change.** Map every retractable venue to one constant
  `retractable` category in every row, played or not. It changes feature values, so it needs two
  dataset builds and the user's approval. It could share builds with a climatology arm.

## Ranked list

1. **Team health with the lagged proxy** (last week's report and starters absent last game).
   This is the largest plausible effect on margins and win probability, the snap counts are
   already cached, and it serves the stats-only track. Main cost: a new feature group with an ETL
   flag, maybe a new nflreadpy source (injuries), two dataset builds (must-ask) and eight 2007-2025
   walk-forward runs: two arms, two seeds, with and without game lines. The handoff estimates
   about 31 minutes per comparable run on the GPU (unverified for this configuration).
2. **Venue climatology for weather.** Cheap to build from the schedule's own history, but it
   mainly touches the display-only total. Better done together with the retractable-roof
   question, so the two share one pair of builds.
3. **The referee crew: park.** Public on Tuesday, but no allowed pick-time source while
   nflverse's officials feed is off.
4. **Attendance: drop.** Recorded after the game, with no nflverse source.

## Open questions for the user

1. **The health proxy.** Accept last week's final report plus starters absent last game as the
   health proxy, or wait for an allowed in-week source? Recommendation: accept, because no
   allowed in-week history exists. Fairly sure.
2. **The retractable roofs.** Fix the `stadium_type` skew, for example one constant
   `retractable` category in every row? It changes feature values: two builds, and the
   checkpoints are invalidated. Recommendation: yes, grouped with the climatology builds. Fairly
   sure it is a real gap; unsure of its size.
3. **Both tracks.** Measure each idea both with and without game lines, as the handoff's
   stats-only note says? Recommendation: yes for team health. That doubles the runs.
4. **Retire two ideas.** Drop attendance and park the referee crew? Recommendation: yes to both.
   Sure.
5. **The pick-time line.** Does it mean before the weekly run (Tuesday or Wednesday), as these
   notes assume, or before the week's first kickoff? It decides whether Wednesday's practice
   report and the Thursday teams' early game statuses count.

## Not verified

- why the referee differs between the two nflverse sources in 27 games from 2015 to 2025;
- whether the Next Gen Stats feed carries crews before the games every week, or only in week 1;
- whether the schedule referee comes from Pro-Football-Reference;
- the injury report clock times;
- the current league injury policy text: the 2015 PDF and the 2016 page returned 404, so the
  timing rests on a search excerpt, ESPN 2016 and Legion Report;
- the current retractable-roof policy (the sources are from 2013 and 2018);
- when the Thursday-game teams' game status is published;
- the timing of game-day inactives;
- how soon nflverse publishes snap counts after a Monday game;
- whether nflreadpy offers defense snap counts;
- how long players listed `Out` stay out;
- attendance on Pro-Football-Reference and ESPN;
- the 2020 crowd announcements;
- the Paris and Munich roof override branch;
- the run-time estimate.

## Sources and commands

Scripts, all in `~/scratch/step8_research/scripts/` and run read-only on the repo:

- `download.py`: saves the schedule, officials and injury files 2009-2026 to `downloads/`;
- `analyze2.py`: coverage, crews, roof states, wind against the total line, and the shape of the
  injury files (output saved as `~/scratch/step8_research/analyze2_output.txt`);
- `officials_week1.py`: the 2026 week-1 crews against the schedule, and the 2015-2025
  mismatches;
- `games_history.py` and `games_history2.py`: ten `nfldata` snapshots;
- `week_dates.py`: the 2026 calendar and fill counts;
- `depth_charts.py`: depth-chart shape and timestamps;
- `cache_peek.py`: the cached snap-count and weekly-roster columns;
- `stadium_type_check.py`: `stadium_type` in `data/all_data_ml.csv`.

GitHub, through the `gh` command:

- `gh api` on `nflverse/nfldata` (`release_games.yml`, the commit history of `data/games.csv`);
- `gh api` on `nflverse/nflverse-rosters` (`exec/update_officials.R`, `exec/update-injuries.R`,
  the workflows, commit `ecc6ef18`);
- `gh api` on `nflverse/nflverse-data` (the release assets for injuries and officials);
- `gh api` on `nflverse/nflverse-data-archives` (README);
- `gh run list -R nflverse/nflverse-rosters` for the officials and injuries workflows.

Web pages read:

- <https://footballzebras.com/?p=39395>
- <https://www.sportsbettingdime.com/?p=804427>
- <https://www.footballzebras.com/2025/10/week-7-referee-assignments-2025>
- <https://www.footballzebras.com/2025/12/week-16-referee-assignments-2025>
- <https://footballzebras.com/2018/10/week-5-referee-assignments-2018>
- <https://www.footballzebras.com/2025/07/officiating-crews-for-the-2025-season/>
- <https://footballzebras.com/2013/05/nfl-retractable-roof-policies>
- <https://www.ajc.com/sports/finished-roof-opens-new-option-for-falcons-home-games/iQVQA7iiItfuFNPUcI218L/>
- <https://www.nesdis.noaa.gov/about/k-12-education/weather-forecasting/how-reliable-are-weather-forecasts>
- <https://open-meteo.com/en/docs/historical-forecast-api>
- <https://africa.espn.com/nfl/story/_/id/17361111/nfl-streamlines-injured-players-game-status-removes-probable-designation>
- <https://legionreport.com/nfl-week-3-injury-report/>

Failed fetches:

- <https://operations.nfl.com/updates/football-ops/2016-nfl-injury-report-policy> (404);
- <https://operations.nfl.com/media/1818/2015-injury-report-policy.pdf> (404; a search excerpt
  was used);
- <https://scijinks.gov/forecast-reliability/> (DNS failure; the NOAA NESDIS page was used
  instead);
- the nflreadr dictionary web pages, which rendered no fields (the `data-raw` CSVs were used).
