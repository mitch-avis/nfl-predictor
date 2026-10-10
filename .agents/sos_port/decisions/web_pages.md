# Step 6 decisions: the team and quarterback web pages

Judgment calls made on 2026-10-10 while folding the `nfl-sos-ratings` web pages (its `web/` and
its `ui_api.py` and `ui_data.py` at `9081bce`) into this app as Milestone 58 task 58.2, on the
implementer branch that fast-forwarded to `feat/step6-sos-port` and merged the season statistics
branch. Each entry gives what differs, the options, the decision, the reasoning, any data, and how
to reverse it. Commit hashes are on that branch. The delegating session reviewed the branch the
same day: entries 17 to 21 come from that round, and the earlier entries say where it corrected or
extended them ("Corrected in review", "since review"). The branch then merged the reviewed step-6
branch (`25ca44a`).

Data marked "published tables" comes from a manual check: the `nfl-sos-ratings` files of 1999-2026
(534 files) copied read-only into a scratch folder with this repository's team codes (`WAS` to
`WSH`; its other codes were already canonical), served from the worktree's code on a spare port.

**Status.** No job in this repository writes `data/ratings/` yet: the season-statistics ETL step
that is to write those tables is planned separately. Until it lands, the pages show an empty state
naming the folder and the files they read, and the phase's done-criterion in
`.agents/web_ui_plan.md` (the current-season team page shows rating heatmaps and a team detail with
game logs) waits on that step; so far the pages have been checked only against the copied tables.

## 1. The pages read `data/ratings/` under the configured data directory

- **What differs:** the source app served its own `data/`; this app's plan read the sibling
  checkout through `NFLP_SOS_DATA_DIR` (`root/nfl-sos-ratings/data`).
- **Options:** (a) keep `NFLP_SOS_DATA_DIR` and point its default at `data/ratings`; (b) derive
  the folder from the data directory (`Settings.ratings_path = data_path / "ratings"`) and drop
  the separate setting; (c) a new `NFLP_RATINGS_DIR` setting.
- **Decision:** (b).
- **Reasoning:** the step's data contract puts the tables in this repository's data directory,
  where the season-statistics ETL step is to write them, so the folder follows `NFLP_DATA_DIR` like
  every other dataset, and a scratch tree needs one variable. A second knob would only let the two
  drift apart. The sibling repository is read-only and no longer a source for the app. An
  `NFLP_SOS_DATA_DIR` left in an environment is ignored (the settings ignore unknown variables),
  so nothing breaks.
- **How to reverse it:** revert `f164a3f` (with the routes that read `ratings_path`).

## 2. One route family, `/api/ratings/{season}/{kind}/...`

- **What differs:** the source had `/api/seasons/{season}` (both tables in one payload) and
  separate `/teams/...` and `/qbs/...` routes for each section; the plan named `/sos/...`.
- **Options:** (a) the source's paths; (b) `/api/ratings/...` with `kind` a path parameter that
  is `teams` or `qbs`, one handler per section; (c) separate team and quarterback routers.
- **Decision:** (b): `/ratings/seasons`, `/ratings/{season}/{kind}`,
  `/ratings/{season}/{kind}/rank-ranges`,
  `/ratings/{season}/{kind}/{id}/{game-logs,rating-history,rank-history,head-to-head}`,
  `/ratings/schedules`, `/ratings/metrics`. The head-to-head route is the source's
  `rating-pairs`.
- **Reasoning:** this app prefixes routes by area (`/power`, `/pool`); "ratings" names the data
  now that it is this repository's own. One handler per section keeps the two kinds identical,
  and a separate team and quarterback request each loads only what its page needs (the source
  sent both tables together). An unknown kind is a 422 from the path's type.
- **How to reverse it:** rename the router's paths (`3ee20d8`) and the frontend's
  `ratingsPath` (`c7de9e1`).

## 3. A season is listed once its two index tables exist

- **What differs:** the source listed a season when six kinds existed (`team_per_game_stats`,
  `qb_per_game_stats`, `combined`, `qb_combined`, `ratings`, `qb_ratings`).
- **Options:** (a) the source's six; (b) the two the index pages read (`combined`,
  `qb_combined`); (c) every kind.
- **Decision:** (b), and every other table is optional: its route answers 404 `no_ratings_table`
  naming the file, and the page hides that section.
- **Reasoning:** the per-game stats tables are already inside `combined` and no page reads them,
  so requiring them hides a season that would display. A partial build then shows what it has.
- **Data:** the published tables hold every kind for all 28 seasons (19 kinds a completed season,
  21 for 2026), so both rules list the same 28 seasons.
- **How to reverse it:** change `INDEX_KINDS` in `api/ratings/files.py` (`f164a3f`).

## 4. Registry metadata in the app's column format, not the source's

- **What differs:** the source served the registry's own column metadata and drew its tables with
  its own components; this app's tables read `ColumnMetaOut` (kind, polarity, heatmap, sticky).
- **Options:** (a) serve the registry's metadata and port the source's table components; (b) map
  each resolved column into `ColumnMetaOut` and add the registry's fields (`full_name`, category,
  subcategory, `contextual`, `shape`, `percent`) as `MetricColumnOut`, so the existing
  `DataTable` renders the tables.
- **Decision:** (b). Kind comes from the shape first (identity text, flags), then a proportion
  becomes a percentage, then the Parquet type (whole numbers or decimals); graded and contextual
  numbers carry a gradient whose direction is the registry's polarity; the team code and the
  quarterback's name are pinned. A column the registry cannot resolve is left out, as the
  existing registry projection does, with a warning in the server log naming it (added in review,
  `743846a`). Decimals are left null and chosen on the page from the values
  shown (the source's `columnDecimals` rule), because the season-totals view changes the values.
- **Reasoning:** one table component keeps the app's look, sorting and tooltips, and the
  registry stays the single source of every label and description.
- **Data:** on the published tables every column of every kind resolves; the 2026 team table has
  492 columns (210 per game, 36 per snap, 239 opponent profile, 6 ratings, the team).
- **How to reverse it:** `75059f4` (metadata) and `824f5ba` (shading).

## 5. Context columns shade in a muted slate hue; graded columns keep the app's green

- **What differs:** the source shaded grades on a blue-to-orange diverging scale in the chosen
  team palette and context columns in the palette's main color; this app shades grades from
  clear to green.
- **Options:** (a) port the source's scales and team palettes; (b) keep the app's green for
  grades and add one muted slate hue for context, deeper toward the tougher end the polarity
  names; (c) leave context columns unshaded.
- **Decision:** (b), through an optional `contextual` flag on `ColumnMeta`.
- **Reasoning:** the pages should read like the rest of the app, and context (schedule strength,
  the opponents faced) is not better or worse, so it must not wear the grade color. Columns
  without the flag shade exactly as before.
- **How to reverse it:** revert `824f5ba`.

## 6. Each game carries its opponent's season ratings from the server

- **What differs:** the source attached the opponent's ratings to each game in the browser, from
  the season's team table, and showed a rank-range chip beside each opponent.
- **Options:** (a) the browser, as the source did; (b) the server, joining the season's `ratings`
  table on the opponent as `opp_team_rating`, `opp_offense_rating`, `opp_defense_rating`,
  `opp_special_teams_rating` and `opp_SRS`.
- **Decision:** (b), with each column's text rewritten: the registry's `opp_` rule describes an
  average over every opponent faced, but here the value is one opponent's whole season. A season
  without a ratings table still serves its game log. The rank-range chip beside each opponent is
  not ported; the opponent links to its own page instead, which has its rank range. Since review
  the join also carries `opp_team_rank`, the opponent's league rank on Team Rating (1 is best),
  and the payload the league's size, for the tiers in entry 11.
- **Reasoning:** a quarterback page would otherwise download the 0.9 MB team table only for six
  numbers per game, and the description must not claim an average that the column is not.
- **How to reverse it:** `sections.with_opponent_ratings` and `_opponent_overrides`
  (`3301099`, the rank `f541e36`).

## 7. The season under way comes from the calendar

- **What differs:** the source marked a season in progress when it equalled its hard-coded
  `config.SEASON = 2026` and a team had games left.
- **Options:** (a) a constant; (b) the season today's date falls in (`clock.nfl_season`, the
  ETL's calendar), with games per team from the regular-season weeks minus the bye (16 before
  2021, 17 since).
- **Decision:** (b). Without `games_played` the current season counts as in progress.
- **Reasoning:** a constant goes stale every year; the ETL already defines the calendar.
- **Data:** on the published tables 2026 reads as in progress with 4 games played; 2025 does not.
- **How to reverse it:** `tables.season_progress` and the route's `current_season` (`3301099`,
  `3ee20d8`).

## 8. Franchise names come from `constants.TEAM_MAPPING`

- **What differs:** the source kept a team-name map in its frontend (`WAS` as Washington,
  `LA` and `LAR` both the Rams).
- **Options:** (a) copy the map into the frontend; (b) send the names of the table's teams with
  each entity table.
- **Decision:** (b), `team_names` in the entity table payload; an unknown code names itself.
  Since review the rank-range payload carries them too, so the Rank ranges page no longer
  downloads the team table only for the names.
- **Reasoning:** `constants.py` is the canonical team mapping (`AGENTS.md`), so a second list
  could drift from it.
- **How to reverse it:** revert `53344e6` (the rank-range names `b28683c`).

## 9. The glossary leaves out the garbage-time filter

- **What differs:** the source's glossary listed every registry metric and explained the
  garbage-time filter, an exploration view.
- **Options:** (a) port the filter view; (b) list everything, the filter's metrics included;
  (c) leave out the filter's metrics, its `filtered_` prefix and its concept entry.
- **Decision:** (c). The filter view and its `wp-ratings` routes are not ported: no page here
  depends on them (the source's detail page showed one filter line, which is left out too).
- **Reasoning:** the task puts the filter out of scope unless a page depends on it, and none
  does; glossary entries for columns no page shows would only confuse. The registry's metrics are
  unchanged (they belong to the season statistics port).
- **Data:** the registry has 319 metrics; the glossary shows 311. The 8 left out are `wp_unit`,
  `wp_bin`, `wp_bin_plays`, `wp_bin_epa`, `wp_kept_play_share`, `wp_kept_dropback_share`,
  `qb_wp_bin_epa` and `qb_wp_bin_dropbacks`. Corrected in review: two registry category
  descriptions mentioned the filter's data, not one (the team Overall and the quarterback Passing
  Volume categories). On the delegating session's decision both now describe only what the pages
  show (`22fcdfd`, on top of the reviewed registry), and a test keeps the glossary's categories
  free of the filter.
- **How to reverse it:** drop the filter in `api/ratings/glossary.py` (`3301099`) and port the
  source's `wp_filter.py` view and routes; restore the two category texts from `22fcdfd`.

## 10. Categories come from the registry with each table

- **What differs:** the source's frontend hard-coded the team categories and subcategories and
  the quarterback categories; its special-teams subcategories (kicking, coverage, returns) are not
  in the registry this repository ported, which gives special teams none.
- **Options:** (a) copy the lists; (b) send the entity's categories in display order with each
  table.
- **Decision:** (b); the views and filters read them.
- **Reasoning:** the registry is the single source of the taxonomy, so the filters cannot offer
  a subcategory no column has.
- **Data:** on the 2026 published tables the team views show 15 Overall, 100 Offense, 68 Defense
  and 2 Special Teams columns per game.
- **How to reverse it:** `tables.categories` (`3301099`) and `views.resolveViewState`
  (`10077c9`).

## 11. One row per opponent faced, without the source's "vs season" columns

- **What differs:** the source's detail page built a ledger of unique opponents whose columns
  followed the chosen stat group, averaged each stat over the games against that opponent, tiered
  each opponent (tougher, middle, softer) and added one "vs season" delta.
- **Options:** (a) port it whole; (b) one row per opponent with the games, weeks and results,
  the opponent's season ratings and the average score of those games, plus the opponents'
  head-to-head-excluded profiles through the "Opponents per game" and "per play" views; (c) no
  ledger.
- **Decision:** (b). Accepted in review as a simplification of the source's ledger. Since review
  each opponent also carries the source's tier, by its league rank on Team Rating: Tougher for the
  top 10, Softer for the bottom 10, Middle for the rest (the source's rule). The source picked the
  ranking metric by stat group (the opponent's defense rating for offense stats and a
  quarterback's passing stats, its offense rating for defense stats, Team Rating otherwise); here
  it is always Team Rating, the one rank the game log carries, and the tier is plain text rather
  than shaded. An opponent without a rank (one the season's ratings table does not list) is
  Middle, as in the source.
- **Reasoning:** averaging per-game rates over one or two games mixes rates without their
  denominators, which the season statistics port takes care to avoid; scores and the opponent's
  own season ratings say how the games went and how strong the opponent was. The opponent
  profiles (what the opponents did against everyone else) are the views the stats card already
  offers.
- **How to reverse it:** `opponents.ts` (`10077c9`, the tiers `68b381a` and `ca25600`) and the
  detail page's "Opponents faced" card (`b4144de`).

## 12. A missing team or quarterback is a message on its page

- **What differs:** the source redirected to the index with a "not found" note.
- **Options:** (a) redirect; (b) say "No team LV in 2025" in place, with the season picker and a
  link back.
- **Decision:** (b).
- **Reasoning:** switching seasons on a detail page keeps the same id; a quarterback who did not
  play that season should not lose the page's season, and the reader can pick another.
- **How to reverse it:** `RatingsDetail.tsx` (`b4144de`).

## 13. What the pages do not port

- **What differs:** the source app also had a refresh button (`/api/refresh`, which rebuilt its
  data), team color palettes and team chips, CSV export buttons, a link to its methodology on
  GitHub, and game ids linked to an outside site (nflsavant.com).
- **Options:** port each, or leave it out.
- **Decision:** leave them out. Data refreshes belong to this app's ETL jobs (the season-statistics
  ETL step, planned separately, is to write these tables); the app keeps one palette; the glossary
  carries the method's ideas in plain words instead of linking to the read-only repository; game
  ids are not shown (the week and the opponent identify a game, home `vs` or away `@`).
- **Reasoning:** the task excludes anything that writes data; the other items are presentation
  the app does not have elsewhere, and an outside link would send readers to a third-party site.
- **How to reverse it:** each is additive; nothing here prevents it.

## 14. The sidebar and the overview group the pages in sections

- **What differs:** the sidebar was one "Navigate" list of 11 pages (10 for a viewer); the five
  new pages make 16 (15 for a viewer).
- **Options:** (a) one longer list; (b) four sections: This week (overview, predictions, pool,
  betting), Teams and players (power rankings, teams, quarterbacks, schedules, rank ranges, head
  to head), Model and data (model, data, jobs, runs, users) and Reference (glossary), the
  overview's cards grouped the same way.
- **Decision:** (b). Betting moves up beside the week's pages, and Model above Data.
- **Reasoning:** a 16-item list is hard to scan, above all on a phone; the sections put the new
  pages beside the power rankings they relate to.
- **How to reverse it:** `nav.ts`, `AppSidebar.tsx` and `Dashboard.tsx` (`d8eb61f`).

## 15. Power Rankings stays; it and Teams link to each other

- **What differs:** the source's teams index and this app's power rankings both rank teams.
- **Options:** (a) replace Power Rankings with the ported ratings; (b) keep both, linked, each
  saying what it is.
- **Decision:** (b). Power Rankings is the model's Bradley-Terry strength on margins with
  projected standings; Teams is the ported EPA ratings adjusted for every opponent, with each
  team's season stats.
- **Reasoning:** they measure different things, and the step-6 phase on power rankings decides by
  measurement how the ported ratings and the model combine; replacing the page now would pre-empt
  it.
- **How to reverse it:** the links in `PowerRankings.tsx` and `Ratings.tsx` (`d8eb61f`,
  `b4144de`).

## 16. Whole season tables, not sliced by view

- **What differs:** nothing; the source also sent whole tables. Recorded because of the size.
- **Options:** (a) as is; (b) page or slice the table by view on the server; (c) compress
  responses app-wide.
- **Decision:** (a) for the payload's shape. The first pass also left responses uncompressed;
  review added (c), entry 17.
- **Reasoning:** the view filters, totals and search run in the browser over one payload, so a
  view switch costs nothing.
- **Corrected in review:** the first pass argued against compression that it would apply to the
  job log stream too. It does not: Starlette's `GZipMiddleware` (1.7.0) leaves
  `text/event-stream` alone by default. Its sizes were also those of the 2026 season in progress,
  which understate a completed season's.
- **Data:** published tables: the team table is 0.90 MB (2026) and 0.96 MB (2025) of JSON,
  served in about 0.12 s. Corrected in review: a completed season's quarterback table is 0.49 MB
  (0.33 MB for 2026), and a game log runs up to 0.27 MB (one team's 0.16 MB for 2026).
- **How to reverse it:** add slicing to `api/ratings/tables.py`.

## 17. Responses are gzipped app-wide, at level 6

- **What differs:** the source app sent its JSON uncompressed, as this app did.
- **Options:** (a) leave responses uncompressed; (b) Starlette's `GZipMiddleware` on the whole
  app at its default level, 9; (c) the same at level 6, gzip's own default; (d) compress only the
  ratings routes.
- **Decision:** (c). The delegating session chose app-wide compression in review, and level 6
  after the independent review timed both levels. `create_app` adds the middleware with
  `compresslevel=6`; it compresses responses of 500 bytes or more for clients that send
  `Accept-Encoding: gzip`.
- **Reasoning:** the season tables are the app's largest payloads and repetitive JSON, so they
  shrink five- to seven-fold, and one middleware covers every route. Level 9 saves a few percent
  more bytes for noticeably more CPU on every response, which pays only on slow links. Starlette
  skips `text/event-stream` by default, so the job log stream is sent uncompressed as before.
  `tests/api/test_compression.py` pins the level, the plain response to a client without gzip, and
  the uncompressed stream.
- **Data:** measured by the independent review on the published tables at `5536417`, in bytes,
  for the 2025 team table, the 2025 quarterback table, the Chiefs' 2025 game log and the metric
  list: plain JSON 960,476 / 490,845 / 265,378 / 145,521; level 6 179,732 / 82,391 / 38,427 /
  20,444 (zlib on the served payloads); level 9 171,389 / 75,331 / 37,431 / 20,237 (served
  through the app). Level 9 saves 4.6% / 8.6% / 2.6% / 1.0% over level 6 for +7.2 / +10.2 / +1.4 /
  +0.8 ms of CPU (one pinned core, median of 100 runs), worth it only on links slower than about
  2-9 Mbit/s.
- **How to reverse it:** remove the middleware from `create_app` (`c4362e8`), or drop
  `compresslevel=6` for Starlette's level 9 (`8149a6d`).

## 18. A table a page needs that cannot be read answers 503; one it can do without is left out

- **What differs:** review found paths to a 500 and to needless 503s. A file Polars cannot read
  (empty, cut short, not Parquet) or the server may not open (a `PermissionError`, which Polars
  raises as the builtin `OSError`) was a 500, as was a table removed between the schedule
  leaderboard's folder scan and its read. Once those were 503s, an unreadable table that only
  adds to a page (the season ratings beside a game log, the team history that starts a rank
  history) failed the whole page, and one unreadable season failed the whole leaderboard.
- **Options:** for an unreadable table a route needs, (a) the 500, (b) a 404 as if missing, or
  (c) a 503 naming the file; for an unreadable table that only adds to a page, or one season of
  the leaderboard, (a) a 503 for the whole page or (b) leave it out as if missing, with a warning;
  for a vanished table, (a) the 500 or (b) leave that season out.
- **Decision:** (c), (b) and (b), from review. A route answers an unreadable table it needs with a
  503, code `ratings_table_unreadable`, whose message names the file and says it may be cut short,
  still being written, or not readable by the server. A table that only adds to a page is treated
  as missing when it cannot be read, with a warning in the server log naming it; the leaderboard
  leaves out a season it cannot read (with the same warning) or that vanished.
- **Reasoning:** a 503 says the data is there but cannot be used yet (a table being rewritten),
  which a 404 would hide, and the page shows that message instead of a bare server error. A page
  that can be served without a table should not fail because of it; the warning keeps the broken
  file in view for whoever runs the server.
- **How to reverse it:** `files.read_table` and the router's `_unreadable` (`b7b1f1a`, the
  `OSError` case `8c2b224`); the router's `_read_optional` and `schedules._ratings_by_season`
  (`63b6b0f`, `a9e24b6`).

## 19. The index table and the comparison show no rank ranges

- **What differs:** the source's index table had a "Rank range" column whenever the view showed
  the headline rating (each row's middle 50% of ranks across the redraws, with a small interval
  track; `RANK_RANGE_COLUMN` in its `EntityTable.tsx`), and its comparison panel showed each
  compared row's published rank, middle 50% and track under its name (`ComparisonPanel.tsx`).
  Neither is ported; the ranges are on the Rank ranges page and on each detail page.
- **Options:** (a) port both, fetching the season's rank ranges on the index page and joining
  them by id; (b) leave them out.
- **Decision:** (b), as delivered. Not a choice made in the first pass: it was missed there and
  found in review, so it is logged here as a difference.
- **Reasoning:** (a) costs the index page a second request (the rank-range payload) and a
  column built from another payload; both features are additive and can follow.
- **How to reverse it:** call `useRankRanges` in `Ratings.tsx`, add the column through a renderer,
  and add a range line to `ComparisonPanel.tsx`, reusing `RankInterval`.

## 20. No selection bar for the compared rows

- **What differs:** once a row was ticked, the source's toolbar showed "n selected" with "View
  comparison" (a jump to the panel) and "Clear selection". Here the comparison panel appears under
  the table as soon as a row is ticked, each compared row has its own remove button, and a note
  under the table says up to four rows can be compared.
- **Options:** (a) port the bar; (b) leave it out.
- **Decision:** (b), as delivered; like entry 19, missed in the first pass and logged after review.
- **Reasoning:** without the bar, reaching the panel below a 32-row table takes a scroll, and
  clearing four compared rows takes four clicks; the bar is small and additive.
- **How to reverse it:** add the bar to `Ratings.tsx` above the table; clearing sets `compare` to
  empty in the URL.

## 21. Search, sort and the qualifier switch reset when the reader leaves the index

- **What differs:** the source kept every index control (view, categories, compared rows, search,
  sort, the qualifier switch) in an app-wide store, so all of them survived a visit to a detail
  page and back. Here the season, view, category, hidden subcategories and compared rows are in
  the URL, so they survive the round trip and can be shared, but the search text, the sort and
  the "Show QBs below the qualifier" switch are the page's own state and reset.
- **Options:** (a) put search, sort and the switch in the URL as well; (b) an app-wide store like
  the source's; (c) leave them as they are.
- **Decision:** (c), as delivered; missed in the first pass and logged after review.
- **Reasoning:** (a) would keep the index's whole state in one place, the URL, and make a sorted,
  searched view shareable; it needs `DataTable` to take its sort from its caller (today it holds
  the sort itself and takes only a default).
- **How to reverse it:** add search, sort and switch parameters through `useRatingsParams`, with a
  controlled sort in `DataTable`.
