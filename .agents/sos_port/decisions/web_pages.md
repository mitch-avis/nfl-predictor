# Step 6 decisions: the team and quarterback web pages

Judgment calls made on 2026-10-10 while folding the `nfl-sos-ratings` web pages (its `web/` and
its `ui_api.py` and `ui_data.py` at `9081bce`) into this app as Milestone 58 task 58.2, on the
implementer branch that fast-forwarded to `feat/step6-sos-port` and merged the season statistics
branch. Each entry gives what differs, the options, the decision, the reasoning, any data, and how
to reverse it. Commit hashes are on that branch.

Data marked "published tables" comes from a manual check: the `nfl-sos-ratings` files of 1999-2026
(534 files) copied read-only into a scratch folder with this repository's team codes (`WAS` to
`WSH`; its other codes were already canonical), served from the worktree's code on a spare port.

## 1. The pages read `data/ratings/` under the configured data directory

- **What differs:** the source app served its own `data/`; this app's plan read the sibling
  checkout through `NFLP_SOS_DATA_DIR` (`root/nfl-sos-ratings/data`).
- **Options:** (a) keep `NFLP_SOS_DATA_DIR` and point its default at `data/ratings`; (b) derive
  the folder from the data directory (`Settings.ratings_path = data_path / "ratings"`) and drop
  the separate setting; (c) a new `NFLP_RATINGS_DIR` setting.
- **Decision:** (b).
- **Reasoning:** the ETL writes the tables into this repository's data directory (the step's data
  contract), so the folder follows `NFLP_DATA_DIR` like every other dataset, and a scratch tree
  needs one variable. A second knob would only let the two drift apart. The sibling repository is
  read-only and no longer a source for the app. An `NFLP_SOS_DATA_DIR` left in an environment is
  ignored (the settings ignore unknown variables), so nothing breaks.
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
  existing registry projection does. Decimals are left null and chosen on the page from the values
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
  not ported; the opponent links to its own page instead, which has its rank range.
- **Reasoning:** a quarterback page would otherwise download the 0.9 MB team table only for five
  numbers per game, and the description must not claim an average that the column is not.
- **How to reverse it:** `sections.with_opponent_ratings` and `_opponent_overrides`
  (`3301099`).

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
- **Reasoning:** `constants.py` is the canonical team mapping (`AGENTS.md`), so a second list
  could drift from it.
- **How to reverse it:** revert `53344e6`.

## 9. The glossary leaves out the garbage-time filter

- **What differs:** the source's glossary listed every registry metric and explained the
  garbage-time filter, an exploration view.
- **Options:** (a) port the filter view; (b) list everything, the filter's metrics included;
  (c) leave out the filter's metrics, its `filtered_` prefix and its concept entry.
- **Decision:** (c). The filter view and its `wp-ratings` routes are not ported: no page here
  depends on them (the source's detail page showed one filter line, which is left out too).
- **Reasoning:** the task puts the filter out of scope unless a page depends on it, and none
  does; glossary entries for columns no page shows would only confuse. The registry itself is
  unchanged (it belongs to the season statistics port).
- **Data:** the registry has 319 metrics; the glossary shows 311. The 8 left out are `wp_unit`,
  `wp_bin`, `wp_bin_plays`, `wp_bin_epa`, `wp_kept_play_share`, `wp_kept_dropback_share`,
  `qb_wp_bin_epa` and `qb_wp_bin_dropbacks`. One registry category description (Overall) still
  mentions the filter's data; that text is the registry's.
- **How to reverse it:** drop the filter in `api/ratings/glossary.py` (`3301099`) and port the
  source's `wp_filter.py` view and routes.

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
- **Decision:** (b).
- **Reasoning:** averaging per-game rates over one or two games mixes rates without their
  denominators, which the season statistics port takes care to avoid; scores and the opponent's
  own season ratings say how the games went and how strong the opponent was. The opponent
  profiles (what the opponents did against everyone else) are the views the stats card already
  offers.
- **How to reverse it:** `opponents.ts` (`10077c9`) and the detail page's "Opponents faced" card
  (`b4144de`).

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
- **Decision:** leave them out. Data refreshes run as this app's ETL job; the app keeps one
  palette; the glossary carries the method's ideas in plain words instead of linking to the
  read-only repository; game ids are not shown (the week and the opponent identify a game, home
  `vs` or away `@`).
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

## 16. Whole season tables, uncompressed

- **What differs:** nothing; the source also sent whole tables. Recorded because of the size.
- **Options:** (a) as is; (b) page or slice the table by view on the server; (c) compress
  responses app-wide.
- **Decision:** (a).
- **Reasoning:** the view filters, totals and search run in the browser over one payload, so a
  view switch costs nothing. Compression would apply to every route, the job log stream included,
  and is a separate change.
- **Data:** published tables: the team table is 0.90 MB (2026) and 0.96 MB (2025) of JSON,
  served in about 0.12 s; the quarterback table 0.33 MB; one team's game log 0.16 MB.
- **How to reverse it:** add slicing to `api/ratings/tables.py` or a compression middleware.
