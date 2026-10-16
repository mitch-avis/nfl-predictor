# Frontend audit notes

Observations about the web app (`web/` and the API under `src/nfl_predictor/api/` that feeds it)
noticed while working on something else: things the UI shows wrongly or incompletely, and code or
accessibility problems in the components. This file is a collecting point, not a worklist. An
entry becomes work only when it moves into `.agents/TODO.md`; when it is fixed, it moves to
"Fixed since noted" with the release that fixed it.

## How to add an entry

Add a bullet under the matching section with:

- the date noticed and the page or component;
- `file:line` on `main` at that date (line numbers drift; the file and the symbol are the anchor);
- what was seen, checked against the code, not inferred;
- why it matters to someone using the page;
- the suggested direction, if one is clear, and whether a fix is already planned.

Verify each `file:line` before writing it, and drop an entry that no longer holds rather than
leaving it stale.

## Data shown in the UI

- 2026-10-16, Data status page, leakage audit card: flagged columns show as `[object Object]`.
  `ml/leakage_audit.py:154` declares `flagged_columns` as a list of `{"column", "reason"}`
  objects (appended at lines 113-114 and 165), but `web/src/api/types.ts:366` types it as
  `string[]` and `web/src/pages/DataStatus.tsx:135` renders it with `.join(', ')`. The card is
  clean today because nothing is flagged, so the bug shows only when the audit finds something,
  which is when the card matters. A fix is planned: list it as open until it lands.
- 2026-10-16, Data status page, files card: the card says "Everything under data/ that the model
  and reports read" (`web/src/pages/DataStatus.tsx:88`), but `DATASET_FILES`
  (`api/readers/data_status.py:28`) lists only `all_data_ml.csv`, `all_data.csv`,
  `completed_games_ml.csv`, `completed_games.csv` and `qb_elos.csv`. Files the code does read
  are missing: `meta_data.csv` (`utils/polars/ratings_build.py:278`), `strength_snapshots.csv`
  (the composite power-ranking method), `market_lines_metadata.json` (read by the lines refresh),
  `own_elo_metadata.json` and the ported ratings under `data/ratings/`. Direction: list those, or
  reword the description to "the main dataset files".
- 2026-10-16, Data status page, cache coverage: `CACHE_FILE_RE`
  (`api/readers/data_status.py:35`) matches only `pbp_` and `schedule_` files, so the coverage
  shown leaves out the other per-season nflreadpy caches on disk (`official_team_stats`,
  `player_stats_week`, `rosters`, `rosters_weekly`, `snap_counts`, `team_stats`) and the
  `data/cache/nfelo/` market-lines cache. A season missing from one of those does not show.
- 2026-10-16, Data status page, leakage audit card: `latest_leakage_audit`
  (`api/readers/data_status.py:119-139`) shows the newest `*leakage_audit*.json` under the roots
  by modification time, and the card (`web/src/pages/DataStatus.tsx:121-136`) names no dataset or
  run. An audit of an experiment or an old file can therefore stand in for the production
  dataset's. The newest audit's recorded config also has `market_transform: False`, while the
  production run uses the market transform, so "Clean" may describe a different feature set from
  the one the weekly run trains on. Direction: show the audited file and its settings, and pick
  the audit attached to the active run.
- 2026-10-16, Runs page and Power rankings page: `scan_runs` (`api/runs/indexer.py:135`) takes
  every directory under `models/` that holds a `metadata.json`, so the run list includes smoke
  tests, a training run and the week-1 2026 experiment runs (`week01_2026_refreshed`,
  `_strength`, `_totalfix`, `_totalref`) beside the production weekly runs. The power page's
  week-over-week change uses `previous_rankings` (`api/routers/power.py:22-33`): the newest run
  whose rankings are stamped one week earlier, which can be one of those experiments rather than
  the production run for that week. Direction: mark runs as production or experiment, and take
  the previous week only from production runs.

## UI and code

- 2026-10-16, `DataTable`: a descending sort puts missing values first, although the comment at
  `web/src/components/table/DataTable.tsx:84` says "missing values last". `compareCells`
  (line 85) ranks `null` last, react-table reverses the comparison for a descending sort, and
  `sortUndefined: 'last'` (line 125) only catches `undefined`, while the API sends `null`. A fix
  is planned on its own branch: open until it lands.
- 2026-10-16, `DataTable`: a button inside a button. The column's info button
  (`web/src/components/common/InfoTooltip.tsx:11`, rendered by `HeaderLabel` at
  `DataTable.tsx:70`) sits inside the sort button (`DataTable.tsx:168-176`). That is invalid HTML,
  and the sort button's `aria-label` ("Sort by X", line 172) hides the column description from
  screen readers. Direction: put the info button beside the sort button, not inside it.
- 2026-10-16, `DataTable` table semantics: no `aria-sort` on the header cells (`<th>` at
  `DataTable.tsx:158`), no `scope="col"`, no `<caption>` (`<table>` at line 146); the sort state
  shows only as an icon (line 175), and multi-column sorting by shift-click is not discoverable.
- 2026-10-16, inconsistent tables: `web/src/pages/Runs.tsx`, `DataStatus.tsx`, `Users.tsx` and
  `Jobs.tsx` use the plain `web/src/components/ui/table.tsx` (no sorting, heat colors or sticky
  header), while `web/src/pages/HeadToHead.tsx:85`,
  `web/src/components/ratings/ComparisonPanel.tsx:63` and
  `web/src/components/ratings/UnitRankRanges.tsx:8` are hand-built tables. The hand-built ones do
  use `<caption>` and `scope`, which `DataTable` lacks.
- 2026-10-16, `DataTable` first-click sort direction: react-table picks it from the column's first
  non-missing value (text ascending, numbers descending; `getAutoSortDir` in table-core 9.3.1),
  and nothing documents it. On rank columns, where 1 is best, the first click shows the worst
  first. Direction: set `sortDescFirst` per column from the registry metadata.
- 2026-10-16, dead script: `web/package.json:14` defines `gen:api` as
  `node scripts/gen-api-types.mjs`, and `web/scripts/` does not exist. `.agents/web_ui_plan.md:410`
  describes it as a type-drift check that CI runs; CI does not.
- 2026-10-16, `DataTable` empty state: the empty message (`DataTable.tsx:141`) is a plain `div`
  without `role="status"`. Loading and error states are handled per page and were not reviewed.

## Fixed since noted

- Data status page, files card: `nfl_elo.csv` was listed although nothing reads it; removed from
  `DATASET_FILES` in 0.63.2.
