import { CalendarClock, ListOrdered, Search, Swords } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'

import { useEntityTable } from '@/api/queries'
import type { EntityKind, EntityTable, Row } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { ComparisonPanel } from '@/components/ratings/ComparisonPanel'
import { RatingsSeasonState, UnavailableSeason } from '@/components/ratings/RatingsStates'
import { SeasonPicker } from '@/components/ratings/SeasonPicker'
import { ViewControls, type ViewChange } from '@/components/ratings/ViewControls'
import { withDecimals } from '@/components/ratings/decimals'
import { entityConfig, entityId, entityLabel, findEntityRow } from '@/components/ratings/entities'
import { carriedSearch, useRatingsParams, useRatingsSeason, viewPatch } from '@/components/ratings/useRatingsParams'
import { resolveViewState, seasonView, VIEWS } from '@/components/ratings/views'
import { DataTable, type CellRenderer } from '@/components/table/DataTable'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'

/** The most rows the comparison puts side by side. */
const MAX_COMPARE = 4

function matches(kind: EntityKind, row: Row, names: Record<string, string>, needle: string): boolean {
  const team = String(row.team ?? '')
  return [entityId(kind, row), entityLabel(kind, row, names), team, names[team] ?? ''].some((text) => text.toLowerCase().includes(needle))
}

function SeasonProgress({ kind, games }: { kind: EntityKind; games: number | null }) {
  return (
    <p className="flex max-w-prose items-start gap-2 text-sm text-muted-foreground">
      <CalendarClock className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden />
      <span>
        <span className="font-medium text-foreground">Season in progress: up to {games ?? 0} games per team.</span>{' '}
        {kind === 'teams'
          ? 'Until a team has played 9 games its rating also leans on its rating last season, a little less after each game, so ratings will move as the season goes on.'
          : 'Ratings use only the games played so far, so they lean toward the league average and will move as the season goes on.'}
      </span>
    </p>
  )
}

function PageNotes({ kind }: { kind: EntityKind }) {
  const config = entityConfig(kind)
  return (
    <details className="rounded-xl border bg-card px-4 py-3 text-sm">
      <summary className="cursor-pointer font-medium">How to read this page</summary>
      <ul className="mt-2 list-disc space-y-1.5 pl-5 text-muted-foreground">
        <li>Sort any column, switch views, and open a row for its game-by-game page.</li>
        {config.notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </details>
  )
}

function SeasonTable({ kind, data }: { kind: EntityKind; data: EntityTable }) {
  const config = entityConfig(kind)
  const { searchParams, update, request } = useRatingsParams()
  const [query, setQuery] = useState('')
  const [showUnranked, setShowUnranked] = useState(false)
  const state = resolveViewState(kind, data.categories, request)
  const view = seasonView(kind, data.table, state)
  const needle = query.trim().toLowerCase()
  const rows = view.table.rows.filter((row) => (kind !== 'qbs' || showUnranked || row.qb_is_eligible === true) && (needle === '' || matches(kind, row, data.team_names, needle)))
  const table = withDecimals({ ...view.table, rows }, view.columns)
  const search = carriedSearch(searchParams, data.season)
  const nameColumn = config.labelColumn
  const compareIds = (searchParams.get('compare') ?? '').split(',').filter((id) => id !== '' && findEntityRow(kind, data.table.rows, id) !== undefined)
  const toggleCompare = (id: string) =>
    update({ compare: (compareIds.includes(id) ? compareIds.filter((picked) => picked !== id) : [...compareIds, id].slice(0, MAX_COMPARE)).join(',') })
  const renderers: Record<string, CellRenderer> = {
    [nameColumn]: (value, row) => {
      const id = entityId(kind, row)
      const name = String(value ?? id)
      const picked = compareIds.includes(id)
      return (
        <span className="inline-flex items-center gap-2">
          <Checkbox aria-label={`Compare ${name}`} checked={picked} disabled={!picked && compareIds.length >= MAX_COMPARE} onCheckedChange={() => toggleCompare(id)} />
          <Link to={`/${kind}/${encodeURIComponent(id)}${search}`} className="font-medium text-primary underline-offset-4 hover:underline">
            {name}
          </Link>
        </span>
      )
    },
  }
  const compareColumns = view.metricColumns.length > 0 ? view.metricColumns : (data.table.column_groups.ratings ?? [])
  const onChange = (change: ViewChange) => update(viewPatch(change))
  const sortable = view.columns.includes(config.ratingColumn)
  return (
    <div className="flex flex-col gap-3">
      <Card className="gap-0 px-4 py-3">
        <ViewControls kind={kind} state={state} onChange={onChange} />
        <p className="mt-2 text-xs text-muted-foreground">{VIEWS.find((option) => option.id === state.view)?.description(kind)}</p>
      </Card>
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative w-full max-w-xs">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input type="search" aria-label={`Find a ${config.singular}`} placeholder={`Find a ${config.singular}`} className="pl-8" value={query} onChange={(event) => setQuery(event.target.value)} />
        </div>
        {kind === 'qbs' ? (
          <div className="flex items-center gap-2">
            <Switch id="show-unranked" checked={showUnranked} onCheckedChange={setShowUnranked} aria-label="Show QBs below the qualifier" />
            <Label htmlFor="show-unranked">Show QBs below the qualifier</Label>
          </div>
        ) : null}
      </div>
      <DataTable
        key={`${data.season}-${state.view}-${state.category}`}
        table={table}
        columns={view.columns}
        defaultSort={sortable ? { key: config.ratingColumn, desc: true } : undefined}
        renderers={renderers}
        rowKey={(row) => entityId(kind, row)}
        maxHeight="70vh"
        emptyMessage={needle ? `No ${config.singular} matches “${query.trim()}”.` : 'No rows.'}
      />
      <p className="text-xs text-muted-foreground">Tick up to {MAX_COMPARE} rows to compare them side by side on this view’s columns.</p>
      <ComparisonPanel kind={kind} season={data.season} data={data} ids={compareIds} columns={compareColumns} onRemove={toggleCompare} />
    </div>
  )
}

/** The team or quarterback table of a season: ratings, stats and opponent profiles, by view. */
export function RatingsIndexPage({ kind }: { kind: EntityKind }) {
  const config = entityConfig(kind)
  const { query: seasonsQuery, season, seasons, unavailable, setSeason } = useRatingsSeason()
  const tableQuery = useEntityTable(kind, season)
  const data = tableQuery.data
  return (
    <>
      <PageHeader
        title={season === null ? config.title : `${config.title} · ${season}`}
        description={config.ratingDescription}
        actions={
          <>
            <SeasonPicker seasons={seasons} season={season} onChange={setSeason} />
            <Button asChild variant="outline" size="sm">
              <Link to={`/rank-ranges?kind=${kind}${season === null ? '' : `&season=${season}`}`}>
                <ListOrdered /> Rank ranges
              </Link>
            </Button>
            <Button asChild variant="outline" size="sm">
              <Link to={`/head-to-head?kind=${kind}${season === null ? '' : `&season=${season}`}`}>
                <Swords /> Head to head
              </Link>
            </Button>
          </>
        }
      />
      {season === null ? <RatingsSeasonState isLoading={seasonsQuery.isLoading} error={seasonsQuery.error} /> : null}
      <div className="flex flex-col gap-4">
        {unavailable !== null && season !== null ? <UnavailableSeason requested={unavailable} shown={season} /> : null}
        {data?.in_progress ? <SeasonProgress kind={kind} games={data.games_played} /> : null}
        {tableQuery.isLoading ? <Skeleton className="h-96 w-full" /> : null}
        {tableQuery.isError ? <ErrorState error={tableQuery.error} title={`Could not load the ${season ?? ''} ${config.title.toLowerCase()}`} /> : null}
        {data ? <SeasonTable kind={kind} data={data} /> : null}
        {data ? <PageNotes kind={kind} /> : null}
        {data && kind === 'teams' ? (
          <p className="text-sm text-muted-foreground">
            These ratings come from play-by-play: EPA per play, adjusted for every opponent. The model’s own team strengths, fit on game margins, and its projected standings are on{' '}
            <Link to="/power" className="font-medium text-primary underline-offset-4 hover:underline">
              Power Rankings
            </Link>
            .
          </p>
        ) : null}
      </div>
    </>
  )
}
