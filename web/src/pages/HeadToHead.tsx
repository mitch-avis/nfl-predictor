import { ArrowLeftRight } from 'lucide-react'
import { useMemo } from 'react'
import { Link } from 'react-router'

import { isMissing, useEntitySection, useEntityTable, useRankRanges } from '@/api/queries'
import type { EntityKind, EntityTable, Row } from '@/api/types'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { RatingsSeasonState } from '@/components/ratings/RatingsStates'
import { SeasonPicker } from '@/components/ratings/SeasonPicker'
import { SectionCard } from '@/components/ratings/SectionCard'
import { entityConfig, entityId, entityLabel, entityPath, findEntityRow } from '@/components/ratings/entities'
import { decimalsFor, formatMetric } from '@/components/ratings/format'
import { formatChance, middleRankText, ordinal, parseRankRanges, type RankRange } from '@/components/ratings/rankRanges'
import { describeRatingPair, neighborId, parseRatingPairs, signed, type RatingPair } from '@/components/ratings/ratingPairs'
import { summaryTiles } from '@/components/ratings/summary'
import { useRatingsParams, useRatingsSeason } from '@/components/ratings/useRatingsParams'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { columnStats, heatStyle } from '@/utils/cells'

/** The rows that can be compared, best first: every team, or the qualifying quarterbacks. */
function candidates(kind: EntityKind, data: EntityTable): Row[] {
  const rating = entityConfig(kind).ratingColumn
  return data.table.rows
    .filter((row) => typeof row[rating] === 'number' && (kind !== 'qbs' || row.qb_is_eligible === true))
    .sort((left, right) => Number(right[rating]) - Number(left[rating]))
}

function EntitySelect({ label, value, rows, kind, names, onChange }: { label: string; value: string; rows: Row[]; kind: EntityKind; names: Record<string, string>; onChange: (id: string) => void }) {
  const options = [...rows].sort((left, right) => entityLabel(kind, left, names).localeCompare(entityLabel(kind, right, names)))
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="w-64 max-w-full" aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((row) => (
          <SelectItem key={entityId(kind, row)} value={entityId(kind, row)}>
            {entityLabel(kind, row, names)}
            {kind === 'qbs' ? ` · ${String(row.team ?? '')}` : ''}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** The rating gap's 95% range and median on an axis centered on even (0); decorative. */
function GapBar({ pair, unit }: { pair: RatingPair; unit: string }) {
  if (pair.gapLow === null || pair.gapHigh === null || pair.gapMid === null) return null
  const extent = Math.max(Math.abs(pair.gapLow), Math.abs(pair.gapHigh)) * 1.1 || 1
  const at = (value: number) => `${((value + extent) / (2 * extent)) * 100}%`
  return (
    <div className="flex flex-col gap-1" aria-hidden>
      <div className="relative h-6 w-full">
        <span className="absolute inset-y-0 w-px bg-muted-foreground/60" style={{ left: '50%' }} />
        <span className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-chart-1/60" style={{ left: at(pair.gapLow), width: `calc(${at(pair.gapHigh)} - ${at(pair.gapLow)})` }} />
        <span className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-chart-1 ring-2 ring-card" style={{ left: at(pair.gapMid) }} />
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>Second rated higher</span>
        <span>Even</span>
        <span>First rated higher</span>
      </div>
      <p className="text-xs text-muted-foreground">
        Rating gap, {unit}: the bar spans the middle 95% of redraws ({signed(pair.gapLow, 2)} to {signed(pair.gapHigh, 2)}), the dot the typical gap.
      </p>
    </div>
  )
}

function ComparisonTable({ kind, season, data, ids, ranges }: { kind: EntityKind; season: number; data: EntityTable; ids: [string, string]; ranges: RankRange[] }) {
  const { rows, column_metadata: metadata } = data.table
  const pairRows = ids.map((id) => findEntityRow(kind, rows, id))
  const config = entityConfig(kind)
  const columns = [...config.summaryColumns, config.unadjustedColumn].filter((column) => column in metadata)
  const stats = columnStats(rows, columns)
  const rangeOf = (id: string) => ranges.find((range) => range.id === id)
  return (
    <div className="overflow-x-auto rounded-xl border">
      <table className="w-full min-w-max text-sm">
        <caption className="sr-only">Side by side</caption>
        <thead className="bg-muted/50">
          <tr>
            <th scope="col" className="px-3 py-2 text-left font-medium text-muted-foreground">
              Metric
            </th>
            {ids.map((id, index) => (
              <th key={`${id}-${index}`} scope="col" className="px-3 py-2 text-right font-medium">
                <Link to={entityPath(kind, id, season)} className="text-primary underline-offset-4 hover:underline">
                  {pairRows[index] ? entityLabel(kind, pairRows[index], data.team_names) : id}
                </Link>
                {rangeOf(id) ? <div className="text-xs font-normal text-muted-foreground">Rank {ordinal(rangeOf(id)?.publishedRank ?? 0)} · middle 50% {middleRankText(rangeOf(id) as RankRange)}</div> : null}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {columns.map((column) => {
            const meta = metadata[column]
            const decimals = decimalsFor(meta, rows, column)
            return (
              <tr key={column} className="border-t">
                <th scope="row" className="px-3 py-1.5 text-left font-medium">
                  {meta?.label ?? column}
                </th>
                {pairRows.map((row, index) => {
                  const tile = row ? summaryTiles(kind, row, rows, metadata).find((candidate) => candidate.column === column) : undefined
                  const value = row?.[column] ?? null
                  return (
                    <td key={`${column}-${index}`} className="tabular px-3 py-1.5 text-right" style={meta ? heatStyle(meta, value, stats[column]) : undefined}>
                      {formatMetric(meta, value, decimals)}
                      {tile?.rank ? <span className="ml-1 text-xs text-muted-foreground">({ordinal(tile.rank.rank)})</span> : null}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Matchup({ kind, season, data }: { kind: EntityKind; season: number; data: EntityTable }) {
  const { searchParams, update } = useRatingsParams()
  const ranked = useMemo(() => candidates(kind, data), [data, kind])
  const ids = ranked.map((row) => entityId(kind, row))
  const askedA = searchParams.get('a')
  const a = askedA !== null && ids.includes(askedA) ? askedA : (ids[0] ?? null)
  const askedB = searchParams.get('b')
  const b = askedB !== null && ids.includes(askedB) && askedB !== a ? askedB : a === null ? null : neighborId(ids, a)
  const pairs = useEntitySection('head-to-head', kind, season, a)
  const rankRanges = useRankRanges(kind, season)
  const ranges = useMemo(() => (rankRanges.data ? parseRankRanges(kind, rankRanges.data) : []), [kind, rankRanges.data])
  if (a === null || b === null) return <EmptyState title="Nothing to compare" description="The season has fewer than two rated entries." />

  const labelOf = (id: string) => {
    const row = findEntityRow(kind, data.table.rows, id)
    return row ? entityLabel(kind, row, data.team_names) : id
  }
  const pair = pairs.data ? parseRatingPairs(kind, pairs.data).find((candidate) => candidate.otherId === b) : undefined
  const noun = kind === 'teams' ? 'team' : 'quarterback'
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-2">
        <EntitySelect label={`First ${noun}`} value={a} rows={ranked} kind={kind} names={data.team_names} onChange={(id) => update({ a: id, b: id === b ? null : b })} />
        <Button type="button" variant="ghost" size="icon" aria-label="Swap the two" onClick={() => update({ a: b, b: a })}>
          <ArrowLeftRight />
        </Button>
        <EntitySelect label={`Second ${noun}`} value={b} rows={ranked.filter((row) => entityId(kind, row) !== a)} kind={kind} names={data.team_names} onChange={(id) => update({ a, b: id })} />
      </div>
      <SectionCard
        title="Chance of being rated higher"
        description={`How often the first ${noun} was rated above the second across 1,000 redraws of the ${season} season, both re-rated in every redraw${kind === 'qbs' ? ', counting only redraws with both' : ''}.`}
      >
        {pairs.isLoading ? <Skeleton className="h-24 w-full" /> : null}
        {pairs.isError && isMissing(pairs.error) ? <p className="text-sm text-muted-foreground">This season was built without head-to-head chances.</p> : null}
        {pairs.isError && !isMissing(pairs.error) ? <ErrorState error={pairs.error} title="Could not load the head-to-head chances" /> : null}
        {pair ? (
          <>
            {pair.aboveChance !== null ? <p className="tabular text-3xl font-semibold">{formatChance(pair.aboveChance)}</p> : null}
            <p className="text-sm">{describeRatingPair(kind, labelOf(a), labelOf(b), pair)}</p>
            <GapBar pair={pair} unit={kind === 'teams' ? 'points per game' : 'EPA per dropback'} />
          </>
        ) : null}
      </SectionCard>
      <SectionCard title="Side by side" description="The published ratings with each one’s rank in the season; shading compares them with the whole league.">
        <ComparisonTable kind={kind} season={season} data={data} ids={[a, b]} ranges={ranges} />
      </SectionCard>
    </div>
  )
}

/** Two teams or two quarterbacks compared: how often one was rated above the other, and their ratings. */
export function HeadToHeadPage() {
  const { searchParams, update } = useRatingsParams()
  const kind: EntityKind = searchParams.get('kind') === 'qbs' ? 'qbs' : 'teams'
  const { query: seasonsQuery, season, seasons, setSeason } = useRatingsSeason()
  const table = useEntityTable(kind, season)
  return (
    <>
      <PageHeader
        title={season === null ? 'Head to head' : `Head to head · ${season}`}
        description="Pick two teams or two quarterbacks to see how often one came out ahead of the other when the season’s games were redrawn, and their ratings side by side."
        actions={<SeasonPicker seasons={seasons} season={season} onChange={setSeason} />}
      />
      {season === null ? <RatingsSeasonState isLoading={seasonsQuery.isLoading} error={seasonsQuery.error} /> : null}
      {season !== null ? (
        <div className="flex flex-col gap-4">
          <Tabs value={kind} onValueChange={(value) => update({ kind: value === 'teams' ? null : value, a: null, b: null })}>
            <TabsList>
              <TabsTrigger value="teams">Teams</TabsTrigger>
              <TabsTrigger value="qbs">Quarterbacks</TabsTrigger>
            </TabsList>
          </Tabs>
          {table.isLoading ? <Skeleton className="h-64 w-full" /> : null}
          {table.isError ? <ErrorState error={table.error} title={`Could not load the ${season} ${kind === 'teams' ? 'teams' : 'quarterbacks'}`} /> : null}
          {table.data ? <Matchup kind={kind} season={season} data={table.data} /> : null}
        </div>
      ) : null}
    </>
  )
}
