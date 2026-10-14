import { Link } from 'react-router'

import { useScheduleRanks } from '@/api/queries'
import type { MetricColumnMeta, Row, ScheduleRanks } from '@/api/types'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { SectionCard } from '@/components/ratings/SectionCard'
import { withDecimals } from '@/components/ratings/decimals'
import { entityPath } from '@/components/ratings/entities'
import { decimalsFor, formatMetric } from '@/components/ratings/format'
import { DataTable, type CellRenderer } from '@/components/table/DataTable'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { useQueryParam } from '@/hooks/useQueryParam'

const ALL = 'all'
const LEADERS = 10

function seasonOf(row: Row): number {
  return Number(row.season)
}

function ScheduleList({ title, rows, rankColumn, sos }: { title: string; rows: Row[]; rankColumn: 'softest_rank' | 'hardest_rank'; sos: { meta: MetricColumnMeta | undefined; decimals: number | null } }) {
  const leaders = [...rows].sort((left, right) => Number(left[rankColumn]) - Number(right[rankColumn])).slice(0, LEADERS)
  return (
    <SectionCard title={title}>
      <ol className="flex flex-col gap-1 text-sm">
        {leaders.map((row) => (
          <li key={`${row.season}-${row.team}`} className="grid grid-cols-[2rem_1fr_auto] items-baseline gap-2">
            <span className="tabular text-muted-foreground">{String(row[rankColumn])}.</span>
            <Link to={entityPath('teams', String(row.team), seasonOf(row))} className="font-medium text-primary underline-offset-4 hover:underline">
              {String(row.season)} {String(row.team)}
            </Link>
            <span className="tabular">SoS {formatMetric(sos.meta, row.sos, sos.decimals)}</span>
          </li>
        ))}
      </ol>
    </SectionCard>
  )
}

function Filter({ label, value, options, onChange }: { label: string; value: string | null; options: string[]; onChange: (value: string | null) => void }) {
  const id = `filter-${label.toLowerCase()}`
  return (
    <div className="flex items-center gap-2">
      <Label htmlFor={id}>{label}</Label>
      <Select value={value ?? ALL} onValueChange={(next) => onChange(next === ALL ? null : next)}>
        <SelectTrigger id={id} className="w-[130px]" size="sm" aria-label={label}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>All</SelectItem>
          {options.map((option) => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

function Leaderboard({ data }: { data: ScheduleRanks }) {
  const [team, setTeam] = useQueryParam('team')
  const [season, setSeason] = useQueryParam('season')
  const { rows, column_metadata: metadata } = data.table
  const sos = { meta: metadata.sos, decimals: decimalsFor(metadata.sos, rows, 'sos') }
  const teams = [...new Set(rows.map((row) => String(row.team)))].sort()
  const seasons = [...new Set(rows.map((row) => String(row.season)))].sort().reverse()
  const shown = rows.filter((row) => (team === null || row.team === team) && (season === null || String(row.season) === season))
  const table = withDecimals({ ...data.table, rows: shown }, data.table.visible_columns)
  const renderers: Record<string, CellRenderer> = {
    team: (value, row) => (
      <Link to={entityPath('teams', String(value), seasonOf(row))} className="font-medium text-primary underline-offset-4 hover:underline">
        {String(value)}
      </Link>
    ),
  }
  const first = data.seasons[0]
  const last = data.seasons.at(-1)
  const skipped = data.skipped_seasons
  return (
    <div className="flex flex-col gap-5">
      <p className="text-sm text-muted-foreground">
        {data.count} team-seasons, {first === last ? first : `${first}–${last}`}.{' '}
        {skipped.length > 0 ? (skipped.length === 1 ? `${skipped[0]} is left out until it is complete.` : `${skipped.join(', ')} are left out until they are complete.`) : null}
      </p>
      <div className="grid gap-5 md:grid-cols-2">
        <ScheduleList title="Softest schedules" rows={rows} rankColumn="softest_rank" sos={sos} />
        <ScheduleList title="Hardest schedules" rows={rows} rankColumn="hardest_rank" sos={sos} />
      </div>
      <SectionCard title="Every schedule" description="Rank 1 is the softest (or hardest) schedule of every completed team-season. Open a team for that season’s page.">
        <div className="flex flex-wrap items-center gap-4">
          <Filter label="Team" value={team} options={teams} onChange={setTeam} />
          <Filter label="Season" value={season} options={seasons} onChange={setSeason} />
        </div>
        <DataTable table={table} defaultSort={{ key: 'softest_rank' }} renderers={renderers} rowKey={(row) => `${row.season}-${row.team}`} maxHeight="70vh" emptyMessage="No team-season matches these filters." />
      </SectionCard>
    </div>
  )
}

/** Why the ranking is empty: no team ratings at all (run the ETL), or only seasons still in progress. */
function EmptyRanking({ skipped }: { skipped: number[] }) {
  const description =
    skipped.length === 0
      ? 'The ranking reads each completed season’s team ratings, {season}_ratings.parquet in data/ratings/, and that folder holds none yet. The ETL writes them (nfl-predictor data, or the Full ETL rebuild on the Jobs page); run it, then reload.'
      : `The ranking reads each completed season’s team ratings, and ${skipped.join(', ')} ${skipped.length === 1 ? 'is' : 'are'} still in progress, left out until complete. The ETL writes each season’s ratings again on every run.`
  return <EmptyState title="No completed seasons to rank yet" description={description} />
}

/** Every completed team-season ranked on schedule strength, across every season with ratings. */
export function SchedulesPage() {
  const query = useScheduleRanks()
  return (
    <>
      <PageHeader
        title="Schedules"
        description="Every completed team-season ranked on schedule strength (SoS): the average Team Rating of the opponents played, one entry per game, each opponent rated without its games against that team. Higher is a harder schedule."
      />
      {query.isLoading ? <Skeleton className="h-96 w-full" /> : null}
      {query.isError ? <ErrorState error={query.error} title="Could not load the schedule rankings" /> : null}
      {query.data && query.data.count === 0 ? <EmptyRanking skipped={query.data.skipped_seasons} /> : null}
      {query.data && query.data.count > 0 ? <Leaderboard data={query.data} /> : null}
    </>
  )
}
