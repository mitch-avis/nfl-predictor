import { useMemo } from 'react'

import { usePower } from '@/api/queries'
import type { Row, TablePayload } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { MovementChip } from '@/components/power/MovementChip'
import { RatingBar } from '@/components/power/RatingBar'
import { DataTable } from '@/components/table/DataTable'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useQueryParam } from '@/hooks/useQueryParam'

function groupBy(table: TablePayload, key: string): { name: string; table: TablePayload }[] {
  const groups = new Map<string, Row[]>()
  for (const row of table.rows) {
    const name = String(row[key] ?? '—')
    groups.set(name, [...(groups.get(name) ?? []), row])
  }
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([name, rows]) => ({ name, table: { ...table, rows } }))
}

const STANDING_COLUMNS = ['team_abbr', 'record', 'projected_wins', 'projected_losses', 'projected_win_pct', 'exp_wins', 'games_remaining', 'projected_division_rank']

export function PowerRankingsPage() {
  const [run] = useQueryParam('run')
  const query = usePower(run)
  const conferences = useMemo(() => (query.data?.standings ? groupBy(query.data.standings, 'conference') : []), [query.data])
  const divisions = useMemo(() => (query.data?.division_standings ? groupBy(query.data.division_standings, 'division') : []), [query.data])

  return (
    <>
      <PageHeader
        title="Power Rankings"
        description="Bradley-Terry team strength fit on margins over a two-season window (prior season down-weighted), plus projected standings from the model's remaining-schedule win probabilities."
      />
      {query.isLoading ? <Skeleton className="h-64 w-full" /> : null}
      {query.isError ? <ErrorState error={query.error} title="No power rankings for this run" /> : null}
      {query.data ? (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline">
              {query.data.season} · through week {query.data.through_week}
            </Badge>
            <span>Run {query.data.run_id}</span>
            {query.data.previous_run_id ? <span>· movement vs {query.data.previous_run_id}</span> : <span>· no prior week on disk for movement</span>}
          </div>
          <Tabs defaultValue="rankings">
            <TabsList>
              <TabsTrigger value="rankings">Rankings</TabsTrigger>
              <TabsTrigger value="conference" disabled={conferences.length === 0}>Conference standings</TabsTrigger>
              <TabsTrigger value="division" disabled={divisions.length === 0}>Division standings</TabsTrigger>
            </TabsList>
            <TabsContent value="rankings" className="mt-3">
              <DataTable
                table={query.data.rankings}
                columns={['rank', 'team_abbr', 'rank_change', 'power_rating_1_10', 'record', 'division', 'rating_raw', 'power_rating_0_10', 'previous_rank', 'home_advantage_prob']}
                defaultSort={{ key: 'rank' }}
                rowKey={(row) => String(row.team_abbr)}
                renderers={{
                  power_rating_1_10: (value) => <RatingBar value={typeof value === 'number' ? value : null} />,
                  rank_change: (value) => <MovementChip delta={typeof value === 'number' ? value : null} />,
                  rank: (value) => <span className="font-semibold">{String(value)}</span>,
                }}
              />
            </TabsContent>
            <TabsContent value="conference" className="mt-3 grid gap-4 xl:grid-cols-2">
              {conferences.map((group) => (
                <section key={group.name} className="space-y-2">
                  <h2 className="text-sm font-semibold">{group.name}</h2>
                  <DataTable table={group.table} columns={STANDING_COLUMNS.filter((c) => c !== 'projected_division_rank')} defaultSort={{ key: 'projected_win_pct', desc: true }} rowKey={(row) => String(row.team_abbr)} dense />
                </section>
              ))}
            </TabsContent>
            <TabsContent value="division" className="mt-3 grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
              {divisions.map((group) => (
                <section key={group.name} className="space-y-2">
                  <h2 className="text-sm font-semibold">{group.name}</h2>
                  <DataTable table={group.table} columns={['projected_division_rank', 'team_abbr', 'record', 'projected_wins', 'projected_win_pct']} defaultSort={{ key: 'projected_division_rank' }} rowKey={(row) => String(row.team_abbr)} dense heat={false} />
                </section>
              ))}
            </TabsContent>
          </Tabs>
        </div>
      ) : null}
    </>
  )
}
