import { Link } from 'react-router'

import { ApiError } from '@/api/client'
import { picksCsvUrl, useConfidenceSheet, useTiebreakers, useWeeks } from '@/api/queries'
import type { WeekParams } from '@/api/queries'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { WeekSelector } from '@/components/common/WeekSelector'
import { weekKey } from '@/components/common/weekKey'
import { ConfidenceSheetView } from '@/components/pool/ConfidenceSheetView'
import { SurvivorPanel } from '@/components/pool/SurvivorPanel'
import { TiebreakerPanel } from '@/components/pool/TiebreakerPanel'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useQueryParam } from '@/hooks/useQueryParam'
import { useWeekParams } from '@/hooks/useWeekParams'
import { formatRelative } from '@/utils/format'

const TABS = ['confidence', 'tiebreakers', 'survivor'] as const
type PoolTab = (typeof TABS)[number]

function isNotPredicted(error: unknown): boolean {
  return error instanceof ApiError && error.code === 'no_predictions'
}

function NotPredicted({ week }: { week: number | null }) {
  return (
    <EmptyState
      title={week === null ? 'No predictions yet' : `Week ${week} has no predictions yet`}
      description={
        <>
          Predict the week first: the{' '}
          <Link to="/predictions" className="underline underline-offset-2">
            Predictions page
          </Link>{' '}
          can generate it from the active model.
        </>
      }
    />
  )
}

function TiebreakersTab({ params }: { params: WeekParams }) {
  const query = useTiebreakers(params)
  if (query.isLoading) return <Skeleton className="h-64 w-full" />
  if (query.isError) return isNotPredicted(query.error) ? <NotPredicted week={params.week ?? null} /> : <ErrorState error={query.error} title="No tiebreakers" />
  if (!query.data) return null
  return <TiebreakerPanel key={`${query.data.season}-${query.data.week}-${query.data.run_id}`} card={query.data} />
}

/** Pool helpers for the selected week: the confidence sheet, the tiebreakers and survivor plans. */
export function PoolPage() {
  const { params, select } = useWeekParams()
  const weeksQuery = useWeeks()
  const sheet = useConfidenceSheet(params)
  const [tabParam, setTab] = useQueryParam('tab')
  const tab: PoolTab = TABS.find((value) => value === tabParam) ?? 'confidence'
  const notPredicted = sheet.isError && isNotPredicted(sheet.error)
  const data = sheet.data
  const current = data
    ? weekKey({ season: data.season, week: data.week, source: data.source, run_id: data.run_id })
    : notPredicted && params.week !== null
      ? weekKey({ season: params.season, week: params.week, source: 'available' })
      : null
  const season = data?.season ?? params.season
  const week = data?.week ?? params.week

  return (
    <>
      <PageHeader
        title="Pool helpers"
        description="What to enter in a confidence pool, its tiebreakers and a survivor pool, from the model’s predictions. They only rearrange the predictions; no probability or rank changes."
        actions={<WeekSelector weeks={weeksQuery.data ?? []} value={current} onChange={select} />}
      />
      {data ? (
        <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="outline">
            {data.season} · Week {data.week}
          </Badge>
          <span>
            {data.source === 'unattached' ? 'Unattached file in data/predict' : `Run ${data.run_id}`} · predictions written {formatRelative(data.generated_at)}
          </span>
        </div>
      ) : null}
      <Tabs value={tab} onValueChange={(value) => setTab(value === 'confidence' ? null : value)}>
        <TabsList>
          <TabsTrigger value="confidence">Confidence sheet</TabsTrigger>
          <TabsTrigger value="tiebreakers">Tiebreakers</TabsTrigger>
          <TabsTrigger value="survivor">Survivor</TabsTrigger>
        </TabsList>
        <TabsContent value="confidence" className="mt-3">
          {sheet.isLoading ? <Skeleton className="h-64 w-full" /> : null}
          {notPredicted ? <NotPredicted week={params.week} /> : null}
          {sheet.isError && !notPredicted ? <ErrorState error={sheet.error} title="No confidence sheet" /> : null}
          {data ? <ConfidenceSheetView sheet={data} csvUrl={picksCsvUrl(params)} /> : null}
        </TabsContent>
        <TabsContent value="tiebreakers" className="mt-3">
          <TiebreakersTab params={params} />
        </TabsContent>
        <TabsContent value="survivor" className="mt-3">
          {/* Keyed by season: the used teams are stored per season and read when the panel mounts. */}
          <SurvivorPanel key={season ?? 'none'} season={season} week={week} />
        </TabsContent>
      </Tabs>
    </>
  )
}
