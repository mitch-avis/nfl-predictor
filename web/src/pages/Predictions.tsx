import { LayoutGrid, Table2, Wand2 } from 'lucide-react'
import { useMemo, useState } from 'react'

import { ApiError } from '@/api/client'
import { usePicks, usePredictions, useWeeks } from '@/api/queries'
import type { Row, WeekRef } from '@/api/types'
import { CopyButton } from '@/components/common/CopyButton'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { StatTile } from '@/components/common/StatTile'
import { RunJobButton } from '@/components/jobs/RunJobButton'
import { WeekSelector, weekKey } from '@/components/common/WeekSelector'
import { MatchupCard } from '@/components/predictions/MatchupCard'
import { WinProbBar } from '@/components/predictions/WinProbBar'
import { ColumnPicker, useColumnGroups } from '@/components/table/ColumnPicker'
import { DataTable } from '@/components/table/DataTable'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useIsMobile } from '@/hooks/use-mobile'
import { useNumberParam, useQueryParam } from '@/hooks/useQueryParam'
import { cn } from '@/utils/cn'
import { formatDateTime, formatNumber, formatPercent, formatRelative } from '@/utils/format'

/** A week the model has not predicted yet: offer to generate it rather than showing an error. */
function NotPredictedYet({ season, week }: { season: number | null; week: number | null }) {
  return (
    <EmptyState
      title={`Week ${week ?? '?'} has no predictions yet`}
      description={
        <>
          The games are in the dataset, so the active model can predict them now. A future week
          keeps the market lines, rest, and quarterbacks of the last ETL run, so refresh the lines
          and predict again once the week is close.
        </>
      }
      action={
        <RunJobButton
          templateId="predict_week"
          label="Generate predictions"
          icon={Wand2}
          variant="default"
          preset={season !== null && week !== null ? { season, week } : undefined}
        />
      }
    />
  )
}

function useWeekParams() {
  const [run, setRun] = useQueryParam('run')
  const [season, setSeason] = useNumberParam('season')
  const [week, setWeek] = useNumberParam('week')
  const [source, setSource] = useQueryParam('source')
  const select = (ref: WeekRef | null) => {
    setRun(ref?.source === 'run' ? ref.run_id : null)
    setSeason(ref && ref.source !== 'run' ? ref.season : null)
    setWeek(ref && ref.source !== 'run' ? ref.week : null)
    setSource(ref?.source === 'unattached' ? 'unattached' : null)
  }
  return { params: { run, season, week, source }, select }
}

function PicksList({ params }: { params: { run: string | null; season: number | null; week: number | null } }) {
  const picks = usePicks(params)
  if (picks.isLoading) return <Skeleton className="h-40 w-full" />
  if (picks.isError) return <ErrorState error={picks.error} />
  if (!picks.data) return null
  const rows = picks.data.table.rows
  const text = rows.map((r) => `${r.confidence_rank}\t${r.predicted_winner}\t${r.away_abbr} @ ${r.home_abbr}\t${formatPercent(Math.max(Number(r.home_win_prob), Number(r.away_win_prob)), 1)}`).join('\n')
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground">
          Assign the points shown to each pick. Highest confidence first; {rows.length} games → max {rows.length * (rows.length + 1) / 2} points.
        </p>
        <CopyButton text={text} label="Copy picks" />
      </div>
      <ol className="divide-y rounded-xl border bg-card">
        {rows.map((r) => {
          const homeProb = Number(r.home_win_prob)
          const pickProb = String(r.predicted_winner) === String(r.home_abbr) ? homeProb : 1 - homeProb
          return (
            <li key={String(r.game_id)} className="flex items-center gap-3 px-3 py-2">
              <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">{String(r.confidence_rank)}</span>
              <div className="min-w-0 flex-1">
                <div className="font-medium">
                  {String(r.predicted_winner)} <span className="text-xs font-normal text-muted-foreground">over {String(r.predicted_winner) === String(r.home_abbr) ? String(r.away_abbr) : String(r.home_abbr)}</span>
                </div>
                <div className="text-xs text-muted-foreground">{String(r.away_abbr)} @ {String(r.home_abbr)}</div>
              </div>
              <div className="w-28 shrink-0 sm:w-40">
                <WinProbBar homeProb={homeProb} away={String(r.away_abbr)} home={String(r.home_abbr)} />
              </div>
              <span className={cn('tabular w-14 text-right text-sm font-semibold', pickProb >= 0.65 ? 'text-success' : pickProb < 0.55 ? 'text-muted-foreground' : '')}>{formatPercent(pickProb, 1)}</span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

export function PredictionsPage() {
  const { params, select } = useWeekParams()
  const query = usePredictions(params)
  const weeksQuery = useWeeks()
  const isMobile = useIsMobile()
  const [view, setView] = useState<'auto' | 'cards' | 'table'>('auto')
  const showCards = view === 'cards' || (view === 'auto' && isMobile)
  const groups = useColumnGroups('predictions', query.data?.table, ['Uncertainty', 'Context'])
  const weeks = query.data?.weeks ?? weeksQuery.data ?? []
  const notPredicted = query.isError && query.error instanceof ApiError && query.error.code === 'no_predictions'
  const current = query.data
    ? weekKey({ season: query.data.season, week: query.data.week, source: query.data.source, run_id: query.data.run_id })
    : notPredicted && params.week !== null
      ? weekKey({ season: params.season, week: params.week, source: 'available' })
      : null
  const rows = useMemo<Row[]>(() => query.data?.table.rows ?? [], [query.data])

  return (
    <>
      <PageHeader
        title="Predictions"
        description="Calibrated win probabilities, predicted scores, and how they compare with the market lines captured at the last data refresh."
        actions={
          <>
            <WeekSelector weeks={weeks} value={current} onChange={select} />
            <div className="inline-flex rounded-md border">
              <Button variant={showCards ? 'secondary' : 'ghost'} size="sm" aria-label="Card view" onClick={() => setView('cards')}>
                <LayoutGrid className="size-4" />
              </Button>
              <Button variant={!showCards ? 'secondary' : 'ghost'} size="sm" aria-label="Table view" onClick={() => setView('table')}>
                <Table2 className="size-4" />
              </Button>
            </div>
          </>
        }
      />
      {query.isLoading ? <Skeleton className="h-64 w-full" /> : null}
      {notPredicted ? <NotPredictedYet season={params.season} week={params.week} /> : null}
      {query.isError && !notPredicted ? <ErrorState error={query.error} title="No predictions to show" /> : null}
      {query.data ? (
        <div className="space-y-5">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline">{query.data.season} · Week {query.data.week}</Badge>
            <span>
              {query.data.source === 'unattached' ? 'Unattached file in data/predict' : `Run ${query.data.run_id}`} · generated {formatRelative(query.data.generated_at)}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <StatTile label="Games" value={query.data.summary.games} footnote={query.data.summary.first_kickoff ? `${formatDateTime(query.data.summary.first_kickoff)} → ${formatDateTime(query.data.summary.last_kickoff)}` : undefined} />
            <StatTile label="Avg. confidence" value={formatNumber(query.data.summary.avg_confidence, 3)} hint="Mean of |p − 0.5| across the slate. Higher weeks have clearer favorites." />
            <StatTile label="Disagrees with market" value={query.data.summary.market_disagreements} hint="Games where the model's pick differs from the moneyline favorite. These decide whether the model earns its keep." />
            <StatTile label="Games with lines" value={`${query.data.summary.games_with_lines}/${query.data.summary.games}`} hint="Games that had a moneyline when the data was refreshed." />
          </div>
          <Tabs defaultValue="games">
            <TabsList>
              <TabsTrigger value="games">Games</TabsTrigger>
              <TabsTrigger value="picks">Confidence picks</TabsTrigger>
            </TabsList>
            <TabsContent value="games" className="mt-3">
              {showCards ? (
                <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
                  {rows.map((row) => (
                    <MatchupCard key={String(row.game_id)} row={row} />
                  ))}
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex justify-end">
                    <ColumnPicker groups={groups.groups} hidden={groups.hidden} onToggle={groups.toggle} onReset={groups.reset} />
                  </div>
                  <DataTable
                    table={query.data.table}
                    columns={groups.columns}
                    defaultSort={{ key: 'game_datetime' }}
                    rowKey={(row) => String(row.game_id)}
                    rowClassName={(row) => (row.agrees_with_market === false ? 'bg-warning/10' : undefined)}
                    renderers={{
                      predicted_winner: (value, row) => (
                        <span className={cn('font-semibold', row.agrees_with_market === false && 'text-warning-foreground')}>{String(value ?? '—')}</span>
                      ),
                    }}
                  />
                  <p className="text-xs text-muted-foreground">Rows tinted amber disagree with the market favorite. Hover any header for what the column means.</p>
                </div>
              )}
            </TabsContent>
            <TabsContent value="picks" className="mt-3">
              <PicksList params={params} />
            </TabsContent>
          </Tabs>
        </div>
      ) : null}
    </>
  )
}
