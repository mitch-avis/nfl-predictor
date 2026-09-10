import { ArrowDown, ArrowUp, ChevronDown } from 'lucide-react'
import { useState } from 'react'

import { useModel } from '@/api/queries'
import type { MetricStrategyEntry } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { InfoTooltip } from '@/components/common/InfoTooltip'
import { PageHeader } from '@/components/common/PageHeader'
import { StatTile } from '@/components/common/StatTile'
import { CalibrationChart } from '@/components/model/CalibrationChart'
import { FeatureImportanceChart } from '@/components/model/FeatureImportanceChart'
import { DataTable } from '@/components/table/DataTable'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useRegistry } from '@/api/queries'
import { useQueryParam } from '@/hooks/useQueryParam'
import { formatNumber, formatPercent, formatRelative, humanize, shortHash } from '@/utils/format'

const METRIC_ORDER = ['brier', 'log_loss', 'reliability_ece', 'winner_accuracy', 'pick_accuracy', 'margin_mae', 'total_mae', 'home_mae', 'away_mae']

function directionFor(metric: string, strategy: Record<string, MetricStrategyEntry[]> | null | undefined): 'higher' | 'lower' | null {
  if (!strategy) return null
  for (const entries of Object.values(strategy)) {
    const hit = entries.find((e) => e.metric === metric)
    if (hit) return hit.direction
  }
  return null
}

function MetricTiles({ metrics, strategy, title }: { metrics: Record<string, number>; strategy: Record<string, MetricStrategyEntry[]> | null | undefined; title: string }) {
  const registry = useRegistry()
  const keys = [...METRIC_ORDER.filter((k) => k in metrics), ...Object.keys(metrics).filter((k) => !METRIC_ORDER.includes(k) && !['games', 'weeks', 'picks_correct', 'expected_points', 'actual_points'].includes(k))]
  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold">{title}</h2>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {keys.map((key) => {
          const meta = registry.data?.columns[key]
          const direction = meta?.polarity && meta.polarity !== 'neutral' ? meta.polarity : directionFor(key, strategy)
          const value = metrics[key]
          const isPct = meta?.kind === 'prob' || key.includes('accuracy')
          return (
            <StatTile
              key={key}
              label={meta?.label ?? humanize(key)}
              value={isPct ? formatPercent(value, 1) : formatNumber(value, meta?.decimals ?? (Math.abs(value) < 1 ? 4 : 2))}
              hint={meta?.description}
              footnote={
                direction ? (
                  <span className="inline-flex items-center gap-1">
                    {direction === 'lower' ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" />}
                    {direction === 'lower' ? 'lower is better' : 'higher is better'}
                  </span>
                ) : undefined
              }
            />
          )
        })}
      </div>
    </section>
  )
}

function KeyValue({ items }: { items: [string, string | number | null | undefined][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {items.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted-foreground">{k}</dt>
          <dd className="truncate font-mono text-xs leading-6" title={v === null || v === undefined ? undefined : String(v)}>
            {v === null || v === undefined ? '—' : String(v)}
          </dd>
        </div>
      ))}
    </dl>
  )
}

function JsonBlock({ title, value }: { title: string; value: unknown }) {
  const [open, setOpen] = useState(false)
  if (value === null || value === undefined) return null
  return (
    <div className="rounded-lg border">
      <Button variant="ghost" size="sm" className="w-full justify-between" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {title}
        <ChevronDown className={open ? 'size-4 rotate-180 transition-transform' : 'size-4 transition-transform'} />
      </Button>
      {open ? <pre className="max-h-96 overflow-auto border-t bg-muted/40 p-3 text-xs">{JSON.stringify(value, null, 2)}</pre> : null}
    </div>
  )
}

export function ModelPage() {
  const [run] = useQueryParam('run')
  const query = useModel(run)
  const data = query.data
  const config = data?.metadata.config ?? {}

  return (
    <>
      <PageHeader title="Model" description="What the active model is, how it was trained, how it scored, and which features it leans on." />
      {query.isLoading ? <Skeleton className="h-64 w-full" /> : null}
      {query.isError ? <ErrorState error={query.error} title="No model to show" /> : null}
      {data ? (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline">{data.run_id}</Badge>
            <Badge variant="secondary">{data.kind.replace('_', ' ')}</Badge>
            <span>created {formatRelative(data.metadata.created_at)}</span>
          </div>
          {data.metrics.holdout ? <MetricTiles metrics={data.metrics.holdout} strategy={data.metrics.metric_strategy} title="Holdout metrics" /> : null}
          {data.metrics.overall ? <MetricTiles metrics={data.metrics.overall} strategy={data.metrics.metric_strategy} title="Walk-forward metrics" /> : null}
          {data.metrics.pool ? (
            <div className="grid gap-3 sm:grid-cols-3">
              <StatTile label="Pool weeks" value={formatNumber(data.metrics.pool.weeks, 0)} hint="Weeks scored under confidence-pool rules." />
              <StatTile label="Picks correct / wk" value={formatNumber(data.metrics.pool.weekly_picks_correct_avg, 2)} hint="Average number of correct picks per week." />
              <StatTile label="Pool pts / wk" value={`${formatNumber(data.metrics.pool.weekly_actual_points_avg, 1)} of ${formatNumber(data.metrics.pool.weekly_expected_points_avg, 1)} exp.`} hint="Realized confidence points per week versus the expectation implied by the probabilities. A large gap means overconfidence." />
            </div>
          ) : null}
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-1 text-base">
                  Feature importance <InfoTooltip content="XGBoost gain, summed across the margin and total heads. Higher means the feature contributed more to the trees' splits." />
                </CardTitle>
                <CardDescription>Top {Math.min(25, data.feature_importance.length)} of {data.metadata.feature_count ?? '?'} features.</CardDescription>
              </CardHeader>
              <CardContent>
                <FeatureImportanceChart rows={data.feature_importance} />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-1 text-base">
                  Calibration <InfoTooltip content="Reliability diagram: for games predicted at each probability, how often the home team actually won. Points on the diagonal are perfectly calibrated." />
                </CardTitle>
                <CardDescription>{data.calibration ? `${data.calibration.bin_count} bins${data.calibration.source ? ` · ${data.calibration.source}` : ''}` : 'Only available for runs with a walk-forward report.'}</CardDescription>
              </CardHeader>
              <CardContent>{data.calibration ? <CalibrationChart bins={data.calibration.bins} /> : <div className="text-sm text-muted-foreground">No calibration data for this run.</div>}</CardContent>
            </Card>
          </div>
          {data.wf_compare ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Walk-forward candidates</CardTitle>
                <CardDescription>Every configuration the weekly run compared; the top row was trained as the final model.</CardDescription>
              </CardHeader>
              <CardContent>
                <DataTable table={data.wf_compare} defaultSort={{ key: 'wf_rank' }} dense rowClassName={(row) => (row.wf_rank === 1 ? 'bg-accent/40' : undefined)} />
              </CardContent>
            </Card>
          ) : null}
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Provenance</CardTitle>
              </CardHeader>
              <CardContent>
                <KeyValue
                  items={[
                    ['Model kind', String(config.model_kind ?? '—')],
                    ['Calibration', String(config.win_prob_calibration ?? config.calibration ?? '—')],
                    ['Market blend', config.market_prob_blend !== undefined ? String(config.market_prob_blend) : '—'],
                    ['Score rounding', String(config.score_rounding ?? '—')],
                    ['Features', data.metadata.feature_count ?? null],
                    ['Git commit', shortHash(data.metadata.git_commit_hash, 12)],
                    ['Dataset hash', shortHash(data.metadata.dataset_hash, 12)],
                    ['Data path', String(config.data_path ?? '—')],
                    ['XGBoost', data.metadata.library_versions?.xgboost ?? null],
                    ['Python', data.metadata.library_versions?.python ?? null],
                  ]}
                />
              </CardContent>
            </Card>
            <div className="space-y-2">
              <JsonBlock title="Training config" value={data.metadata.config} />
              <JsonBlock title="Model parameters" value={data.metadata.params} />
              <JsonBlock title="Tuned parameters" value={data.metadata.tuned_params} />
              <JsonBlock title="Splits" value={data.metadata.splits} />
              <JsonBlock title="Early stopping" value={data.metadata.early_stopping} />
              <JsonBlock title="Missing data by group" value={data.metrics.missing_data} />
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}
