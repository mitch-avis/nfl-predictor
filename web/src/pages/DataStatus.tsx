import { CheckCircle2, Database, XCircle } from 'lucide-react'

import { useDataStatus } from '@/api/queries'
import { ErrorState } from '@/components/common/ErrorState'
import { InfoTooltip } from '@/components/common/InfoTooltip'
import { PageHeader } from '@/components/common/PageHeader'
import { StatTile } from '@/components/common/StatTile'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatNumber, formatRelative, shortHash } from '@/utils/format'

function bytes(size: number | null): string {
  if (size === null) return '—'
  if (size > 1e9) return `${(size / 1e9).toFixed(2)} GB`
  if (size > 1e6) return `${(size / 1e6).toFixed(1)} MB`
  if (size > 1e3) return `${(size / 1e3).toFixed(0)} KB`
  return `${size} B`
}

function SeasonStrip({ label, seasons, hint }: { label: string; seasons: number[]; hint: string }) {
  if (seasons.length === 0) return <div className="text-sm text-muted-foreground">{label}: none cached</div>
  const min = Math.min(...seasons)
  const max = Math.max(...seasons)
  const all = Array.from({ length: max - min + 1 }, (_, i) => min + i)
  return (
    <div>
      <div className="mb-1 flex items-center gap-1 text-xs font-medium text-muted-foreground">
        {label} <InfoTooltip content={hint} />
        <span className="ml-auto">{min}–{max}</span>
      </div>
      <div className="flex gap-0.5">
        {all.map((season) => (
          <span key={season} title={String(season)} className={seasons.includes(season) ? 'h-2 flex-1 rounded-sm bg-primary' : 'h-2 flex-1 rounded-sm bg-muted'} />
        ))}
      </div>
    </div>
  )
}

export function DataStatusPage() {
  const query = useDataStatus()
  const main = query.data?.files.find((f) => f.name === 'all_data_ml.csv')
  const training = query.data?.files.find((f) => f.name === 'completed_games_ml.csv')

  return (
    <>
      <PageHeader title="Data & ETL" description="What the pipeline last produced, how fresh it is, and whether the leakage audit is clean. Refresh and job controls arrive in phase 2." />
      {query.isLoading ? <Skeleton className="h-64 w-full" /> : null}
      {query.isError ? <ErrorState error={query.error} /> : null}
      {query.data ? (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <StatTile label="Current" value={`${query.data.current_season} · Wk ${query.data.current_week}`} hint="Season and week from the ETL's calendar rules (weeks start Tuesday)." />
            <StatTile label="Games in dataset" value={formatNumber(main?.rows ?? null, 0)} footnote={main?.seasons ? `${main.seasons[0]}–${main.seasons[1]}` : 'all_data_ml.csv missing'} />
            <StatTile label="Last ETL" value={main?.modified_at ? formatRelative(main.modified_at) : '—'} footnote={main?.modified_at ? new Date(main.modified_at).toLocaleString() : undefined} hint="Modification time of all_data_ml.csv." />
            <StatTile
              label="Training set hash"
              value={<span className="font-mono text-base">{query.data.fingerprint ? shortHash(query.data.fingerprint.sha256, 12) : 'computing…'}</span>}
              hint="SHA-256 of completed_games_ml.csv. Runs record this hash, so it tells you which runs were trained on the current data."
              footnote={training?.rows ? `${formatNumber(training.rows, 0)} completed games` : undefined}
            />
          </div>
          <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Database className="size-4" /> Dataset files
                </CardTitle>
                <CardDescription>Everything under data/ that the model and reports read.</CardDescription>
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>File</TableHead>
                      <TableHead className="text-right">Rows</TableHead>
                      <TableHead>Seasons</TableHead>
                      <TableHead className="text-right">Size</TableHead>
                      <TableHead>Updated</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {query.data.files.map((file) => (
                      <TableRow key={file.name} className={file.exists ? '' : 'opacity-60'}>
                        <TableCell>
                          <div className="font-mono text-xs">{file.name}</div>
                          <div className="text-xs text-muted-foreground">{file.description}</div>
                        </TableCell>
                        <TableCell className="tabular text-right">{formatNumber(file.rows, 0)}</TableCell>
                        <TableCell className="tabular">{file.seasons ? `${file.seasons[0]}–${file.seasons[1]}` : '—'}</TableCell>
                        <TableCell className="tabular text-right">{bytes(file.size)}</TableCell>
                        <TableCell className="whitespace-nowrap text-xs">{file.exists ? formatRelative(file.modified_at) : 'missing'}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Leakage audit</CardTitle>
                  <CardDescription>Checks that no feature encodes the outcome.</CardDescription>
                </CardHeader>
                <CardContent>
                  {query.data.leakage_audit ? (
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center gap-2">
                        {query.data.leakage_audit.ok ? <CheckCircle2 className="size-5 text-success" /> : <XCircle className="size-5 text-destructive" />}
                        <span className="font-medium">{query.data.leakage_audit.ok ? 'Clean' : 'Findings'}</span>
                        <Badge variant="outline" className="ml-auto">{formatRelative(query.data.leakage_audit.modified_at)}</Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {query.data.leakage_audit.feature_count ?? '?'} features · {formatNumber(query.data.leakage_audit.row_count ?? null, 0)} rows · {(query.data.leakage_audit.flagged_columns ?? []).length} flagged
                      </div>
                      {(query.data.leakage_audit.flagged_columns ?? []).length > 0 ? <div className="font-mono text-xs">{(query.data.leakage_audit.flagged_columns ?? []).join(', ')}</div> : null}
                    </div>
                  ) : (
                    <div className="text-sm text-muted-foreground">No audit report found under models/ or reports/.</div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">nflreadpy cache</CardTitle>
                  <CardDescription>Seasons cached as Parquet under data/cache.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <SeasonStrip label="Schedules" seasons={query.data.cache.schedule} hint="One file per season; the current season is refreshed on every ETL run." />
                  <SeasonStrip label="Play-by-play" seasons={query.data.cache.pbp} hint="Regular-season plays per season, used for EPA and success-rate features." />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Ad-hoc outputs</CardTitle>
                  <CardDescription>Prediction files and workbooks not tied to a run.</CardDescription>
                </CardHeader>
                <CardContent>
                  {query.data.predict_files.length === 0 ? (
                    <div className="text-sm text-muted-foreground">None.</div>
                  ) : (
                    <ul className="space-y-1 text-xs">
                      {query.data.predict_files.map((file) => (
                        <li key={file.path} className="flex items-center justify-between gap-2">
                          <span className="truncate font-mono">{file.name}</span>
                          <span className="shrink-0 text-muted-foreground">{formatRelative(file.modified_at)}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}
