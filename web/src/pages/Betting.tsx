import { Download, Info } from 'lucide-react'
import { useState } from 'react'

import { qs } from '@/api/client'
import { useBetting, useSession } from '@/api/queries'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { ActionBadge } from '@/components/betting/ActionBadge'
import { ColumnPicker, useColumnGroups } from '@/components/table/ColumnPicker'
import { DataTable } from '@/components/table/DataTable'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { useNumberParam, useQueryParam } from '@/hooks/useQueryParam'
import { formatRelative } from '@/utils/format'

export function BettingPage() {
  const [run] = useQueryParam('run')
  const [season] = useNumberParam('season')
  const [week] = useNumberParam('week')
  const params = { run, season, week }
  const query = useBetting(params)
  const session = useSession()
  const [showTotals, setShowTotals] = useState(false)
  const groups = useColumnGroups('betting', query.data?.table, ['Fair odds', 'Total (informational)', 'Uncertainty'])
  const columns = groups.columns.filter((key) => showTotals || query.data?.table.column_metadata[key]?.actionable !== false)

  return (
    <>
      <PageHeader
        title="Betting"
        description="Where the model disagrees with the price. Edges are model probability minus the implied probability of the offered line; the ladder turns edge into an action size."
        actions={
          <>
            {query.data ? <ColumnPicker groups={groups.groups.filter((g) => g !== 'Total (informational)')} hidden={groups.hidden} onToggle={groups.toggle} onReset={groups.reset} /> : null}
            {query.data?.xlsx_available ? (
              <Button asChild variant="outline" size="sm">
                <a href={`/api/betting/xlsx${qs({ run })}`}>
                  <Download className="size-4" /> Workbook
                </a>
              </Button>
            ) : session.data?.user.role === 'admin' ? (
              <Button variant="outline" size="sm" disabled title="Generate from the Jobs page (phase 2)">
                <Download className="size-4" /> No workbook yet
              </Button>
            ) : null}
          </>
        }
      />
      {query.isLoading ? <Skeleton className="h-64 w-full" /> : null}
      {query.isError ? <ErrorState error={query.error} title="No betting report to show" /> : null}
      {query.data ? (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline">
              {query.data.season} · Week {query.data.week}
            </Badge>
            <span>{query.data.run_id ? `Run ${query.data.run_id}` : 'Unattached predictions'} · lines as of {formatRelative(query.data.generated_at)}</span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {query.data.ladder.map((step) => (
              <span key={step.action} className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                <ActionBadge action={step.action} /> ≥ {(step.min_edge * 100).toFixed(0)}%
              </span>
            ))}
            <label className="ml-auto inline-flex items-center gap-2 text-xs">
              <Switch checked={showTotals} onCheckedChange={setShowTotals} aria-label="Show total columns" /> Show totals (informational)
            </label>
          </div>
          <DataTable
            table={query.data.table}
            columns={columns}
            defaultSort={{ key: 'moneyline_edge_prob', desc: true }}
            rowKey={(row) => String(row.game_id)}
            renderers={{
              moneyline_action: (value) => <ActionBadge action={value === null ? null : String(value)} />,
              spread_action: (value) => <ActionBadge action={value === null ? null : String(value)} />,
              total_action: (value) => <ActionBadge action={value === null ? null : String(value)} muted />,
            }}
          />
          <Alert>
            <Info className="size-4" />
            <AlertTitle>How to read this</AlertTitle>
            <AlertDescription>
              <ul className="list-disc space-y-1 pl-4">
                {query.data.notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        </div>
      ) : null}
    </>
  )
}
