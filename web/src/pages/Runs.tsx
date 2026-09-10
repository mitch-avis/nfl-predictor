import { Check, Pin, PinOff } from 'lucide-react'
import { useState } from 'react'

import { useActivateRun, useClearActiveRun, useRuns, useSession } from '@/api/queries'
import type { RunKind, RunSummary } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { EmptyState } from '@/components/common/EmptyState'
import { InfoTooltip } from '@/components/common/InfoTooltip'
import { PageHeader } from '@/components/common/PageHeader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { formatNumber, formatPercent, formatRelative, seasonWeekLabel, shortHash } from '@/utils/format'

const KIND_LABEL: Record<RunKind, string> = {
  weekly: 'Weekly',
  training: 'Training',
  walk_forward: 'Walk-forward',
}

const STAGE_ORDER = ['wf_compare', 'train', 'predictions', 'reports'] as const
const STAGE_LABEL: Record<(typeof STAGE_ORDER)[number], string> = {
  wf_compare: 'Compare',
  train: 'Train',
  predictions: 'Predict',
  reports: 'Reports',
}

function StageChips({ run }: { run: RunSummary }) {
  if (run.kind !== 'weekly') return <span className="text-xs text-muted-foreground">—</span>
  return (
    <div className="flex flex-nowrap gap-1">
      {STAGE_ORDER.map((stage) => (
        <Badge
          key={stage}
          variant={run.stages[stage] ? 'secondary' : 'outline'}
          className={run.stages[stage] ? '' : 'text-muted-foreground'}
        >
          {run.stages[stage] ? <Check className="size-3" /> : null}
          {STAGE_LABEL[stage]}
        </Badge>
      ))}
    </div>
  )
}

function Holdout({ run }: { run: RunSummary }) {
  if (!run.holdout) return <span className="text-muted-foreground">—</span>
  const accuracy = run.holdout.winner_accuracy ?? run.holdout.pick_accuracy
  return (
    <span className="tabular whitespace-nowrap text-xs">
      Brier {formatNumber(run.holdout.brier, 4)}
      {accuracy !== undefined ? <> · Acc {formatPercent(accuracy, 1)}</> : null}
    </span>
  )
}

export function RunsPage() {
  const [kind, setKind] = useState<RunKind | 'all'>('all')
  const runs = useRuns(kind)
  const session = useSession()
  const activate = useActivateRun()
  const clearActive = useClearActiveRun()
  const [confirm, setConfirm] = useState<RunSummary | null>(null)
  const isAdmin = session.data?.user.role === 'admin'

  return (
    <>
      <PageHeader
        title="Runs"
        description={
          <>
            Every run directory under <code className="rounded bg-muted px-1">models/</code>, newest first. The{' '}
            <strong>active</strong> run feeds every other page.
          </>
        }
        actions={
          isAdmin && runs.data?.pinned_run_id ? (
            <Button variant="outline" size="sm" onClick={() => clearActive.mutate()} disabled={clearActive.isPending}>
              <PinOff className="size-4" /> Unpin (use newest complete)
            </Button>
          ) : null
        }
      />
      <Tabs value={kind} onValueChange={(value) => setKind(value as RunKind | 'all')} className="mb-4">
        <TabsList>
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="weekly">Weekly</TabsTrigger>
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="walk_forward">Walk-forward</TabsTrigger>
        </TabsList>
      </Tabs>
      {runs.isLoading ? <Skeleton className="h-64 w-full" /> : null}
      {runs.isError ? <ErrorState error={runs.error} /> : null}
      {runs.data && runs.data.runs.length === 0 ? (
        <EmptyState
          title="No runs found"
          description="Run scripts/weekly_run.py or the training CLI; run directories with a metadata.json appear here."
        />
      ) : null}
      {runs.data && runs.data.runs.length > 0 ? (
        <div className="overflow-x-auto rounded-xl border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-8" />
                <TableHead>Run</TableHead>
                <TableHead>Kind</TableHead>
                <TableHead>Week</TableHead>
                <TableHead>
                  <span className="inline-flex items-center gap-1">
                    Stages <InfoTooltip content="Weekly runs pass through compare, train, predict, and reports. A run is complete when reports finished." />
                  </span>
                </TableHead>
                <TableHead>
                  <span className="inline-flex items-center gap-1">
                    Holdout <InfoTooltip content="Brier score (lower is better) and winner accuracy on the run's holdout window." />
                  </span>
                </TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Data</TableHead>
                {isAdmin ? <TableHead className="text-right">Action</TableHead> : null}
              </TableRow>
            </TableHeader>
            <TableBody>
              {runs.data.runs.map((run) => (
                <TableRow key={run.run_id} data-active={run.is_active || undefined} className="data-[active]:bg-accent/40">
                  <TableCell>
                    {run.is_active ? (
                      <Pin className="size-4 text-primary" aria-label="Active run" />
                    ) : null}
                  </TableCell>
                  <TableCell className="font-medium">
                    <div className="max-w-xs break-all">{run.run_id}</div>
                    <div className="text-xs text-muted-foreground">
                      {run.model_kind ?? '—'} · {run.files.model ? 'model' : 'no model'}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{KIND_LABEL[run.kind]}</Badge>
                  </TableCell>
                  <TableCell className="whitespace-nowrap">{seasonWeekLabel(run.season, run.week)}</TableCell>
                  <TableCell>
                    <StageChips run={run} />
                  </TableCell>
                  <TableCell>
                    <Holdout run={run} />
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs" title={run.created_at}>
                    {formatRelative(run.created_at)}
                  </TableCell>
                  <TableCell className="font-mono text-xs" title={run.dataset_hash ?? undefined}>
                    {shortHash(run.dataset_hash)}
                  </TableCell>
                  {isAdmin ? (
                    <TableCell className="text-right">
                      {run.files.model && !run.is_active ? (
                        <Button size="sm" variant="outline" onClick={() => setConfirm(run)}>
                          Activate
                        </Button>
                      ) : null}
                    </TableCell>
                  ) : null}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      ) : null}
      <Dialog open={confirm !== null} onOpenChange={(open) => !open && setConfirm(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Activate {confirm?.run_id}?</DialogTitle>
            <DialogDescription>
              Predictions, betting, power rankings, and model pages will read from this run for everyone.
            </DialogDescription>
          </DialogHeader>
          {activate.isError ? <ErrorState error={activate.error} title="Activation failed" /> : null}
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirm(null)}>
              Cancel
            </Button>
            <Button
              disabled={activate.isPending}
              onClick={() =>
                confirm && activate.mutate(confirm.run_id, { onSuccess: () => setConfirm(null) })
              }
            >
              <Pin className="size-4" /> Activate
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
