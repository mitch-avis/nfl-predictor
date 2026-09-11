import { ArrowLeft, Ban } from 'lucide-react'
import { useMemo } from 'react'
import { Link, useParams } from 'react-router'

import { isTerminal, useCancelJob, useJob, useSession } from '@/api/queries'
import { useJobStream } from '@/api/sse'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { JobStatusBadge } from '@/components/jobs/JobStatusBadge'
import { LogConsole } from '@/components/jobs/LogConsole'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { formatRelative } from '@/utils/format'

export function JobDetailPage() {
  const { jobId = '' } = useParams()
  const job = useJob(jobId || null)
  const session = useSession()
  const cancel = useCancelJob()
  const stream = useJobStream(jobId || null)
  const isAdmin = session.data?.user.role === 'admin'
  const status = job.data?.status ?? stream.status
  const progress = job.data?.progress ?? null
  const percent = useMemo(
    () => (progress && progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : null),
    [progress],
  )

  return (
    <>
      <PageHeader
        title={job.data?.template_label ?? 'Job'}
        description={
          job.data ? (
            <>
              Started {formatRelative(job.data.started_at ?? job.data.created_at)} by {job.data.created_by ?? 'unknown'}
              {job.data.parent_job_id ? (
                <>
                  {' · '}
                  <Link className="underline" to={`/jobs/${job.data.parent_job_id}`}>
                    chained from another job
                  </Link>
                </>
              ) : null}
            </>
          ) : null
        }
        actions={
          <div className="flex items-center gap-2">
            {status ? <JobStatusBadge status={status} /> : null}
            {isAdmin && status && !isTerminal(status) ? (
              <Button
                size="sm"
                variant="outline"
                disabled={cancel.isPending}
                onClick={() => cancel.mutate(jobId)}
              >
                <Ban className="size-4" /> Cancel
              </Button>
            ) : null}
            <Button asChild size="sm" variant="ghost">
              <Link to="/jobs">
                <ArrowLeft className="size-4" /> All jobs
              </Link>
            </Button>
          </div>
        }
      />
      {job.isLoading ? <Skeleton className="h-64 w-full" /> : null}
      {job.isError ? <ErrorState error={job.error} /> : null}
      {cancel.isError ? <ErrorState error={cancel.error} title="Could not cancel the job" /> : null}
      {job.data?.error ? <ErrorState error={new Error(job.data.error)} title="The job reported a problem" /> : null}
      {progress ? (
        <div className="mb-4 grid gap-1.5">
          <div className="flex items-baseline justify-between text-xs text-muted-foreground">
            <span className="truncate">{progress.label}</span>
            <span className="tabular">
              {progress.current}/{progress.total}
            </span>
          </div>
          <Progress value={percent ?? 0} />
        </div>
      ) : null}
      {job.data ? (
        <LogConsole
          lines={stream.lines}
          empty={isTerminal(job.data.status) ? 'This job produced no output.' : 'Waiting for output…'}
        />
      ) : null}
    </>
  )
}
