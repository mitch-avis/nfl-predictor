import { AlertTriangle, Play } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router'

import { useCreateJob, useJobCatalog, useJobs, useSession } from '@/api/queries'
import type { JobParams, JobTemplate } from '@/api/types'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { JobForm } from '@/components/jobs/JobForm'
import { JobStatusBadge } from '@/components/jobs/JobStatusBadge'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatRelative } from '@/utils/format'

const CATEGORY_ORDER = ['Data', 'Pipeline', 'Model', 'Reports', 'Validation']

function categoryRank(category: string): number {
  const index = CATEGORY_ORDER.indexOf(category)
  return index === -1 ? CATEGORY_ORDER.length : index
}

export function JobsPage() {
  const catalog = useJobCatalog()
  const jobs = useJobs()
  const session = useSession()
  const create = useCreateJob()
  const navigate = useNavigate()
  const [open, setOpen] = useState<JobTemplate | null>(null)
  const isAdmin = session.data?.user.role === 'admin'
  const busy = new Set(catalog.data?.busy_groups ?? [])

  const categories = Array.from(new Set((catalog.data?.templates ?? []).map((t) => t.category))).sort(
    (a, b) => categoryRank(a) - categoryRank(b),
  )

  const launch = (template: JobTemplate, params: JobParams) => {
    create.mutate(
      { templateId: template.id, params },
      {
        onSuccess: (job) => {
          setOpen(null)
          void navigate(`/jobs/${job.id}`)
        },
      },
    )
  }

  return (
    <>
      <PageHeader
        title="Jobs"
        description="Launch the project's pipelines and watch their logs stream. Long jobs keep running if you close the page."
      />
      {catalog.isLoading ? <Skeleton className="h-48 w-full" /> : null}
      {catalog.isError ? <ErrorState error={catalog.error} /> : null}
      {!isAdmin && catalog.data ? (
        <p className="mb-4 text-sm text-muted-foreground">
          You can watch jobs here; launching and cancelling need an admin account.
        </p>
      ) : null}
      {categories.map((category) => (
        <section key={category} className="mb-6">
          <h2 className="mb-2 text-sm font-semibold tracking-wide text-muted-foreground uppercase">{category}</h2>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {catalog.data?.templates
              .filter((template) => template.category === category)
              .map((template) => {
                const blocked = template.exclusive_group !== null && busy.has(template.exclusive_group)
                return (
                  <Card key={template.id} className="flex flex-col">
                    <CardHeader>
                      <CardTitle className="text-base">{template.label}</CardTitle>
                      <CardDescription>{template.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="mt-auto flex flex-wrap items-center gap-2">
                      <Button size="sm" disabled={!isAdmin || blocked} onClick={() => setOpen(template)}>
                        <Play className="size-4" /> Run
                      </Button>
                      {blocked ? <Badge variant="outline">Group busy</Badge> : null}
                      {template.writes_datasets ? <Badge variant="outline">Writes data</Badge> : null}
                      {template.needs_active_run ? <Badge variant="outline">Uses active run</Badge> : null}
                      {template.chain_template_id ? (
                        <Badge variant="secondary">then {template.chain_template_id}</Badge>
                      ) : null}
                    </CardContent>
                  </Card>
                )
              })}
          </div>
        </section>
      ))}

      <h2 className="mb-2 text-sm font-semibold tracking-wide text-muted-foreground uppercase">History</h2>
      {jobs.isError ? <ErrorState error={jobs.error} /> : null}
      {jobs.data && jobs.data.jobs.length === 0 ? (
        <EmptyState title="No jobs yet" description="Every job you run from this page is listed here with its logs." />
      ) : null}
      {jobs.data && jobs.data.jobs.length > 0 ? (
        <div className="overflow-x-auto rounded-xl border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Job</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Started</TableHead>
                <TableHead>By</TableHead>
                <TableHead className="text-right">Logs</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.data.jobs.map((job) => (
                <TableRow key={job.id}>
                  <TableCell className="font-medium">
                    <div>{job.template_label}</div>
                    <div className="text-xs text-muted-foreground">
                      {Object.entries(job.params)
                        .map(([key, value]) => `${key}=${String(value)}`)
                        .join(' · ') || 'no options'}
                    </div>
                  </TableCell>
                  <TableCell>
                    <JobStatusBadge status={job.status} />
                    {job.error ? (
                      <div className="mt-1 flex items-start gap-1 text-xs text-muted-foreground">
                        <AlertTriangle className="mt-0.5 size-3 shrink-0" />
                        <span className="max-w-60 truncate" title={job.error}>
                          {job.error}
                        </span>
                      </div>
                    ) : null}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs" title={job.started_at ?? job.created_at}>
                    {formatRelative(job.started_at ?? job.created_at)}
                  </TableCell>
                  <TableCell className="text-xs">{job.created_by ?? '—'}</TableCell>
                  <TableCell className="text-right">
                    <Button asChild size="sm" variant="outline">
                      <Link to={`/jobs/${job.id}`}>Open</Link>
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      ) : null}

      <Dialog open={open !== null} onOpenChange={(next) => !next && setOpen(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{open?.label}</DialogTitle>
            <DialogDescription>{open?.description}</DialogDescription>
          </DialogHeader>
          {create.isError ? <ErrorState error={create.error} title="Could not start the job" /> : null}
          {open ? (
            <JobForm template={open} pending={create.isPending} onSubmit={(params) => launch(open, params)} />
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  )
}
