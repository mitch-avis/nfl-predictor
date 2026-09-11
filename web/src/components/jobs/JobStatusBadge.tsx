import { Ban, CheckCircle2, Clock, Loader2, XCircle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

import type { JobStatus } from '@/api/types'
import { cn } from '@/utils/cn'

const STYLES: Record<JobStatus, { label: string; icon: LucideIcon; className: string }> = {
  queued: { label: 'Queued', icon: Clock, className: 'bg-muted text-muted-foreground' },
  running: { label: 'Running', icon: Loader2, className: 'bg-secondary text-secondary-foreground' },
  succeeded: { label: 'Succeeded', icon: CheckCircle2, className: 'bg-success text-success-foreground' },
  failed: { label: 'Failed', icon: XCircle, className: 'bg-destructive text-white' },
  canceled: { label: 'Canceled', icon: Ban, className: 'bg-muted text-muted-foreground line-through' },
}

/** Colored chip naming a job's current state. */
export function JobStatusBadge({ status, className }: { status: JobStatus; className?: string }) {
  const style = STYLES[status]
  const Icon = style.icon
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] font-semibold tracking-wide',
        style.className,
        className,
      )}
    >
      <Icon className={cn('size-3', status === 'running' && 'animate-spin')} aria-hidden />
      {style.label}
    </span>
  )
}
