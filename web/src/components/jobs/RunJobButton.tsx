import type { LucideIcon } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router'

import { useCreateJob, useJobCatalog, useSession } from '@/api/queries'
import type { JobParams } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { JobForm } from '@/components/jobs/JobForm'

/**
 * Admin-only shortcut that launches one job template from any page.
 *
 * Templates with parameters open a prefilled form; parameterless ones start immediately. Either
 * way the browser lands on the job's log page. Nothing renders for viewers.
 */
export function RunJobButton({
  templateId,
  label,
  preset,
  icon: Icon,
  variant = 'outline',
}: {
  templateId: string
  label: string
  preset?: JobParams
  icon?: LucideIcon
  variant?: 'default' | 'outline' | 'secondary' | 'ghost'
}) {
  const catalog = useJobCatalog()
  const session = useSession()
  const create = useCreateJob()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)

  const template = catalog.data?.templates.find((item) => item.id === templateId)
  if (session.data?.user.role !== 'admin' || template === undefined) return null
  const busy = template.exclusive_group !== null && (catalog.data?.busy_groups ?? []).includes(template.exclusive_group)

  const launch = (params: JobParams) => {
    create.mutate(
      { templateId, params },
      {
        onSuccess: (job) => {
          setOpen(false)
          void navigate(`/jobs/${job.id}`)
        },
      },
    )
  }

  return (
    <>
      <Button
        size="sm"
        variant={variant}
        disabled={busy || create.isPending}
        onClick={() => (template.params.length === 0 ? launch({}) : setOpen(true))}
      >
        {Icon ? <Icon className="size-4" /> : null}
        {busy ? `${label} (busy)` : label}
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{template.label}</DialogTitle>
            <DialogDescription>{template.description}</DialogDescription>
          </DialogHeader>
          {create.isError ? <ErrorState error={create.error} title="Could not start the job" /> : null}
          <JobForm template={template} preset={preset} pending={create.isPending} onSubmit={launch} />
        </DialogContent>
      </Dialog>
    </>
  )
}
