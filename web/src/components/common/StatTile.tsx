import type { ReactNode } from 'react'

import { Card } from '@/components/ui/card'
import { cn } from '@/utils/cn'

import { InfoTooltip } from './InfoTooltip'

/** A compact labeled value with an optional hint and footnote. */
export function StatTile({
  label,
  value,
  hint,
  footnote,
  className,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  footnote?: ReactNode
  className?: string
}) {
  return (
    <Card className={cn('gap-1 px-4 py-3', className)}>
      <div className="flex items-center gap-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
        {hint ? <InfoTooltip content={hint} label={`About ${label}`} /> : null}
      </div>
      <div className="tabular truncate text-xl font-semibold leading-tight sm:text-2xl">{value}</div>
      {footnote ? <div className="text-xs text-muted-foreground">{footnote}</div> : null}
    </Card>
  )
}
