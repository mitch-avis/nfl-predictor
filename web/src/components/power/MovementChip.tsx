import { ArrowDown, ArrowUp, Minus } from 'lucide-react'

import { cn } from '@/utils/cn'

/** Rank movement since last week: up is good. */
export function MovementChip({ delta }: { delta: number | null }) {
  if (delta === null || delta === undefined) return <span className="text-xs text-muted-foreground">new</span>
  if (delta === 0)
    return (
      <span className="inline-flex items-center gap-0.5 text-xs text-muted-foreground">
        <Minus className="size-3" /> 0
      </span>
    )
  const up = delta > 0
  return (
    <span className={cn('inline-flex items-center gap-0.5 text-xs font-medium', up ? 'text-success' : 'text-destructive')}>
      {up ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />}
      {Math.abs(delta)}
    </span>
  )
}
