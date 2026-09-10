import { formatNumber } from '@/utils/format'

/** A 1-10 power rating as a filled bar with the number beside it. */
export function RatingBar({ value }: { value: number | null }) {
  if (value === null) return <span className="text-muted-foreground">—</span>
  const pct = Math.max(0, Math.min(100, ((value - 1) / 9) * 100))
  return (
    <span className="inline-flex items-center gap-2">
      <span className="h-2 w-20 overflow-hidden rounded-full bg-muted" aria-hidden>
        <span className="block h-full rounded-full bg-primary" style={{ width: `${pct}%` }} />
      </span>
      <span className="tabular">{formatNumber(value, 2)}</span>
    </span>
  )
}
