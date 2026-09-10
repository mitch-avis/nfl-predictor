import { cn } from '@/utils/cn'
import { formatPercent } from '@/utils/format'

/** Two-sided bar: away share on the left, home share on the right, with an optional market marker. */
export function WinProbBar({ homeProb, marketHomeProb, away, home, className }: { homeProb: number | null; marketHomeProb?: number | null; away: string; home: string; className?: string }) {
  if (homeProb === null || homeProb === undefined) return <div className="text-xs text-muted-foreground">No probability</div>
  const homePct = Math.max(0, Math.min(1, homeProb))
  const awayFavored = homePct < 0.5
  return (
    <div className={cn('space-y-1', className)}>
      <div className="flex justify-between text-xs font-medium">
        <span className={cn(awayFavored ? 'text-foreground' : 'text-muted-foreground')}>
          {away} {formatPercent(1 - homePct)}
        </span>
        <span className={cn(!awayFavored ? 'text-foreground' : 'text-muted-foreground')}>
          {formatPercent(homePct)} {home}
        </span>
      </div>
      <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-muted" role="img" aria-label={`${home} win probability ${formatPercent(homePct)}`}>
        <div className="absolute inset-y-0 left-0 bg-chart-3/80" style={{ width: `${(1 - homePct) * 100}%` }} />
        <div className="absolute inset-y-0 right-0 bg-primary/80" style={{ width: `${homePct * 100}%` }} />
        {marketHomeProb !== null && marketHomeProb !== undefined ? (
          <div
            className="absolute inset-y-0 w-0.5 bg-foreground"
            style={{ left: `${(1 - marketHomeProb) * 100}%` }}
            title={`Market: ${home} ${formatPercent(marketHomeProb)}`}
          />
        ) : null}
      </div>
    </div>
  )
}
