import { cn } from '@/utils/cn'

import { rankCenter, rankSpan, type RankRange } from './rankRanges'

type TrackSize = 'row' | 'mini'

function at(left: number): { left: string } {
  return { left: `${left}%` }
}

/**
 * One rank range on a track whose left edge is rank 1: a thin bar for the middle 95% of redraws,
 * a thick bar for the middle 50%, a dot for the median, and a hollow diamond for the published
 * rank when it differs from the median. Decorative: the caller supplies the text alternative.
 */
export function RankIntervalTrack({
  range,
  count,
  size = 'row',
  ticks = [],
  showPublished = size === 'row',
}: {
  range: Pick<RankRange, 'rank' | 'publishedRank'>
  count: number
  size?: TrackSize
  ticks?: number[]
  showPublished?: boolean
}) {
  const { q025, q250, q500, q750, q975 } = range.rank
  return (
    <div aria-hidden className={cn('relative w-full', size === 'row' ? 'h-5' : 'h-3')}>
      {ticks.map((tick) => (
        <span key={tick} className="absolute inset-y-0 w-px bg-border" style={at(rankCenter(tick, count))} />
      ))}
      {q025 !== null && q975 !== null ? (
        <span className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-chart-1" style={{ ...at(rankSpan(q025, q975, count).left), width: `${rankSpan(q025, q975, count).width}%` }} />
      ) : null}
      {q250 !== null && q750 !== null ? (
        <span
          className={cn('absolute top-1/2 -translate-y-1/2 rounded-sm bg-chart-1', size === 'row' ? 'h-2.5' : 'h-1.5')}
          style={{ ...at(rankSpan(q250, q750, count).left), width: `${rankSpan(q250, q750, count).width}%` }}
        />
      ) : null}
      {q500 !== null ? (
        <span
          className={cn('absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground ring-2 ring-card', size === 'row' ? 'size-2.5' : 'size-2')}
          style={at(rankCenter(q500, count))}
        />
      ) : null}
      {showPublished && range.publishedRank !== q500 && range.publishedRank > 0 ? (
        <span
          className={cn('absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rotate-45 border-2 border-foreground bg-card', size === 'row' ? 'size-2.5' : 'size-2')}
          style={at(rankCenter(range.publishedRank, count))}
        />
      ) : null}
    </div>
  )
}

/** The key to the interval marks, drawn with the same marks. */
export function RankIntervalKey() {
  const items = [
    { label: 'Middle 50% of redraws', mark: <span className="h-2.5 w-5 rounded-sm bg-chart-1" /> },
    { label: 'Middle 95%', mark: <span className="h-0.5 w-5 rounded-full bg-chart-1" /> },
    { label: 'Median rank', mark: <span className="size-2.5 rounded-full bg-foreground" /> },
    { label: 'Published rank, when different', mark: <span className="size-2.5 rotate-45 border-2 border-foreground bg-card" /> },
  ]
  return (
    <ul aria-label="Key" className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {items.map((item) => (
        <li key={item.label} className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-flex w-5 items-center justify-center">
            {item.mark}
          </span>
          {item.label}
        </li>
      ))}
    </ul>
  )
}
