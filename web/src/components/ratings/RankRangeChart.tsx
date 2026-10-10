import { useState } from 'react'
import { Link } from 'react-router'

import type { EntityKind } from '@/api/types'
import { cn } from '@/utils/cn'

import { entityPath } from './entities'
import { RankIntervalTrack } from './RankInterval'
import { rankCenter, rankChanceText, rankRangeHeadline, rankRangeSummary, rankRangesByMedian, rankTicks, type RankRange } from './rankRanges'

// One label width per kind keeps every row's track aligned: team codes are short, names long.
const ROW_GRID: Record<EntityKind, string> = {
  teams: 'grid grid-cols-[3.5rem_minmax(0,1fr)] items-center gap-x-2',
  qbs: 'grid grid-cols-[7rem_minmax(0,1fr)] items-center gap-x-2 sm:grid-cols-[10rem_minmax(0,1fr)]',
}

/**
 * The league's rank ranges, one row per team or quarterback ordered by median rank, rank 1 at the
 * left. The readout above the rows describes the hovered or focused row; every row links to its
 * detail page, and each link's name is its interval in words.
 */
export function RankRangeChart({ kind, season, ranges, labels }: { kind: EntityKind; season: number; ranges: RankRange[]; labels: Record<string, string> }) {
  const [activeId, setActiveId] = useState<string | null>(null)
  const count = ranges.length
  const ticks = rankTicks(count)
  const nameOf = (range: RankRange) => labels[range.id] ?? range.label
  const active = ranges.find((range) => range.id === activeId)
  return (
    <div className="flex flex-col gap-3">
      <div role="status" aria-live="polite" className="sticky top-14 z-10 min-h-12 rounded-lg border bg-card px-3 py-2 text-sm">
        {active ? (
          <>
            <div className="font-medium">
              {nameOf(active)}: {rankRangeHeadline(active)}
            </div>
            <div className="text-muted-foreground">{rankChanceText(kind, active)}</div>
          </>
        ) : (
          <span className="text-muted-foreground">Hover or focus a row for its numbers; open it for the full page.</span>
        )}
      </div>
      <div className={ROW_GRID[kind]} aria-hidden>
        <span className="text-xs text-muted-foreground">Rank</span>
        <div className="tabular relative h-4 text-xs text-muted-foreground">
          {ticks.map((tick) => (
            <span key={tick} className="absolute -translate-x-1/2" style={{ left: `${rankCenter(tick, count)}%` }}>
              {tick}
            </span>
          ))}
        </div>
      </div>
      <ol className="flex flex-col">
        {rankRangesByMedian(ranges).map((range) => (
          <li key={range.id}>
            <Link
              to={entityPath(kind, range.id, season)}
              aria-label={rankRangeSummary({ ...range, label: nameOf(range) })}
              className={cn(ROW_GRID[kind], 'w-full rounded-sm px-1 py-1 hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring', range.id === activeId && 'bg-muted/60')}
              onMouseEnter={() => setActiveId(range.id)}
              onFocus={() => setActiveId(range.id)}
            >
              <span className="truncate text-sm font-medium">{range.label}</span>
              <RankIntervalTrack range={range} count={count} ticks={ticks} />
            </Link>
          </li>
        ))}
      </ol>
    </div>
  )
}
