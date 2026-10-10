import type { EntityKind, MetricColumnMeta, Row } from '@/api/types'
import { InfoTooltip } from '@/components/common/InfoTooltip'
import { cn } from '@/utils/cn'

import { decimalsFor, formatMetric } from './format'
import { ordinal } from './rankRanges'
import { summaryTiles, type Rank } from './summary'

function rankText(rank: Rank | null): string | null {
  return rank ? `${ordinal(rank.rank)} of ${rank.count}` : null
}

/**
 * The detail page's lead: each published rating with its rank, headline first, and the headline's
 * value before the schedule adjustment, so the adjustment's size is visible.
 */
export function RatingSummary({ kind, row, rows, metadata }: { kind: EntityKind; row: Row; rows: Row[]; metadata: Record<string, MetricColumnMeta> }) {
  const tiles = summaryTiles(kind, row, rows, metadata)
  return (
    <section aria-label="Season ratings" className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
      {tiles.map((tile, index) => {
        const meta = metadata[tile.column]
        const label = meta?.label ?? tile.column
        const unadjusted = tile.unadjusted
        const unadjustedMeta = unadjusted ? metadata[unadjusted.column] : undefined
        return (
          <div
            key={tile.column}
            role="group"
            aria-label={label}
            className={cn('rounded-lg border bg-card px-3 py-2', index === 0 && 'col-span-2 border-primary/40 bg-primary/5')}
          >
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              {label}
              {meta?.description ? <InfoTooltip content={meta.description} label={`About ${label}`} /> : null}
            </div>
            <div className={cn('tabular font-semibold', index === 0 ? 'text-2xl' : 'text-lg')}>{formatMetric(meta, tile.value, decimalsFor(meta, rows, tile.column))}</div>
            {rankText(tile.rank) ? <div className="text-xs text-muted-foreground">{rankText(tile.rank)}</div> : null}
            {unadjusted ? (
              <p className="mt-1 text-xs text-muted-foreground">
                Before the schedule adjustment: {unadjustedMeta?.label ?? unadjusted.column}{' '}
                <span className="tabular font-medium text-foreground">{formatMetric(unadjustedMeta, unadjusted.value, decimalsFor(unadjustedMeta, rows, unadjusted.column))}</span>
                {rankText(unadjusted.rank) ? ` (${rankText(unadjusted.rank)})` : ''}
              </p>
            ) : null}
          </div>
        )
      })}
    </section>
  )
}
