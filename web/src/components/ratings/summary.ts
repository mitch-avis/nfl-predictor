/**
 * The detail page's lead tiles: each published rating with its rank, headline first, and the
 * headline's value before the schedule adjustment, so the adjustment's size is visible.
 */
import type { EntityKind, MetricColumnMeta, Polarity, Row } from '@/api/types'

import { entityConfig } from './entities'

export interface Rank {
  rank: number
  count: number
}

export interface SummaryTile {
  column: string
  value: Row[string]
  rank: Rank | null
  /** The headline rating's counterpart before the schedule adjustment; on the headline tile only. */
  unadjusted: { column: string; value: Row[string]; rank: Rank | null } | null
}

/**
 * `value`'s rank among `values` (1 = best by `polarity`), tied values sharing the better rank;
 * missing values are left out, and a neutral column has no rank.
 */
export function rankAmong(values: Row[string][], value: Row[string] | undefined, polarity: Polarity): Rank | null {
  if (polarity === 'neutral' || typeof value !== 'number' || !Number.isFinite(value)) return null
  const numbers = values.filter((candidate): candidate is number => typeof candidate === 'number' && Number.isFinite(candidate))
  const better = numbers.filter((candidate) => (polarity === 'higher' ? candidate > value : candidate < value)).length
  return { rank: better + 1, count: numbers.length }
}

/**
 * The tiles for one team or quarterback among the season's `rows`. A quarterback is ranked among
 * the qualifiers, as on the index, and one below the qualifier is not ranked; context columns
 * (schedule strength, the pass defense faced) are never ranked as better or worse.
 */
export function summaryTiles(kind: EntityKind, row: Row, rows: Row[], metadata: Record<string, MetricColumnMeta>): SummaryTile[] {
  const config = entityConfig(kind)
  const ranked = kind === 'qbs' ? rows.filter((candidate) => candidate.qb_is_eligible === true) : rows
  const rankable = kind !== 'qbs' || row.qb_is_eligible === true
  const rankOf = (column: string): Rank | null => {
    const meta = metadata[column]
    if (!rankable || !meta || meta.contextual) return null
    return rankAmong(ranked.map((candidate) => candidate[column] ?? null), row[column] ?? null, meta.polarity)
  }
  const headline = config.summaryColumns[0]
  return config.summaryColumns
    .filter((column) => column in row)
    .map((column) => ({
      column,
      value: row[column] ?? null,
      rank: rankOf(column),
      unadjusted:
        column === headline && config.unadjustedColumn in row
          ? { column: config.unadjustedColumn, value: row[config.unadjustedColumn] ?? null, rank: rankOf(config.unadjustedColumn) }
          : null,
    }))
}
