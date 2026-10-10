/** Formatting of a ratings value outside a table: with the column's kind and fixed decimals. */
import type { MetricColumnMeta, Row } from '@/api/types'
import { formatCell } from '@/utils/cells'

import { columnDecimals } from './decimals'

/** One value as its table shows it, with `decimals` (a percentage's when the column is one). */
export function formatMetric(meta: MetricColumnMeta | undefined, value: Row[string] | undefined, decimals: number | null): string {
  if (!meta) return value === null || value === undefined ? '—' : String(value)
  return formatCell({ ...meta, decimals }, value ?? null)
}

/** The decimals a column's values call for across `rows`. */
export function decimalsFor(meta: MetricColumnMeta | undefined, rows: Row[], column: string): number | null {
  return columnDecimals(
    rows.map((row) => row[column] ?? null),
    meta?.shape,
    meta?.percent ?? false,
  )
}
