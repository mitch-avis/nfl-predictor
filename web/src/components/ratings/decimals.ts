/** Fixed decimals per ratings column, so a column's decimal points line up down the table. */
import type { MetricShape, MetricTable, Row } from '@/api/types'

/** Decimals of a percentage column that is not whole percentages. */
const PERCENT_DECIMALS = 1

/**
 * Decimal places for one column: whole numbers stay whole, points-per-game scores get 2, and the
 * rest follow their scale (3 below 1, 2 below 100, 1 above). A column of proportions shown as
 * percentages gets 1, or none when every value is a whole percentage. Null without numbers.
 */
export function columnDecimals(values: ReadonlyArray<Row[string]>, shape: MetricShape | null | undefined, percent = false): number | null {
  const numbers = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  if (numbers.length === 0) return null
  if (percent) {
    // Rounded first, so 0.29 (28.999999999999996 when multiplied) counts as a whole 29%.
    return numbers.every((value) => Number.isInteger(Number((value * 100).toFixed(6)))) ? 0 : PERCENT_DECIMALS
  }
  if (numbers.every((value) => Number.isInteger(value))) return 0
  if (shape === 'score') return 2
  const scale = Math.max(...numbers.map((value) => Math.abs(value)))
  if (scale < 1) return 3
  if (scale < 100) return 2
  return 1
}

/** A copy of `table` whose `columns` carry the decimals their values call for. */
export function withDecimals(table: MetricTable, columns: string[]): MetricTable {
  const metadata = { ...table.column_metadata }
  for (const column of columns) {
    const meta = metadata[column]
    if (!meta) continue
    metadata[column] = { ...meta, decimals: columnDecimals(table.rows.map((row) => row[column] ?? null), meta.shape, meta.percent) }
  }
  return { ...table, column_metadata: metadata }
}
