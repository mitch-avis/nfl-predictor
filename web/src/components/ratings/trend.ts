/** Points, reference lines and round axis ticks for the week-by-week charts. */
import type { Row } from '@/api/types'

export interface TrendPoint {
  week: number
  value: number
  opponent: string
}

/** A dashed horizontal line on a trend chart, with the caption that explains it. */
export interface TrendReference {
  value: number
  caption: string
}

/** Week-by-week points for one numeric column, skipping games without a value. */
export function buildTrendPoints(rows: Row[], column: string): TrendPoint[] {
  return rows
    .filter((row) => typeof row.week === 'number' && typeof row[column] === 'number')
    .map((row) => ({ week: row.week as number, value: row[column] as number, opponent: String(row.opponent_team ?? '') }))
    .sort((left, right) => left.week - right.week)
}

/** A line at the mean of the games shown, written as the chart writes its values. */
export function meanReference(points: TrendPoint[], format: (value: number) => string = String): TrendReference | null {
  if (points.length === 0) return null
  const mean = points.reduce((total, point) => total + point.value, 0) / points.length
  return { value: mean, caption: `Dashed line: the average of the games shown (${format(mean)}).` }
}

const NICE_STEP_FACTORS = [1, 2, 2.5, 5, 10]

/**
 * Round axis ticks that cover `min` to `max` in about `target` steps of 1, 2, 2.5 or 5 times a
 * power of ten (-4, -2, 0 rather than -3.6, -2.7, ...). A flat line gets a band around its value.
 */
export function niceTicks(min: number, max: number, target = 5): number[] {
  let low = Math.min(min, max)
  let high = Math.max(min, max)
  if (low === high) {
    const pad = low === 0 ? 1 : Math.abs(low) / 2
    low -= pad
    high += pad
  }
  const raw = (high - low) / (target - 1)
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const step = (NICE_STEP_FACTORS.find((factor) => factor * magnitude >= raw) ?? 10) * magnitude
  const start = Math.floor(low / step) * step
  const count = Math.round((Math.ceil(high / step) * step - start) / step)
  return Array.from({ length: count + 1 }, (_, index) => Number((start + index * step).toFixed(10)))
}

/** The columns a trend chart offers: the preferred ones with values first, then the rest; never the week. */
export function trendColumns(rows: Row[], columns: readonly string[], preferred: readonly string[] = []): string[] {
  const hasValues = (column: string) => column !== 'week' && rows.some((row) => typeof row[column] === 'number')
  return [...new Set([...preferred.filter(hasValues), ...columns.filter(hasValues)])]
}
