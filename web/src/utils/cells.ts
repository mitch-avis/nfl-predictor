/** Format a table cell according to its registry metadata. */
import type { ColumnMeta, Row } from '@/api/types'

import { formatDate, formatDateTime, formatMoneyline, formatNumber, formatPercent, formatSigned } from './format'

export type CellValue = Row[string]

export function formatCell(meta: ColumnMeta | undefined, value: CellValue): string {
  if (value === null || value === undefined) return '—'
  if (!meta) return String(value)
  switch (meta.kind) {
    case 'int':
      return typeof value === 'number' ? formatNumber(value, 0) : String(value)
    case 'float':
      return typeof value === 'number' ? formatNumber(value, meta.decimals ?? 1) : String(value)
    case 'pct':
      return typeof value === 'number' ? `${value > 0 ? '+' : ''}${(value * 100).toFixed(meta.decimals ?? 1)}%` : String(value)
    case 'prob':
      return typeof value === 'number' ? formatPercent(value, meta.decimals ?? 0) : String(value)
    case 'money':
      return typeof value === 'number' ? formatMoneyline(value) : String(value)
    case 'spread':
      return typeof value === 'number' ? formatSigned(value, meta.decimals ?? 1) : String(value)
    case 'datetime':
      return formatDateTime(String(value))
    case 'date':
      return formatDate(String(value))
    case 'bool':
      return value ? 'Yes' : 'No'
    default:
      return String(value)
  }
}

export interface ColumnStats {
  min: number
  max: number
  absMax: number
}

/** Per-column numeric ranges used by the heatmap. */
export function columnStats(rows: Row[], keys: string[]): Record<string, ColumnStats> {
  const stats: Record<string, ColumnStats> = {}
  for (const key of keys) {
    let min = Infinity
    let max = -Infinity
    for (const row of rows) {
      const value = row[key]
      if (typeof value === 'number' && Number.isFinite(value)) {
        if (value < min) min = value
        if (value > max) max = value
      }
    }
    if (min !== Infinity) stats[key] = { min, max, absMax: Math.max(Math.abs(min), Math.abs(max)) }
  }
  return stats
}

/**
 * Background color for a heatmap cell. Polar columns fade from transparent (worst) to green (best);
 * neutral columns diverge around zero (blue for positive, orange for negative).
 */
export function heatStyle(meta: ColumnMeta, value: CellValue, stats: ColumnStats | undefined): React.CSSProperties | undefined {
  if (!meta.heatmap || typeof value !== 'number' || !stats) return undefined
  if (meta.polarity === 'neutral') {
    if (stats.absMax === 0) return undefined
    const t = Math.min(1, Math.abs(value) / stats.absMax)
    const alpha = (0.08 + 0.42 * t).toFixed(3)
    return { backgroundColor: value >= 0 ? `oklch(0.72 0.14 240 / ${alpha})` : `oklch(0.75 0.15 55 / ${alpha})` }
  }
  const span = stats.max - stats.min
  if (span === 0) return undefined
  let t = (value - stats.min) / span
  if (meta.polarity === 'lower') t = 1 - t
  const alpha = (0.06 + 0.44 * t).toFixed(3)
  return { backgroundColor: `oklch(0.75 0.16 150 / ${alpha})` }
}

export function isNumericKind(meta: ColumnMeta | undefined): boolean {
  return !!meta && ['int', 'float', 'pct', 'prob', 'money', 'spread'].includes(meta.kind)
}
