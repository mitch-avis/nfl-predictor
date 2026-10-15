/**
 * Rank ranges by week for a season in progress: the rank when the games through each week are
 * redrawn and refit. The API carries every number; this module shapes them for the band chart and
 * its text alternative.
 */
import type { EntityKind, Row, SeasonTable } from '@/api/types'

import { ordinal } from './rankRanges'

/** One week's median rank and its 50% and 95% bands, rank 1 the best. */
export interface RankHistoryPoint {
  week: number
  published: number | null
  median: number | null
  low50: number | null
  high50: number | null
  low95: number | null
  high95: number | null
}

const RANK_COLUMN: Record<EntityKind, string> = { teams: 'team_rank', qbs: 'qb_rank' }

function numberOrNull(value: Row[string] | undefined): number | null {
  return typeof value === 'number' ? value : null
}

/** Read every week of one team's or quarterback's rank history, in week order. */
export function parseRankHistory(kind: EntityKind, payload: SeasonTable): RankHistoryPoint[] {
  const rank = RANK_COLUMN[kind]
  return payload.table.rows
    .map((row) => ({
      week: numberOrNull(row.week) ?? 0,
      published: numberOrNull(row[rank]),
      median: numberOrNull(row[`${rank}_q500`]),
      low50: numberOrNull(row[`${rank}_q250`]),
      high50: numberOrNull(row[`${rank}_q750`]),
      low95: numberOrNull(row[`${rank}_q025`]),
      high95: numberOrNull(row[`${rank}_q975`]),
    }))
    .sort((left, right) => left.week - right.week)
}

function band(low: number | null, high: number | null): [number, number] | null {
  return low === null || high === null ? null : [low, high]
}

/** The rows the band chart draws: the median line and the two bands as `[low, high]`. */
export function rankBandData(points: readonly RankHistoryPoint[]) {
  return points.map((point) => ({
    week: point.week,
    median: point.median,
    band50: band(point.low50, point.high50),
    band95: band(point.low95, point.high95),
  }))
}

function weekText(point: RankHistoryPoint): string {
  const median = point.median === null ? 'n/a' : ordinal(point.median)
  const range = point.low95 === null || point.high95 === null ? 'n/a' : `${ordinal(point.low95)}–${ordinal(point.high95)}`
  return `${median} after week ${point.week} (95%: ${range})`
}

/** One sentence for screen readers and the card: the first and the latest week. */
export function describeRankHistory(points: readonly RankHistoryPoint[]): string {
  const first = points[0]
  const last = points.at(-1)
  if (!first || !last) return ''
  if (first === last) return `Median rank ${weekText(first)}.`
  return `Median rank ${weekText(first)} and ${weekText(last)}.`
}

/** The bottom of the rank axis: the worst rank any week's 95% band reaches, at least 1. */
export function rankAxisMax(points: readonly RankHistoryPoint[]): number {
  return Math.max(1, ...points.map((point) => point.high95 ?? point.median ?? 1))
}
