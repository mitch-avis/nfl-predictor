/** Head-to-head chances: how often one team or quarterback was rated above another across redraws. */
import type { EntityKind, Row, SeasonTable } from '@/api/types'

/** One compared team or quarterback: how often the subject was rated above it, and by how much. */
export interface RatingPair {
  otherId: string
  aboveChance: number | null
  gapLow: number | null
  gapMid: number | null
  gapHigh: number | null
  share: number | null
}

const COLUMNS: Record<EntityKind, { other: string; above: string; gap: string; share: string }> = {
  teams: { other: 'other_team', above: 'team_rated_above_probability', gap: 'team_rating_gap', share: 'team_pair_share' },
  qbs: { other: 'other_qb_id', above: 'qb_rated_above_probability', gap: 'qb_rating_gap', share: 'qb_pair_share' },
}

/** Decimals and unit of a rating gap: points per game for teams, EPA per dropback for passers. */
const GAP_FORMAT: Record<EntityKind, { decimals: number; unit: string }> = {
  teams: { decimals: 1, unit: 'points per game' },
  qbs: { decimals: 3, unit: 'EPA per dropback' },
}

function numberOrNull(value: Row[string] | undefined): number | null {
  return typeof value === 'number' ? value : null
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`
}

/** A signed number such as `+2.5` or `-1.2`; one that rounds to zero shows unsigned. */
export function signed(value: number | null, decimals: number): string {
  if (value === null) return '—'
  const rounded = Number(value.toFixed(decimals))
  const text = Math.abs(rounded).toFixed(decimals)
  if (rounded === 0) return text
  return rounded > 0 ? `+${text}` : `-${text}`
}

/** Read one subject's head-to-head payload into typed pairs, in the payload's order. */
export function parseRatingPairs(kind: EntityKind, payload: SeasonTable): RatingPair[] {
  const columns = COLUMNS[kind]
  return payload.table.rows.map((row) => ({
    otherId: String(row[columns.other] ?? ''),
    aboveChance: numberOrNull(row[columns.above]),
    gapLow: numberOrNull(row[`${columns.gap}_q025`]),
    gapMid: numberOrNull(row[`${columns.gap}_q500`]),
    gapHigh: numberOrNull(row[`${columns.gap}_q975`]),
    share: numberOrNull(row[columns.share]),
  }))
}

/**
 * One sentence: how often `subject` was rated above `other` across the redraws, the typical rating
 * gap and its 95% range. Quarterback chances count only the redraws with both passers, so the
 * sentence says how many those were when it is not all of them.
 */
export function describeRatingPair(kind: EntityKind, subject: string, other: string, pair: RatingPair): string {
  if (pair.aboveChance === null) return `${subject} and ${other} never appeared in the same redraw.`
  const { decimals, unit } = GAP_FORMAT[kind]
  const among = kind === 'qbs' ? 'the redraws with both' : 'redraws'
  const sentence =
    `${subject} rated above ${other} in ${percent(pair.aboveChance)} of ${among}. Typical gap ` +
    `(${subject} minus ${other}): ${signed(pair.gapMid, decimals)} ${unit}; 95% of redraws: ${signed(pair.gapLow, decimals)} to ${signed(pair.gapHigh, decimals)}.`
  if (pair.share === null || pair.share >= 1) return sentence
  return `${sentence} Both appeared in ${percent(pair.share)} of redraws.`
}

/** The default comparison: the one ranked just above, or the runner-up for the leader. */
export function neighborId(rankedIds: readonly string[], id: string): string | null {
  const index = rankedIds.indexOf(id)
  if (index === -1 || rankedIds.length < 2) return null
  return rankedIds[index === 0 ? 1 : index - 1] ?? null
}
