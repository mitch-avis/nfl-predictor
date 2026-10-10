/**
 * Rank ranges: how much a team's or quarterback's rank moves when the season's games are redrawn
 * at random (with repeats) and everyone is re-rated each time. The API carries every number; this
 * module reads the payload, orders it, and turns it into text and track geometry.
 */
import type { EntityKind, RankRangeRow, RankRangeTable, Row } from '@/api/types'

export const QUANTILE_KEYS = ['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975'] as const
export type QuantileKey = (typeof QUANTILE_KEYS)[number]
type Quantiles = Record<QuantileKey, number | null>

/** One team's or quarterback's rank range, read from one payload row. */
export interface RankRange {
  id: string
  label: string
  team: string | null
  publishedRank: number
  rank: Quantiles
  rating: Quantiles
  top5: number
  top10: number
  missingShare: number
  probabilities: number[]
}

/** A team's offense, defense or special-teams rank range, carried on its rank-range row. */
export interface UnitRankRange {
  unit: string
  label: string
  publishedRank: number
  rank: Quantiles
  rating: Quantiles
}

const COLUMNS: Record<EntityKind, { id: string; label: string; rank: string; rating: string }> = {
  teams: { id: 'team', label: 'team', rank: 'team_rank', rating: 'team_rating' },
  qbs: { id: 'qb_id', label: 'qb_name', rank: 'qb_rank', rating: 'adj_qb_epa_per_dropback' },
}

const UNITS = [
  { key: 'offense', label: 'Offense' },
  { key: 'defense', label: 'Defense' },
  { key: 'special_teams', label: 'Special teams' },
] as const

function numberOrNull(value: RankRangeRow[string] | undefined): number | null {
  return typeof value === 'number' ? value : null
}

function quantiles(row: RankRangeRow, base: string): Quantiles {
  return Object.fromEntries(QUANTILE_KEYS.map((key) => [key, numberOrNull(row[`${base}_${key}`])])) as Quantiles
}

/** Read every row of a rank-range payload, in its (published-rank) order. */
export function parseRankRanges(kind: EntityKind, payload: RankRangeTable): RankRange[] {
  const columns = COLUMNS[kind]
  return payload.table.rows.map((row) => {
    const probabilities = row[`${columns.rank}_probabilities`]
    return {
      id: String(row[columns.id] ?? ''),
      label: String(row[columns.label] ?? row[columns.id] ?? ''),
      team: kind === 'qbs' && typeof row.team === 'string' ? row.team : null,
      publishedRank: numberOrNull(row[columns.rank]) ?? 0,
      rank: quantiles(row, columns.rank),
      rating: quantiles(row, columns.rating),
      top5: numberOrNull(row[`${columns.rank}_top5_probability`]) ?? 0,
      top10: numberOrNull(row[`${columns.rank}_top10_probability`]) ?? 0,
      missingShare: numberOrNull(row[`${columns.rank}_missing_share`]) ?? 0,
      probabilities: Array.isArray(probabilities) ? probabilities : [],
    }
  })
}

/** The unit rank ranges on one team's row, in unit order; empty for seasons built without them. */
export function parseUnitRankRanges(payload: RankRangeTable, teamId: string): UnitRankRange[] {
  const row = payload.table.rows.find((candidate) => candidate.team === teamId)
  if (!row) return []
  return UNITS.flatMap(({ key, label }) => {
    const publishedRank = numberOrNull(row[`${key}_rank`])
    if (publishedRank === null) return []
    return [{ unit: key, label, publishedRank, rank: quantiles(row, `${key}_rank`), rating: quantiles(row, `${key}_rating`) }]
  })
}

/**
 * Each team's season-long rank range as an opponent, by team: the team's own on team pages, its
 * defense's on quarterback pages (none when the season has no unit ranges).
 */
export function opponentRankRanges(kind: EntityKind, payload: RankRangeTable): Map<string, Pick<RankRange, 'publishedRank' | 'rank'>> {
  if (kind === 'teams') return new Map(parseRankRanges('teams', payload).map((range) => [range.id, range]))
  return new Map(
    payload.table.rows.flatMap((row) => {
      const team = String(row.team ?? '')
      const defense = parseUnitRankRanges(payload, team).find((unit) => unit.unit === 'defense')
      return defense ? [[team, defense] as const] : []
    }),
  )
}

/** Order ranges for the league chart: median rank first, the published rank breaking ties. */
export function rankRangesByMedian(ranges: RankRange[]): RankRange[] {
  const median = (range: RankRange) => range.rank.q500 ?? Number.POSITIVE_INFINITY
  return [...ranges].sort((a, b) => median(a) - median(b) || a.publishedRank - b.publishedRank)
}

/** `1st`, `2nd`, `3rd`, `4th`, ..., `11th`, ..., `21st`. */
export function ordinal(rank: number): string {
  const lastTwo = rank % 100
  if (lastTwo >= 11 && lastTwo <= 13) return `${rank}th`
  const suffix = ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[rank % 10] ?? 'th'
  return `${rank}${suffix}`
}

function rankRangeText(low: number | null, high: number | null): string {
  if (low === null || high === null) return 'n/a'
  return low === high ? ordinal(low) : `${ordinal(low)}–${ordinal(high)}`
}

/** The middle 50% of redrawn ranks, such as `3rd–9th`, or one rank when both ends agree. */
export function middleRankText(range: Pick<RankRange, 'rank'>): string {
  return rankRangeText(range.rank.q250, range.rank.q750)
}

/** The detail page's headline, for example `6th; middle 50%: 4th–8th; 95%: 1st–16th`. */
export function rankRangeHeadline(range: Pick<RankRange, 'publishedRank' | 'rank'>): string {
  const published = ordinal(range.publishedRank)
  if (range.rank.q500 === null) return `${published}; not ranked in any redraw`
  return [published, `middle 50%: ${rankRangeText(range.rank.q250, range.rank.q750)}`, `95%: ${rankRangeText(range.rank.q025, range.rank.q975)}`].join('; ')
}

/** A one-line text alternative for one interval in the league chart. */
export function rankRangeSummary(range: RankRange): string {
  const median = range.rank.q500 === null ? 'n/a' : ordinal(range.rank.q500)
  return `${range.label}: published rank ${ordinal(range.publishedRank)}, median ${median}; middle 50%: ${rankRangeText(range.rank.q250, range.rank.q750)}; 95%: ${rankRangeText(range.rank.q025, range.rank.q975)}`
}

/** A share of redraws as a percent that never rounds a possible outcome to 0% or 100%. */
export function formatChance(share: number): string {
  if (share > 0 && share < 0.005) return '<1%'
  if (share < 1 && share > 0.995) return '>99%'
  return `${Math.round(share * 100)}%`
}

/** The top-5 and top-10 chances, plus how often a redraw left the team or quarterback out. */
export function rankChanceText(kind: EntityKind, range: RankRange): string {
  const text = `Top 5 in ${formatChance(range.top5)} of redraws, top 10 in ${formatChance(range.top10)}`
  if (range.missingShare <= 0) return text
  const missing = kind === 'qbs' ? 'no dropbacks' : 'no games'
  return `${text}; left out of ${formatChance(range.missingShare)} (${missing} drawn)`
}

/** Left offset and width, in percent of the track, of ranks `low` through `high` out of `count`. */
export function rankSpan(low: number, high: number, count: number): { left: number; width: number } {
  return { left: ((low - 1) / count) * 100, width: ((high - low + 1) / count) * 100 }
}

/** The center of one rank's cell, in percent of the track. */
export function rankCenter(rank: number, count: number): number {
  return ((rank - 0.5) / count) * 100
}

/**
 * Axis ticks for `count` ranks: 1, every `step`-th rank, then the last. A tick within three ranks
 * of the last is dropped so the labels stay apart on a phone-width track.
 */
export function rankTicks(count: number, step = 5): number[] {
  const ticks = [1]
  for (let rank = step; rank < count; rank += step) {
    if (count - rank >= 4) ticks.push(rank)
  }
  if (count > 1) ticks.push(count)
  return ticks
}

/** The P(rank = k) bars of the detail page, the published rank marked. */
export function rankHistogram(range: RankRange): Array<{ rank: number; probability: number; published: boolean }> {
  return range.probabilities.map((probability, index) => ({ rank: index + 1, probability, published: index + 1 === range.publishedRank }))
}

/** For a quarterback below the qualifier, the sentence that says why he has no rank range. */
export function belowQualifierText(row: Row): string | null {
  if (row.qb_is_eligible !== false) return null
  const attempts = row.qb_attempts_total
  const needed = row.qb_attempt_qualifier
  const detail = typeof attempts === 'number' && typeof needed === 'number' ? `${attempts} of ${needed} pass attempts` : 'too few pass attempts'
  return `Not ranked: below the qualifier (${detail}), so no rank range.`
}
