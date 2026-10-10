import type { SheetRow } from '@/api/types'
import { formatPercent } from '@/utils/format'

export type SheetOrder = 'points' | 'kickoff'

function kickoffTime(row: SheetRow): number {
  const time = row.kickoff ? new Date(row.kickoff).getTime() : Number.NaN
  return Number.isNaN(time) ? Number.POSITIVE_INFINITY : time
}

/** The sheet by points (most first) or by kickoff (the order pool sites list the games in). */
export function sortSheetRows(rows: SheetRow[], order: SheetOrder): SheetRow[] {
  const sorted = [...rows]
  if (order === 'points') sorted.sort((a, b) => (b.points ?? 0) - (a.points ?? 0) || kickoffTime(a) - kickoffTime(b))
  else sorted.sort((a, b) => kickoffTime(a) - kickoffTime(b) || (b.points ?? 0) - (a.points ?? 0))
  return sorted
}

/** Tab-separated lines for the clipboard: points, pick, matchup and the pick's chance. */
export function sheetText(rows: SheetRow[]): string {
  return rows
    .map((r) => `${r.points ?? ''}\t${r.pick ?? ''}\t${r.away_abbr} @ ${r.home_abbr}\t${formatPercent(r.pick_win_prob, 1)}`)
    .join('\n')
}

/** Where the pick plays: `vs` the visitor for a home pick, `at` the home team for an away one. */
export function matchupLabel(row: SheetRow): string {
  if (row.pick === null || row.opponent === null) return `${row.away_abbr} @ ${row.home_abbr}`
  return row.pick_is_home ? `vs ${row.opponent}` : `at ${row.opponent}`
}

/** A kickoff in the viewer's time zone with its weekday: `Thu, Oct 8, 8:15 PM`. */
export function formatKickoff(value: string | null): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString(undefined, { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}

/** Week numbers as ranges: `5–7, 9`. */
export function weekRanges(weeks: number[]): string {
  if (weeks.length === 0) return 'none'
  const sorted = [...weeks].sort((a, b) => a - b)
  const parts: string[] = []
  let start = sorted[0]
  let previous = sorted[0]
  for (const week of [...sorted.slice(1), Number.NaN]) {
    if (week === previous + 1) {
      previous = week
      continue
    }
    parts.push(start === previous ? String(start) : `${start}–${previous}`)
    start = week
    previous = week
  }
  return parts.join(', ')
}
