import type { JobLogLine } from '@/api/types'

/** Levels in severity order; picking one shows it and everything above it. */
export const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR'] as const
export type Level = (typeof LEVELS)[number]

const RANK: Record<string, number> = { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 4 }

/** Rendering every line of a long walk-forward run would stall the page; keep the newest ones. */
export const MAX_RENDERED = 2000

/** Filter log lines to those at or above `level`. */
export function filterLines(lines: JobLogLine[], level: Level): JobLogLine[] {
  const floor = RANK[level]
  return lines.filter((line) => (RANK[line.level] ?? RANK.INFO) >= floor)
}
