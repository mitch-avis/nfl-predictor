import { describe, expect, it } from 'vitest'

import {
  formatMoneyline,
  formatNumber,
  formatPercent,
  formatRelative,
  formatSigned,
  humanize,
  seasonWeekLabel,
  shortHash,
} from './format'

describe('format helpers', () => {
  it('formats numbers and percentages with placeholders for missing values', () => {
    expect(formatNumber(3.14159, 2)).toBe('3.14')
    expect(formatNumber(null)).toBe('—')
    expect(formatPercent(0.4023)).toBe('40%')
    expect(formatPercent(0.4023, 1)).toBe('40.2%')
    expect(formatPercent(undefined)).toBe('—')
  })

  it('formats signed spreads and moneylines', () => {
    expect(formatSigned(3.5)).toBe('+3.5')
    expect(formatSigned(-1)).toBe('-1.0')
    expect(formatSigned(0)).toBe('PK')
    expect(formatMoneyline(130)).toBe('+130')
    expect(formatMoneyline(-155)).toBe('-155')
    expect(formatMoneyline(null)).toBe('—')
  })

  it('formats relative times', () => {
    const now = new Date('2026-09-10T12:00:00Z')
    expect(formatRelative('2026-09-10T11:59:30Z', now)).toBe('30s ago')
    expect(formatRelative('2026-09-10T11:30:00Z', now)).toBe('30 min ago')
    expect(formatRelative('2026-09-10T06:00:00Z', now)).toBe('6 hr ago')
    expect(formatRelative('2026-09-08T12:00:00Z', now)).toBe('2 days ago')
    expect(formatRelative('2026-09-11T12:00:00Z', now)).toBe('1 days from now')
    expect(formatRelative('garbage', now)).toBe('garbage')
  })

  it('humanizes keys, hashes, and season labels', () => {
    expect(humanize('home_win_prob')).toBe('Home win prob')
    expect(shortHash('5b6af6aa0000')).toBe('5b6af6aa')
    expect(shortHash(null)).toBe('—')
    expect(seasonWeekLabel(2026, 1)).toBe('2026 · Week 1')
    expect(seasonWeekLabel(2026, null)).toBe('2026')
    expect(seasonWeekLabel(null, null)).toBe('—')
  })
})
