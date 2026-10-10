import { describe, expect, it } from 'vitest'

import type { SheetRow } from '@/api/types'

import { matchupLabel, sheetText, sortSheetRows, weekRanges } from './poolFormat'

function row(overrides: Partial<SheetRow>): SheetRow {
  return {
    game_id: 'g',
    points: 1,
    pick: 'DAL',
    opponent: 'TB',
    pick_is_home: true,
    away_abbr: 'TB',
    home_abbr: 'DAL',
    kickoff: '2026-10-08T20:15:00-04:00',
    pick_win_prob: 0.7928,
    expected_points: 0.7928,
    away_score: null,
    home_score: null,
    result: null,
    points_won: null,
    newer_points: null,
    newer_pick: null,
    ...overrides,
  }
}

const rows: SheetRow[] = [
  row({ game_id: 'late', points: 3, kickoff: '2026-10-11T16:25:00-04:00' }),
  row({ game_id: 'thursday', points: 1, kickoff: '2026-10-08T20:15:00-04:00' }),
  row({ game_id: 'early', points: 2, kickoff: '2026-10-11T13:00:00-04:00' }),
]

describe('sortSheetRows', () => {
  it('orders by points, most first', () => {
    // Act
    const sorted = sortSheetRows(rows, 'points')

    // Assert
    expect(sorted.map((r) => r.game_id)).toEqual(['late', 'early', 'thursday'])
  })

  it('orders by kickoff, the way pool sites list the games', () => {
    // Act
    const sorted = sortSheetRows(rows, 'kickoff')

    // Assert
    expect(sorted.map((r) => r.game_id)).toEqual(['thursday', 'early', 'late'])
  })
})

describe('sheetText', () => {
  it('writes one tab-separated line per game: points, pick, matchup, chance', () => {
    // Arrange
    const sheet = [row({ points: 3 }), row({ points: 2, pick: 'HOU', opponent: 'TEN', pick_is_home: false, away_abbr: 'HOU', home_abbr: 'TEN', pick_win_prob: 0.709 })]

    // Act
    const text = sheetText(sheet)

    // Assert
    expect(text).toBe('3\tDAL\tTB @ DAL\t79.3%\n2\tHOU\tHOU @ TEN\t70.9%')
  })
})

describe('matchupLabel', () => {
  it('reads a home pick as playing against the visitor', () => {
    // Act
    const label = matchupLabel(row({}))

    // Assert
    expect(label).toBe('vs TB')
  })

  it('reads an away pick as playing at the home team', () => {
    // Act
    const label = matchupLabel(row({ pick: 'TB', opponent: 'DAL', pick_is_home: false }))

    // Assert
    expect(label).toBe('at DAL')
  })
})

describe('weekRanges', () => {
  it('joins consecutive weeks into ranges', () => {
    // Act
    const text = weekRanges([5, 6, 7, 9, 11, 12])

    // Assert
    expect(text).toBe('5–7, 9, 11–12')
  })

  it('says none for no weeks', () => {
    // Act
    const text = weekRanges([])

    // Assert
    expect(text).toBe('none')
  })
})
