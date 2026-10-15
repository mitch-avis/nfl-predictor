import { describe, expect, it } from 'vitest'

import { buildTrendPoints, meanReference, niceTicks, trendColumns } from './trend'

const rows = [
  { week: 2, opponent_team: 'KC', points_for: 31, completion_pct: 0.7 },
  { week: 1, opponent_team: 'NE', points_for: 17, completion_pct: null },
  { week: 3, opponent_team: 'WSH', points_for: 24, completion_pct: 0.64 },
]

describe('buildTrendPoints', () => {
  it('lists one column by week, skipping games without a value', () => {
    // Act
    const points = buildTrendPoints(rows, 'completion_pct')

    // Assert
    expect(points).toEqual([
      { week: 2, value: 0.7, opponent: 'KC' },
      { week: 3, value: 0.64, opponent: 'WSH' },
    ])
  })
})

describe('meanReference', () => {
  it('draws the mean of the points with its caption', () => {
    // Arrange
    const points = buildTrendPoints(rows, 'points_for')

    // Act
    const reference = meanReference(points, (value) => value.toFixed(1))

    // Assert
    expect(reference).toEqual({ value: 24, caption: 'Dashed line: the average of the games shown (24.0).' })
  })

  it('draws nothing without points', () => {
    // Act
    const reference = meanReference([])

    // Assert
    expect(reference).toBeNull()
  })
})

describe('niceTicks', () => {
  it.each([
    [-3.6, 4.2, [-4, -2, 0, 2, 4, 6]],
    [0.12, 0.31, [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]],
    [5, 5, [2, 4, 6, 8]],
  ])('covers %s to %s with round steps', (min, max, expected) => {
    // Act
    const ticks = niceTicks(min, max)

    // Assert
    expect(ticks).toEqual(expected)
  })
})

describe('trendColumns', () => {
  it('offers the preferred columns with values first, then the rest, never the week', () => {
    // Act
    const columns = trendColumns(rows, ['week', 'completion_pct', 'points_for', 'turnover_margin'], ['points_for', 'missing'])

    // Assert
    expect(columns).toEqual(['points_for', 'completion_pct'])
  })
})
