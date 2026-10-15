import { describe, expect, it } from 'vitest'

import type { SeasonTable } from '@/api/types'

import { describeRankHistory, parseRankHistory, rankAxisMax, rankBandData } from './rankHistory'

function history(rows: SeasonTable['table']['rows']): SeasonTable {
  return { kind: 'teams', season: 2026, entity_id: 'KC', table: { rows, visible_columns: [], column_groups: {}, column_metadata: {} } }
}

const weeks = history([
  { week: 4, team: 'KC', team_rank: 5, team_rank_q025: 1, team_rank_q250: 3, team_rank_q500: 5, team_rank_q750: 8, team_rank_q975: 14 },
  { week: 3, team: 'KC', team_rank: 7, team_rank_q025: 2, team_rank_q250: 4, team_rank_q500: 7, team_rank_q750: 11, team_rank_q975: 20 },
])

describe('parseRankHistory', () => {
  it('reads each week’s median rank and its bands, in week order', () => {
    // Act
    const points = parseRankHistory('teams', weeks)

    // Assert
    expect(points.map((point) => [point.week, point.median, point.low50, point.high50, point.low95, point.high95])).toEqual([
      [3, 7, 4, 11, 2, 20],
      [4, 5, 3, 8, 1, 14],
    ])
  })
})

describe('rankBandData', () => {
  it('draws the bands as low-high pairs beside the median', () => {
    // Arrange
    const points = parseRankHistory('teams', weeks)

    // Act
    const data = rankBandData(points)

    // Assert
    expect(data[0]).toEqual({ week: 3, median: 7, band50: [4, 11], band95: [2, 20] })
  })
})

describe('describeRankHistory', () => {
  it('summarizes the first and the latest week', () => {
    // Arrange
    const points = parseRankHistory('teams', weeks)

    // Act
    const text = describeRankHistory(points)

    // Assert
    expect(text).toBe('Median rank 7th after week 3 (95%: 2nd–20th) and 5th after week 4 (95%: 1st–14th).')
  })

  it('says nothing without weeks', () => {
    // Act
    const text = describeRankHistory([])

    // Assert
    expect(text).toBe('')
  })
})

describe('rankAxisMax', () => {
  it('reaches the worst rank any week’s 95% band reaches', () => {
    // Arrange
    const points = parseRankHistory('teams', weeks)

    // Act
    const max = rankAxisMax(points)

    // Assert
    expect(max).toBe(20)
  })
})
