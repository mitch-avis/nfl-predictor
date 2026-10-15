import { describe, expect, it } from 'vitest'

import type { SeasonTable } from '@/api/types'

import { describeRatingPair, neighborId, parseRatingPairs, type RatingPair } from './ratingPairs'

function pairs(kind: 'teams' | 'qbs', rows: SeasonTable['table']['rows']): SeasonTable {
  return { kind, season: 2025, entity_id: 'KC', table: { rows, visible_columns: [], column_groups: {}, column_metadata: {} } }
}

const pair = (overrides: Partial<RatingPair>): RatingPair => ({
  otherId: 'BUF',
  aboveChance: 0.79,
  gapLow: -1.2,
  gapMid: 2.46,
  gapHigh: 6.04,
  share: 1,
  ...overrides,
})

describe('parseRatingPairs', () => {
  it('reads the chance and the rating-gap percentiles of each pair', () => {
    // Arrange
    const table = pairs('teams', [
      { team: 'KC', other_team: 'BUF', team_rated_above_probability: 0.2, team_rating_gap_q025: -6, team_rating_gap_q500: -2.5, team_rating_gap_q975: 1, team_pair_share: 1 },
    ])

    // Act
    const [parsed] = parseRatingPairs('teams', table)

    // Assert
    expect(parsed).toEqual({ otherId: 'BUF', aboveChance: 0.2, gapLow: -6, gapMid: -2.5, gapHigh: 1, share: 1 })
  })

  it('reads a quarterback pair by the other passer’s id', () => {
    // Arrange
    const table = pairs('qbs', [{ qb_id: 'a', other_qb_id: 'b', qb_rated_above_probability: 0.6, qb_pair_share: 0.9 }])

    // Act
    const [parsed] = parseRatingPairs('qbs', table)

    // Assert
    expect([parsed.otherId, parsed.aboveChance, parsed.share, parsed.gapMid]).toEqual(['b', 0.6, 0.9, null])
  })
})

describe('describeRatingPair', () => {
  it('says how often the team was rated above the other and by how much', () => {
    // Act
    const text = describeRatingPair('teams', 'Kansas City Chiefs', 'Buffalo Bills', pair({}))

    // Assert
    expect(text).toBe(
      'Kansas City Chiefs rated above Buffalo Bills in 79% of redraws. Typical gap (Kansas City Chiefs minus Buffalo Bills): +2.5 points per game; 95% of redraws: -1.2 to +6.0.',
    )
  })

  it('counts a quarterback pair among the redraws with both and says how many those were', () => {
    // Act
    const text = describeRatingPair('qbs', 'J.Allen', 'P.Mahomes', pair({ gapLow: -0.051, gapMid: 0.02, gapHigh: 0.0904, share: 0.9 }))

    // Assert
    expect(text).toBe(
      'J.Allen rated above P.Mahomes in 79% of the redraws with both. Typical gap (J.Allen minus P.Mahomes): +0.020 EPA per dropback; 95% of redraws: -0.051 to +0.090. Both appeared in 90% of redraws.',
    )
  })

  it('says when the two never shared a redraw', () => {
    // Act
    const text = describeRatingPair('qbs', 'J.Allen', 'M.Backup', pair({ aboveChance: null }))

    // Assert
    expect(text).toBe('J.Allen and M.Backup never appeared in the same redraw.')
  })
})

describe('neighborId', () => {
  it.each([
    ['the one ranked just above', ['A', 'B', 'C'], 'B', 'A'],
    ['the runner-up for the leader', ['A', 'B', 'C'], 'A', 'B'],
    ['nothing for a unit not ranked', ['A', 'B'], 'Z', null],
    ['nothing for a unit alone', ['A'], 'A', null],
  ] as const)('picks %s', (_name, ranked, id, expected) => {
    // Act
    const neighbor = neighborId(ranked, id)

    // Assert
    expect(neighbor).toBe(expected)
  })
})
