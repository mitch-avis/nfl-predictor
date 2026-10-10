import { describe, expect, it } from 'vitest'

import type { RankRangeRow, RankRangeTable } from '@/api/types'

import {
  belowQualifierText,
  formatChance,
  middleRankText,
  opponentRankRanges,
  ordinal,
  parseRankRanges,
  parseUnitRankRanges,
  rankCenter,
  rankChanceText,
  rankHistogram,
  rankRangeHeadline,
  rankRangesByMedian,
  rankSpan,
  rankTicks,
} from './rankRanges'

function quantiles(base: string, values: number[]): RankRangeRow {
  const keys = ['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975']
  return Object.fromEntries(keys.map((key, index) => [`${base}_${key}`, values[index]]))
}

function teamRow(team: string, rank: number, ranks: number[], extra: RankRangeRow = {}): RankRangeRow {
  return {
    team,
    team_rank: rank,
    ...quantiles('team_rank', ranks),
    ...quantiles('team_rating', [1, 2, 3, 4, 5, 6, 7]),
    team_rank_top5_probability: 0.62,
    team_rank_top10_probability: 0.97,
    team_rank_missing_share: 0,
    team_rank_probabilities: [0.1, 0.3, 0.4, 0.2],
    ...extra,
  }
}

function payload(rows: RankRangeRow[]): RankRangeTable {
  return {
    kind: 'teams',
    season: 2025,
    entity_id: null,
    table: { rows, visible_columns: [], column_groups: {}, column_metadata: {} },
    team_names: {},
  }
}

describe('parseRankRanges', () => {
  it('reads each row’s published rank, percentiles, chances and rank probabilities', () => {
    // Arrange
    const table = payload([teamRow('KC', 2, [1, 1, 2, 3, 3, 4, 4])])

    // Act
    const [range] = parseRankRanges('teams', table)

    // Assert
    expect(range.id).toBe('KC')
    expect(range.publishedRank).toBe(2)
    expect(range.rank.q250).toBe(2)
    expect(range.rank.q975).toBe(4)
    expect(range.rating.q500).toBe(4)
    expect([range.top5, range.top10, range.missingShare]).toEqual([0.62, 0.97, 0])
    expect(range.probabilities).toEqual([0.1, 0.3, 0.4, 0.2])
  })

  it('names a quarterback and keeps his team', () => {
    // Arrange
    const table = payload([{ qb_id: '00-1', qb_name: 'J.Allen', team: 'BUF', qb_rank: 1, ...quantiles('qb_rank', [1, 1, 1, 1, 2, 2, 3]) }])

    // Act
    const [range] = parseRankRanges('qbs', table)

    // Assert
    expect([range.id, range.label, range.team]).toEqual(['00-1', 'J.Allen', 'BUF'])
  })
})

describe('rankRangesByMedian', () => {
  it('orders by median rank, the published rank breaking ties', () => {
    // Arrange
    const ranges = parseRankRanges(
      'teams',
      payload([teamRow('A', 1, [1, 1, 2, 3, 3, 4, 4]), teamRow('B', 2, [1, 1, 1, 2, 2, 3, 3]), teamRow('C', 3, [2, 2, 3, 3, 4, 4, 4])]),
    )

    // Act
    const ordered = rankRangesByMedian(ranges)

    // Assert
    expect(ordered.map((range) => range.id)).toEqual(['B', 'A', 'C'])
  })
})

describe('rank text', () => {
  it.each([
    [1, '1st'],
    [2, '2nd'],
    [3, '3rd'],
    [4, '4th'],
    [11, '11th'],
    [12, '12th'],
    [13, '13th'],
    [21, '21st'],
    [32, '32nd'],
  ])('writes %i as %s', (rank, expected) => {
    // Act
    const text = ordinal(rank)

    // Assert
    expect(text).toBe(expected)
  })

  it('writes the middle 50% and 95% beside the published rank', () => {
    // Arrange
    const [range] = parseRankRanges('teams', payload([teamRow('KC', 6, [1, 2, 4, 6, 8, 12, 16])]))

    // Act
    const headline = rankRangeHeadline(range)

    // Assert
    expect(headline).toBe('6th; middle 50%: 4th–8th; 95%: 1st–16th')
  })

  it('writes the middle 50% of redrawn ranks', () => {
    // Arrange
    const [range] = parseRankRanges('teams', payload([teamRow('KC', 6, [1, 2, 4, 6, 8, 12, 16])]))

    // Act
    const text = middleRankText(range)

    // Assert
    expect(text).toBe('4th–8th')
  })

  it('writes one rank when both ends of a range agree', () => {
    // Arrange
    const [range] = parseRankRanges('teams', payload([teamRow('KC', 1, [1, 1, 1, 1, 1, 1, 1])]))

    // Act
    const text = middleRankText(range)

    // Assert
    expect(text).toBe('1st')
  })

  it.each([
    [0.001, '<1%'],
    [0.999, '>99%'],
    [0.62, '62%'],
    [0, '0%'],
    [1, '100%'],
  ])('writes a share of %s as %s', (share, expected) => {
    // Act
    const text = formatChance(share)

    // Assert
    expect(text).toBe(expected)
  })

  it('adds how often the team was left out of the redraws', () => {
    // Arrange
    const [range] = parseRankRanges('teams', payload([teamRow('KC', 2, [1, 1, 2, 3, 3, 4, 4], { team_rank_missing_share: 0.02 })]))

    // Act
    const text = rankChanceText('teams', range)

    // Assert
    expect(text).toBe('Top 5 in 62% of redraws, top 10 in 97%; left out of 2% (no games drawn)')
  })
})

describe('track geometry', () => {
  it('spans ranks on a track whose left edge is rank 1', () => {
    // Act
    const span = rankSpan(2, 3, 4)

    // Assert
    expect(span).toEqual({ left: 25, width: 50 })
  })

  it('centers a rank in its cell', () => {
    // Act
    const center = rankCenter(1, 4)

    // Assert
    expect(center).toBe(12.5)
  })

  it.each([
    [32, 5, [1, 5, 10, 15, 20, 25, 32]],
    [4, 5, [1, 4]],
    [12, 5, [1, 5, 12]],
  ])('ticks %i ranks every %i', (count, step, expected) => {
    // Act
    const ticks = rankTicks(count, step)

    // Assert
    expect(ticks).toEqual(expected)
  })
})

describe('rankHistogram', () => {
  it('lists the chance of each rank and marks the published one', () => {
    // Arrange
    const [range] = parseRankRanges('teams', payload([teamRow('KC', 3, [1, 1, 2, 3, 3, 4, 4])]))

    // Act
    const bars = rankHistogram(range)

    // Assert
    expect(bars.map((bar) => [bar.rank, bar.published])).toEqual([
      [1, false],
      [2, false],
      [3, true],
      [4, false],
    ])
  })
})

describe('unit and opponent ranges', () => {
  const row = teamRow('KC', 2, [1, 1, 2, 3, 3, 4, 4], {
    offense_rank: 1,
    ...quantiles('offense_rank', [1, 1, 1, 1, 2, 2, 3]),
    defense_rank: 4,
    ...quantiles('defense_rank', [2, 3, 3, 4, 4, 4, 4]),
  })

  it('reads a team’s unit ranges in unit order, skipping units the season lacks', () => {
    // Act
    const units = parseUnitRankRanges(payload([row]), 'KC')

    // Assert
    expect(units.map((unit) => [unit.label, unit.publishedRank])).toEqual([
      ['Offense', 1],
      ['Defense', 4],
    ])
  })

  it('gives a quarterback page each team’s defense range as an opponent', () => {
    // Act
    const ranges = opponentRankRanges('qbs', payload([row]))

    // Assert
    expect(ranges.get('KC')?.publishedRank).toBe(4)
  })

  it('gives a team page each team’s own range as an opponent', () => {
    // Act
    const ranges = opponentRankRanges('teams', payload([row]))

    // Assert
    expect(ranges.get('KC')?.publishedRank).toBe(2)
  })
})

describe('belowQualifierText', () => {
  it('says how far a quarterback below the qualifier is from it', () => {
    // Act
    const text = belowQualifierText({ qb_is_eligible: false, qb_attempts_total: 25, qb_attempt_qualifier: 42 })

    // Assert
    expect(text).toBe('Not ranked: below the qualifier (25 of 42 pass attempts), so no rank range.')
  })

  it('says nothing for a qualifier', () => {
    // Act
    const text = belowQualifierText({ qb_is_eligible: true })

    // Assert
    expect(text).toBeNull()
  })
})
