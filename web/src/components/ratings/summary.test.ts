import { describe, expect, it } from 'vitest'

import type { MetricColumnMeta } from '@/api/types'

import { rankAmong, summaryTiles } from './summary'

function meta(key: string, overrides: Partial<MetricColumnMeta> = {}): MetricColumnMeta {
  return {
    key,
    label: key,
    description: '',
    group: 'ratings',
    kind: 'float',
    polarity: 'higher',
    heatmap: true,
    decimals: null,
    actionable: true,
    sticky: false,
    full_name: key,
    category: null,
    subcategory: null,
    contextual: false,
    shape: 'score',
    percent: false,
    ...overrides,
  }
}

const METADATA: Record<string, MetricColumnMeta> = {
  team_rating: meta('team_rating'),
  offense_rating: meta('offense_rating'),
  sos: meta('sos', { contextual: true }),
  epa_margin_per_play: meta('epa_margin_per_play', { shape: 'rate' }),
}

const ROWS = [
  { team: 'BUF', team_rating: 6.5, offense_rating: 4.25, sos: 0.5, epa_margin_per_play: 0.12 },
  { team: 'KC', team_rating: 4, offense_rating: 3, sos: -0.2, epa_margin_per_play: 0.14 },
  { team: 'NE', team_rating: 4, offense_rating: 1, sos: 0.1, epa_margin_per_play: 0.01 },
]

describe('rankAmong', () => {
  it.each([
    ['a higher-is-better value', [3, 1, 2], 2, 'higher', { rank: 2, count: 3 }],
    ['a lower-is-better value', [3, 1, 2], 2, 'lower', { rank: 2, count: 3 }],
    ['tied values sharing the better rank', [4, 4, 6], 4, 'higher', { rank: 2, count: 3 }],
    ['a neutral column', [1, 2], 1, 'neutral', null],
    ['a missing value', [1, 2], null, 'higher', null],
  ] as const)('ranks %s', (_name, values, value, polarity, expected) => {
    // Act
    const rank = rankAmong([...values], value, polarity)

    // Assert
    expect(rank).toEqual(expected)
  })
})

describe('summaryTiles', () => {
  it('lists the ratings with their ranks, headline first with its unadjusted value', () => {
    // Act
    const tiles = summaryTiles('teams', ROWS[2], ROWS, METADATA)

    // Assert
    expect(tiles.map((tile) => [tile.column, tile.rank])).toEqual([
      ['team_rating', { rank: 2, count: 3 }],
      ['offense_rating', { rank: 3, count: 3 }],
      ['sos', null],
    ])
    expect(tiles[0].unadjusted).toEqual({ column: 'epa_margin_per_play', value: 0.01, rank: { rank: 3, count: 3 } })
  })

  it('ranks a quarterback among the qualifiers only', () => {
    // Arrange
    const qbMeta = { adj_qb_epa_per_dropback: meta('adj_qb_epa_per_dropback') }
    const rows = [
      { qb_id: 'a', adj_qb_epa_per_dropback: 0.2, qb_is_eligible: true },
      { qb_id: 'b', adj_qb_epa_per_dropback: 0.4, qb_is_eligible: false },
      { qb_id: 'c', adj_qb_epa_per_dropback: 0.1, qb_is_eligible: true },
    ]

    // Act
    const tiles = summaryTiles('qbs', rows[0], rows, qbMeta)

    // Assert
    expect(tiles[0].rank).toEqual({ rank: 1, count: 2 })
  })

  it('leaves a quarterback below the qualifier unranked', () => {
    // Arrange
    const qbMeta = { adj_qb_epa_per_dropback: meta('adj_qb_epa_per_dropback') }
    const rows = [{ qb_id: 'b', adj_qb_epa_per_dropback: 0.4, qb_is_eligible: false }]

    // Act
    const tiles = summaryTiles('qbs', rows[0], rows, qbMeta)

    // Assert
    expect(tiles[0].rank).toBeNull()
  })
})
