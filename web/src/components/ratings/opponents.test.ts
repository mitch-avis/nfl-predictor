import { describe, expect, it } from 'vitest'

import type { MetricColumnMeta, MetricTable } from '@/api/types'

import { opponentLedger } from './opponents'

function meta(key: string, overrides: Partial<MetricColumnMeta> = {}): MetricColumnMeta {
  return {
    key,
    label: key,
    description: `${key} described.`,
    group: 'results',
    kind: 'int',
    polarity: 'higher',
    heatmap: true,
    decimals: null,
    actionable: true,
    sticky: false,
    full_name: key,
    category: 'Overall',
    subcategory: null,
    contextual: false,
    shape: 'count',
    percent: false,
    ...overrides,
  }
}

const LOGS: MetricTable = {
  rows: [
    { week: 1, opponent_team: 'KC', is_home: true, points_for: 31, points_allowed: 17, point_margin: 14, win_value: 1, opp_team_rating: 4, opp_defense_rating: 1.5 },
    { week: 2, opponent_team: 'NE', is_home: false, points_for: 20, points_allowed: 20, point_margin: 0, win_value: 0.5, opp_team_rating: -2, opp_defense_rating: -1 },
    { week: 4, opponent_team: 'KC', is_home: false, points_for: 20, points_allowed: 27, point_margin: -7, win_value: 0, opp_team_rating: 4, opp_defense_rating: 1.5 },
  ],
  visible_columns: ['week', 'opponent_team', 'is_home', 'points_for', 'points_allowed', 'point_margin', 'win_value', 'opp_team_rating', 'opp_defense_rating'],
  column_groups: {},
  column_metadata: {
    opponent_team: meta('opponent_team', { kind: 'text', shape: 'id', polarity: 'neutral', group: 'identity' }),
    points_for: meta('points_for'),
    points_allowed: meta('points_allowed', { polarity: 'lower' }),
    point_margin: meta('point_margin'),
    opp_team_rating: meta('opp_team_rating', { kind: 'float', shape: 'score', contextual: true, group: 'opponent_ratings' }),
    opp_defense_rating: meta('opp_defense_rating', { kind: 'float', shape: 'score', contextual: true, group: 'opponent_ratings' }),
  },
}

const RANKED: MetricTable = {
  rows: [
    { week: 1, opponent_team: 'KC', win_value: 1, opp_team_rank: 4 },
    { week: 2, opponent_team: 'DEN', win_value: 0, opp_team_rank: 15 },
    { week: 3, opponent_team: 'NE', win_value: 1, opp_team_rank: 30 },
  ],
  visible_columns: ['week', 'opponent_team', 'win_value', 'opp_team_rank'],
  column_groups: {},
  column_metadata: {
    opponent_team: LOGS.column_metadata.opponent_team,
    opp_team_rank: meta('opp_team_rank', { label: 'Opp Rank', kind: 'int', shape: 'score', polarity: 'lower', contextual: true, group: 'opponent_ratings' }),
  },
}

describe('opponentLedger', () => {
  it('tiers each opponent by its league rank: the top 10 tougher, the bottom 10 softer', () => {
    // Act
    const ledger = opponentLedger(RANKED, 32)

    // Assert
    expect(ledger.rows.map((row) => [row.opponent_team, row.opp_team_rank, row.opp_tier])).toEqual([
      ['DEN', 15, 'Middle'],
      ['KC', 4, 'Tougher'],
      ['NE', 30, 'Softer'],
    ])
    expect(ledger.visible_columns).toEqual(['opponent_team', 'games', 'weeks', 'results', 'opp_team_rank', 'opp_tier'])
    expect(ledger.column_metadata.opp_tier.description).toContain('top 10')
  })

  it('leaves the tier out when the league size is not known', () => {
    // Act
    const ledger = opponentLedger(RANKED, null)

    // Assert
    expect(ledger.visible_columns).not.toContain('opp_tier')
  })

  it('gives each opponent one row: its games, weeks, results and the average score', () => {
    // Act
    const ledger = opponentLedger(LOGS, null)

    // Assert
    expect(ledger.rows).toEqual([
      { opponent_team: 'KC', games: 2, weeks: '1, 4', results: 'W, L', points_for: 25.5, points_allowed: 22, point_margin: 3.5, opp_team_rating: 4, opp_defense_rating: 1.5 },
      { opponent_team: 'NE', games: 1, weeks: '2', results: 'T', points_for: 20, points_allowed: 20, point_margin: 0, opp_team_rating: -2, opp_defense_rating: -1 },
    ])
  })

  it('describes every column, the averages as averages over the games against that opponent', () => {
    // Act
    const ledger = opponentLedger(LOGS, null)

    // Assert
    expect(ledger.visible_columns).toEqual(['opponent_team', 'games', 'weeks', 'results', 'opp_team_rating', 'opp_defense_rating', 'points_for', 'points_allowed', 'point_margin'])
    expect(ledger.column_metadata.points_for.kind).toBe('float')
    expect(ledger.column_metadata.points_for.description).toContain('Averaged over the games against this opponent.')
    expect(ledger.column_metadata.opponent_team.sticky).toBe(true)
  })
})
