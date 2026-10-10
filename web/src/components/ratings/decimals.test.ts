import { describe, expect, it } from 'vitest'

import type { MetricColumnMeta, MetricTable } from '@/api/types'

import { columnDecimals, withDecimals } from './decimals'

describe('columnDecimals', () => {
  it.each([
    ['whole numbers', [3, 10, 0], 'count', false, 0],
    ['a points-per-game score', [6.5, -2.25], 'score', false, 2],
    ['values below one', [0.21, -0.05], 'rate', false, 3],
    ['values below one hundred', [27.25, 3.9], 'count', false, 2],
    ['values from one hundred up', [250.5, 1001.75], 'count', false, 1],
    ['proportions that are whole percentages', [0.29, 0.5], 'rate', true, 0],
    ['proportions with tenths of a percent', [0.681, 0.5], 'rate', true, 1],
    ['a column without numbers', [null, 'x'], 'count', false, null],
  ] as const)('gives %s their decimals', (_name, values, shape, percent, expected) => {
    // Act
    const decimals = columnDecimals(values, shape, percent)

    // Assert
    expect(decimals).toBe(expected)
  })
})

describe('withDecimals', () => {
  it('fills each shown column’s decimals from its values and leaves the rest', () => {
    // Arrange
    const meta = (key: string, overrides: Partial<MetricColumnMeta>): MetricColumnMeta => ({
      key,
      label: key,
      description: '',
      group: 'g',
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
      shape: 'count',
      percent: false,
      ...overrides,
    })
    const table: MetricTable = {
      rows: [{ team_rating: 6.5, completion_pct: 0.681, games_played: 17 }],
      visible_columns: ['team_rating', 'completion_pct', 'games_played'],
      column_groups: {},
      column_metadata: {
        team_rating: meta('team_rating', { shape: 'score' }),
        completion_pct: meta('completion_pct', { shape: 'rate', percent: true, kind: 'prob' }),
        games_played: meta('games_played', { kind: 'int' }),
      },
    }

    // Act
    const filled = withDecimals(table, ['team_rating', 'completion_pct'])

    // Assert
    expect(filled.column_metadata.team_rating.decimals).toBe(2)
    expect(filled.column_metadata.completion_pct.decimals).toBe(1)
    expect(filled.column_metadata.games_played.decimals).toBeNull()
    expect(table.column_metadata.team_rating.decimals).toBeNull()
  })
})
