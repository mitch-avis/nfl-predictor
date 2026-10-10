import { describe, expect, it } from 'vitest'

import type { ColumnMeta } from '@/api/types'

import { columnStats, formatCell, heatStyle } from './cells'

const meta = (overrides: Partial<ColumnMeta>): ColumnMeta => ({
  key: 'k',
  label: 'K',
  description: '',
  group: 'G',
  kind: 'float',
  polarity: 'neutral',
  heatmap: false,
  decimals: null,
  actionable: true,
  sticky: false,
  ...overrides,
})

const rows = [{ a: 1, b: -2 }, { a: 3, b: 4 }, { a: null, b: 'x' }]

describe('cells', () => {
  it.each([
    ['prob', meta({ kind: 'prob', decimals: 1 }), 0.4023, '40.2%'],
    ['pct', meta({ kind: 'pct', decimals: 1 }), 0.05, '+5.0%'],
    ['money', meta({ kind: 'money' }), -155, '-155'],
    ['spread', meta({ kind: 'spread' }), -3, '-3.0'],
    ['int', meta({ kind: 'int' }), 7, '7'],
    ['bool', meta({ kind: 'bool' }), true, 'Yes'],
    ['text', meta({ kind: 'text' }), 'KC', 'KC'],
    ['missing value', undefined, null, '—'],
    ['value without metadata', undefined, 3, '3'],
  ] as const)('formats a %s cell', (_name, columnMeta, value, expected) => {
    // Act
    const text = formatCell(columnMeta, value)

    // Assert
    expect(text).toBe(expected)
  })

  it('computes the numeric range of each column', () => {
    // Act
    const stats = columnStats(rows, ['a', 'b', 'c'])

    // Assert
    expect(stats.a).toEqual({ min: 1, max: 3, absMax: 3 })
    expect(stats.b).toEqual({ min: -2, max: 4, absMax: 4 })
    expect(stats.c).toBeUndefined()
  })

  it.each([
    ['the best value of a higher-is-better column', meta({ heatmap: true, polarity: 'higher' }), 3, 'a', '0.500'],
    ['the worst value of a higher-is-better column', meta({ heatmap: true, polarity: 'higher' }), 1, 'a', '0.060'],
    ['the best value of a lower-is-better column', meta({ heatmap: true, polarity: 'lower' }), 1, 'a', '0.500'],
    ['a negative value of a neutral column', meta({ heatmap: true, polarity: 'neutral' }), -2, 'b', '55'],
    ['a positive value of a neutral column', meta({ heatmap: true, polarity: 'neutral' }), 4, 'b', '240'],
  ] as const)('shades %s', (_name, columnMeta, value, column, expected) => {
    // Arrange
    const stats = columnStats(rows, ['a', 'b'])

    // Act
    const style = heatStyle(columnMeta, value, stats[column])

    // Assert
    expect(style?.backgroundColor).toContain(expected)
  })

  it.each([
    ['the tougher end of a context column', meta({ heatmap: true, polarity: 'higher', contextual: true }), 3, '0.400'],
    ['the easier end of a context column', meta({ heatmap: true, polarity: 'higher', contextual: true }), 1, '0.040'],
    ['the tougher end of a lower-is-tougher context column', meta({ heatmap: true, polarity: 'lower', contextual: true }), 1, '0.400'],
  ] as const)('shades %s in the muted context hue, not the grade colors', (_name, columnMeta, value, alpha) => {
    // Arrange
    const stats = columnStats(rows, ['a'])

    // Act
    const style = heatStyle(columnMeta, value, stats.a)

    // Assert
    expect(style?.backgroundColor).toBe(`oklch(0.6 0.04 262 / ${alpha})`)
  })

  it.each([
    ['a neutral column without a spread', meta({ heatmap: true, polarity: 'neutral' }), 4, { min: 0, max: 0, absMax: 0 }],
    ['a graded column without a spread', meta({ heatmap: true, polarity: 'higher' }), 4, { min: 4, max: 4, absMax: 4 }],
    ['a text value', meta({ heatmap: true, polarity: 'higher' }), 'x', { min: 1, max: 3, absMax: 3 }],
    ['a column without a heatmap', meta({}), 1, { min: 1, max: 3, absMax: 3 }],
  ] as const)('leaves %s unshaded', (_name, columnMeta, value, stats) => {
    // Act
    const style = heatStyle(columnMeta, value, stats)

    // Assert
    expect(style).toBeUndefined()
  })
})
