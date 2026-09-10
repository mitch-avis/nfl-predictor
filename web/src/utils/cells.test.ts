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

describe('cells', () => {
  it('formats by kind', () => {
    expect(formatCell(meta({ kind: 'prob', decimals: 1 }), 0.4023)).toBe('40.2%')
    expect(formatCell(meta({ kind: 'pct', decimals: 1 }), 0.05)).toBe('+5.0%')
    expect(formatCell(meta({ kind: 'money' }), -155)).toBe('-155')
    expect(formatCell(meta({ kind: 'spread' }), -3)).toBe('-3.0')
    expect(formatCell(meta({ kind: 'int' }), 7)).toBe('7')
    expect(formatCell(meta({ kind: 'bool' }), true)).toBe('Yes')
    expect(formatCell(meta({ kind: 'text' }), 'KC')).toBe('KC')
    expect(formatCell(undefined, null)).toBe('—')
    expect(formatCell(undefined, 3)).toBe('3')
  })

  it('computes column stats and heat styles', () => {
    const rows = [{ a: 1, b: -2 }, { a: 3, b: 4 }, { a: null, b: 'x' }]
    const stats = columnStats(rows, ['a', 'b', 'c'])
    expect(stats.a).toEqual({ min: 1, max: 3, absMax: 3 })
    expect(stats.b).toEqual({ min: -2, max: 4, absMax: 4 })
    expect(stats.c).toBeUndefined()
    const higher = meta({ heatmap: true, polarity: 'higher' })
    expect(heatStyle(higher, 3, stats.a)?.backgroundColor).toContain('0.500')
    expect(heatStyle(higher, 1, stats.a)?.backgroundColor).toContain('0.060')
    const lower = meta({ heatmap: true, polarity: 'lower' })
    expect(heatStyle(lower, 1, stats.a)?.backgroundColor).toContain('0.500')
    const neutral = meta({ heatmap: true, polarity: 'neutral' })
    expect(heatStyle(neutral, -2, stats.b)?.backgroundColor).toContain('55')
    expect(heatStyle(neutral, 4, stats.b)?.backgroundColor).toContain('240')
    expect(heatStyle(neutral, 4, { min: 0, max: 0, absMax: 0 })).toBeUndefined()
    expect(heatStyle(higher, 4, { min: 4, max: 4, absMax: 4 })).toBeUndefined()
    expect(heatStyle(higher, 'x', stats.a)).toBeUndefined()
    expect(heatStyle(meta({}), 1, stats.a)).toBeUndefined()
  })
})
