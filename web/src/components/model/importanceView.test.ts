import { describe, expect, it } from 'vitest'

import type { FeatureImportance } from '@/api/types'

import { importanceCaption, importanceRows } from './importanceView'

const importance: FeatureImportance = {
  measure: 'mean_abs_shap',
  rows: [
    { feature: 'total_line', value: 4, margin_value: 0.5, total_value: 3.5, splits: 9 },
    { feature: 'home_spread', value: 3, margin_value: 2.8, total_value: 0.2, splits: 7 },
  ],
  margin_rows: [
    { feature: 'home_spread', value: 2.8, margin_value: 2.8, total_value: 0.2, splits: 5 },
    { feature: 'away_rest', value: 1.1, margin_value: 1.1, total_value: 0.1, splits: 2 },
    { feature: 'total_line', value: 0.5, margin_value: 0.5, total_value: 3.5, splits: 1 },
  ],
}

describe('importanceRows', () => {
  it('plots the combined ranking for both heads', () => {
    // Act
    const data = importanceRows(importance, 'combined', 25)

    // Assert
    expect(data).toEqual([
      { name: 'Total line', value: 4 },
      { name: 'Home spread', value: 3 },
    ])
  })

  it('plots the margin head ranking for the margin view', () => {
    // Act
    const data = importanceRows(importance, 'margin', 25)

    // Assert
    expect(data.map((row) => row.name)).toEqual(['Home spread', 'Away rest', 'Total line'])
    expect(data[0].value).toBe(2.8)
  })

  it('keeps only the first rows up to the limit', () => {
    // Act
    const data = importanceRows(importance, 'margin', 2)

    // Assert
    expect(data).toHaveLength(2)
  })
})

describe('importanceCaption', () => {
  it('describes total gain over both heads', () => {
    // Act
    const caption = importanceCaption('total_gain', 'combined')

    // Assert
    expect(caption).toMatch(/^Total gain:.*across the margin and total heads/)
  })

  it('describes total gain for the margin head alone', () => {
    // Act
    const caption = importanceCaption('total_gain', 'margin')

    // Assert
    expect(caption).toMatch(/^Total gain:.*margin head alone/)
    expect(caption).not.toMatch(/total heads/)
  })

  it('keeps the many-categories warning for the oldest runs in the margin view', () => {
    // Act
    const caption = importanceCaption('summed_average_gain', 'margin')

    // Assert
    expect(caption).toMatch(/margin head alone/)
    expect(caption).toMatch(/favors features with many categories/)
  })
})
