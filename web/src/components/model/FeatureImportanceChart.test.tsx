import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { FeatureImportance } from '@/api/types'

import { FeatureImportanceChart } from './FeatureImportanceChart'

const rows = [
  { feature: 'away_elo_pre', value: 35, margin_value: 30, total_value: 5, splits: 4 },
  { feature: 'home_rest', value: 25, margin_value: 20, total_value: 5, splits: 3 },
]

describe('FeatureImportanceChart', () => {
  it('says the bars show total gain for current runs', () => {
    const importance: FeatureImportance = { measure: 'total_gain', rows }
    render(<FeatureImportanceChart importance={importance} />)
    expect(screen.getByText(/^Total gain:/)).toBeInTheDocument()
    expect(screen.queryByText(/older run/i)).not.toBeInTheDocument()
  })

  it('labels older runs as average gain summed over columns', () => {
    const importance: FeatureImportance = { measure: 'summed_average_gain', rows }
    render(<FeatureImportanceChart importance={importance} />)
    expect(screen.getByText(/average gain per split, summed over/i)).toBeInTheDocument()
    expect(screen.getByText(/older run/i)).toBeInTheDocument()
  })

  it('reports runs without importance', () => {
    render(<FeatureImportanceChart importance={{ measure: null, rows: [] }} />)
    expect(screen.getByText('No feature importance recorded.')).toBeInTheDocument()
  })
})
