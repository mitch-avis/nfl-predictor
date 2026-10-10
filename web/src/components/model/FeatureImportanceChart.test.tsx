import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { FeatureImportance } from '@/api/types'

import { FeatureImportanceChart } from './FeatureImportanceChart'

const rows = [
  { feature: 'away_elo_pre', value: 35, margin_value: 30, total_value: 5, splits: 4 },
  { feature: 'home_rest', value: 25, margin_value: 20, total_value: 5, splits: 3 },
]

describe('FeatureImportanceChart', () => {
  it('says the bars show mean absolute SHAP in points for current runs', () => {
    // Arrange
    const importance: FeatureImportance = { measure: 'mean_abs_shap', rows }

    // Act
    render(<FeatureImportanceChart importance={importance} />)

    // Assert
    expect(screen.getByText(/^Mean \|SHAP\|:/)).toBeInTheDocument()
    expect(screen.getByText(/in points/)).toBeInTheDocument()
    expect(screen.getByText(/adjustment to the market line/)).toBeInTheDocument()
    expect(screen.queryByText(/older run/i)).not.toBeInTheDocument()
  })

  it('labels runs without SHAP as total gain', () => {
    // Arrange
    const importance: FeatureImportance = { measure: 'total_gain', rows }

    // Act
    render(<FeatureImportanceChart importance={importance} />)

    // Assert
    expect(screen.getByText(/^Total gain:/)).toBeInTheDocument()
    expect(screen.getByText(/did not record SHAP/i)).toBeInTheDocument()
  })

  it('labels older runs as average gain summed over columns', () => {
    // Arrange
    const importance: FeatureImportance = { measure: 'summed_average_gain', rows }

    // Act
    render(<FeatureImportanceChart importance={importance} />)

    // Assert
    expect(screen.getByText(/average gain per split, summed over/i)).toBeInTheDocument()
    expect(screen.getByText(/older run/i)).toBeInTheDocument()
  })

  it('reports runs without importance', () => {
    // Act
    render(<FeatureImportanceChart importance={{ measure: null, rows: [] }} />)

    // Assert
    expect(screen.getByText('No feature importance recorded.')).toBeInTheDocument()
  })
})
