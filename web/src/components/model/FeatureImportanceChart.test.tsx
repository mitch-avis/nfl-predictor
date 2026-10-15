import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import type { FeatureImportance } from '@/api/types'

import { FeatureImportanceChart } from './FeatureImportanceChart'

const rows = [
  { feature: 'away_elo_pre', value: 35, margin_value: 30, total_value: 5, splits: 4 },
  { feature: 'home_rest', value: 25, margin_value: 20, total_value: 5, splits: 3 },
]

const marginRows = [
  { feature: 'away_elo_pre', value: 30, margin_value: 30, total_value: 5, splits: 3 },
  { feature: 'home_spread', value: 22, margin_value: 22, total_value: 1, splits: 2 },
]

describe('FeatureImportanceChart', () => {
  it('says the bars show mean absolute SHAP in points for current runs', () => {
    // Arrange
    const importance: FeatureImportance = { measure: 'mean_abs_shap', rows, margin_rows: marginRows }

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
    const importance: FeatureImportance = { measure: 'total_gain', rows, margin_rows: marginRows }

    // Act
    render(<FeatureImportanceChart importance={importance} />)

    // Assert
    expect(screen.getByText(/^Total gain:/)).toBeInTheDocument()
    expect(screen.getByText(/did not record SHAP/i)).toBeInTheDocument()
  })

  it('labels older runs as average gain summed over columns', () => {
    // Arrange
    const importance: FeatureImportance = { measure: 'summed_average_gain', rows, margin_rows: marginRows }

    // Act
    render(<FeatureImportanceChart importance={importance} />)

    // Assert
    expect(screen.getByText(/average gain per split, summed over/i)).toBeInTheDocument()
    expect(screen.getByText(/older run/i)).toBeInTheDocument()
  })

  it('reports runs without importance', () => {
    // Act
    render(<FeatureImportanceChart importance={{ measure: null, rows: [], margin_rows: [] }} />)

    // Assert
    expect(screen.getByText('No feature importance recorded.')).toBeInTheDocument()
  })

  it('shows both heads by default with the margin-only switch off', () => {
    // Arrange
    const importance: FeatureImportance = { measure: 'mean_abs_shap', rows, margin_rows: marginRows }

    // Act
    render(<FeatureImportanceChart importance={importance} />)

    // Assert
    expect(screen.getByRole('switch', { name: 'Margin head only' })).not.toBeChecked()
    expect(screen.getByText(/added across the margin and total heads/)).toBeInTheDocument()
  })

  it('switches the caption to the margin head alone', async () => {
    // Arrange
    const user = userEvent.setup()
    const importance: FeatureImportance = { measure: 'mean_abs_shap', rows, margin_rows: marginRows }
    render(<FeatureImportanceChart importance={importance} />)

    // Act
    await user.click(screen.getByRole('switch', { name: 'Margin head only' }))

    // Assert
    expect(screen.getByRole('switch', { name: 'Margin head only' })).toBeChecked()
    expect(screen.getByText(/margin head alone/)).toBeInTheDocument()
    expect(screen.queryByText(/added across the margin and total heads/)).not.toBeInTheDocument()
  })

  it('offers no switch when the run recorded no per-head ranking', () => {
    // Arrange
    const importance: FeatureImportance = { measure: 'total_gain', rows, margin_rows: [] }

    // Act
    render(<FeatureImportanceChart importance={importance} />)

    // Assert
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
  })
})
