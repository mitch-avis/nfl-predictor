import { screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { renderWithProviders } from '@/test/render'
import { TEAM_CATEGORIES, teamTable } from '@/test/ratingsFixtures'

import { MetricSections } from './MetricSections'

describe('MetricSections', () => {
  it('groups the values by category and subcategory in the registry’s order', () => {
    // Arrange
    const { table } = teamTable()

    // Act
    renderWithProviders(
      <MetricSections row={table.rows[0]} rows={table.rows} columns={['rushing_yards', 'games_played', 'passing_yards']} metadata={table.column_metadata} categories={TEAM_CATEGORIES} />,
    )

    // Assert
    expect(screen.getAllByRole('heading').map((heading) => heading.textContent)).toEqual(['Overall', 'Offense — Passing', 'Offense — Rushing'])
    const passing = screen.getByRole('region', { name: 'Offense — Passing' })
    expect(within(passing).getByText('250.5')).toBeInTheDocument()
  })

  it('says when the filters leave nothing to show', () => {
    // Arrange
    const { table } = teamTable()

    // Act
    renderWithProviders(<MetricSections row={table.rows[0]} rows={table.rows} columns={[]} metadata={table.column_metadata} categories={TEAM_CATEGORIES} />)

    // Assert
    expect(screen.getByText('No stats match these filters.')).toBeInTheDocument()
  })
})
