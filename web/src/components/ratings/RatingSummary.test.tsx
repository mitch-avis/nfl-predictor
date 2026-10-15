import { screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { renderWithProviders } from '@/test/render'
import { teamTable } from '@/test/ratingsFixtures'

import { RatingSummary } from './RatingSummary'

describe('RatingSummary', () => {
  it('leads with the headline rating and its rank among the season', () => {
    // Arrange
    const { table } = teamTable()

    // Act
    renderWithProviders(<RatingSummary kind="teams" row={table.rows[0]} rows={table.rows} metadata={table.column_metadata} />)

    // Assert
    const headline = screen.getByRole('group', { name: 'Team Rating' })
    expect(within(headline).getByText('6.50')).toBeInTheDocument()
    expect(within(headline).getByText('1st of 3')).toBeInTheDocument()
  })

  it('shows the headline before the schedule adjustment', () => {
    // Arrange
    const { table } = teamTable()

    // Act
    renderWithProviders(<RatingSummary kind="teams" row={table.rows[2]} rows={table.rows} metadata={table.column_metadata} />)

    // Assert
    expect(screen.getByText(/Before the schedule adjustment/)).toHaveTextContent('Before the schedule adjustment: EPA Margin/Play -0.110 (3rd of 3)')
  })

  it('does not rank schedule strength as better or worse', () => {
    // Arrange
    const { table } = teamTable()

    // Act
    renderWithProviders(<RatingSummary kind="teams" row={table.rows[0]} rows={table.rows} metadata={table.column_metadata} />)

    // Assert
    const sos = screen.getByRole('group', { name: 'SoS' })
    expect(within(sos).queryByText(/of 3/)).not.toBeInTheDocument()
  })
})
