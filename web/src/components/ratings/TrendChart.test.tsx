import { screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { renderWithProviders } from '@/test/render'
import { teamGameLogs, teamRatingHistory } from '@/test/ratingsFixtures'

import { TrendChart } from './TrendChart'

describe('TrendChart', () => {
  it('charts the first preferred column with values and lists its points for screen readers', () => {
    // Arrange
    const { table } = teamGameLogs()

    // Act
    renderWithProviders(<TrendChart rows={table.rows} columns={['points_for', 'passing_yards']} metadata={table.column_metadata} preferred={['passing_yards']} />)

    // Assert
    expect(screen.getByText('Pass Yds by week')).toBeInTheDocument()
    const points = screen.getByRole('table', { name: 'Pass Yds, week by week' })
    expect(within(points).getAllByRole('row').map((row) => row.textContent)).toEqual(['WeekOpponentPass Yds', '1KC280', '2WSH210'])
  })

  it('draws the reference it is given and says what it is', () => {
    // Arrange
    const { table } = teamRatingHistory()

    // Act
    renderWithProviders(<TrendChart rows={table.rows} columns={['team_rating']} metadata={table.column_metadata} reference={{ value: 0, caption: 'Dashed line: an average team (0).' }} />)

    // Assert
    expect(screen.getByText('Dashed line: an average team (0).')).toBeInTheDocument()
  })

  it('draws nothing with fewer than two points', () => {
    // Arrange
    const { table } = teamRatingHistory()

    // Act
    const view = renderWithProviders(<TrendChart rows={table.rows.slice(0, 1)} columns={['team_rating']} metadata={table.column_metadata} />)

    // Assert
    expect(view.container).toBeEmptyDOMElement()
  })
})
