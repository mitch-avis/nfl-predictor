import { screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { renderWithProviders } from '@/test/render'
import { teamGameLogs } from '@/test/ratingsFixtures'

import { GameLogTable } from './GameLogTable'

describe('GameLogTable', () => {
  it('names each opponent with home or away and links to its page for the season', () => {
    // Arrange
    const { table } = teamGameLogs()

    // Act
    renderWithProviders(<GameLogTable kind="teams" season={2025} logs={table} columns={['week', 'opponent_team', 'points_for', 'win_value']} />)

    // Assert
    const kc = screen.getByRole('link', { name: '@ KC' })
    expect(kc).toHaveAttribute('href', '/teams/KC?season=2025')
    expect(screen.getByRole('link', { name: 'vs WSH' })).toHaveAttribute('href', '/teams/WSH?season=2025')
  })

  it('writes each result as a win, loss or tie', () => {
    // Arrange
    const { table } = teamGameLogs()

    // Act
    renderWithProviders(<GameLogTable kind="teams" season={2025} logs={table} columns={['week', 'opponent_team', 'win_value']} />)

    // Assert
    const rows = screen.getAllByRole('row').slice(1)
    expect(rows.map((row) => within(row).getAllByRole('cell').at(-1)?.textContent)).toEqual(['W', 'L'])
  })
})
