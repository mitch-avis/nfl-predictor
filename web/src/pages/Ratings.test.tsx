import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'
import { qbTable, teamTable } from '@/test/ratingsFixtures'

import { RatingsIndexPage } from './Ratings'

const SEASONS = { seasons: [2025, 2024], current_season: 2026 }

function bodyRows() {
  return screen.getAllByRole('row').slice(1)
}

describe('RatingsIndexPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('lists the season’s teams by rating, each linking to its page', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable() })

    // Act
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams' })

    // Assert
    const buffalo = await screen.findByRole('link', { name: 'BUF' })
    expect(buffalo).toHaveAttribute('href', '/teams/BUF?season=2025')
    expect(bodyRows().map((row) => within(row).getAllByRole('cell')[0].textContent)).toEqual(['BUF', 'KC', 'WSH'])
    expect(screen.getByRole('button', { name: /Sort by Team Rating/ })).toBeInTheDocument()
  })

  it('switches to a view of per-game stats and carries it to the team pages', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable() })
    const user = userEvent.setup()
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams' })
    await screen.findByRole('link', { name: 'BUF' })

    // Act
    await user.click(screen.getByRole('button', { name: 'Per game' }))

    // Assert
    expect(screen.getByRole('button', { name: /Sort by Pass Yds/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Sort by Team Rating/ })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'BUF' })).toHaveAttribute('href', '/teams/BUF?season=2025&view=per_game')
  })

  it('finds a team by name or code', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable() })
    const user = userEvent.setup()
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams' })
    await screen.findByRole('link', { name: 'BUF' })

    // Act
    await user.type(screen.getByRole('searchbox', { name: 'Find a team' }), 'washington')

    // Assert
    expect(bodyRows()).toHaveLength(1)
    expect(screen.getByRole('link', { name: 'WSH' })).toBeInTheDocument()
  })

  it('links the model’s power rankings from the team page', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable() })

    // Act
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams' })

    // Assert
    expect(await screen.findByRole('link', { name: 'Power Rankings' })).toHaveAttribute('href', '/power')
  })

  it('says how far a season in progress has gone', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable({ in_progress: true, games_played: 5 }) })

    // Act
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams' })

    // Assert
    expect(await screen.findByText(/Season in progress: up to 5 games per team/)).toBeInTheDocument()
  })

  it('leaves quarterbacks below the qualifier out of the table', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/qbs': qbTable() })

    // Act
    renderWithProviders(<RatingsIndexPage kind="qbs" />, { route: '/qbs' })

    // Assert
    expect(await screen.findByRole('link', { name: 'J.Allen' })).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'M.Backup' })).not.toBeInTheDocument()
  })

  it('lists quarterbacks below the qualifier when asked', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/qbs': qbTable() })
    const user = userEvent.setup()
    renderWithProviders(<RatingsIndexPage kind="qbs" />, { route: '/qbs' })
    await screen.findByRole('link', { name: 'J.Allen' })

    // Act
    await user.click(screen.getByRole('switch', { name: 'Show QBs below the qualifier' }))

    // Assert
    expect(screen.getByRole('link', { name: 'M.Backup' })).toHaveAttribute('href', '/qbs/00-9?season=2025')
  })

  it('compares the teams ticked side by side', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable() })
    const user = userEvent.setup()
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams' })
    await user.click(await screen.findByRole('checkbox', { name: 'Compare BUF' }))

    // Act
    await user.click(screen.getByRole('checkbox', { name: 'Compare WSH' }))

    // Assert
    const comparison = screen.getByRole('table', { name: 'Comparison' })
    const headers = within(comparison).getAllByRole('columnheader').map((header) => header.textContent)
    expect(headers.slice(1)).toEqual(['Buffalo Bills', 'Washington Commanders'])
  })

  it('restores a comparison from the page address', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable() })

    // Act
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams?compare=KC,WSH' })

    // Assert
    const comparison = await screen.findByRole('table', { name: 'Comparison' })
    expect(within(comparison).getAllByRole('columnheader').map((header) => header.textContent).slice(1)).toEqual(['Kansas City Chiefs', 'Washington Commanders'])
    expect(screen.getByRole('checkbox', { name: 'Compare KC' })).toBeChecked()
  })

  it('keeps the keyboard on a compare box after ticking it', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable() })
    const user = userEvent.setup()
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams' })
    ;(await screen.findByRole('checkbox', { name: 'Compare BUF' })).focus()

    // Act
    await user.keyboard(' ')

    // Assert
    const box = screen.getByRole('checkbox', { name: 'Compare BUF' })
    expect(box).toBeChecked()
    expect(box).toHaveFocus()
  })

  it('compares the season totals the table shows in the totals view', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable() })

    // Act
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams?view=totals&compare=BUF,KC' })

    // Assert
    const comparison = await screen.findByRole('table', { name: 'Comparison' })
    const passing = within(comparison).getByRole('rowheader', { name: 'Pass Yds' }).closest('tr') as HTMLElement
    expect(within(passing).getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['4,258.5', '4,590.0'])
  })

  it('explains how to get tables when there are none', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': { seasons: [], current_season: 2026 } })

    // Act
    renderWithProviders(<RatingsIndexPage kind="teams" />, { route: '/teams' })

    // Assert
    expect(await screen.findByText('No team or quarterback tables yet')).toBeInTheDocument()
  })
})
