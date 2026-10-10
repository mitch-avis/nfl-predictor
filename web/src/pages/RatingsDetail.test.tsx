import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'
import { qbTable, teamGameLogs, teamPairs, teamRankRanges, teamRatingHistory, teamTable } from '@/test/ratingsFixtures'

import { RatingsDetailPage } from './RatingsDetail'

const SEASONS = { seasons: [2025, 2024], current_season: 2026 }

function stubTeamSeason() {
  return stubApi({
    '/api/ratings/seasons': SEASONS,
    '/api/ratings/2025/teams': teamTable(),
    '/api/ratings/2025/teams/rank-ranges': teamRankRanges(),
    '/api/ratings/2025/teams/BUF/game-logs': teamGameLogs('BUF'),
    '/api/ratings/2025/teams/BUF/rating-history': teamRatingHistory('BUF'),
    '/api/ratings/2025/teams/BUF/head-to-head': teamPairs('BUF'),
  })
}

describe('RatingsDetailPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('titles a team by its franchise name and leads with its ratings', async () => {
    // Arrange
    stubTeamSeason()

    // Act
    renderWithProviders(<RatingsDetailPage kind="teams" entityId="BUF" />, { route: '/teams/BUF?season=2025' })

    // Assert
    expect(await screen.findByRole('heading', { name: 'Buffalo Bills' })).toBeInTheDocument()
    expect(screen.getByText('Team · 2025 regular season')).toBeInTheDocument()
    expect(within(screen.getByRole('group', { name: 'Team Rating' })).getByText('6.50')).toBeInTheDocument()
  })

  it('lists the games, each opponent linked to its page', async () => {
    // Arrange
    stubTeamSeason()

    // Act
    renderWithProviders(<RatingsDetailPage kind="teams" entityId="BUF" />, { route: '/teams/BUF?season=2025' })

    // Assert
    const games = await screen.findByRole('region', { name: 'Game by game' })
    expect(within(games).getByRole('link', { name: '@ KC' })).toHaveAttribute('href', '/teams/KC?season=2025')
  })

  it('gives one row per opponent faced, with its season rating and the results', async () => {
    // Arrange
    stubTeamSeason()

    // Act
    renderWithProviders(<RatingsDetailPage kind="teams" entityId="BUF" />, { route: '/teams/BUF?season=2025' })

    // Assert
    const opponents = await screen.findByRole('region', { name: 'Opponents faced' })
    const rows = within(opponents).getAllByRole('row').slice(1)
    expect(rows.map((row) => within(row).getAllByRole('cell')[0].textContent)).toEqual(['KC', 'WSH'])
  })

  it('shows the season stats of the view chosen, by category', async () => {
    // Arrange
    stubTeamSeason()
    const user = userEvent.setup()
    renderWithProviders(<RatingsDetailPage kind="teams" entityId="BUF" />, { route: '/teams/BUF?season=2025' })
    const stats = await screen.findByRole('region', { name: 'Season stats' })

    // Act
    await user.click(within(stats).getByRole('button', { name: 'Season totals' }))

    // Assert
    const passing = within(stats).getByRole('region', { name: 'Offense — Passing' })
    expect(within(passing).getByText('4,258.5')).toBeInTheDocument()
  })

  it('says when a team has no rows that season', async () => {
    // Arrange
    stubTeamSeason()

    // Act
    renderWithProviders(<RatingsDetailPage kind="teams" entityId="LV" />, { route: '/teams/LV?season=2025' })

    // Assert
    expect(await screen.findByText('No team LV in 2025')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'All teams' })).toHaveAttribute('href', '/teams?season=2025')
  })

  it('titles a quarterback by his name and his team', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/qbs': qbTable() })

    // Act
    renderWithProviders(<RatingsDetailPage kind="qbs" entityId="00-1" />, { route: '/qbs/00-1?season=2025' })

    // Assert
    expect(await screen.findByRole('heading', { name: 'J.Allen' })).toBeInTheDocument()
    expect(screen.getByText('Quarterback · Buffalo Bills · 2025 regular season')).toBeInTheDocument()
  })
})
