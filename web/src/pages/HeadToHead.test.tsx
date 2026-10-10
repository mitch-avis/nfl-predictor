import { screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'
import { teamPairs, teamRankRanges, teamTable } from '@/test/ratingsFixtures'

import { HeadToHeadPage } from './HeadToHead'

const SEASONS = { seasons: [2025], current_season: 2026 }

function stubTeams() {
  return stubApi({
    '/api/ratings/seasons': SEASONS,
    '/api/ratings/2025/teams': teamTable(),
    '/api/ratings/2025/teams/rank-ranges': teamRankRanges(),
    '/api/ratings/2025/teams/BUF/head-to-head': teamPairs('BUF'),
    '/api/ratings/2025/teams/KC/head-to-head': teamPairs('KC'),
  })
}

describe('HeadToHeadPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('says how often the first team was rated above the second', async () => {
    // Arrange
    stubTeams()

    // Act
    renderWithProviders(<HeadToHeadPage />, { route: '/head-to-head?kind=teams&season=2025&a=BUF&b=WSH' })

    // Assert
    expect(await screen.findByText(/Buffalo Bills rated above Washington Commanders in 81% of redraws/)).toBeInTheDocument()
    expect(screen.getByText('81%')).toBeInTheDocument()
  })

  it('compares the two side by side, each linked to its page', async () => {
    // Arrange
    stubTeams()

    // Act
    renderWithProviders(<HeadToHeadPage />, { route: '/head-to-head?kind=teams&season=2025&a=BUF&b=WSH' })

    // Assert
    const comparison = await screen.findByRole('table', { name: 'Side by side' })
    expect(within(comparison).getByRole('link', { name: 'Buffalo Bills' })).toHaveAttribute('href', '/teams/BUF?season=2025')
    expect(within(comparison).getByRole('rowheader', { name: /Team Rating/ })).toBeInTheDocument()
  })

  it('starts on the top-rated team and its neighbor', async () => {
    // Arrange
    stubTeams()

    // Act
    renderWithProviders(<HeadToHeadPage />, { route: '/head-to-head?kind=teams&season=2025' })

    // Assert
    expect(await screen.findByText(/Buffalo Bills rated above Kansas City Chiefs in 81% of redraws/)).toBeInTheDocument()
  })

  it('compares the second with the first’s neighbor when only one is given', async () => {
    // Arrange
    stubTeams()

    // Act
    renderWithProviders(<HeadToHeadPage />, { route: '/head-to-head?kind=teams&season=2025&a=KC' })

    // Assert
    expect(await screen.findByText(/Kansas City Chiefs rated above Buffalo Bills in 30% of redraws/)).toBeInTheDocument()
  })
})
