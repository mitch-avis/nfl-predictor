import { screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'
import { teamPairs, teamTable } from '@/test/ratingsFixtures'

import { HeadToHeadCard } from './HeadToHeadCard'

describe('HeadToHeadCard', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('compares with the team ranked just above by default', async () => {
    // Arrange
    stubApi({ '/api/ratings/2025/teams/KC/head-to-head': teamPairs('KC') })
    const { table, team_names } = teamTable()

    // Act
    renderWithProviders(<HeadToHeadCard kind="teams" season={2025} entityId="KC" rows={table.rows} names={team_names} />)

    // Assert
    expect(await screen.findByText(/rated above Buffalo Bills in 30% of redraws/)).toBeInTheDocument()
  })

  it('links to the head-to-head page for the pair', async () => {
    // Arrange
    stubApi({ '/api/ratings/2025/teams/KC/head-to-head': teamPairs('KC') })
    const { table, team_names } = teamTable()

    // Act
    renderWithProviders(<HeadToHeadCard kind="teams" season={2025} entityId="KC" rows={table.rows} names={team_names} />)

    // Assert
    const link = await screen.findByRole('link', { name: 'Compare side by side' })
    expect(link).toHaveAttribute('href', '/head-to-head?kind=teams&season=2025&a=KC&b=BUF')
  })

  it('stays hidden for a season built without head-to-head chances', async () => {
    // Arrange
    const fetchMock = stubApi({})
    const { table, team_names } = teamTable()

    // Act
    renderWithProviders(<HeadToHeadCard kind="teams" season={2025} entityId="KC" rows={table.rows} names={team_names} />)

    // Assert
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(screen.queryByText(/Head to head/)).not.toBeInTheDocument()
  })
})
