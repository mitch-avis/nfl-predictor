import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'
import { teamPairs, teamTable } from '@/test/ratingsFixtures'

import { ComparisonPanel } from './ComparisonPanel'

describe('ComparisonPanel', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('puts the picked teams side by side on the view’s columns', () => {
    // Arrange
    stubApi({})
    const data = teamTable()

    // Act
    renderWithProviders(<ComparisonPanel kind="teams" season={2025} table={data.table} teamNames={data.team_names} ids={['WSH', 'BUF']} columns={['team_rating', 'sos']} onRemove={() => {}} />)

    // Assert
    const table = screen.getByRole('table', { name: 'Comparison' })
    const headers = within(table).getAllByRole('columnheader').map((header) => header.textContent)
    expect(headers[1]).toContain('Washington Commanders')
    expect(headers[2]).toContain('Buffalo Bills')
    expect(within(table).getAllByRole('rowheader').map((header) => header.textContent)).toEqual(['Team Rating', 'SoS'])
  })

  it('takes a team out of the comparison', async () => {
    // Arrange
    stubApi({})
    const onRemove = vi.fn()
    const user = userEvent.setup()
    const data = teamTable()
    renderWithProviders(<ComparisonPanel kind="teams" season={2025} table={data.table} teamNames={data.team_names} ids={['WSH', 'BUF']} columns={['team_rating']} onRemove={onRemove} />)

    // Act
    await user.click(screen.getByRole('button', { name: 'Remove Buffalo Bills from the comparison' }))

    // Assert
    expect(onRemove).toHaveBeenCalledWith('BUF')
  })

  it('adds the head-to-head chance when two are compared', async () => {
    // Arrange
    stubApi({ '/api/ratings/2025/teams/BUF/head-to-head': teamPairs('BUF') })

    const data = teamTable()

    // Act
    renderWithProviders(<ComparisonPanel kind="teams" season={2025} table={data.table} teamNames={data.team_names} ids={['BUF', 'WSH']} columns={['team_rating']} onRemove={() => {}} />)

    // Assert
    expect(await screen.findByText(/Buffalo Bills rated above Washington Commanders in 81% of redraws/)).toBeInTheDocument()
  })
})
