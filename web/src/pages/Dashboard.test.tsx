import { screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'

import { DashboardPage } from './Dashboard'

describe('DashboardPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('links the other pages in the sidebar’s sections', async () => {
    // Arrange
    stubApi({ '/api/runs': { active_run_id: null, pinned_run_id: null, runs: [] } })

    // Act
    renderWithProviders(<DashboardPage />)

    // Assert
    const players = await screen.findByRole('region', { name: 'Teams and players' })
    expect(within(players).getByRole('link', { name: /Teams/ })).toHaveAttribute('href', '/teams')
    expect(within(screen.getByRole('region', { name: 'This week' })).queryByRole('link', { name: /Overview/ })).not.toBeInTheDocument()
  })
})
