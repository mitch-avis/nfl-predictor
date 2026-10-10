import { screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'

import { PowerRankingsPage } from './PowerRankings'

describe('PowerRankingsPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('points to the play-by-play team ratings on the Teams page', () => {
    // Arrange
    stubApi({})

    // Act
    renderWithProviders(<PowerRankingsPage />, { route: '/power' })

    // Assert
    expect(screen.getByRole('link', { name: 'Teams' })).toHaveAttribute('href', '/teams')
  })
})
