import { screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'

import { carriedSearch, useRatingsSeason, viewPatch } from './useRatingsParams'

function SeasonProbe() {
  const { season, unavailable } = useRatingsSeason()
  return (
    <p>
      season {String(season)} unavailable {String(unavailable)}
    </p>
  )
}

describe('useRatingsSeason', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('opens on the newest season with tables', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': { seasons: [2025, 2024], current_season: 2026 } })

    // Act
    renderWithProviders(<SeasonProbe />, { route: '/teams' })

    // Assert
    expect(await screen.findByText('season 2025 unavailable null')).toBeInTheDocument()
  })

  it('keeps a season asked for in the URL', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': { seasons: [2025, 2024], current_season: 2026 } })

    // Act
    renderWithProviders(<SeasonProbe />, { route: '/teams?season=2024' })

    // Assert
    expect(await screen.findByText('season 2024 unavailable null')).toBeInTheDocument()
  })

  it('names a season without tables and shows the newest instead', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': { seasons: [2025, 2024], current_season: 2026 } })

    // Act
    renderWithProviders(<SeasonProbe />, { route: '/teams?season=1990' })

    // Assert
    expect(await screen.findByText('season 2025 unavailable 1990')).toBeInTheDocument()
  })
})

describe('viewPatch', () => {
  it('writes the hidden subcategories as a comma list', () => {
    // Act
    const patch = viewPatch({ view: 'per_game', hidden: ['Rushing', 'Passing'] })

    // Assert
    expect(patch).toEqual({ view: 'per_game', hide: 'Rushing,Passing' })
  })
})

describe('carriedSearch', () => {
  it('carries the season and the view filters, and drops the rest', () => {
    // Arrange
    const params = new URLSearchParams('view=per_play&category=Defense&q=buf&compare=KC')

    // Act
    const search = carriedSearch(params, 2024)

    // Assert
    expect(search).toBe('?season=2024&view=per_play&category=Defense')
  })
})
