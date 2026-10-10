import { screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { RankRangeTable } from '@/api/types'
import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'
import { teamRankRanges, teamTable } from '@/test/ratingsFixtures'

import { RankRangesPage } from './RankRanges'

const SEASONS = { seasons: [2025], current_season: 2026 }

describe('RankRangesPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('charts every team’s rank range for the newest season', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams/rank-ranges': teamRankRanges() })

    // Act
    renderWithProviders(<RankRangesPage />, { route: '/rank-ranges' })

    // Assert
    expect(await screen.findByRole('link', { name: /^Buffalo Bills: published rank 1st/ })).toHaveAttribute('href', '/teams/BUF?season=2025')
  })

  it('names the teams from the rank ranges without fetching the team table', async () => {
    // Arrange
    const fetchMock = stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams/rank-ranges': teamRankRanges() })

    // Act
    renderWithProviders(<RankRangesPage />, { route: '/rank-ranges' })

    // Assert
    await screen.findByRole('link', { name: /^Buffalo Bills:/ })
    const paths = fetchMock.mock.calls.map(([input]) => new URL(String(input), 'http://localhost').pathname)
    expect(paths).not.toContain('/api/ratings/2025/teams')
  })

  it('charts the quarterbacks when asked', async () => {
    // Arrange
    const qbRanges: RankRangeTable = {
      kind: 'qbs',
      season: 2025,
      entity_id: null,
      table: {
        rows: [{ qb_id: '00-1', qb_name: 'J.Allen', team: 'BUF', qb_rank: 1, qb_rank_q025: 1, qb_rank_q250: 1, qb_rank_q500: 1, qb_rank_q750: 2, qb_rank_q975: 2 }],
        visible_columns: [],
        column_groups: {},
        column_metadata: {},
      },
      team_names: { BUF: 'Buffalo Bills' },
    }
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/qbs/rank-ranges': qbRanges })

    // Act
    renderWithProviders(<RankRangesPage />, { route: '/rank-ranges?kind=qbs' })

    // Assert
    expect(await screen.findByRole('link', { name: /^J.Allen: published rank 1st/ })).toHaveAttribute('href', '/qbs/00-1?season=2025')
  })

  it('says when the season was built without rank ranges', async () => {
    // Arrange
    stubApi({ '/api/ratings/seasons': SEASONS, '/api/ratings/2025/teams': teamTable() })

    // Act
    renderWithProviders(<RankRangesPage />, { route: '/rank-ranges' })

    // Assert
    expect(await screen.findByText('No rank ranges for 2025')).toBeInTheDocument()
  })
})
