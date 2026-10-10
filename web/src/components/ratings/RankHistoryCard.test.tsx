import { screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SeasonTable } from '@/api/types'
import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'

import { RankHistoryCard } from './RankHistoryCard'

function history(rows: SeasonTable['table']['rows']): SeasonTable {
  return { kind: 'teams', season: 2026, entity_id: 'KC', table: { rows, visible_columns: [], column_groups: {}, column_metadata: {} } }
}

describe('RankHistoryCard', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('describes the rank week by week for a season in progress', async () => {
    // Arrange
    stubApi({
      '/api/ratings/2026/teams/KC/rank-history': history([
        { week: 3, team: 'KC', team_rank: 7, team_rank_q025: 2, team_rank_q250: 4, team_rank_q500: 7, team_rank_q750: 11, team_rank_q975: 20 },
        { week: 4, team: 'KC', team_rank: 5, team_rank_q025: 1, team_rank_q250: 3, team_rank_q500: 5, team_rank_q750: 8, team_rank_q975: 14 },
      ]),
    })

    // Act
    renderWithProviders(<RankHistoryCard kind="teams" season={2026} entityId="KC" count={32} />)

    // Assert
    expect(await screen.findByText('Median rank 7th after week 3 (95%: 2nd–20th) and 5th after week 4 (95%: 1st–14th).')).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Rank by week' })).toBeInTheDocument()
  })

  it('explains the wait until every team has played three games', async () => {
    // Arrange
    stubApi({ '/api/ratings/2026/teams/KC/rank-history': history([]) })

    // Act
    renderWithProviders(<RankHistoryCard kind="teams" season={2026} entityId="KC" count={32} />)

    // Assert
    expect(await screen.findByText(/starts once every team has played three games/)).toBeInTheDocument()
  })

  it('stays hidden for a completed season', async () => {
    // Arrange
    const fetchMock = stubApi({})

    // Act
    renderWithProviders(<RankHistoryCard kind="teams" season={2025} entityId="KC" count={32} />)

    // Assert
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(screen.queryByText('Rank by week')).not.toBeInTheDocument()
  })
})
