import { screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ScheduleRanks } from '@/api/types'
import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'
import { metricMeta } from '@/test/ratingsFixtures'

import { SchedulesPage } from './Schedules'

function ranks(rows: Array<[number, string, number]>): ScheduleRanks {
  const count = rows.length
  return {
    count,
    seasons: [...new Set(rows.map(([season]) => season))].sort(),
    skipped_seasons: [2026],
    table: {
      rows: rows.map(([season, team, sos], index) => ({ season, team, sos, softest_rank: index + 1, hardest_rank: count - index })),
      visible_columns: ['season', 'team', 'sos', 'softest_rank', 'hardest_rank'],
      column_groups: { identity: ['season', 'team'], schedule: ['sos'], ranks: ['softest_rank', 'hardest_rank'] },
      column_metadata: {
        season: metricMeta('season', { label: 'Season', kind: 'text', shape: 'id', polarity: 'neutral', heatmap: false, group: 'identity' }),
        team: metricMeta('team', { label: 'Team', kind: 'text', shape: 'id', polarity: 'neutral', heatmap: false, group: 'identity', sticky: true }),
        sos: metricMeta('sos', { label: 'SoS', shape: 'score', contextual: true, group: 'schedule' }),
        softest_rank: metricMeta('softest_rank', { label: 'Softest Rank', kind: 'int', polarity: 'neutral', heatmap: false, group: 'ranks' }),
        hardest_rank: metricMeta('hardest_rank', { label: 'Hardest Rank', kind: 'int', polarity: 'neutral', heatmap: false, group: 'ranks' }),
      },
    },
  }
}

const LEADERBOARD = ranks([
  [2024, 'BUF', -2.25],
  [2023, 'KC', -0.5],
  [2024, 'WSH', 0.25],
  [2023, 'BUF', 1.5],
])

describe('SchedulesPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('leads with the softest and the hardest schedules, each linked to the team’s season', async () => {
    // Arrange
    stubApi({ '/api/ratings/schedules': LEADERBOARD })

    // Act
    renderWithProviders(<SchedulesPage />, { route: '/schedules' })

    // Assert
    const softest = await screen.findByRole('region', { name: 'Softest schedules' })
    expect(within(softest).getAllByRole('link').map((link) => link.getAttribute('href'))[0]).toBe('/teams/BUF?season=2024')
    const hardest = screen.getByRole('region', { name: 'Hardest schedules' })
    expect(within(hardest).getAllByRole('listitem')[0]).toHaveTextContent('2023 BUF')
  })

  it('says which seasons the ranking covers and which it leaves out', async () => {
    // Arrange
    stubApi({ '/api/ratings/schedules': LEADERBOARD })

    // Act
    renderWithProviders(<SchedulesPage />, { route: '/schedules' })

    // Assert
    expect(await screen.findByText(/4 team-seasons, 2023–2024/)).toBeInTheDocument()
    expect(screen.getByText(/2026 is left out until it is complete/)).toBeInTheDocument()
  })

  it('narrows the full table to one team', async () => {
    // Arrange
    stubApi({ '/api/ratings/schedules': LEADERBOARD })

    // Act
    renderWithProviders(<SchedulesPage />, { route: '/schedules?team=BUF' })

    // Assert
    const table = await screen.findByRole('region', { name: 'Every schedule' })
    expect(within(table).getAllByRole('row').slice(1)).toHaveLength(2)
  })

  it('explains an empty ranking', async () => {
    // Arrange
    stubApi({ '/api/ratings/schedules': ranks([]) })

    // Act
    renderWithProviders(<SchedulesPage />, { route: '/schedules' })

    // Assert
    expect(await screen.findByText('No completed seasons to rank yet')).toBeInTheDocument()
  })
})
