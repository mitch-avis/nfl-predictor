import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import type { TiebreakGame, Tiebreakers } from '@/api/types'
import { renderWithProviders } from '@/test/render'

import { TiebreakerPanel } from './TiebreakerPanel'

function game(overrides: Partial<TiebreakGame>): TiebreakGame {
  return {
    game_id: 'g',
    away_abbr: 'A',
    home_abbr: 'H',
    kickoff: '2026-10-11T13:00:00-04:00',
    slot: null,
    total_points: 44,
    home_points: 23,
    away_points: 21,
    total_raw: 44.2,
    home_raw: 23.1,
    away_raw: 21.1,
    total_p10: 31.0,
    total_p90: 58.0,
    home_p10: 13.0,
    home_p90: 33.0,
    away_p10: 11.0,
    away_p90: 31.0,
    ...overrides,
  }
}

const card: Tiebreakers = {
  run_id: 'weekly_w5',
  source: 'active',
  season: 2026,
  week: 5,
  generated_at: null,
  games: [
    game({ game_id: 'thu', away_abbr: 'TB', home_abbr: 'DAL', slot: 'Thursday night', total_points: 48, home_points: 29, away_points: 19 }),
    game({ game_id: 'snf', away_abbr: 'BAL', home_abbr: 'ATL', slot: 'Sunday night', total_points: 45, home_points: 24, away_points: 21 }),
    game({ game_id: 'mnf', away_abbr: 'BUF', home_abbr: 'LAR', slot: 'Monday night', total_points: 55, home_points: 28, away_points: 27 }),
  ],
  default_game_ids: ['snf', 'mnf'],
  highest: { points: 39, median: 38.6, p10: 33.2, p90: 45.9, projected_team: 'DAL', projected_points: 29.4 },
  lowest: { points: 8, median: 8.2, p10: 3.1, p90: 12.4, projected_team: 'TB', projected_points: 18.6 },
}

describe('TiebreakerPanel', () => {
  it('starts with the suggested Sunday and Monday night games', () => {
    // Act
    renderWithProviders(<TiebreakerPanel card={card} />)

    // Assert
    const first = screen.getByRole('region', { name: 'First tiebreak game' })
    expect(within(first).getByText('BAL @ ATL')).toBeInTheDocument()
    expect(within(first).getByLabelText('Total points')).toHaveTextContent('45')
    const second = screen.getByRole('region', { name: 'Second tiebreak game' })
    expect(within(second).getByText('BUF @ LAR')).toBeInTheDocument()
    expect(within(second).getByLabelText('LAR points')).toHaveTextContent('28')
    expect(within(second).getByLabelText('BUF points')).toHaveTextContent('27')
  })

  it('switches a tiebreak game to the one the pool names', async () => {
    // Arrange
    const user = userEvent.setup()
    renderWithProviders(<TiebreakerPanel card={card} />)

    // Act
    await user.selectOptions(screen.getByLabelText('Choose the first tiebreak game'), 'thu')

    // Assert
    const first = screen.getByRole('region', { name: 'First tiebreak game' })
    expect(within(first).getByText('TB @ DAL')).toBeInTheDocument()
    expect(within(first).getByLabelText('Total points')).toHaveTextContent('48')
  })

  it('answers the highest and lowest team score of the week', () => {
    // Act
    renderWithProviders(<TiebreakerPanel card={card} />)

    // Assert
    expect(screen.getByLabelText('Highest team score')).toHaveTextContent('39')
    expect(screen.getByLabelText('Lowest team score')).toHaveTextContent('8')
    expect(screen.getByText(/DAL projected highest at 29.4/)).toBeInTheDocument()
  })

  it('says when a week has no games to break ties with', () => {
    // Act
    renderWithProviders(<TiebreakerPanel card={{ ...card, games: [], default_game_ids: [] }} />)

    // Assert
    expect(screen.getByText('No games to break ties with this week.')).toBeInTheDocument()
  })
})
