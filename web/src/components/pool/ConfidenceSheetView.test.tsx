import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import type { ConfidenceSheet, SheetRow } from '@/api/types'

import { renderWithProviders } from '@/test/render'

import { ConfidenceSheetView } from './ConfidenceSheetView'

function row(overrides: Partial<SheetRow>): SheetRow {
  return {
    game_id: 'g',
    points: 1,
    pick: 'DAL',
    opponent: 'TB',
    pick_is_home: true,
    away_abbr: 'TB',
    home_abbr: 'DAL',
    kickoff: '2026-10-08T20:15:00-04:00',
    pick_win_prob: 0.79,
    expected_points: 0.79,
    away_score: null,
    home_score: null,
    result: null,
    points_won: null,
    ...overrides,
  }
}

function sheet(overrides: Partial<ConfidenceSheet> = {}): ConfidenceSheet {
  return {
    run_id: 'weekly_w5',
    source: 'active',
    season: 2026,
    week: 5,
    generated_at: '2026-10-07T12:00:00+00:00',
    rows: [
      row({ game_id: 'a', points: 3, pick: 'DAL', opponent: 'TB', kickoff: '2026-10-11T16:25:00-04:00' }),
      row({ game_id: 'b', points: 2, pick: 'HOU', opponent: 'TEN', pick_is_home: false, away_abbr: 'HOU', home_abbr: 'TEN', kickoff: '2026-10-08T20:15:00-04:00' }),
      row({ game_id: 'c', points: 1, pick: 'PIT', opponent: 'IND', away_abbr: 'IND', home_abbr: 'PIT', kickoff: '2026-10-11T13:00:00-04:00' }),
    ],
    games: 3,
    max_points: 6,
    expected_points: 4.2,
    games_final: 0,
    points_won: null,
    points_possible: 6,
    lock_at: '2026-10-08T20:15:00-04:00',
    generated_after_lock: false,
    rank_problem: null,
    ...overrides,
  }
}

function picksInOrder(): string[] {
  return within(screen.getByRole('list', { name: 'Confidence sheet' }))
    .getAllByRole('listitem')
    .map((item) => item.getAttribute('data-pick') ?? '')
}

describe('ConfidenceSheetView', () => {
  it('lists the picks with the most points first', () => {
    // Act
    renderWithProviders(<ConfidenceSheetView sheet={sheet()} csvUrl="/csv" />)

    // Assert
    expect(picksInOrder()).toEqual(['DAL', 'HOU', 'PIT'])
    expect(screen.getByText('at TEN')).toBeInTheDocument()
  })

  it('reorders the picks by kickoff on request', async () => {
    // Arrange
    const user = userEvent.setup()
    renderWithProviders(<ConfidenceSheetView sheet={sheet()} csvUrl="/csv" />)

    // Act
    await user.click(screen.getByRole('button', { name: 'By kickoff' }))

    // Assert
    expect(picksInOrder()).toEqual(['HOU', 'PIT', 'DAL'])
  })

  it('offers the sheet as a CSV download', () => {
    // Act
    renderWithProviders(<ConfidenceSheetView sheet={sheet()} csvUrl="/api/predictions/picks?format=csv" />)

    // Assert
    expect(screen.getByRole('link', { name: /Download CSV/ })).toHaveAttribute('href', '/api/predictions/picks?format=csv')
  })

  it('warns when the predictions were written after the first kickoff', () => {
    // Act
    renderWithProviders(<ConfidenceSheetView sheet={sheet({ generated_after_lock: true })} csvUrl="/csv" />)

    // Assert
    expect(screen.getByRole('alert')).toHaveTextContent(/written after the week’s first kickoff/)
  })

  it('reports ranks that are not one to N', () => {
    // Act
    renderWithProviders(<ConfidenceSheetView sheet={sheet({ rank_problem: 'Confidence ranks should run 1 to 3 once each: 3 repeats.' })} csvUrl="/csv" />)

    // Assert
    expect(screen.getByRole('alert')).toHaveTextContent('3 repeats')
  })

  it('scores final games, a tie counting for neither side', () => {
    // Arrange
    const scored = sheet({
      rows: [
        row({ game_id: 'a', points: 3, result: 'win', points_won: 3, away_score: 10, home_score: 31 }),
        row({ game_id: 'b', points: 2, pick: 'HOU', result: 'tie', points_won: 0, away_score: 20, home_score: 20 }),
        row({ game_id: 'c', points: 1, pick: 'PIT' }),
      ],
      games_final: 2,
      points_won: 3,
      points_possible: 4,
    })

    // Act
    renderWithProviders(<ConfidenceSheetView sheet={scored} csvUrl="/csv" />)

    // Assert
    expect(screen.getByText('+3')).toBeInTheDocument()
    expect(screen.getByText('Tie, no points')).toBeInTheDocument()
    expect(screen.getByText(/2 of 3 games final · 4 still possible/)).toBeInTheDocument()
  })
})
