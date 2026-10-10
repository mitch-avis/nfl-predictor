import { screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { SurvivorOut, SurvivorPick } from '@/api/types'
import { renderWithProviders } from '@/test/render'

import { SurvivorPlans } from './SurvivorPlans'

function pick(week: number, team: string, opponent: string, winProb: number, later = false): SurvivorPick {
  return { week, team, opponent, is_home: true, win_prob: winProb, game_id: `${week}${team}`, kickoff: null, later, cumulative_survival: winProb }
}

const result: SurvivorOut = {
  season: 2026,
  start_week: 5,
  planned_weeks: [5, 6],
  later_weeks: [7],
  missing_weeks: [8, 9, 10],
  used_teams: ['KC'],
  teams: ['BUF', 'GB', 'KC', 'NE'],
  plans: [
    { rank: 1, picks: [pick(5, 'NE', 'NYJ', 0.71), pick(6, 'BUF', 'HOU', 0.67), pick(7, 'GB', 'MIN', 0.63, true)], survival: 0.4757, later_survival: 0.63, total_survival: 0.2997 },
    { rank: 2, picks: [pick(5, 'BUF', 'HOU', 0.67), pick(6, 'NE', 'NYJ', 0.71), pick(7, 'GB', 'MIN', 0.63, true)], survival: 0.4757, later_survival: 0.63, total_survival: 0.2997 },
  ],
  future_values: [
    { team: 'GB', value: 0.12 },
    { team: 'NE', value: 0.0 },
  ],
}

describe('SurvivorPlans', () => {
  it('ranks one plan per pick this week', () => {
    // Act
    renderWithProviders(<SurvivorPlans result={result} />)

    // Assert
    const plans = within(screen.getByRole('list', { name: 'Survivor plans' })).getAllByRole('listitem')
    expect(plans.map((plan) => plan.getAttribute('data-first'))).toEqual(['NE', 'BUF'])
    expect(within(plans[0]).getByText('30.0%')).toBeInTheDocument()
  })

  it('marks the picks for later weeks', () => {
    // Act
    renderWithProviders(<SurvivorPlans result={result} />)

    // Assert
    const plans = within(screen.getByRole('list', { name: 'Survivor plans' })).getAllByRole('listitem')
    const later = within(plans[0]).getByTitle('Week 7: a later pick in the plan')
    expect(later).toHaveTextContent('GB')
  })

  it('says which weeks have no predictions and where to make them', () => {
    // Act
    renderWithProviders(<SurvivorPlans result={result} />)

    // Assert
    expect(screen.getByText(/No predictions for weeks 8–10/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Predictions page' })).toHaveAttribute('href', '/predictions')
  })

  it('lists the teams worth saving for later', () => {
    // Act
    renderWithProviders(<SurvivorPlans result={result} />)

    // Assert
    const saving = screen.getByRole('list', { name: 'Teams worth saving' })
    expect(within(saving).getAllByRole('listitem').map((item) => item.textContent)).toEqual(['GB12%'])
  })

  it('explains an empty result', () => {
    // Act
    renderWithProviders(<SurvivorPlans result={{ ...result, plans: [] }} />)

    // Assert
    expect(screen.getByText('No plan from this week')).toBeInTheDocument()
  })
})
