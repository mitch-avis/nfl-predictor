import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SurvivorOut } from '@/api/types'
import { renderWithProviders } from '@/test/render'

import { SurvivorPanel } from './SurvivorPanel'

const result: SurvivorOut = {
  season: 2026,
  start_week: 5,
  planned_weeks: [5],
  later_weeks: [],
  missing_weeks: [],
  used_teams: [],
  teams: ['BUF', 'KC', 'NE'],
  plans: [],
  future_values: [],
}

function stubSurvivorApi() {
  const fetchMock = vi.fn(async () => new Response(JSON.stringify(result), { status: 200, headers: { 'Content-Type': 'application/json' } }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('SurvivorPanel', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('asks for the plans with the near-term weeks and no search width', async () => {
    // Arrange
    const fetchMock = stubSurvivorApi()

    // Act
    renderWithProviders(<SurvivorPanel season={2026} week={5} />)

    // Assert
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(JSON.parse(String(init.body))).toEqual({ season: 2026, start_week: 5, used_teams: [], horizon_weeks: 4 })
    expect(screen.queryByLabelText('Search width')).not.toBeInTheDocument()
  })

  it('keeps the season’s stored teams when the season arrives after the panel opens', async () => {
    // Arrange
    window.localStorage.setItem('nflp.survivor.used.2026', JSON.stringify(['KC']))
    stubSurvivorApi()
    const user = userEvent.setup()
    const view = renderWithProviders(<SurvivorPanel season={null} week={null} />)
    view.rerender(<SurvivorPanel season={2026} week={5} />)
    const buf = await screen.findByRole('button', { name: 'BUF' })

    // Act
    await user.click(buf)

    // Assert
    expect(JSON.parse(window.localStorage.getItem('nflp.survivor.used.2026') ?? '[]')).toEqual(['BUF', 'KC'])
  })
})
