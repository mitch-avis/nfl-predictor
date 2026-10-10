import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ThemeProvider } from '@/app/ThemeProvider'
import { TooltipProvider } from '@/components/ui/tooltip'
import { stubApi } from '@/test/api'
import { qbTable, teamTable } from '@/test/ratingsFixtures'

import { routes } from './router'

const SESSION = { user: { id: 1, username: 'viewer', role: 'viewer', created_at: '' } }

/**
 * How long a test waits for its page. Each test renders the whole app and a lazily loaded page; in
 * a full run on a busy machine the team page has taken more than Testing Library's default second.
 */
const PAGE_LOAD = { timeout: 3000 }

function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(routes, { initialEntries: [path] })
  return render(
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <TooltipProvider>
          <RouterProvider router={router} />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>,
  )
}

describe('router', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('opens a team’s page from its path', async () => {
    // Arrange
    stubApi({
      '/api/auth/me': SESSION,
      '/api/runs': { active_run_id: null, pinned_run_id: null, runs: [] },
      '/api/ratings/seasons': { seasons: [2025], current_season: 2026 },
      '/api/ratings/2025/teams': teamTable(),
    })

    // Act
    renderAt('/teams/WSH?season=2025')

    // Assert
    expect(await screen.findByRole('heading', { name: 'Washington Commanders' }, PAGE_LOAD)).toBeInTheDocument()
  })

  it('opens the quarterback index from its path', async () => {
    // Arrange
    stubApi({
      '/api/auth/me': SESSION,
      '/api/runs': { active_run_id: null, pinned_run_id: null, runs: [] },
      '/api/ratings/seasons': { seasons: [2025], current_season: 2026 },
      '/api/ratings/2025/qbs': qbTable(),
    })

    // Act
    renderAt('/qbs')

    // Assert
    expect(await screen.findByRole('heading', { name: 'Quarterbacks · 2025' }, PAGE_LOAD)).toBeInTheDocument()
  })

  it.each([
    ['/schedules', 'Schedules'],
    ['/rank-ranges', 'Rank ranges · 2025'],
    ['/head-to-head', 'Head to head · 2025'],
  ])('opens %s', async (path, heading) => {
    // Arrange
    stubApi({
      '/api/auth/me': SESSION,
      '/api/runs': { active_run_id: null, pinned_run_id: null, runs: [] },
      '/api/ratings/seasons': { seasons: [2025], current_season: 2026 },
    })

    // Act
    renderAt(path)

    // Assert
    expect(await screen.findByRole('heading', { name: heading }, PAGE_LOAD)).toBeInTheDocument()
  })
})
