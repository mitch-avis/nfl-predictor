import { screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { User } from '@/api/types'
import { SidebarProvider } from '@/components/ui/sidebar'
import { renderWithProviders } from '@/test/render'

import { AppSidebar } from './AppSidebar'

const viewer: User = { id: 2, username: 'viewer', role: 'viewer', created_at: '' }

function linksIn(section: string): string[] {
  const group = screen.getByRole('group', { name: section })
  return within(group)
    .getAllByRole('link')
    .map((link) => link.textContent ?? '')
}

describe('AppSidebar', () => {
  it('groups the pages into sections, the team and quarterback pages beside the power rankings', () => {
    // Act
    renderWithProviders(
      <SidebarProvider>
        <AppSidebar user={viewer} />
      </SidebarProvider>,
      { route: '/teams' },
    )

    // Assert
    expect(linksIn('Teams and players')).toEqual(['Power Rankings', 'Teams', 'Quarterbacks', 'Schedules', 'Rank ranges', 'Head to head'])
    expect(linksIn('This week')).toEqual(['Overview', 'Predictions', 'Pool', 'Betting'])
  })

  it('marks the page in view, its detail pages included', () => {
    // Act
    renderWithProviders(
      <SidebarProvider>
        <AppSidebar user={viewer} />
      </SidebarProvider>,
      { route: '/qbs/00-1' },
    )

    // Assert
    expect(screen.getByRole('link', { name: 'Quarterbacks' })).toHaveAttribute('data-active', 'true')
  })

  it('keeps the admin pages from a viewer', () => {
    // Act
    renderWithProviders(
      <SidebarProvider>
        <AppSidebar user={viewer} />
      </SidebarProvider>,
    )

    // Assert
    expect(screen.queryByRole('link', { name: 'Users' })).not.toBeInTheDocument()
  })
})
