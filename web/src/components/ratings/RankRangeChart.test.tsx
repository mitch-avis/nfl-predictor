import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { renderWithProviders } from '@/test/render'
import { teamRankRanges } from '@/test/ratingsFixtures'

import { RankRangeChart } from './RankRangeChart'
import { parseRankRanges } from './rankRanges'

describe('RankRangeChart', () => {
  it('lists every team by median rank, each linking to its page for the season', () => {
    // Arrange
    const ranges = parseRankRanges('teams', teamRankRanges())

    // Act
    renderWithProviders(<RankRangeChart kind="teams" season={2025} ranges={ranges} labels={{ BUF: 'Buffalo Bills' }} />)

    // Assert
    const links = within(screen.getByRole('list')).getAllByRole('link')
    expect(links.map((link) => link.getAttribute('href'))).toEqual(['/teams/BUF?season=2025', '/teams/KC?season=2025', '/teams/WSH?season=2025'])
    expect(links[0]).toHaveAccessibleName('Buffalo Bills: published rank 1st, median 1st; middle 50%: 1st–2nd; 95%: 1st–2nd')
  })

  it('reads out the hovered row’s ranges', async () => {
    // Arrange
    const user = userEvent.setup()
    const ranges = parseRankRanges('teams', teamRankRanges())
    renderWithProviders(<RankRangeChart kind="teams" season={2025} ranges={ranges} labels={{}} />)

    // Act
    await user.hover(screen.getByRole('link', { name: /^KC:/ }))

    // Assert
    expect(screen.getByRole('status')).toHaveTextContent('KC: 2nd; middle 50%: 2nd; 95%: 1st–3rd')
  })
})
