import { screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { renderWithProviders } from '@/test/render'
import { qbTable, teamRankRanges } from '@/test/ratingsFixtures'

import { RankRangeCard } from './RankRangeCard'

describe('RankRangeCard', () => {
  it('gives a team’s published rank, its ranges and its chances', () => {
    // Act
    renderWithProviders(<RankRangeCard kind="teams" season={2025} entityId="KC" ranges={teamRankRanges()} row={{ team: 'KC' }} />)

    // Assert
    expect(screen.getByText('2nd; middle 50%: 2nd; 95%: 1st–3rd')).toBeInTheDocument()
    expect(screen.getByText('Top 5 in 90% of redraws, top 10 in 100%')).toBeInTheDocument()
  })

  it('lists each unit’s rank range beside the team’s', () => {
    // Act
    renderWithProviders(<RankRangeCard kind="teams" season={2025} entityId="KC" ranges={teamRankRanges()} row={{ team: 'KC' }} />)

    // Assert
    const units = screen.getByRole('table', { name: 'Rank range by unit' })
    expect(within(units).getByRole('rowheader', { name: 'Offense' })).toBeInTheDocument()
    expect(within(units).getByRole('rowheader', { name: 'Defense' })).toBeInTheDocument()
  })

  it('says a quarterback below the qualifier has no rank range', () => {
    // Arrange
    const backup = qbTable().table.rows[2]

    // Act
    renderWithProviders(<RankRangeCard kind="qbs" season={2025} entityId="00-9" ranges={undefined} row={backup} />)

    // Assert
    expect(screen.getByText('Not ranked: below the qualifier (too few pass attempts), so no rank range.')).toBeInTheDocument()
  })

  it('shows nothing for a season without rank ranges', () => {
    // Act
    const view = renderWithProviders(<RankRangeCard kind="teams" season={2025} entityId="KC" ranges={undefined} row={{ team: 'KC' }} />)

    // Assert
    expect(view.container).toBeEmptyDOMElement()
  })
})
