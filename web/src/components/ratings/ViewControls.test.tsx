import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { QB_CATEGORIES, TEAM_CATEGORIES } from '@/test/ratingsFixtures'

import { ViewControls } from './ViewControls'
import { resolveViewState } from './views'

describe('ViewControls', () => {
  it('marks the active view and hides the filters in the ratings view', () => {
    // Arrange
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'ratings' })

    // Act
    render(<ViewControls kind="teams" state={state} onChange={() => {}} />)

    // Assert
    expect(screen.getByRole('button', { name: 'Ratings' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.queryByRole('group', { name: 'Category' })).not.toBeInTheDocument()
  })

  it('switches the view', async () => {
    // Arrange
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<ViewControls kind="teams" state={resolveViewState('teams', TEAM_CATEGORIES, {})} onChange={onChange} />)

    // Act
    await user.click(screen.getByRole('button', { name: 'Per play' }))

    // Assert
    expect(onChange).toHaveBeenCalledWith({ view: 'per_play' })
  })

  it('switches the team category and clears the hidden subcategories', async () => {
    // Arrange
    const onChange = vi.fn()
    const user = userEvent.setup()
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'per_game', hidden: ['Rushing'] })
    render(<ViewControls kind="teams" state={state} onChange={onChange} />)

    // Act
    await user.click(within(screen.getByRole('group', { name: 'Category' })).getByRole('button', { name: 'Defense' }))

    // Assert
    expect(onChange).toHaveBeenCalledWith({ category: 'Defense', hidden: [] })
  })

  it('hides a subcategory that is switched on', async () => {
    // Arrange
    const onChange = vi.fn()
    const user = userEvent.setup()
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'per_game' })
    render(<ViewControls kind="teams" state={state} onChange={onChange} />)

    // Act
    await user.click(within(screen.getByRole('group', { name: 'Subcategories' })).getByRole('button', { name: 'Rushing' }))

    // Assert
    expect(onChange).toHaveBeenCalledWith({ hidden: ['Rushing'] })
  })

  it('shows a hidden quarterback category switched off and switches it back on', async () => {
    // Arrange
    const onChange = vi.fn()
    const user = userEvent.setup()
    const state = resolveViewState('qbs', QB_CATEGORIES, { view: 'per_game', hidden: ['Passing Volume'] })
    render(<ViewControls kind="qbs" state={state} onChange={onChange} />)
    const volume = within(screen.getByRole('group', { name: 'Categories' })).getByRole('button', { name: 'Passing Volume' })

    // Act
    await user.click(volume)

    // Assert
    expect(volume).toHaveAttribute('aria-pressed', 'false')
    expect(onChange).toHaveBeenCalledWith({ hidden: [] })
  })
})
