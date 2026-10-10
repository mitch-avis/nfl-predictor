import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { TeamPicker } from './TeamPicker'

const teams = ['BUF', 'KC', 'NE']

describe('TeamPicker', () => {
  it('shows which teams are already used', () => {
    // Act
    render(<TeamPicker teams={teams} selected={['KC']} onChange={() => {}} />)

    // Assert
    expect(screen.getByRole('button', { name: 'KC' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'NE' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('adds a team when it is chosen', async () => {
    // Arrange
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TeamPicker teams={teams} selected={['KC']} onChange={onChange} />)

    // Act
    await user.click(screen.getByRole('button', { name: 'BUF' }))

    // Assert
    expect(onChange).toHaveBeenCalledWith(['BUF', 'KC'])
  })

  it('removes a used team when it is chosen again', async () => {
    // Arrange
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TeamPicker teams={teams} selected={['KC', 'NE']} onChange={onChange} />)

    // Act
    await user.click(screen.getByRole('button', { name: 'KC' }))

    // Assert
    expect(onChange).toHaveBeenCalledWith(['NE'])
  })

  it('clears every used team at once', async () => {
    // Arrange
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TeamPicker teams={teams} selected={['KC', 'NE']} onChange={onChange} />)

    // Act
    await user.click(screen.getByRole('button', { name: 'Clear' }))

    // Assert
    expect(onChange).toHaveBeenCalledWith([])
  })
})
