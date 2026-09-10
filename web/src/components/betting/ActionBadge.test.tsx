import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ActionBadge } from './ActionBadge'

describe('ActionBadge', () => {
  it('renders the ladder label with a rung-specific style', () => {
    render(<ActionBadge action="STRONG" />)
    expect(screen.getByText('STRONG').className).toContain('bg-success')
  })

  it('falls back for unknown or missing actions', () => {
    render(<ActionBadge action={null} muted />)
    const el = screen.getByText('—')
    expect(el.className).toContain('grayscale')
  })
})
