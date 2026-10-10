import { screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { renderWithProviders } from '@/test/render'

import { NotFoundPage } from './NotFound'

describe('NotFoundPage', () => {
  it('says a path without a page does not exist', () => {
    // Act
    renderWithProviders(<NotFoundPage />, { route: '/no-such-page' })

    // Assert
    expect(screen.getByText('This page doesn’t exist.')).toBeInTheDocument()
    expect(screen.getByText('/no-such-page')).toBeInTheDocument()
    expect(screen.queryByText(/phase/i)).not.toBeInTheDocument()
  })

  it('links back to the overview', () => {
    // Act
    renderWithProviders(<NotFoundPage />, { route: '/no-such-page' })

    // Assert
    expect(screen.getByRole('link', { name: 'Go to the overview' })).toHaveAttribute('href', '/')
  })
})
