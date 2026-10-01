import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useIsMobile } from './use-mobile'

function ViewportProbe() {
  return <div>{useIsMobile() ? 'mobile' : 'desktop'}</div>
}

describe('useIsMobile', () => {
  const originalInnerWidth = window.innerWidth
  const originalMatchMedia = window.matchMedia
  let listeners: Array<() => void> = []

  beforeEach(() => {
    listeners = []
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      writable: true,
      value: 1024,
    })
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: window.innerWidth < 768,
      media: query,
      onchange: null,
      addEventListener: (_event: string, listener: () => void) => {
        listeners.push(listener)
      },
      removeEventListener: (_event: string, listener: () => void) => {
        listeners = listeners.filter((candidate) => candidate !== listener)
      },
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    })) as typeof window.matchMedia
  })

  afterEach(() => {
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      writable: true,
      value: originalInnerWidth,
    })
    window.matchMedia = originalMatchMedia
  })

  it('derives the initial viewport state during render', () => {
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      writable: true,
      value: 640,
    })

    render(<ViewportProbe />)

    expect(screen.getByText('mobile')).toBeInTheDocument()
  })

  it('updates when the viewport listener fires', () => {
    render(<ViewportProbe />)
    expect(screen.getByText('desktop')).toBeInTheDocument()

    act(() => {
      Object.defineProperty(window, 'innerWidth', {
        configurable: true,
        writable: true,
        value: 640,
      })
      listeners.forEach((listener) => listener())
    })

    expect(screen.getByText('mobile')).toBeInTheDocument()
  })
})
