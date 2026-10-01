import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { SidebarMenuSkeleton } from './sidebar'

describe('SidebarMenuSkeleton', () => {
  it('keeps a stable width across rerenders', () => {
    const { container, rerender } = render(<SidebarMenuSkeleton />)
    const skeleton = container.querySelector('[data-sidebar="menu-skeleton-text"]') as HTMLElement | null

    expect(skeleton).not.toBeNull()
    const width = skeleton?.style.getPropertyValue('--skeleton-width') ?? ''
    expect(width).toMatch(/^\d+%$/)

    rerender(<SidebarMenuSkeleton showIcon />)

    const next = container.querySelector('[data-sidebar="menu-skeleton-text"]') as HTMLElement | null
    expect(next?.style.getPropertyValue('--skeleton-width')).toBe(width)
  })
})
