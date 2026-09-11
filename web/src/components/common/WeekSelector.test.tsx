import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { WeekRef } from '@/api/types'

import { WeekSelector, weekKey } from './WeekSelector'

const weeks: WeekRef[] = [
  { season: 2026, week: 1, source: 'active', run_id: 'weekly_a', label: 'Week 1 (active run)' },
  { season: 2026, week: 0, source: 'run', run_id: 'weekly_b', label: 'Week 0 (weekly_b)' },
  { season: 2026, week: 3, source: 'unattached', run_id: null, label: 'Week 3 (data/predict)' },
  { season: 2026, week: 5, source: 'available', run_id: null, label: 'Week 5 (not predicted yet)' },
]

describe('WeekSelector', () => {
  it('keys the active week independently of its run id', () => {
    expect(weekKey(weeks[0])).toBe(weekKey({ season: 2026, week: 1, source: 'active', run_id: 'anything' }))
    expect(weekKey(weeks[1])).not.toBe(weekKey({ ...weeks[1], run_id: 'other' }))
  })

  it('renders the selected week label', () => {
    render(<WeekSelector weeks={weeks} value={weekKey(weeks[2])} onChange={() => {}} />)
    expect(screen.getByText('2026 · Week 3 (data/predict)')).toBeInTheDocument()
  })

  it('offers weeks that have not been predicted yet', () => {
    const onChange = vi.fn()
    render(<WeekSelector weeks={weeks} value={weekKey(weeks[3])} onChange={onChange} />)
    expect(screen.getByText('2026 · Week 5 (not predicted yet)')).toBeInTheDocument()
    expect(weekKey(weeks[3])).not.toBe(weekKey(weeks[2]))
  })

  it('renders nothing without weeks', () => {
    const { container } = render(<WeekSelector weeks={[]} value={null} onChange={() => {}} />)
    expect(container).toBeEmptyDOMElement()
  })
})
