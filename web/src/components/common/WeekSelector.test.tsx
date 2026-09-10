import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { WeekRef } from '@/api/types'

import { WeekSelector, weekKey } from './WeekSelector'

const weeks: WeekRef[] = [
  { season: 2026, week: 1, source: 'active', run_id: 'weekly_a', label: 'Week 1 (active run)' },
  { season: 2026, week: 0, source: 'run', run_id: 'weekly_b', label: 'Week 0 (weekly_b)' },
  { season: 2026, week: 3, source: 'unattached', run_id: null, label: 'Week 3 (data/predict)' },
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

  it('renders nothing without weeks', () => {
    const { container } = render(<WeekSelector weeks={[]} value={null} onChange={() => {}} />)
    expect(container).toBeEmptyDOMElement()
  })
})
