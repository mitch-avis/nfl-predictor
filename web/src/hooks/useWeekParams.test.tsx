import { act, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'

import type { WeekRef } from '@/api/types'

import { useWeekParams } from './useWeekParams'

function wrapper(initial: string) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <MemoryRouter initialEntries={[initial]}>{children}</MemoryRouter>
  }
}

const runWeek: WeekRef = { season: 2026, week: 4, source: 'run', run_id: 'weekly_w4', label: 'Week 4 (weekly_w4)' }
const unattached: WeekRef = { season: 2026, week: 3, source: 'unattached', run_id: null, label: 'Week 3 (data/predict)' }
const activeWeek: WeekRef = { season: 2026, week: 6, source: 'active', run_id: 'weekly_w5', label: 'Week 6 (active run)' }

describe('useWeekParams', () => {
  it('reads the week from the URL', () => {
    // Act
    const { result } = renderHook(() => useWeekParams(), { wrapper: wrapper('/pool?season=2026&week=6') })

    // Assert
    expect(result.current.params).toEqual({ run: null, season: 2026, week: 6, source: null })
  })

  it('selects another run by its id alone', () => {
    // Arrange
    const { result } = renderHook(() => useWeekParams(), { wrapper: wrapper('/pool?season=2026&week=6') })

    // Act
    act(() => result.current.select(runWeek))

    // Assert
    expect(result.current.params).toEqual({ run: 'weekly_w4', season: null, week: null, source: null })
  })

  it('selects an unattached file by its season, week and source', () => {
    // Arrange
    const { result } = renderHook(() => useWeekParams(), { wrapper: wrapper('/pool') })

    // Act
    act(() => result.current.select(unattached))

    // Assert
    expect(result.current.params).toEqual({ run: null, season: 2026, week: 3, source: 'unattached' })
  })

  it('selects a week of the active run by its season and week', () => {
    // Arrange
    const { result } = renderHook(() => useWeekParams(), { wrapper: wrapper('/pool?run=weekly_w4') })

    // Act
    act(() => result.current.select(activeWeek))

    // Assert
    expect(result.current.params).toEqual({ run: null, season: 2026, week: 6, source: null })
  })

  it('clears the selection', () => {
    // Arrange
    const { result } = renderHook(() => useWeekParams(), { wrapper: wrapper('/pool?season=2026&week=6&source=unattached') })

    // Act
    act(() => result.current.select(null))

    // Assert
    expect(result.current.params).toEqual({ run: null, season: null, week: null, source: null })
  })
})
