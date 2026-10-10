import { useSearchParams } from 'react-router'

import type { WeekRef } from '@/api/types'

const WEEK_KEYS = ['run', 'season', 'week', 'source'] as const

function numberParam(raw: string | null): number | null {
  return raw === null || raw === '' || Number.isNaN(Number(raw)) ? null : Number(raw)
}

/**
 * Which week's predictions a page shows, kept in the URL: a non-active run by its id alone,
 * anything else by season and week, plus `source=unattached` for a `data/predict` file.
 *
 * `select` rewrites all four parameters in one navigation. Separate setters would each start from
 * the URL of the last render, so only the last of them would take effect.
 */
export function useWeekParams() {
  const [searchParams, setSearchParams] = useSearchParams()
  const params = {
    run: searchParams.get('run'),
    season: numberParam(searchParams.get('season')),
    week: numberParam(searchParams.get('week')),
    source: searchParams.get('source'),
  }
  const select = (ref: WeekRef | null) => {
    const next: Record<(typeof WEEK_KEYS)[number], string | number | null | undefined> = {
      run: ref?.source === 'run' ? ref.run_id : null,
      season: ref && ref.source !== 'run' ? ref.season : null,
      week: ref && ref.source !== 'run' ? ref.week : null,
      source: ref?.source === 'unattached' ? 'unattached' : null,
    }
    setSearchParams(
      (current) => {
        const copy = new URLSearchParams(current)
        for (const key of WEEK_KEYS) {
          const value = next[key]
          if (value === null || value === undefined || value === '') copy.delete(key)
          else copy.set(key, String(value))
        }
        return copy
      },
      { replace: true },
    )
  }
  return { params, select }
}
