import { useCallback } from 'react'
import { useSearchParams } from 'react-router'

/** Read and write one URL query parameter; the URL is the source of truth. */
export function useQueryParam(key: string): [string | null, (value: string | null) => void] {
  const [params, setParams] = useSearchParams()
  const value = params.get(key)
  const set = useCallback(
    (next: string | null) => {
      setParams(
        (current) => {
          const copy = new URLSearchParams(current)
          if (next === null || next === '') copy.delete(key)
          else copy.set(key, next)
          return copy
        },
        { replace: true },
      )
    },
    [key, setParams],
  )
  return [value, set]
}

export function useNumberParam(key: string): [number | null, (value: number | null) => void] {
  const [raw, set] = useQueryParam(key)
  const value = raw === null || raw === '' || Number.isNaN(Number(raw)) ? null : Number(raw)
  const setNumber = useCallback((next: number | null) => set(next === null ? null : String(next)), [set])
  return [value, setNumber]
}
