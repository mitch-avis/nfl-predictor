import { useCallback } from 'react'
import { useSearchParams } from 'react-router'

import { useRatingsSeasons } from '@/api/queries'

import { isStatView, type StatView, type ViewRequest } from './views'

/** The URL parameters the ratings pages share, so a view survives the trip to a detail page. */
export const VIEW_PARAMS = ['view', 'category', 'hide'] as const

export type ParamPatch = Record<string, string | number | null>

/**
 * The ratings pages' URL state: the season, the view and its filters. `update` rewrites several
 * parameters in one navigation; separate setters would each start from the last render's URL, so
 * only the last would take effect.
 */
export function useRatingsParams() {
  const [searchParams, setSearchParams] = useSearchParams()
  const update = useCallback(
    (patch: ParamPatch) =>
      setSearchParams(
        (current) => {
          const copy = new URLSearchParams(current)
          for (const [key, value] of Object.entries(patch)) {
            if (value === null || value === '') copy.delete(key)
            else copy.set(key, String(value))
          }
          return copy
        },
        { replace: true },
      ),
    [setSearchParams],
  )
  const view = searchParams.get('view')
  const request: ViewRequest = {
    view: isStatView(view) ? view : null,
    category: searchParams.get('category'),
    hidden: (searchParams.get('hide') ?? '').split(',').filter(Boolean),
  }
  return { searchParams, update, request }
}

/** The patch that records a view request in the URL (`hide` as a comma list). */
export function viewPatch(request: { view?: StatView; category?: string; hidden?: string[] }): ParamPatch {
  const patch: ParamPatch = {}
  if (request.view !== undefined) patch.view = request.view
  if (request.category !== undefined) patch.category = request.category
  if (request.hidden !== undefined) patch.hide = request.hidden.join(',')
  return patch
}

/** The query string a link carries to another ratings page: the season and the view's filters. */
export function carriedSearch(searchParams: URLSearchParams, season: number | null): string {
  const carried = new URLSearchParams()
  if (season !== null) carried.set('season', String(season))
  for (const key of VIEW_PARAMS) {
    const value = searchParams.get(key)
    if (value) carried.set(key, value)
  }
  const text = carried.toString()
  return text ? `?${text}` : ''
}

/**
 * The season in `?season=`, falling back to the newest season with tables. `unavailable` is a
 * requested season the list does not have, so the page can say what it shows instead.
 */
export function useRatingsSeason() {
  const query = useRatingsSeasons()
  const { searchParams, update } = useRatingsParams()
  const seasons = query.data?.seasons ?? []
  const requestedText = searchParams.get('season')
  const requested = requestedText === null ? null : Number(requestedText)
  const season = requested !== null && seasons.includes(requested) ? requested : (seasons[0] ?? null)
  const unavailable = requestedText !== null && query.isSuccess && seasons.length > 0 && !seasons.includes(Number(requestedText)) ? requestedText : null
  const setSeason = useCallback((next: number) => update({ season: next }), [update])
  return { query, season, seasons, unavailable, setSeason, currentSeason: query.data?.current_season ?? null }
}
