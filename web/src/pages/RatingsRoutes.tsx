/**
 * The team and quarterback pages, loaded on first visit: they bring the chart library, which the
 * rest of the app loads only for the Model page.
 */
import { lazy, Suspense, type ReactNode } from 'react'
import { useParams } from 'react-router'

import type { EntityKind } from '@/api/types'
import { Skeleton } from '@/components/ui/skeleton'

const RatingsIndexPage = lazy(() => import('./Ratings').then((module) => ({ default: module.RatingsIndexPage })))
const RatingsDetailPage = lazy(() => import('./RatingsDetail').then((module) => ({ default: module.RatingsDetailPage })))
const SchedulesPage = lazy(() => import('./Schedules').then((module) => ({ default: module.SchedulesPage })))
const RankRangesPage = lazy(() => import('./RankRanges').then((module) => ({ default: module.RankRangesPage })))
const HeadToHeadPage = lazy(() => import('./HeadToHead').then((module) => ({ default: module.HeadToHeadPage })))

function Lazy({ children }: { children: ReactNode }) {
  return <Suspense fallback={<Skeleton className="h-64 w-full" />}>{children}</Suspense>
}

export function RatingsIndexRoute({ kind }: { kind: EntityKind }) {
  return (
    <Lazy>
      <RatingsIndexPage kind={kind} />
    </Lazy>
  )
}

/** One team's or quarterback's page; a new id starts the page afresh (its pickers included). */
export function RatingsDetailRoute({ kind }: { kind: EntityKind }) {
  const { entityId = '' } = useParams()
  return (
    <Lazy>
      <RatingsDetailPage key={`${kind}-${entityId}`} kind={kind} entityId={entityId} />
    </Lazy>
  )
}

export function SchedulesRoute() {
  return (
    <Lazy>
      <SchedulesPage />
    </Lazy>
  )
}

export function RankRangesRoute() {
  return (
    <Lazy>
      <RankRangesPage />
    </Lazy>
  )
}

export function HeadToHeadRoute() {
  return (
    <Lazy>
      <HeadToHeadPage />
    </Lazy>
  )
}
