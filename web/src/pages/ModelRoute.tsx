import { lazy, Suspense } from 'react'

import { Skeleton } from '@/components/ui/skeleton'

const ModelPage = lazy(() => import('./Model').then((module) => ({ default: module.ModelPage })))

export function ModelRoute() {
  return (
    <Suspense fallback={<Skeleton className="h-64 w-full" />}>
      <ModelPage />
    </Suspense>
  )
}
