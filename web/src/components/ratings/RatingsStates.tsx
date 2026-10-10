import { Link } from 'react-router'

import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { Skeleton } from '@/components/ui/skeleton'

/** What a ratings page shows until it has a season: loading, an error, or no tables yet. */
export function RatingsSeasonState({ isLoading, error }: { isLoading: boolean; error: unknown }) {
  if (isLoading) return <Skeleton className="h-64 w-full" />
  if (error) return <ErrorState error={error} title="Could not load the ratings seasons" />
  return (
    <EmptyState
      title="No team or quarterback tables yet"
      description={
        <>
          The ETL writes each season’s ratings and statistics to <code className="rounded bg-muted px-1">data/ratings/</code>. Run it from the{' '}
          <Link to="/jobs" className="underline underline-offset-2">
            Jobs page
          </Link>
          , then reload.
        </>
      }
    />
  )
}

/** A note that the season asked for has no tables, naming the one shown instead. */
export function UnavailableSeason({ requested, shown }: { requested: string; shown: number }) {
  return (
    <p role="status" className="rounded-lg border border-dashed px-3 py-2 text-sm text-muted-foreground">
      No tables for the {requested} season; showing {shown}.
    </p>
  )
}
