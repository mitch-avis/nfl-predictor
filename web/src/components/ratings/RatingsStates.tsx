import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { Skeleton } from '@/components/ui/skeleton'

const CODE = 'rounded bg-muted px-1'

/** What a ratings page shows until it has a season: loading, an error, or no tables yet. */
export function RatingsSeasonState({ isLoading, error }: { isLoading: boolean; error: unknown }) {
  if (isLoading) return <Skeleton className="h-64 w-full" />
  if (error) return <ErrorState error={error} title="Could not load the ratings seasons" />
  return (
    <EmptyState
      title="No team or quarterback tables yet"
      description={
        <>
          These pages read each season’s tables from <code className={CODE}>data/ratings/</code>, starting with <code className={CODE}>{'{season}_combined.parquet'}</code> and{' '}
          <code className={CODE}>{'{season}_qb_combined.parquet'}</code>, and that folder holds no season with both yet. The ETL writes them (<code className={CODE}>nfl-predictor data</code>, or
          the Full ETL rebuild on the Jobs page); run it, then reload.
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
