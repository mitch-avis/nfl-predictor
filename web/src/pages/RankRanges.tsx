import { useMemo } from 'react'

import { isMissing, useEntityTable, useRankRanges } from '@/api/queries'
import type { EntityKind } from '@/api/types'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { RankIntervalKey } from '@/components/ratings/RankInterval'
import { RankRangeChart } from '@/components/ratings/RankRangeChart'
import { RatingsSeasonState } from '@/components/ratings/RatingsStates'
import { SeasonPicker } from '@/components/ratings/SeasonPicker'
import { SectionCard } from '@/components/ratings/SectionCard'
import { parseRankRanges } from '@/components/ratings/rankRanges'
import { useRatingsSeason } from '@/components/ratings/useRatingsParams'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useQueryParam } from '@/hooks/useQueryParam'

/** Where every team or qualifying quarterback ranks across the season's redraws. */
export function RankRangesPage() {
  const [kindParam, setKind] = useQueryParam('kind')
  const kind: EntityKind = kindParam === 'qbs' ? 'qbs' : 'teams'
  const { query: seasonsQuery, season, seasons, setSeason } = useRatingsSeason()
  const ranges = useRankRanges(kind, season)
  const teams = useEntityTable('teams', kind === 'teams' ? season : null)
  const parsed = useMemo(() => (ranges.data ? parseRankRanges(kind, ranges.data) : []), [kind, ranges.data])
  const labels = kind === 'teams' ? (teams.data?.team_names ?? {}) : {}
  const unit = kind === 'teams' ? 'team' : 'qualifying quarterback'
  return (
    <>
      <PageHeader
        title={season === null ? 'Rank ranges' : `Rank ranges · ${season}`}
        description="How much each rank depends on which games happened to be played: the season’s games are redrawn at random, with repeats, 1,000 times and everyone is re-rated each time."
        actions={<SeasonPicker seasons={seasons} season={season} onChange={setSeason} />}
      />
      {season === null ? <RatingsSeasonState isLoading={seasonsQuery.isLoading} error={seasonsQuery.error} /> : null}
      {season !== null ? (
        <div className="flex flex-col gap-4">
          <Tabs value={kind} onValueChange={(value) => setKind(value === 'teams' ? null : value)}>
            <TabsList>
              <TabsTrigger value="teams">Teams</TabsTrigger>
              <TabsTrigger value="qbs">Quarterbacks</TabsTrigger>
            </TabsList>
          </Tabs>
          {ranges.isLoading ? <Skeleton className="h-96 w-full" /> : null}
          {ranges.isError && isMissing(ranges.error) ? <EmptyState title={`No rank ranges for ${season}`} description="The season was built without them." /> : null}
          {ranges.isError && !isMissing(ranges.error) ? <ErrorState error={ranges.error} title="Could not load the rank ranges" /> : null}
          {parsed.length > 0 ? (
            <SectionCard
              title={kind === 'teams' ? 'Teams' : 'Quarterbacks'}
              description={`One row per ${unit}, ordered by median rank, rank 1 at the left. The spread shows how much a rank depends on which games were played, not whether the rating is right; a season in progress shows wider ranges.`}
            >
              <RankIntervalKey />
              <RankRangeChart kind={kind} season={season} ranges={parsed} labels={labels} />
            </SectionCard>
          ) : null}
        </div>
      ) : null}
    </>
  )
}
