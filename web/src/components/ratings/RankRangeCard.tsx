import type { EntityKind, RankRangeTable, Row } from '@/api/types'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

import { RankHistogram } from './RankHistogram'
import { belowQualifierText, parseRankRanges, parseUnitRankRanges, rankChanceText, rankRangeHeadline } from './rankRanges'
import { UnitRankRanges } from './UnitRankRanges'

/**
 * Where one team or quarterback ranked across the season's redraws: the headline ranges, the top-5
 * and top-10 chances, the chance of each rank and, for a team, each unit's range. A quarterback
 * below the qualifier gets the reason he has none; a season without rank ranges shows nothing.
 */
export function RankRangeCard({ kind, season, entityId, ranges, row }: { kind: EntityKind; season: number; entityId: string; ranges: RankRangeTable | undefined; row: Row }) {
  const range = ranges ? parseRankRanges(kind, ranges).find((candidate) => candidate.id === entityId) : undefined
  const units = ranges && kind === 'teams' ? parseUnitRankRanges(ranges, entityId) : []
  if (!range) {
    const reason = belowQualifierText(row)
    if (!reason) return null
    return (
      <Card className="gap-2 px-4 py-3">
        <div className="text-sm font-medium">Rank range</div>
        <p className="text-sm text-muted-foreground">{reason}</p>
      </Card>
    )
  }
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle className="text-base">Rank range</CardTitle>
        <CardDescription>
          Where this {kind === 'teams' ? 'team' : 'quarterback'} ranked across 1,000 redraws of the {season} season: its games drawn at random, with repeats, and everyone re-rated each time. It shows how much the
          rank depends on which games happened to be played, not whether the rating is right.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div>
          <p className="tabular text-lg font-semibold">{rankRangeHeadline(range)}</p>
          <p className="text-sm text-muted-foreground">{rankChanceText(kind, range)}</p>
        </div>
        <RankHistogram range={range} />
        {units.length > 0 ? <UnitRankRanges ranges={units} count={ranges?.table.rows.length ?? 0} /> : null}
      </CardContent>
    </Card>
  )
}
