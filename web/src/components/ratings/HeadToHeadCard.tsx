import { useMemo, useState } from 'react'
import { Link } from 'react-router'

import { isMissing, useEntitySection } from '@/api/queries'
import type { EntityKind, Row } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

import { entityConfig, entityId, entityLabel, headToHeadPath } from './entities'
import { describeRatingPair, neighborId, parseRatingPairs } from './ratingPairs'

/**
 * How often this team or quarterback was rated above another across the season's redraws, with a
 * picker that starts on its neighbor in the published ranking. Hidden for a season built without
 * head-to-head chances.
 */
export function HeadToHeadCard({ kind, season, entityId: subject, rows, names }: { kind: EntityKind; season: number; entityId: string; rows: Row[]; names: Record<string, string> }) {
  const query = useEntitySection('head-to-head', kind, season, subject)
  const [picked, setPicked] = useState<string | null>(null)
  const pairs = useMemo(() => (query.data ? parseRatingPairs(kind, query.data) : []), [kind, query.data])
  const labels = useMemo(() => new Map(rows.map((row) => [entityId(kind, row), entityLabel(kind, row, names)])), [kind, names, rows])
  const rating = entityConfig(kind).ratingColumn
  const ranked = useMemo(() => {
    const compared = new Set([subject, ...pairs.map((pair) => pair.otherId)])
    return rows
      .filter((row) => compared.has(entityId(kind, row)) && typeof row[rating] === 'number')
      .sort((left, right) => Number(right[rating]) - Number(left[rating]))
      .map((row) => entityId(kind, row))
  }, [kind, pairs, rating, rows, subject])

  if (query.isError) return isMissing(query.error) ? null : <ErrorState error={query.error} title="Could not load the head-to-head chances" />
  if (pairs.length === 0) return null

  const labelOf = (id: string) => labels.get(id) ?? id
  const selected = picked ?? neighborId(ranked, subject) ?? pairs[0].otherId
  const pair = pairs.find((candidate) => candidate.otherId === selected)
  const options = [...pairs].sort((left, right) => labelOf(left.otherId).localeCompare(labelOf(right.otherId)))
  const unit = kind === 'teams' ? 'team' : 'quarterback'

  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle className="text-base">Head to head</CardTitle>
        <CardDescription>
          How often this {unit} was rated above the one you pick across the same 1,000 redraws of the {season} season, counting only redraws with both. Both are re-rated in every redraw, so this answers the
          question more directly than two rank ranges.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Select value={selected} onValueChange={setPicked}>
          <SelectTrigger className="w-64 max-w-full" aria-label="Compare with">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map((option) => (
              <SelectItem key={option.otherId} value={option.otherId}>
                {labelOf(option.otherId)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {pair ? <p className="text-sm">{describeRatingPair(kind, labelOf(subject), labelOf(pair.otherId), pair)}</p> : null}
        <Link to={headToHeadPath(kind, season, subject, selected)} className="text-sm font-medium text-primary underline-offset-4 hover:underline">
          Compare side by side
        </Link>
      </CardContent>
    </Card>
  )
}
