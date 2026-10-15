import { useMemo } from 'react'
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { isMissing, useEntitySection } from '@/api/queries'
import type { EntityKind } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

import { AXIS_TICK, TOOLTIP_STYLE } from './chartStyle'
import { describeRankHistory, parseRankHistory, rankAxisMax, rankBandData, type RankHistoryPoint } from './rankHistory'
import { ordinal, rankTicks } from './rankRanges'

function rangeText(low: number | null, high: number | null): string {
  return low === null || high === null ? 'n/a' : `${ordinal(low)}–${ordinal(high)}`
}

function medianText(point: RankHistoryPoint): string {
  return point.median === null ? 'n/a' : ordinal(point.median)
}

const WAIT_NOTE =
  'It starts once every team has played three games: with fewer, a redraw can only repeat or drop a team’s games, never change their results, so the bands would look more certain than they are.'

/**
 * The rank range week by week for a season in progress: the median rank as a line with the middle
 * 50% and 95% of redraws as bands, rank 1 at the top. Hidden for a season without weekly ranges.
 */
export function RankHistoryCard({ kind, season, entityId, count = 0 }: { kind: EntityKind; season: number; entityId: string; count?: number }) {
  const query = useEntitySection('rank-history', kind, season, entityId)
  const points = useMemo(() => (query.data ? parseRankHistory(kind, query.data) : []), [kind, query.data])

  if (query.isError) return isMissing(query.error) ? null : <ErrorState error={query.error} title="Could not load the rank by week" />
  if (!query.data) return null
  if (points.length === 0) {
    return (
      <Card className="gap-2">
        <CardHeader>
          <CardTitle className="text-base">Rank by week</CardTitle>
          <CardDescription>The weekly rank chart starts once every team has played three games: with fewer, the redraws would look more certain than they are.</CardDescription>
        </CardHeader>
      </Card>
    )
  }
  const byWeek = new Map(points.map((point) => [point.week, point]))
  const axisMax = Math.max(count, rankAxisMax(points))
  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle className="text-base">Rank by week</CardTitle>
        <CardDescription>
          The rank as the {season} season went on, rank 1 at the top, from 1,000 redraws of the games played through each week. Line: median rank; darker band: middle 50%; lighter band: middle 95%. {WAIT_NOTE}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <p className="text-sm">{describeRankHistory(points)}</p>
        <div className="h-56 w-full" aria-hidden>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rankBandData(points)} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="week" tickLine={false} tick={AXIS_TICK} stroke="var(--border)" />
              <YAxis
                reversed
                domain={[1, axisMax]}
                ticks={rankTicks(axisMax, axisMax > 20 ? 8 : 5)}
                interval={0}
                allowDecimals={false}
                tickLine={false}
                width={44}
                tick={AXIS_TICK}
                stroke="var(--border)"
                tickFormatter={(value: number) => ordinal(value)}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                content={({ active, label }) => {
                  const point = typeof label === 'number' ? byWeek.get(label) : undefined
                  if (!active || !point) return null
                  return (
                    <div style={TOOLTIP_STYLE} className="px-2.5 py-1.5">
                      <div className="font-medium">Week {point.week}</div>
                      <div>Median rank {medianText(point)}</div>
                      <div>Middle 50% {rangeText(point.low50, point.high50)}</div>
                      <div>Middle 95% {rangeText(point.low95, point.high95)}</div>
                    </div>
                  )
                }}
              />
              <Area dataKey="band95" stroke="none" fill="var(--chart-1)" fillOpacity={0.15} isAnimationActive={false} />
              <Area dataKey="band50" stroke="none" fill="var(--chart-1)" fillOpacity={0.35} isAnimationActive={false} />
              <Line dataKey="median" stroke="var(--chart-1)" strokeWidth={2} dot={{ r: 4, fill: 'var(--chart-1)' }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <table className="sr-only">
          <caption>Rank by week</caption>
          <thead>
            <tr>
              <th scope="col">Week</th>
              <th scope="col">Median rank</th>
              <th scope="col">Middle 50%</th>
              <th scope="col">Middle 95%</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point) => (
              <tr key={point.week}>
                <th scope="row">{point.week}</th>
                <td>{medianText(point)}</td>
                <td>{rangeText(point.low50, point.high50)}</td>
                <td>{rangeText(point.low95, point.high95)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}
