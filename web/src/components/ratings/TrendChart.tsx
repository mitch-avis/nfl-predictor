import { useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { MetricColumnMeta, Row } from '@/api/types'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

import { AXIS_TICK, TOOLTIP_STYLE } from './chartStyle'
import { columnDecimals } from './decimals'
import { decimalsFor, formatMetric } from './format'
import { buildTrendPoints, meanReference, niceTicks, trendColumns, type TrendReference } from './trend'

/**
 * One metric by week, straight segments between games, with a picker over the numeric `columns`
 * (any `preferred` one with values first). `reference` is the dashed line: `'mean'` draws the mean
 * of the points, a `TrendReference` a fixed value, `null` none. A hidden table carries the points.
 */
export function TrendChart({
  rows,
  columns,
  metadata,
  preferred = [],
  reference = 'mean',
}: {
  rows: Row[]
  columns: string[]
  metadata: Record<string, MetricColumnMeta>
  preferred?: readonly string[]
  reference?: 'mean' | TrendReference | null
}) {
  const options = useMemo(() => trendColumns(rows, columns, preferred), [columns, preferred, rows])
  const [picked, setPicked] = useState<string | null>(null)
  const column = picked !== null && options.includes(picked) ? picked : options[0]
  const points = useMemo(() => (column ? buildTrendPoints(rows, column) : []), [column, rows])
  if (!column || points.length < 2) return null

  const meta = metadata[column]
  const label = meta?.label ?? column
  const decimals = decimalsFor(meta, rows, column)
  const format = (value: number) => formatMetric(meta, value, decimals)
  const line = reference === 'mean' ? meanReference(points, format) : reference
  const values = [...points.map((point) => point.value), ...(line ? [line.value] : [])]
  const ticks = niceTicks(Math.min(...values), Math.max(...values))
  const tickDecimals = columnDecimals(ticks, meta?.shape, meta?.percent ?? false)
  const withOpponents = points.some((point) => point.opponent !== '')

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-medium">{label} by week</div>
        {options.length > 1 ? (
          <Select value={column} onValueChange={setPicked}>
            <SelectTrigger size="sm" className="w-56" aria-label="Metric to chart">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {options.map((option) => (
                <SelectItem key={option} value={option}>
                  {metadata[option]?.label ?? option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : null}
      </div>
      <div className="h-56 w-full" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis dataKey="week" tickLine={false} tick={AXIS_TICK} stroke="var(--border)" />
            <YAxis
              tickLine={false}
              width={56}
              tick={AXIS_TICK}
              stroke="var(--border)"
              domain={[ticks[0], ticks.at(-1) ?? ticks[0]]}
              ticks={ticks}
              tickFormatter={(value: number) => formatMetric(meta, value, tickDecimals)}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value) => [typeof value === 'number' ? format(value) : String(value), label]}
              labelFormatter={(week, payload) => {
                const opponent = (payload?.[0]?.payload as { opponent?: string } | undefined)?.opponent
                return opponent ? `Week ${week} vs ${opponent}` : `Week ${week}`
              }}
            />
            {line ? <ReferenceLine y={line.value} stroke="var(--muted-foreground)" strokeDasharray="4 4" /> : null}
            <Line type="linear" dataKey="value" stroke="var(--chart-1)" strokeWidth={2} dot={{ r: 3, fill: 'var(--chart-1)' }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {line ? <p className="text-xs text-muted-foreground">{line.caption}</p> : null}
      <table className="sr-only">
        <caption>{label}, week by week</caption>
        <thead>
          <tr>
            <th scope="col">Week</th>
            {withOpponents ? <th scope="col">Opponent</th> : null}
            <th scope="col">{label}</th>
          </tr>
        </thead>
        <tbody>
          {points.map((point) => (
            <tr key={`${point.week}-${point.opponent}`}>
              <td>{point.week}</td>
              {withOpponents ? <td>{point.opponent}</td> : null}
              <td>{format(point.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
