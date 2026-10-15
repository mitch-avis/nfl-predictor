import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { AXIS_TICK, TOOLTIP_STYLE } from './chartStyle'
import { formatChance, ordinal, rankHistogram, type RankRange } from './rankRanges'

/**
 * P(rank = k) across the redraws: one column per rank, the published rank in the full series color
 * and the others lighter. A hidden table carries the same numbers for screen readers.
 */
export function RankHistogram({ range }: { range: RankRange }) {
  const bars = rankHistogram(range)
  if (bars.length === 0) return null
  // Plot whole percents so the axis picks clean ticks (0, 4, 8, ...).
  const points = bars.map((bar) => ({ ...bar, percent: bar.probability * 100 }))
  return (
    <div className="flex flex-col gap-2">
      <div className="h-44 w-full" aria-hidden>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={points} margin={{ top: 8, right: 8, bottom: 4, left: 4 }} barCategoryGap={2}>
            <CartesianGrid vertical={false} stroke="var(--border)" />
            <XAxis dataKey="rank" tickLine={false} interval="preserveStartEnd" tick={AXIS_TICK} stroke="var(--border)" />
            <YAxis tickLine={false} width={40} tick={AXIS_TICK} stroke="var(--border)" allowDecimals={false} tickFormatter={(value: number) => `${value}%`} />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              cursor={{ fill: 'var(--muted)', fillOpacity: 0.6 }}
              formatter={(_value, _name, item) => [formatChance((item.payload as { probability: number }).probability), 'Share of redraws']}
              labelFormatter={(label) => {
                const rank = Number(label)
                return rank === range.publishedRank ? `Ranked ${ordinal(rank)} (published rank)` : `Ranked ${ordinal(rank)}`
              }}
            />
            <Bar dataKey="percent" maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false}>
              {points.map((bar) => (
                <Cell key={bar.rank} fill="var(--chart-1)" fillOpacity={bar.published ? 1 : 0.45} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">
        How often each rank came up across the redraws. Rank 1 is at the left; the full-color column is the published rank ({ordinal(range.publishedRank)}).
      </p>
      <table className="sr-only">
        <caption>Chance of each rank</caption>
        <thead>
          <tr>
            <th scope="col">Rank</th>
            <th scope="col">Chance</th>
          </tr>
        </thead>
        <tbody>
          {bars.map((bar) => (
            <tr key={bar.rank}>
              <th scope="row">{ordinal(bar.rank)}</th>
              <td>{formatChance(bar.probability)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
