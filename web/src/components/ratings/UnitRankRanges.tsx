import { RankIntervalTrack } from './RankInterval'
import { rankRangeHeadline, type UnitRankRange } from './rankRanges'

/** A team's offense, defense and special-teams rank ranges, each on a track of `count` ranks. */
export function UnitRankRanges({ ranges, count }: { ranges: UnitRankRange[]; count: number }) {
  return (
    <div className="flex flex-col gap-1.5">
      <table className="w-full text-sm">
        <caption className="sr-only">Rank range by unit</caption>
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th scope="col" className="py-1.5 pr-3 font-medium">
              Unit
            </th>
            <th scope="col" className="py-1.5 pr-3 font-medium">
              Rank range
            </th>
            <th scope="col" className="w-1/4 py-1.5 font-medium">
              <span className="sr-only">Interval</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {ranges.map((range) => (
            <tr key={range.unit} className="border-b last:border-0">
              <th scope="row" className="py-1.5 pr-3 text-left font-medium">
                {range.label}
              </th>
              <td className="tabular py-1.5 pr-3">{rankRangeHeadline(range)}</td>
              <td className="py-1.5">
                <RankIntervalTrack range={range} count={count} size="mini" showPublished />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-muted-foreground">Offense, defense and special teams are each ranked on their own in every redraw, so their ranges need not add up to the overall one.</p>
    </div>
  )
}
