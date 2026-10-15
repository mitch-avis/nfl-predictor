import { X } from 'lucide-react'
import { Link } from 'react-router'

import { useEntitySection } from '@/api/queries'
import type { EntityKind, MetricTable } from '@/api/types'
import { Button } from '@/components/ui/button'
import { columnStats, heatStyle } from '@/utils/cells'

import { entityLabel, entityPath, findEntityRow, headToHeadPath } from './entities'
import { decimalsFor, formatMetric } from './format'
import { describeRatingPair, parseRatingPairs } from './ratingPairs'
import { SectionCard } from './SectionCard'

function PairSentence({ kind, season, first, second, labels }: { kind: EntityKind; season: number; first: string; second: string; labels: [string, string] }) {
  const query = useEntitySection('head-to-head', kind, season, first)
  const pair = query.data ? parseRatingPairs(kind, query.data).find((candidate) => candidate.otherId === second) : undefined
  if (!pair) return null
  return (
    <p className="text-sm">
      {describeRatingPair(kind, labels[0], labels[1], pair)}{' '}
      <Link to={headToHeadPath(kind, season, first, second)} className="font-medium text-primary underline-offset-4 hover:underline">
        Head to head
      </Link>
    </p>
  )
}

/**
 * The picked teams or quarterbacks side by side: one column each, one row per column of the
 * current view, shaded against the whole season so two picks are not painted as each other's
 * extremes. `table` is the view's table, so the values are the ones the table above shows (season
 * totals in the totals view). Two picks add their head-to-head chance.
 */
export function ComparisonPanel({
  kind,
  season,
  table,
  teamNames,
  ids,
  columns,
  onRemove,
}: {
  kind: EntityKind
  season: number
  table: MetricTable
  teamNames: Record<string, string>
  ids: string[]
  columns: string[]
  onRemove: (id: string) => void
}) {
  const { rows, column_metadata: metadata } = table
  const picked = ids.flatMap((id) => {
    const row = findEntityRow(kind, rows, id)
    return row ? [{ id, row, label: entityLabel(kind, row, teamNames) }] : []
  })
  if (picked.length === 0) return null
  const stats = columnStats(rows, columns)
  const [first, second] = picked
  return (
    <SectionCard title="Comparison" description={`${picked.length} picked, side by side on the view’s columns. The page address keeps the picks, so the comparison can be shared.`}>
      {picked.length === 2 && first && second ? <PairSentence kind={kind} season={season} first={first.id} second={second.id} labels={[first.label, second.label]} /> : null}
      <div className="max-h-[70vh] overflow-auto rounded-xl border">
        <table className="w-full min-w-max text-sm">
          <caption className="sr-only">Comparison</caption>
          <thead className="sticky top-0 z-10 bg-card">
            <tr>
              <th scope="col" className="sticky left-0 z-20 border-b bg-card px-3 py-2 text-left font-medium text-muted-foreground">
                Metric
              </th>
              {picked.map((entity) => (
                <th key={entity.id} scope="col" className="min-w-36 border-b px-3 py-2 text-right font-medium">
                  <span className="inline-flex items-center gap-1">
                    <Link to={entityPath(kind, entity.id, season)} className="text-primary underline-offset-4 hover:underline">
                      {entity.label}
                    </Link>
                    <Button type="button" variant="ghost" size="icon-xs" aria-label={`Remove ${entity.label} from the comparison`} onClick={() => onRemove(entity.id)}>
                      <X />
                    </Button>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {columns.map((column) => {
              const meta = metadata[column]
              const decimals = decimalsFor(meta, rows, column)
              return (
                <tr key={column} className="border-b last:border-0">
                  <th scope="row" className="sticky left-0 z-10 bg-card px-3 py-1.5 text-left font-medium whitespace-nowrap">
                    {meta?.label ?? column}
                  </th>
                  {picked.map((entity) => (
                    <td key={entity.id} className="tabular px-3 py-1.5 text-right" style={meta ? heatStyle(meta, entity.row[column] ?? null, stats[column]) : undefined}>
                      {formatMetric(meta, entity.row[column], decimals)}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </SectionCard>
  )
}
