import { useMemo } from 'react'
import { Link } from 'react-router'

import type { EntityKind, MetricTable } from '@/api/types'
import { DataTable, type CellRenderer } from '@/components/table/DataTable'

import { withDecimals } from './decimals'
import { entityPath } from './entities'

const RESULT_LETTERS: Record<number, string> = { 1: 'W', 0: 'L', 0.5: 'T' }

/**
 * One row per game: the week and opponent (home `vs`, away `@`, linked to the opponent's page for
 * the season), the score and result, then the columns of the current view.
 */
export function GameLogTable({ kind, season, logs, columns }: { kind: EntityKind; season: number; logs: MetricTable; columns: string[] }) {
  const table = useMemo(() => withDecimals(logs, columns), [columns, logs])
  const renderers = useMemo<Record<string, CellRenderer>>(
    () => ({
      opponent_team: (value, row) =>
        typeof value === 'string' ? (
          <Link to={entityPath('teams', value, season)} className="font-medium text-primary underline-offset-4 hover:underline">
            {row.is_home === false ? '@' : 'vs'} {value}
          </Link>
        ) : (
          '—'
        ),
      win_value: (value) => (typeof value === 'number' ? (RESULT_LETTERS[value] ?? String(value)) : '—'),
    }),
    [season],
  )
  return (
    <DataTable
      table={table}
      columns={columns}
      defaultSort={{ key: 'week' }}
      renderers={renderers}
      rowKey={(row, index) => `${String(row.game_id ?? index)}-${kind}`}
      dense
      emptyMessage="No games this season."
    />
  )
}
