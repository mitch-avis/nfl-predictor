import {
  createSortedRowModel,
  flexRender,
  metaHelper,
  rowSortingFeature,
  tableFeatures,
  useTable,
  type CellContext,
  type ColumnDef,
  type HeaderContext,
  type Row as TableRow,
  type SortingState,
  type Table,
} from '@tanstack/react-table'
import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react'
import { useMemo, useState, type CSSProperties, type ReactNode } from 'react'

import type { ColumnMeta, Row, TablePayload } from '@/api/types'
import { InfoTooltip } from '@/components/common/InfoTooltip'
import { cn } from '@/utils/cn'
import { columnStats, formatCell, heatStyle, isNumericKind } from '@/utils/cells'

export type CellRenderer = (value: Row[string], row: Row, meta: ColumnMeta) => ReactNode

export interface DataTableProps {
  table: TablePayload
  /** Column keys to show, in order. Defaults to `table.visible_columns`. */
  columns?: string[]
  defaultSort?: { key: string; desc?: boolean }
  renderers?: Record<string, CellRenderer>
  rowKey?: (row: Row, index: number) => string
  rowClassName?: (row: Row) => string | undefined
  dense?: boolean
  heat?: boolean
  maxHeight?: string
  emptyMessage?: string
}

const STICKY_WIDTH = 88

/** What the cells read on every render, passed through the table's `meta` option. */
interface DataTableMeta {
  columnMetadata: Record<string, ColumnMeta>
  renderers?: Record<string, CellRenderer>
}

/** The table features: sorting only, with the cells' metadata typed through `tableMeta`. */
const features = tableFeatures({
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  tableMeta: metaHelper<DataTableMeta>(),
})

type Features = typeof features

function tableMeta(table: Table<Features, Row>): DataTableMeta {
  return table.options.meta as DataTableMeta
}

/**
 * A column's header: its label and description. The header and cell components live at module
 * level and read the current metadata and renderers from the table, so their identity never
 * changes and React updates cells in place instead of remounting them (which dropped focus).
 */
function HeaderLabel({ column, table }: HeaderContext<Features, Row, unknown>) {
  const meta = tableMeta(table).columnMetadata[column.id]
  return (
    <span className={cn('inline-flex items-center gap-1', isNumericKind(meta) && 'justify-end w-full')}>
      <span>{meta?.label ?? column.id}</span>
      {meta?.description ? <InfoTooltip content={meta.description} label={`About ${meta.label}`} /> : null}
    </span>
  )
}

/**
 * A cell: the page's renderer for its column when there is one, else the formatted value. It reads
 * the payload value rather than the accessor's, so a renderer still sees `null` as `null`.
 */
function BodyCell({ column, row, table }: CellContext<Features, Row, unknown>) {
  const { columnMetadata, renderers } = tableMeta(table)
  const meta = columnMetadata[column.id]
  const value = row.original[column.id]
  const custom = renderers?.[column.id]
  return custom ? custom(value, row.original, meta) : formatCell(meta, value)
}

/**
 * Sort numbers as numbers and anything else as text. Missing values never reach it: the accessor
 * reads `null` as `undefined`, which `sortUndefined: 'last'` keeps last in both directions (the
 * table flips this comparator's result on a descending sort, so it cannot place them itself).
 */
function compareCells(left: TableRow<Features, Row>, right: TableRow<Features, Row>, columnId: string): number {
  const av = left.original[columnId]
  const bv = right.original[columnId]
  if (av === null || av === undefined || bv === null || bv === undefined) return 0
  if (typeof av === 'number' && typeof bv === 'number') return av - bv
  return String(av).localeCompare(String(bv))
}

/** A sortable table driven entirely by the registry metadata in the payload. */
export function DataTable({
  table,
  columns,
  defaultSort,
  renderers,
  rowKey,
  rowClassName,
  dense = false,
  heat = true,
  maxHeight,
  emptyMessage = 'No rows.',
}: DataTableProps) {
  const keys = useMemo(() => (columns ?? table.visible_columns).filter((k) => k in table.column_metadata), [columns, table])
  const stats = useMemo(() => columnStats(table.rows, keys), [table.rows, keys])
  const [sorting, setSorting] = useState<SortingState>(defaultSort ? [{ id: defaultSort.key, desc: !!defaultSort.desc }] : [])

  const stickyKeys = keys.filter((k) => table.column_metadata[k]?.sticky)
  const stickyOffsets: Record<string, number> = {}
  stickyKeys.forEach((k, i) => {
    stickyOffsets[k] = i * STICKY_WIDTH
  })

  const columnDefs = useMemo<ColumnDef<Features, Row>[]>(
    () =>
      keys.map((key) => ({
        id: key,
        accessorFn: (row) => row[key] ?? undefined,
        header: HeaderLabel,
        cell: BodyCell,
        sortFn: compareCells,
        sortUndefined: 'last',
      })),
    [keys],
  )
  const meta: DataTableMeta = { columnMetadata: table.column_metadata, renderers }

  const instance = useTable({
    features,
    data: table.rows,
    columns: columnDefs,
    state: { sorting },
    onSortingChange: setSorting,
    meta,
  })

  if (table.rows.length === 0) {
    return <div className="rounded-xl border p-6 text-center text-sm text-muted-foreground">{emptyMessage}</div>
  }

  return (
    <div className="relative overflow-auto rounded-xl border bg-card" style={maxHeight ? { maxHeight } : undefined}>
      <table className={cn('w-max min-w-full border-separate border-spacing-0 text-sm', dense && 'text-xs')}>
        <thead className="sticky top-0 z-20 bg-card">
          {instance.getHeaderGroups().map((group) => (
            <tr key={group.id}>
              {group.headers.map((header) => {
                const meta = table.column_metadata[header.column.id]
                const sorted = header.column.getIsSorted()
                const sticky = header.column.id in stickyOffsets
                const primary = sorted && header.column.getSortIndex() === 0
                const ariaSort = primary ? (sorted === 'asc' ? 'ascending' : 'descending') : undefined
                const style: CSSProperties | undefined = sticky
                  ? { position: 'sticky', left: stickyOffsets[header.column.id], minWidth: STICKY_WIDTH, zIndex: 30 }
                  : undefined
                return (
                  <th
                    key={header.id}
                    style={style}
                    aria-sort={ariaSort}
                    className={cn(
                      'whitespace-nowrap border-b bg-card px-2.5 font-medium text-muted-foreground',
                      dense ? 'py-1.5' : 'py-2.5',
                      isNumericKind(meta) ? 'text-right' : 'text-left',
                      sticky && 'shadow-[inset_-1px_0_0_var(--border)]',
                    )}
                  >
                    <button
                      type="button"
                      className="inline-flex max-w-full items-center gap-1 hover:text-foreground"
                      onClick={header.column.getToggleSortingHandler()}
                      aria-label={`Sort by ${meta?.label ?? header.column.id}`}
                    >
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {sorted === 'asc' ? <ArrowUp className="size-3" /> : sorted === 'desc' ? <ArrowDown className="size-3" /> : <ArrowUpDown className="size-3 opacity-40" />}
                    </button>
                  </th>
                )
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {instance.getRowModel().rows.map((row, index) => (
            <tr key={rowKey ? rowKey(row.original, index) : row.id} className={cn('group/row hover:bg-accent/30', rowClassName?.(row.original))}>
              {row.getAllCells().map((cell) => {
                const key = cell.column.id
                const meta = table.column_metadata[key]
                const value = row.original[key]
                const sticky = key in stickyOffsets
                const style: CSSProperties = {
                  ...(sticky ? { position: 'sticky', left: stickyOffsets[key], minWidth: STICKY_WIDTH, zIndex: 10 } : {}),
                  ...(heat ? heatStyle(meta, value, stats[key]) : {}),
                }
                return (
                  <td
                    key={cell.id}
                    style={style}
                    className={cn(
                      'whitespace-nowrap border-b px-2.5 tabular',
                      dense ? 'py-1' : 'py-2',
                      isNumericKind(meta) ? 'text-right' : 'text-left',
                      sticky && 'bg-card font-medium shadow-[inset_-1px_0_0_var(--border)] group-hover/row:bg-accent/30',
                      meta?.actionable === false && 'text-muted-foreground',
                    )}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
