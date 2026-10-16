import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import type { ColumnMeta, Row, TablePayload } from '@/api/types'
import { TooltipProvider } from '@/components/ui/tooltip'

import { DataTable, type CellRenderer, type DataTableProps } from './DataTable'

function meta(key: string, overrides: Partial<ColumnMeta> = {}): ColumnMeta {
  return { key, label: key, description: '', group: 'g', kind: 'text', polarity: 'neutral', heatmap: false, decimals: null, actionable: true, sticky: false, ...overrides }
}

function payload(decimals: number | null): TablePayload {
  return {
    rows: [{ team: 'BUF', rating: 6.5 }],
    visible_columns: ['team', 'rating'],
    column_groups: {},
    column_metadata: { team: meta('team'), rating: meta('rating', { kind: 'float', decimals }) },
  }
}

function renderers(): Record<string, CellRenderer> {
  return { team: (value) => <input aria-label={`Pick ${String(value)}`} /> }
}

describe('DataTable', () => {
  it('keeps a cell mounted when the page passes new renderers and metadata', () => {
    // Arrange
    const view = render(
      <TooltipProvider>
        <DataTable table={payload(1)} renderers={renderers()} />
      </TooltipProvider>,
    )
    const input = screen.getByRole('textbox', { name: 'Pick BUF' })
    input.focus()

    // Act
    view.rerender(
      <TooltipProvider>
        <DataTable table={payload(2)} renderers={renderers()} />
      </TooltipProvider>,
    )

    // Assert
    expect(screen.getByRole('textbox', { name: 'Pick BUF' })).toBe(input)
    expect(input).toHaveFocus()
    expect(screen.getByText('6.50')).toBeInTheDocument()
  })
})

const SORT_ROWS: Row[] = [
  { team: 'KC', rating: 3 },
  { team: 'BUF', rating: null },
  { team: 'DET', rating: 7 },
  { team: 'ARI', rating: -2 },
]

function sortPayload(): TablePayload {
  return {
    rows: SORT_ROWS,
    visible_columns: ['team', 'rating'],
    column_groups: {},
    column_metadata: { team: meta('team', { label: 'Team' }), rating: meta('rating', { label: 'Rating', kind: 'float', decimals: 1 }) },
  }
}

function renderTable(props: DataTableProps) {
  return render(
    <TooltipProvider>
      <DataTable {...props} />
    </TooltipProvider>,
  )
}

/** The text of one column's body cells, top to bottom. */
function columnText(index: number): string[] {
  return screen
    .getAllByRole('row')
    .slice(1)
    .map((row) => within(row).getAllByRole('cell')[index].textContent ?? '')
}

/** The class list of the sort icon in a column's header button. */
function sortIcon(label: string): string {
  const button = screen.getByRole('button', { name: `Sort by ${label}` })
  return button.querySelector('svg')?.getAttribute('class') ?? ''
}

describe('DataTable rendering', () => {
  it('renders the header labels and formatted cells in the visible column order', () => {
    // Act
    renderTable({ table: sortPayload() })

    // Assert
    const headers = screen.getAllByRole('columnheader').map((th) => th.textContent)
    expect(headers).toEqual(['Team', 'Rating'])
    expect(columnText(0)).toEqual(['KC', 'BUF', 'DET', 'ARI'])
    expect(columnText(1)).toEqual(['3.0', '—', '7.0', '-2.0'])
  })

  it('shows only the requested columns, in order, dropping keys without metadata', () => {
    // Act
    renderTable({ table: sortPayload(), columns: ['rating', 'missing', 'team'] })

    // Assert
    const headers = screen.getAllByRole('columnheader').map((th) => th.textContent)
    expect(headers).toEqual(['Rating', 'Team'])
  })

  it('shows the default empty message when there are no rows', () => {
    // Arrange
    const table = { ...sortPayload(), rows: [] }

    // Act
    renderTable({ table })

    // Assert
    expect(screen.getByText('No rows.')).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('shows a custom empty message', () => {
    // Arrange
    const table = { ...sortPayload(), rows: [] }

    // Act
    renderTable({ table, emptyMessage: 'Nothing yet.' })

    // Assert
    expect(screen.getByText('Nothing yet.')).toBeInTheDocument()
  })

  it('passes the value, row and metadata to a column renderer', () => {
    // Arrange
    const seen: unknown[] = []
    const renderers: Record<string, CellRenderer> = {
      team: (value, row, columnMeta) => {
        seen.push([value, row.rating, columnMeta.label])
        return `<${String(value)}>`
      },
    }

    // Act
    renderTable({ table: sortPayload(), renderers })

    // Assert
    expect(columnText(0)).toEqual(['<KC>', '<BUF>', '<DET>', '<ARI>'])
    expect(seen).toContainEqual(['KC', 3, 'Team'])
  })

  it('calls rowKey with each row and its index', () => {
    // Arrange
    const calls: [string, number][] = []
    const rowKey = (row: Row, index: number) => {
      calls.push([String(row.team), index])
      return String(row.team)
    }

    // Act
    renderTable({ table: sortPayload(), rowKey })

    // Assert
    expect(calls.slice(0, 4)).toEqual([
      ['KC', 0],
      ['BUF', 1],
      ['DET', 2],
      ['ARI', 3],
    ])
  })

  it('applies rowClassName to each body row', () => {
    // Arrange
    const rowClassName = (row: Row) => (row.team === 'DET' ? 'highlight-row' : undefined)

    // Act
    renderTable({ table: sortPayload(), rowClassName })

    // Assert
    const highlighted = screen.getAllByRole('row').filter((row) => row.classList.contains('highlight-row'))
    expect(highlighted).toHaveLength(1)
    expect(within(highlighted[0]).getAllByRole('cell')[0]).toHaveTextContent('DET')
  })

  it('pins sticky columns at stacked left offsets', () => {
    // Arrange
    const table = sortPayload()
    table.column_metadata = {
      team: meta('team', { sticky: true }),
      rating: meta('rating', { kind: 'float', decimals: 1, sticky: true }),
    }

    // Act
    renderTable({ table })

    // Assert
    const [teamHeader, ratingHeader] = screen.getAllByRole('columnheader')
    expect(teamHeader).toHaveStyle({ position: 'sticky', left: '0px' })
    expect(ratingHeader).toHaveStyle({ position: 'sticky', left: '88px' })
    expect(within(screen.getAllByRole('row')[1]).getAllByRole('cell')[1]).toHaveStyle({ position: 'sticky', left: '88px' })
  })

  it('shades heatmap cells by default', () => {
    // Arrange
    const table = sortPayload()
    table.column_metadata.rating = meta('rating', { kind: 'float', decimals: 1, heatmap: true, polarity: 'higher' })

    // Act
    renderTable({ table })

    // Assert
    const detCell = within(screen.getAllByRole('row')[3]).getAllByRole('cell')[1]
    expect(detCell.style.backgroundColor).not.toBe('')
  })

  it('leaves heatmap cells unshaded when heat is off', () => {
    // Arrange
    const table = sortPayload()
    table.column_metadata.rating = meta('rating', { kind: 'float', decimals: 1, heatmap: true, polarity: 'higher' })

    // Act
    renderTable({ table, heat: false })

    // Assert
    const detCell = within(screen.getAllByRole('row')[3]).getAllByRole('cell')[1]
    expect(detCell.style.backgroundColor).toBe('')
  })

  it('shows an info button for a column with a description', () => {
    // Arrange
    const table = sortPayload()
    table.column_metadata.rating = meta('rating', { label: 'Rating', kind: 'float', description: 'Points above average' })

    // Act
    renderTable({ table })

    // Assert
    expect(screen.getByRole('button', { name: 'About Rating' })).toBeInTheDocument()
  })
})

describe('DataTable sorting', () => {
  it('starts unsorted, in payload order, with neutral sort icons', () => {
    // Act
    renderTable({ table: sortPayload() })

    // Assert
    expect(columnText(0)).toEqual(['KC', 'BUF', 'DET', 'ARI'])
    expect(sortIcon('Team')).toContain('lucide-arrow-up-down')
  })

  it('applies the default sort ascending', () => {
    // Act
    renderTable({ table: sortPayload(), defaultSort: { key: 'team' } })

    // Assert
    expect(columnText(0)).toEqual(['ARI', 'BUF', 'DET', 'KC'])
    expect(sortIcon('Team')).toMatch(/lucide-arrow-up(?!-down)/)
  })

  it('applies a descending default sort', () => {
    // Act
    renderTable({ table: sortPayload(), defaultSort: { key: 'team', desc: true } })

    // Assert
    expect(columnText(0)).toEqual(['KC', 'DET', 'BUF', 'ARI'])
    expect(sortIcon('Team')).toContain('lucide-arrow-down')
  })

  it('sorts a text column ascending on the first click', async () => {
    // Arrange
    const user = userEvent.setup()
    renderTable({ table: sortPayload() })

    // Act
    await user.click(screen.getByRole('button', { name: 'Sort by Team' }))

    // Assert
    expect(columnText(0)).toEqual(['ARI', 'BUF', 'DET', 'KC'])
    expect(sortIcon('Team')).toMatch(/lucide-arrow-up(?!-down)/)
  })

  it('sorts a text column ascending on the first click when its first row is null', async () => {
    // Arrange
    const user = userEvent.setup()
    const rows = [{ team: null, rating: 1 }, { team: 'KC', rating: 2 }, { team: 'ARI', rating: 3 }]
    renderTable({ table: { ...sortPayload(), rows } })

    // Act
    await user.click(screen.getByRole('button', { name: 'Sort by Team' }))

    // Assert
    expect(columnText(0)).toEqual(['ARI', 'KC', '—'])
  })

  it('sorts a text column descending on the second click', async () => {
    // Arrange
    const user = userEvent.setup()
    renderTable({ table: sortPayload() })
    const button = screen.getByRole('button', { name: 'Sort by Team' })
    await user.click(button)

    // Act
    await user.click(button)

    // Assert
    expect(columnText(0)).toEqual(['KC', 'DET', 'BUF', 'ARI'])
    expect(sortIcon('Team')).toContain('lucide-arrow-down')
  })

  it('clears the sort on the third click', async () => {
    // Arrange
    const user = userEvent.setup()
    renderTable({ table: sortPayload() })
    const button = screen.getByRole('button', { name: 'Sort by Team' })
    await user.click(button)
    await user.click(button)

    // Act
    await user.click(button)

    // Assert
    expect(columnText(0)).toEqual(['KC', 'BUF', 'DET', 'ARI'])
    expect(sortIcon('Team')).toContain('lucide-arrow-up-down')
  })

  it('sorts a numeric column descending on the first click, with null values first', async () => {
    // Arrange
    const user = userEvent.setup()
    renderTable({ table: sortPayload() })

    // Act
    await user.click(screen.getByRole('button', { name: 'Sort by Rating' }))

    // Assert
    expect(columnText(0)).toEqual(['BUF', 'DET', 'KC', 'ARI'])
    expect(sortIcon('Rating')).toContain('lucide-arrow-down')
  })

  it('sorts a numeric column ascending on the second click, with null values last', async () => {
    // Arrange
    const user = userEvent.setup()
    renderTable({ table: sortPayload() })
    const button = screen.getByRole('button', { name: 'Sort by Rating' })
    await user.click(button)

    // Act
    await user.click(button)

    // Assert
    expect(columnText(0)).toEqual(['ARI', 'KC', 'DET', 'BUF'])
  })

  it('replaces the sort when another header is clicked', async () => {
    // Arrange
    const user = userEvent.setup()
    renderTable({ table: sortPayload(), defaultSort: { key: 'rating' } })

    // Act
    await user.click(screen.getByRole('button', { name: 'Sort by Team' }))

    // Assert
    expect(columnText(0)).toEqual(['ARI', 'BUF', 'DET', 'KC'])
    expect(sortIcon('Rating')).toContain('lucide-arrow-up-down')
  })

  it('adds a secondary sort on shift-click', async () => {
    // Arrange
    const user = userEvent.setup()
    const table = sortPayload()
    table.rows = [
      { team: 'KC', rating: 3 },
      { team: 'BUF', rating: 7 },
      { team: 'ARI', rating: 3 },
    ]
    renderTable({ table, defaultSort: { key: 'rating' } })
    await user.keyboard('{Shift>}')

    // Act
    await user.click(screen.getByRole('button', { name: 'Sort by Team' }))

    // Assert
    expect(columnText(0)).toEqual(['ARI', 'KC', 'BUF'])
  })
})
