import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { ColumnMeta, TablePayload } from '@/api/types'
import { TooltipProvider } from '@/components/ui/tooltip'

import { DataTable, type CellRenderer } from './DataTable'

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
