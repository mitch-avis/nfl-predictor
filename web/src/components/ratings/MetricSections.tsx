import { useId } from 'react'

import type { MetricCategory, MetricColumnMeta, Row } from '@/api/types'
import { InfoTooltip } from '@/components/common/InfoTooltip'

import { decimalsFor, formatMetric } from './format'

interface Section {
  title: string
  rank: number
  columns: string[]
}

/** Group columns by category and subcategory, in the order of `categories`. */
function sectionsOf(columns: string[], metadata: Record<string, MetricColumnMeta>, categories: MetricCategory[]): Section[] {
  const sections = new Map<string, Section>()
  for (const column of columns) {
    const meta = metadata[column]
    const category = meta?.category ?? 'Other'
    const categoryIndex = categories.findIndex((candidate) => candidate.name === category)
    const subcategories = categoryIndex >= 0 ? categories[categoryIndex].subcategories : []
    const title = meta?.subcategory ? `${category} — ${meta.subcategory}` : category
    const rank = (categoryIndex >= 0 ? categoryIndex : categories.length) * 1000 + (meta?.subcategory ? subcategories.indexOf(meta.subcategory) + 1 : 0)
    const section = sections.get(title) ?? { title, rank, columns: [] }
    section.columns.push(column)
    sections.set(title, section)
  }
  return [...sections.values()].sort((left, right) => left.rank - right.rank)
}

function SectionBlock({ section, row, rows, metadata }: { section: Section; row: Row; rows: Row[]; metadata: Record<string, MetricColumnMeta> }) {
  const titleId = useId()
  return (
    <section aria-labelledby={titleId} className="flex flex-col gap-2">
      <h3 id={titleId} className="text-sm font-semibold">
        {section.title}
      </h3>
      <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
        {section.columns.map((column) => {
          const meta = metadata[column]
          const label = meta?.label ?? column
          return (
            <div key={column} className="rounded-lg border bg-muted/30 px-3 py-2">
              <dt className="flex items-center gap-1 text-xs text-muted-foreground">
                {label}
                {meta?.description ? <InfoTooltip content={meta.description} label={`About ${label}`} /> : null}
              </dt>
              <dd className="tabular text-lg font-semibold">{formatMetric(meta, row[column], decimalsFor(meta, rows, column))}</dd>
            </div>
          )
        })}
      </dl>
    </section>
  )
}

/** One team's or quarterback's season values for the current view, grouped by category. */
export function MetricSections({ row, rows, columns, metadata, categories }: { row: Row; rows: Row[]; columns: string[]; metadata: Record<string, MetricColumnMeta>; categories: MetricCategory[] }) {
  if (columns.length === 0) return <p className="text-sm text-muted-foreground">No stats match these filters.</p>
  return (
    <div className="flex flex-col gap-5">
      {sectionsOf(columns, metadata, categories).map((section) => (
        <SectionBlock key={section.title} section={section} row={row} rows={rows} metadata={metadata} />
      ))}
    </div>
  )
}
