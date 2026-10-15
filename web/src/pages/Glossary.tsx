import { Search } from 'lucide-react'
import { useId, useMemo, useState } from 'react'

import { useMetricRegistry, useRegistry } from '@/api/queries'
import type { ColumnMeta, Registry } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { buildGlossary, filterGlossary, type GlossaryEntry, type GlossarySection } from '@/components/ratings/glossary'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useQueryParam } from '@/hooks/useQueryParam'

const RATINGS_TAB = 'ratings'

function matchesColumn(meta: ColumnMeta, needle: string): boolean {
  return [meta.key, meta.label, meta.description].some((text) => text.toLowerCase().includes(needle))
}

function ModelGroup({ group, keys, registry }: { group: string; keys: string[]; registry: Registry }) {
  const titleId = useId()
  return (
    <section aria-labelledby={titleId} className="mb-6 break-inside-avoid rounded-xl border bg-card p-4">
      <h2 id={titleId} className="mb-3 text-sm font-semibold">
        {group}
      </h2>
      <dl className="space-y-2.5">
        {keys.map((key) => {
          const meta = registry.columns[key]
          return (
            <div key={key}>
              <dt className="flex flex-wrap items-center gap-1.5 text-sm font-medium">
                {meta.label}
                <code className="rounded bg-muted px-1 text-[10px] text-muted-foreground">{key}</code>
                {meta.polarity !== 'neutral' ? (
                  <Badge variant="outline" className="text-[10px]">
                    {meta.polarity} is better
                  </Badge>
                ) : null}
                {!meta.actionable ? (
                  <Badge variant="outline" className="text-[10px]">
                    informational
                  </Badge>
                ) : null}
              </dt>
              <dd className="text-xs text-muted-foreground">{meta.description}</dd>
            </div>
          )
        })}
      </dl>
    </section>
  )
}

/** The prediction, betting, power-ranking and model columns, grouped by where they appear. */
function ModelColumns({ needle }: { needle: string }) {
  const registry = useRegistry()
  if (registry.isLoading) return <Skeleton className="h-64 w-full" />
  if (registry.isError) return <ErrorState error={registry.error} />
  if (!registry.data) return null
  const data = registry.data
  const groups = Object.entries(data.groups)
    .map(([group, keys]) => [group, needle === '' ? keys : keys.filter((key) => matchesColumn(data.columns[key], needle))] as const)
    .filter(([, keys]) => keys.length > 0)
  if (groups.length === 0) return <p className="text-sm text-muted-foreground">No model column matches the search.</p>
  return (
    <div className="columns-1 gap-6 md:columns-2 xl:columns-3">
      {groups.map(([group, keys]) => (
        <ModelGroup key={group} group={group} keys={keys} registry={data} />
      ))}
    </div>
  )
}

function GlossaryItem({ entry }: { entry: GlossaryEntry }) {
  return (
    <div className="rounded-lg border bg-muted/30 px-3 py-2">
      <dt className="font-medium">{entry.title}</dt>
      <dd className="text-sm text-muted-foreground">
        {entry.shownAs ? `Shown in the tables as ${entry.shownAs}. ` : ''}
        {entry.description}
      </dd>
      {entry.direction ? <dd className="mt-1 text-xs text-muted-foreground">{entry.direction}</dd> : null}
      {entry.formula ? <dd className="mt-1 text-xs text-muted-foreground">How it’s computed: {entry.formula}</dd> : null}
      {entry.sameAs ? <dd className="mt-1 text-xs text-muted-foreground">The same number as {entry.sameAs}.</dd> : null}
      {entry.since ? <dd className="mt-1 text-xs text-muted-foreground">Data from {entry.since} on.</dd> : null}
      {entry.note ? <dd className="mt-1 text-xs text-muted-foreground">{entry.note}</dd> : null}
    </div>
  )
}

function GlossaryCard({ section }: { section: GlossarySection }) {
  const titleId = useId()
  return (
    <section aria-labelledby={titleId} className="rounded-xl border bg-card p-4">
      <h2 id={titleId} className="text-sm font-semibold">
        {section.title}
      </h2>
      {section.description ? <p className="mt-0.5 text-sm text-muted-foreground">{section.description}</p> : null}
      <dl className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {section.entries.map((entry) => (
          <GlossaryItem key={entry.id} entry={entry} />
        ))}
      </dl>
    </section>
  )
}

/** The team and quarterback ratings and stats, from the metric registry. */
function RatingsMetrics({ query }: { query: string }) {
  const registry = useMetricRegistry()
  const sections = useMemo(() => (registry.data ? buildGlossary(registry.data) : []), [registry.data])
  const shown = useMemo(() => filterGlossary(sections, query), [query, sections])
  if (registry.isLoading) return <Skeleton className="h-64 w-full" />
  if (registry.isError) return <ErrorState error={registry.error} title="Could not load the metric definitions" />
  if (shown.length === 0) return <p className="text-sm text-muted-foreground">No team or quarterback stat matches the search.</p>
  return (
    <div className="flex flex-col gap-4">
      {shown.map((section) => (
        <GlossaryCard key={section.id} section={section} />
      ))}
    </div>
  )
}

export function GlossaryPage() {
  const [tab, setTab] = useQueryParam('tab')
  const [query, setQuery] = useState('')
  const active = tab === RATINGS_TAB ? RATINGS_TAB : 'model'
  return (
    <>
      <PageHeader
        title="Glossary"
        description="Every column the app shows, in plain words: the model’s predictions, betting and run columns, and the team and quarterback ratings and stats. The same text backs the header tooltips."
      />
      <div className="relative mb-4 max-w-md">
        <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
        <Input type="search" aria-label="Search the glossary" placeholder="Search the glossary" className="pl-8" value={query} onChange={(event) => setQuery(event.target.value)} />
      </div>
      <Tabs value={active} onValueChange={(value) => setTab(value === 'model' ? null : value)}>
        <TabsList>
          <TabsTrigger value="model">Predictions and model</TabsTrigger>
          <TabsTrigger value={RATINGS_TAB}>Teams and quarterbacks</TabsTrigger>
        </TabsList>
        <TabsContent value="model" className="mt-3">
          <ModelColumns needle={query.trim().toLowerCase()} />
        </TabsContent>
        <TabsContent value={RATINGS_TAB} className="mt-3">
          <RatingsMetrics query={query} />
        </TabsContent>
      </Tabs>
    </>
  )
}
