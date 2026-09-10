import { useRegistry } from '@/api/queries'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

export function GlossaryPage() {
  const registry = useRegistry()
  return (
    <>
      <PageHeader title="Glossary" description="Every column the app can show, grouped by where it appears. The same text backs the header tooltips." />
      {registry.isLoading ? <Skeleton className="h-64 w-full" /> : null}
      {registry.isError ? <ErrorState error={registry.error} /> : null}
      {registry.data ? (
        <div className="columns-1 gap-6 md:columns-2 xl:columns-3">
          {Object.entries(registry.data.groups).map(([group, keys]) => (
            <section key={group} className="mb-6 break-inside-avoid rounded-xl border bg-card p-4">
              <h2 className="mb-3 text-sm font-semibold">{group}</h2>
              <dl className="space-y-2.5">
                {keys.map((key) => {
                  const meta = registry.data.columns[key]
                  return (
                    <div key={key}>
                      <dt className="flex flex-wrap items-center gap-1.5 text-sm font-medium">
                        {meta.label}
                        <code className="rounded bg-muted px-1 text-[10px] text-muted-foreground">{key}</code>
                        {meta.polarity !== 'neutral' ? <Badge variant="outline" className="text-[10px]">{meta.polarity} is better</Badge> : null}
                        {!meta.actionable ? <Badge variant="outline" className="text-[10px]">informational</Badge> : null}
                      </dt>
                      <dd className="text-xs text-muted-foreground">{meta.description}</dd>
                    </div>
                  )
                })}
              </dl>
            </section>
          ))}
        </div>
      ) : null}
    </>
  )
}
