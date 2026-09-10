import { Link } from 'react-router'

import { useRuns } from '@/api/queries'
import { NAV_ITEMS } from '@/app/nav'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { StatTile } from '@/components/common/StatTile'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { formatNumber, formatPercent, formatRelative, seasonWeekLabel } from '@/utils/format'

export function DashboardPage() {
  const runs = useRuns('all')
  const active = runs.data?.runs.find((run) => run.is_active) ?? null

  return (
    <>
      <PageHeader title="Overview" description="Where the project stands right now: the active run, its week, and how it scored on its holdout." />
      {runs.isLoading ? <Skeleton className="h-28 w-full" /> : null}
      {runs.isError ? <ErrorState error={runs.error} /> : null}
      {runs.data ? (
        <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
          <StatTile label="Active run" value={<span className="truncate text-base">{active?.run_id ?? 'None'}</span>} footnote={active ? formatRelative(active.created_at) : 'Pin one on the Runs page'} />
          <StatTile label="Week" value={active ? seasonWeekLabel(active.season, active.week) : '—'} hint="The season and week the active run predicted." />
          <StatTile label="Holdout Brier" value={formatNumber(active?.holdout?.brier, 4)} hint="Mean squared error of the win probabilities on held-out games. Lower is better; 0.25 is a coin flip." />
          <StatTile label="Holdout accuracy" value={formatPercent(active?.holdout?.winner_accuracy ?? active?.holdout?.pick_accuracy, 1)} hint="Share of held-out games where the favored side won." />
        </div>
      ) : null}
      <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {NAV_ITEMS.filter((item) => item.to !== '/').map((item) => (
          <Link key={item.to} to={item.to} className="group focus-visible:outline-2">
            <Card className="h-full transition-colors group-hover:bg-accent/40">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <item.icon className="size-4 text-primary" /> {item.label}
                </CardTitle>
                <CardDescription>{item.description}</CardDescription>
              </CardHeader>
              <CardContent className="text-xs text-muted-foreground">
                {item.phase !== undefined ? `Coming in phase ${item.phase}` : 'Available now'}
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </>
  )
}
