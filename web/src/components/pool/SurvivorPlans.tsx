import { Link } from 'react-router'

import type { SurvivorOut, SurvivorPick, SurvivorPlan } from '@/api/types'
import { EmptyState } from '@/components/common/EmptyState'
import { cn } from '@/utils/cn'
import { formatPercent } from '@/utils/format'

import { weekRanges } from './poolFormat'

const WORTH_SAVING = 8
const MIN_VALUE = 0.005

function PickChip({ pick }: { pick: SurvivorPick }) {
  return (
    <span
      title={pick.later ? `Week ${pick.week}: a later week the lookahead filled in` : `Week ${pick.week}`}
      className={cn(
        'inline-flex items-baseline gap-1 rounded-md border px-1.5 py-0.5 text-xs',
        pick.later ? 'border-dashed text-muted-foreground' : 'bg-card font-medium',
      )}
    >
      <span className="text-muted-foreground">W{pick.week}</span>
      <span className="font-semibold">{pick.team}</span>
      <span className="text-muted-foreground">{pick.is_home ? 'vs' : 'at'} {pick.opponent}</span>
      <span className="tabular">{formatPercent(pick.win_prob, 0)}</span>
    </span>
  )
}

function PlanItem({ plan, plannedWeeks }: { plan: SurvivorPlan; plannedWeeks: number[] }) {
  const first = plan.picks[0]
  return (
    <li data-first={first?.team ?? ''} className={cn('space-y-2 px-3 py-3', plan.rank === 1 && 'bg-accent/40')}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="font-medium">
          <span className="text-muted-foreground">#{plan.rank}</span> Pick {first?.team ?? '—'} now
        </div>
        <div className="text-right">
          <div className="tabular text-lg font-semibold">{formatPercent(plan.total_survival, 1)}</div>
          <div className="text-xs text-muted-foreground">
            weeks {weekRanges(plannedWeeks)}: {formatPercent(plan.survival, 1)}
            {plan.later_survival !== null ? ` · later: ${formatPercent(plan.later_survival, 1)}` : ''}
          </div>
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {plan.picks.map((pick) => (
          <PickChip key={pick.week} pick={pick} />
        ))}
      </div>
    </li>
  )
}

/** Ranked survivor plans, one per pick this week, and the teams worth saving for later weeks. */
export function SurvivorPlans({ result }: { result: SurvivorOut }) {
  const saving = result.future_values.filter((item) => item.value >= MIN_VALUE).slice(0, WORTH_SAVING)
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Planned pick by pick: weeks {weekRanges(result.planned_weeks)}. Later weeks with predictions ({weekRanges(result.later_weeks)}) are filled in with the best use of
        the teams left, which decides which teams to save.
        {result.missing_weeks.length > 0 ? (
          <>
            {' '}
            No predictions for weeks {weekRanges(result.missing_weeks)}, so the plan cannot save teams for them; predict them from the{' '}
            <Link to="/predictions" className="underline underline-offset-2">
              Predictions page
            </Link>
            .
          </>
        ) : null}
      </p>
      {result.plans.length === 0 ? (
        <EmptyState title="No plan from this week" description="Every team playing this week is already used, or its game is over." />
      ) : (
        <ol aria-label="Survivor plans" className="divide-y rounded-xl border bg-card">
          {result.plans.map((plan) => (
            <PlanItem key={plan.rank} plan={plan} plannedWeeks={result.planned_weeks} />
          ))}
        </ol>
      )}
      {saving.length > 0 ? (
        <div className="space-y-1">
          <h3 className="text-sm font-semibold">Worth saving</h3>
          <p className="text-xs text-muted-foreground">Share of the later weeks’ survival chance lost by using the team now.</p>
          <ul aria-label="Teams worth saving" className="flex flex-wrap gap-1.5">
            {saving.map((item) => (
              <li key={item.team} className="inline-flex items-baseline gap-1 rounded-md border bg-card px-2 py-0.5 text-xs">
                <span className="font-semibold">{item.team}</span>
                <span className="tabular text-muted-foreground">{formatPercent(item.value, 0)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}
