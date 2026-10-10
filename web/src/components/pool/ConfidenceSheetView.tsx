import { AlertTriangle, Download, Lock } from 'lucide-react'
import { useState } from 'react'

import type { ConfidenceSheet, SheetRow } from '@/api/types'
import { CopyButton } from '@/components/common/CopyButton'
import { StatTile } from '@/components/common/StatTile'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { cn } from '@/utils/cn'
import { formatNumber, formatPercent } from '@/utils/format'

import { type SheetOrder, formatKickoff, matchupLabel, sheetText, sortSheetRows } from './poolFormat'

function ResultChip({ row }: { row: SheetRow }) {
  if (row.result === null) return null
  if (row.result === 'tie') {
    return <span className="rounded-md bg-muted px-1.5 py-0.5 text-xs text-muted-foreground" title="A tie counts as a miss for both sides.">Tie, no points</span>
  }
  const won = row.result === 'win'
  return (
    <span className={cn('tabular rounded-md px-1.5 py-0.5 text-xs font-semibold', won ? 'bg-success/15 text-success' : 'bg-muted text-muted-foreground')}>
      {won ? `+${row.points_won ?? 0}` : 'Miss'}
    </span>
  )
}

function SheetLine({ row }: { row: SheetRow }) {
  const final = row.away_score !== null && row.home_score !== null
  return (
    <li data-pick={row.pick ?? ''} className="flex items-center gap-3 px-3 py-2">
      <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-primary text-sm font-bold text-primary-foreground">
        {row.points ?? '—'}
        <span className="sr-only"> points</span>
      </span>
      <div className="min-w-0 flex-1">
        <div className="font-medium">
          {row.pick ?? '—'} <span className="text-xs font-normal text-muted-foreground">{matchupLabel(row)}</span>
        </div>
        <div className="text-xs text-muted-foreground">
          {formatKickoff(row.kickoff)}
          {final ? ` · final ${row.away_abbr} ${row.away_score}, ${row.home_abbr} ${row.home_score}` : ''}
        </div>
      </div>
      <ResultChip row={row} />
      <div className="w-20 shrink-0 text-right">
        <div className="tabular text-sm font-semibold">{formatPercent(row.pick_win_prob, 1)}</div>
        <div className="tabular text-xs text-muted-foreground">{formatNumber(row.expected_points, 1)} exp.</div>
      </div>
    </li>
  )
}

/** A week's confidence pool sheet: what to enter, when picks lock, and how the week has scored. */
export function ConfidenceSheetView({ sheet, csvUrl }: { sheet: ConfidenceSheet; csvUrl: string }) {
  const [order, setOrder] = useState<SheetOrder>('points')
  const rows = sortSheetRows(sheet.rows, order)
  return (
    <div className="space-y-4">
      {sheet.generated_after_lock ? (
        <Alert>
          <AlertTriangle />
          <AlertTitle>Written after the lock</AlertTitle>
          <AlertDescription>
            These predictions were written after the week’s first kickoff ({formatKickoff(sheet.lock_at)}). Pool picks are made once, before that game, so the picks you
            entered then still stand; this sheet may differ from them.
          </AlertDescription>
        </Alert>
      ) : null}
      {sheet.rank_problem ? (
        <Alert variant="destructive">
          <AlertTriangle />
          <AlertTitle>Ranks are not one to {sheet.games}</AlertTitle>
          <AlertDescription>{sheet.rank_problem} The sheet shows the file as written.</AlertDescription>
        </Alert>
      ) : null}
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <StatTile label="Points on offer" value={sheet.max_points} footnote={`${sheet.games} games, 1 to ${sheet.games} once each`} />
        <StatTile
          label="Expected points"
          value={formatNumber(sheet.expected_points, 1)}
          hint="Each pick's points times its chance to win, added up. A tie scores as a miss for both sides."
        />
        <StatTile
          label="Points won"
          value={sheet.points_won ?? '—'}
          footnote={`${sheet.games_final} of ${sheet.games} games final · ${sheet.points_possible ?? '—'} still possible`}
        />
        <StatTile
          label="Picks lock"
          value={<span className="inline-flex items-center gap-1 text-base"><Lock className="size-4" />{formatKickoff(sheet.lock_at)}</span>}
          hint="Picks are single-shot: made once, before the week's first game, and never updated during the week."
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-md border" role="group" aria-label="Order">
          <Button variant={order === 'points' ? 'secondary' : 'ghost'} size="sm" aria-pressed={order === 'points'} onClick={() => setOrder('points')}>
            By points
          </Button>
          <Button variant={order === 'kickoff' ? 'secondary' : 'ghost'} size="sm" aria-pressed={order === 'kickoff'} onClick={() => setOrder('kickoff')}>
            By kickoff
          </Button>
        </div>
        <div className="flex gap-2">
          <CopyButton text={sheetText(rows)} label="Copy sheet" />
          <Button variant="outline" size="sm" asChild>
            <a href={csvUrl} download>
              <Download className="size-4" /> Download CSV
            </a>
          </Button>
        </div>
      </div>
      <ol aria-label="Confidence sheet" className="divide-y rounded-xl border bg-card">
        {rows.map((row) => (
          <SheetLine key={row.game_id} row={row} />
        ))}
      </ol>
    </div>
  )
}
