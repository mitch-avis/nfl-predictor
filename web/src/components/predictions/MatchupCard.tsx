import { ArrowRight, Minus } from 'lucide-react'

import type { Row } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { cn } from '@/utils/cn'
import { formatDateTime, formatMoneyline, formatNumber, formatSigned } from '@/utils/format'

import { QuantileBand } from './QuantileBand'
import { WinProbBar } from './WinProbBar'

const num = (v: Row[string]): number | null => (typeof v === 'number' ? v : null)
const str = (v: Row[string]): string => (v === null || v === undefined ? '—' : String(v))

/** One game as a card, used on narrow screens. */
export function MatchupCard({ row }: { row: Row }) {
  const away = str(row.away_abbr)
  const home = str(row.home_abbr)
  const pick = str(row.predicted_winner)
  const agrees = row.agrees_with_market
  const margin = num(row.predicted_margin_raw) ?? num(row.predicted_margin)
  return (
    <Card className="gap-3 p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-base font-semibold">
            {away} <span className="text-muted-foreground">@</span> {home}
          </div>
          <div className="text-xs text-muted-foreground">
            {formatDateTime(str(row.game_datetime))}
            {row.stadium_name ? ` · ${str(row.stadium_name)}` : ''}
            {row.is_divisional_matchup ? ' · Division' : ''}
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <Badge>Pick: {pick}</Badge>
          <span className="text-xs text-muted-foreground">Conf. rank {str(row.confidence_rank)}</span>
        </div>
      </div>
      <WinProbBar homeProb={num(row.home_win_prob)} marketHomeProb={num(row.market_home_prob_novig)} away={away} home={home} />
      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="rounded-lg bg-muted/60 p-2">
          <div className="text-muted-foreground">Score</div>
          <div className="tabular text-sm font-semibold">
            {formatNumber(num(row.predicted_away_score), 0)}–{formatNumber(num(row.predicted_home_score), 0)}
          </div>
        </div>
        <div className="rounded-lg bg-muted/60 p-2">
          <div className="text-muted-foreground">Margin · spread</div>
          <div className="tabular text-sm font-semibold">
            {formatSigned(margin)} <span className="text-muted-foreground">·</span> {formatSigned(num(row.home_spread))}
          </div>
        </div>
        <div className="rounded-lg bg-muted/60 p-2">
          <div className="text-muted-foreground">Total · O/U</div>
          <div className="tabular text-sm font-semibold">
            {formatNumber(num(row.predicted_total_raw) ?? num(row.predicted_total), 1)} <span className="text-muted-foreground">·</span> {formatNumber(num(row.total_line), 1)}
          </div>
        </div>
      </div>
      <QuantileBand p10={num(row.predicted_margin_p10)} p50={num(row.predicted_margin_p50)} p90={num(row.predicted_margin_p90)} point={margin} market={num(row.market_home_margin)} />
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span>
          ML {away} {formatMoneyline(num(row.away_moneyline))} / {home} {formatMoneyline(num(row.home_moneyline))}
        </span>
        {agrees === false ? (
          <span className={cn('inline-flex items-center gap-1 font-medium text-warning-foreground')}>
            <ArrowRight className="size-3" /> Disagrees with market ({str(row.market_favorite)})
          </span>
        ) : agrees === true ? (
          <span className="inline-flex items-center gap-1">
            <Minus className="size-3" /> Agrees with market
          </span>
        ) : null}
      </div>
    </Card>
  )
}
