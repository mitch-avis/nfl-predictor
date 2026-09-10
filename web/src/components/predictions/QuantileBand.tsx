import { formatSigned } from '@/utils/format'

/** A mini axis showing the p10-p90 margin band with the point prediction and the market spread. */
export function QuantileBand({ p10, p50, p90, point, market }: { p10: number | null; p50: number | null; p90: number | null; point: number | null; market: number | null }) {
  if (p10 === null || p90 === null) return null
  const lo = Math.min(p10, market ?? p10, -21)
  const hi = Math.max(p90, market ?? p90, 21)
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`
  return (
    <div className="space-y-1">
      <div className="relative h-4 w-full">
        <div className="absolute inset-x-0 top-1/2 h-px bg-border" />
        <div className="absolute top-1/2 h-2 -translate-y-1/2 rounded-full bg-primary/25" style={{ left: x(p10), width: `calc(${x(p90)} - ${x(p10)})` }} />
        {p50 !== null ? <div className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-primary/70" style={{ left: x(p50) }} title={`p50 ${formatSigned(p50)}`} /> : null}
        {point !== null ? <div className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary" style={{ left: x(point) }} title={`Predicted ${formatSigned(point)}`} /> : null}
        {market !== null ? <div className="absolute top-1/2 h-3.5 w-0.5 -translate-y-1/2 bg-foreground" style={{ left: x(market) }} title={`Market ${formatSigned(market)}`} /> : null}
        <div className="absolute top-1/2 h-2 w-px -translate-y-1/2 bg-muted-foreground/50" style={{ left: x(0) }} />
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>{formatSigned(p10)}</span>
        <span>home margin · p10–p90</span>
        <span>{formatSigned(p90)}</span>
      </div>
    </div>
  )
}
