import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'

import type { CalibrationBin } from '@/api/types'

/** Reliability diagram: observed win rate against predicted probability per bin. */
export function CalibrationChart({ bins }: { bins: CalibrationBin[] }) {
  const data = bins.filter((b) => b.count > 0).map((b) => ({ x: b.avg_pred, y: b.avg_actual, n: b.count, label: `${(b.bin_lower * 100).toFixed(0)}–${(b.bin_upper * 100).toFixed(0)}%` }))
  if (data.length === 0) return <div className="text-sm text-muted-foreground">No calibration bins.</div>
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ left: 0, right: 12, top: 8, bottom: 8 }}>
          <CartesianGrid stroke="var(--border)" />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} stroke="var(--border)" label={{ value: 'Predicted', position: 'insideBottom', offset: -4, fontSize: 11, fill: 'var(--muted-foreground)' }} />
          <YAxis type="number" dataKey="y" domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} stroke="var(--border)" width={44} />
          <Tooltip
            contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--popover-foreground)', fontSize: 12 }}
            formatter={(value, name) => [typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : String(value), name === 'y' ? 'Observed' : 'Predicted']}
            labelFormatter={(_label, payload) => {
              const point = payload?.[0]?.payload as { label?: string; n?: number } | undefined
              return point ? `${point.label} · ${point.n} games` : ''
            }}
          />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="y" stroke="var(--chart-1)" strokeWidth={2} dot={{ r: 3, fill: 'var(--chart-1)' }} isAnimationActive={false} />
          <Scatter dataKey="y" fill="var(--chart-1)" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
