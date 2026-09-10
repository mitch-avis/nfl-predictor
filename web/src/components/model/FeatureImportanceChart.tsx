import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { FeatureImportanceRow } from '@/api/types'
import { humanize } from '@/utils/format'

/** Horizontal bars of combined gain for the top features. */
export function FeatureImportanceChart({ rows, limit = 25 }: { rows: FeatureImportanceRow[]; limit?: number }) {
  const data = rows.slice(0, limit).map((row) => ({ name: humanize(row.feature), gain: row.gain, margin: row.margin_gain, total: row.total_gain }))
  if (data.length === 0) return <div className="text-sm text-muted-foreground">No feature importance recorded.</div>
  return (
    <div style={{ height: Math.max(240, data.length * 22 + 40) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke="var(--border)" />
          <XAxis type="number" tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} stroke="var(--border)" />
          <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 11, fill: 'var(--foreground)' }} stroke="var(--border)" interval={0} />
          <Tooltip
            cursor={{ fill: 'var(--accent)' }}
            contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--popover-foreground)', fontSize: 12 }}
            formatter={(value) => (typeof value === 'number' ? value.toFixed(2) : String(value))}
          />
          <Bar dataKey="gain" name="Combined gain" fill="var(--chart-1)" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
