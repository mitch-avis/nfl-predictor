import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { FeatureImportance, FeatureImportanceMeasure } from '@/api/types'
import { humanize } from '@/utils/format'

const MEASURES: Record<FeatureImportanceMeasure, { name: string; caption: string }> = {
  mean_abs_shap: {
    name: 'Mean |SHAP| (points)',
    caption:
      'Mean |SHAP|: how far the feature moves the model’s adjustment to the market line (or the prediction itself when not anchored), in points, averaged over the model’s training games and added across the margin and total heads.',
  },
  total_gain: {
    name: 'Total gain',
    caption:
      'Total gain: loss reduction summed over every split on the feature, across the margin and total heads (a run that did not record SHAP).',
  },
  summed_average_gain: {
    name: 'Summed average gain',
    caption:
      'Average gain per split, summed over the feature’s encoded columns and both heads (an older run that did not record total gain). This favors features with many categories.',
  },
}

/** Horizontal bars of the top features, captioned with the measure they show. */
export function FeatureImportanceChart({ importance, limit = 25 }: { importance: FeatureImportance; limit?: number }) {
  const data = importance.rows.slice(0, limit).map((row) => ({ name: humanize(row.feature), value: row.value }))
  if (data.length === 0 || importance.measure === null) return <div className="text-sm text-muted-foreground">No feature importance recorded.</div>
  const measure = MEASURES[importance.measure]
  return (
    <div className="space-y-2">
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
            <Bar dataKey="value" name={measure.name} fill="var(--chart-1)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">{measure.caption}</p>
    </div>
  )
}
