import { useId, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { FeatureImportance, ImportanceHead } from '@/api/types'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'

import { MEASURE_NAMES, importanceCaption, importanceRows } from './importanceView'

/**
 * Horizontal bars of the top features, captioned with the measure they show. Both heads are added
 * by default; a switch shows the margin head's own ranking when the run recorded one.
 */
export function FeatureImportanceChart({ importance, limit = 25 }: { importance: FeatureImportance; limit?: number }) {
  const [head, setHead] = useState<ImportanceHead>('combined')
  const switchId = useId()
  const hasMargin = importance.margin_rows.length > 0
  const view: ImportanceHead = hasMargin ? head : 'combined'
  const data = importanceRows(importance, view, limit)
  if (data.length === 0 || importance.measure === null) return <div className="text-sm text-muted-foreground">No feature importance recorded.</div>
  return (
    <div className="space-y-2">
      {hasMargin ? (
        <div className="flex justify-end">
          <Label htmlFor={switchId} className="cursor-pointer text-xs font-normal text-muted-foreground">
            <Switch id={switchId} size="sm" checked={view === 'margin'} onCheckedChange={(checked) => setHead(checked ? 'margin' : 'combined')} />
            Margin head only
          </Label>
        </div>
      ) : null}
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
            <Bar dataKey="value" name={MEASURE_NAMES[importance.measure]} fill="var(--chart-1)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">{importanceCaption(importance.measure, view)}</p>
    </div>
  )
}
