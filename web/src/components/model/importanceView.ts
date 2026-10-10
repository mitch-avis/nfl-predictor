import type { FeatureImportance, FeatureImportanceMeasure, ImportanceHead } from '@/api/types'
import { humanize } from '@/utils/format'

/** Axis and tooltip name of each measure. */
export const MEASURE_NAMES: Record<FeatureImportanceMeasure, string> = {
  mean_abs_shap: 'Mean |SHAP| (points)',
  total_gain: 'Total gain',
  summed_average_gain: 'Summed average gain',
}

const CAPTIONS: Record<FeatureImportanceMeasure, Record<ImportanceHead, string>> = {
  mean_abs_shap: {
    combined:
      'Mean |SHAP|: how far the feature moves the model’s adjustment to the market line (or the prediction itself when not anchored), in points, averaged over the model’s training games and added across the margin and total heads.',
    margin:
      'Mean |SHAP|: how far the feature moves the predicted margin’s adjustment to the market line (or the predicted margin itself when not anchored), in points, averaged over the model’s training games, for the margin head alone.',
  },
  total_gain: {
    combined:
      'Total gain: loss reduction summed over every split on the feature, across the margin and total heads (a run that did not record SHAP).',
    margin:
      'Total gain: loss reduction summed over every split on the feature, for the margin head alone (a run that did not record SHAP).',
  },
  summed_average_gain: {
    combined:
      'Average gain per split, summed over the feature’s encoded columns and both heads (an older run that did not record total gain). This favors features with many categories.',
    margin:
      'Average gain per split, summed over the feature’s encoded columns, for the margin head alone (an older run that did not record total gain). This favors features with many categories.',
  },
}

/** The note under the chart: what the measure is and which heads it covers. */
export function importanceCaption(measure: FeatureImportanceMeasure, head: ImportanceHead): string {
  return CAPTIONS[measure][head]
}

/** The bars to plot for a view: the view's ranking, first `limit` features, readable names. */
export function importanceRows(importance: FeatureImportance, head: ImportanceHead, limit: number): { name: string; value: number }[] {
  const rows = head === 'margin' ? importance.margin_rows : importance.rows
  return rows.slice(0, limit).map((row) => ({ name: humanize(row.feature), value: row.value }))
}
