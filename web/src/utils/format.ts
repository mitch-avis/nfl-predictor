/** Number, date, and label formatting shared across pages. */

export function formatNumber(value: number | null | undefined, decimals = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export function formatPercent(value: number | null | undefined, decimals = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(decimals)}%`
}

/** Render a signed number such as a spread or margin: `+3.5`, `-1.0`, `PK` for zero. */
export function formatSigned(value: number | null | undefined, decimals = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  if (Math.abs(value) < 1e-9) return 'PK'
  const text = Math.abs(value).toFixed(decimals)
  return value > 0 ? `+${text}` : `-${text}`
}

/** Render American moneyline odds with an explicit sign. */
export function formatMoneyline(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const rounded = Math.round(value)
  return rounded > 0 ? `+${rounded}` : String(rounded)
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value.length === 10 ? `${value}T12:00:00` : value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })
}

/** Humanize a relative timestamp: "3 min ago", "2 days ago". */
export function formatRelative(value: string | null | undefined, now: Date = new Date()): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  const seconds = Math.round((now.getTime() - date.getTime()) / 1000)
  const abs = Math.abs(seconds)
  const suffix = seconds >= 0 ? 'ago' : 'from now'
  if (abs < 60) return `${abs}s ${suffix}`
  if (abs < 3600) return `${Math.round(abs / 60)} min ${suffix}`
  if (abs < 86400) return `${Math.round(abs / 3600)} hr ${suffix}`
  return `${Math.round(abs / 86400)} days ${suffix}`
}

export function shortHash(value: string | null | undefined, length = 8): string {
  if (!value) return '—'
  return value.slice(0, length)
}

/** Turn `snake_case_key` into `Snake case key`. */
export function humanize(key: string): string {
  const spaced = key.replace(/_/g, ' ').trim()
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

export function seasonWeekLabel(season: number | null, week: number | null): string {
  if (season === null && week === null) return '—'
  if (week === null) return String(season)
  return `${season ?? '?'} · Week ${week}`
}
