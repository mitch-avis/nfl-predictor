import type { EntityKind } from '@/api/types'
import { Button } from '@/components/ui/button'

import { VIEWS, type StatView, type ViewOption, type ViewState } from './views'

export interface ViewChange {
  view?: StatView
  category?: string
  hidden?: string[]
}

function ToggleRow({ label, options, isActive, onSelect }: { label: string; options: { id: string; label: string }[]; isActive: (id: string) => boolean; onSelect: (id: string) => void }) {
  return (
    // On phones a row scrolls sideways instead of wrapping into several lines.
    <div role="group" aria-label={label} className="flex w-full min-w-0 gap-1.5 overflow-x-auto pb-1 sm:w-auto sm:flex-wrap sm:overflow-visible sm:pb-0">
      {options.map((option) => {
        const active = isActive(option.id)
        return (
          <Button key={option.id} type="button" size="sm" variant={active ? 'default' : 'outline'} className="shrink-0" aria-pressed={active} onClick={() => onSelect(option.id)}>
            {option.label}
          </Button>
        )
      })}
    </div>
  )
}

/**
 * The stat controls of the ratings pages: one of the views, then (outside the ratings view) the
 * team category and the subcategories to include; a quarterback page filters by category.
 */
export function ViewControls({ kind, state, onChange, views = VIEWS }: { kind: EntityKind; state: ViewState; onChange: (change: ViewChange) => void; views?: ViewOption[] }) {
  const showFilters = state.view !== 'ratings'
  const toggle = (option: string) =>
    onChange({ hidden: state.hidden.includes(option) ? state.hidden.filter((hidden) => hidden !== option) : [...state.hidden, option] })
  return (
    <div className="flex flex-col gap-2">
      <ToggleRow label="View" options={views} isActive={(id) => state.view === id} onSelect={(id) => onChange({ view: id as StatView })} />
      {showFilters && kind === 'teams' ? (
        <ToggleRow
          label="Category"
          options={state.categoryOptions.map((name) => ({ id: name, label: name }))}
          isActive={(id) => state.category === id}
          onSelect={(id) => onChange({ category: id, hidden: [] })}
        />
      ) : null}
      {showFilters && state.subcategoryOptions.length > 0 ? (
        <ToggleRow
          label={kind === 'teams' ? 'Subcategories' : 'Categories'}
          options={state.subcategoryOptions.map((name) => ({ id: name, label: name }))}
          isActive={(id) => !state.hidden.includes(id)}
          onSelect={toggle}
        />
      ) : null}
    </div>
  )
}
