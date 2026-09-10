import { Columns3 } from 'lucide-react'

import type { TablePayload } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useLocalStorage } from '@/hooks/useLocalStorage'

/** Toggle column groups on and off; the choice is remembered per `tableId`. */
export function useColumnGroups(tableId: string, table: TablePayload | undefined, defaultHidden: string[] = []) {
  const [hidden, setHidden] = useLocalStorage<string[]>(`nflp.columns.${tableId}`, defaultHidden)
  const groups = table ? Object.keys(table.column_groups) : []
  const columns = table ? table.visible_columns.filter((key) => !hidden.includes(table.column_metadata[key]?.group)) : []
  const toggle = (group: string) => setHidden((prev) => (prev.includes(group) ? prev.filter((g) => g !== group) : [...prev, group]))
  return { hidden, groups, columns, toggle, reset: () => setHidden(defaultHidden) }
}

export function ColumnPicker({ groups, hidden, onToggle, onReset }: { groups: string[]; hidden: string[]; onToggle: (g: string) => void; onReset: () => void }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm">
          <Columns3 className="size-4" /> Columns
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-56">
        <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">Column groups</div>
        <ul className="space-y-2">
          {groups.map((group) => (
            <li key={group} className="flex items-center gap-2">
              <Checkbox id={`col-${group}`} checked={!hidden.includes(group)} onCheckedChange={() => onToggle(group)} />
              <label htmlFor={`col-${group}`} className="text-sm">
                {group}
              </label>
            </li>
          ))}
        </ul>
        <Button variant="ghost" size="sm" className="mt-3 w-full" onClick={onReset}>
          Reset
        </Button>
      </PopoverContent>
    </Popover>
  )
}
