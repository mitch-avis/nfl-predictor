import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router'

import { TooltipProvider } from '@/components/ui/tooltip'

/** Render `ui` inside the providers the app shell supplies: a router and the tooltip provider. */
export function renderWithProviders(ui: ReactElement, { route = '/' }: { route?: string } = {}) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <TooltipProvider>{ui}</TooltipProvider>
    </MemoryRouter>,
  )
}
