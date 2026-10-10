import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import type { ReactElement, ReactNode } from 'react'
import { MemoryRouter } from 'react-router'

import { TooltipProvider } from '@/components/ui/tooltip'

/**
 * Render `ui` inside the providers the app shell supplies: a router, a query client of its own
 * (no retries) and the tooltip provider. `rerender` keeps the same providers and query client.
 */
export function renderWithProviders(ui: ReactElement, { route = '/' }: { route?: string } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  function Providers({ children }: { children: ReactNode }) {
    return (
      <MemoryRouter initialEntries={[route]}>
        <QueryClientProvider client={client}>
          <TooltipProvider>{children}</TooltipProvider>
        </QueryClientProvider>
      </MemoryRouter>
    )
  }
  return render(ui, { wrapper: Providers })
}
