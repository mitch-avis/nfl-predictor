import { LogOut, Moon, Sun, SunMoon } from 'lucide-react'
import { Outlet, useNavigate } from 'react-router'

import { useLogout, useRuns } from '@/api/queries'
import type { User } from '@/api/types'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Separator } from '@/components/ui/separator'
import { SidebarInset, SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { seasonWeekLabel } from '@/utils/format'

import { AppSidebar } from './AppSidebar'
import { useTheme, type Theme } from './ThemeProvider'

const THEME_ICONS = { light: Sun, dark: Moon, system: SunMoon } as const
const THEME_ORDER: Theme[] = ['light', 'dark', 'system']

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const Icon = THEME_ICONS[theme]
  const next = THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length]
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={`Theme: ${theme}. Switch to ${next}`}
          onClick={() => setTheme(next)}
        >
          <Icon className="size-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>Theme: {theme}</TooltipContent>
    </Tooltip>
  )
}

function ActiveRunChip() {
  const runs = useRuns('all')
  const active = runs.data?.runs.find((run) => run.is_active)
  if (runs.isLoading) return <div className="h-5 w-32 animate-pulse rounded bg-muted" />
  if (!active) {
    return (
      <Badge variant="outline" className="text-muted-foreground">
        No active run
      </Badge>
    )
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="secondary" className="max-w-[60vw] truncate font-normal">
          <span className="hidden sm:inline">Active run:&nbsp;</span>
          <span className="font-medium">{active.run_id}</span>
          <span className="ml-1 text-muted-foreground">· {seasonWeekLabel(active.season, active.week)}</span>
        </Badge>
      </TooltipTrigger>
      <TooltipContent>
        Every page reads from this run. Change it on the Runs page.
      </TooltipContent>
    </Tooltip>
  )
}

function UserMenu({ user }: { user: User }) {
  const logout = useLogout()
  const navigate = useNavigate()
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <span className="grid size-5 place-items-center rounded-full bg-primary text-[11px] font-semibold uppercase text-primary-foreground">
            {user.username.slice(0, 1)}
          </span>
          <span className="hidden sm:inline">{user.username}</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>
          {user.username} <span className="font-normal text-muted-foreground">· {user.role}</span>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={() => logout.mutate(undefined, { onSuccess: () => navigate('/login') })}
        >
          <LogOut className="size-4" /> Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** Sidebar + header + routed content. */
export function AppShell({ user }: { user: User }) {
  return (
    <SidebarProvider>
      <AppSidebar user={user} />
      <SidebarInset className="min-w-0">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-2 border-b bg-background/85 px-3 backdrop-blur sm:px-4">
          <SidebarTrigger aria-label="Toggle navigation" />
          <Separator orientation="vertical" className="mr-1 h-5" />
          <ActiveRunChip />
          <div className="ml-auto flex items-center gap-1">
            <ThemeToggle />
            <UserMenu user={user} />
          </div>
        </header>
        <main className="min-w-0 flex-1 px-3 py-4 sm:px-6 sm:py-6 safe-bottom">
          <Outlet />
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}
