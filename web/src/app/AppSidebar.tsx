import { NavLink, useLocation } from 'react-router'

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  useSidebar,
} from '@/components/ui/sidebar'
import type { User } from '@/api/types'

import { NAV_ITEMS, NAV_SECTIONS } from './nav'

/** Left navigation in sections: an icon rail on desktop when collapsed, a drawer on mobile. */
export function AppSidebar({ user }: { user: User }) {
  const location = useLocation()
  const { setOpenMobile, isMobile } = useSidebar()
  const items = NAV_ITEMS.filter((item) => !item.adminOnly || user.role === 'admin')
  const sections = NAV_SECTIONS.map((section) => ({ section, items: items.filter((item) => item.section === section) })).filter(
    (group) => group.items.length > 0,
  )

  return (
    <Sidebar collapsible="icon" variant="sidebar">
      <SidebarHeader>
        <div className="flex items-center gap-2 px-1 py-1">
          <img src="/favicon.svg" alt="" className="size-8 shrink-0 rounded-lg" />
          <div className="min-w-0 group-data-[collapsible=icon]:hidden">
            <div className="truncate text-sm font-semibold leading-tight">NFL Predictor</div>
            <div className="truncate text-xs text-muted-foreground">Picks, ratings, edges</div>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        {sections.map(({ section, items: sectionItems }) => (
          <SidebarGroup key={section} role="group" aria-label={section}>
            <SidebarGroupLabel>{section}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {sectionItems.map((item) => {
                  const active =
                    item.to === '/' ? location.pathname === '/' : location.pathname.startsWith(item.to)
                  return (
                    <SidebarMenuItem key={item.to}>
                      <SidebarMenuButton asChild isActive={active} tooltip={item.label}>
                        <NavLink to={item.to} onClick={() => isMobile && setOpenMobile(false)}>
                          <item.icon />
                          <span>{item.label}</span>
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  )
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>
      <SidebarFooter>
        <div className="px-2 py-1 text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">
          Signed in as <span className="font-medium text-foreground">{user.username}</span> ·{' '}
          {user.role}
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
