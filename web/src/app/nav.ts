import {
  Activity,
  BarChart3,
  BookOpen,
  Database,
  FlaskConical,
  LayoutDashboard,
  ListOrdered,
  PlayCircle,
  Trophy,
  Users,
  type LucideIcon,
} from 'lucide-react'

export interface NavItem {
  to: string
  label: string
  icon: LucideIcon
  description: string
  adminOnly?: boolean
  phase?: number
}

/** Sidebar navigation, in display order. Items with a `phase` beyond the current build show a placeholder. */
export const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, description: 'This week at a glance' },
  { to: '/predictions', label: 'Predictions', icon: Trophy, description: 'Game-by-game model output with market lines' },
  { to: '/power', label: 'Power Rankings', icon: ListOrdered, description: 'Team ratings and projected standings' },
  { to: '/betting', label: 'Betting', icon: BarChart3, description: 'Moneyline and spread edges' },
  { to: '/data', label: 'Data & ETL', icon: Database, description: 'Dataset freshness and pipeline health' },
  { to: '/model', label: 'Model', icon: FlaskConical, description: 'Metrics, calibration, and feature importance' },
  { to: '/jobs', label: 'Jobs', icon: PlayCircle, description: 'Run ETL, training, and reports' },
  { to: '/runs', label: 'Runs', icon: Activity, description: 'Every run on disk; pin the active one' },
  { to: '/users', label: 'Users', icon: Users, description: 'Accounts and roles', adminOnly: true },
  { to: '/glossary', label: 'Glossary', icon: BookOpen, description: 'What every column means' },
]

export const CURRENT_PHASE = 2
