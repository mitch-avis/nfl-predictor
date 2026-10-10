import {
  Activity,
  BarChart3,
  BookOpen,
  CalendarRange,
  ChartNoAxesGantt,
  Database,
  FlaskConical,
  LayoutDashboard,
  ListChecks,
  ListOrdered,
  PlayCircle,
  Shield,
  Swords,
  Trophy,
  UserRound,
  Users,
  type LucideIcon,
} from 'lucide-react'

/** The sidebar's sections, in display order. */
export const NAV_SECTIONS = ['This week', 'Teams and players', 'Model and data', 'Reference'] as const
export type NavSection = (typeof NAV_SECTIONS)[number]

export interface NavItem {
  to: string
  label: string
  icon: LucideIcon
  description: string
  section: NavSection
  adminOnly?: boolean
}

/** Sidebar navigation, in display order within each section. */
export const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, description: 'This week at a glance', section: 'This week' },
  { to: '/predictions', label: 'Predictions', icon: Trophy, description: 'Game-by-game model output with market lines', section: 'This week' },
  { to: '/pool', label: 'Pool', icon: ListChecks, description: 'Confidence sheet, tiebreakers and survivor plans', section: 'This week' },
  { to: '/betting', label: 'Betting', icon: BarChart3, description: 'Moneyline and spread edges', section: 'This week' },
  { to: '/power', label: 'Power Rankings', icon: ListOrdered, description: 'The model’s team strengths and projected standings', section: 'Teams and players' },
  { to: '/teams', label: 'Teams', icon: Shield, description: 'Schedule-adjusted team ratings and season stats', section: 'Teams and players' },
  { to: '/qbs', label: 'Quarterbacks', icon: UserRound, description: 'Schedule-adjusted quarterback ratings and stats', section: 'Teams and players' },
  { to: '/schedules', label: 'Schedules', icon: CalendarRange, description: 'Every team-season ranked by schedule strength', section: 'Teams and players' },
  { to: '/rank-ranges', label: 'Rank ranges', icon: ChartNoAxesGantt, description: 'How firm each rank is when the season is redrawn', section: 'Teams and players' },
  { to: '/head-to-head', label: 'Head to head', icon: Swords, description: 'How often one team or quarterback rates above another', section: 'Teams and players' },
  { to: '/model', label: 'Model', icon: FlaskConical, description: 'Metrics, calibration, and feature importance', section: 'Model and data' },
  { to: '/data', label: 'Data & ETL', icon: Database, description: 'Dataset freshness and pipeline health', section: 'Model and data' },
  { to: '/jobs', label: 'Jobs', icon: PlayCircle, description: 'Run ETL, training, and reports', section: 'Model and data' },
  { to: '/runs', label: 'Runs', icon: Activity, description: 'Every run on disk; pin the active one', section: 'Model and data' },
  { to: '/users', label: 'Users', icon: Users, description: 'Accounts and roles', section: 'Model and data', adminOnly: true },
  { to: '/glossary', label: 'Glossary', icon: BookOpen, description: 'What every column means', section: 'Reference' },
]
