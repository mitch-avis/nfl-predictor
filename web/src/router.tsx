import { createBrowserRouter, type RouteObject } from 'react-router'

import { RequireAuth } from './app/RequireAuth'
import { BettingPage } from './pages/Betting'
import { DashboardPage } from './pages/Dashboard'
import { DataStatusPage } from './pages/DataStatus'
import { GlossaryPage } from './pages/Glossary'
import { JobDetailPage } from './pages/JobDetail'
import { JobsPage } from './pages/Jobs'
import { LoginPage } from './pages/Login'
import { ModelRoute } from './pages/ModelRoute'
import { NotFoundPage } from './pages/NotFound'
import { PoolPage } from './pages/Pool'
import { PowerRankingsPage } from './pages/PowerRankings'
import { PredictionsPage } from './pages/Predictions'
import { HeadToHeadRoute, RankRangesRoute, RatingsDetailRoute, RatingsIndexRoute, SchedulesRoute } from './pages/RatingsRoutes'
import { RunsPage } from './pages/Runs'
import { UsersPage } from './pages/Users'

export const routes: RouteObject[] = [
  { path: '/login', element: <LoginPage /> },
  {
    path: '/',
    element: <RequireAuth />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'predictions', element: <PredictionsPage /> },
      { path: 'pool', element: <PoolPage /> },
      { path: 'power', element: <PowerRankingsPage /> },
      { path: 'teams', element: <RatingsIndexRoute kind="teams" /> },
      { path: 'teams/:entityId', element: <RatingsDetailRoute kind="teams" /> },
      { path: 'qbs', element: <RatingsIndexRoute kind="qbs" /> },
      { path: 'qbs/:entityId', element: <RatingsDetailRoute kind="qbs" /> },
      { path: 'schedules', element: <SchedulesRoute /> },
      { path: 'rank-ranges', element: <RankRangesRoute /> },
      { path: 'head-to-head', element: <HeadToHeadRoute /> },
      { path: 'betting', element: <BettingPage /> },
      { path: 'data', element: <DataStatusPage /> },
      { path: 'model', element: <ModelRoute /> },
      { path: 'glossary', element: <GlossaryPage /> },
      { path: 'jobs', element: <JobsPage /> },
      { path: 'jobs/:jobId', element: <JobDetailPage /> },
      { path: 'runs', element: <RunsPage /> },
      { path: 'users', element: <UsersPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
]

export const router = createBrowserRouter(routes)
