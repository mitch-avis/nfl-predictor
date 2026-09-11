import { lazy, Suspense, type ReactNode } from 'react'
import { createBrowserRouter } from 'react-router'

import { Skeleton } from './components/ui/skeleton'

import { RequireAuth } from './app/RequireAuth'
import { BettingPage } from './pages/Betting'
import { DashboardPage } from './pages/Dashboard'
import { DataStatusPage } from './pages/DataStatus'
import { GlossaryPage } from './pages/Glossary'
import { JobDetailPage } from './pages/JobDetail'
import { JobsPage } from './pages/Jobs'
import { LoginPage } from './pages/Login'
import { PlaceholderPage } from './pages/Placeholder'
import { PowerRankingsPage } from './pages/PowerRankings'
import { PredictionsPage } from './pages/Predictions'
import { RunsPage } from './pages/Runs'
import { UsersPage } from './pages/Users'

const ModelPage = lazy(() => import('./pages/Model').then((m) => ({ default: m.ModelPage })))

function Lazy({ children }: { children: ReactNode }) {
  return <Suspense fallback={<Skeleton className="h-64 w-full" />}>{children}</Suspense>
}

export const router = createBrowserRouter([
  { path: '/login', element: <LoginPage /> },
  {
    path: '/',
    element: <RequireAuth />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'predictions', element: <PredictionsPage /> },
      { path: 'power', element: <PowerRankingsPage /> },
      { path: 'betting', element: <BettingPage /> },
      { path: 'data', element: <DataStatusPage /> },
      { path: 'model', element: <Lazy><ModelPage /></Lazy> },
      { path: 'glossary', element: <GlossaryPage /> },
      { path: 'jobs', element: <JobsPage /> },
      { path: 'jobs/:jobId', element: <JobDetailPage /> },
      { path: 'runs', element: <RunsPage /> },
      { path: 'users', element: <UsersPage /> },
      { path: '*', element: <PlaceholderPage /> },
    ],
  },
])
