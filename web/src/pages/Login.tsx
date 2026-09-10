import { useState, type FormEvent } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router'

import { ApiError } from '@/api/client'
import { useLogin, useSession } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export function LoginPage() {
  const session = useSession()
  const login = useLogin()
  const navigate = useNavigate()
  const location = useLocation()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const from = (location.state as { from?: string } | null)?.from ?? '/'

  if (session.data) return <Navigate to={from} replace />

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    login.mutate({ username, password }, { onSuccess: () => navigate(from, { replace: true }) })
  }
  const error = login.error instanceof ApiError ? login.error : null

  return (
    <div className="grid min-h-dvh place-items-center bg-muted/40 px-4 py-8">
      <Card className="w-full max-w-sm">
        <CardHeader className="text-center">
          <img src="/favicon.svg" alt="" className="mx-auto mb-2 size-12 rounded-xl" />
          <CardTitle className="text-xl">NFL Predictor</CardTitle>
          <CardDescription>Sign in to view predictions, rankings, and reports.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4" noValidate>
            <div className="space-y-1.5">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                autoComplete="username"
                autoCapitalize="none"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            {error ? (
              <p role="alert" className="text-sm text-destructive">
                {error.status === 429 ? 'Too many attempts. Wait a few minutes.' : error.message}
              </p>
            ) : null}
            <Button type="submit" className="w-full" disabled={login.isPending || !username || !password}>
              {login.isPending ? 'Signing in…' : 'Sign in'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
