import { Trash2 } from 'lucide-react'
import { useState, type FormEvent } from 'react'

import { useCreateUser, useDeleteUser, useSession, useUpdateUser, useUsers } from '@/api/queries'
import type { Role, User } from '@/api/types'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatRelative } from '@/utils/format'

function RoleSelect({ value, onChange, disabled }: { value: Role; onChange: (role: Role) => void; disabled?: boolean }) {
  return (
    <Select value={value} onValueChange={(next) => onChange(next as Role)} disabled={disabled}>
      <SelectTrigger className="w-28" aria-label="Role">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="viewer">viewer</SelectItem>
        <SelectItem value="admin">admin</SelectItem>
      </SelectContent>
    </Select>
  )
}

function UserRow({ user, self }: { user: User; self: boolean }) {
  const update = useUpdateUser()
  const remove = useDeleteUser()
  return (
    <TableRow>
      <TableCell className="font-medium">
        {user.username}
        {self ? <span className="ml-1 text-xs text-muted-foreground">(you)</span> : null}
      </TableCell>
      <TableCell>
        <RoleSelect value={user.role} onChange={(role) => update.mutate({ id: user.id, role })} disabled={update.isPending} />
      </TableCell>
      <TableCell className="text-xs text-muted-foreground" title={user.created_at}>
        {formatRelative(user.created_at)}
      </TableCell>
      <TableCell className="text-right">
        <Button
          size="icon"
          variant="ghost"
          aria-label={`Delete ${user.username}`}
          disabled={self || remove.isPending}
          onClick={() => window.confirm(`Delete ${user.username}?`) && remove.mutate(user.id)}
        >
          <Trash2 className="size-4" />
        </Button>
        {update.isError ? <div className="text-xs text-destructive">{(update.error as Error).message}</div> : null}
        {remove.isError ? <div className="text-xs text-destructive">{(remove.error as Error).message}</div> : null}
      </TableCell>
    </TableRow>
  )
}

export function UsersPage() {
  const users = useUsers()
  const session = useSession()
  const create = useCreateUser()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<Role>('viewer')

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    create.mutate(
      { username, password, role },
      {
        onSuccess: () => {
          setUsername('')
          setPassword('')
          setRole('viewer')
        },
      },
    )
  }

  return (
    <>
      <PageHeader title="Users" description="Viewers can browse everything. Admins can also run jobs, pin runs, and manage users." />
      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="overflow-x-auto rounded-xl border">
          {users.isLoading ? <Skeleton className="h-40 w-full" /> : null}
          {users.isError ? <ErrorState error={users.error} /> : null}
          {users.data ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Username</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right">Remove</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {users.data.map((user) => (
                  <UserRow key={user.id} user={user} self={user.id === session.data?.user.id} />
                ))}
              </TableBody>
            </Table>
          ) : null}
        </div>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Add user</CardTitle>
            <CardDescription>Passwords need at least 8 characters.</CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-3" onSubmit={onSubmit}>
              <div className="space-y-1.5">
                <Label htmlFor="new-username">Username</Label>
                <Input id="new-username" value={username} onChange={(e) => setUsername(e.target.value)} autoCapitalize="none" required />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="new-password">Password</Label>
                <Input id="new-password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="new-password" required />
              </div>
              <div className="space-y-1.5">
                <Label>Role</Label>
                <RoleSelect value={role} onChange={setRole} />
              </div>
              {create.isError ? <ErrorState error={create.error} title="Could not create user" /> : null}
              <Button type="submit" disabled={create.isPending || !username || password.length < 8}>
                Create
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </>
  )
}
