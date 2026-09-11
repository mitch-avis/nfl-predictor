/** TanStack Query hooks for every backend route. Query keys live here so invalidation is uniform. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { ApiError, apiFetch, qs } from './client'
import type {
  BettingOut,
  DataStatusOut,
  Job,
  JobCatalog,
  JobList,
  JobLogs,
  JobParams,
  JobStatus,
  ModelOut,
  PicksOut,
  PowerOut,
  PredictionsOut,
  Registry,
  WeekRef,
  Role,
  RunDetail,
  RunKind,
  RunList,
  Session,
  User,
} from './types'

export const keys = {
  session: ['session'] as const,
  users: ['users'] as const,
  runs: (kind?: RunKind | 'all') => ['runs', kind ?? 'all'] as const,
  run: (runId: string) => ['run', runId] as const,
  jobCatalog: ['job-catalog'] as const,
  jobs: ['jobs'] as const,
  job: (jobId: string) => ['job', jobId] as const,
}

/** How often job lists refresh while something is still running. */
const JOB_POLL_MS = 2000

export function useSession() {
  return useQuery({
    queryKey: keys.session,
    queryFn: async () => {
      try {
        return await apiFetch<Session>('/api/auth/me')
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) return null
        throw error
      }
    },
    staleTime: 5 * 60 * 1000,
    retry: false,
  })
}

export function useLogin() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (credentials: { username: string; password: string }) =>
      apiFetch<Session>('/api/auth/login', { method: 'POST', body: credentials }),
    onSuccess: (session) => client.setQueryData(keys.session, session),
  })
}

export function useLogout() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: () => apiFetch<void>('/api/auth/logout', { method: 'POST' }),
    onSuccess: () => {
      client.setQueryData(keys.session, null)
      client.removeQueries({ predicate: (query) => query.queryKey[0] !== 'session' })
    },
  })
}

export function useRuns(kind: RunKind | 'all' = 'all') {
  return useQuery({
    queryKey: keys.runs(kind),
    queryFn: () => apiFetch<RunList>(`/api/runs${qs({ kind })}`),
  })
}

export function useRun(runId: string | null) {
  return useQuery({
    queryKey: keys.run(runId ?? ''),
    queryFn: () => apiFetch<RunDetail>(`/api/runs/${encodeURIComponent(runId ?? '')}`),
    enabled: runId !== null,
  })
}

export function useActivateRun() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) =>
      apiFetch<RunDetail>(`/api/runs/${encodeURIComponent(runId)}/activate`, { method: 'POST' }),
    onSuccess: () => client.invalidateQueries({ predicate: () => true }),
  })
}

export function useClearActiveRun() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: () => apiFetch<void>('/api/runs/active', { method: 'DELETE' }),
    onSuccess: () => client.invalidateQueries({ predicate: () => true }),
  })
}

export function useUsers() {
  return useQuery({ queryKey: keys.users, queryFn: () => apiFetch<User[]>('/api/users') })
}

export function useCreateUser() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (payload: { username: string; password: string; role: Role }) =>
      apiFetch<User>('/api/users', { method: 'POST', body: payload }),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.users }),
  })
}

export function useUpdateUser() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...payload }: { id: number; role?: Role; password?: string }) =>
      apiFetch<User>(`/api/users/${id}`, { method: 'PATCH', body: payload }),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.users }),
  })
}

export function useDeleteUser() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => apiFetch<void>(`/api/users/${id}`, { method: 'DELETE' }),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.users }),
  })
}

export interface WeekParams {
  run?: string | null
  season?: number | null
  week?: number | null
  source?: string | null
}

export function useRegistry() {
  return useQuery({
    queryKey: ['registry'],
    queryFn: () => apiFetch<Registry>('/api/registry'),
    staleTime: Infinity,
  })
}

export function usePredictions(params: WeekParams) {
  return useQuery({
    queryKey: ['predictions', params],
    queryFn: () => apiFetch<PredictionsOut>(`/api/predictions${qs(params)}`),
  })
}

/** Every week the selector can offer, including weeks that could still be generated. */
export function useWeeks() {
  return useQuery({
    queryKey: ['prediction-weeks'],
    queryFn: () => apiFetch<WeekRef[]>('/api/predictions/weeks'),
  })
}

export function usePicks(params: WeekParams) {
  return useQuery({
    queryKey: ['picks', params],
    queryFn: () => apiFetch<PicksOut>(`/api/predictions/picks${qs(params)}`),
  })
}

export function useBetting(params: WeekParams) {
  return useQuery({
    queryKey: ['betting', params],
    queryFn: () => apiFetch<BettingOut>(`/api/betting${qs(params)}`),
  })
}

export function usePower(run: string | null) {
  return useQuery({
    queryKey: ['power', run],
    queryFn: () => apiFetch<PowerOut>(`/api/power${qs({ run })}`),
  })
}

export function useModel(run: string | null) {
  return useQuery({
    queryKey: ['model', run],
    queryFn: () => apiFetch<ModelOut>(`/api/model${qs({ run })}`),
  })
}

export function useDataStatus() {
  return useQuery({
    queryKey: ['data-status'],
    queryFn: () => apiFetch<DataStatusOut>('/api/data/status'),
    refetchInterval: 60_000,
  })
}

export function useJobCatalog() {
  return useQuery({ queryKey: keys.jobCatalog, queryFn: () => apiFetch<JobCatalog>('/api/jobs/catalog') })
}

export function useJobs(limit = 50) {
  return useQuery({
    queryKey: [...keys.jobs, limit],
    queryFn: () => apiFetch<JobList>(`/api/jobs${qs({ limit })}`),
    refetchInterval: (query) =>
      query.state.data?.jobs.some((job) => job.status === 'running' || job.status === 'queued')
        ? JOB_POLL_MS
        : false,
  })
}

export function useJob(jobId: string | null, { poll = true }: { poll?: boolean } = {}) {
  return useQuery({
    queryKey: keys.job(jobId ?? ''),
    queryFn: () => apiFetch<Job>(`/api/jobs/${encodeURIComponent(jobId ?? '')}`),
    enabled: jobId !== null,
    refetchInterval: (query) =>
      poll && query.state.data && !isTerminal(query.state.data.status) ? JOB_POLL_MS : false,
  })
}

/** Whether a job has finished, one way or another. */
export function isTerminal(status: JobStatus): boolean {
  return status === 'succeeded' || status === 'failed' || status === 'canceled'
}

export function useCreateJob() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: ({ templateId, params }: { templateId: string; params: JobParams }) =>
      apiFetch<Job>('/api/jobs', { method: 'POST', body: { template_id: templateId, params } }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: keys.jobs })
      void client.invalidateQueries({ queryKey: keys.jobCatalog })
    },
  })
}

export function useCancelJob() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) =>
      apiFetch<Job>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),
    onSuccess: (job) => {
      client.setQueryData(keys.job(job.id), job)
      void client.invalidateQueries({ queryKey: keys.jobs })
    },
  })
}

/** Fetch a page of a job's logs starting after `after`. */
export function fetchJobLogs(jobId: string, after: number) {
  return apiFetch<JobLogs>(`/api/jobs/${encodeURIComponent(jobId)}/logs${qs({ after })}`)
}
