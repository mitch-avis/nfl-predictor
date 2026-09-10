/**
 * Thin fetch wrapper for the FastAPI backend.
 *
 * Every request carries the session cookie and the `X-Requested-With` header the backend requires
 * on mutating calls. Errors are normalized to `ApiError` so pages can branch on `code`.
 */

export const CSRF_HEADER = { 'X-Requested-With': 'nflp' } as const

export class ApiError extends Error {
  readonly status: number
  readonly code: string

  constructor(status: number, code: string, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

interface ErrorBody {
  error?: { code?: string; message?: string }
  detail?: unknown
}

async function toApiError(response: Response): Promise<ApiError> {
  let body: ErrorBody | null = null
  try {
    body = (await response.json()) as ErrorBody
  } catch {
    body = null
  }
  const code = body?.error?.code ?? (response.status === 422 ? 'validation_error' : 'http_error')
  const message =
    body?.error?.message ??
    (typeof body?.detail === 'string' ? body.detail : null) ??
    `${response.status} ${response.statusText}`
  return new ApiError(response.status, code, message)
}

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE'
  body?: unknown
  signal?: AbortSignal
}

/** Perform a JSON request against `/api/...`; resolves to the parsed body (or `undefined` for 204). */
export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { ...CSRF_HEADER }
  let body: BodyInit | undefined
  if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(options.body)
  }
  const response = await fetch(path, {
    method: options.method ?? 'GET',
    headers,
    body,
    credentials: 'same-origin',
    signal: options.signal,
  })
  if (!response.ok) {
    throw await toApiError(response)
  }
  if (response.status === 204) {
    return undefined as T
  }
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) {
    throw new ApiError(response.status, 'not_json', `The API returned ${contentType || 'no content type'} for ${path}; is the backend running the current code?`)
  }
  return (await response.json()) as T
}

/** Build a query string, dropping empty values. */
export function qs(params: object): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params as Record<string, unknown>)) {
    if (value !== undefined && value !== null && value !== '') search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}
