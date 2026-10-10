import { vi } from 'vitest'

/** A stubbed route's answer: a JSON body (status 200), or a status with an API error body. */
export type StubAnswer = unknown | { status: number; code: string; message?: string }

function isError(answer: StubAnswer): answer is { status: number; code: string; message?: string } {
  return typeof answer === 'object' && answer !== null && 'status' in answer && 'code' in answer
}

/**
 * Replace `fetch` with one that answers by path (the query string ignored): each listed path gets
 * its body, and any other path the API's 404. Returns the mock, so a test can read the calls.
 */
export function stubApi(routes: Record<string, StubAnswer>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), 'http://localhost')
    const answer = routes[url.pathname]
    if (answer === undefined) {
      return new Response(JSON.stringify({ error: { code: 'not_found', message: `No stub for ${url.pathname}` } }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' },
      })
    }
    if (isError(answer)) {
      return new Response(JSON.stringify({ error: { code: answer.code, message: answer.message ?? answer.code } }), {
        status: answer.status,
        headers: { 'Content-Type': 'application/json' },
      })
    }
    return new Response(JSON.stringify(answer), { status: 200, headers: { 'Content-Type': 'application/json' } })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}
