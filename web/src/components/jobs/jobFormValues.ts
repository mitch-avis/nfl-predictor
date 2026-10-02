import type { JobParams, JobTemplate } from '@/api/types'

/** Initial form values: whatever the caller passed, else the template's declared defaults. */
export function initialValues(template: JobTemplate, preset: JobParams = {}): JobParams {
  const values: JobParams = {}
  for (const spec of template.params) {
    const supplied = preset[spec.name]
    if (supplied !== undefined) values[spec.name] = supplied
    else if (spec.default !== null) values[spec.name] = spec.default
    else if (spec.kind === 'bool') values[spec.name] = false
    else values[spec.name] = ''
  }
  return values
}

/** Drop blanks so the backend applies its own defaults rather than rejecting empty strings. */
export function submittableValues(values: JobParams): JobParams {
  return Object.fromEntries(Object.entries(values).filter(([, value]) => value !== ''))
}
