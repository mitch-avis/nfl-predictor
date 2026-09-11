import { useState } from 'react'

import type { JobParams, JobTemplate, ParamSpec } from '@/api/types'
import { InfoTooltip } from '@/components/common/InfoTooltip'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'

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

function Field({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec
  value: string | number | boolean
  onChange: (next: string | number | boolean) => void
}) {
  const id = `param-${spec.name}`
  const label = (
    <Label htmlFor={id} className="flex items-center gap-1">
      {spec.label}
      {spec.required ? <span className="text-destructive">*</span> : null}
      {spec.description ? <InfoTooltip content={spec.description} /> : null}
    </Label>
  )
  if (spec.kind === 'bool') {
    return (
      <div className="flex items-center justify-between gap-3 py-1">
        {label}
        <Switch id={id} checked={value === true} onCheckedChange={onChange} />
      </div>
    )
  }
  if (spec.kind === 'choice') {
    return (
      <div className="grid gap-1.5">
        {label}
        <Select value={String(value)} onValueChange={onChange}>
          <SelectTrigger id={id}>
            <SelectValue placeholder="Choose" />
          </SelectTrigger>
          <SelectContent>
            {spec.choices.map((choice) => (
              <SelectItem key={choice} value={choice}>
                {choice}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    )
  }
  const numeric = spec.kind === 'int' || spec.kind === 'float'
  return (
    <div className="grid gap-1.5">
      {label}
      <Input
        id={id}
        type={numeric ? 'number' : 'text'}
        inputMode={numeric ? 'numeric' : undefined}
        step={spec.kind === 'float' ? 'any' : undefined}
        min={spec.minimum ?? undefined}
        max={spec.maximum ?? undefined}
        value={String(value)}
        required={spec.required}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  )
}

/** A form generated from a template's parameter schema. */
export function JobForm({
  template,
  preset,
  pending,
  onSubmit,
  submitLabel = 'Run job',
}: {
  template: JobTemplate
  preset?: JobParams
  pending?: boolean
  onSubmit: (params: JobParams) => void
  submitLabel?: string
}) {
  const [values, setValues] = useState<JobParams>(() => initialValues(template, preset))
  return (
    <form
      className="grid gap-3"
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit(submittableValues(values))
      }}
    >
      {template.params.map((spec) => (
        <Field
          key={spec.name}
          spec={spec}
          value={values[spec.name] ?? ''}
          onChange={(next) => setValues((current) => ({ ...current, [spec.name]: next }))}
        />
      ))}
      {template.params.length === 0 ? (
        <p className="text-sm text-muted-foreground">This job takes no options.</p>
      ) : null}
      <Button type="submit" disabled={pending} className="mt-1 justify-self-start">
        {submitLabel}
      </Button>
    </form>
  )
}
