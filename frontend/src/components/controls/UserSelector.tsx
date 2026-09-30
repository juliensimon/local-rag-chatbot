/**
 * User ID input selecting whose document collection RAG searches
 */

import { useEffect, useState } from 'react'
import { Input } from '@/components/ui/input'
import { User } from 'lucide-react'
import { useSettings } from '@/context/SettingsContext'
import { useSources } from '@/hooks/useSources'
import { USER_ID_PATTERN } from '@/types/api'
import { cn } from '@/lib/utils'

export function UserSelector() {
  const { userId, setUserId, ragEnabled } = useSettings()
  const { isError } = useSources(userId)
  // Edits stay local until committed: selecting a user may index their PDFs server-side
  const [draft, setDraft] = useState(userId ?? '')

  useEffect(() => {
    setDraft(userId ?? '')
  }, [userId])

  if (!ragEnabled) return null

  const trimmed = draft.trim()
  const invalid = trimmed !== '' && !USER_ID_PATTERN.test(trimmed)

  const commit = () => {
    if (!invalid) setUserId(trimmed || null)
  }

  return (
    <div className="space-y-2">
      <label htmlFor="user-id" className="flex items-center gap-1 text-sm font-medium">
        <User className="h-4 w-4" />
        Collection
      </label>
      <Input
        id="user-id"
        className={cn('w-48', invalid && 'border-destructive')}
        placeholder="Shared (enter user ID)"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === 'Enter' && commit()}
        aria-invalid={invalid}
        aria-describedby="user-id-error"
      />
      <p id="user-id-error" className="text-xs text-destructive">
        {invalid
          ? 'Letters, digits, - and _ only'
          : isError && userId
            ? `No documents for '${userId}'`
            : null}
      </p>
    </div>
  )
}
