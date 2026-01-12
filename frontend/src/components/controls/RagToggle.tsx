/**
 * RAG mode toggle switch
 */

import { Search } from 'lucide-react'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { useSettings } from '@/context/SettingsContext'

export function RagToggle() {
  const { ragEnabled, setRagEnabled } = useSettings()

  return (
    <div className="flex items-center gap-2">
      <Switch
        id="rag-mode"
        checked={ragEnabled}
        onCheckedChange={setRagEnabled}
        aria-label="Enable RAG mode"
      />
      <Label
        htmlFor="rag-mode"
        className="flex cursor-pointer items-center gap-1.5 text-sm"
      >
        <Search className="h-4 w-4" />
        RAG Mode
      </Label>
    </div>
  )
}
