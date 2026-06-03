/**
 * Persistent memory (mem0) toggle switch
 */

import { Brain } from 'lucide-react'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { useSettings } from '@/context/SettingsContext'

export function MemoryToggle() {
  const { memoryEnabled, setMemoryEnabled } = useSettings()

  return (
    <div className="flex items-center gap-2">
      <Switch
        id="memory-mode"
        checked={memoryEnabled}
        onCheckedChange={setMemoryEnabled}
        aria-label="Enable persistent memory"
      />
      <Label
        htmlFor="memory-mode"
        className="flex cursor-pointer items-center gap-1.5 text-sm"
      >
        <Brain className="h-4 w-4" />
        Memory
      </Label>
    </div>
  )
}
