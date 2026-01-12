/**
 * Hybrid search balance slider
 */

import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import { useSettings } from '@/context/SettingsContext'

export function HybridAlphaSlider() {
  const { hybridAlpha, setHybridAlpha, searchType, ragEnabled } = useSettings()

  if (!ragEnabled || searchType !== 'hybrid') return null

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">Search Balance</Label>
        <span className="text-xs text-muted-foreground">{hybridAlpha}%</span>
      </div>
      <div className="space-y-1">
        <Slider
          value={[hybridAlpha]}
          onValueChange={([value]) => setHybridAlpha(value)}
          min={0}
          max={100}
          step={5}
          aria-label="Hybrid search balance"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Keyword</span>
          <span>Semantic</span>
        </div>
      </div>
    </div>
  )
}
