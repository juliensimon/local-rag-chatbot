/**
 * Control panel container for all RAG controls
 */

import { Card, CardContent } from '@/components/ui/card'
import { RagToggle } from './RagToggle'
import { SearchTypeSelector } from './SearchTypeSelector'
import { DocumentFilter } from './DocumentFilter'
import { AdvancedOptions } from './AdvancedOptions'

export function ControlPanel() {
  return (
    <Card>
      <CardContent className="space-y-4 p-4">
        {/* Main controls row */}
        <div className="flex flex-wrap items-start gap-6">
          <RagToggle />
          <SearchTypeSelector />
          <DocumentFilter />
        </div>

        {/* Advanced options */}
        <AdvancedOptions />
      </CardContent>
    </Card>
  )
}
