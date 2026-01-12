/**
 * Search type radio selector
 */

import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { HelpCircle } from 'lucide-react'
import { useSettings } from '@/context/SettingsContext'
import type { SearchType } from '@/types/api'

const SEARCH_TYPES: { value: SearchType; label: string; description: string }[] = [
  {
    value: 'mmr',
    label: 'MMR',
    description: 'Maximal Marginal Relevance - balances relevance and diversity',
  },
  {
    value: 'similarity',
    label: 'Similarity',
    description: 'Pure vector similarity search - most relevant results',
  },
  {
    value: 'hybrid',
    label: 'Hybrid',
    description: 'Combines semantic and keyword search for best coverage',
  },
]

export function SearchTypeSelector() {
  const { searchType, setSearchType, ragEnabled } = useSettings()

  if (!ragEnabled) return null

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1 text-sm font-medium">
        Search Type
        <Tooltip>
          <TooltipTrigger asChild>
            <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />
          </TooltipTrigger>
          <TooltipContent>
            <p className="max-w-xs">
              Choose how documents are retrieved for answering your question.
            </p>
          </TooltipContent>
        </Tooltip>
      </div>
      <RadioGroup
        value={searchType}
        onValueChange={(value) => setSearchType(value as SearchType)}
        className="flex flex-wrap gap-4"
        aria-label="Search type selection"
      >
        {SEARCH_TYPES.map((type) => (
          <div key={type.value} className="flex items-center gap-2">
            <RadioGroupItem value={type.value} id={`search-${type.value}`} />
            <Label
              htmlFor={`search-${type.value}`}
              className="cursor-pointer text-sm"
            >
              <Tooltip>
                <TooltipTrigger asChild>
                  <span>{type.label}</span>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">{type.description}</p>
                </TooltipContent>
              </Tooltip>
            </Label>
          </div>
        ))}
      </RadioGroup>
    </div>
  )
}
