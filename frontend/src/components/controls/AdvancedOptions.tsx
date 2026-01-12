/**
 * Advanced options accordion (query rewriting, reranking)
 */

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { Settings, HelpCircle } from 'lucide-react'
import { useSettings } from '@/context/SettingsContext'
import { HybridAlphaSlider } from './HybridAlphaSlider'

export function AdvancedOptions() {
  const {
    ragEnabled,
    useQueryRewriting,
    setUseQueryRewriting,
    useReranking,
    setUseReranking,
  } = useSettings()

  if (!ragEnabled) return null

  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="advanced" className="border-none">
        <AccordionTrigger className="py-2 text-sm text-muted-foreground hover:no-underline">
          <span className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Advanced Options
          </span>
        </AccordionTrigger>
        <AccordionContent>
          <div className="space-y-4 pt-2">
            {/* Query Rewriting */}
            <div className="flex items-center gap-2">
              <Checkbox
                id="query-rewriting"
                checked={useQueryRewriting}
                onCheckedChange={(checked) =>
                  setUseQueryRewriting(checked === true)
                }
              />
              <Label
                htmlFor="query-rewriting"
                className="flex cursor-pointer items-center gap-1 text-sm"
              >
                Query Rewriting
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">
                      Optimizes your query for better retrieval by expanding
                      terms and focusing on key concepts.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </Label>
            </div>

            {/* Re-ranking */}
            <div className="flex items-center gap-2">
              <Checkbox
                id="reranking"
                checked={useReranking}
                onCheckedChange={(checked) => setUseReranking(checked === true)}
              />
              <Label
                htmlFor="reranking"
                className="flex cursor-pointer items-center gap-1 text-sm"
              >
                Re-ranking
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">
                      Uses a cross-encoder to re-score and reorder retrieved
                      documents for higher precision.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </Label>
            </div>

            {/* Hybrid Alpha Slider */}
            <HybridAlphaSlider />
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}
