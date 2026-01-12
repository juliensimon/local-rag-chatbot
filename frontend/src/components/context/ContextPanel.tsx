/**
 * Context panel showing retrieved documents
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { FileSearch, HelpCircle } from 'lucide-react'
import { SourceCard } from './SourceCard'
import { RewrittenQuery } from './RewrittenQuery'
import { useChat } from '@/context/ChatContext'
import { useSettings } from '@/context/SettingsContext'

export function ContextPanel() {
  const { context } = useChat()
  const { ragEnabled, searchType } = useSettings()

  if (!ragEnabled) {
    return (
      <Card className="h-full">
        <CardContent className="flex h-full items-center justify-center p-6">
          <p className="text-center text-sm text-muted-foreground">
            Enable RAG mode to see retrieved context here.
          </p>
        </CardContent>
      </Card>
    )
  }

  if (!context || context.sources.length === 0) {
    return (
      <Card className="h-full">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <FileSearch className="h-4 w-4" />
            Retrieved Context
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-1 items-center justify-center">
          <p className="text-center text-sm text-muted-foreground">
            Ask a question to see retrieved documents.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <FileSearch className="h-4 w-4" />
          Retrieved Context
          <span className="text-xs font-normal text-muted-foreground">
            ({context.sources.length} sources)
          </span>
          {searchType === 'hybrid' && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-xs">
                  <p className="font-medium mb-1">Score Legend</p>
                  <ul className="text-xs space-y-1">
                    <li><strong>f:</strong> Fused score (combined ranking)</li>
                    <li><strong>s:</strong> Semantic score (vector similarity)</li>
                    <li><strong>k:</strong> Keyword score (BM25 text match)</li>
                  </ul>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        <ScrollArea className="h-full px-4 pb-4">
          <div className="space-y-3">
            {/* Rewritten query if present */}
            {context.rewritten_query && (
              <RewrittenQuery query={context.rewritten_query} />
            )}

            {/* Source documents */}
            {context.sources.map((source, index) => (
              <SourceCard
                key={`${source.source}-${source.page}-${index}`}
                source={source}
                searchType={searchType}
              />
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
