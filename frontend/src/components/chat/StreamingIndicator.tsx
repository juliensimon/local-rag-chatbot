/**
 * Streaming response indicator (typing animation)
 */

import { Bot } from 'lucide-react'

interface StreamingIndicatorProps {
  content: string
}

export function StreamingIndicator({ content }: StreamingIndicatorProps) {
  return (
    <div className="flex gap-3 p-4">
      {/* Avatar */}
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted">
        <Bot className="h-4 w-4" />
      </div>

      {/* Message content */}
      <div className="flex max-w-[80%] flex-col gap-1 items-start">
        <div className="rounded-lg px-4 py-2 text-sm bg-muted text-foreground">
          <div className="whitespace-pre-wrap break-words">
            {content}
            <span className="ml-1 inline-block h-4 w-1 animate-pulse bg-current" />
          </div>
        </div>
      </div>
    </div>
  )
}
