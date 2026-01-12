/**
 * Scrollable list of chat messages
 */

import { useEffect, useRef } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ChatMessage } from './ChatMessage'
import { StreamingIndicator } from './StreamingIndicator'
import type { Message } from '@/types/api'

interface MessageListProps {
  messages: Message[]
  isStreaming: boolean
  partialResponse: string
}

export function MessageList({
  messages,
  isStreaming,
  partialResponse,
}: MessageListProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, partialResponse])

  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="flex flex-1 items-center justify-center text-muted-foreground">
        <p className="text-center">
          Start a conversation by typing a message below.
        </p>
      </div>
    )
  }

  return (
    <ScrollArea className="flex-1" ref={scrollRef}>
      <div className="flex flex-col">
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
        {isStreaming && partialResponse && (
          <StreamingIndicator content={partialResponse} />
        )}
      </div>
    </ScrollArea>
  )
}
