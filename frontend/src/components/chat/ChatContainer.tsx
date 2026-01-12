/**
 * Main chat container assembling all chat components
 */

import { Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'
import { ExampleQuestions } from './ExampleQuestions'
import { useChat } from '@/context/ChatContext'

export function ChatContainer() {
  const {
    messages,
    isStreaming,
    partialResponse,
    error,
    sendMessage,
    clearChat,
    cancelStream,
  } = useChat()

  const handleExampleSelect = (question: string) => {
    sendMessage(question)
  }

  return (
    <Card className="flex h-[calc(100vh-12rem)] flex-col">
      {/* Header with clear button */}
      <div className="flex items-center justify-between border-b px-4 py-2">
        <h2 className="text-sm font-medium">Chat</h2>
        <Button
          variant="ghost"
          size="sm"
          onClick={clearChat}
          disabled={messages.length === 0 && !isStreaming}
          className="text-muted-foreground hover:text-foreground"
        >
          <Trash2 className="mr-2 h-4 w-4" />
          Clear
        </Button>
      </div>

      {/* Message list */}
      <MessageList
        messages={messages}
        isStreaming={isStreaming}
        partialResponse={partialResponse}
      />

      {/* Error display */}
      {error && (
        <div className="mx-4 mb-2 rounded-md bg-destructive/10 p-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Example questions */}
      <div className="border-t px-4 py-2">
        <ExampleQuestions
          onSelect={handleExampleSelect}
          disabled={isStreaming}
        />
      </div>

      {/* Input */}
      <ChatInput
        onSend={sendMessage}
        onCancel={cancelStream}
        isStreaming={isStreaming}
      />
    </Card>
  )
}
