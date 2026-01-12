/**
 * Chat input with send button
 */

import { useState, useRef, useEffect } from 'react'
import { Send, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'

const MAX_LENGTH = 10000

interface ChatInputProps {
  onSend: (message: string) => void
  onCancel: () => void
  isStreaming: boolean
  disabled?: boolean
}

export function ChatInput({
  onSend,
  onCancel,
  isStreaming,
  disabled,
}: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'
    }
  }, [value])

  const handleSubmit = () => {
    if (value.trim() && !isStreaming && !disabled) {
      onSend(value.trim())
      setValue('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const isOverLimit = value.length > MAX_LENGTH
  const canSend = value.trim() && !isOverLimit && !disabled

  return (
    <div className="border-t bg-background p-4">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            disabled={isStreaming || disabled}
            className="min-h-[44px] resize-none pr-12"
            rows={1}
          />
          {/* Character count */}
          {value.length > MAX_LENGTH * 0.8 && (
            <span
              className={`absolute bottom-2 right-2 text-xs ${
                isOverLimit ? 'text-destructive' : 'text-muted-foreground'
              }`}
            >
              {value.length}/{MAX_LENGTH}
            </span>
          )}
        </div>

        {isStreaming ? (
          <Button
            variant="destructive"
            size="icon"
            onClick={onCancel}
            aria-label="Stop generating"
          >
            <Square className="h-4 w-4" />
          </Button>
        ) : (
          <Button
            size="icon"
            onClick={handleSubmit}
            disabled={!canSend}
            aria-label="Send message"
          >
            <Send className="h-4 w-4" />
          </Button>
        )}
      </div>

      {isOverLimit && (
        <p className="mt-1 text-xs text-destructive">
          Message exceeds maximum length of {MAX_LENGTH} characters
        </p>
      )}
    </div>
  )
}
