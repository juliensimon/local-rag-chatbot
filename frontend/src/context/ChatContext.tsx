/**
 * Chat context for message state and streaming
 */

import { createContext, useContext, useState, useCallback, useRef } from 'react'
import type { Message, ContextResponse, ChatRequest } from '@/types/api'
import { streamChat } from '@/api/sse'
import { useSettings } from './SettingsContext'

interface ChatContextValue {
  messages: Message[]
  isStreaming: boolean
  partialResponse: string
  context: ContextResponse | null
  error: string | null
  sendMessage: (content: string) => Promise<void>
  clearChat: () => void
  cancelStream: () => void
}

const ChatContext = createContext<ChatContextValue | undefined>(undefined)

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [partialResponse, setPartialResponse] = useState('')
  const [context, setContext] = useState<ContextResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const abortControllerRef = useRef<AbortController | null>(null)
  const settings = useSettings()

  const sendMessage = useCallback(async (content: string) => {
    if (isStreaming || !content.trim()) return

    // Clear any previous error
    setError(null)

    // Add user message
    const userMessage: Message = { role: 'user', content: content.trim() }
    setMessages((prev) => [...prev, userMessage])

    // Prepare for streaming
    setIsStreaming(true)
    setPartialResponse('')
    setContext(null)

    // Create abort controller for cancellation
    abortControllerRef.current = new AbortController()

    // Build chat request
    const request: ChatRequest = {
      message: content.trim(),
      history: messages,
      rag_enabled: settings.ragEnabled,
      search_type: settings.searchType,
      doc_filter: settings.docFilter,
      use_query_rewriting: settings.useQueryRewriting,
      use_reranking: settings.useReranking,
      hybrid_alpha: settings.hybridAlpha,
    }

    let fullResponse = ''

    try {
      await streamChat(
        request,
        {
          onToken: (token) => {
            fullResponse += token
            setPartialResponse(fullResponse)
          },
          onContext: (ctx) => {
            setContext(ctx)
          },
          onDone: () => {
            // Add assistant message when complete
            const assistantMessage: Message = {
              role: 'assistant',
              content: fullResponse,
            }
            setMessages((prev) => [...prev, assistantMessage])
            setPartialResponse('')
            setIsStreaming(false)
          },
          onError: (errorMsg) => {
            setError(errorMsg)
            setIsStreaming(false)
            // Still add partial response if we have one
            if (fullResponse) {
              const assistantMessage: Message = {
                role: 'assistant',
                content: fullResponse + '\n\n[Error: ' + errorMsg + ']',
              }
              setMessages((prev) => [...prev, assistantMessage])
            }
            setPartialResponse('')
          },
        },
        abortControllerRef.current.signal
      )
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMsg)
      setIsStreaming(false)
      // Still add partial response if we have one
      if (fullResponse) {
        const assistantMessage: Message = {
          role: 'assistant',
          content: fullResponse,
        }
        setMessages((prev) => [...prev, assistantMessage])
      }
      setPartialResponse('')
    }
  }, [isStreaming, messages, settings])

  const clearChat = useCallback(() => {
    // Cancel any ongoing stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    setMessages([])
    setIsStreaming(false)
    setPartialResponse('')
    setContext(null)
    setError(null)
  }, [])

  const cancelStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    // Keep partial response as final message
    if (partialResponse) {
      const assistantMessage: Message = {
        role: 'assistant',
        content: partialResponse,
      }
      setMessages((prev) => [...prev, assistantMessage])
    }
    setIsStreaming(false)
    setPartialResponse('')
  }, [partialResponse])

  return (
    <ChatContext.Provider
      value={{
        messages,
        isStreaming,
        partialResponse,
        context,
        error,
        sendMessage,
        clearChat,
        cancelStream,
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const context = useContext(ChatContext)
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider')
  }
  return context
}
