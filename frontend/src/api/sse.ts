/**
 * SSE streaming utilities for chat endpoint
 */

import type { ChatRequest, ContextResponse, SSEEvent } from '@/types/api'
import { api } from './client'
import { parseSSEBuffer } from '@/lib/sse-parser'

export interface StreamCallbacks {
  onToken: (content: string) => void
  onContext: (context: ContextResponse) => void
  onDone: () => void
  onError: (error: string) => void
}

/**
 * Stream chat response with SSE
 * @param request Chat request parameters
 * @param callbacks Event callbacks
 * @param signal AbortSignal for cancellation
 */
export async function streamChat(
  request: ChatRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const response = await api.chatStream(request, signal)
  const reader = response.body?.getReader()

  if (!reader) {
    callbacks.onError('No response body')
    return
  }

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        // Process any remaining buffer
        if (buffer.trim()) {
          const { events } = parseSSEBuffer(buffer + '\n\n')
          processEvents(events, callbacks)
        }
        break
      }

      buffer += decoder.decode(value, { stream: true })
      const { events, remaining } = parseSSEBuffer(buffer)
      buffer = remaining

      processEvents(events, callbacks)
    }
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        // Stream was cancelled, not an error
        return
      }
      callbacks.onError(error.message)
    } else {
      callbacks.onError('Unknown streaming error')
    }
  } finally {
    reader.releaseLock()
  }
}

function processEvents(events: SSEEvent[], callbacks: StreamCallbacks): void {
  for (const event of events) {
    switch (event.type) {
      case 'token':
        if ('content' in event.data) {
          callbacks.onToken(event.data.content)
        }
        break
      case 'context':
        callbacks.onContext(event.data as ContextResponse)
        break
      case 'done':
        callbacks.onDone()
        break
      case 'error':
        if ('message' in event.data) {
          callbacks.onError(event.data.message)
        }
        break
    }
  }
}
