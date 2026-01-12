/**
 * Server-Sent Events parser
 * Handles SSE format: "event: <type>\ndata: <json>\n\n"
 */

import type { SSEEvent, SSEEventType, SSEEventData } from '@/types/api'

interface ParseResult {
  events: SSEEvent[]
  remaining: string
}

/**
 * Parse SSE buffer into events
 * Handles partial events across chunks
 */
export function parseSSEBuffer(buffer: string): ParseResult {
  const events: SSEEvent[] = []
  const lines = buffer.split('\n')
  let remaining = ''

  let currentEvent: string | null = null
  let currentData: string | null = null

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // Check if this is the last line and buffer doesn't end with newline
    // This means we have an incomplete line
    if (i === lines.length - 1 && !buffer.endsWith('\n')) {
      remaining = line
      break
    }

    if (line.startsWith('event: ')) {
      currentEvent = line.slice(7).trim()
    } else if (line.startsWith('data: ')) {
      currentData = line.slice(6)
    } else if (line === '' && currentEvent !== null && currentData !== null) {
      // Empty line marks end of event
      try {
        const parsedData = JSON.parse(currentData) as SSEEventData
        events.push({
          type: currentEvent as SSEEventType,
          data: parsedData,
        })
      } catch {
        // Skip malformed JSON
        console.warn('Failed to parse SSE data:', currentData)
      }
      currentEvent = null
      currentData = null
    }
  }

  // If we have partial event data, add it to remaining
  if (currentEvent !== null || currentData !== null) {
    const partialLines: string[] = []
    if (currentEvent !== null) {
      partialLines.push(`event: ${currentEvent}`)
    }
    if (currentData !== null) {
      partialLines.push(`data: ${currentData}`)
    }
    remaining = partialLines.join('\n') + (remaining ? '\n' + remaining : '')
  }

  return { events, remaining }
}
