/**
 * Markdown rendering component using react-markdown
 */

import ReactMarkdown from 'react-markdown'
import { cn } from '@/lib/utils'

interface MarkdownProps {
  content: string
  className?: string
}

export function Markdown({ content, className }: MarkdownProps) {
  return (
    <div
      className={cn(
        'prose prose-sm dark:prose-invert max-w-none',
        // Tighter spacing for chat context
        'prose-p:my-1 prose-headings:my-2',
        // Code styling using shadcn theme colors
        'prose-pre:bg-muted prose-pre:text-foreground',
        'prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none',
        // List styling
        'prose-ul:my-1 prose-ol:my-1 prose-li:my-0',
        className
      )}
    >
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  )
}
