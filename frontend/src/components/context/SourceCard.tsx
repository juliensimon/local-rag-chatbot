/**
 * Card displaying a retrieved source document
 */

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Star, FileText, ChevronDown, ChevronUp, Copy, Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { SourceDocument } from '@/types/api'
import { Markdown } from '@/components/ui/markdown'

interface SourceCardProps {
  source: SourceDocument
  searchType?: 'mmr' | 'similarity' | 'hybrid'
}

function formatScore(score: number | null | undefined): string {
  if (score == null) return '-'
  return score.toFixed(2)
}

export function SourceCard({ source, searchType }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(source.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Truncate content for preview
  const previewLength = 200
  const needsTruncation = source.content.length > previewLength
  const displayContent = expanded
    ? source.content
    : source.content.slice(0, previewLength) + (needsTruncation ? '...' : '')

  return (
    <Card className={cn('transition-shadow', source.is_top && 'ring-2 ring-primary/50')}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            {source.is_top && (
              <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
            )}
            <FileText className="h-4 w-4 text-muted-foreground" />
            <span className="truncate">{source.source}</span>
          </CardTitle>
          <Badge variant="outline" className="shrink-0 text-xs">
            p.{source.page}
          </Badge>
        </div>

        {/* Score display */}
        <div className="flex flex-wrap gap-1.5 text-xs text-muted-foreground">
          {searchType === 'hybrid' ? (
            <>
              {source.score != null && (
                <Badge variant="secondary" className="text-xs">
                  f:{formatScore(source.score)}
                </Badge>
              )}
              {source.semantic_score != null && (
                <Badge variant="secondary" className="text-xs">
                  s:{formatScore(source.semantic_score)}
                </Badge>
              )}
              {source.keyword_score != null && (
                <Badge variant="secondary" className="text-xs">
                  k:{formatScore(source.keyword_score)}
                </Badge>
              )}
            </>
          ) : (
            source.score != null && (
              <Badge variant="secondary" className="text-xs">
                rel:{formatScore(source.score)}
              </Badge>
            )
          )}
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <Markdown
          content={displayContent}
          className="text-muted-foreground"
        />

        <div className="mt-2 flex items-center gap-2">
          {needsTruncation && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(!expanded)}
              className="h-7 text-xs"
            >
              {expanded ? (
                <>
                  <ChevronUp className="mr-1 h-3 w-3" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="mr-1 h-3 w-3" />
                  Show more
                </>
              )}
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="h-7 text-xs"
          >
            {copied ? (
              <>
                <Check className="mr-1 h-3 w-3" />
                Copied
              </>
            ) : (
              <>
                <Copy className="mr-1 h-3 w-3" />
                Copy
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
