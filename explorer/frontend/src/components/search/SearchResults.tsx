/**
 * Search results display component
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { DocumentScore } from '@/types/explorer'
import { FileText } from 'lucide-react'

interface SearchResultsProps {
  results: DocumentScore[]
  query: string
  searchType: string
}

export function SearchResults({ results, query, searchType }: SearchResultsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Search Results</CardTitle>
        <CardDescription>
          Found {results.length} results for "{query}" using {searchType} search
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[600px]">
          <div className="space-y-4">
            {results.map((result, idx) => (
              <div key={idx} className="rounded-lg border p-4 space-y-2">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <span className="font-semibold">{result.source}</span>
                    <Badge variant="outline">Page {result.page}</Badge>
                  </div>
                  {result.score !== null && (
                    <Badge variant="secondary">
                      Score: {result.score.toFixed(3)}
                    </Badge>
                  )}
                </div>

                {result.semantic_score !== null && result.keyword_score !== null && (
                  <div className="flex gap-4 text-xs text-muted-foreground">
                    <span>Semantic: {result.semantic_score.toFixed(3)}</span>
                    <span>Keyword: {result.keyword_score.toFixed(3)}</span>
                  </div>
                )}

                <div className="text-sm text-muted-foreground bg-muted p-3 rounded">
                  {result.content.length > 500
                    ? `${result.content.substring(0, 500)}...`
                    : result.content}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
