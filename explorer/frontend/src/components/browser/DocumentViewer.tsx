/**
 * Document viewer component with similar documents
 */

import { useQuery } from '@tanstack/react-query'
import { explorerAPI } from '@/api/client'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import type { DocumentResponse } from '@/types/explorer'
import { ArrowLeft, FileText } from 'lucide-react'
import { Separator } from '@/components/ui/separator'

interface DocumentViewerProps {
  document: DocumentResponse
  onBack: () => void
}

export function DocumentViewer({ document, onBack }: DocumentViewerProps) {
  const { data: similar, isLoading } = useQuery({
    queryKey: ['similar', document.doc_id],
    queryFn: () => explorerAPI.getSimilarDocuments(document.doc_id, 5),
  })

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                {document.source}
              </CardTitle>
              <CardDescription className="mt-1">
                Page {document.page} • {document.chunk_size} characters
              </CardDescription>
            </div>
            <Button variant="outline" onClick={onBack}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <p className="whitespace-pre-wrap">{document.content}</p>
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {isLoading ? (
        <Card>
          <CardHeader>
            <CardTitle>Similar Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <Skeleton className="h-32 w-full" />
          </CardContent>
        </Card>
      ) : similar && similar.similar_docs.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Similar Documents</CardTitle>
            <CardDescription>Documents with similar content</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[300px]">
              <div className="space-y-4">
                {similar.similar_docs.map((doc, idx) => (
                  <div key={idx}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <span className="font-semibold">{doc.source}</span>
                        <Badge variant="outline">Page {doc.page}</Badge>
                      </div>
                      {doc.score !== null && (
                        <Badge variant="secondary">Similarity: {doc.score.toFixed(3)}</Badge>
                      )}
                    </div>
                    <div className="text-sm text-muted-foreground bg-muted p-3 rounded">
                      {doc.content.length > 200
                        ? `${doc.content.substring(0, 200)}...`
                        : doc.content}
                    </div>
                    {idx < similar.similar_docs.length - 1 && (
                      <Separator className="mt-4" />
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}
