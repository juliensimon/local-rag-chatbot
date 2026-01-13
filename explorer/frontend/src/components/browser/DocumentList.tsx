/**
 * Document list component
 */

import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { DocumentResponse } from '@/types/explorer'
import { FileText } from 'lucide-react'
import { DocumentViewer } from './DocumentViewer'
import { useState } from 'react'

interface DocumentListProps {
  documents: DocumentResponse[]
}

export function DocumentList({ documents }: DocumentListProps) {
  const [selectedDoc, setSelectedDoc] = useState<DocumentResponse | null>(null)

  if (selectedDoc) {
    return <DocumentViewer document={selectedDoc} onBack={() => setSelectedDoc(null)} />
  }

  return (
    <Card>
      <CardContent className="pt-6">
        <ScrollArea className="h-[600px]">
          <div className="space-y-4">
            {documents.map((doc) => (
              <div
                key={doc.doc_id}
                className="rounded-lg border p-4 space-y-2 cursor-pointer hover:bg-accent transition-colors"
                onClick={() => setSelectedDoc(doc)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <span className="font-semibold">{doc.source}</span>
                    <Badge variant="outline">Page {doc.page}</Badge>
                    <Badge variant="secondary">{doc.chunk_size} chars</Badge>
                  </div>
                </div>

                <div className="text-sm text-muted-foreground bg-muted p-3 rounded">
                  {doc.content.length > 300
                    ? `${doc.content.substring(0, 300)}...`
                    : doc.content}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
