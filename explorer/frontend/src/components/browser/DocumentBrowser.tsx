/**
 * Document browser component with filtering and pagination
 */

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { explorerAPI } from '@/api/client'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { DocumentList } from './DocumentList'
import { Skeleton } from '@/components/ui/skeleton'
import { AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react'

export function DocumentBrowser() {
  const [page, setPage] = useState(1)
  const [pageSize] = useState(100)
  const [sourceFilter, setSourceFilter] = useState<string>('')
  const [minPage, setMinPage] = useState<number | undefined>(undefined)
  const [maxPage, setMaxPage] = useState<number | undefined>(undefined)
  const [minChunkSize, setMinChunkSize] = useState<number | undefined>(undefined)
  const [maxChunkSize, setMaxChunkSize] = useState<number | undefined>(undefined)

  const { data, isLoading, error } = useQuery({
    queryKey: ['documents', page, pageSize, sourceFilter, minPage, maxPage, minChunkSize, maxChunkSize],
    queryFn: () =>
      explorerAPI.getDocuments({
        page,
        page_size: pageSize,
        source: sourceFilter || undefined,
        min_page: minPage,
        max_page: maxPage,
        min_chunk_size: minChunkSize,
        max_chunk_size: maxChunkSize,
      }),
  })

  const handleReset = () => {
    setSourceFilter('')
    setMinPage(undefined)
    setMaxPage(undefined)
    setMinChunkSize(undefined)
    setMaxChunkSize(undefined)
    setPage(1)
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-destructive" />
            Error Loading Documents
          </CardTitle>
          <CardDescription>
            {error instanceof Error ? error.message : 'Failed to load documents'}
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Browse Documents</CardTitle>
          <CardDescription>Filter and browse document chunks in the vector store</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>Source Filter</Label>
                <Input
                  placeholder="Filter by source filename..."
                  value={sourceFilter}
                  onChange={(e) => {
                    setSourceFilter(e.target.value)
                    setPage(1)
                  }}
                />
              </div>

              <div className="space-y-2">
                <Label>Min Page</Label>
                <Input
                  type="number"
                  placeholder="Min page number"
                  value={minPage ?? ''}
                  onChange={(e) => {
                    const val = e.target.value ? parseInt(e.target.value) : undefined
                    setMinPage(val)
                    setPage(1)
                  }}
                />
              </div>

              <div className="space-y-2">
                <Label>Max Page</Label>
                <Input
                  type="number"
                  placeholder="Max page number"
                  value={maxPage ?? ''}
                  onChange={(e) => {
                    const val = e.target.value ? parseInt(e.target.value) : undefined
                    setMaxPage(val)
                    setPage(1)
                  }}
                />
              </div>

              <div className="space-y-2">
                <Label>Min Chunk Size</Label>
                <Input
                  type="number"
                  placeholder="Min characters"
                  value={minChunkSize ?? ''}
                  onChange={(e) => {
                    const val = e.target.value ? parseInt(e.target.value) : undefined
                    setMinChunkSize(val)
                    setPage(1)
                  }}
                />
              </div>

              <div className="space-y-2">
                <Label>Max Chunk Size</Label>
                <Input
                  type="number"
                  placeholder="Max characters"
                  value={maxChunkSize ?? ''}
                  onChange={(e) => {
                    const val = e.target.value ? parseInt(e.target.value) : undefined
                    setMaxChunkSize(val)
                    setPage(1)
                  }}
                />
              </div>

              <div className="space-y-2 flex items-end">
                <Button variant="outline" onClick={handleReset} className="w-full">
                  Reset Filters
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      ) : data ? (
        <>
          <DocumentList documents={data.documents} />
          {data.total_pages > 1 && (
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-muted-foreground">
                    Page {data.page} of {data.total_pages} ({data.total} total documents)
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page === 1}
                    >
                      <ChevronLeft className="h-4 w-4" />
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
                      disabled={page === data.total_pages}
                    >
                      Next
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      ) : null}
    </div>
  )
}
