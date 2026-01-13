/**
 * Search interface component
 */

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { explorerAPI } from '@/api/client'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { SearchResults } from './SearchResults'
import { Search, Loader2 } from 'lucide-react'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'

export function SearchInterface() {
  const [query, setQuery] = useState('')
  const [searchType, setSearchType] = useState<'mmr' | 'similarity' | 'hybrid'>('mmr')
  const [k, setK] = useState(5)
  const [hybridAlpha, setHybridAlpha] = useState(70)

  const searchMutation = useMutation({
    mutationFn: (searchQuery: string) =>
      explorerAPI.search({
        query: searchQuery,
        search_type: searchType,
        k,
        hybrid_alpha: hybridAlpha / 100,
      }),
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      searchMutation.mutate(query)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Search Vector Store</CardTitle>
          <CardDescription>
            Test different search strategies (MMR, similarity, hybrid) to see how they perform
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder="Enter your search query..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="flex-1"
              />
              <Button type="submit" disabled={searchMutation.isPending || !query.trim()}>
                {searchMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Search className="h-4 w-4" />
                )}
                <span className="ml-2">Search</span>
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>Search Type</Label>
                <Select value={searchType} onValueChange={(v) => setSearchType(v as any)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mmr">MMR (Diverse)</SelectItem>
                    <SelectItem value="similarity">Similarity</SelectItem>
                    <SelectItem value="hybrid">Hybrid</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Number of Results (k)</Label>
                <Input
                  type="number"
                  min="1"
                  max="50"
                  value={k}
                  onChange={(e) => setK(parseInt(e.target.value) || 5)}
                />
              </div>

              {searchType === 'hybrid' && (
                <div className="space-y-2">
                  <Label>Hybrid Alpha: {hybridAlpha}% semantic</Label>
                  <Slider
                    value={[hybridAlpha]}
                    onValueChange={(vals) => setHybridAlpha(vals[0])}
                    min={0}
                    max={100}
                    step={5}
                  />
                </div>
              )}
            </div>
          </form>
        </CardContent>
      </Card>

      {searchMutation.data && (
        <SearchResults
          results={searchMutation.data.results}
          query={searchMutation.data.query}
          searchType={searchMutation.data.search_type}
        />
      )}

      {searchMutation.error && (
        <Card>
          <CardHeader>
            <CardTitle className="text-destructive">Search Error</CardTitle>
            <CardDescription>
              {searchMutation.error instanceof Error
                ? searchMutation.error.message
                : 'Failed to perform search'}
            </CardDescription>
          </CardHeader>
        </Card>
      )}
    </div>
  )
}
