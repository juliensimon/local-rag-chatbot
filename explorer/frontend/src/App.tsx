/**
 * Main application component
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from '@/context/ThemeContext'
import { ExplorerLayout } from '@/components/layout/ExplorerLayout'
import { StatisticsDashboard } from '@/components/dashboard/StatisticsDashboard'
import { RecommendationsPanel } from '@/components/recommendations/RecommendationsPanel'
import { SearchInterface } from '@/components/search/SearchInterface'
import { DocumentBrowser } from '@/components/browser/DocumentBrowser'
import { useState } from 'react'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000, // 1 minute
      retry: 1,
    },
  },
})

type Tab = 'dashboard' | 'recommendations' | 'search' | 'browser'

function AppContent() {
  const [activeTab, setActiveTab] = useState<Tab>('dashboard')

  return (
    <ExplorerLayout activeTab={activeTab} onTabChange={setActiveTab}>
      {activeTab === 'dashboard' && <StatisticsDashboard />}
      {activeTab === 'recommendations' && <RecommendationsPanel />}
      {activeTab === 'search' && <SearchInterface />}
      {activeTab === 'browser' && <DocumentBrowser />}
    </ExplorerLayout>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AppContent />
      </ThemeProvider>
    </QueryClientProvider>
  )
}
