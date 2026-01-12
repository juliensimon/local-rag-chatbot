/**
 * Main application component
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ThemeProvider } from '@/context/ThemeContext'
import { SettingsProvider } from '@/context/SettingsContext'
import { ChatProvider } from '@/context/ChatContext'
import { AppLayout } from '@/components/layout/AppLayout'
import { ChatContainer } from '@/components/chat/ChatContainer'
import { ControlPanel } from '@/components/controls/ControlPanel'
import { ContextPanel } from '@/components/context/ContextPanel'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000, // 1 minute
      retry: 1,
    },
  },
})

function AppContent() {
  return (
    <AppLayout sidebar={<ContextPanel />}>
      <div className="flex flex-col gap-4">
        <ControlPanel />
        <ChatContainer />

        {/* Mobile context panel - shown below chat on small screens */}
        <div className="lg:hidden">
          <ContextPanel />
        </div>
      </div>
    </AppLayout>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <SettingsProvider>
            <ChatProvider>
              <AppContent />
            </ChatProvider>
          </SettingsProvider>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  )
}
