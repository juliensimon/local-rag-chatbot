/**
 * Main layout for explorer application
 */

import { BarChart3, FileSearch, Search, Lightbulb } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/context/ThemeContext'
import { Moon, Sun } from 'lucide-react'

type Tab = 'dashboard' | 'recommendations' | 'search' | 'browser'

interface ExplorerLayoutProps {
  children: React.ReactNode
  activeTab: Tab
  onTabChange: (tab: Tab) => void
}

export function ExplorerLayout({ children, activeTab, onTabChange }: ExplorerLayoutProps) {
  const { resolvedTheme, setTheme } = useTheme()

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <BarChart3 className="h-4 w-4" /> },
    { id: 'recommendations', label: 'Recommendations', icon: <Lightbulb className="h-4 w-4" /> },
    { id: 'search', label: 'Search', icon: <Search className="h-4 w-4" /> },
    { id: 'browser', label: 'Browse', icon: <FileSearch className="h-4 w-4" /> },
  ]

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b bg-background">
        <div className="container mx-auto flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-6 w-6" />
            <h1 className="text-xl font-semibold">Vector Store Explorer</h1>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
          >
            {resolvedTheme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </Button>
        </div>
        <div className="container mx-auto border-t px-4">
          <nav className="flex gap-1">
            {tabs.map((tab) => (
              <Button
                key={tab.id}
                variant={activeTab === tab.id ? 'secondary' : 'ghost'}
                className="rounded-b-none"
                onClick={() => onTabChange(tab.id)}
              >
                {tab.icon}
                <span className="ml-2">{tab.label}</span>
              </Button>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1">
        <div className="container mx-auto h-full px-4 py-6">{children}</div>
      </main>
    </div>
  )
}
