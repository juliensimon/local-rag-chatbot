/**
 * App header with title and controls
 */

import { BookOpen } from 'lucide-react'
import { ThemeToggle } from './ThemeToggle'
import { useHealthCheck } from '@/hooks/useHealthCheck'

export function Header() {
  const { data: health, isError } = useHealthCheck()

  const getStatusColor = () => {
    if (isError) return 'bg-red-500'
    if (!health) return 'bg-gray-400'
    if (health.status === 'healthy') return 'bg-green-500'
    return 'bg-yellow-500'
  }

  const getStatusLabel = () => {
    if (isError) return 'API unavailable'
    if (!health) return 'Checking...'
    if (health.status === 'healthy') return 'Connected'
    return 'Degraded'
  }

  return (
    <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <BookOpen className="h-6 w-6 text-primary" />
          <h1 className="text-lg font-semibold">Document Q&A</h1>
        </div>

        <div className="flex items-center gap-4">
          {/* Health status indicator */}
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span
              className={`h-2 w-2 rounded-full ${getStatusColor()}`}
              aria-hidden="true"
            />
            <span className="hidden sm:inline">{getStatusLabel()}</span>
          </div>

          <ThemeToggle />
        </div>
      </div>
    </header>
  )
}
