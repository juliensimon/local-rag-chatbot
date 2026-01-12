/**
 * Main application layout with responsive grid
 */

import { Header } from './Header'

interface AppLayoutProps {
  children: React.ReactNode
  sidebar?: React.ReactNode
}

export function AppLayout({ children, sidebar }: AppLayoutProps) {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">
        <div className="container mx-auto h-full px-4 py-6">
          <div className="grid h-full gap-6 lg:grid-cols-[1fr_350px]">
            {/* Main content area */}
            <div className="flex flex-col">{children}</div>

            {/* Sidebar (context panel) - hidden on mobile, shown on lg+ */}
            {sidebar && (
              <aside className="hidden lg:block">
                <div className="sticky top-6">{sidebar}</div>
              </aside>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
