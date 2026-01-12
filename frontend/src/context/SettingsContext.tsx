/**
 * Settings context for RAG configuration
 */

import { createContext, useContext, useState, useCallback } from 'react'
import type { SearchType } from '@/types/api'
import { DEFAULT_CHAT_SETTINGS } from '@/types/api'

interface Settings {
  ragEnabled: boolean
  searchType: SearchType
  docFilter: string | null
  useQueryRewriting: boolean
  useReranking: boolean
  hybridAlpha: number
}

interface SettingsContextValue extends Settings {
  setRagEnabled: (enabled: boolean) => void
  setSearchType: (type: SearchType) => void
  setDocFilter: (filter: string | null) => void
  setUseQueryRewriting: (enabled: boolean) => void
  setUseReranking: (enabled: boolean) => void
  setHybridAlpha: (alpha: number) => void
  resetSettings: () => void
}

const SettingsContext = createContext<SettingsContextValue | undefined>(undefined)

const initialSettings: Settings = {
  ragEnabled: DEFAULT_CHAT_SETTINGS.rag_enabled,
  searchType: DEFAULT_CHAT_SETTINGS.search_type,
  docFilter: DEFAULT_CHAT_SETTINGS.doc_filter,
  useQueryRewriting: DEFAULT_CHAT_SETTINGS.use_query_rewriting,
  useReranking: DEFAULT_CHAT_SETTINGS.use_reranking,
  hybridAlpha: DEFAULT_CHAT_SETTINGS.hybrid_alpha,
}

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<Settings>(initialSettings)

  const setRagEnabled = useCallback((enabled: boolean) => {
    setSettings((prev) => ({ ...prev, ragEnabled: enabled }))
  }, [])

  const setSearchType = useCallback((type: SearchType) => {
    setSettings((prev) => ({ ...prev, searchType: type }))
  }, [])

  const setDocFilter = useCallback((filter: string | null) => {
    setSettings((prev) => ({ ...prev, docFilter: filter }))
  }, [])

  const setUseQueryRewriting = useCallback((enabled: boolean) => {
    setSettings((prev) => ({ ...prev, useQueryRewriting: enabled }))
  }, [])

  const setUseReranking = useCallback((enabled: boolean) => {
    setSettings((prev) => ({ ...prev, useReranking: enabled }))
  }, [])

  const setHybridAlpha = useCallback((alpha: number) => {
    setSettings((prev) => ({ ...prev, hybridAlpha: alpha }))
  }, [])

  const resetSettings = useCallback(() => {
    setSettings(initialSettings)
  }, [])

  return (
    <SettingsContext.Provider
      value={{
        ...settings,
        setRagEnabled,
        setSearchType,
        setDocFilter,
        setUseQueryRewriting,
        setUseReranking,
        setHybridAlpha,
        resetSettings,
      }}
    >
      {children}
    </SettingsContext.Provider>
  )
}

export function useSettings() {
  const context = useContext(SettingsContext)
  if (!context) {
    throw new Error('useSettings must be used within a SettingsProvider')
  }
  return context
}
