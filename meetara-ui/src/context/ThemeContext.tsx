'use client'

import React, { createContext, useContext, useEffect, useState } from 'react'
import { Theme } from '@/lib/types'

interface ThemeContextType {
  theme: Theme
  setTheme: (theme: Theme) => void
  resolvedTheme: 'light' | 'dark'
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('system')
  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('light')
  const [mounted, setMounted] = useState(false)

  // Load theme from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('meetara-theme') as Theme | null
    if (stored) {
      setTheme(stored)
    }
    setMounted(true)
  }, [])

  // Update resolved theme based on system preference or explicit choice
  useEffect(() => {
    const updateResolvedTheme = () => {
      if (theme === 'system') {
        const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches
        setResolvedTheme(systemDark ? 'dark' : 'light')
      } else {
        setResolvedTheme(theme)
      }
    }

    updateResolvedTheme()

    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = () => updateResolvedTheme()
    mediaQuery.addEventListener('change', handler)

    return () => mediaQuery.removeEventListener('change', handler)
  }, [theme])

  // Apply theme to document
  useEffect(() => {
    if (!mounted) return

    const root = document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(resolvedTheme)
    localStorage.setItem('meetara-theme', theme)
  }, [resolvedTheme, theme, mounted])

  // Prevent flash of wrong theme
  if (!mounted) {
    return null
  }

  return (
    <ThemeContext.Provider value={{ theme, setTheme, resolvedTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

