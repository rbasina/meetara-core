'use client'

import React, { useState } from 'react'
import { 
  ChevronDown, 
  ChevronUp, 
  Upload, 
  Trash2, 
  Moon, 
  Sun,
  Settings,
  HelpCircle,
  Sparkles
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Category, CategorizedDomain, Domain } from '@/lib/types'
import { useTheme } from '@/context/ThemeContext'
import AnimatedLogo from './AnimatedLogo'

interface SidebarProps {
  categories: { [key: string]: Category }
  domains: Domain[]
  sessionId: string | null
  messageCount: number
  onUploadClick: () => void
  onClearHistory: () => void
  expandedCategories: Set<string>
  onToggleCategory: (categoryName: string) => void
}

// Category icons mapping
const categoryIcons: { [key: string]: string } = {
  healthcare: '🏥',
  legal_financial: '⚖️',
  emergency_crisis: '🚨',
  business: '💼',
  business_professional: '👔',
  education: '🎓',
  technology: '💻',
  daily_life: '🏠',
  creative: '🎨',
  research_academic: '🔬',
  aerospace_transportation: '🚀',
  industrial_manufacturing: '🏭',
  specialized: '🔧',
  entertainment_media: '🎬',
  food_cooking: '🍳',
  psychology_wellness: '🧘',
  sports_recreation: '⚽',
  travel_tourism: '✈️'
}

// Tier colors
const tierColors: { [key: string]: string } = {
  safety_critical: 'tier-safety-critical',
  expert: 'tier-expert',
  quality: 'tier-quality'
}

export default function Sidebar({
  categories,
  domains,
  sessionId,
  messageCount,
  onUploadClick,
  onClearHistory,
  expandedCategories,
  onToggleCategory
}: SidebarProps) {
  const { theme, setTheme, resolvedTheme } = useTheme()
  const [showSettings, setShowSettings] = useState(false)

  const getCategoryIcon = (categoryName: string): string => {
    return categoryIcons[categoryName] || '📁'
  }

  const getTierClass = (tier: string): string => {
    return tierColors[tier] || 'tier-quality'
  }

  const filterDomains = (category: Category): CategorizedDomain[] => {
    return category.domains.filter(domain => domain.has_content)
  }

  const totalDomainsWithContent = Object.values(categories).reduce(
    (sum, cat) => sum + cat.domains_with_content, 0
  )

  return (
    <aside className="hidden md:flex w-80 flex-col bg-[hsl(var(--sidebar-bg))] border-r border-[hsl(var(--sidebar-border))] h-screen">
      {/* Header with Logo */}
      <div className="p-5 border-b border-[hsl(var(--sidebar-border))]">
        <div className="flex items-center justify-between">
          <AnimatedLogo 
            size="md" 
            showText={true} 
            showTagline={false}
            glowIntensity="high"
            className="justify-start"
          />
          
          {/* Theme Toggle */}
          <button
            onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
            className="p-2 rounded-lg hover:bg-[hsl(var(--sidebar-hover))] transition-colors"
            title={`Switch to ${resolvedTheme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {resolvedTheme === 'dark' ? (
              <Sun className="w-5 h-5 text-amber-500" />
            ) : (
              <Moon className="w-5 h-5 text-slate-600" />
            )}
          </button>
        </div>
        <p className="text-xs text-muted-foreground mt-1 ml-[68px]">
          Knowledge-Based AI Assistant
        </p>
      </div>

      {/* Quick Stats */}
      <div className="px-4 py-3 border-b border-[hsl(var(--sidebar-border))] bg-gradient-to-r from-primary/5 to-secondary/5">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="font-medium">{totalDomainsWithContent} Domains Active</span>
          </div>
          <span className="text-muted-foreground">
            {Object.keys(categories).length} Categories
          </span>
        </div>
      </div>

      {/* Domains Section */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            Knowledge Domains
          </h3>
        </div>
        
        {/* Categories List */}
        <div className="space-y-1 px-2 pb-4">
          {Object.entries(categories).map(([categoryName, category]) => {
            const filteredDomains = filterDomains(category)
            const isExpanded = expandedCategories.has(categoryName)
            
            if (filteredDomains.length === 0) return null
            
            return (
              <div 
                key={categoryName} 
                className={cn(
                  "rounded-xl overflow-hidden transition-all duration-200",
                  isExpanded ? "bg-card shadow-sm" : ""
                )}
              >
                {/* Category Header */}
                <button
                  onClick={() => onToggleCategory(categoryName)}
                  className={cn(
                    "category-header w-full",
                    isExpanded && "bg-primary/5"
                  )}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xl">{getCategoryIcon(categoryName)}</span>
                    <div className="flex flex-col items-start">
                      <span className="font-medium text-sm">
                        {category.display_name}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {category.domains_with_content} of {category.total_domains} active
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={cn(
                      "badge text-[10px]",
                      getTierClass(category.tier)
                    )}>
                      {category.tier.replace('_', ' ')}
                    </span>
                    {isExpanded ? (
                      <ChevronUp className="w-4 h-4 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-muted-foreground" />
                    )}
                  </div>
                </button>
                
                {/* Domain List */}
                {isExpanded && (
                  <div className="px-2 pb-2 space-y-1 animate-slide-in-up">
                    {filteredDomains.map((domain, index) => (
                      <div
                        key={domain.name}
                        className={cn(
                          "sidebar-item ml-6",
                          "animate-fade-in",
                          `stagger-${Math.min(index + 1, 5)}`
                        )}
                        style={{ animationFillMode: 'backwards' }}
                      >
                        <div className="flex-1">
                          <span className="text-sm">
                            {domain.display_name}
                          </span>
                        </div>
                        <span className={cn(
                          "badge text-xs",
                          domain.has_content ? "badge-success" : "badge-secondary"
                        )}>
                          {domain.documents !== undefined ? domain.documents : domain.chunks}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Bottom Actions */}
      <div className="p-4 space-y-2 border-t border-[hsl(var(--sidebar-border))] bg-[hsl(var(--sidebar-bg))]">
        {/* Upload Button */}
        <button
          onClick={onUploadClick}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 transition-all duration-200 font-medium shadow-md hover:shadow-lg"
        >
          <Upload className="w-4 h-4" />
          <span>Upload Documents</span>
        </button>
        
        {/* Clear History Button */}
        {messageCount > 0 && (
          <button
            onClick={onClearHistory}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-muted text-muted-foreground rounded-xl hover:bg-destructive/10 hover:text-destructive transition-colors text-sm"
          >
            <Trash2 className="w-4 h-4" />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {/* Session Info */}
      {sessionId && (
        <div className="px-4 py-3 border-t border-[hsl(var(--sidebar-border))] bg-muted/30">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Session: {sessionId.slice(0, 8)}...</span>
            <span>{messageCount} messages</span>
          </div>
        </div>
      )}
    </aside>
  )
}

