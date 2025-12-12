'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Send, X, Loader2, Database, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Category } from '@/lib/types'
import ModelSelector from './ModelSelector'

interface InputAreaProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  isLoading: boolean
  messageCount: number
  categories: { [key: string]: Category }
  detectedDomain: string | null
  onDetectedDomainChange: (domain: string | null) => void
  selectedModel: string | null
  onModelChange: (modelId: string) => void
}

interface DomainKeywordsResponse {
  keywords: { [domain: string]: string[] }
  stop_words: string[]
  generic_terms: string[]
  total_domains: number
}

export default function InputArea({
  value,
  onChange,
  onSend,
  isLoading,
  messageCount,
  categories,
  detectedDomain,
  onDetectedDomainChange,
  selectedModel,
  onModelChange
}: InputAreaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [querySuggestions, setQuerySuggestions] = useState<string[]>([])
  const [domainKeywords, setDomainKeywords] = useState<{ [key: string]: string[] }>({})
  const [stopWords, setStopWords] = useState<Set<string>>(new Set())
  const [genericTerms, setGenericTerms] = useState<Set<string>>(new Set())

  // Load domain keywords from API (config-driven)
  useEffect(() => {
    const fetchDomainKeywords = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/chat/domains/keywords')
        if (response.ok) {
          const data: DomainKeywordsResponse = await response.json()
          setDomainKeywords(data.keywords || {})
          setStopWords(new Set(data.stop_words?.map(w => w.toLowerCase()) || []))
          setGenericTerms(new Set(data.generic_terms?.map(w => w.toLowerCase()) || []))
          console.log(`Loaded keywords for ${data.total_domains} domains from config`)
        }
      } catch (error) {
        console.error('Failed to load domain keywords from API:', error)
        // No fallback - rely on server-side detection if API fails
      }
    }

    fetchDomainKeywords()
  }, [])

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`
    }
  }, [value])

  // Detect domain from query using config-driven keywords
  const detectDomainFromQuery = (query: string) => {
    if (!query.trim() || Object.keys(domainKeywords).length === 0) {
      onDetectedDomainChange(null)
      return
    }
    
    const queryLower = query.toLowerCase()
    
    // Extract meaningful words (filter out stop words)
    const queryWords = queryLower
      .split(/\s+/)
      .filter(word => word.length > 2 && !stopWords.has(word))
    
    let bestMatch: { domain: string; score: number } | null = null
    
    // Score each domain based on keyword matches
    for (const [domain, keywords] of Object.entries(domainKeywords)) {
      let score = 0
      
      for (const keyword of keywords) {
        const keywordLower = keyword.toLowerCase()
        
        // Check if keyword appears in query
        if (queryLower.includes(keywordLower)) {
          // Higher score for exact word matches
          if (queryWords.includes(keywordLower)) {
            score += 2
          } else {
            score += 1
          }
          
          // Reduce score for generic terms
          if (genericTerms.has(keywordLower)) {
            score -= 0.5
          }
        }
      }
      
      if (score > 0 && (!bestMatch || score > bestMatch.score)) {
        bestMatch = { domain, score }
      }
    }
    
    // Only set domain if we have a reasonable match (score >= 2)
    if (bestMatch && bestMatch.score >= 2) {
      onDetectedDomainChange(bestMatch.domain)
    } else {
      onDetectedDomainChange(null)
    }
  }

  // Generate query suggestions based on available domains
  useEffect(() => {
    if (!value.trim() && Object.keys(categories).length > 0) {
      const suggestions: string[] = []
      
      // Define suggestions for common categories
      const suggestionMap: { [key: string]: string } = {
        'healthcare': 'What are the symptoms of diabetes?',
        'business': 'What is project management?',
        'education': 'Explain calculus derivatives',
        'technology': 'What is software development?',
        'legal_financial': 'What are my tenant rights?',
        'psychology_wellness': 'How can I manage stress better?',
        'travel_tourism': 'Best places to visit in Europe?',
        'food_cooking': 'How to make pasta from scratch?'
      }
      
      // Add suggestions for categories that have content
      for (const [categoryName, suggestion] of Object.entries(suggestionMap)) {
        const category = categories[categoryName]
        if (category && category.domains_with_content > 0) {
          suggestions.push(suggestion)
        }
      }
      
      // Limit to 3 suggestions
      setQuerySuggestions(suggestions.slice(0, 3))
    }
  }, [categories, value])

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value
    onChange(newValue)
    detectDomainFromQuery(newValue)
  }

  const totalDomainsAvailable = Object.values(categories).reduce(
    (sum, cat) => sum + cat.domains_with_content, 0
  )

  return (
    <div className="border-t border-border bg-card p-4">
      {/* Domain Detection Feedback */}
      {detectedDomain && value.trim() && (
        <div className="mb-3 px-4 py-2 bg-primary/5 border border-primary/20 rounded-xl flex items-center gap-2 animate-fade-in">
          <Database className="w-4 h-4 text-primary" />
          <span className="text-sm text-primary">
            Detected domain: <span className="font-semibold capitalize">{detectedDomain.replace(/_/g, ' ')}</span>
          </span>
        </div>
      )}
      
      {/* Query Suggestions */}
      {querySuggestions.length > 0 && !value.trim() && (
        <div className="mb-3">
          <p className="text-xs text-muted-foreground px-1 mb-2 flex items-center gap-1">
            <Sparkles className="w-3 h-3" />
            Try asking:
          </p>
          <div className="flex flex-wrap gap-2">
            {querySuggestions.map((suggestion, idx) => (
              <button
                key={idx}
                onClick={() => onChange(suggestion)}
                className="quick-action animate-fade-in"
                style={{ animationDelay: `${idx * 0.1}s` }}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}
      
      {/* Input Container */}
      <div className="flex gap-3">
        <div className="flex-1 relative input-glow rounded-xl border border-border bg-background">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleChange}
            onKeyPress={handleKeyPress}
            placeholder="Ask me²TARA anything... Type @ to mention domains"
            className={cn(
              "w-full resize-none px-4 py-3 pr-12 rounded-xl bg-transparent",
              "focus:outline-none transition-all",
              "placeholder:text-muted-foreground/60"
            )}
            rows={1}
            style={{ minHeight: '50px', maxHeight: '150px' }}
            disabled={isLoading}
          />
          {value.trim() && (
            <button
              onClick={() => {
                onChange('')
                onDetectedDomainChange(null)
              }}
              className="absolute right-3 top-3 p-1.5 text-muted-foreground hover:text-foreground rounded-lg hover:bg-muted transition-colors"
              aria-label="Clear input"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        
        {/* Send Button */}
        <button
          onClick={onSend}
          disabled={!value.trim() || isLoading}
          className="send-button"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Thinking...</span>
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              <span>Send</span>
            </>
          )}
        </button>
      </div>
      
      {/* Bottom Bar: Model Selector + Stats */}
      <div className="mt-3 flex items-center justify-between">
        {/* Model Selector */}
        <ModelSelector
          selectedModel={selectedModel}
          onModelChange={onModelChange}
          disabled={isLoading}
        />
        
        {/* Quick Stats */}
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span>{messageCount} message{messageCount !== 1 ? 's' : ''}</span>
          {totalDomainsAvailable > 0 && (
            <span className="flex items-center gap-1">
              <Database className="w-3 h-3" />
              {totalDomainsAvailable} domains
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
