'use client'

import React from 'react'
import { Sparkles, MessageSquare, BookOpen, Zap, Brain } from 'lucide-react'
import { cn } from '@/lib/utils'
import AnimatedLogo from './AnimatedLogo'

interface EmptyStateProps {
  onSuggestionClick: (suggestion: string) => void
}

const suggestions = [
  {
    icon: <BookOpen className="w-5 h-5" />,
    title: "Learn Something New",
    query: "Explain calculus derivatives",
    category: "Education"
  },
  {
    icon: <Zap className="w-5 h-5" />,
    title: "Health Information",
    query: "What are the symptoms of diabetes?",
    category: "Healthcare"
  },
  {
    icon: <Brain className="w-5 h-5" />,
    title: "Technical Knowledge",
    query: "What is software development?",
    category: "Technology"
  },
  {
    icon: <MessageSquare className="w-5 h-5" />,
    title: "Business Insights",
    query: "What is project management?",
    category: "Business"
  }
]

export default function EmptyState({ onSuggestionClick }: EmptyStateProps) {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center max-w-2xl">
        {/* Logo Animation */}
        <div className="mb-8 flex justify-center">
          <AnimatedLogo 
            size="xl" 
            showText={false} 
            glowIntensity="high"
          />
        </div>
        
        {/* Welcome Text */}
        <h1 className="text-3xl font-bold mb-3">
          Welcome to <span className="gradient-text">me²TARA</span>
        </h1>
        <p className="text-muted-foreground mb-8 max-w-md mx-auto">
          Your intelligent knowledge assistant with domain-specific expertise. 
          Ask me anything about health, education, technology, business, and more.
        </p>
        
        {/* Feature Highlights */}
        <div className="flex items-center justify-center gap-6 mb-10 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[hsl(var(--rag-color))]" />
            <span>RAG-Powered</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-secondary" />
            <span>Multi-Domain</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary" />
            <span>Image Analysis</span>
          </div>
        </div>
        
        {/* Suggestion Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {suggestions.map((suggestion, idx) => (
            <button
              key={idx}
              onClick={() => onSuggestionClick(suggestion.query)}
              className={cn(
                "group p-4 rounded-xl text-left transition-all duration-300",
                "bg-card border border-border hover:border-primary/50",
                "hover:shadow-lg hover:-translate-y-1",
                "animate-fade-in"
              )}
              style={{ animationDelay: `${idx * 0.1}s` }}
            >
              <div className="flex items-start gap-3">
                <div className={cn(
                  "p-2 rounded-lg transition-colors",
                  "bg-muted group-hover:bg-primary/10",
                  "text-muted-foreground group-hover:text-primary"
                )}>
                  {suggestion.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm">{suggestion.title}</span>
                    <span className="text-xs text-muted-foreground">{suggestion.category}</span>
                  </div>
                  <p className="text-sm text-muted-foreground truncate">
                    "{suggestion.query}"
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
        
        {/* Keyboard Shortcut Hint */}
        <p className="mt-8 text-xs text-muted-foreground">
          Press <kbd className="px-1.5 py-0.5 rounded bg-muted font-mono text-xs">Enter</kbd> to send a message
        </p>
      </div>
    </div>
  )
}

