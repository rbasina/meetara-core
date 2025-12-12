'use client'

import React from 'react'
import { Loader2, Sparkles, Brain, Database } from 'lucide-react'
import { cn } from '@/lib/utils'

interface LoadingIndicatorProps {
  variant?: 'default' | 'thinking' | 'searching'
  className?: string
}

export default function LoadingIndicator({ 
  variant = 'default',
  className 
}: LoadingIndicatorProps) {
  return (
    <div className={cn("flex justify-start animate-slide-in-up", className)}>
      <div className="message-assistant max-w-5xl w-full px-6 py-5 rounded-2xl rounded-tl-sm shadow-md">
        <div className="flex items-center gap-4">
          {/* Animated Icon */}
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center">
              {variant === 'thinking' ? (
                <Brain className="w-5 h-5 text-primary animate-pulse" />
              ) : variant === 'searching' ? (
                <Database className="w-5 h-5 text-primary animate-pulse" />
              ) : (
                <Sparkles className="w-5 h-5 text-primary animate-pulse" />
              )}
            </div>
            <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-card border-2 border-primary flex items-center justify-center">
              <Loader2 className="w-2.5 h-2.5 text-primary animate-spin" />
            </div>
          </div>
          
          {/* Text Content */}
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-foreground">
                me²TARA is thinking...
              </span>
              <TypingDots />
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              {variant === 'searching' 
                ? 'Searching knowledge base for relevant information'
                : variant === 'thinking'
                ? 'Analyzing your question and formulating response'
                : 'Analyzing your question and searching knowledge base'
              }
            </p>
          </div>
        </div>
        
        {/* Progress Steps (Optional) */}
        <div className="mt-4 pt-4 border-t border-border">
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <Step active completed icon={<Database className="w-3 h-3" />} label="Query received" />
            <Step active icon={<Brain className="w-3 h-3" />} label="Processing" />
            <Step icon={<Sparkles className="w-3 h-3" />} label="Generating response" />
          </div>
        </div>
      </div>
    </div>
  )
}

// Typing dots animation
function TypingDots() {
  return (
    <div className="typing-indicator">
      <span />
      <span />
      <span />
    </div>
  )
}

// Step indicator
interface StepProps {
  active?: boolean
  completed?: boolean
  icon: React.ReactNode
  label: string
}

function Step({ active, completed, icon, label }: StepProps) {
  return (
    <div className={cn(
      "flex items-center gap-1.5 transition-colors",
      completed ? "text-primary" : active ? "text-foreground" : "text-muted-foreground/50"
    )}>
      <div className={cn(
        "p-1 rounded-full",
        completed ? "bg-primary/20" : active ? "bg-primary/10" : "bg-muted"
      )}>
        {icon}
      </div>
      <span className={cn(
        completed && "font-medium",
        active && "font-medium"
      )}>
        {label}
      </span>
    </div>
  )
}

