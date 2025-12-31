'use client'

import React, { useState, useEffect } from 'react'
import { Brain, Sparkles, Search, BookOpen } from 'lucide-react'
import { cn } from '@/lib/utils'

interface LoadingIndicatorProps {
  variant?: 'default' | 'thinking' | 'searching'
  className?: string
}

// me²TARA-specific thinking messages (user-friendly, no technical jargon)
const thinkingMessages = [
  "Understanding your question",
  "Searching me²TARA knowledge base",
  "Finding relevant information",
  "Analyzing expert sources",
  "Preparing personalized response",
]

export default function LoadingIndicator({ 
  variant = 'default',
  className 
}: LoadingIndicatorProps) {
  const [messageIndex, setMessageIndex] = useState(0)
  
  // Rotate through thinking messages
  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % thinkingMessages.length)
    }, 2500) // Change message every 2.5 seconds
    
    return () => clearInterval(interval)
  }, [])
  
  return (
    <div className={cn("flex justify-start animate-slide-in-up", className)}>
      <div className="message-assistant max-w-5xl w-full px-5 py-4 rounded-2xl rounded-tl-sm shadow-md">
        <div className="flex flex-col gap-3">
          {/* me²TARA Thinking Header */}
          <div className="flex items-center gap-3">
            {/* Animated Brain Icon */}
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary/30 to-secondary/30 flex items-center justify-center relative">
              <Brain className="w-5 h-5 text-primary animate-pulse" />
              <Sparkles className="w-3 h-3 text-secondary absolute -top-1 -right-1 animate-bounce" />
            </div>
            
            {/* me²TARA Thinking Text */}
            <div className="flex flex-col">
              <span className="text-base font-semibold text-foreground">
                me²TARA Thinking
              </span>
              <span className="text-xs text-muted-foreground animate-fade-in-out">
                {thinkingMessages[messageIndex]}
              </span>
            </div>
            
            <TypingDots />
          </div>
          
          {/* Progress Indicator */}
          <div className="ml-13 flex items-center gap-2 text-xs text-muted-foreground">
            <ThinkingProgress />
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

// Thinking progress animation - shows me²TARA processing steps (user-friendly)
function ThinkingProgress() {
  const [step, setStep] = useState(0)
  const steps = [
    { icon: Search, label: "Knowledge Search" },
    { icon: BookOpen, label: "Expert Sources" },
    { icon: Brain, label: "Analysis" },
    { icon: Sparkles, label: "Response" },
  ]
  
  useEffect(() => {
    const interval = setInterval(() => {
      setStep((prev) => Math.min(prev + 1, steps.length - 1))
    }, 8000) // Move to next step every 8 seconds
    
    return () => clearInterval(interval)
  }, [])
  
  return (
    <div className="flex items-center gap-1">
      {steps.map((s, i) => {
        const Icon = s.icon
        const isActive = i === step
        const isComplete = i < step
        
        return (
          <div 
            key={s.label}
            className={cn(
              "flex items-center gap-1 px-2 py-1 rounded-full transition-all duration-300",
              isActive && "bg-primary/10 text-primary",
              isComplete && "text-green-500",
              !isActive && !isComplete && "text-muted-foreground/50"
            )}
          >
            <Icon className={cn(
              "w-3 h-3",
              isActive && "animate-pulse"
            )} />
            {isActive && (
              <span className="text-[10px] font-medium animate-fade-in">
                {s.label}
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}
