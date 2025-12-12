'use client'

import React from 'react'
import { Loader2, Sparkles } from 'lucide-react'
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
      <div className="message-assistant max-w-5xl w-full px-5 py-4 rounded-2xl rounded-tl-sm shadow-md">
        <div className="flex items-center gap-3">
          {/* Simple Animated Icon */}
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center">
            <Loader2 className="w-4 h-4 text-primary animate-spin" />
          </div>
          
          {/* Simple Text */}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-foreground">
              Generating response
            </span>
            <TypingDots />
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
