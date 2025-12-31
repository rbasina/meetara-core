'use client'

import React, { useState, useEffect } from 'react'
import { Brain, Sparkles, Search, BookOpen } from 'lucide-react'
import { cn } from '@/lib/utils'

interface LoadingIndicatorProps {
  variant?: 'default' | 'thinking' | 'searching'
  className?: string
}

// Progress steps shown during me²TARA thinking (icon + short label)
const progressSteps = [
  { icon: Search, label: "Knowledge Search" },
  { icon: BookOpen, label: "Expert Sources" },
  { icon: Brain, label: "Analysis" },
  { icon: Sparkles, label: "Response" },
]

export default function LoadingIndicator({ 
  variant = 'default',
  className 
}: LoadingIndicatorProps) {
  const [currentStep, setCurrentStep] = useState(0)
  
  // Progress through steps sequentially (synchronized message + icon)
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        // Move to next step, stay at last step when reached
        if (prev < progressSteps.length - 1) {
          return prev + 1
        }
        return prev // Stay at "Response" step
      })
    }, 6000) // Move to next step every 6 seconds
    
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
            <span className="text-base font-semibold text-foreground">
              me²TARA Thinking
            </span>
            
            <TypingDots />
          </div>
          
          {/* Progress Indicator - SYNCHRONIZED with message above */}
          <div className="ml-[52px] flex items-center gap-2 text-xs text-muted-foreground">
            <ThinkingProgress currentStep={currentStep} />
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

// Thinking progress animation - shows icon + message for active step
function ThinkingProgress({ currentStep }: { currentStep: number }) {
  const currentStepData = progressSteps[currentStep]
  const Icon = currentStepData.icon
  
  return (
    <div className="flex items-center gap-2">
      {/* Show all step icons with completion status */}
      <div className="flex items-center gap-1">
        {progressSteps.map((s, i) => {
          const StepIcon = s.icon
          const isActive = i === currentStep
          const isComplete = i < currentStep
          
          return (
            <div 
              key={i}
              className={cn(
                "w-6 h-6 rounded-full flex items-center justify-center transition-all duration-300",
                isActive && "bg-primary/20 text-primary",
                isComplete && "bg-green-500/20 text-green-500",
                !isActive && !isComplete && "text-muted-foreground/40"
              )}
            >
              <StepIcon className={cn(
                "w-3 h-3",
                isActive && "animate-pulse"
              )} />
            </div>
          )
        })}
      </div>
      
      {/* Show current step label */}
      <span className="text-xs text-muted-foreground animate-fade-in">
        {currentStepData.label}
      </span>
    </div>
  )
}
