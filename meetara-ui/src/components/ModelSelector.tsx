'use client'

import React, { useState, useEffect } from 'react'
import { ChevronDown, Cpu, Zap, Brain, Sparkles, Check, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Model {
  id: string
  name: string
  description: string
  size: number
  parameters: string
  tier: string
  default: boolean
  recommended_for: string[]
  loaded?: boolean
}

interface ModelSelectorProps {
  selectedModel: string | null
  onModelChange: (modelId: string) => void
  disabled?: boolean
}

const tierIcons: { [key: string]: React.ReactNode } = {
  fast: <Zap className="w-4 h-4" />,
  balanced: <Cpu className="w-4 h-4" />,
  thinking: <Brain className="w-4 h-4" />,
  powerful: <Sparkles className="w-4 h-4" />
}

const tierColors: { [key: string]: string } = {
  fast: 'text-green-500',
  balanced: 'text-blue-500',
  thinking: 'text-purple-500',
  powerful: 'text-amber-500'
}

export default function ModelSelector({
  selectedModel,
  onModelChange,
  disabled = false
}: ModelSelectorProps) {
  const [models, setModels] = useState<Model[]>([])
  const [isOpen, setIsOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [autoSelect, setAutoSelect] = useState(true)

  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/models/')
        if (response.ok) {
          const data = await response.json()
          setModels(data.models || [])
          setAutoSelect(data.auto_select_enabled)
          
          // Set default model if none selected
          if (!selectedModel && data.current_model) {
            onModelChange(data.current_model)
          }
        }
      } catch (error) {
        console.error('Failed to fetch models:', error)
        // Use fallback models if API fails
        setModels([
          {
            id: 'meetara-1.7b',
            name: 'Meetara 1.7B (Fast)',
            description: 'Fast and efficient for general queries',
            size: 1.2,
            parameters: '1.7B',
            tier: 'fast',
            default: true,
            recommended_for: ['general', 'daily_life']
          },
          {
            id: 'meetara-4b-thinking',
            name: 'Meetara 4B Thinking',
            description: 'Deep reasoning for complex tasks',
            size: 2.8,
            parameters: '4B',
            tier: 'thinking',
            default: false,
            recommended_for: ['healthcare', 'education']
          }
        ])
      } finally {
        setIsLoading(false)
      }
    }

    fetchModels()
  }, [selectedModel, onModelChange])

  const currentModel = models.find(m => m.id === selectedModel) || models.find(m => m.default)

  const handleModelSelect = async (modelId: string) => {
    onModelChange(modelId)
    setIsOpen(false)
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Loading models...</span>
      </div>
    )
  }

  return (
    <div className="relative">
      {/* Model Selector Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-lg border border-border",
          "bg-background hover:bg-muted transition-colors text-sm",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          isOpen && "ring-2 ring-primary/50"
        )}
      >
        {currentModel && (
          <>
            <span className={tierColors[currentModel.tier]}>
              {tierIcons[currentModel.tier]}
            </span>
            <span className="font-medium">{currentModel.name}</span>
            {autoSelect && (
              <span className="text-xs text-muted-foreground">(Auto)</span>
            )}
          </>
        )}
        <ChevronDown className={cn(
          "w-4 h-4 text-muted-foreground transition-transform",
          isOpen && "rotate-180"
        )} />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 w-80 bg-card border border-border rounded-xl shadow-lg z-50 overflow-hidden animate-slide-in-up">
          <div className="p-3 border-b border-border bg-muted/30">
            <h3 className="font-semibold text-sm">Select Model</h3>
            <p className="text-xs text-muted-foreground mt-1">
              Choose a model based on your needs
            </p>
          </div>
          
          <div className="max-h-64 overflow-y-auto">
            {models.map((model) => (
              <button
                key={model.id}
                onClick={() => handleModelSelect(model.id)}
                className={cn(
                  "w-full px-4 py-3 text-left hover:bg-muted/50 transition-colors",
                  "border-b border-border last:border-0",
                  selectedModel === model.id && "bg-primary/5"
                )}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <span className={tierColors[model.tier]}>
                      {tierIcons[model.tier]}
                    </span>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm">{model.name}</span>
                        {model.default && (
                          <span className="text-[10px] px-1.5 py-0.5 bg-primary/10 text-primary rounded">
                            Default
                          </span>
                        )}
                        {model.loaded && (
                          <span className="text-[10px] px-1.5 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded">
                            Loaded
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {model.description}
                      </p>
                      <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                        <span>{model.parameters}</span>
                        <span>•</span>
                        <span>{model.size} GB</span>
                      </div>
                    </div>
                  </div>
                  {selectedModel === model.id && (
                    <Check className="w-4 h-4 text-primary" />
                  )}
                </div>
              </button>
            ))}
          </div>
          
          {/* Auto-select Toggle */}
          <div className="p-3 border-t border-border bg-muted/30">
            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-xs text-muted-foreground">
                Auto-select based on domain
              </span>
              <div className={cn(
                "w-8 h-4 rounded-full transition-colors relative",
                autoSelect ? "bg-primary" : "bg-muted"
              )}>
                <div className={cn(
                  "absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform",
                  autoSelect ? "translate-x-4" : "translate-x-0.5"
                )} />
              </div>
            </label>
          </div>
        </div>
      )}

      {/* Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  )
}

