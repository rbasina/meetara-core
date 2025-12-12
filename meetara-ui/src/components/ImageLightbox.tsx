'use client'

import React, { useEffect, useCallback } from 'react'
import { X, ChevronLeft, ChevronRight, BookOpen, ZoomIn, ZoomOut } from 'lucide-react'
import { cn } from '@/lib/utils'
import { ImageData, Message } from '@/lib/types'

interface ImageLightboxProps {
  messages: Message[]
  selectedImageIndex: number | null
  expandedImageMessageId: string | null
  onClose: () => void
  onNavigate: (index: number) => void
}

export default function ImageLightbox({
  messages,
  selectedImageIndex,
  expandedImageMessageId,
  onClose,
  onNavigate
}: ImageLightboxProps) {
  // Find the message with images
  const message = messages.find(m => m.id === expandedImageMessageId)
  
  // Keyboard navigation
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!message?.images || selectedImageIndex === null) return
    
    if (e.key === 'Escape') {
      onClose()
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault()
      const newIndex = selectedImageIndex > 0 
        ? selectedImageIndex - 1 
        : message.images.length - 1
      onNavigate(newIndex)
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      const newIndex = selectedImageIndex < message.images.length - 1
        ? selectedImageIndex + 1
        : 0
      onNavigate(newIndex)
    }
  }, [selectedImageIndex, message, onClose, onNavigate])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  // Don't render if no image selected
  if (selectedImageIndex === null || !expandedImageMessageId || !message?.images) {
    return null
  }

  const currentImage = message.images[selectedImageIndex]
  const imageUrl = currentImage.image_url.startsWith('http') 
    ? currentImage.image_url 
    : `http://localhost:8000${currentImage.image_url}`

  const getDomainSpecificTips = (): string[] => {
    const domain = message.domain || ''
    const tips: string[] = []
    
    const isAcademic = domain.includes('academic') || domain.includes('education') || domain.includes('tutoring')
    const isHealth = domain.includes('health') || domain.includes('medical')
    const isBusiness = domain.includes('business') || domain.includes('professional')
    const isTechnical = domain.includes('programming') || domain.includes('technology')

    if (isAcademic) {
      tips.push('Study the labels and annotations carefully')
      tips.push('Relate this visual to the concepts discussed in the response')
      if (currentImage.page) {
        tips.push(`Cross-reference with page ${currentImage.page} for complete context`)
      }
    } else if (isHealth) {
      tips.push('Review all anatomical labels and medical terminology')
      tips.push('Always consult healthcare professionals for medical advice')
    } else if (isBusiness) {
      tips.push('Analyze the data trends and key metrics shown')
      tips.push('Consider how this applies to real-world scenarios')
    } else if (isTechnical) {
      tips.push('Study the architecture and data flow shown')
      tips.push('Note the technical specifications and patterns')
    } else {
      tips.push('Observe all details and key elements')
      tips.push('Relate this image to the concepts discussed above')
    }

    return tips
  }

  return (
    <div 
      className="lightbox-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose()
        }
      }}
    >
      <div className="bg-card rounded-2xl max-w-6xl max-h-[90vh] w-full mx-4 flex flex-col shadow-2xl overflow-hidden animate-slide-in-up">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border bg-gradient-to-r from-primary/5 to-secondary/5">
          <div className="flex items-center gap-3">
            <BookOpen className="w-5 h-5 text-primary" />
            <div>
              <h3 className="font-bold text-foreground">
                Image {selectedImageIndex + 1} of {message.images.length}
              </h3>
              {currentImage.filename && (
                <p className="text-sm text-muted-foreground">
                  {currentImage.filename}
                  {currentImage.page && ` • Page ${currentImage.page}`}
                </p>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            aria-label="Close lightbox"
          >
            <X className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>

        {/* Image Container */}
        <div className="flex-1 overflow-auto p-6 bg-muted/50">
          <div className="bg-card rounded-xl p-4 shadow-inner mb-4">
            <img 
              src={imageUrl}
              alt={currentImage.ocr_text || `Image ${selectedImageIndex + 1}`}
              className="max-w-full h-auto mx-auto rounded-lg"
              onError={(e) => {
                e.currentTarget.style.display = 'none'
              }}
            />
          </div>

          {/* Caption Section */}
          {(currentImage.figure_number || currentImage.caption || (currentImage.captions && currentImage.captions.length > 0)) && (
            <div className="bg-primary/5 border-l-4 border-primary p-4 rounded-r-xl mb-4">
              <div className="flex items-start gap-3">
                {currentImage.figure_number && (
                  <span className="badge badge-primary font-bold whitespace-nowrap">
                    Figure {currentImage.figure_number}
                  </span>
                )}
                <div className="flex-1">
                  {currentImage.caption ? (
                    <p className="text-base font-medium">{currentImage.caption}</p>
                  ) : currentImage.captions && currentImage.captions.length > 0 ? (
                    <div className="space-y-2">
                      {currentImage.captions.map((cap, i) => (
                        <p key={i} className="text-base">{cap}</p>
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          )}

          {/* Description */}
          {(currentImage.visual_description || currentImage.ocr_text) && (
            <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-400 p-4 rounded-r-xl mb-4">
              <p className="text-sm font-semibold text-amber-800 dark:text-amber-300 mb-2">
                📝 Image Description:
              </p>
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                {currentImage.visual_description || currentImage.ocr_text || 'No description available'}
              </p>
            </div>
          )}

          {/* Study Tips */}
          <div className="bg-secondary/5 border-l-4 border-secondary p-4 rounded-r-xl">
            <p className="text-sm font-semibold text-secondary mb-2">
              💡 Study Guide:
            </p>
            <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside">
              {getDomainSpecificTips().map((tip, i) => (
                <li key={i}>{tip}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Navigation Footer */}
        <div className="flex flex-col gap-3 p-4 border-t border-border bg-card">
          <div className="flex items-center justify-between">
            <button
              onClick={() => {
                const newIndex = selectedImageIndex > 0 
                  ? selectedImageIndex - 1 
                  : message.images!.length - 1
                onNavigate(newIndex)
              }}
              disabled={message.images.length === 1}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all",
                message.images.length === 1
                  ? "bg-muted text-muted-foreground cursor-not-allowed"
                  : "bg-primary text-primary-foreground hover:bg-primary/90"
              )}
            >
              <ChevronLeft className="w-4 h-4" />
              Previous
            </button>
            
            {/* Pagination Dots */}
            <div className="flex gap-2">
              {message.images.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => onNavigate(idx)}
                  className={cn(
                    "rounded-full transition-all",
                    idx === selectedImageIndex 
                      ? "w-8 h-2 bg-primary" 
                      : "w-2 h-2 bg-muted-foreground/30 hover:bg-muted-foreground/50"
                  )}
                  aria-label={`Go to image ${idx + 1}`}
                />
              ))}
            </div>

            <button
              onClick={() => {
                const newIndex = selectedImageIndex < message.images!.length - 1
                  ? selectedImageIndex + 1
                  : 0
                onNavigate(newIndex)
              }}
              disabled={message.images.length === 1}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all",
                message.images.length === 1
                  ? "bg-muted text-muted-foreground cursor-not-allowed"
                  : "bg-primary text-primary-foreground hover:bg-primary/90"
              )}
            >
              Next
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
          
          {/* Keyboard Shortcuts Hint */}
          <p className="text-xs text-center text-muted-foreground">
            💡 Keyboard shortcuts: ← → to navigate, ESC to close
          </p>
        </div>
      </div>
    </div>
  )
}

