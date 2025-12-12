'use client'

import React from 'react'
import { 
  Database, 
  Brain, 
  FileText, 
  ZoomIn, 
  BookOpen,
  ChevronRight,
  AlertCircle
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Message, ImageData } from '@/lib/types'
import { formatMarkdown } from '@/lib/markdown'

interface ChatMessageProps {
  message: Message
  onImageClick: (index: number, messageId: string) => void
  onToggleImageSection: (messageId: string) => void
  expandedImageSections: Set<string>
}

export default function ChatMessage({
  message,
  onImageClick,
  onToggleImageSection,
  expandedImageSections
}: ChatMessageProps) {
  const isUser = message.role === 'user'
  const hasImages = message.images && message.images.length > 0
  const sectionId = `images-${message.id}`
  const isImageSectionExpanded = expandedImageSections.has(sectionId)

  const getRAGStatusIcon = (status: 'rag' | 'llm' | 'mixed') => {
    switch (status) {
      case 'rag':
        return <Database className="w-3.5 h-3.5" />
      case 'llm':
        return <Brain className="w-3.5 h-3.5" />
      case 'mixed':
        return <FileText className="w-3.5 h-3.5" />
    }
  }

  const getRAGStatusText = (status: 'rag' | 'llm' | 'mixed') => {
    switch (status) {
      case 'rag':
        return 'RAG Documents'
      case 'llm':
        return 'LLM Knowledge'
      case 'mixed':
        return 'Mixed Sources'
    }
  }

  const getRAGStatusClass = (status: 'rag' | 'llm' | 'mixed') => {
    switch (status) {
      case 'rag':
        return 'status-rag'
      case 'llm':
        return 'status-llm'
      case 'mixed':
        return 'status-mixed'
    }
  }

  const getConfidenceClass = (confidence: number) => {
    if (confidence >= 0.8) return 'badge-success'
    if (confidence >= 0.6) return 'badge-warning'
    return 'badge-destructive'
  }

  if (isUser) {
    return (
      <div className="flex justify-end animate-slide-in-up">
        <div className="max-w-3xl px-5 py-3 rounded-2xl rounded-tr-sm message-user shadow-md">
          {message.timestamp && (
            <div className="text-xs text-white/70 mb-1">
              {message.timestamp.toLocaleString()}
            </div>
          )}
          <div className="whitespace-pre-wrap">{message.content}</div>
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex justify-start animate-slide-in-up">
      <div className={cn(
        "max-w-5xl w-full px-6 py-5 rounded-2xl rounded-tl-sm shadow-md",
        message.error 
          ? "bg-destructive/5 border-destructive/20" 
          : "message-assistant"
      )}>
        {/* Image Preview Section */}
        {hasImages && (
          <div className="mb-4 p-4 bg-gradient-to-r from-primary/5 to-secondary/5 rounded-xl border border-primary/20">
            <div className="flex items-center gap-2 mb-3">
              <BookOpen className="w-4 h-4 text-primary" />
              <h3 className="text-sm font-semibold">
                Quick Preview: {message.images!.length} Image{message.images!.length > 1 ? 's' : ''} Available
              </h3>
              <span className="ml-auto badge badge-primary text-xs">
                {message.images!.length} Image{message.images!.length > 1 ? 's' : ''}
              </span>
            </div>
            
            {/* Thumbnail Gallery */}
            <div className="flex gap-2 mb-3 overflow-x-auto pb-2">
              {message.images!.map((img, idx) => {
                const imageUrl = img.image_url.startsWith('http') 
                  ? img.image_url 
                  : `http://localhost:8000${img.image_url}`
                return (
                  <button
                    key={idx}
                    onClick={() => onImageClick(idx, message.id)}
                    className="image-thumbnail flex-shrink-0"
                    style={{ width: '80px', height: '60px' }}
                  >
                    <img 
                      src={imageUrl}
                      alt={`Image ${idx + 1}`}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        e.currentTarget.style.display = 'none'
                      }}
                    />
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-black/30">
                      <ZoomIn className="w-4 h-4 text-white" />
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-1">
                      <p className="text-[10px] text-white font-medium text-center">#{idx + 1}</p>
                    </div>
                  </button>
                )
              })}
            </div>
            
            {/* Action Buttons */}
            <div className="flex items-center justify-between">
              <button
                onClick={() => onImageClick(0, message.id)}
                className="text-xs text-primary hover:text-primary/80 font-medium flex items-center gap-1 transition-colors"
              >
                <ZoomIn className="w-3 h-3" />
                Open in full screen
              </button>
              <button
                onClick={() => onToggleImageSection(message.id)}
                className="text-xs text-primary hover:text-primary/80 font-medium flex items-center gap-1 transition-colors"
              >
                {isImageSectionExpanded ? 'Hide' : 'Show'} detailed images
                <ChevronRight className={cn(
                  "w-3 h-3 transition-transform",
                  isImageSectionExpanded && "rotate-90"
                )} />
              </button>
            </div>
          </div>
        )}
        
        {/* Timestamp */}
        {message.responseTimestamp && (
          <div className="text-xs text-muted-foreground mb-3">
            {new Date(message.responseTimestamp).toLocaleString()}
          </div>
        )}
        
        {/* Message Content */}
        {message.error ? (
          <div className="flex items-start gap-3 text-destructive">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <p>{message.content}</p>
          </div>
        ) : (
          <div 
            className="prose prose-sm max-w-none"
            dangerouslySetInnerHTML={{ __html: formatMarkdown(message.content) }} 
          />
        )}
        
        {/* Expanded Image Section */}
        {hasImages && isImageSectionExpanded && (
          <div className="mt-6 space-y-4 animate-slide-in-up">
            <div className="flex items-center gap-2 pb-2 border-b border-border">
              <div className="w-1 h-6 bg-gradient-to-b from-primary to-secondary rounded-full"></div>
              <h3 className="text-lg font-bold">📚 Detailed Images & Visual Explanations</h3>
            </div>
            
            {message.images!.map((img, idx) => (
              <ImageCard 
                key={idx} 
                image={img} 
                index={idx} 
                total={message.images!.length}
                domain={message.domain}
                onZoom={() => onImageClick(idx, message.id)}
              />
            ))}
          </div>
        )}
        
        {/* Message Metadata */}
        {!message.error && (
          <div className="mt-4 pt-3 border-t border-border">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              {/* Domain Badge */}
              {message.domain && (
                <div className="badge badge-primary flex items-center gap-1">
                  <Database className="w-3 h-3" />
                  <span className="capitalize">{message.domain.replace('_', ' ')}</span>
                </div>
              )}
              
              {/* RAG Status */}
              {message.ragStatus && (
                <div className={cn("badge flex items-center gap-1", getRAGStatusClass(message.ragStatus))}>
                  {getRAGStatusIcon(message.ragStatus)}
                  <span>{getRAGStatusText(message.ragStatus)}</span>
                </div>
              )}
              
              {/* Confidence */}
              {message.confidence !== undefined && (
                <div className={cn("badge", getConfidenceClass(message.confidence))}>
                  {(message.confidence * 100).toFixed(0)}% confidence
                </div>
              )}
              
              {/* Source Count */}
              {hasImages && (
                <div className="badge badge-secondary flex items-center gap-1">
                  <FileText className="w-3 h-3" />
                  <span>{message.images!.length} source{message.images!.length > 1 ? 's' : ''}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Image Card Component
interface ImageCardProps {
  image: ImageData
  index: number
  total: number
  domain?: string
  onZoom: () => void
}

function ImageCard({ image, index, total, domain, onZoom }: ImageCardProps) {
  const imageUrl = image.image_url.startsWith('http') 
    ? image.image_url 
    : `http://localhost:8000${image.image_url}`

  const getDomainTip = () => {
    if (!domain) return 'Review this visual carefully to understand the concept.'
    
    const d = domain.toLowerCase()
    if (d.includes('academic') || d.includes('education') || d.includes('tutoring')) {
      return 'Study this diagram carefully and relate it to the concepts discussed above.'
    }
    if (d.includes('health') || d.includes('medical')) {
      return 'Review this illustration in context. Always consult healthcare professionals for medical advice.'
    }
    if (d.includes('business') || d.includes('professional')) {
      return 'Analyze this visual to understand the business concepts and strategies discussed.'
    }
    if (d.includes('programming') || d.includes('technology') || d.includes('software')) {
      return 'Study this diagram alongside the technical explanations provided above.'
    }
    return 'Take time to understand how this visual relates to the concepts discussed.'
  }

  return (
    <div className="border border-border rounded-xl p-5 bg-card shadow-sm hover:shadow-md transition-shadow card-hover">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="badge badge-primary text-xs">
              Image {index + 1} of {total}
            </span>
            {image.page && (
              <span className="badge badge-secondary text-xs">
                Page {image.page}
              </span>
            )}
          </div>
          {image.filename && (
            <p className="text-sm text-muted-foreground">
              Source: {image.filename}
            </p>
          )}
        </div>
        <button
          onClick={onZoom}
          className="flex items-center gap-1 px-3 py-1.5 text-sm font-medium text-primary bg-primary/10 hover:bg-primary/20 rounded-lg transition-colors"
        >
          <ZoomIn className="w-4 h-4" />
          Expand
        </button>
      </div>
      
      {/* Image */}
      <div className="bg-muted rounded-lg p-4 mb-3">
        <img 
          src={imageUrl}
          alt={image.ocr_text || `Image ${index + 1}`}
          className="max-w-full h-auto rounded-md mx-auto cursor-pointer hover:opacity-90 transition-opacity"
          onClick={onZoom}
          onError={(e) => {
            e.currentTarget.style.display = 'none'
          }}
        />
      </div>
      
      {/* Caption */}
      {(image.figure_number || image.caption || (image.captions && image.captions.length > 0)) && (
        <div className="bg-primary/5 border-l-4 border-primary p-3 rounded-r-lg mb-3">
          <div className="flex items-start gap-2">
            {image.figure_number && (
              <span className="badge badge-primary text-xs font-bold whitespace-nowrap">
                Figure {image.figure_number}
              </span>
            )}
            <div className="flex-1">
              {image.caption ? (
                <p className="text-sm font-medium">{image.caption}</p>
              ) : image.captions && image.captions.length > 0 ? (
                <div className="space-y-1">
                  {image.captions.map((cap, i) => (
                    <p key={i} className="text-sm">{cap}</p>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}
      
      {/* Description */}
      {(image.visual_description || image.ocr_text) && (
        <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-400 p-3 rounded-r-lg mb-3">
          <p className="text-xs font-semibold text-amber-800 dark:text-amber-300 mb-1">
            📝 Image Description:
          </p>
          <p className="text-sm text-muted-foreground">
            {image.visual_description 
              ? (image.visual_description.length > 200 ? `${image.visual_description.substring(0, 200)}...` : image.visual_description)
              : (image.ocr_text && image.ocr_text.length > 200 ? `${image.ocr_text.substring(0, 200)}...` : image.ocr_text)
            }
          </p>
        </div>
      )}
      
      {/* Study Tip */}
      <div className="pt-3 border-t border-border">
        <p className="text-xs text-muted-foreground">
          <strong>💡 Tip:</strong> {getDomainTip()}
        </p>
      </div>
    </div>
  )
}

