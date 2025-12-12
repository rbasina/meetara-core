'use client'

import { useState, useEffect, useRef } from 'react'
import { 
  Sidebar, 
  ChatMessage, 
  InputArea, 
  ImageLightbox, 
  UploadModal, 
  LoadingIndicator,
  EmptyState 
} from '@/components'
import { ThemeProvider } from '@/context/ThemeContext'
import { Message, Domain, Category, CategorizedDomainsResponse, ChatResponse } from '@/lib/types'
import { generateId } from '@/lib/utils'

function MeetaraChat() {
  // State
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [domains, setDomains] = useState<Domain[]>([])
  const [categories, setCategories] = useState<{ [key: string]: Category }>({})
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['healthcare', 'business', 'education'])
  )
  const [detectedDomain, setDetectedDomain] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showUpload, setShowUpload] = useState(false)
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [selectedImageIndex, setSelectedImageIndex] = useState<number | null>(null)
  const [expandedImageMessageId, setExpandedImageMessageId] = useState<string | null>(null)
  const [expandedImageSections, setExpandedImageSections] = useState<Set<string>>(new Set())
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Load conversation history from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem('meetara_conversation_history')
      if (stored) {
        const parsed = JSON.parse(stored)
        const loadedMessages = parsed.map((msg: Message & { timestamp: string }) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
        setMessages(loadedMessages)
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error)
    }
    
    // Load session ID
    try {
      const storedSession = localStorage.getItem('meetara_session_id')
      if (storedSession) {
        setSessionId(storedSession)
      }
    } catch (error) {
      console.error('Failed to load session ID:', error)
    }
    
    // Load expanded categories
    try {
      const savedCategories = localStorage.getItem('meetara_expanded_categories')
      if (savedCategories) {
        setExpandedCategories(new Set(JSON.parse(savedCategories)))
      }
    } catch (error) {
      console.error('Failed to load expanded categories:', error)
    }
  }, [])

  // Save conversation history
  useEffect(() => {
    if (messages.length > 0) {
      try {
        localStorage.setItem('meetara_conversation_history', JSON.stringify(messages))
      } catch (error) {
        console.error('Failed to save conversation history:', error)
      }
    }
  }, [messages])

  // Save session ID
  useEffect(() => {
    if (sessionId) {
      localStorage.setItem('meetara_session_id', sessionId)
    }
  }, [sessionId])

  // Load domains on mount
  useEffect(() => {
    loadDomains()
  }, [])

  const loadDomains = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/chat/domains/categorized')
      const data: CategorizedDomainsResponse = await response.json()
      
      setCategories(data.categories || {})
      
      // Flatten for backward compatibility
      const flatDomains: Domain[] = []
      Object.values(data.categories || {}).forEach(category => {
        category.domains.forEach(domain => {
          flatDomains.push({
            name: domain.name,
            count: domain.chunks,
            status: domain.has_content ? 'active' : 'empty'
          })
        })
      })
      setDomains(flatDomains)
    } catch (error) {
      console.error('Failed to load domains:', error)
      // Fallback to old endpoint
      try {
        const response = await fetch('http://localhost:8000/api/vectorstore/all')
        const data = await response.json()
        const domainData = Object.values(data.domains || {}).map((domainInfo: unknown) => {
          const info = domainInfo as { domain: string; stats?: { count?: number }; status?: string }
          return {
            name: info.domain,
            count: info.stats?.count || 0,
            status: (info.status === 'active' ? 'active' : 'empty') as 'active' | 'empty'
          }
        })
        setDomains(domainData)
      } catch (fallbackError) {
        console.error('Fallback also failed:', fallbackError)
      }
    }
  }

  const toggleCategory = (categoryName: string) => {
    setExpandedCategories(prev => {
      const newSet = new Set(prev)
      if (newSet.has(categoryName)) {
        newSet.delete(categoryName)
      } else {
        newSet.add(categoryName)
      }
      localStorage.setItem('meetara_expanded_categories', JSON.stringify(Array.from(newSet)))
      return newSet
    })
  }

  const sendMessage = async () => {
    if (!input.trim()) return

    const currentInput = input.trim()
    const currentDetectedDomain = detectedDomain
    const requestTime = new Date()
    
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: currentInput,
      timestamp: requestTime,
      requestTimestamp: requestTime.toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setDetectedDomain(null)
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: currentInput,
          session_id: sessionId,
          topic: currentDetectedDomain || undefined,
          context: {},
          model: selectedModel || undefined  // Include selected model
        }),
        signal: AbortSignal.timeout(120000)
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`)
      }

      const data: ChatResponse = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      // Determine RAG status
      let ragStatus: 'rag' | 'llm' | 'mixed' = 'llm'
      if (data.response.includes('Based on the documents') || 
          data.response.includes('According to the information')) {
        ragStatus = 'rag'
      } else if (data.response.includes('general knowledge')) {
        ragStatus = 'llm'
      } else if (data.response.includes('context') && 
                 data.response.includes('general knowledge')) {
        ragStatus = 'mixed'
      }

      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: data.response,
        domain: data.domain,
        confidence: data.confidence,
        ragStatus,
        images: data.images || [],
        timestamp: new Date(),
        responseTimestamp: data.response_timestamp ?? data.responseTimestamp ?? new Date().toISOString()
      }

      setMessages(prev => [...prev, assistantMessage])
      
      if (data.session_id) {
        setSessionId(data.session_id)
      }

      loadDomains()

    } catch (error: unknown) {
      console.error('Failed to send message:', error)
      
      let errorMsg = 'Sorry, I encountered an error. Please try again.'
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMsg = 'Request timed out. The server may be slow or unresponsive.'
        } else if (error.message) {
          errorMsg = error.message
        }
      }
      
      const errorMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: errorMsg,
        error: true,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleImageClick = (index: number, messageId: string) => {
    setSelectedImageIndex(index)
    setExpandedImageMessageId(messageId)
  }

  const handleToggleImageSection = (messageId: string) => {
    const sectionId = `images-${messageId}`
    setExpandedImageSections(prev => {
      const newSet = new Set(prev)
      if (newSet.has(sectionId)) {
        newSet.delete(sectionId)
      } else {
        newSet.add(sectionId)
      }
      return newSet
    })
  }

  const handleClearHistory = () => {
    if (confirm('Clear all conversation history?')) {
      setMessages([])
      localStorage.removeItem('meetara_conversation_history')
      setSessionId(null)
      localStorage.removeItem('meetara_session_id')
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion)
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <Sidebar
        categories={categories}
        domains={domains}
        sessionId={sessionId}
        messageCount={messages.length}
        onUploadClick={() => setShowUpload(true)}
        onClearHistory={handleClearHistory}
        expandedCategories={expandedCategories}
        onToggleCategory={toggleCategory}
      />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          {messages.length === 0 ? (
            <EmptyState onSuggestionClick={handleSuggestionClick} />
          ) : (
            <>
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  onImageClick={handleImageClick}
                  onToggleImageSection={handleToggleImageSection}
                  expandedImageSections={expandedImageSections}
                />
              ))}
              
              {isLoading && <LoadingIndicator />}
            </>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <InputArea
          value={input}
          onChange={setInput}
          onSend={sendMessage}
          isLoading={isLoading}
          messageCount={messages.length}
          categories={categories}
          detectedDomain={detectedDomain}
          onDetectedDomainChange={setDetectedDomain}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
        />
      </main>

      {/* Upload Modal */}
      <UploadModal
        isOpen={showUpload}
        onClose={() => setShowUpload(false)}
        onUploadComplete={loadDomains}
        domains={domains}
      />

      {/* Image Lightbox */}
      <ImageLightbox
        messages={messages}
        selectedImageIndex={selectedImageIndex}
        expandedImageMessageId={expandedImageMessageId}
        onClose={() => {
          setSelectedImageIndex(null)
          setExpandedImageMessageId(null)
        }}
        onNavigate={setSelectedImageIndex}
      />
    </div>
  )
}

// Wrap with ThemeProvider
export default function MeetaraChatPage() {
  return (
    <ThemeProvider>
      <MeetaraChat />
    </ThemeProvider>
  )
}
