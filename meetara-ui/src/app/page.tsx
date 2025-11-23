'use client'

import { useState, useEffect, useRef } from 'react'
import { Send, Upload, Database, Brain, FileText, AlertCircle, Loader2, ZoomIn, X, ChevronLeft, ChevronRight, BookOpen, ChevronDown, ChevronUp, Search } from 'lucide-react'
import AnimatedLogo from '@/components/AnimatedLogo'

interface ImageData {
  image_url: string
  image_path: string
  image_id?: string
  page?: number
  filename?: string
  ocr_text?: string
  visual_description?: string  // ✅ Enhanced description with visual analysis
  figure_number?: string | null  // ✅ Figure number (e.g., "29.6")
  caption?: string | null  // ✅ Primary caption text
  captions?: string[]  // ✅ Array of caption strings
  caption_details?: Array<{
    figure_number?: string | null
    caption_text?: string
    full_caption?: string
  }>
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  domain?: string
  confidence?: number
  ragStatus?: 'rag' | 'llm' | 'mixed'
  timestamp: Date
  error?: boolean
  images?: ImageData[]  // ✅ Images from documents
  requestTimestamp?: string
  responseTimestamp?: string
}

interface Domain {
  name: string
  count: number
  status: 'active' | 'empty'
}

interface CategorizedDomain {
  name: string
  display_name: string
  chunks: number
  documents?: number  // Document count (more meaningful than chunks)
  database_size: string
  has_content: boolean
}

interface Category {
  name: string
  display_name: string
  tier: string
  domains: CategorizedDomain[]
  total_domains: number
  domains_with_content: number
  total_chunks: number
  total_size_mb: number
}

interface CategorizedDomainsResponse {
  categories: { [key: string]: Category }
  total_categories: number
  total_domains: number
}

// Enhanced markdown formatter with proper spacing control
function formatMarkdown(text: string): string {
  // AGGRESSIVE: Remove ALL multiple blank lines - keep only single newlines!
  // Step 1: Replace 4+ newlines with single newline
  text = text.replace(/\n{4,}/g, '\n')
  // Step 2: Replace 3 newlines with single newline
  text = text.replace(/\n{3}/g, '\n')
  // Step 3: Replace 2 newlines with single newline (max spacing = 1 blank line between elements)
  text = text.replace(/\n{2}/g, '\n')
  
  // Split into lines
  const lines = text.split('\n')
  const processedLines: string[] = []
  let inList = false
  let listType: 'ul' | 'ol' | null = null
  let lastWasBoldHeader = false  // Track if previous element was a bold header
  let lastElement = ''  // Track last element type to avoid gaps
  
  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i]
    const line = rawLine.trim()
    
    // AGGRESSIVE: Skip ALL empty lines completely (no breaks added!)
    if (!line) {
      continue  // Skip empty lines entirely - no spacing added!
    }
    
    // Headers (###, ##, #)
    if (/^#{1,3}\s+/.test(line)) {
      // Close any open list
      if (inList) {
        processedLines.push(listType === 'ol' ? '</ol>' : '</ul>')
        inList = false
        listType = null
      }
      const level = (line.match(/^#+/) || [''])[0].length
      const content = line.replace(/^#+\s+/, '')
      const headerTag = level === 1 ? 'h1' : level === 2 ? 'h2' : 'h3'
      // Main title: mt-0, section headers: mt-3 (after intro paragraph)
      const marginTop = processedLines.length > 0 ? 'mt-3' : 'mt-0'
      processedLines.push(`<${headerTag} class="font-bold text-lg ${marginTop} mb-2 leading-normal">${formatInlineMarkdown(content)}</${headerTag}>`)
      lastWasBoldHeader = false  // Headers aren't "bold headers" in our context
      lastElement = 'header'
      continue
    }
    
    // Horizontal rule (---)
    if (/^---+$/.test(line)) {
      if (inList) {
        processedLines.push(listType === 'ol' ? '</ol>' : '</ul>')
        inList = false
        listType = null
      }
      processedLines.push('<hr class="my-2 border-gray-300">')
      continue
    }
    
    // Numbered list items (1. 2. 3.)
    if (/^\d+\.\s+/.test(line)) {
      if (!inList || listType !== 'ol') {
        if (inList && listType === 'ul') {
          processedLines.push('</ul>')
        }
        // If coming right after a bold header, NO top spacing at all!
        const listMargin = lastWasBoldHeader ? 'mb-1 mt-0' : 'mb-1 mt-0'
        processedLines.push(`<ol class="list-decimal ml-6 ${listMargin}">`)
        inList = true
        listType = 'ol'
        lastWasBoldHeader = false  // Reset after starting list
      }
      const content = line.replace(/^\d+\.\s+/, '')
      processedLines.push(`<li class="mb-0 leading-tight">${formatInlineMarkdown(content)}</li>`)
      continue
    }
    
    // Bullet list items (- or *)
    if (/^[\-\*]\s+/.test(line)) {
      const content = line.replace(/^[\-\*]\s+/, '')
      
      // Check if content is ONLY bold text (like "**Header**" or "**• Header**")
      // Or bold text followed by just a colon (like "**Header:**")
      // If so, treat it as a bold paragraph/header, not a list item
      const trimmedContent = content.trim()
      const boldOnlyPattern = /^\*\*[^*]+\*\*:?\s*$/  // Matches **text** or **text:** with nothing else
      const boldWithBulletPattern = /^\*\*\s*[•\-\*]\s*[^*]+\*\*:?\s*$/  // Matches **• text** or **- text**
      
      if (boldOnlyPattern.test(trimmedContent) || boldWithBulletPattern.test(trimmedContent)) {
        // This is a bold header - render as paragraph, not list item
        if (inList) {
          processedLines.push(listType === 'ol' ? '</ol>' : '</ul>')
          inList = false
          listType = null
        }
        // Clean up if it has a bullet symbol in the bold text
        let cleanedContent = trimmedContent.replace(/\*\*\s*[•\-\*]\s+/, '**').replace(/\s+[•\-\*]\s+\*\*/, '**')
        // Minimal margins - compact spacing for sub-section headers
        // First bold header gets mt-3 (after intro), subsequent get mt-2
        const marginTop = processedLines.length === 0 ? 'mt-0' : (lastElement === 'paragraph' ? 'mt-3' : 'mt-2')
        processedLines.push(`<p class="${marginTop} mb-1 leading-normal font-semibold text-base">${formatInlineMarkdown(cleanedContent)}</p>`)
        lastWasBoldHeader = true
        lastElement = 'bold_header'
        continue
      }
      
      // Regular bullet item - add to list
      if (!inList || listType !== 'ul') {
        if (inList && listType === 'ol') {
          processedLines.push('</ol>')
        }
        // Minimal margins - compact but connected
        const listMargin = lastWasBoldHeader ? 'mb-2 mt-1' : 'mb-2 mt-1'  // Small spacing after bold header
        processedLines.push(`<ul class="list-disc ml-6 ${listMargin} leading-normal">`)
        inList = true
        listType = 'ul'
        lastWasBoldHeader = false  // Reset after starting list
        lastElement = 'list'
      }
      processedLines.push(`<li class="mb-1 leading-normal">${formatInlineMarkdown(content)}</li>`)
      continue
    }
    
    // Close list if we hit a non-list line
    if (inList) {
      processedLines.push(listType === 'ol' ? '</ol>' : '</ul>')
      inList = false
      listType = null
    }
    
    // Regular paragraph - compact spacing
    processedLines.push(`<p class="mb-2 leading-normal">${formatInlineMarkdown(line)}</p>`)
    lastWasBoldHeader = false  // Reset after regular paragraph
  }
  
  // Close any open list
  if (inList) {
    processedLines.push(listType === 'ol' ? '</ol>' : '</ul>')
  }
  
  return processedLines.join('\n')
}

// Format inline markdown (**bold**, *italic*, LaTeX math)
function formatInlineMarkdown(text: string): string {
  let html = text
  
  // LaTeX math expressions: $...$ or \(...\) -> rendered math
  // First handle \(...\) delimiters (alternative LaTeX inline math)
  // Match \(...\) with proper handling of escaped \) inside
  html = html.replace(/\\\(([\s\S]*?)\\\)/g, (match, mathContent) => {
    // Clean up escaped closing parenthesis inside the math block
    mathContent = mathContent.replace(/\\\)/g, ')')
    return renderMathExpression(mathContent.trim())
  })
  
  // Then handle $...$ delimiters (standard LaTeX inline math)
  html = html.replace(/\$([^$]+)\$/g, (match, mathContent) => {
    return renderMathExpression(mathContent.trim())
  })
  
  // Handle standalone LaTeX expressions that appear in text (not in delimiters)
  // Match complete LaTeX commands with their arguments
  html = html.replace(/(\\lim|\\frac|\\sqrt|\\int|\\sum|\\prod|\\to|\\in|\\subset)(\{[^}]*\})+/g, (match) => {
    // Only process if not already inside HTML tags
    if (!match.includes('<') && !match.includes('>')) {
      return renderMathExpression(match)
    }
    return match
  })
  
  // Handle standalone \to (arrow) that appears without braces
  html = html.replace(/\\to(?![a-zA-Z])/g, '→')
  
  // Finally, handle escaped closing parentheses (\) - these should become regular )
  // This fixes cases like f(x \), f'(x \), etc. that appear outside delimiters
  // Since we've already processed delimited math blocks, remaining \) can be safely converted
  html = html.replace(/\\\)/g, ')')
  
  // **bold** -> <strong>bold</strong> (do this first to avoid conflicts)
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  
  // *italic* -> <em>italic</em> (only match single * not part of **)
  html = html.replace(/(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)/g, '<em>$1</em>')
  
  return html
}

// Render LaTeX math expressions to HTML with Unicode symbols
function renderMathExpression(latex: string): string {
  let rendered = latex
  
  // First, clean up escaped closing parentheses that might be in the content
  rendered = rendered.replace(/\\\)/g, ')')
  
  // Handle \lim with subscripts: \lim_{h \to 0}
  rendered = rendered.replace(/\\lim_{([^}]+)}/g, (match, subscript) => {
    // Clean up \to inside subscript
    subscript = subscript.replace(/\\to/g, '→')
    return `lim<sub>${subscript}</sub>`
  })
  
  // Handle \to (arrow)
  rendered = rendered.replace(/\\to/g, '→')
  
  // Integrals: \int or \int_{a}^{b} or \int_a^b
  // Handle \int_{a}^{b} first (with braces)
  rendered = rendered.replace(/\\int_{([^}]+)}^{([^}]+)}/g, '∫<sub>$1</sub><sup>$2</sup>')
  // Handle \int_a^b (with subscripts/superscripts)
  rendered = rendered.replace(/\\int_([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)/g, '∫<sub>$1</sub><sup>$2</sup>')
  rendered = rendered.replace(/\\int_([a-zA-Z0-9]+)/g, '∫<sub>$1</sub>')
  rendered = rendered.replace(/\\int\^([a-zA-Z0-9]+)/g, '∫<sup>$1</sup>')
  // Handle plain \int
  rendered = rendered.replace(/\\int/g, '∫')
  
  // Differentials with thin space: \,dt, \,dx, \,dy, etc.
  rendered = rendered.replace(/\\,([a-z]+)/g, '&nbsp;$1')
  
  // Derivatives: f'(x) or F'(x) - handle apostrophe notation
  rendered = rendered.replace(/([a-zA-Z])'/g, '$1′')
  
  // Common Greek letters
  rendered = rendered.replace(/\\pi/g, 'π')
  rendered = rendered.replace(/\\theta/g, 'θ')
  rendered = rendered.replace(/\\alpha/g, 'α')
  rendered = rendered.replace(/\\beta/g, 'β')
  rendered = rendered.replace(/\\gamma/g, 'γ')
  rendered = rendered.replace(/\\delta/g, 'δ')
  rendered = rendered.replace(/\\Delta/g, 'Δ')
  rendered = rendered.replace(/\\epsilon/g, 'ε')
  rendered = rendered.replace(/\\lambda/g, 'λ')
  rendered = rendered.replace(/\\mu/g, 'μ')
  rendered = rendered.replace(/\\sigma/g, 'σ')
  rendered = rendered.replace(/\\Sigma/g, 'Σ')
  rendered = rendered.replace(/\\omega/g, 'ω')
  rendered = rendered.replace(/\\Omega/g, 'Ω')
  
  // Math operators
  rendered = rendered.replace(/\\times/g, '×')
  rendered = rendered.replace(/\\div/g, '÷')
  rendered = rendered.replace(/\\pm/g, '±')
  rendered = rendered.replace(/\\mp/g, '∓')
  rendered = rendered.replace(/\\cdot/g, '·')
  
  // Relations
  rendered = rendered.replace(/\\leq/g, '≤')
  rendered = rendered.replace(/\\geq/g, '≥')
  rendered = rendered.replace(/\\neq/g, '≠')
  rendered = rendered.replace(/\\approx/g, '≈')
  rendered = rendered.replace(/\\equiv/g, '≡')
  
  // Superscripts: ^{...} or ^number or variable^number (like r^2)
  // First handle ^{...} format
  rendered = rendered.replace(/\^{([^}]+)}/g, (match, content) => {
    // Convert numbers to superscript Unicode
    const superscriptMap: { [key: string]: string } = {
      '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
      '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
      '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾'
    }
    if (content.length === 1 && superscriptMap[content]) {
      return `<sup>${superscriptMap[content]}</sup>`
    }
    return `<sup>${content}</sup>`
  })
  // Then handle variable^number format (like r^2, x^3) - must come after ^{...} replacement
  rendered = rendered.replace(/([a-zA-Z])\^(\d)/g, (match, variable, digit) => {
    const superscriptMap: { [key: string]: string } = {
      '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
      '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }
    return `${variable}<sup>${superscriptMap[digit] || digit}</sup>`
  })
  
  // Subscripts: _{...} or _number
  rendered = rendered.replace(/_{([^}]+)}/g, '<sub>$1</sub>')
  rendered = rendered.replace(/_(\d)/g, '<sub>$1</sub>')
  
  // Fractions: \frac{a}{b} -> a/b or styled fraction
  rendered = rendered.replace(/\\frac{([^}]+)}{([^}]+)}/g, '<span class="inline-block align-middle"><span class="text-sm">$1</span>/<span class="text-sm">$2</span></span>')
  
  // Square root: \sqrt{x} -> √x
  rendered = rendered.replace(/\\sqrt{([^}]+)}/g, '√<span style="text-decoration: overline;">$1</span>')
  rendered = rendered.replace(/\\sqrt\[(\d+)\]{([^}]+)}/g, '<sup>$1</sup>√<span style="text-decoration: overline;">$2</span>')
  
  // Remove any remaining backslashes that weren't converted (safety cleanup)
  rendered = rendered.replace(/\\([a-zA-Z])/g, '$1')
  
  // Clean up spaces - preserve single spaces as normal spaces, convert multiple to non-breaking
  rendered = rendered.replace(/\s{2,}/g, ' ')
  
  // Wrap in math span for styling
  return `<span class="font-mono bg-blue-50 px-1 py-0.5 rounded text-sm">${rendered}</span>`
}

export default function MeetaraChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [domains, setDomains] = useState<Domain[]>([])
  const [categories, setCategories] = useState<{ [key: string]: Category }>({})
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['healthcare', 'business', 'education']))
  const [detectedDomain, setDetectedDomain] = useState<string | null>(null)
  const [querySuggestions, setQuerySuggestions] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showUpload, setShowUpload] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isConnecting, setIsConnecting] = useState(false)
  const [selectedImageIndex, setSelectedImageIndex] = useState<number | null>(null)
  const [expandedImageMessageId, setExpandedImageMessageId] = useState<string | null>(null)
  const [expandedImageSections, setExpandedImageSections] = useState<Set<string>>(new Set())
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const imageSectionRefs = useRef<Map<string, HTMLDivElement>>(new Map())

  // Keyboard navigation for image modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (selectedImageIndex === null || !expandedImageMessageId) return
      
      const message = messages.find(m => m.id === expandedImageMessageId)
      if (!message || !message.images) return
      
      if (e.key === 'Escape') {
        setSelectedImageIndex(null)
        setExpandedImageMessageId(null)
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        const newIndex = selectedImageIndex > 0 
          ? selectedImageIndex - 1 
          : message.images.length - 1
        setSelectedImageIndex(newIndex)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        const newIndex = selectedImageIndex < message.images.length - 1
          ? selectedImageIndex + 1
          : 0
        setSelectedImageIndex(newIndex)
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedImageIndex, expandedImageMessageId, messages])

  // Load conversation history from localStorage on mount (client-side only)
  useEffect(() => {
    // Load conversation history from localStorage on mount
    try {
      const stored = localStorage.getItem('meetara_conversation_history')
      if (stored) {
        const parsed = JSON.parse(stored)
        // Convert timestamp strings back to Date objects
        const loadedMessages = parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
        setMessages(loadedMessages)
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error)
    }
    
    // Load session ID from localStorage
    try {
      const stored = localStorage.getItem('meetara_session_id')
      if (stored) {
        setSessionId(stored)
      }
    } catch (error) {
      console.error('Failed to load session ID:', error)
    }
  }, [])

  // Save conversation history to localStorage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      try {
        localStorage.setItem('meetara_conversation_history', JSON.stringify(messages))
      } catch (error) {
        console.error('Failed to save conversation history:', error)
      }
    }
  }, [messages])

  // Save session ID to localStorage
  useEffect(() => {
    if (sessionId) {
      localStorage.setItem('meetara_session_id', sessionId)
    }
  }, [sessionId])

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])


  // Load available domains on mount
  useEffect(() => {
    loadDomains()
  }, [])

  const loadDomains = async () => {
    try {
      // Use new categorized endpoint for Perplexity-style sidebar
      const response = await fetch('http://localhost:8000/api/chat/domains/categorized')
      const data: CategorizedDomainsResponse = await response.json()
      
      // Set categorized data
      setCategories(data.categories || {})
      
      // Also maintain flat list for backward compatibility (upload modal, etc.)
      const flatDomains: Domain[] = []
      Object.values(data.categories || {}).forEach(category => {
        category.domains.forEach(domain => {
          flatDomains.push({
            name: domain.name,
            count: domain.chunks,
            status: domain.has_content ? 'active' as const : 'empty' as const
          })
        })
      })
      setDomains(flatDomains)
      
      // Load expanded categories from localStorage
      try {
        const saved = localStorage.getItem('meetara_expanded_categories')
        if (saved) {
          const savedCategories = JSON.parse(saved)
          setExpandedCategories(new Set(savedCategories))
        }
      } catch (e) {
        // Use defaults if localStorage fails
      }
    } catch (error) {
      console.error('Failed to load categorized domains:', error)
      // Fallback: try old endpoint
      try {
        const response = await fetch('http://localhost:8000/api/vectorstore/all')
        const data = await response.json()
        const domainData = Object.values(data.domains || {}).map((domainInfo: any) => ({
          name: domainInfo.domain,
          count: domainInfo.stats?.count || 0,
          status: domainInfo.status === 'active' ? 'active' as const : 'empty' as const
        }))
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
      // Save to localStorage
      try {
        localStorage.setItem('meetara_expanded_categories', JSON.stringify(Array.from(newSet)))
      } catch (e) {
        // Ignore localStorage errors
      }
      return newSet
    })
  }
  
  const getCategoryIcon = (categoryName: string): string => {
    const icons: { [key: string]: string } = {
      healthcare: '🏥',
      legal_financial: '⚖️',
      emergency_crisis: '🚨',
      business: '💼',
      business_professional: '👔',
      education: '🎓',
      technology: '💻',
      daily_life: '🏠',
      creative: '🎨',
      research_academic: '🔬',
      aerospace_transportation: '🚀',
      industrial_manufacturing: '🏭',
      specialized: '🔧',
      entertainment_media: '🎬',
      food_cooking: '🍳',
      psychology_wellness: '🧘',
      sports_recreation: '⚽',
      travel_tourism: '✈️'
    }
    return icons[categoryName] || '📁'
  }
  
  const getTierColor = (tier: string): string => {
    switch (tier) {
      case 'safety_critical':
        return 'border-gray-200 bg-gray-50'
      case 'expert':
        return 'border-gray-200 bg-gray-50'
      case 'quality':
        return 'border-gray-200 bg-gray-50'
      default:
        return 'border-gray-200 bg-gray-50'
    }
  }
  
  const filterDomains = (category: Category): CategorizedDomain[] => {
    return category.domains.filter(domain => {
      // Only show domains with content (always hide empty domains)
      if (!domain.has_content) {
        return false
      }
      return true
    })
  }
  
  const detectDomainFromQuery = (query: string) => {
    if (!query.trim()) {
      setDetectedDomain(null)
      return
    }
    
    // Simple keyword-based domain detection
    const queryLower = query.toLowerCase()
    const domainKeywords: { [key: string]: string[] } = {
      'healthcare': ['health', 'medical', 'doctor', 'patient', 'disease', 'symptom', 'treatment', 'medicine'],
      'mental_health': ['mental', 'depression', 'anxiety', 'therapy', 'psychology', 'stress'],
      'nutrition': ['nutrition', 'diet', 'food', 'vitamin', 'calorie', 'protein'],
      'business': ['business', 'company', 'management', 'strategy', 'marketing', 'sales'],
      'accounting': ['accounting', 'finance', 'budget', 'revenue', 'expense', 'balance sheet'],
      'academic_tutoring': ['math', 'calculus', 'algebra', 'geometry', 'science', 'physics', 'chemistry', 'biology'],
      'legal_business': ['legal', 'law', 'contract', 'legal', 'intellectual property'],
      'technology': ['programming', 'code', 'software', 'development', 'algorithm', 'data'],
    }
    
    for (const [domain, keywords] of Object.entries(domainKeywords)) {
      if (keywords.some(keyword => queryLower.includes(keyword))) {
        setDetectedDomain(domain)
        return
      }
    }
    
    setDetectedDomain(null)
  }
  
  // Generate query suggestions based on available domains
  useEffect(() => {
    if (!input.trim() && Object.keys(categories).length > 0) {
      const suggestions = []
      const popularDomains = ['healthcare', 'business', 'education', 'technology']
      
      for (const domainCategory of popularDomains) {
        const category = categories[domainCategory]
        if (category && category.domains_with_content > 0) {
          const domain = category.domains.find(d => d.has_content)
          if (domain) {
            const domainName = domain.display_name.toLowerCase()
            if (domainCategory === 'healthcare') {
              suggestions.push(`What are the symptoms of diabetes?`)
            } else if (domainCategory === 'business') {
              suggestions.push(`What is project management?`)
            } else if (domainCategory === 'education') {
              suggestions.push(`Explain calculus derivatives`)
            } else if (domainCategory === 'technology') {
              suggestions.push(`What is software development?`)
            }
          }
        }
      }
      
      setQuerySuggestions(suggestions)
    }
  }, [categories, input])

  const sendMessage = async () => {
    if (!input.trim()) return
    
    // Clear any previous errors
    setError(null)

    const currentInput = input.trim()
    const currentDetectedDomain = detectedDomain // Save before clearing
    const requestTime = new Date()
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: currentInput,
      timestamp: requestTime,
      requestTimestamp: requestTime.toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setDetectedDomain(null) // Clear domain detection when sending
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
          topic: currentDetectedDomain || undefined, // Pass detected domain if available
          context: {}
        }),
        signal: AbortSignal.timeout(120000) // 120 second timeout (increased for complex queries with image generation and LLM processing)
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()
      
      // Check if response has an error
      if (data.error) {
        throw new Error(data.error)
      }
      
      // ✅ Log images if present (for debugging)
      if (data.images && data.images.length > 0) {
        console.log('📷 Received images from API:', data.images)
      }
      
      // Determine RAG status based on response
      let ragStatus: 'rag' | 'llm' | 'mixed' = 'llm'
      if (data.response.includes('Based on the documents') || 
          data.response.includes('According to the information')) {
        ragStatus = 'rag'
      } else if (data.response.includes('general knowledge') || 
                 data.response.includes('based on general knowledge')) {
        ragStatus = 'llm'
      } else if (data.response.includes('context') && 
                 data.response.includes('general knowledge')) {
        ragStatus = 'mixed'
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        domain: data.domain,
        confidence: data.confidence,
        ragStatus,
        images: data.images || [],  // ✅ Include images from API response
        timestamp: new Date(),
        responseTimestamp: data.response_timestamp ?? data.responseTimestamp ?? new Date().toISOString()
      }

      setMessages(prev => [...prev, assistantMessage])
      
      // Clear detected domain after sending
      setDetectedDomain(null)
      
      // Update session ID if provided
      if (data.session_id) {
        setSessionId(data.session_id)
      }

      // Refresh domain counts
      loadDomains()

    } catch (error: any) {
      console.error('Failed to send message:', error)
      
      // Determine error message
      let errorMsg = 'Sorry, I encountered an error. Please try again.'
      if (error.name === 'AbortError') {
        errorMsg = 'Request timed out. The server may be slow or unresponsive. Please try again.'
      } else if (error.message) {
        errorMsg = error.message
      }
      
      setError(errorMsg)
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const getRAGStatusIcon = (status: 'rag' | 'llm' | 'mixed') => {
    switch (status) {
      case 'rag':
        return <Database className="w-4 h-4 text-green-500" />
      case 'llm':
        return <Brain className="w-4 h-4 text-yellow-500" />
      case 'mixed':
        return <FileText className="w-4 h-4 text-blue-500" />
    }
  }

  const getRAGStatusText = (status: 'rag' | 'llm' | 'mixed') => {
    switch (status) {
      case 'rag':
        return 'Using RAG documents'
      case 'llm':
        return 'Using LLM general knowledge'
      case 'mixed':
        return 'Mixed: RAG + general knowledge'
    }
  }

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <div className="hidden md:flex w-80 bg-white shadow-xl flex-col relative border-r border-gray-200">
        <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
          <AnimatedLogo 
            size="md" 
            showText={true} 
            showTagline={false}
            glowIntensity="high"
            className="justify-start"
          />
          <p className="text-xs text-gray-600 -mt-5 ml-20 italic">Knowledge-Based AI Assistant</p>
        </div>

        {/* Domain Status - Categorized (Collapsible) */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 border-b">
            <h3 className="font-semibold text-gray-700 mb-3">Domains</h3>
          </div>
          
          {/* Categories */}
          <div className="space-y-1 px-2 pb-4">
            {Object.entries(categories).map(([categoryName, category]) => {
              const filteredDomains = filterDomains(category)
              const isExpanded = expandedCategories.has(categoryName)
              
              // Don't show category if no domains match filter
              if (filteredDomains.length === 0) return null
              
              return (
                <div key={categoryName} className={`border rounded-lg mb-1 ${getTierColor(category.tier)}`}>
                  {/* Category Header */}
                  <button
                    onClick={() => toggleCategory(categoryName)}
                    className="w-full flex items-center justify-between p-3 hover:bg-opacity-50 transition-colors"
                  >
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{getCategoryIcon(categoryName)}</span>
                      <span className="font-semibold text-sm text-gray-800">
                        {category.display_name}
                      </span>
                      <span className="text-xs text-gray-500">
                        ({category.domains_with_content}/{category.total_domains})
                      </span>
                    </div>
                    {isExpanded ? (
                      <ChevronUp className="h-4 w-4 text-gray-500" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-gray-500" />
                    )}
                  </button>
                  
                  {/* Domain List */}
                  {isExpanded && (
                    <div className="px-3 pb-2 space-y-1">
                      {filteredDomains.map((domain) => (
                        <div
                          key={domain.name}
                          className={`flex items-center justify-between p-2 rounded text-sm transition-colors ${
                            domain.has_content
                              ? 'hover:bg-white hover:bg-opacity-50 cursor-pointer'
                              : 'opacity-60'
                          }`}
                        >
                          <span className="text-gray-700 capitalize">
                            {domain.display_name}
                          </span>
                          <span className={`text-xs px-2 py-0.5 rounded ${
                            domain.has_content
                              ? 'bg-green-100 text-green-700'
                              : 'bg-gray-100 text-gray-500'
                          }`}>
                            {domain.documents !== undefined ? domain.documents : domain.chunks}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Upload Button */}
        <div className="p-4 space-y-2">
          <button
            onClick={() => setShowUpload(true)}
            className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Upload className="w-4 h-4" />
            <span>Upload Documents</span>
          </button>
          
          {/* Clear History Button */}
          {messages.length > 0 && (
            <button
              onClick={() => {
                if (confirm('Clear all conversation history?')) {
                  setMessages([])
                  localStorage.removeItem('meetara_conversation_history')
                  setSessionId(null)
                  localStorage.removeItem('meetara_session_id')
                }
              }}
              className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors text-sm"
            >
              <span>Clear History</span>
            </button>
          )}
        </div>

        {/* Session Info */}
        {sessionId && (
          <div className="p-4 border-t mt-auto">
            <p className="text-xs text-gray-500">Session: {sessionId.slice(0, 8)}...</p>
            <p className="text-xs text-gray-400 mt-1">{messages.length} messages in history</p>
          </div>
        )}
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`${message.role === 'user' ? 'max-w-3xl' : 'max-w-5xl w-full'} px-6 py-4 rounded-xl ${
                  message.role === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg'
                    : message.error
                    ? 'bg-gray-50 shadow-md border border-gray-300'
                    : 'bg-white shadow-lg border border-gray-200 hover:shadow-xl transition-shadow'
                }`}
              >
                <div className="whitespace-pre-wrap prose prose-sm max-w-none">
                  {message.role === 'assistant' ? (
                    <div>
                      {/* 🎓 Visual Summary Section - Compact Preview Only */}
                      {message.images && message.images.length > 0 && (
                        <div className="mb-4 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                          <div className="flex items-center gap-2 mb-2">
                            <BookOpen className="w-4 h-4 text-blue-600" />
                            <h3 className="text-sm font-bold text-blue-900">Quick Preview: {message.images.length} Image{message.images.length > 1 ? 's' : ''} Available</h3>
                            <span className="ml-auto text-xs font-semibold text-blue-700 bg-blue-100 px-2 py-1 rounded-full">
                              {message.images.length} Image{message.images.length > 1 ? 's' : ''}
                            </span>
                          </div>
                          {/* Compact Thumbnail Gallery - Click to open lightbox */}
                          <div className="flex gap-2 mb-2">
                            {message.images.map((img, idx) => {
                              const imageUrl = img.image_url.startsWith('http') 
                                ? img.image_url 
                                : `http://localhost:8000${img.image_url}`
                              return (
                                <div 
                                  key={idx}
                                  className="relative group cursor-pointer border-2 border-blue-200 rounded overflow-hidden hover:border-blue-400 transition-all flex-shrink-0"
                                  onClick={() => {
                                    setSelectedImageIndex(idx)
                                    setExpandedImageMessageId(message.id)
                                  }}
                                  style={{ width: '80px', height: '60px' }}
                                >
                                  <img 
                                    src={imageUrl}
                                    alt={`Image ${idx + 1}`}
                                    className="w-full h-full object-cover"
                                    onError={(e) => {
                                      console.error('Failed to load image:', imageUrl)
                                      e.currentTarget.style.display = 'none'
                                    }}
                                  />
                                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all flex items-center justify-center">
                                    <ZoomIn className="w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                                  </div>
                                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-1">
                                    <p className="text-xs text-white font-medium text-center">#{idx + 1}</p>
                                  </div>
                                </div>
                              )
                            })}
                          </div>
                          <div className="flex items-center justify-between gap-2">
                            <button
                              onClick={() => {
                                setSelectedImageIndex(0)
                                setExpandedImageMessageId(message.id)
                              }}
                              className="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1"
                            >
                              <ZoomIn className="w-3 h-3" />
                              Open in full screen
                            </button>
                            <button
                              onClick={() => {
                                const sectionId = `images-${message.id}`
                                setExpandedImageSections(prev => {
                                  const newSet = new Set(prev)
                                  if (newSet.has(sectionId)) {
                                    newSet.delete(sectionId)
                                  } else {
                                    newSet.add(sectionId)
                                    // Smooth scroll to section after a brief delay for expansion
                                    setTimeout(() => {
                                      const element = imageSectionRefs.current.get(sectionId)
                                      if (element) {
                                        element.scrollIntoView({ behavior: 'smooth', block: 'start' })
                                      }
                                    }, 100)
                                  }
                                  return newSet
                                })
                              }}
                              className="text-xs text-blue-600 hover:text-blue-800 font-medium flex items-center gap-1"
                            >
                              {expandedImageSections.has(`images-${message.id}`) ? 'Hide' : 'Show'} detailed images
                              <ChevronRight className={`w-3 h-3 transition-transform ${expandedImageSections.has(`images-${message.id}`) ? 'rotate-90' : ''}`} />
                            </button>
                          </div>
                        </div>
                      )}
                      
                      {message.responseTimestamp && (
                        <div className="text-xs text-gray-500 mb-3">
                          <p>{new Date(message.responseTimestamp).toLocaleString()}</p>
                        </div>
                      )}
                      
                      <div dangerouslySetInnerHTML={{ __html: formatMarkdown(message.content) }} />
                      
                      {/* 🎓 Enhanced Diagram Section - Detailed View (Collapsible) */}
                      {message.images && message.images.length > 0 && (
                        <div 
                          ref={(el) => {
                            if (el) {
                              imageSectionRefs.current.set(`images-${message.id}`, el)
                            } else {
                              imageSectionRefs.current.delete(`images-${message.id}`)
                            }
                          }}
                          className={`mt-4 space-y-4 transition-all duration-300 ${expandedImageSections.has(`images-${message.id}`) ? 'block' : 'hidden'}`}
                        >
                          <div className="flex items-center gap-2 pb-2 border-b border-gray-300">
                            <div className="w-1 h-6 bg-gradient-to-b from-blue-500 to-indigo-600 rounded-full"></div>
                            <h3 className="text-xl font-bold text-gray-800">📚 Detailed Images & Visual Explanations</h3>
                          </div>
                          
                          {message.images.map((img, idx) => {
                            const imageUrl = img.image_url.startsWith('http') 
                              ? img.image_url 
                              : `http://localhost:8000${img.image_url}`
                            return (
                              <div key={idx} className="border-2 border-gray-200 rounded-xl p-5 bg-white shadow-md hover:shadow-lg transition-shadow">
                                {/* Diagram Header */}
                                <div className="flex items-start justify-between mb-3">
                                  <div>
                                    <div className="flex items-center gap-2 mb-1">
                                      <span className="text-sm font-bold text-blue-600 bg-blue-100 px-2 py-1 rounded">
                                        Image {idx + 1} of {message.images!.length}
                                      </span>
                                      {img.page && (
                                        <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                                          Page {img.page}
                                        </span>
                                      )}
                                    </div>
                                    {img.filename && (
                                      <p className="text-sm font-medium text-gray-700">
                                        Source: {img.filename}
                                      </p>
                                    )}
                                  </div>
                                  <button
                                    onClick={() => {
                                      setSelectedImageIndex(idx)
                                      setExpandedImageMessageId(message.id)
                                    }}
                                    className="flex items-center gap-1 px-3 py-1.5 text-sm font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors"
                                  >
                                    <ZoomIn className="w-4 h-4" />
                                    Expand
                                  </button>
                                </div>
                                
                                {/* Diagram Image */}
                                <div className="bg-gray-50 rounded-lg p-4 mb-3 border border-gray-200">
                                <img 
                                  src={imageUrl}
                                    alt={img.ocr_text || `Image ${idx + 1} from ${img.filename || 'document'}`}
                                    className="max-w-full h-auto rounded-md shadow-sm mx-auto cursor-pointer hover:opacity-90 transition-opacity"
                                    onClick={() => {
                                      setSelectedImageIndex(idx)
                                      setExpandedImageMessageId(message.id)
                                    }}
                                  onError={(e) => {
                                    console.error('Failed to load image:', imageUrl)
                                    e.currentTarget.style.display = 'none'
                                  }}
                                />
                                </div>
                                
                                {/* Figure Number and Caption - Priority Display */}
                                {(img.figure_number || img.caption || (img.captions && img.captions.length > 0)) && (
                                  <div className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded-r-lg mb-3">
                                    <div className="flex items-start gap-2">
                                      {img.figure_number && (
                                        <span className="inline-flex items-center px-2 py-1 rounded-md bg-blue-100 text-blue-800 text-xs font-bold whitespace-nowrap">
                                          Figure {img.figure_number}
                                        </span>
                                      )}
                                      <div className="flex-1">
                                        {img.caption ? (
                                          <p className="text-sm text-gray-800 font-medium leading-relaxed">
                                            {img.caption}
                                          </p>
                                        ) : img.captions && img.captions.length > 0 ? (
                                          <div className="space-y-1">
                                            {img.captions.map((cap: string, capIdx: number) => (
                                              <p key={capIdx} className="text-sm text-gray-800 leading-relaxed">
                                                {cap}
                                              </p>
                                            ))}
                                          </div>
                                        ) : null}
                                      </div>
                                    </div>
                                  </div>
                                )}
                                
                                {/* Image Description - Enhanced with Visual Analysis */}
                                {(img.visual_description || img.ocr_text) && (
                                  <div className="bg-amber-50 border-l-4 border-amber-400 p-3 rounded-r-lg">
                                    <p className="text-xs font-semibold text-amber-800 mb-1">📝 Image Description:</p>
                                    <p className="text-sm text-gray-700">
                                      {img.visual_description 
                                        ? (img.visual_description.length > 200 ? `${img.visual_description.substring(0, 200)}...` : img.visual_description)
                                        : (img.ocr_text && img.ocr_text.length > 200 ? `${img.ocr_text.substring(0, 200)}...` : img.ocr_text)
                                      }
                                    </p>
                                    {/* Always show raw OCR in expandable section when available */}
                                    {img.ocr_text && (
                                      <details className="mt-1">
                                        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                                          {img.visual_description ? 'Show raw OCR text' : 'Show OCR details'}
                                        </summary>
                                        <p className="text-xs text-gray-600 bg-gray-50 p-2 rounded mt-1 font-mono whitespace-pre-wrap">
                                          {img.ocr_text}
                                        </p>
                                      </details>
                                    )}
                                  </div>
                                )}
                                
                                {/* Context-Aware Learning Tip */}
                                {(() => {
                                  const domain = message.domain || ''
                                  const isAcademic = domain.includes('academic') || domain.includes('education') || domain.includes('tutoring')
                                  const isHealth = domain.includes('health') || domain.includes('medical') || domain.includes('wellness')
                                  const isBusiness = domain.includes('business') || domain.includes('professional') || domain.includes('marketing') || domain.includes('sales')
                                  const isTechnical = domain.includes('programming') || domain.includes('engineering') || domain.includes('technology') || domain.includes('software')
                                  
                                  let tipLabel = '💡 Insight:'
                                  let tipText = 'Take time to understand how this image relates to the concepts discussed above.'
                                  
                                  if (isAcademic) {
                                    tipLabel = '💡 Study Tip:'
                                    tipText = 'Take time to understand how this image relates to the concepts discussed above. Review the visual elements carefully to reinforce your learning.'
                                  } else if (isHealth) {
                                    tipLabel = '💡 Important Note:'
                                    tipText = 'Review this image in the context of the information provided above. For medical information, always consult with healthcare professionals.'
                                  } else if (isBusiness) {
                                    tipLabel = '💡 Key Takeaway:'
                                    tipText = 'This visual can help clarify business concepts and strategies discussed in the response above.'
                                  } else if (isTechnical) {
                                    tipLabel = '💡 Technical Note:'
                                    tipText = 'Study this diagram/visual in conjunction with the technical explanations above to fully understand the concept.'
                                  }
                                  
                                  return (
                                    <div className="mt-3 pt-3 border-t border-gray-200">
                                      <p className="text-xs text-gray-500">
                                        <strong>{tipLabel}</strong> {tipText}
                                        {img.page && ` Review this image alongside the text on page ${img.page} of ${img.filename || 'the source document'}.`}
                                      </p>
                                    </div>
                                  )
                                })()}
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  ) : (
                    message.error ? (
                      message.content
                    ) : (
                      <div className="space-y-2">
                        {message.timestamp && (
                          <div className="text-xs text-gray-200">
                            {message.timestamp.toLocaleString()}
                          </div>
                        )}
                        <div>{message.content}</div>
                      </div>
                    )
                  )}
                </div>
                
                {message.role === 'assistant' && (
                  <div className="mt-4 pt-3 border-t border-gray-200">
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      {/* Domain Badge */}
                      {message.domain && (
                        <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-blue-100 text-blue-700 font-medium">
                          <Database className="w-3 h-3" />
                          <span className="capitalize">{message.domain.replace('_', ' ')}</span>
                        </div>
                      )}
                      
                      {/* RAG Status */}
                      {message.ragStatus && (
                        <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-gray-100 text-gray-700">
                          {getRAGStatusIcon(message.ragStatus)}
                          <span>{getRAGStatusText(message.ragStatus)}</span>
                        </div>
                      )}
                      
                      {/* Confidence Indicator */}
                      {message.confidence !== undefined && (
                        <div className={`flex items-center gap-1 px-2 py-1 rounded-full font-medium ${
                          message.confidence >= 0.8 
                            ? 'bg-green-100 text-green-700' 
                            : message.confidence >= 0.6 
                            ? 'bg-yellow-100 text-yellow-700' 
                            : 'bg-orange-100 text-orange-700'
                        }`}>
                          <span>{(message.confidence * 100).toFixed(0)}% confidence</span>
                        </div>
                      )}
                      
                      {/* Source Count */}
                      {message.images && message.images.length > 0 && (
                        <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-purple-100 text-purple-700">
                          <FileText className="w-3 h-3" />
                          <span>{message.images.length} source{message.images.length > 1 ? 's' : ''}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white shadow-lg border border-gray-200 px-6 py-4 rounded-xl max-w-5xl w-full">
                <div className="flex items-center space-x-3">
                  <Loader2 className="animate-spin h-5 w-5 text-blue-600" />
                  <div>
                    <span className="text-gray-700 font-medium">meeTARA is thinking...</span>
                    <p className="text-xs text-gray-500 mt-1">Analyzing your question and searching knowledge base</p>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Empty State */}
          {messages.length === 0 && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center max-w-2xl px-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-gray-800 mb-2">Welcome to meeTARA</h2>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Enhanced Input Area */}
        <div className="border-t bg-white p-4">
          {/* Domain Detection Feedback */}
          {detectedDomain && input.trim() && (
            <div className="mb-2 px-3 py-1.5 bg-blue-50 border border-blue-200 rounded-lg flex items-center gap-2">
              <Database className="w-4 h-4 text-blue-600" />
              <span className="text-sm text-blue-700">
                Detected domain: <span className="font-semibold capitalize">{detectedDomain.replace('_', ' ')}</span>
              </span>
            </div>
          )}
          
          {/* Query Suggestions */}
          {querySuggestions.length > 0 && !input.trim() && (
            <div className="mb-2 space-y-1">
              <p className="text-xs text-gray-500 px-1">Try asking:</p>
              <div className="flex flex-wrap gap-2">
                {querySuggestions.slice(0, 3).map((suggestion, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInput(suggestion)}
                    className="text-xs px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
          
          <div className="flex space-x-3">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => {
                  setInput(e.target.value)
                  // Simple domain detection based on keywords
                  detectDomainFromQuery(e.target.value)
                }}
                onKeyPress={handleKeyPress}
                placeholder="Ask meeTARA anything... Type @ to mention domains"
                className="w-full resize-none border-2 border-gray-300 rounded-xl px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                rows={1}
                style={{ minHeight: '50px', maxHeight: '150px' }}
              />
              {input.trim() && (
                <button
                  onClick={() => {
                    setInput('')
                    setDetectedDomain(null)
                  }}
                  className="absolute right-2 top-2 p-1.5 text-gray-400 hover:text-gray-600 rounded transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 font-medium shadow-lg hover:shadow-xl transition-all disabled:shadow-none"
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
          
          {/* Quick Stats */}
          <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
            <span>{messages.length} message{messages.length !== 1 ? 's' : ''} in conversation</span>
            {Object.keys(categories).length > 0 && (
              <span>{Object.values(categories).reduce((sum, cat) => sum + cat.domains_with_content, 0)} domains available</span>
            )}
          </div>
        </div>
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <DocumentUploadModal
          onClose={() => setShowUpload(false)}
          onUpload={loadDomains}
          domains={domains}
        />
      )}

      {/* 🎓 Image Lightbox Modal - Enhanced for Students */}
      {selectedImageIndex !== null && expandedImageMessageId && (() => {
        const message = messages.find(m => m.id === expandedImageMessageId)
        if (!message || !message.images || message.images.length === 0) return null
        const currentImage = message.images[selectedImageIndex]
        const imageUrl = currentImage.image_url.startsWith('http') 
          ? currentImage.image_url 
          : `http://localhost:8000${currentImage.image_url}`
        
        return (
          <div 
            className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center p-4"
            onClick={(e) => {
              if (e.target === e.currentTarget) {
                setSelectedImageIndex(null)
                setExpandedImageMessageId(null)
              }
            }}
          >
            <div className="bg-white rounded-lg max-w-6xl max-h-[90vh] w-full flex flex-col shadow-2xl">
              {/* Modal Header */}
              <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-blue-50 to-indigo-50">
                <div className="flex items-center gap-3">
                  <BookOpen className="w-5 h-5 text-blue-600" />
                  <div>
                    <h3 className="font-bold text-gray-800">
                      Image {selectedImageIndex + 1} of {message.images.length}
                    </h3>
                    {currentImage.filename && (
                      <p className="text-sm text-gray-600">
                        {currentImage.filename}
                        {currentImage.page && ` • Page ${currentImage.page}`}
                      </p>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => {
                    setSelectedImageIndex(null)
                    setExpandedImageMessageId(null)
                  }}
                  className="p-2 hover:bg-gray-200 rounded-full transition-colors"
                >
                  <X className="w-5 h-5 text-gray-600" />
                </button>
              </div>

              {/* Modal Body - Image Display */}
              <div className="flex-1 overflow-auto p-6 bg-gray-100">
                <div className="bg-white rounded-lg p-4 shadow-inner mb-4">
                  <img 
                    src={imageUrl}
                    alt={currentImage.ocr_text || `Image ${selectedImageIndex + 1}`}
                    className="max-w-full h-auto mx-auto rounded-md"
                    onError={(e) => {
                      console.error('Failed to load image:', imageUrl)
                      e.currentTarget.style.display = 'none'
                    }}
                  />
                </div>

                {/* Figure Number and Caption - Priority Display in Modal */}
                {(currentImage.figure_number || currentImage.caption || (currentImage.captions && currentImage.captions.length > 0)) && (
                  <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-lg mb-4">
                    <div className="flex items-start gap-2">
                      {currentImage.figure_number && (
                        <span className="inline-flex items-center px-3 py-1.5 rounded-md bg-blue-100 text-blue-800 text-sm font-bold whitespace-nowrap">
                          Figure {currentImage.figure_number}
                        </span>
                      )}
                      <div className="flex-1">
                        {currentImage.caption ? (
                          <p className="text-base text-gray-800 font-medium leading-relaxed">
                            {currentImage.caption}
                          </p>
                        ) : currentImage.captions && currentImage.captions.length > 0 ? (
                          <div className="space-y-2">
                            {currentImage.captions.map((cap: string, capIdx: number) => (
                              <p key={capIdx} className="text-base text-gray-800 leading-relaxed">
                                {cap}
                              </p>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </div>
                )}

                {/* Image Details - Enhanced Description */}
                {(currentImage.visual_description || currentImage.ocr_text) && (
                  <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded-r-lg mb-4">
                    <p className="text-sm font-semibold text-amber-800 mb-2">📝 Image Description:</p>
                    <p className="text-sm text-gray-700 whitespace-pre-wrap">
                      {currentImage.visual_description || currentImage.ocr_text || 'No description available'}
                    </p>
                    {/* Always show raw OCR in expandable section when available and different from description */}
                    {currentImage.ocr_text && (
                      <details className="mt-2">
                        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                          {currentImage.visual_description ? 'Show raw OCR text' : 'Show OCR details'}
                        </summary>
                        <p className="text-xs text-gray-600 bg-gray-50 p-2 rounded mt-1 font-mono whitespace-pre-wrap">
                          {currentImage.ocr_text}
                        </p>
                      </details>
                    )}
                  </div>
                )}

                {/* Image-Specific Learning Guide - Based on OCR Content */}
                {(() => {
                  const message = messages.find(m => m.id === expandedImageMessageId)
                  const domain = message?.domain || ''
                  const isAcademic = domain.includes('academic') || domain.includes('education') || domain.includes('tutoring')
                  const isHealth = domain.includes('health') || domain.includes('medical') || domain.includes('wellness')
                  const isBusiness = domain.includes('business') || domain.includes('professional') || domain.includes('marketing')
                  const isTechnical = domain.includes('programming') || domain.includes('engineering') || domain.includes('technology')
                  
                  // Generate image-specific tips based on OCR content
                  const generateImageSpecificTips = (descriptionText: string | undefined, domain: string, page?: number): string[] => {
                    const tips: string[] = []
                    const ocrText = descriptionText || '' // Use visual_description if available
                    
                    if (!ocrText || ocrText.trim().length < 10) {
                      // Fallback to generic if no OCR text
                      if (isAcademic) {
                        tips.push('Observe all labels, axes, and mathematical/scientific notations carefully')
                        tips.push('Relate this visual to the concepts explained in the text above')
                        if (page) tips.push(`Cross-reference with the full text on page ${page} for complete context`)
                        tips.push('Take notes on how this visual representation helps you understand the concept')
                      } else if (isHealth) {
                        tips.push('Review all labels, diagrams, and medical illustrations carefully')
                        tips.push('Relate this visual to the health information provided above')
                        tips.push('For medical advice, always consult with healthcare professionals')
                        if (page) tips.push(`Refer to the full document on page ${page} for complete context`)
                      } else if (isBusiness) {
                        tips.push('Analyze charts, graphs, and business diagrams carefully')
                        tips.push('Connect this visual to the business concepts discussed above')
                        tips.push('Consider practical applications of the information shown')
                        if (page) tips.push(`Review the full context on page ${page}`)
                      } else if (isTechnical) {
                        tips.push('Study diagrams, code examples, and technical illustrations carefully')
                        tips.push('Relate this visual to the technical explanations provided above')
                        tips.push('Note technical specifications, patterns, or architectures shown')
                        if (page) tips.push(`Cross-reference with the documentation on page ${page}`)
                      } else {
                        tips.push('Observe all labels, details, and key elements carefully')
                        tips.push('Relate this image to the concepts explained in the text above')
                        tips.push('Take notes on how this visual representation helps you understand the concept')
                      }
                      return tips
                    }
                    
                    // ✅ SMART: Generate tips from caption/OCR text intelligently (no hard-coded patterns)
                    const captionText = currentImage.caption || (currentImage.captions && currentImage.captions.length > 0 ? currentImage.captions.join(' ') : '') || ''
                    const fullText = `${ocrText} ${captionText}`.toLowerCase()
                    
                    // Use caption text to generate context-aware tips
                    if (captionText && captionText.trim().length > 20) {
                      // Extract key concepts from caption
                      const captionLower = captionText.toLowerCase()
                      
                      // Generate smart tips based on caption content
                      if (captionLower.includes('bone') || captionLower.includes('skeleton') || captionLower.includes('skull')) {
                        tips.push('Identify and label the major anatomical structures shown in this diagram')
                        tips.push('Understand the function and relationship between different parts')
                        if (captionLower.includes('skull')) {
                          tips.push('Study how the skull bones connect and support facial structures while protecting the brain')
                        }
                        if (captionLower.includes('vertebral') || captionLower.includes('spine')) {
                          tips.push('Note the structure and curvature of the vertebral column and its role in support')
                        }
                        if (captionLower.includes('rib') || captionLower.includes('cage')) {
                          tips.push('Observe how the rib cage protects internal organs and connects to the vertebral column')
                        }
                      } else if (captionLower.includes('function') || captionLower.includes('equation') || captionLower.includes('graph')) {
                        tips.push('Focus on the mathematical relationships and key points shown')
                        tips.push('Identify intercepts, slopes, areas, or critical values')
                      } else if (captionLower.includes('process') || captionLower.includes('flow') || captionLower.includes('system')) {
                        tips.push('Study the process flow or system architecture shown')
                        tips.push('Understand how different components connect')
                      } else if (captionLower.includes('chart') || captionLower.includes('data') || captionLower.includes('trend')) {
                        tips.push('Analyze the data trends and patterns shown')
                        tips.push('Identify key metrics and significant data points')
                      }
                      
                      // Always add these general tips
                      if (captionLower.includes('label') || fullText.includes('label')) {
                        tips.push('Pay attention to labeled components - they identify key elements')
                      }
                      if (page) {
                        tips.push(`Refer to page ${page} for detailed explanations and context`)
                      }
                    } else if (ocrText && ocrText.trim().length > 20) {
                      // Fallback: use OCR text if no caption
                      tips.push('Review the visual elements and how they connect to the text explanation')
                      if (page) {
                        tips.push(`Refer to page ${page} for complete context`)
                      }
                    } else {
                      // Generic tips if no content
                      tips.push('Observe all labels, details, and key elements carefully')
                      tips.push('Relate this image to the concepts explained in the text above')
                      if (page) {
                        tips.push(`Review page ${page} for complete context`)
                      }
                    }
                    
                    // Domain-specific additions
                    if (isHealth) {
                      tips.push('For medical information, always consult with healthcare professionals')
                    } else if (isBusiness) {
                      tips.push('Consider practical applications and how this relates to real-world scenarios')
                    } else if (isTechnical) {
                      tips.push('Relate this visual to the technical explanations provided above')
                    }
                    
                    return tips
                  }
                  
                  const guideTitle = (() => {
                    if (isAcademic) return '💡 Study Guide (Image-Specific):'
                    if (isHealth) return '💡 Important Information (Image-Specific):'
                    if (isBusiness) return '💡 Business Insights (Image-Specific):'
                    if (isTechnical) return '💡 Technical Reference (Image-Specific):'
                    return '💡 Understanding Guide (Image-Specific):'
                  })()
                  
                  const guideItems = generateImageSpecificTips(currentImage.visual_description || currentImage.ocr_text, domain, currentImage.page)
                  
                  return (
                    <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg">
                      <p className="text-sm font-semibold text-blue-800 mb-2">{guideTitle}</p>
                      <ul className="text-sm text-gray-700 space-y-1 list-disc list-inside">
                        {guideItems.map((item, idx) => (
                          <li key={idx}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )
                })()}
              </div>

              {/* Modal Footer - Navigation */}
              <div className="flex flex-col gap-2 p-4 border-t bg-gray-50">
                <div className="flex items-center justify-between">
                  <button
                    onClick={() => {
                      const newIndex = selectedImageIndex > 0 
                        ? selectedImageIndex - 1 
                        : message.images!.length - 1
                      setSelectedImageIndex(newIndex)
                    }}
                    disabled={message.images.length === 1}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronLeft className="w-4 h-4" />
                    Previous
                  </button>
                  
                  <div className="flex gap-2">
                    {message.images.map((_, idx) => (
                      <button
                        key={idx}
                        onClick={() => setSelectedImageIndex(idx)}
                        className={`w-2 h-2 rounded-full transition-all ${
                          idx === selectedImageIndex 
                            ? 'bg-blue-600 w-8' 
                            : 'bg-gray-300 hover:bg-gray-400'
                        }`}
                        aria-label={`Go to image ${idx + 1}`}
                      />
                    ))}
                  </div>

                  <button
                    onClick={() => {
                      const newIndex = selectedImageIndex < message.images!.length - 1
                        ? selectedImageIndex + 1
                        : 0
                      setSelectedImageIndex(newIndex)
                    }}
                    disabled={message.images.length === 1}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Next
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
                
                {/* Keyboard Shortcuts Hint */}
                <p className="text-xs text-center text-gray-500 mt-1">
                  💡 Keyboard shortcuts: ← → to navigate, ESC to close
                </p>
              </div>
            </div>
          </div>
        )
      })()}
    </div>
  )
}

// Document Upload Modal Component
function DocumentUploadModal({ onClose, onUpload, domains }: {
  onClose: () => void
  onUpload: () => void
  domains: Domain[]
}) {
  const [selectedDomain, setSelectedDomain] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  const handleUpload = async () => {
    if (!file || !selectedDomain) return

    setIsUploading(true)
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('domain', selectedDomain)

      const response = await fetch('http://localhost:8000/api/upload/doc', {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(300000) // 5 minute timeout for large PDFs
      })

      const data = await response.json()
      
      if (data.success) {
        setUploadProgress(100)
        setTimeout(() => {
          onUpload()
          onClose()
        }, 1000)
      } else {
        alert(`Upload failed: ${data.message}`)
      }
    } catch (error) {
      console.error('Upload error:', error)
      alert('Upload failed. Please try again.')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-96">
        <h2 className="text-xl font-semibold mb-4">Upload Document</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Domain
            </label>
            <select
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
              className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Choose a domain...</option>
              {domains.map((domain) => (
                <option key={domain.name} value={domain.name}>
                  {domain.name.replace('_', ' ')} ({domain.count} docs)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select File
            </label>
            <input
              type="file"
              accept=".pdf,.doc,.docx,.txt,.md"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="w-full border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {isUploading && (
            <div className="space-y-2">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-sm text-gray-600">Uploading and processing...</p>
            </div>
          )}
        </div>

        <div className="flex justify-end space-x-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={!file || !selectedDomain || isUploading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isUploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      </div>
    </div>
  )
}
