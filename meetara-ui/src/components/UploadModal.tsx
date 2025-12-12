'use client'

import React, { useState, useRef } from 'react'
import { X, Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Domain } from '@/lib/types'

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
  onUploadComplete: () => void
  domains: Domain[]
}

export default function UploadModal({
  isOpen,
  onClose,
  onUploadComplete,
  domains
}: UploadModalProps) {
  const [selectedDomain, setSelectedDomain] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [errorMessage, setErrorMessage] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  if (!isOpen) return null

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setUploadStatus('idle')
      setErrorMessage('')
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    const droppedFile = e.dataTransfer.files?.[0]
    if (droppedFile) {
      // Validate file type
      const validTypes = ['.pdf', '.doc', '.docx', '.txt', '.md']
      const fileExt = '.' + droppedFile.name.split('.').pop()?.toLowerCase()
      
      if (validTypes.includes(fileExt)) {
        setFile(droppedFile)
        setUploadStatus('idle')
        setErrorMessage('')
      } else {
        setErrorMessage('Invalid file type. Please upload PDF, DOC, DOCX, TXT, or MD files.')
        setUploadStatus('error')
      }
    }
  }

  const handleUpload = async () => {
    if (!file || !selectedDomain) return

    setIsUploading(true)
    setUploadStatus('uploading')
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('domain', selectedDomain)

      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90))
      }, 500)

      const response = await fetch('http://localhost:8000/api/upload/doc', {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(300000) // 5 minute timeout
      })

      clearInterval(progressInterval)
      const data = await response.json()
      
      if (data.success) {
        setUploadProgress(100)
        setUploadStatus('success')
        
        // Wait a moment to show success, then close
        setTimeout(() => {
          onUploadComplete()
          handleClose()
        }, 1500)
      } else {
        setUploadStatus('error')
        setErrorMessage(data.message || 'Upload failed. Please try again.')
      }
    } catch (error) {
      console.error('Upload error:', error)
      setUploadStatus('error')
      setErrorMessage('Upload failed. Please check your connection and try again.')
    } finally {
      setIsUploading(false)
    }
  }

  const handleClose = () => {
    if (!isUploading) {
      setFile(null)
      setSelectedDomain('')
      setUploadProgress(0)
      setUploadStatus('idle')
      setErrorMessage('')
      onClose()
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
      <div className="bg-card rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden animate-slide-in-up">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-border bg-gradient-to-r from-primary/5 to-secondary/5">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-xl">
              <Upload className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-bold">Upload Document</h2>
              <p className="text-xs text-muted-foreground">Add knowledge to your domains</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            disabled={isUploading}
            className="p-2 rounded-lg hover:bg-muted transition-colors disabled:opacity-50"
          >
            <X className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 space-y-5">
          {/* Domain Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Select Domain
            </label>
            <select
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
              disabled={isUploading}
              className={cn(
                "w-full px-4 py-2.5 rounded-xl border border-border bg-background",
                "focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary",
                "transition-all disabled:opacity-50"
              )}
            >
              <option value="">Choose a domain...</option>
              {domains.map((domain) => (
                <option key={domain.name} value={domain.name}>
                  {domain.name.replace('_', ' ')} ({domain.count} docs)
                </option>
              ))}
            </select>
          </div>

          {/* File Drop Zone */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Select File
            </label>
            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all",
                "hover:border-primary hover:bg-primary/5",
                file ? "border-primary bg-primary/5" : "border-border",
                isUploading && "pointer-events-none opacity-50"
              )}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.doc,.docx,.txt,.md"
                onChange={handleFileSelect}
                className="hidden"
                disabled={isUploading}
              />
              
              {file ? (
                <div className="flex items-center justify-center gap-3">
                  <FileText className="w-8 h-8 text-primary" />
                  <div className="text-left">
                    <p className="font-medium text-sm">{file.name}</p>
                    <p className="text-xs text-muted-foreground">{formatFileSize(file.size)}</p>
                  </div>
                </div>
              ) : (
                <>
                  <Upload className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                  <p className="text-sm font-medium">Drop file here or click to browse</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Supports PDF, DOC, DOCX, TXT, MD
                  </p>
                </>
              )}
            </div>
          </div>

          {/* Upload Progress */}
          {uploadStatus === 'uploading' && (
            <div className="space-y-2 animate-fade-in">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Uploading...</span>
                <span className="font-medium">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                <div
                  className="bg-primary h-full rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Success Message */}
          {uploadStatus === 'success' && (
            <div className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-xl animate-fade-in">
              <CheckCircle className="w-5 h-5" />
              <span className="text-sm font-medium">Document uploaded successfully!</span>
            </div>
          )}

          {/* Error Message */}
          {uploadStatus === 'error' && errorMessage && (
            <div className="flex items-center gap-3 p-4 bg-destructive/10 text-destructive rounded-xl animate-fade-in">
              <AlertCircle className="w-5 h-5" />
              <span className="text-sm">{errorMessage}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-5 border-t border-border bg-muted/30">
          <button
            onClick={handleClose}
            disabled={isUploading}
            className="px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={!file || !selectedDomain || isUploading || uploadStatus === 'success'}
            className={cn(
              "px-5 py-2 rounded-xl font-medium text-sm transition-all",
              "bg-primary text-primary-foreground",
              "hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed",
              "flex items-center gap-2"
            )}
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Uploading...
              </>
            ) : uploadStatus === 'success' ? (
              <>
                <CheckCircle className="w-4 h-4" />
                Done
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Upload
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

