#!/usr/bin/env python3
"""
Test script to extract text from Python PDF and check keyword matching
"""

import sys
from pathlib import Path
sys.path.append('.')

from app.rag.vector_loader import VectorLoader
from app.core.config_loader import ConfigLoader

def test_pdf_extraction():
    """Test PDF text extraction and keyword matching."""
    print("🔍 Testing PDF text extraction and keyword matching...")
    
    # Initialize components
    vector_loader = VectorLoader()
    config_loader = ConfigLoader()
    
    # Test file path
    pdf_path = Path("downloads/programming/python311.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"📄 Testing file: {pdf_path}")
    
    # Extract text from PDF
    print("📖 Extracting text from PDF...")
    extracted_text = vector_loader.extract_text_from_file(pdf_path)
    
    print(f"📊 Extracted text length: {len(extracted_text)} characters")
    print(f"📝 First 500 characters: {extracted_text[:500]}")
    
    # Check programming keywords
    print("\n🔍 Checking programming keywords...")
    programming_keywords = config_loader.get_domain_keywords("programming")
    
    if programming_keywords:
        print(f"📋 Found {len(programming_keywords)} programming keywords")
        
        # Check for keyword matches
        text_lower = extracted_text.lower()
        matches = []
        
        for keyword in programming_keywords[:20]:  # Check first 20 keywords
            if keyword.lower() in text_lower:
                count = text_lower.count(keyword.lower())
                matches.append((keyword, count))
        
        print(f"✅ Found {len(matches)} keyword matches:")
        for keyword, count in matches:
            print(f"  - '{keyword}': {count} occurrences")
        
        # Calculate relevance score
        total_matches = sum(count for _, count in matches)
        relevance_score = total_matches / len(programming_keywords) if programming_keywords else 0
        print(f"📊 Relevance score: {relevance_score:.3f}")
        
    else:
        print("❌ No programming keywords found in config")

if __name__ == "__main__":
    test_pdf_extraction() 