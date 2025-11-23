#!/usr/bin/env python3
"""
Simple test script to check LLM functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.llm_processor import SmartLLMProcessor
from app.core.logger import agent_logger

def test_llm():
    """Test LLM generation directly."""
    try:
        print("Initializing LLM processor...")
        llm_processor = SmartLLMProcessor()
        
        print("Testing simple generation...")
        test_prompt = "Hello, how are you?"
        
        response = llm_processor.generate_intelligent_response(
            query="Hello",
            context_docs=["This is a test document."]
        )
        
        print(f"Response: {response}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_llm()
    if success:
        print("✅ LLM test successful!")
    else:
        print("❌ LLM test failed!") 