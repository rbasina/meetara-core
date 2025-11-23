#!/usr/bin/env python3
"""
Test script for Meetara GGUF models integration
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_gguf_processor():
    """Test the GGUF processor directly."""
    try:
        print("Testing Meetara GGUF Models Integration...")
        
        # Test configuration
        from app.core.config import Settings
        config = Settings()
        
        print("Configuration loaded")
        print(f"   - Meetara models enabled: {config.use_meetara_models}")
        print(f"   - Models path: {config.meetara_models_path}")
        print(f"   - Instruct model: {config.meetara_instruct_model}")
        print(f"   - Thinking model: {config.meetara_thinking_model}")
        
        # Check if models exist
        models_path = config.meetara_models_path
        if models_path.exists():
            print("Models path exists")
            
            instruct_path = models_path / config.meetara_instruct_model
            thinking_path = models_path / config.meetara_thinking_model
            
            print(f"   - Instruct model exists: {instruct_path.exists()}")
            print(f"   - Thinking model exists: {thinking_path.exists()}")
            
            if instruct_path.exists():
                size_mb = instruct_path.stat().st_size / (1024 * 1024)
                print(f"   - Instruct model size: {size_mb:.1f} MB")
                
            if thinking_path.exists():
                size_mb = thinking_path.stat().st_size / (1024 * 1024)
                print(f"   - Thinking model size: {size_mb:.1f} MB")
        else:
            print(f"Models path not found: {models_path}")
            return False
        
        # Test GGUF processor
        print("\nTesting GGUF Processor...")
        from app.core.gguf_llm_processor import MeetaraGGUFProcessor
        
        processor = MeetaraGGUFProcessor()
        
        # Get model info
        info = processor.get_model_info()
        print("GGUF Processor initialized")
        print(f"   - Models enabled: {info['meetara_models_enabled']}")
        print(f"   - Loaded models: {info['loaded_models']}")
        print(f"   - Available models: {info['available_models']}")
        
        # Health check
        health = processor.health_check()
        print(f"   - Status: {health['status']}")
        print(f"   - Models loaded: {health['models_loaded']}")
        print(f"   - Instruct available: {health['instruct_available']}")
        print(f"   - Thinking available: {health['thinking_available']}")
        
        # Test response generation
        if health['instruct_available']:
            print("\nTesting Response Generation...")
            result = processor.generate_response(
                query="Hello, how are you?",
                context="This is a test context.",
                domain="general_health",
                emotion="neutral"
            )
            
            print("Response generated successfully")
            print(f"   - Model used: {result['model_used']}")
            print(f"   - Success: {result['success']}")
            print(f"   - Generation time: {result['generation_time']}s")
            print(f"   - Response: {result['response'][:100]}...")
        
        print("\nAll tests passed! Meetara GGUF models are ready!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gguf_processor()
    sys.exit(0 if success else 1)
