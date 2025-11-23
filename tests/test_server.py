#!/usr/bin/env python3
"""
Simple test to start the Meetara server
"""

import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_server():
    """Test server startup step by step."""
    try:
        print("Step 1: Testing basic imports...")
        from app.core.config import settings
        print("✅ Config loaded")
        
        print("Step 2: Testing logger...")
        from app.core.logger import setup_logging
        print("✅ Logger loaded")
        
        print("Step 3: Testing API imports...")
        from app.api import chat_router, emotion_router, upload_router
        print("✅ API routers loaded")
        
        print("Step 4: Testing FastAPI app creation...")
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI(
            title="Meetara Core Test",
            description="Test server",
            version="1.0.0"
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        app.include_router(chat_router, prefix="/api")
        app.include_router(emotion_router, prefix="/api")
        app.include_router(upload_router, prefix="/api")
        
        print("✅ FastAPI app created successfully")
        
        print("Step 5: Testing server startup...")
        import uvicorn
        
        print("Starting server on http://localhost:8000")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_server()
