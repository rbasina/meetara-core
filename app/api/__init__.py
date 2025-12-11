"""
FastAPI route handlers for Meetara Core.

This module provides all the API endpoints for the Meetara Core backend,
including chat, document upload, emotion detection, image generation, and vectorstore operations.
"""

from .chat import router as chat_router
from .emotion import router as emotion_router
from .upload import router as upload_router
from .image_generation import router as image_generation_router
from .vectorstore import router as vectorstore_router

__all__ = [
    "chat_router",
    "emotion_router", 
    "upload_router",
    "image_generation_router",
    "vectorstore_router"
] 