"""
Image Generation API endpoints for Meetara Core.
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.core.logger import api_logger
from app.services.image_generator import get_image_generator

router = APIRouter(prefix="/generate-image", tags=["image-generation"])


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""
    query: str = Field(..., description="Natural language query describing the image to generate")
    image_type: Optional[str] = Field(None, description="Type of image: function_graph, geometry, calculus_diagram")
    domain: Optional[str] = Field("academic_tutoring", description="Domain context")


class ImageGenerationResponse(BaseModel):
    """Response model for image generation."""
    success: bool
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    image_type: Optional[str] = None
    message: str


@router.post("/", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    """
    Generate an image based on a natural language query.
    
    Examples:
    - "Show me a graph of y = x^2"
    - "Draw a triangle with angles 45°, 60°, 75°"
    - "Create a circle with radius 5"
    """
    try:
        generator = get_image_generator()
        
        # Generate image from query
        result = generator.generate_from_query(request.query, request.domain or "academic_tutoring")
        
        if result:
            return ImageGenerationResponse(
                success=True,
                image_url=result["image_url"],
                image_path=result.get("image_path"),
                image_type=result.get("type"),
                message=f"Image generated successfully: {result.get('filename')}"
            )
        else:
            return ImageGenerationResponse(
                success=False,
                message="Could not determine what image to generate from the query. Try being more specific (e.g., 'graph of y = x^2' or 'triangle with angles 45°, 60°, 75°')."
            )
            
    except Exception as e:
        api_logger.error(f"Error in image generation endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@router.get("/check")
async def check_generation_capability(query: str = Query(..., description="Query to check")):
    """
    Check if a query should trigger image generation.
    
    Returns whether the query would generate an image and what type.
    """
    try:
        generator = get_image_generator()
        should_gen, img_type = generator.should_generate_image(query)
        
        return {
            "should_generate": should_gen,
            "image_type": img_type,
            "query": query
        }
    except Exception as e:
        api_logger.error(f"Error checking generation capability: {e}")
        return {
            "should_generate": False,
            "image_type": None,
            "error": str(e)
        }

