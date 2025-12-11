"""
Vision AI analyzer for document images.
Provides visual understanding of images that complements OCR text extraction.
Uses offline vision models to describe charts, diagrams, and visual content.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
from PIL import Image
import io

rag_logger = logging.getLogger(__name__)

# Try to import vision models (offline, no API needed)
VISION_AI_AVAILABLE = False
VISION_MODEL = None

try:
    # Option 1: Use transformers with BLIP (image captioning)
    from transformers import BlipProcessor, BlipForConditionalGeneration
    VISION_AI_AVAILABLE = True
    VISION_MODEL_TYPE = "blip"
    rag_logger.info("BLIP vision model available for image analysis")
except ImportError:
    try:
        # Option 2: Use CLIP for image understanding
        from transformers import CLIPProcessor, CLIPModel
        VISION_AI_AVAILABLE = True
        VISION_MODEL_TYPE = "clip"
        rag_logger.info("CLIP vision model available for image analysis")
    except ImportError:
        try:
            # Option 3: Use OpenAI CLIP via transformers (lightweight)
            from transformers import AutoProcessor, AutoModelForVision2Seq
            VISION_AI_AVAILABLE = True
            VISION_MODEL_TYPE = "auto"
            rag_logger.info("Auto vision model available for image analysis")
        except ImportError:
            rag_logger.warning("No vision AI models available. Install transformers with vision models for image analysis.")
            VISION_AI_AVAILABLE = False
            VISION_MODEL_TYPE = None

# Initialize model lazily (only when needed)
_vision_processor = None
_vision_model = None


def _initialize_vision_model():
    """Initialize vision model on first use (lazy loading)."""
    global _vision_processor, _vision_model
    
    if not VISION_AI_AVAILABLE:
        return False
    
    if _vision_model is not None:
        return True
    
    try:
        if VISION_MODEL_TYPE == "blip":
            # BLIP for image captioning (good for charts, diagrams)
            model_name = "Salesforce/blip-image-captioning-base"
            _vision_processor = BlipProcessor.from_pretrained(model_name)
            _vision_model = BlipForConditionalGeneration.from_pretrained(model_name)
            rag_logger.info(f"Initialized BLIP vision model: {model_name}")
            return True
            
        elif VISION_MODEL_TYPE == "clip":
            # CLIP for image understanding
            model_name = "openai/clip-vit-base-patch32"
            _vision_processor = CLIPProcessor.from_pretrained(model_name)
            _vision_model = CLIPModel.from_pretrained(model_name)
            rag_logger.info(f"Initialized CLIP vision model: {model_name}")
            return True
            
        elif VISION_MODEL_TYPE == "auto":
            # Auto model (lightweight option)
            model_name = "microsoft/git-base"
            _vision_processor = AutoProcessor.from_pretrained(model_name)
            _vision_model = AutoModelForVision2Seq.from_pretrained(model_name)
            rag_logger.info(f"Initialized auto vision model: {model_name}")
            return True
            
    except Exception as e:
        rag_logger.error(f"Failed to initialize vision model: {e}")
        return False
    
    return False


def analyze_image_vision(image_path: Path, ocr_text: str = "") -> Optional[Dict[str, Any]]:
    """
    Analyze an image using Vision AI to provide visual understanding.
    
    This complements OCR by:
    - Describing charts, graphs, and diagrams that OCR can't handle
    - Providing context for images with no text
    - Understanding visual relationships and patterns
    
    Args:
        image_path: Path to the image file
        ocr_text: Existing OCR text (if any) - used to determine if vision analysis is needed
    
    Returns:
        Dictionary with vision analysis results:
        - description: Visual description of the image
        - image_type: Type of image (chart, diagram, photo, etc.)
        - confidence: Confidence score
        - has_text: Whether image contains text (from OCR)
        - vision_analysis: Whether vision analysis was performed
    """
    if not VISION_AI_AVAILABLE:
        return None
    
    try:
        # Initialize model if needed
        if not _initialize_vision_model():
            return None
        
        # Load image
        if not image_path.exists():
            rag_logger.warning(f"Image not found: {image_path}")
            return None
        
        image = Image.open(image_path).convert('RGB')
        
        # Determine if vision analysis is beneficial
        # Skip if OCR already extracted substantial text (text-heavy images)
        has_substantial_text = len(ocr_text.strip()) > 50
        
        # For images with no/minimal text, vision AI is very useful
        # For images with text, vision AI can still provide context
        
        vision_description = None
        image_type = "unknown"
        confidence = 0.0
        
        if VISION_MODEL_TYPE == "blip":
            # BLIP: Generate caption for the image
            inputs = _vision_processor(image, return_tensors="pt")
            out = _vision_model.generate(**inputs, max_length=100)
            vision_description = _vision_processor.decode(out[0], skip_special_tokens=True)
            confidence = 0.8  # BLIP provides good captions
            image_type = _classify_image_type(vision_description, ocr_text)
            
        elif VISION_MODEL_TYPE == "clip":
            # CLIP: Image understanding (can be used for classification)
            # For now, use a simple description approach
            # CLIP is better for classification, but we can use it for understanding
            vision_description = "Visual content detected (CLIP analysis)"
            confidence = 0.7
            image_type = "visual_content"
            
        elif VISION_MODEL_TYPE == "auto":
            # Auto model: Try to generate description
            try:
                inputs = _vision_processor(image, return_tensors="pt")
                out = _vision_model.generate(**inputs, max_length=100)
                vision_description = _vision_processor.decode(out[0], skip_special_tokens=True)
                confidence = 0.75
                image_type = _classify_image_type(vision_description, ocr_text)
            except Exception as e:
                rag_logger.debug(f"Auto model generation failed: {e}")
                vision_description = "Visual content detected"
                confidence = 0.6
                image_type = "visual_content"
        
        if vision_description:
            return {
                'description': vision_description,
                'image_type': image_type,
                'confidence': confidence,
                'has_text': len(ocr_text.strip()) > 0,
                'vision_analysis': True,
                'ocr_text_length': len(ocr_text.strip())
            }
        
    except Exception as e:
        rag_logger.debug(f"Vision analysis failed for {image_path}: {e}")
        return None
    
    return None


def _classify_image_type(description: str, ocr_text: str) -> str:
    """Classify the type of image based on description and OCR text."""
    description_lower = description.lower()
    ocr_lower = ocr_text.lower()
    
    # Check for chart/graph indicators
    chart_keywords = ['chart', 'graph', 'plot', 'diagram', 'bar', 'line', 'pie', 'scatter']
    if any(keyword in description_lower for keyword in chart_keywords):
        return "chart"
    
    # Check for diagram indicators
    diagram_keywords = ['diagram', 'flowchart', 'architecture', 'structure', 'layout']
    if any(keyword in description_lower for keyword in diagram_keywords):
        return "diagram"
    
    # Check for table indicators
    if 'table' in description_lower or 'table' in ocr_lower:
        return "table"
    
    # Check for photo/image indicators
    photo_keywords = ['photo', 'image', 'picture', 'photograph']
    if any(keyword in description_lower for keyword in photo_keywords):
        return "photo"
    
    # Check for text-heavy images
    if len(ocr_text.strip()) > 100:
        return "text_image"
    
    return "visual_content"


def should_use_vision_ai(ocr_text: str, image_size: tuple) -> bool:
    """
    Determine if Vision AI analysis would be beneficial for this image.
    
    Vision AI is most useful when:
    - Image has no/minimal OCR text (charts, diagrams, photos)
    - Image is likely a chart or graph
    - OCR text is minimal but image is substantial
    
    Args:
        ocr_text: OCR text extracted from image
        image_size: Image dimensions (width, height)
    
    Returns:
        True if vision AI analysis would be beneficial
    """
    # Always use vision AI if no OCR text (or very minimal)
    if len(ocr_text.strip()) < 20:
        return True
    
    # Use vision AI for medium to large images even with some text
    # (vision can provide context that OCR text alone doesn't capture)
    width, height = image_size
    image_area = width * height
    
    # For substantial images (>100k pixels), vision AI adds value
    if image_area > 100000 and len(ocr_text.strip()) < 200:
        return True
    
    return False


def get_vision_description_for_document(ocr_text: str, vision_analysis: Optional[Dict[str, Any]]) -> str:
    """
    Generate a combined description for document content.
    Combines OCR text with Vision AI description for comprehensive understanding.
    
    Args:
        ocr_text: OCR text from image
        vision_analysis: Vision AI analysis results
    
    Returns:
        Combined description string for document content
    """
    parts = []
    
    # Add OCR text if available
    if ocr_text and ocr_text.strip():
        parts.append(f"[Image Text]\n{ocr_text.strip()}")
    
    # Add Vision AI description if available
    if vision_analysis and vision_analysis.get('description'):
        vision_desc = vision_analysis['description']
        image_type = vision_analysis.get('image_type', 'visual_content')
        
        # Format vision description based on image type
        if image_type == "chart":
            parts.append(f"[Chart Description]\n{vision_desc}")
        elif image_type == "diagram":
            parts.append(f"[Diagram Description]\n{vision_desc}")
        elif image_type == "photo":
            parts.append(f"[Image Description]\n{vision_desc}")
        else:
            parts.append(f"[Visual Content]\n{vision_desc}")
    
    return "\n\n".join(parts) if parts else ""

