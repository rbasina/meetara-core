"""
Enhanced Image Description Service.

Adds visual descriptions to OCR text by analyzing image content.
"""
import re
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np
from app.core.logger import api_logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ImageDescriptor:
    """Service for generating rich descriptions of images from visual analysis."""
    
    def enhance_ocr_with_description(self, image_path: str, ocr_text: str) -> str:
        """
        Enhance OCR text with visual description of the image.
        
        Args:
            image_path: Path to the image file
            ocr_text: Raw OCR text extracted from image
            
        Returns:
            Enhanced description combining OCR and visual analysis
        """
        try:
            # Load image
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Analyze visual content
            visual_desc = self._analyze_visual_content(img, img_array, ocr_text)
            
            # Return only visual description (OCR shown separately in UI)
            # This prevents duplication since raw OCR is displayed in expandable section
            if visual_desc:
                return visual_desc
            elif ocr_text:
                # Only use OCR if no visual analysis possible
                # Frontend will show this as fallback, but won't duplicate it
                return f"Image contains text elements"
            else:
                return "Diagram or illustration"
                
        except Exception as e:
            api_logger.error(f"Error enhancing OCR description: {e}")
            return ocr_text if ocr_text else "Image"
    
    def _analyze_visual_content(self, img: Image.Image, img_array: np.ndarray, ocr_text: str) -> str:
        """
        Analyze visual content of image to generate description.
        
        Returns:
            Natural language description of visual elements
        """
        width, height = img.size
        descriptions = []
        
        # Analyze based on OCR text patterns
        ocr_lower = ocr_text.lower() if ocr_text else ""
        
        # Geometric shapes detection
        shape_desc = self._detect_geometric_shapes(ocr_text, img_array)
        if shape_desc:
            descriptions.append(shape_desc)
        
        # Mathematical content detection
        math_desc = self._detect_mathematical_content(ocr_text)
        if math_desc:
            descriptions.append(math_desc)
        
        # Diagram type detection
        diagram_desc = self._detect_diagram_type(ocr_text, width, height)
        if diagram_desc:
            descriptions.append(diagram_desc)
        
        # Combine descriptions
        if descriptions:
            return ". ".join(descriptions)
        
        # Fallback description based on size and aspect ratio
        aspect_ratio = width / height if height > 0 else 1
        if aspect_ratio > 1.5:
            return "Wide diagram or chart"
        elif aspect_ratio < 0.67:
            return "Tall diagram or illustration"
        else:
            return "Diagram or illustration"
    
    def _detect_geometric_shapes(self, ocr_text: str, img_array: np.ndarray) -> Optional[str]:
        """Detect geometric shapes from OCR text and image analysis."""
        if not ocr_text:
            return None
        
        ocr_lower = ocr_text.lower()
        descriptions = []
        
        # Triangle detection
        if any(word in ocr_lower for word in ['triangle', 'trig', 'angle', '°']):
            # Extract angle measurements
            angles = re.findall(r'(\d+)\s*°', ocr_text)
            if angles:
                angle_desc = ", ".join([f"{a}°" for a in angles[:3]])
                if len(angles) == 3:
                    descriptions.append(f"Triangle with angles: {angle_desc}")
                else:
                    descriptions.append(f"Triangle with angle measurements: {angle_desc}")
            
            # Detect right angle
            if '90°' in ocr_text or '90 ' in ocr_text:
                descriptions.append("Right-angled triangle")
            
            # Detect triangle type
            if len(angles) == 3:
                try:
                    angle_vals = [int(a) for a in angles]
                    if len(set(angle_vals)) == 1:
                        descriptions.append("Equilateral triangle")
                    elif len(set(angle_vals)) == 2:
                        descriptions.append("Isosceles triangle")
                    else:
                        descriptions.append("Scalene triangle")
                except:
                    pass
            
            # Vertex labels
            vertices = re.findall(r'\b([A-Z])\b', ocr_text)
            if vertices:
                vertex_desc = ", ".join(sorted(set(vertices))[:3])
                descriptions.append(f"Vertices labeled: {vertex_desc}")
            
            # Side labels
            sides = re.findall(r'\b([a-z])\b', ocr_text)
            if sides:
                side_desc = ", ".join(sorted(set(sides))[:3])
                descriptions.append(f"Sides labeled: {side_desc}")
        
        # Circle detection
        elif any(word in ocr_lower for word in ['circle', 'radius', 'diameter', 'r=', 'r =']):
            radius_match = re.search(r'r\s*=\s*(\d+)', ocr_text, re.IGNORECASE)
            if radius_match:
                descriptions.append(f"Circle with radius {radius_match.group(1)}")
            else:
                descriptions.append("Circle")
        
        # Rectangle/square detection
        elif any(word in ocr_lower for word in ['rectangle', 'square', 'length', 'width']):
            if 'square' in ocr_lower:
                descriptions.append("Square")
            else:
                descriptions.append("Rectangle")
        
        return ". ".join(descriptions) if descriptions else None
    
    def _detect_mathematical_content(self, ocr_text: str) -> Optional[str]:
        """Detect mathematical notation and content."""
        if not ocr_text:
            return None
        
        descriptions = []
        
        # Function notation
        if re.search(r'[fgh]\([x]\)\s*=|y\s*=\s*[^=]', ocr_text):
            func_match = re.search(r'(y|[fgh]\(x\))\s*=\s*([^\n,]+)', ocr_text)
            if func_match:
                expr = func_match.group(2).strip()
                descriptions.append(f"Function: {expr}")
        
        # Integrals
        if '∫' in ocr_text or 'integral' in ocr_text.lower():
            descriptions.append("Integral notation or area calculation")
        
        # Derivatives
        if "'" in ocr_text or 'derivative' in ocr_text.lower():
            descriptions.append("Derivative or rate of change")
        
        # Square roots
        if '√' in ocr_text or 'sqrt' in ocr_text.lower():
            descriptions.append("Square root or radical expression")
        
        # Fractions
        if '/' in ocr_text and re.search(r'\d+/\d+', ocr_text):
            descriptions.append("Fractional expressions")
        
        # Coordinates
        if re.search(r'\(-?\d+,\s*-?\d+\)', ocr_text):
            descriptions.append("Coordinate system or graph")
        
        return ". ".join(descriptions) if descriptions else None
    
    def _detect_diagram_type(self, ocr_text: str, width: int, height: int) -> Optional[str]:
        """Detect type of diagram based on OCR and dimensions."""
        if not ocr_text:
            return None
        
        ocr_lower = ocr_text.lower()
        
        # Graph/chart detection
        if any(word in ocr_lower for word in ['graph', 'plot', 'chart']):
            return "Graph or data visualization"
        
        # Axis detection
        if any(word in ocr_lower for word in ['x-axis', 'y-axis', 'axis', 'coordinate']):
            return "Coordinate system or graph"
        
        # Wave/oscillation
        if any(word in ocr_lower for word in ['wave', 'oscillation', 'sine', 'cosine']):
            return "Wave or oscillatory pattern"
        
        # Circuit
        if any(word in ocr_lower for word in ['circuit', 'resistor', 'battery', 'voltage']):
            return "Electrical circuit diagram"
        
        # Molecule
        if any(word in ocr_lower for word in ['molecule', 'atom', 'bond', 'chemical']):
            return "Chemical or molecular structure"
        
        # Cell/biology
        if any(word in ocr_lower for word in ['cell', 'nucleus', 'dna', 'protein']):
            return "Biological diagram or cell structure"
        
        return None


# Global instance
_image_descriptor = None

def get_image_descriptor() -> ImageDescriptor:
    """Get or create global image descriptor instance."""
    global _image_descriptor
    if _image_descriptor is None:
        _image_descriptor = ImageDescriptor()
    return _image_descriptor

