"""
Image Path Utilities for Meetara Core.
Centralized image path normalization and URL generation.
"""
from pathlib import Path
from urllib.parse import quote
from typing import Optional, Tuple
from app.core.logger import agent_logger


def normalize_image_path_to_url(image_path: str, domain: Optional[str] = None) -> Tuple[Optional[str], Optional[Path]]:
    """
    Normalize image path and generate API URL.
    
    Handles multiple path formats:
    - Absolute paths: "C:/path/to/images/domain/file.png"
    - Relative paths: "images/domain/file.png"
    - Windows paths: "images\\domain\\file.png"
    - Paths with hash suffixes: "images/domain_hash/file.png"
    
    Args:
        image_path: Original image path from metadata
        domain: Optional domain name for validation
        
    Returns:
        Tuple of (image_url, resolved_path) where:
        - image_url: API-accessible URL (e.g., "/api/images/domain/file.png")
        - resolved_path: Resolved Path object if file exists, None otherwise
    """
    if not image_path:
        return None, None
    
    # Strategy 1: Direct Path (handles both absolute and relative)
    test_path = Path(image_path)
    if test_path.exists():
        resolved_path = test_path.resolve()
    else:
        # Strategy 2: Normalize backslashes and try relative to cwd
        normalized = str(image_path).replace('\\', '/').replace('images/', '').lstrip('/')
        test_path = (Path.cwd() / "images" / normalized).resolve()
        if test_path.exists():
            resolved_path = test_path
        else:
            # Strategy 3: Try with original path relative to cwd
            test_path = (Path.cwd() / image_path).resolve()
            if test_path.exists():
                resolved_path = test_path
            else:
                # Strategy 4: If path starts with "images\", try from cwd
                if str(image_path).startswith('images\\') or str(image_path).startswith('images/'):
                    cleaned = str(image_path).replace('images\\', '').replace('images/', '')
                    test_path = (Path.cwd() / "images" / cleaned).resolve()
                    if test_path.exists():
                        resolved_path = test_path
                    else:
                        resolved_path = None
                else:
                    resolved_path = None
    
    # Generate API URL regardless of file existence (frontend will handle 404s)
    try:
        # Normalize path string
        stored_path_str = str(image_path).replace('\\', '/')
        
        # Remove leading "images/" or "images\" to get relative path
        if stored_path_str.startswith('images/'):
            relative_path = stored_path_str.replace('images/', '', 1)
        elif stored_path_str.startswith('images\\'):
            relative_path = stored_path_str.replace('images\\', '', 1).replace('\\', '/')
        else:
            # Try to find 'images' in the path
            if '/images/' in stored_path_str:
                idx = stored_path_str.index('/images/')
                relative_path = stored_path_str[idx + 8:]  # Skip '/images/'
            elif '\\images\\' in stored_path_str:
                idx = stored_path_str.index('\\images\\')
                relative_path = stored_path_str[idx + 8:].replace('\\', '/')
            else:
                # Fallback: assume it's already relative
                relative_path = stored_path_str.lstrip('/')
        
        # Remove hash suffixes from folder names (e.g., "APBiology-OP_5meoFaG" -> "APBiology-OP")
        # This handles the case where metadata has hash but actual folders don't
        path_segments = relative_path.lstrip('/').split('/')
        cleaned_segments = []
        for segment in path_segments:
            # Remove hash suffix pattern: "name_hash" -> "name"
            # Hash pattern: short alphanumeric segment that contains digits (e.g., "_5meoFaG")
            if '_' in segment:
                suffix_candidate = segment.split('_')[-1]
                if (
                    suffix_candidate
                    and len(suffix_candidate) <= 10
                    and len(suffix_candidate) >= 4  # Ignore short suffixes like edition markers (e.g., 2e)
                    and suffix_candidate.isalnum()
                    and any(ch.isdigit() for ch in suffix_candidate)
                    and not suffix_candidate.isalpha()
                ):
                    segment = '_'.join(segment.split('_')[:-1])
            cleaned_segments.append(segment)
        
        # URL-encode path segments to handle spaces and special characters
        encoded_segments = [quote(segment, safe='') for segment in cleaned_segments]
        encoded_path = '/'.join(encoded_segments)
        image_url = '/api/images/' + encoded_path
        
        if resolved_path and resolved_path.exists():
            agent_logger.debug(f"   Image file verified: {resolved_path}")
        else:
            agent_logger.warning(f"   ⚠️ Image file not found at {image_path}, but returning URL anyway: {image_url}")
        
        return image_url, resolved_path
        
    except Exception as e:
        # Fallback: construct URL from filename only (URL-encoded)
        img_name = Path(image_path).name
        image_url = f'/api/images/{quote(img_name, safe="")}'
        agent_logger.warning(f"Error constructing image URL from {image_path}: {e}, using filename: {image_url}")
        return image_url, resolved_path

