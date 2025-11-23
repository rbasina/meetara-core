"""
Image extraction utility for PDF documents.
Extracts images from PDFs, performs OCR, and stores them in structured directories.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import io
import platform
import os
import gc

try:
    from pypdf import PdfReader
    import pytesseract
    from PIL import Image
    IMAGE_EXTRACTION_AVAILABLE = True
except ImportError:
    IMAGE_EXTRACTION_AVAILABLE = False

from app.core.logger import rag_logger

# Memory management constants
MAX_IMAGE_DIMENSION = 5000  # Maximum width or height in pixels (prevents huge images)
MAX_IMAGE_MEMORY_MB = 50  # Maximum estimated memory per image in MB
OCR_MAX_DIMENSION = 3000  # Maximum dimension for OCR processing (resize larger images)

# Smart OCR configuration
ENABLE_SMART_OCR_SKIPPING = True  # Skip OCR when captions exist or image is chart/graph
FORCE_OCR_FOR_SCANNED_PDFS = True  # Always OCR for scanned PDFs (text only in images)


def sanitize_filename_part(part: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        part = part.replace(char, '_')
    part = part.strip('. ')
    return part


def _configure_tesseract():
    """Configure Tesseract path for Windows if needed."""
    if platform.system() == "Windows" and IMAGE_EXTRACTION_AVAILABLE:
        # Common installation paths
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        
        # Check if tesseract is in PATH
        try:
            pytesseract.get_tesseract_version()
            return True
        except:
            # Try common paths
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    rag_logger.debug(f"Found Tesseract at: {path}")
                    return True
        
        return False
    return True  # For non-Windows, assume tesseract is in PATH


def extract_images_from_pdf(
    pdf_path: Path,
    output_base_dir: Path,
    domain: str,
    filename: str,
    silent: bool = False,
    page_texts: Optional[Dict[int, str]] = None  # Optional: page number -> text mapping
) -> List[Dict[str, Any]]:
    """
    Extract images from PDF along with their page numbers and OCR text.
    
    Args:
        pdf_path: Path to the PDF file
        output_base_dir: Base directory for storing images (e.g., Path("images"))
        domain: Domain name for organizing images
        filename: Source filename (without extension) for organizing images
        silent: If True, don't log progress messages
        page_texts: Optional dict mapping page numbers to extracted text (for smart OCR skipping)
        
    Returns:
        List of image metadata dictionaries with keys:
        - page: Page number
        - image_id: Unique image identifier
        - image_path: Path to saved image file
        - ocr_text: Extracted OCR text
        - image_size: (width, height) tuple
        - format: Image format
    """
    if not IMAGE_EXTRACTION_AVAILABLE:
        if not silent:
            rag_logger.warning("Image extraction libraries not available (pypdf, pytesseract, Pillow)")
        return []
    
    # Configure Tesseract
    tesseract_available = _configure_tesseract()
    if not tesseract_available:
        if not silent:
            rag_logger.warning("Tesseract OCR not found - images will be extracted but OCR will be skipped")
    
    images_data = []
    
    try:
        pdf_reader = PdfReader(str(pdf_path))
        total_pages = len(pdf_reader.pages)
        
        if not silent:
            rag_logger.info(f"Extracting images from PDF: {pdf_path.name} ({total_pages} pages)")
        
        # Detect if PDF is scanned (minimal text extraction)
        # Check first few pages for text content
        is_scanned_pdf = False
        if page_texts:
            # Use provided page texts to detect scanned PDF
            sample_pages = [p for p in range(1, min(6, total_pages + 1)) if p in page_texts]
            if sample_pages:
                avg_text_length = sum(len(page_texts[p].strip()) for p in sample_pages) / len(sample_pages)
                # If average text per page is < 100 chars, likely scanned
                is_scanned_pdf = avg_text_length < 100
                if not silent and is_scanned_pdf:
                    rag_logger.info("Detected scanned PDF - OCR will be performed on all images")
        else:
            # Try to extract text from first page to detect scanned PDF
            try:
                first_page = pdf_reader.pages[0]
                first_page_text = first_page.extract_text() if hasattr(first_page, 'extract_text') else ""
                is_scanned_pdf = len(first_page_text.strip()) < 100
                if not silent and is_scanned_pdf:
                    rag_logger.info("Detected scanned PDF - OCR will be performed on all images")
            except:
                pass  # Can't determine, assume not scanned
        
        # Create output directory: images/{domain}/{filename}/
        output_dir = output_base_dir / domain / filename
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_per_page = {}  # Track images per page for statistics
        ocr_skipped_count = 0  # Track how many images skipped OCR
        
        # Log initial progress (even in silent mode for large files)
        log_interval = 50 if total_pages > 500 else 20 if total_pages > 100 else 10
        if total_pages > 100:
            rag_logger.info(f"📷 Starting image extraction from {total_pages} pages (this may take several minutes for large PDFs)...")
        
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            # Extract images from this page
            if '/XObject' in page.get('/Resources', {}):
                try:
                    xobjects = page['/Resources']['/XObject'].get_object()
                    page_image_count = 0
                    
                    for obj_num, obj in xobjects.items():
                        if obj.get('/Subtype') == '/Image':
                            try:
                                # Check image size before extraction to prevent memory issues
                                width = obj.get('/Width', 0)
                                height = obj.get('/Height', 0)
                                
                                # Skip extremely large images that would cause memory issues
                                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                                    if not silent:
                                        rag_logger.warning(
                                            f"Skipping oversized image on page {page_num}: "
                                            f"{width}x{height} (max: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
                                        )
                                    continue
                                
                                # Estimate memory usage (rough calculation)
                                # RGB: 3 bytes per pixel, CMYK: 4 bytes per pixel
                                estimated_mb = (width * height * 4) / (1024 * 1024)  # Conservative estimate
                                if estimated_mb > MAX_IMAGE_MEMORY_MB:
                                    if not silent:
                                        rag_logger.warning(
                                            f"Skipping memory-intensive image on page {page_num}: "
                                            f"{width}x{height} (~{estimated_mb:.1f}MB, max: {MAX_IMAGE_MEMORY_MB}MB)"
                                        )
                                    continue
                                
                                # Get page text for smart OCR skipping
                                page_text = ""
                                if page_texts and page_num in page_texts:
                                    page_text = page_texts[page_num]
                                elif hasattr(page, 'extract_text'):
                                    try:
                                        page_text = page.extract_text() or ""
                                    except:
                                        pass
                                
                                # Check if we should skip OCR (smart skipping)
                                image_size_tuple = (width, height)
                                skip_ocr = False
                                if tesseract_available and ENABLE_SMART_OCR_SKIPPING:
                                    # Check for captions in page text (FIGURE captions)
                                    has_captions = False
                                    if page_text:
                                        import re
                                        has_captions = bool(re.search(r'(?:FIGURE|Figure)\s+[\d\.]+', page_text, re.IGNORECASE))
                                    
                                    skip_ocr = _should_skip_ocr(
                                        image_size=image_size_tuple,
                                        has_captions=has_captions,
                                        is_scanned_pdf=is_scanned_pdf,
                                        page_text=page_text
                                    )
                                    
                                    if skip_ocr:
                                        ocr_skipped_count += 1
                                        if not silent:
                                            rag_logger.debug(f"Page {page_num}: Skipping OCR for image {obj_num} (smart skipping enabled)")
                                
                                image_info = _extract_single_image(
                                    obj, obj_num, page_num, pdf_path.stem,
                                    output_dir, silent, skip_ocr=skip_ocr
                                )
                                if image_info:
                                    images_data.append(image_info)
                                    page_image_count += 1
                                    
                                    # Clear image from memory immediately
                                    del image_info
                            except MemoryError as me:
                                if not silent:
                                    rag_logger.error(
                                        f"Memory error extracting image on page {page_num}: {me}. "
                                        f"Skipping this image to continue processing."
                                    )
                                # Force garbage collection
                                gc.collect()
                                continue
                            except Exception as e:
                                if not silent:
                                    rag_logger.debug(f"Error extracting image on page {page_num}: {e}")
                    
                    if page_image_count > 0:
                        images_per_page[page_num] = page_image_count
                    
                    # Force garbage collection every 10 pages to free memory
                    if page_num % 10 == 0:
                        gc.collect()
                    
                    # Log progress more frequently for large files (even in silent mode)
                    # This prevents users thinking the system hung
                    if page_num % log_interval == 0:
                        rag_logger.info(f"   📄 Page {page_num}/{total_pages} processed, {len(images_data)} images extracted so far...")
                except Exception as e:
                    if not silent:
                        rag_logger.debug(f"Error accessing XObjects on page {page_num}: {e}")
                finally:
                    # Clear page resources
                    del page
                    if page_num % 5 == 0:  # More frequent cleanup for large PDFs
                        gc.collect()
        
        # Log detailed statistics
        if images_data:
            images_with_ocr = len([img for img in images_data if img.get('ocr_text')])
            total_ocr_chars = sum(len(img.get('ocr_text', '')) for img in images_data)
            pages_with_images_count = len(images_per_page)
            
            if not silent:
                rag_logger.info(f"📷 IMAGE EXTRACTION SUMMARY for {pdf_path.name}:")
                rag_logger.info(f"   - Total images extracted: {len(images_data)}")
                rag_logger.info(f"   - Pages with images: {pages_with_images_count}/{total_pages}")
                if ENABLE_SMART_OCR_SKIPPING:
                    rag_logger.info(f"   - OCR skipped (smart skipping): {ocr_skipped_count}/{len(images_data)} ({ocr_skipped_count/len(images_data)*100:.1f}%)")
                rag_logger.info(f"   - Images with OCR text: {images_with_ocr}/{len(images_data)} ({images_with_ocr/len(images_data)*100:.1f}%)")
                rag_logger.info(f"   - Total OCR characters: {total_ocr_chars:,}")
                
                # Image size statistics
                image_sizes = [img.get('image_size', (0, 0)) for img in images_data if img.get('image_size')]
                if image_sizes:
                    avg_w = sum(w for w, h in image_sizes) / len(image_sizes)
                    avg_h = sum(h for w, h in image_sizes) / len(image_sizes)
                    rag_logger.info(f"   - Average image size: {int(avg_w)}x{int(avg_h)} pixels")
        elif not silent:
            rag_logger.info(f"📷 No images found in {pdf_path.name}")
        
        # Final cleanup
        del pdf_reader
        gc.collect()
        
        return images_data
        
    except MemoryError as me:
        rag_logger.error(
            f"Memory allocation failed while extracting images from PDF {pdf_path}: {me}. "
            f"Extracted {len(images_data)} images before failure. "
            f"Consider processing this PDF in smaller batches or with fewer images."
        )
        # Force cleanup
        gc.collect()
        return images_data  # Return what we managed to extract
    except Exception as e:
        rag_logger.error(f"Failed to extract images from PDF {pdf_path}: {e}")
        gc.collect()
        return []


def _should_skip_ocr(
    image_size: tuple,
    has_captions: bool = False,
    is_scanned_pdf: bool = False,
    page_text: str = ""
) -> bool:
    """
    Smart OCR skipping: Determine if OCR should be skipped.
    
    Skip OCR when:
    1. Captions exist (captions are better than OCR for academic images)
    2. Image is clearly a chart/graph (numbers won't help)
    3. PDF text extraction is good (native PDF, not scanned)
    
    Only use OCR for:
    - Scanned PDFs (text only in images)
    - Images without captions that might have meaningful text
    - Large images that might contain text content
    
    Args:
        image_size: (width, height) tuple
        has_captions: Whether captions were found for this image
        is_scanned_pdf: Whether PDF is scanned (text only in images)
        page_text: Text extracted from the page (to check if PDF has good text extraction)
    
    Returns:
        True if OCR should be skipped, False if OCR should be performed
    """
    if not ENABLE_SMART_OCR_SKIPPING:
        return False  # Always do OCR if smart skipping is disabled
    
    # Always OCR for scanned PDFs (text only in images)
    if is_scanned_pdf and FORCE_OCR_FOR_SCANNED_PDFS:
        return False
    
    # Skip OCR if captions exist (captions are more reliable and descriptive)
    if has_captions:
        rag_logger.debug("Skipping OCR: Captions found (captions are better than OCR)")
        return True
    
    # Skip OCR if PDF has good text extraction (native PDF, not scanned)
    # Check if page has substantial text content (indicates good text extraction)
    if page_text and len(page_text.strip()) > 200:
        # PDF has good text extraction - captions should be in text
        # Only OCR if image is very large (might contain text not in captions)
        width, height = image_size
        if width < 800 or height < 800:
            rag_logger.debug("Skipping OCR: PDF has good text extraction and image is small")
            return True
    
    # Check if image is likely a chart/graph (numbers won't help)
    width, height = image_size
    aspect_ratio = width / height if height > 0 else 1
    
    # Charts/graphs are typically rectangular with moderate aspect ratios
    # Very wide or tall images might be charts
    if 0.3 <= aspect_ratio <= 3.0:
        # Moderate aspect ratio - could be chart/graph
        # If image is medium-sized, likely a chart (skip OCR)
        if 300 <= width <= 2000 and 300 <= height <= 2000:
            rag_logger.debug("Skipping OCR: Image appears to be chart/graph (numbers won't help)")
            return True
    
    # Do OCR for:
    # - Large images (might contain text content)
    # - Images without captions
    # - Scanned PDFs
    return False


def _is_generic_image(img: Image.Image, ocr_text: str, image_size: tuple) -> bool:
    """
    Check if an image is generic (logo, decorative, etc.) before saving.
    Returns True if image should be filtered out.
    """
    import re
    
    width, height = image_size
    ocr_lower = ocr_text.lower().strip()
    
    # Filter very small images (icons, logos)
    if width < 50 or height < 50:
        return True
    
    # Filter extreme aspect ratios (headers/footers)
    aspect_ratio = width / height if height > 0 else 1
    if aspect_ratio > 20 or aspect_ratio < 0.05:
        return True
    
    # If no OCR text, check size (small images with no text are likely decorative)
    if not ocr_text or len(ocr_text.strip()) == 0:
        if width < 100 or height < 100:
            return True
        return False  # Large images without OCR might be diagrams
    
    # Generic patterns
    generic_patterns = [
        r'copyright', r'©', r'™', r'®',
        r'all rights reserved', r'logo',
        r'www\.', r'http[s]?://',
        r'\bfoundation\b', r'\buniversity\b',
        r'\bopenstax\b', r'\brice\b', r'\bhewlett\b', r'\barnold\b',
    ]
    
    for pattern in generic_patterns:
        if re.search(pattern, ocr_lower, re.IGNORECASE):
            return True
    
    # Check for brand names in short text (likely logos)
    ocr_clean = re.sub(r'[™®©\W]+', ' ', ocr_lower).strip()
    ocr_words = ocr_clean.split()
    brand_names = ['openstax', 'rice', 'hewlett', 'arnold', 'tjaf', 'foundation']
    
    if len(ocr_words) <= 4:
        brand_matches = sum(1 for word in ocr_words if word in brand_names)
        if brand_matches > 0:
            return True
    
    # Check for trademark symbols with short text
    if re.search(r'[™®]', ocr_text) and len(ocr_words) <= 5:
        return True
    
    # Minimal meaningful text
    meaningful_chars = len(re.sub(r'\W', '', ocr_text))
    if meaningful_chars < 3:
        return True
    
    return False


def _extract_single_image(
    obj: Any,
    obj_num: str,
    page_num: int,
    pdf_stem: str,
    output_dir: Path,
    silent: bool,
    skip_ocr: bool = False
) -> Optional[Dict[str, Any]]:
    """Extract a single image from a PDF object."""
    try:
        # Get image data
        if '/Filter' in obj:
            filter_type = obj['/Filter']
            
            # Handle list of filters (e.g., ['/FlateDecode'])
            if isinstance(filter_type, list):
                filter_type = filter_type[0] if filter_type else None
            
            img = None
            ocr_text = ""
            
            if filter_type == '/DCTDecode':  # JPEG
                image_data = obj._data
                img = Image.open(io.BytesIO(image_data))
                
            elif filter_type == '/FlateDecode':  # FlateDecode (zlib compressed)
                img = _extract_flatedecode_image(obj)
                
            else:
                # Try to extract using get_data() for other formats
                try:
                    image_data = obj.get_data()
                    img = Image.open(io.BytesIO(image_data))
                except Exception:
                    return None
            
            if img is None:
                return None
            
            # Convert to RGB for consistency (needed for OCR)
            # Handle CMYK and other color spaces properly to avoid color inversion
            if img.mode == 'CMYK':
                # CMYK images need proper conversion to RGB to avoid inversion
                # Use ImageCms for proper color profile handling if available, otherwise use standard conversion
                try:
                    from PIL import ImageCms
                    # Try to convert using ICC profile if available
                    if hasattr(img, 'info') and 'icc_profile' in img.info:
                        img = ImageCms.profileToProfile(img, img.info['icc_profile'], 'sRGB')
                    else:
                        # Standard CMYK to RGB conversion
                        # Note: Some PDFs store CMYK images inverted - PIL's convert handles this
                        img = img.convert('RGB')
                except (ImportError, Exception):
                    # Fallback to standard conversion
                    img = img.convert('RGB')
            elif img.mode in ['L', 'LA', 'P', 'PA']:
                # Grayscale, palette - convert to RGB
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            image_size = img.size
            
            # Do OCR FIRST before saving (to check if it's generic)
            # Skip OCR if smart skipping determined it's not needed
            ocr_text = ""
            if not skip_ocr:
                # Resize very large images for OCR to prevent memory issues
                ocr_img = img
                try:
                    if IMAGE_EXTRACTION_AVAILABLE and _configure_tesseract():
                        # Resize if image is too large for OCR (prevents memory issues)
                        width, height = image_size
                        if width > OCR_MAX_DIMENSION or height > OCR_MAX_DIMENSION:
                            # Calculate scaling factor to fit within OCR_MAX_DIMENSION
                            scale = min(OCR_MAX_DIMENSION / width, OCR_MAX_DIMENSION / height)
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            ocr_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            if not silent:
                                rag_logger.debug(
                                    f"Resized image for OCR: {width}x{height} -> {new_width}x{new_height} "
                                    f"(page {page_num})"
                                )
                        
                        ocr_text = pytesseract.image_to_string(ocr_img)
                        ocr_text = ocr_text.strip()
                        
                        # Clean up resized image if different from original
                        if ocr_img is not img:
                            del ocr_img
                except MemoryError as me:
                    if not silent:
                        rag_logger.warning(f"Memory error during OCR on page {page_num}: {me}")
                    ocr_text = ""
                    if ocr_img is not img:
                        del ocr_img
                except Exception as ocr_e:
                    if not silent:
                        rag_logger.debug(f"OCR failed during extraction: {ocr_e}")
                    ocr_text = ""
                    if ocr_img is not img:
                        del ocr_img
            
            # Check if image is generic BEFORE saving
            if _is_generic_image(img, ocr_text, image_size):
                if not silent:
                    rag_logger.debug(f"Skipping generic image (not saving): page {page_num}, size {image_size}, OCR: {ocr_text[:30]}")
                return None  # Don't save generic images
            
            # Image passed filters - save it
            safe_obj_num = sanitize_filename_part(str(obj_num))
            image_id = f"page_{page_num}_img_{safe_obj_num}"
            image_filename = f"{pdf_stem}_{image_id}.png"
            output_path = output_dir / image_filename
            
            img.save(output_path)
            
            return {
                'page': page_num,
                'image_id': image_id,
                'image_path': str(output_path),
                'ocr_text': ocr_text,
                'image_size': image_size,
                'format': 'PNG'
            }
        
        else:
            # Try alternative extraction method for images without explicit filter
            try:
                if '/Length' in obj:
                    image_data = obj._data
                    img = Image.open(io.BytesIO(image_data))
                    
                    # Convert to RGB for consistency
                    # Handle CMYK and other color spaces properly to avoid color inversion
                    if img.mode == 'CMYK':
                        try:
                            from PIL import ImageCms
                            # Try to convert using ICC profile if available
                            if hasattr(img, 'info') and 'icc_profile' in img.info:
                                img = ImageCms.profileToProfile(img, img.info['icc_profile'], 'sRGB')
                            else:
                                # Standard CMYK to RGB conversion
                                img = img.convert('RGB')
                        except (ImportError, Exception):
                            # Fallback to standard conversion
                            img = img.convert('RGB')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    image_size = img.size
                    
                    # Do OCR first to check if generic
                    # Skip OCR if smart skipping determined it's not needed
                    ocr_text = ""
                    if not skip_ocr:
                        # Resize very large images for OCR to prevent memory issues
                        ocr_img = img
                        try:
                            if IMAGE_EXTRACTION_AVAILABLE and _configure_tesseract():
                                # Resize if image is too large for OCR
                                width, height = image_size
                                if width > OCR_MAX_DIMENSION or height > OCR_MAX_DIMENSION:
                                    scale = min(OCR_MAX_DIMENSION / width, OCR_MAX_DIMENSION / height)
                                    new_width = int(width * scale)
                                    new_height = int(height * scale)
                                    ocr_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                
                                ocr_text = pytesseract.image_to_string(ocr_img).strip()
                                
                                # Clean up resized image if different from original
                                if ocr_img is not img:
                                    del ocr_img
                        except MemoryError:
                            ocr_text = ""
                            if ocr_img is not img:
                                del ocr_img
                        except:
                            ocr_text = ""
                            if ocr_img is not img:
                                del ocr_img
                    
                    # Check if generic BEFORE saving
                    if _is_generic_image(img, ocr_text, image_size):
                        if not silent:
                            rag_logger.debug(f"Skipping generic image (not saving): page {page_num}, size {image_size}")
                        return None
                    
                    # Image passed filters - save it
                    safe_obj_num = sanitize_filename_part(str(obj_num))
                    image_id = f"page_{page_num}_img_{safe_obj_num}"
                    image_filename = f"{pdf_stem}_{image_id}.png"
                    output_path = output_dir / image_filename
                    
                    img.save(output_path)
                    
                    return {
                        'page': page_num,
                        'image_id': image_id,
                        'image_path': str(output_path),
                        'ocr_text': ocr_text,
                        'image_size': image_size,
                        'format': 'PNG'
                    }
            except Exception:
                return None
        
        return None
        
    except Exception as e:
        if not silent:
            rag_logger.debug(f"Error extracting image object {obj_num}: {e}")
        return None


def _extract_flatedecode_image(obj: Any) -> Optional[Image.Image]:
    """Extract image from FlateDecode compressed PDF object."""
    try:
        # Get decompressed data (pypdf automatically decompresses)
        image_data = obj.get_data()
        
        # Get image properties
        width = obj.get('/Width', 0)
        height = obj.get('/Height', 0)
        bits_per_component = obj.get('/BitsPerComponent', 8)
        color_space = obj.get('/ColorSpace', '/DeviceRGB')
        decode_array = obj.get('/Decode', None)  # Check for color inversion
        
        if width > 0 and height > 0:
            # Determine image mode based on color space and bits
            if color_space == '/DeviceGray' or (isinstance(color_space, list) and color_space and color_space[0] == '/DeviceGray'):
                mode = 'L'  # Grayscale
                channels = 1
            elif color_space == '/DeviceRGB' or (isinstance(color_space, list) and color_space and color_space[0] == '/DeviceRGB'):
                mode = 'RGB'
                channels = 3
            elif color_space == '/DeviceCMYK' or (isinstance(color_space, list) and color_space and color_space[0] == '/DeviceCMYK'):
                mode = 'CMYK'
                channels = 4
            elif color_space == '/Indexed' or (isinstance(color_space, list) and len(color_space) > 0 and color_space[0] == '/Indexed'):
                mode = 'P'  # Palette mode
                channels = 1
            else:
                # Default to RGB
                mode = 'RGB'
                channels = 3
            
            # Calculate expected data size
            expected_size = width * height * channels * (bits_per_component // 8)
            if bits_per_component == 1:
                expected_size = (width * height * channels + 7) // 8
            
            # Adjust data size if needed
            if len(image_data) < expected_size:
                # Pad with zeros if needed
                image_data += b'\x00' * (expected_size - len(image_data))
            elif len(image_data) > expected_size:
                # Truncate if too large
                image_data = image_data[:expected_size]
            
            # Create image from raw data
            if bits_per_component == 1:
                # 1-bit images need special handling
                img = Image.frombytes('1', (width, height), image_data)
                img = img.convert('L')  # Convert to grayscale for better OCR
            else:
                img = Image.frombytes(mode, (width, height), image_data)
            
            # Convert CMYK to RGB properly to avoid color inversion
            if img.mode == 'CMYK':
                try:
                    from PIL import ImageCms
                    # Try to convert using ICC profile if available
                    if hasattr(img, 'info') and 'icc_profile' in img.info:
                        img = ImageCms.profileToProfile(img, img.info['icc_profile'], 'sRGB')
                    else:
                        # Standard CMYK to RGB conversion
                        img = img.convert('RGB')
                except (ImportError, Exception):
                    # Fallback to standard conversion
                    img = img.convert('RGB')
            
            # Check if Decode array indicates color inversion and fix it
            if decode_array:
                # Decode array [1, 0] means colors are inverted (white=0, black=1)
                # Decode array [0, 1] is normal (white=1, black=0)
                if isinstance(decode_array, list) and len(decode_array) >= 2:
                    if decode_array[0] > decode_array[1]:  # Inverted (e.g., [1, 0])
                        # Invert the image colors
                        if img.mode == 'RGB':
                            from PIL import ImageOps
                            img = ImageOps.invert(img)
                        elif img.mode == 'L':  # Grayscale
                            from PIL import ImageOps
                            img = ImageOps.invert(img)
                        elif img.mode == 'CMYK':
                            # For CMYK, convert to RGB first, then invert
                            img = img.convert('RGB')
                            from PIL import ImageOps
                            img = ImageOps.invert(img)
            
            return img
        return None
        
    except Exception as e:
        rag_logger.debug(f"Failed to extract FlateDecode image: {e}")
        return None


def filter_relevant_images(images_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out generic/decorative images that aren't relevant to the content.
    
    Filters:
    - Very small images (likely icons/logos) - less than 50x50 pixels
    - Images with no meaningful OCR text (decorative elements)
    - Generic patterns (logos, copyright notices, watermarks)
    - Extremely narrow/wide images (likely headers/footers/borders)
    
    Returns:
        Filtered list of relevant images
    """
    filtered = []
    
    import re
    
    # Generic patterns that indicate non-content images (logos, branding, etc.)
    generic_patterns = [
        r'copyright',
        r'©',
        r'™',  # Trademark symbol
        r'®',  # Registered trademark symbol
        r'all rights reserved',
        r'logo',
        r'www\.',
        r'http[s]?://',
        r'page \d+',
        r'chapter \d+',
        r'^\d+$',  # Just page numbers
        r'^\W+$',  # Only punctuation/symbols
        r'\bfoundation\b',  # Foundation logos (Hewlett Foundation, Arnold Foundation, etc.)
        r'\buniversity\b',  # University logos
        r'\bopenstax\b',  # Publisher logos
        r'\brice\b',  # University names (when standalone, not in context)
        r'\bhewlett\b',
        r'\barnold\b',
        r'\blaura.*john.*arnold\b',  # Specific foundation names
        r'\bwilliam.*flora.*hewlett\b',
    ]
    
    # Words that suggest generic header/footer content or branding
    generic_words = [
        'copyright', 'reserved', 'published', 'printed',
        'logo', 'watermark', 'confidential', 'draft',
        'foundation', 'university', 'openstax', 'rice',  # Common brand/publisher names
        'hewlett', 'arnold', 'flora', 'william',
        'trademark', 'tm', 'registered'
    ]
    
    # Brand/publisher names that indicate logos (when OCR text is mostly just the name)
    brand_names = [
        'openstax', 'rice', 'hewlett', 'arnold', 'tjaf',
        'connexions', 'cnx', 'publisher', 'press'
    ]
    
    for img in images_data:
        width, height = img.get('image_size', (0, 0))
        ocr_text = img.get('ocr_text', '').lower().strip()
        
        # Skip very small images (icons, logos) - less than 50x50 pixels
        if width < 50 or height < 50:
            rag_logger.debug(f"Filtered small image: {img.get('image_id')} ({width}x{height})")
            continue
        
        # Skip extremely narrow or wide images (headers/footers/borders)
        aspect_ratio = width / height if height > 0 else 1
        if aspect_ratio > 20 or aspect_ratio < 0.05:  # Very wide or very tall
            rag_logger.debug(f"Filtered narrow/wide image: {img.get('image_id')} (ratio: {aspect_ratio:.2f})")
            continue
        
        # Check OCR text quality
        if ocr_text:
            # Skip if OCR text matches generic patterns
            is_generic = False
            
            for pattern in generic_patterns:
                if re.search(pattern, ocr_text, re.IGNORECASE):
                    is_generic = True
                    rag_logger.debug(f"Filtered generic pattern image: {img.get('image_id')} (pattern: {pattern}, text: {ocr_text[:50]})")
                    break
            
            if is_generic:
                continue
            
            # Check if OCR text is just a brand/publisher name (common in logos)
            ocr_lower = ocr_text.lower().strip()
            # Remove common symbols and punctuation for comparison
            ocr_clean = re.sub(r'[™®©™®\W]+', ' ', ocr_lower).strip()
            ocr_words = ocr_clean.split()
            
            # If OCR text is mostly just brand names with trademark symbols, it's a logo
            brand_matches = sum(1 for word in ocr_words if word in brand_names)
            if len(ocr_words) <= 4 and brand_matches > 0:
                # Short text with brand names = likely logo
                is_generic = True
                rag_logger.debug(f"Filtered brand logo image: {img.get('image_id')} (text: {ocr_text[:50]})")
            
            if is_generic:
                continue
            
            # Check if OCR text contains mostly generic words
            if len(ocr_words) > 0:
                generic_word_count = sum(1 for word in ocr_words if word in generic_words)
                # If more than 50% of words are generic, filter out
                if generic_word_count / len(ocr_words) > 0.5:
                    rag_logger.debug(f"Filtered generic word image: {img.get('image_id')} (text: {ocr_text[:50]})")
                    continue
                
                # If OCR text is too short (less than 3 characters), likely decorative
                meaningful_chars = len(re.sub(r'\W', '', ocr_text))
                if meaningful_chars < 3:
                    rag_logger.debug(f"Filtered minimal text image: {img.get('image_id')} (text: {ocr_text[:20]})")
                    continue
                
                # Check for trademark symbols - strong indicator of logos
                if re.search(r'[™®]', ocr_text):
                    # If text is short and contains trademark, likely a logo
                    if len(ocr_words) <= 5:
                        rag_logger.debug(f"Filtered trademark logo image: {img.get('image_id')} (text: {ocr_text[:50]})")
                        continue
        else:
            # No OCR text - might still be relevant if it's a diagram/chart
            # But if it's very small or has unusual aspect ratio, skip it
            if width < 100 or height < 100:
                rag_logger.debug(f"Filtered small image with no OCR: {img.get('image_id')} ({width}x{height})")
                continue
        
        # Image passed all filters - include it
        filtered.append(img)
    
    return filtered


def get_images_for_page(images_data: List[Dict[str, Any]], page_num: int, filter_generic: bool = True) -> List[Dict[str, Any]]:
    """Get all images associated with a specific page number, optionally filtered for relevance."""
    page_images = [img for img in images_data if img.get('page') == page_num]
    
    if filter_generic:
        return filter_relevant_images(page_images)
    
    return page_images


def get_images_for_pages(images_data: List[Dict[str, Any]], page_nums: List[int]) -> List[Dict[str, Any]]:
    """Get all images associated with a list of page numbers."""
    return [img for img in images_data if img.get('page') in page_nums]


def extract_captions_from_page_text(page_text: str) -> List[str]:
    """
    Extract FIGURE and TABLE captions from page text, including multi-line captions.
    
    Captions are typically formatted as:
    - "FIGURE X.X Description..." (single line)
    - "FIGURE X.X Description that\nspans multiple lines..." (multi-line)
    - "TABLE X.X Description..." (table captions)
    - "Table X Description..." (various formats)
    
    Args:
        page_text: Full page text content
        
    Returns:
        List of caption strings found on the page (including full multi-line captions for both FIGURE and TABLE)
    """
    import re
    
    captions = []
    
    # ✅ MULTI-LINE CAPTION EXTRACTION: Find all FIGURE and TABLE captions and capture full text
    # Pattern to find "FIGURE X.X" or "Figure X.X" or "TABLE X.X" or "Table X.X" at the start
    # Then capture everything until:
    # 1. Next "FIGURE"/"Figure"/"TABLE"/"Table" (start of next caption)
    # 2. Two consecutive newlines (paragraph break)
    # 3. End of text
    
    # Split text into lines for better multi-line handling
    lines = page_text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this line starts with FIGURE, Figure, TABLE, or Table
        figure_match = re.match(r'^(?:FIGURE|Figure)\s+([\d\.]+)(?:\s+|)(.*)$', line, re.IGNORECASE)
        table_match = re.match(r'^(?:TABLE|Table)\s+([\d\.]+)(?:\s+|)(.*)$', line, re.IGNORECASE)
        
        if figure_match:
            # Found a figure caption start
            figure_number = figure_match.group(1)
            caption_start = figure_match.group(2).strip()
            
            # Build the full caption by collecting lines until:
            # - Next FIGURE/Figure/TABLE/Table caption
            # - Two consecutive empty lines (paragraph break)
            # - End of text
            caption_lines = [f"FIGURE {figure_number} {caption_start}".strip()]
            
            # Continue collecting lines for multi-line caption
            i += 1
            empty_line_count = 0
            
            while i < len(lines):
                current_line = lines[i].strip()
                
                # Stop if we hit another FIGURE, Figure, TABLE, or Table caption
                if re.match(r'^(?:FIGURE|Figure|TABLE|Table)\s+[\d\.]+', current_line, re.IGNORECASE):
                    break
                
                # Stop if we hit two consecutive empty lines (paragraph break)
                if not current_line:
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                    i += 1
                    continue
                else:
                    empty_line_count = 0
                
                # Stop if line looks like start of new section (all caps, very short, etc.)
                # But allow continuation if it's clearly part of caption
                if len(current_line) > 0:
                    # Check if this might be a new section (very short line, all caps, etc.)
                    # But be lenient - continue if it looks like caption continuation
                    if len(current_line) < 5 and current_line.isupper() and i < len(lines) - 1:
                        # Might be a section header, check next line
                        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                        if next_line and not re.match(r'^(?:FIGURE|Figure|TABLE|Table)\s+[\d\.]+', next_line, re.IGNORECASE):
                            # Next line is not a figure/table, might be section header
                            # But continue anyway to be safe
                            pass
                    
                    caption_lines.append(current_line)
                
                i += 1
            
            # Join all caption lines
            full_caption = ' '.join(caption_lines).strip()
            
            # Clean up: remove excessive whitespace but preserve line breaks as spaces
            full_caption = re.sub(r'\s+', ' ', full_caption)
            
            # Only include substantial captions (more than just "FIGURE X.X")
            if len(full_caption) > len(f"FIGURE {figure_number}") + 5:
                captions.append(full_caption)
            
            # Don't increment i here - we already did in the while loop
            continue
        
        elif table_match:
            # Found a table caption start
            table_number = table_match.group(1)
            caption_start = table_match.group(2).strip()
            
            # Build the full caption by collecting lines until:
            # - Next FIGURE/Figure/TABLE/Table caption
            # - Two consecutive empty lines (paragraph break)
            # - End of text
            caption_lines = [f"TABLE {table_number} {caption_start}".strip()]
            
            # Continue collecting lines for multi-line caption
            i += 1
            empty_line_count = 0
            
            while i < len(lines):
                current_line = lines[i].strip()
                
                # Stop if we hit another FIGURE, Figure, TABLE, or Table caption
                if re.match(r'^(?:FIGURE|Figure|TABLE|Table)\s+[\d\.]+', current_line, re.IGNORECASE):
                    break
                
                # Stop if we hit two consecutive empty lines (paragraph break)
                if not current_line:
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                    i += 1
                    continue
                else:
                    empty_line_count = 0
                
                # Stop if line looks like start of new section (all caps, very short, etc.)
                # But allow continuation if it's clearly part of caption
                if len(current_line) > 0:
                    # Check if this might be a new section (very short line, all caps, etc.)
                    # But be lenient - continue if it looks like caption continuation
                    if len(current_line) < 5 and current_line.isupper() and i < len(lines) - 1:
                        # Might be a section header, check next line
                        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                        if next_line and not re.match(r'^(?:FIGURE|Figure|TABLE|Table)\s+[\d\.]+', next_line, re.IGNORECASE):
                            # Next line is not a figure/table, might be section header
                            # But continue anyway to be safe
                            pass
                    
                    caption_lines.append(current_line)
                
                i += 1
            
            # Join all caption lines
            full_caption = ' '.join(caption_lines).strip()
            
            # Clean up: remove excessive whitespace but preserve line breaks as spaces
            full_caption = re.sub(r'\s+', ' ', full_caption)
            
            # Only include substantial captions (more than just "TABLE X.X")
            if len(full_caption) > len(f"TABLE {table_number}") + 5:
                captions.append(full_caption)
            
            # Don't increment i here - we already did in the while loop
            continue
        
        i += 1
    
    # Remove duplicates while preserving order
    seen = set()
    unique_captions = []
    for cap in captions:
        # Normalize for comparison (remove extra spaces)
        normalized = re.sub(r'\s+', ' ', cap.lower().strip())
        if normalized not in seen:
            seen.add(normalized)
            unique_captions.append(cap)
    
    return unique_captions


def parse_figure_caption(caption_text: str) -> Dict[str, Any]:
    """
    Parse a FIGURE caption into structured components.
    
    Extracts:
    - figure_number: The numeric identifier (e.g., "1.8", "1.9", "15")
    - caption_text: The description part after the figure number
    - full_caption: The complete original caption
    
    Args:
        caption_text: Full caption string like "FIGURE 1.8 Isaac Newton..."
        
    Returns:
        Dictionary with 'figure_number', 'caption_text', and 'full_caption' keys
    """
    import re
    
    if not caption_text:
        return {
            'figure_number': None,
            'caption_text': '',
            'full_caption': ''
        }
    
    # Pattern to extract figure number and caption text
    # Matches: "FIGURE 1.8 Description..." or "Figure 2.3 Text..." or "FIGURE 15 Text..."
    # Also handles cases without space: "Figure 29.6The bones..." -> extracts "29.6" and "The bones..."
    pattern = r'(?:FIGURE|Figure)\s+([\d\.]+)(?:\s+|)(.+)'
    match = re.search(pattern, caption_text, re.IGNORECASE)
    
    if match:
        figure_number = match.group(1)
        caption_description = match.group(2).strip()
        return {
            'figure_number': figure_number,
            'caption_text': caption_description,
            'full_caption': caption_text
        }
    else:
        # If pattern doesn't match, return the full caption as caption_text
        return {
            'figure_number': None,
            'caption_text': caption_text.strip(),
            'full_caption': caption_text
        }


def parse_table_caption(caption_text: str) -> Dict[str, Any]:
    """
    Parse a TABLE caption into structured components.
    
    Extracts:
    - table_number: The numeric identifier (e.g., "1.8", "1.9", "15")
    - caption_text: The description part after the table number
    - full_caption: The complete original caption
    
    Args:
        caption_text: Full caption string like "TABLE 1.8 Revenue data..."
        
    Returns:
        Dictionary with 'table_number', 'caption_text', and 'full_caption' keys
    """
    import re
    
    if not caption_text:
        return {
            'table_number': None,
            'caption_text': '',
            'full_caption': ''
        }
    
    # Pattern to extract table number and caption text
    # Matches: "TABLE 1.8 Description..." or "Table 2.3 Text..." or "TABLE 15 Text..."
    # Also handles cases without space: "Table 29.6Data..." -> extracts "29.6" and "Data..."
    pattern = r'(?:TABLE|Table)\s+([\d\.]+)(?:\s+|)(.+)'
    match = re.search(pattern, caption_text, re.IGNORECASE)
    
    if match:
        table_number = match.group(1)
        caption_description = match.group(2).strip()
        return {
            'table_number': table_number,
            'caption_text': caption_description,
            'full_caption': caption_text
        }
    else:
        # If pattern doesn't match, return the full caption as caption_text
        return {
            'table_number': None,
            'caption_text': caption_text.strip(),
            'full_caption': caption_text
        }