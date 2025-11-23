"""
Vector store document loader and processor for Meetara Core.
"""
import os
import json
import hashlib
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.logger import rag_logger
from app.rag.domain_retrievers import get_domain_retriever
from app.core.document_validator import document_validator
from app.core.config_loader import ConfigLoader
from app.rag.image_extractor import extract_images_from_pdf, get_images_for_page, filter_relevant_images, extract_captions_from_page_text, parse_figure_caption, parse_table_caption
from app.rag.table_extractor import extract_tables_from_pdf, get_tables_for_page, format_table_reference, TABLE_EXTRACTION_AVAILABLE

# Suppress non-critical PDF rendering warnings (Pat1 color errors are harmless)
warnings.filterwarnings('ignore', message='.*Pat1.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*invalid float value.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*gray non-stroke color.*', category=UserWarning)


class VectorLoader:
    """Document loader and processor for vector store ingestion."""
    
    # Supported file types and their loaders
    LOADER_MAPPING = {
        '.txt': TextLoader,
        '.pdf': PDFMinerLoader,
        '.docx': Docx2txtLoader,
        '.md': UnstructuredMarkdownLoader,
    }
    
    # PDF loaders to try in order
    PDF_LOADERS = [
        PDFMinerLoader,
        # Add fallback PDF loaders
        # 'PDFPlumberLoader',  # Uncomment if pdfplumber is installed
    ]
    
    # Domains that require page number tracking (academic/research use cases)
    PAGE_NUMBER_DOMAINS = ['academic_tutoring', 'academic_tutoring_research', 'research', 'scientific_research', 'research_assistance']
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        self.config_loader = ConfigLoader()
        self.domain_threshold = 0.3  # Minimum relevance score for domain inclusion
    
    def load_document(self, file_path: Path, domain: Optional[str] = None, force: bool = False) -> List[Document]:
        """Load a single document and return Document objects.
        
        Args:
            file_path: Path to the document file
            domain: Optional domain name - if provided and domain requires page numbers,
                   will use PyPDFLoader for PDFs to preserve page numbers
        """
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension not in self.LOADER_MAPPING:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Special handling for text files that might have encoding issues
            if file_extension == '.txt':
                return self._load_text_file_robust(file_path)
            
            # Special handling for PDFs - use page-aware loader for academic domains
            if file_extension == '.pdf':
                return self._load_pdf_robust(file_path, domain, force)
            
            # Load document with standard loader
            loader_class = self.LOADER_MAPPING[file_extension]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Add metadata with validation
            for doc in documents:
                # Ensure doc.metadata is a dictionary
                if not isinstance(doc.metadata, dict):
                    rag_logger.warning(f"Document metadata is not a dict: {type(doc.metadata)}")
                    doc.metadata = {}
                
                # Update with file metadata
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': file_extension,
                    'file_name': file_path.name,
                })
                
                # Debug: Check document content
                content_length = len(doc.page_content.strip())
                rag_logger.info(f"Document {file_path.name} content length: {content_length} characters")
                if content_length == 0:
                    rag_logger.warning(f"Document {file_path.name} has empty content!")
                elif content_length < 100:
                    rag_logger.warning(f"Document {file_path.name} has very short content: {content_length} characters")
                    rag_logger.debug(f"Content preview: {doc.page_content[:200]}...")
            
            rag_logger.info(f"Loaded {len(documents)} documents from {file_path.name}")
            return documents
            
        except Exception as e:
            rag_logger.error(f"Failed to load document {file_path}: {e}")
            raise
    
    def _load_text_file_robust(self, file_path: Path) -> List[Document]:
        """Robust text file loader that handles encoding issues."""
        try:
            # Try to read with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document manually
            doc = Document(
                page_content=content,
                metadata={
                    'source': str(file_path),
                    'file_type': '.txt',
                    'file_name': file_path.name
                }
            )
            
            rag_logger.info(f"Loaded 1 document from {file_path.name}")
            return [doc]
            
        except Exception as e:
            rag_logger.error(f"Failed to load text file {file_path}: {e}")
            raise
    
    def _is_content_already_captured(self, page_content: str, doc_metadata: Optional[Dict] = None, force: bool = False) -> bool:
        """
        Check if image OCR text and figure captions are already captured in the document content.
        
        Args:
            page_content: Document page content
            doc_metadata: Optional document metadata to check for caption data
            force: If True, always return False to force re-processing
        
        Returns:
            True if content appears to be fully processed (OCR + captions), False otherwise
        """
        # If force mode, always re-process
        if force:
            return False
        
        if not page_content:
            return False
        
        content_lower = page_content.lower()
        
        # Check if [Image OCR Text] marker is present (indicates images were processed)
        has_ocr_marker = "[image ocr text]" in content_lower
        
        # Check if [Table Data] marker is present (indicates tables were processed)
        has_table_marker = "[table data]" in content_lower
        
        # Check metadata for actual caption data (more reliable than content patterns)
        has_captions_in_metadata = False
        if doc_metadata and 'associated_images' in doc_metadata:
            try:
                import json
                associated_images = json.loads(doc_metadata['associated_images']) if isinstance(doc_metadata['associated_images'], str) else doc_metadata['associated_images']
                # Check if any image has caption data
                for img_meta in associated_images if isinstance(associated_images, list) else []:
                    if img_meta.get('captions') or img_meta.get('caption_details') or img_meta.get('figure_number'):
                        has_captions_in_metadata = True
                        break
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        
        # Check if FIGURE captions are in content text
        import re
        has_figure_in_text = bool(re.search(r'figure\s+\d+\.\d+', content_lower, re.IGNORECASE))
        
        # Only consider "already captured" if BOTH OCR AND captions are present (or if tables are already processed)
        # This allows re-processing if captions were missing from previous uploads
        if has_table_marker:
            # If tables are already processed, consider content captured
            return True
        
        if has_ocr_marker and (has_captions_in_metadata or has_figure_in_text):
            return True
        
        # If OCR exists but no captions found, return False to allow caption extraction
        return False
    
    def _load_pdf_robust(self, file_path: Path, domain: Optional[str] = None, force: bool = False) -> List[Document]:
        """Robust PDF loader that tries multiple loaders. Uses PyPDFLoader for domains requiring page numbers."""
        # Check file size - skip image extraction for very large files to prevent timeouts
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        # Use more generous threshold for image extraction - only skip for extremely large files
        # 300MB threshold allows most academic PDFs to extract images while still protecting against true monsters
        skip_image_extraction = file_size_mb > 300  # Skip for files > 300MB
        
        # Check if domain requires page number tracking
        requires_page_numbers = domain and domain.lower() in [d.lower() for d in self.PAGE_NUMBER_DOMAINS]
        
        # First, do a quick check to see if content is already captured
        # Load one sample page to check if images are already processed
        # Skip if force mode is enabled
        content_already_captured = False
        if file_path.suffix.lower() == '.pdf' and not skip_image_extraction and not force:
            try:
                # Quick sample: try to load first page to check if content is already processed
                # Try PyPDFLoader first (more reliable for page-aware loading)
                try:
                    loader = PyPDFLoader(str(file_path))
                    sample_docs = loader.load()
                except (ImportError, Exception):
                    # Fallback to PDFMinerLoader if PyPDFLoader not available
                    loader = PDFMinerLoader(str(file_path))
                    sample_docs = loader.load()
                
                if sample_docs and len(sample_docs) > 0:
                    # Check if first few pages already have image content with captions
                    sample_pages = sample_docs[:min(3, len(sample_docs))]  # Check first 3 pages
                    content_already_captured = any(
                        self._is_content_already_captured(doc.page_content, doc.metadata, force) 
                        for doc in sample_pages
                    )
                    if content_already_captured:
                        rag_logger.info(f"Content already captured (with captions) in {file_path.name} - skipping image extraction")
            except Exception as e:
                # If quick check fails, proceed with normal extraction
                rag_logger.debug(f"Quick content check failed for {file_path.name}: {e} - proceeding with normal extraction")
        
        # Check if images already exist on disk (to avoid re-extraction when only captions needed)
        images_base_dir = Path("images")
        domain_name = domain or "general_health"
        file_stem = file_path.stem
        existing_images_dir = images_base_dir / domain_name / file_stem
        
        images_already_exist = existing_images_dir.exists() and any(existing_images_dir.glob("*.png")) if existing_images_dir.exists() else False
        
        # Extract images from PDF (works for all PDFs, but skipped if already captured or for very large files)
        images_data = []
        if file_path.suffix.lower() == '.pdf' and not skip_image_extraction and not content_already_captured:
            # If images already exist on disk and we're just adding captions, load existing images instead of re-extracting
            if images_already_exist and not force:
                rag_logger.info(f"📷 Images already exist for {file_path.name} - loading from disk (skipping re-extraction, only processing captions)")
                images_data = self._load_existing_images(existing_images_dir, file_stem)
                if images_data:
                    rag_logger.info(f"   ✅ Loaded {len(images_data)} existing images from disk (OCR already in document content)")
                else:
                    # If loading failed, fall back to extraction
                    rag_logger.warning(f"   ⚠️ Failed to load existing images, will re-extract")
                    images_already_exist = False  # Force re-extraction
            
            # Only extract if images don't exist or loading failed
            if not images_already_exist:
                try:
                    # Log start of image extraction for large files
                    rag_logger.info(f"📷 Starting image extraction for {file_path.name}...")
                    
                    # Pre-extract page texts for smart OCR skipping
                    # This helps determine if PDF is scanned and if captions exist
                    page_texts = None
                    try:
                        from pypdf import PdfReader
                        pdf_reader = PdfReader(str(file_path))
                        page_texts = {}
                        for page_num, page in enumerate(pdf_reader.pages, start=1):
                            try:
                                page_text = page.extract_text() if hasattr(page, 'extract_text') else ""
                                if page_text:
                                    page_texts[page_num] = page_text
                            except:
                                pass
                        del pdf_reader
                        if page_texts:
                            rag_logger.debug(f"Extracted page texts from {len(page_texts)} pages for smart OCR skipping")
                    except Exception as e:
                        rag_logger.debug(f"Could not pre-extract page texts for smart OCR: {e}")
                    
                    # Extract images (silent mode for batch processing, but progress logs still shown)
                    images_data = extract_images_from_pdf(
                        pdf_path=file_path,
                        output_base_dir=images_base_dir,
                        domain=domain_name,
                        filename=file_stem,
                        silent=True,  # Silent mode, but progress logs still shown for large files
                        page_texts=page_texts  # Pass page texts for smart OCR skipping
                    )
                    
                    if images_data:
                        # Calculate image extraction statistics
                        pages_with_images = len(set(img.get('page', 0) for img in images_data))
                        images_with_ocr = len([img for img in images_data if img.get('ocr_text')])
                        ocr_success_rate = (images_with_ocr / len(images_data) * 100) if images_data else 0
                        total_ocr_chars = sum(len(img.get('ocr_text', '')) for img in images_data)
                        
                        rag_logger.info(f"📷 IMAGE EXTRACTION COMPLETE for {file_path.name}:")
                        rag_logger.info(f"   - Total images extracted: {len(images_data)}")
                        rag_logger.info(f"   - Pages with images: {pages_with_images}")
                        rag_logger.info(f"   - Images with OCR: {images_with_ocr}/{len(images_data)} ({ocr_success_rate:.1f}%)")
                        rag_logger.info(f"   - Total OCR characters: {total_ocr_chars:,}")
                        
                        # Log image size distribution
                        image_sizes = [img.get('image_size', (0, 0)) for img in images_data if img.get('image_size')]
                        if image_sizes:
                            avg_width = sum(w for w, h in image_sizes) / len(image_sizes) if image_sizes else 0
                            avg_height = sum(h for w, h in image_sizes) / len(image_sizes) if image_sizes else 0
                            rag_logger.info(f"   - Average image size: {int(avg_width)}x{int(avg_height)} pixels")
                    else:
                        rag_logger.info(f"📷 No images extracted from {file_path.name}")
                except Exception as e:
                    rag_logger.warning(f"Image extraction failed for {file_path.name}: {e}")
        elif skip_image_extraction:
            rag_logger.info(f"Skipping image extraction for large PDF {file_path.name} ({file_size_mb:.1f} MB) to prevent timeout - document text will be processed")
        
        # Extract tables from PDF (if PyMuPDF is available) - preserves table structure as Markdown
        tables_data = []
        if file_path.suffix.lower() == '.pdf' and TABLE_EXTRACTION_AVAILABLE and not content_already_captured:
            try:
                rag_logger.info(f"📊 Extracting tables from {file_path.name}...")
                tables_data = extract_tables_from_pdf(
                    pdf_path=file_path,
                    silent=True  # Silent mode for batch processing
                )
                if tables_data:
                    pages_with_tables = len(set(t['page'] for t in tables_data))
                    total_table_rows = sum(t['row_count'] for t in tables_data)
                    rag_logger.info(f"📊 TABLE EXTRACTION COMPLETE: {len(tables_data)} table(s) found across {pages_with_tables} page(s), {total_table_rows:,} total rows")
                else:
                    rag_logger.info(f"📊 No tables found in {file_path.name}")
            except Exception as e:
                rag_logger.warning(f"Table extraction failed for {file_path.name}: {e} (will continue with text-only extraction)")
        
        if requires_page_numbers:
            # Use PyPDFLoader for academic domains (preserves page numbers)
            try:
                loader = PyPDFLoader(str(file_path))
                
                # For large PDFs, use lazy loading to process page-by-page and avoid memory spikes
                # This prevents loading thousands of pages into memory at once
                if file_size_mb > 50:  # Use lazy loading for PDFs > 50MB
                    rag_logger.info(f"Using lazy loading for large PDF {file_path.name} ({file_size_mb:.1f} MB)")
                    documents = []
                    try:
                        # Use lazy_load() to process pages incrementally
                        for i, doc in enumerate(loader.lazy_load(), start=0):
                            # Process each page as it loads (memory efficient)
                            if not isinstance(doc.metadata, dict):
                                doc.metadata = {}
                            # CRITICAL: Normalize page numbers to 1-based to match image extraction
                            # Image extraction uses enumerate(..., start=1), so pages are 1-based (1, 2, 3...)
                            # We must use 1-based page numbers so images match documents correctly
                            # PyPDFLoader may return 0-based or 1-based, but we always normalize to 1-based
                            existing_page = doc.metadata.get('page', None)
                            if existing_page is not None:
                                # PyPDFLoader provided a page number - check if it's 0-based or 1-based
                                # If it equals the 0-based index i, it's 0-based → convert to 1-based
                                # If it equals i+1, it's already 1-based → keep it
                                if existing_page == i:
                                    # 0-based: convert to 1-based
                                    doc.metadata['page'] = i + 1
                                # Otherwise, assume PyPDFLoader provided correct 1-based number
                            else:
                                # No page number from PyPDFLoader - set to 1-based (matching image extraction)
                                doc.metadata['page'] = i + 1
                            documents.append(doc)
                            
                            # Log progress every 50 pages
                            if (i + 1) % 50 == 0:
                                rag_logger.info(f"Loaded {i + 1} pages from {file_path.name}...")
                    except AttributeError:
                        # Fallback to regular load if lazy_load() not available
                        rag_logger.warning("lazy_load() not available, falling back to regular load()")
                        documents = loader.load()
                else:
                    documents = loader.load()
                
                # Initialize statistics tracking
                stats = {
                    'total_pages': 0,
                    'pages_with_text': 0,
                    'pages_with_images': 0,
                    'pages_with_captions': 0,
                    'total_text_chars': 0,
                    'total_captions_found': 0,
                    'total_figure_numbers': 0,
                    'images_with_metadata': 0
                }
                
                # PyPDFLoader returns one Document per page with 'page' metadata
                # Ensure all documents have proper metadata and associate images
                # IMPORTANT: Page numbers must be 1-based to match image extraction (which uses 1-based)
                for i, doc in enumerate(documents):
                    if not isinstance(doc.metadata, dict):
                        doc.metadata = {}
                    
                    # CRITICAL: Normalize page numbers to 1-based to match image extraction
                    # Image extraction uses enumerate(..., start=1), so pages are 1-based (1, 2, 3...)
                    # We must use 1-based page numbers so images match documents correctly
                    existing_page = doc.metadata.get('page', None)
                    if existing_page is not None:
                        # PyPDFLoader provided a page number - check if it's 0-based or 1-based
                        # If it equals the 0-based index i, it's 0-based → convert to 1-based
                        # If it equals i+1, it's already 1-based → keep it
                        if existing_page == i:
                            # 0-based: convert to 1-based
                            doc.metadata['page'] = i + 1
                        # Otherwise, assume PyPDFLoader provided correct 1-based number
                    else:
                        # No page number from PyPDFLoader - set to 1-based (matching image extraction)
                        doc.metadata['page'] = i + 1
                    
                    # Get page number (should be 1-based now)
                    page_num = doc.metadata.get('page', i + 1)
                    stats['total_pages'] += 1
                    
                    # Add file metadata
                    doc.metadata.update({
                        'source': str(file_path),
                        'file_type': '.pdf',
                        'file_name': file_path.name,
                    })
                    
                    # Track text extraction statistics
                    text_content = doc.page_content.strip()
                    if text_content:
                        stats['pages_with_text'] += 1
                        stats['total_text_chars'] += len(text_content)
                    
                    # Check if content is already captured for this page - skip if already processed
                    if self._is_content_already_captured(doc.page_content, doc.metadata, force):
                        rag_logger.debug(f"Page {page_num}: Content already captured (with captions) - skipping image/caption extraction")
                        continue
                    
                    # Associate images with this page (with filtering for generic images)
                    if images_data:
                        all_page_images = get_images_for_page(images_data, page_num, filter_generic=False)
                        page_images = get_images_for_page(images_data, page_num, filter_generic=True)
                        if len(all_page_images) > len(page_images):
                            rag_logger.debug(f"Page {page_num}: Filtered {len(all_page_images) - len(page_images)} generic images")
                        if page_images:
                            stats['pages_with_images'] += 1
                            # Extract FIGURE and TABLE captions from page text
                            # Only extract if not already present (check includes caption detection)
                            page_captions = []
                            if not self._is_content_already_captured(doc.page_content, doc.metadata, force):
                                page_captions = extract_captions_from_page_text(doc.page_content)
                                if page_captions:
                                    stats['pages_with_captions'] += 1
                                    stats['total_captions_found'] += len(page_captions)
                            
                            # ✅ SEPARATE FIGURE and TABLE captions
                            figure_captions = [cap for cap in page_captions if cap.upper().startswith('FIGURE')]
                            table_captions = [cap for cap in page_captions if cap.upper().startswith('TABLE')]
                            
                            # Add image metadata to document (serialize as JSON string for ChromaDB compatibility)
                            # ✅ SYNC VALIDATION: Ensure images match page number and have correct metadata
                            images_metadata = []
                            sync_errors = []
                            # Get tables for this page (if available)
                            page_tables_for_metadata = []
                            if tables_data:
                                page_tables_for_metadata = get_tables_for_page(tables_data, page_num)
                            
                            for img in page_images:
                                # ✅ VALIDATE: Image page number must match document page number
                                img_page = img.get('page', None)
                                if img_page is not None and img_page != page_num:
                                    sync_error = f"Page mismatch: Image {img.get('image_id')} has page {img_page} but document is page {page_num}"
                                    sync_errors.append(sync_error)
                                    rag_logger.warning(sync_error)
                                    # Continue anyway - might be adjacent page image
                                
                                # ✅ COMPLETE METADATA: Include captions, figures, and tables from the start
                                img_meta = {
                                    'page': img_page if img_page is not None else page_num,  # Use image's page or fallback to doc page
                                    'image_path': img['image_path'],
                                    'image_id': img['image_id'],
                                    'ocr_text': img.get('ocr_text', ''),
                                    'image_size': img.get('image_size', (0, 0)),  # (width, height) tuple
                                    'format': img.get('format', 'PNG')
                                }
                                
                                # ✅ SYNC: Add structured FIGURE caption details if any were found on this page
                                # FIGURE captions are associated with images
                                if figure_captions:
                                    # Parse FIGURE captions into structured format (figure number, caption text, full caption)
                                    parsed_figure_captions = [parse_figure_caption(cap) for cap in figure_captions]
                                    img_meta['captions'] = figure_captions  # Keep full captions for backward compatibility
                                    img_meta['caption_details'] = parsed_figure_captions  # Structured caption data
                                    # If there's a primary caption, extract its components for easy access
                                    if parsed_figure_captions:
                                        primary_caption = parsed_figure_captions[0]
                                        if primary_caption.get('figure_number'):
                                            img_meta['figure_number'] = primary_caption['figure_number']
                                            stats['total_figure_numbers'] += 1
                                        if primary_caption.get('caption_text'):
                                            img_meta['caption'] = primary_caption['caption_text']
                                
                                # ✅ ADD TABLES: Associate tables with images on the same page
                                if page_tables_for_metadata:
                                    # Add table references to image metadata with TABLE captions
                                    table_refs = []
                                    for table in page_tables_for_metadata:
                                        table_ref = {
                                            'table_id': table.get('table_id'),
                                            'page': table.get('page'),
                                            'row_count': table.get('row_count', 0),
                                            'col_count': table.get('col_count', 0)
                                        }
                                        # ✅ ADD TABLE CAPTIONS: Find matching TABLE caption for this table
                                        # Match by table number if available (e.g., table_42_1 -> TABLE 1.2)
                                        table_id = table.get('table_id', '')
                                        # Try to match table caption by position or number
                                        if table_captions:
                                            # For now, associate all table captions on this page
                                            # In future, could match by table number more precisely
                                            parsed_table_captions = [parse_table_caption(cap) for cap in table_captions]
                                            table_ref['captions'] = table_captions
                                            table_ref['caption_details'] = parsed_table_captions
                                        
                                        table_refs.append(table_ref)
                                    img_meta['associated_tables'] = table_refs
                                
                                # ✅ VALIDATE: Ensure image path exists (sync check)
                                image_path_obj = Path(img_meta['image_path'])
                                if not image_path_obj.exists():
                                    sync_error = f"Image file missing: {img_meta['image_path']} for image {img_meta['image_id']}"
                                    sync_errors.append(sync_error)
                                    rag_logger.warning(sync_error)
                                
                                stats['images_with_metadata'] += 1
                                images_metadata.append(img_meta)
                            
                            # ✅ SYNC: Store synchronized image metadata
                            doc.metadata['associated_images'] = json.dumps(images_metadata)
                            
                            # ✅ VALIDATE: Log sync status
                            if sync_errors:
                                rag_logger.warning(f"Page {page_num}: Found {len(sync_errors)} sync issue(s) with images")
                            else:
                                rag_logger.debug(f"Page {page_num}: ✅ Image-metadata sync validated - {len(images_metadata)} image(s) with metadata")
                            
                            # Enhanced logging for image and caption association
                            if page_captions:
                                caption_summary = []
                                if figure_captions:
                                    caption_summary.append(f"{len(figure_captions)} FIGURE")
                                if table_captions:
                                    caption_summary.append(f"{len(table_captions)} TABLE")
                                caption_info = " + ".join(caption_summary) if caption_summary else "captions"
                                rag_logger.info(f"Page {page_num}: Found {caption_info} caption(s) - associating with {len(page_images)} image(s)")
                                # Log first caption details
                                if figure_captions:
                                    first_cap = figure_captions[0]
                                    parsed_first = parse_figure_caption(first_cap)
                                    fig_num = parsed_first.get('figure_number', 'N/A')
                                    rag_logger.debug(f"   First FIGURE caption: {fig_num} - {parsed_first.get('caption_text', '')[:60]}...")
                                if table_captions:
                                    first_table_cap = table_captions[0]
                                    parsed_table = parse_table_caption(first_table_cap)
                                    table_num = parsed_table.get('table_number', 'N/A')
                                    rag_logger.debug(f"   First TABLE caption: {table_num} - {parsed_table.get('caption_text', '')[:60]}...")
                            else:
                                # Log images without captions
                                rag_logger.debug(f"Page {page_num}: {len(page_images)} image(s) - no captions found")
                            
                            # Log detailed image metadata every 100 pages
                            if page_num % 100 == 0:
                                ocr_count = len([img.get('ocr_text') for img in page_images if img.get('ocr_text')])
                                rag_logger.info(f"   Progress: Page {page_num} - {len(page_images)} image(s), {ocr_count} with OCR, {len(page_captions)} caption(s)")
                            
                            # Add OCR text to document content for searchability (only if not already present)
                            ocr_texts = [img.get('ocr_text', '') for img in page_images if img.get('ocr_text')]
                            if ocr_texts and "[Image OCR Text]" not in doc.page_content:
                                # Append OCR text to page content so it's searchable
                                doc.page_content += "\n\n[Image OCR Text]\n" + "\n".join(ocr_texts)
                            
                            # Add tables to document content for this page (preserves structure as Markdown)
                            if tables_data:
                                page_tables = get_tables_for_page(tables_data, page_num)
                                if page_tables:
                                    # Add table markdown to document content
                                    table_markdowns = []
                                    for table in page_tables:
                                        table_ref = format_table_reference(table['table_id'], table['markdown'])
                                        table_markdowns.append(table_ref)
                                    if table_markdowns and "[Table Data]" not in doc.page_content:
                                        doc.page_content += "\n\n[Table Data]\n" + "\n".join(table_markdowns)
                                        rag_logger.debug(f"Page {page_num}: Added {len(page_tables)} table(s) in Markdown format")
                
                # Extract PDF metadata ONCE (not per page) - moved outside loop for efficiency
                pdf_metadata = None
                try:
                    from pypdf import PdfReader
                    pdf_reader = PdfReader(str(file_path))
                    pdf_metadata = pdf_reader.metadata
                except Exception as e:
                    rag_logger.debug(f"Could not extract PDF metadata: {e}")
                
                # Apply PDF metadata to all documents
                if pdf_metadata:
                    for doc in documents:
                        if pdf_metadata.get('/Author'):
                            doc.metadata['author'] = pdf_metadata.get('/Author')
                        if pdf_metadata.get('/Title'):
                            doc.metadata['title'] = pdf_metadata.get('/Title')
                        if pdf_metadata.get('/CreationDate'):
                            doc.metadata['creation_date'] = pdf_metadata.get('/CreationDate')
                
                if documents and any(len(doc.page_content.strip()) > 0 for doc in documents):
                    # Log comprehensive statistics
                    rag_logger.info(f"📄 TEXT EXTRACTION COMPLETE for {file_path.name}:")
                    rag_logger.info(f"   - Total pages loaded: {stats['total_pages']}")
                    rag_logger.info(f"   - Pages with text: {stats['pages_with_text']} ({stats['pages_with_text']/stats['total_pages']*100:.1f}%)")
                    rag_logger.info(f"   - Total text characters: {stats['total_text_chars']:,}")
                    rag_logger.info(f"   - Average text per page: {stats['total_text_chars']//stats['pages_with_text'] if stats['pages_with_text'] > 0 else 0:,} chars")
                    
                    rag_logger.info(f"📷 IMAGE METADATA COMPLETE for {file_path.name}:")
                    rag_logger.info(f"   - Pages with images: {stats['pages_with_images']} ({stats['pages_with_images']/stats['total_pages']*100:.1f}%)")
                    rag_logger.info(f"   - Images with metadata: {stats['images_with_metadata']}")
                    rag_logger.info(f"   - Pages with captions: {stats['pages_with_captions']} ({stats['pages_with_captions']/stats['total_pages']*100:.1f}%)")
                    rag_logger.info(f"   - Total captions found: {stats['total_captions_found']}")
                    rag_logger.info(f"   - Total figure numbers extracted: {stats['total_figure_numbers']}")
                    
                    rag_logger.info(f"✅ Loaded {len(documents)} pages from {file_path.name} using PyPDFLoader (page numbers preserved)")
                    return documents
                rag_logger.debug(f"PyPDFLoader returned empty documents for {file_path.name}")
            except ImportError:
                rag_logger.warning(f"PyPDFLoader not available - install pypdf package for page number support")
            except Exception as e:
                rag_logger.warning(f"PyPDFLoader failed for {file_path.name}: {e}, falling back to PDFMinerLoader")
        
        # Fallback to standard PDFMinerLoader for non-academic domains or if PyPDFLoader fails
        for loader_class in self.PDF_LOADERS:
            try:
                loader = loader_class(str(file_path))
                documents = loader.load()
                
                # Add metadata with validation and associate images
                for doc in documents:
                    if not isinstance(doc.metadata, dict):
                        doc.metadata = {}
                    
                    # Try to extract page number from metadata if available
                    page_num = doc.metadata.get('page', None)
                    
                    doc.metadata.update({
                        'source': str(file_path),
                        'file_type': '.pdf',
                        'file_name': file_path.name,
                    })
                    
                    # Check if content is already captured for this page - skip if already processed
                    if self._is_content_already_captured(doc.page_content, doc.metadata, force):
                        rag_logger.debug(f"Page {page_num or 'unknown'}: Content already captured (with captions) - skipping image/caption extraction")
                    # Associate images if we have page numbers (some loaders provide this)
                    # Filter out generic/decorative images (logos, watermarks, etc.)
                    elif images_data and page_num is not None:
                        page_images = get_images_for_page(images_data, page_num, filter_generic=True)
                        if page_images:
                            # Extract FIGURE and TABLE captions from page text (only if not already present)
                            page_captions = []
                            if not self._is_content_already_captured(doc.page_content, doc.metadata, force):
                                page_captions = extract_captions_from_page_text(doc.page_content)
                            
                            # ✅ SEPARATE FIGURE and TABLE captions
                            figure_captions = [cap for cap in page_captions if cap.upper().startswith('FIGURE')]
                            table_captions = [cap for cap in page_captions if cap.upper().startswith('TABLE')]
                            
                            # Serialize image metadata as JSON string for ChromaDB compatibility
                            # ✅ SYNC VALIDATION: Ensure images match page number and have correct metadata
                            images_metadata = []
                            sync_errors = []
                            # Get tables for this page (if available)
                            page_tables_for_metadata = []
                            if tables_data:
                                page_tables_for_metadata = get_tables_for_page(tables_data, page_num)
                            
                            for img in page_images:
                                # ✅ VALIDATE: Image page number must match document page number
                                img_page = img.get('page', None)
                                if img_page is not None and img_page != page_num:
                                    sync_error = f"Page mismatch: Image {img.get('image_id')} has page {img_page} but document is page {page_num}"
                                    sync_errors.append(sync_error)
                                    rag_logger.warning(sync_error)
                                
                                # ✅ COMPLETE METADATA: Include captions, figures, and tables from the start
                                img_meta = {
                                    'page': img_page if img_page is not None else page_num,  # Use image's page or fallback to doc page
                                    'image_path': img['image_path'],
                                    'image_id': img['image_id'],
                                    'ocr_text': img.get('ocr_text', ''),
                                    'image_size': img.get('image_size', (0, 0)),  # (width, height) tuple
                                    'format': img.get('format', 'PNG')
                                }
                                
                                # ✅ SYNC: Add structured FIGURE caption details if any were found on this page
                                if figure_captions:
                                    # Parse FIGURE captions into structured format
                                    parsed_figure_captions = [parse_figure_caption(cap) for cap in figure_captions]
                                    img_meta['captions'] = figure_captions  # Full captions for backward compatibility
                                    img_meta['caption_details'] = parsed_figure_captions  # Structured caption data
                                    if parsed_figure_captions:
                                        primary_caption = parsed_figure_captions[0]
                                        if primary_caption.get('figure_number'):
                                            img_meta['figure_number'] = primary_caption['figure_number']
                                        if primary_caption.get('caption_text'):
                                            img_meta['caption'] = primary_caption['caption_text']
                                
                                # ✅ ADD TABLES: Associate tables with images on the same page
                                if page_tables_for_metadata:
                                    # Add table references to image metadata with TABLE captions
                                    table_refs = []
                                    for table in page_tables_for_metadata:
                                        table_ref = {
                                            'table_id': table.get('table_id'),
                                            'page': table.get('page'),
                                            'row_count': table.get('row_count', 0),
                                            'col_count': table.get('col_count', 0)
                                        }
                                        # ✅ ADD TABLE CAPTIONS: Associate TABLE captions with tables
                                        if table_captions:
                                            parsed_table_captions = [parse_table_caption(cap) for cap in table_captions]
                                            table_ref['captions'] = table_captions
                                            table_ref['caption_details'] = parsed_table_captions
                                        
                                        table_refs.append(table_ref)
                                    img_meta['associated_tables'] = table_refs
                                
                                # ✅ VALIDATE: Ensure image path exists (sync check)
                                image_path_obj = Path(img_meta['image_path'])
                                if not image_path_obj.exists():
                                    sync_error = f"Image file missing: {img_meta['image_path']} for image {img_meta['image_id']}"
                                    sync_errors.append(sync_error)
                                    rag_logger.warning(sync_error)
                                
                                images_metadata.append(img_meta)
                            
                            # ✅ SYNC: Store synchronized image metadata
                            doc.metadata['associated_images'] = json.dumps(images_metadata)
                            
                            # ✅ VALIDATE: Log sync status
                            if sync_errors:
                                rag_logger.warning(f"Page {page_num}: Found {len(sync_errors)} sync issue(s) with images")
                            else:
                                rag_logger.debug(f"Page {page_num}: ✅ Image-metadata sync validated - {len(images_metadata)} image(s) with metadata")
                            
                            if page_captions:
                                caption_summary = []
                                if figure_captions:
                                    caption_summary.append(f"{len(figure_captions)} FIGURE")
                                if table_captions:
                                    caption_summary.append(f"{len(table_captions)} TABLE")
                                caption_info = " + ".join(caption_summary) if caption_summary else "captions"
                                rag_logger.info(f"Page {page_num}: Found {caption_info} caption(s)")
                            
                            # Add OCR text to content (only if not already present)
                            ocr_texts = [img.get('ocr_text', '') for img in page_images if img.get('ocr_text')]
                            if ocr_texts and "[Image OCR Text]" not in doc.page_content:
                                doc.page_content += "\n\n[Image OCR Text]\n" + "\n".join(ocr_texts)
                            
                            # Add tables to document content for this page (preserves structure as Markdown)
                            if tables_data:
                                page_tables = get_tables_for_page(tables_data, page_num)
                                if page_tables:
                                    table_markdowns = []
                                    for table in page_tables:
                                        table_ref = format_table_reference(table['table_id'], table['markdown'])
                                        table_markdowns.append(table_ref)
                                    if table_markdowns and "[Table Data]" not in doc.page_content:
                                        doc.page_content += "\n\n[Table Data]\n" + "\n".join(table_markdowns)
                                        rag_logger.debug(f"Page {page_num}: Added {len(page_tables)} table(s) in Markdown format")
                    elif images_data:
                        # If no page number, associate all images with document (fallback)
                        # But still filter out generic images (logos, decorative, etc.)
                        # Only if content not already captured (checks for captions in metadata)
                        if not self._is_content_already_captured(doc.page_content, doc.metadata, force):
                            filtered_images = filter_relevant_images(images_data)
                            if filtered_images:
                                # Extract FIGURE and TABLE captions from page text
                                page_captions = extract_captions_from_page_text(doc.page_content)
                                
                                # ✅ SEPARATE FIGURE and TABLE captions
                                figure_captions = [cap for cap in page_captions if cap.upper().startswith('FIGURE')]
                                table_captions = [cap for cap in page_captions if cap.upper().startswith('TABLE')]
                                
                                # Serialize image metadata as JSON string for ChromaDB compatibility
                                # ✅ SYNC VALIDATION: Ensure images have correct metadata (fallback case - no page numbers)
                                images_metadata = []
                                sync_errors = []
                                # Get tables for this document (if available, for fallback case)
                                doc_tables = []
                                if tables_data:
                                    # In fallback case, associate all tables with document
                                    doc_tables = tables_data
                                
                                for img in filtered_images:
                                    # ✅ COMPLETE METADATA: Include captions, figures, and tables from the start
                                    img_meta = {
                                        'page': img.get('page', None),  # May be None if no page number available
                                        'image_path': img['image_path'],
                                        'image_id': img['image_id'],
                                        'ocr_text': img.get('ocr_text', ''),
                                        'image_size': img.get('image_size', (0, 0)),  # (width, height) tuple
                                        'format': img.get('format', 'PNG')
                                    }
                                    
                                    # ✅ SYNC: Add structured FIGURE caption details if any were found
                                    if figure_captions:
                                        # Parse FIGURE captions into structured format
                                        parsed_figure_captions = [parse_figure_caption(cap) for cap in figure_captions]
                                        img_meta['captions'] = figure_captions  # Full captions for backward compatibility
                                        img_meta['caption_details'] = parsed_figure_captions  # Structured caption data
                                        if parsed_figure_captions:
                                            primary_caption = parsed_figure_captions[0]
                                            if primary_caption.get('figure_number'):
                                                img_meta['figure_number'] = primary_caption['figure_number']
                                            if primary_caption.get('caption_text'):
                                                img_meta['caption'] = primary_caption['caption_text']
                                    
                                    # ✅ ADD TABLES: Associate tables with images (if tables exist on this document)
                                    if doc_tables:
                                        # Add table references to image metadata with TABLE captions
                                        table_refs = []
                                        for table in doc_tables:
                                            table_ref = {
                                                'table_id': table.get('table_id'),
                                                'page': table.get('page'),
                                                'row_count': table.get('row_count', 0),
                                                'col_count': table.get('col_count', 0)
                                            }
                                            # ✅ ADD TABLE CAPTIONS: Associate TABLE captions with tables
                                            if table_captions:
                                                parsed_table_captions = [parse_table_caption(cap) for cap in table_captions]
                                                table_ref['captions'] = table_captions
                                                table_ref['caption_details'] = parsed_table_captions
                                            
                                            table_refs.append(table_ref)
                                        img_meta['associated_tables'] = table_refs
                                    
                                    # ✅ VALIDATE: Ensure image path exists (sync check)
                                    image_path_obj = Path(img_meta['image_path'])
                                    if not image_path_obj.exists():
                                        sync_error = f"Image file missing: {img_meta['image_path']} for image {img_meta['image_id']}"
                                        sync_errors.append(sync_error)
                                        rag_logger.warning(sync_error)
                                    
                                    images_metadata.append(img_meta)
                                
                                # ✅ SYNC: Store synchronized image metadata
                                doc.metadata['associated_images'] = json.dumps(images_metadata)
                                
                                # ✅ VALIDATE: Log sync status
                                if sync_errors:
                                    rag_logger.warning(f"Fallback (no page number): Found {len(sync_errors)} sync issue(s) with images")
                                else:
                                    rag_logger.debug(f"Fallback (no page number): ✅ Image-metadata sync validated - {len(images_metadata)} image(s) with metadata")
                                
                                if page_captions:
                                    caption_summary = []
                                    if figure_captions:
                                        caption_summary.append(f"{len(figure_captions)} FIGURE")
                                    if table_captions:
                                        caption_summary.append(f"{len(table_captions)} TABLE")
                                    caption_info = " + ".join(caption_summary) if caption_summary else "captions"
                                    rag_logger.info(f"Found {caption_info} caption(s) - associating with images")
                                
                                ocr_texts = [img.get('ocr_text', '') for img in filtered_images if img.get('ocr_text')]
                                if ocr_texts:
                                    doc.page_content += "\n\n[Image OCR Text]\n" + "\n".join(ocr_texts)
                                
                                # Add all tables to document (since no page numbers, add all tables)
                                if tables_data:
                                    table_markdowns = []
                                    for table in tables_data:
                                        table_ref = format_table_reference(table['table_id'], table['markdown'])
                                        table_markdowns.append(table_ref)
                                    if table_markdowns and "[Table Data]" not in doc.page_content:
                                        doc.page_content += "\n\n[Table Data]\n" + "\n".join(table_markdowns)
                                        rag_logger.debug(f"Added {len(tables_data)} table(s) in Markdown format")
                                
                                if len(filtered_images) < len(images_data):
                                    rag_logger.debug(f"Filtered {len(images_data) - len(filtered_images)} generic images")
                
                if documents and any(len(doc.page_content.strip()) > 0 for doc in documents):
                    rag_logger.info(f"Loaded {len(documents)} documents from {file_path.name} using {loader_class.__name__}")
                    return documents
                rag_logger.debug(f"PDF loader {loader_class.__name__} returned empty documents for {file_path.name}")
            except Exception as e:
                rag_logger.debug(f"PDF loader {loader_class.__name__} failed for {file_path.name}: {e}")
        
        # Fallback: Try to extract basic text
        try:
            rag_logger.warning(f"Trying fallback text extraction for {file_path.name}")
            fallback_text = self._extract_pdf_text_fallback(file_path)
            if fallback_text.strip():
                doc = Document(
                    page_content=fallback_text,
                    metadata={
                        'source': str(file_path),
                        'file_type': '.pdf',
                        'file_name': file_path.name,
                        'extraction_method': 'fallback'
                    }
                )
                # Add all images to fallback document (with filtering)
                # Only if content not already captured (checks for captions in metadata)
                if images_data and not self._is_content_already_captured(doc.page_content, doc.metadata, force):
                    filtered_images = filter_relevant_images(images_data)
                    if filtered_images:
                        # Extract FIGURE and TABLE captions from page text
                        page_captions = extract_captions_from_page_text(doc.page_content)
                        
                        # ✅ SEPARATE FIGURE and TABLE captions
                        figure_captions = [cap for cap in page_captions if cap.upper().startswith('FIGURE')]
                        table_captions = [cap for cap in page_captions if cap.upper().startswith('TABLE')]
                        
                        # Serialize image metadata as JSON string for ChromaDB compatibility
                        images_metadata = []
                        # Get tables for fallback document (if available)
                        doc_tables = []
                        if tables_data:
                            doc_tables = tables_data
                        
                        for img in filtered_images:
                            img_meta = {
                                'page': img['page'],
                                'image_path': img['image_path'],
                                'image_id': img['image_id'],
                                'ocr_text': img.get('ocr_text', ''),
                                'image_size': img.get('image_size', (0, 0)),  # (width, height) tuple
                                'format': img.get('format', 'PNG')
                            }
                            # ✅ SYNC: Add structured FIGURE caption details if any were found
                            if figure_captions:
                                # Parse FIGURE captions into structured format
                                parsed_figure_captions = [parse_figure_caption(cap) for cap in figure_captions]
                                img_meta['captions'] = figure_captions  # Full captions for backward compatibility
                                img_meta['caption_details'] = parsed_figure_captions  # Structured caption data
                                if parsed_figure_captions:
                                    primary_caption = parsed_figure_captions[0]
                                    if primary_caption.get('figure_number'):
                                        img_meta['figure_number'] = primary_caption['figure_number']
                                    if primary_caption.get('caption_text'):
                                        img_meta['caption'] = primary_caption['caption_text']
                            
                            # ✅ ADD TABLES: Associate tables with images (if tables exist)
                            if doc_tables:
                                # Add table references to image metadata with TABLE captions
                                table_refs = []
                                for table in doc_tables:
                                    table_ref = {
                                        'table_id': table.get('table_id'),
                                        'page': table.get('page'),
                                        'row_count': table.get('row_count', 0),
                                        'col_count': table.get('col_count', 0)
                                    }
                                    # ✅ ADD TABLE CAPTIONS: Associate TABLE captions with tables
                                    if table_captions:
                                        parsed_table_captions = [parse_table_caption(cap) for cap in table_captions]
                                        table_ref['captions'] = table_captions
                                        table_ref['caption_details'] = parsed_table_captions
                                    
                                    table_refs.append(table_ref)
                                img_meta['associated_tables'] = table_refs
                            
                            images_metadata.append(img_meta)
                        
                        doc.metadata['associated_images'] = json.dumps(images_metadata)
                        ocr_texts = [img.get('ocr_text', '') for img in filtered_images if img.get('ocr_text')]
                        if ocr_texts:
                            doc.page_content += "\n\n[Image OCR Text]\n" + "\n".join(ocr_texts)
                        
                        # Add all tables to fallback document
                        if tables_data:
                            table_markdowns = []
                            for table in tables_data:
                                table_ref = format_table_reference(table['table_id'], table['markdown'])
                                table_markdowns.append(table_ref)
                            if table_markdowns and "[Table Data]" not in doc.page_content:
                                doc.page_content += "\n\n[Table Data]\n" + "\n".join(table_markdowns)
                                rag_logger.info(f"Added {len(tables_data)} table(s) in Markdown format to fallback document")
                        
                        if len(filtered_images) < len(images_data):
                            rag_logger.info(f"Filtered {len(images_data) - len(filtered_images)} generic images (logos, decorative, etc.)")
                        if page_captions:
                            caption_summary = []
                            if figure_captions:
                                caption_summary.append(f"{len(figure_captions)} FIGURE")
                            if table_captions:
                                caption_summary.append(f"{len(table_captions)} TABLE")
                            caption_info = " + ".join(caption_summary) if caption_summary else "captions"
                            rag_logger.info(f"Found {caption_info} caption(s) in fallback extraction")
                
                rag_logger.info(f"Loaded 1 document from {file_path.name} using fallback extraction")
                return [doc]
        except Exception as e:
            rag_logger.error(f"Fallback PDF extraction failed for {file_path.name}: {e}")
        
        rag_logger.warning(f"Failed to load PDF {file_path.name} with all methods.")
        return []
    
    def _extract_pdf_text_fallback(self, file_path: Path) -> str:
        """Fallback text extraction for PDFs that can't be loaded with standard loaders."""
        try:
            # Try using pdfplumber if available
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except ImportError:
            rag_logger.warning("pdfplumber not available for fallback PDF extraction")
        except Exception as e:
            rag_logger.debug(f"pdfplumber extraction failed: {e}")
        
        # Final fallback: return filename as content
        return f"PDF file: {file_path.name} (content extraction failed)"
    
    def _check_existing_document_has_captions(self, retriever, file_hash: str) -> bool:
        """
        Check if an existing document with the given hash has caption data.
        
        Args:
            retriever: Domain retriever instance
            file_hash: File hash to check
            
        Returns:
            True if document exists and has captions, False otherwise
        """
        try:
            # Search for documents with this file hash
            results = retriever.vectorstore.get(
                where={"file_hash": file_hash},
                limit=5  # Check first few documents with this hash
            )
            
            if not results or not results.get('ids') or len(results['ids']) == 0:
                return False
            
            # Check metadata of found documents for caption data
            metadatas = results.get('metadatas', [])
            for metadata in metadatas:
                if isinstance(metadata, dict):
                    # Check if associated_images has caption data
                    if 'associated_images' in metadata:
                        try:
                            import json
                            associated_images = json.loads(metadata['associated_images']) if isinstance(metadata['associated_images'], str) else metadata['associated_images']
                            if isinstance(associated_images, list):
                                for img_meta in associated_images:
                                    if img_meta.get('captions') or img_meta.get('caption_details') or img_meta.get('figure_number'):
                                        return True
                        except (json.JSONDecodeError, TypeError, KeyError):
                            pass
            
            return False
        except Exception as e:
            rag_logger.debug(f"Error checking for captions in existing document: {e}")
            return False  # If check fails, assume no captions (allow re-processing)
    
    def _load_existing_images(self, images_dir: Path, file_stem: str) -> List[Dict[str, Any]]:
        """
        Load existing images from disk instead of re-extracting.
        This is used when images already exist but captions need to be extracted.
        """
        images_data = []
        try:
            import re
            from PIL import Image
            
            # Pattern to match image filenames: {file_stem}_page_{page}_img_{id}.png
            pattern = re.compile(rf"{re.escape(file_stem)}_page_(\d+)_img_(.+?)\.png", re.IGNORECASE)
            
            for image_file in images_dir.glob("*.png"):
                match = pattern.match(image_file.name)
                if match:
                    page_num = int(match.group(1))
                    image_id = f"page_{page_num}_img_{match.group(2)}"
                    
                    # Get image size without loading full image data
                    try:
                        with Image.open(image_file) as img:
                            image_size = img.size
                    except Exception:
                        image_size = (0, 0)
                    
                    images_data.append({
                        'page': page_num,
                        'image_id': image_id,
                        'image_path': str(image_file),
                        'ocr_text': '',  # OCR already in document content, don't re-extract
                        'image_size': image_size,
                        'format': 'PNG'
                    })
            
            # Sort by page number for consistency
            images_data.sort(key=lambda x: x['page'])
            
        except Exception as e:
            rag_logger.warning(f"Error loading existing images from {images_dir}: {e}")
            return []
        
        return images_data
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text content from file for validation purposes."""
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_extension == '.pdf':
                loader = PDFMinerLoader(str(file_path))
                documents = loader.load()
                return " ".join([doc.page_content for doc in documents])
            
            elif file_extension == '.docx':
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()
                return " ".join([doc.page_content for doc in documents])
            
            elif file_extension == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
                documents = loader.load()
                return " ".join([doc.page_content for doc in documents])
            
            else:
                rag_logger.warning(f"Unsupported file type for text extraction: {file_extension}")
                return ""
                
        except Exception as e:
            rag_logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def load_documents_from_directory(self, directory_path: Path) -> List[Document]:
        """Load all supported documents from a directory."""
        documents = []
        
        try:
            for file_path in directory_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.LOADER_MAPPING:
                    try:
                        docs = self.load_document(file_path)  # Domain not available at directory level
                        documents.extend(docs)
                    except Exception as e:
                        rag_logger.warning(f"Skipping {file_path}: {e}")
                        continue
            
            rag_logger.info(f"Loaded {len(documents)} total documents from {directory_path}")
            return documents
            
        except Exception as e:
            rag_logger.error(f"Failed to load documents from directory {directory_path}: {e}")
            raise
    
    def process_and_add_to_domain(
        self, 
        documents: List[Document], 
        domain: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Process documents and add them to a domain's vector store."""
        try:
            if not documents:
                rag_logger.warning("No documents to process")
                return False
            
            # Add additional metadata if provided
            if metadata:
                # Handle metadata that might be a JSON string
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        rag_logger.warning(f"Invalid JSON metadata: {metadata}")
                        metadata = {"raw_metadata": metadata}
                
                # Ensure metadata is a dictionary
                if isinstance(metadata, dict):
                    for doc in documents:
                        doc.metadata.update(metadata)
                else:
                    rag_logger.warning(f"Metadata is not a dictionary: {type(metadata)}")
            
            # Get domain retriever
            retriever = get_domain_retriever(domain)
            
            # Get initial stats (before adding documents)
            initial_stats = retriever.get_collection_stats()
            initial_chunks = initial_stats.get('chunk_count', 0)
            
            # Add documents to vector store (with optimized memory management)
            retriever.add_documents(documents)
            
            # Get final stats (after adding documents)
            final_stats = retriever.get_collection_stats()
            final_chunks = final_stats.get('chunk_count', 0)
            chunks_added = final_chunks - initial_chunks
            
            rag_logger.info(f"✅ Successfully added {len(documents)} documents to domain: {domain}")
            rag_logger.info(f"📊 Vector DB Update Summary for domain '{domain}':")
            rag_logger.info(f"   - Documents added: {len(documents)}")
            rag_logger.info(f"   - Chunks added: {chunks_added:,} (from {initial_chunks:,} to {final_chunks:,})")
            rag_logger.info(f"   - Total chunks now: {final_chunks:,}")
            rag_logger.info(f"   - Total documents now: {final_stats.get('document_count', 0):,}")
            rag_logger.info(f"   - Database size: {final_stats.get('db_size', 'Unknown')}")
            
            return True
            
        except Exception as e:
            rag_logger.error(f"Failed to process documents for domain {domain}: {e}")
            return False
    
    def analyze_document_domains(self, content: str) -> Dict[str, float]:
        """Analyze document content and determine relevant domains with scores."""
        try:
            content_lower = content.lower()
            domain_scores = {}
            
            # Get all available domains
            all_domains = self.config_loader.get_all_domains()
            
            for domain in all_domains:
                # Get domain keywords
                domain_keywords = self.config_loader.get_domain_keywords(domain)
                
                if domain_keywords:
                    # Calculate relevance score based on keyword matches
                    matches = sum(1 for keyword in domain_keywords if keyword.lower() in content_lower)
                    if matches > 0:
                        relevance_score = min(matches / len(domain_keywords), 1.0)
                        domain_scores[domain] = relevance_score
            
            # Use config-driven domain detection for additional content type analysis
            # This supplements the keyword-based scoring above
            for domain in all_domains:
                domain_keywords = self.config_loader.get_domain_keywords(domain)
                if domain_keywords:
                    # Calculate additional relevance based on domain-specific content patterns
                    domain_matches = sum(1 for keyword in domain_keywords if keyword.lower() in content_lower)
                    if domain_matches > 0:
                        # Boost score for domains with strong keyword matches
                        additional_score = min(domain_matches / len(domain_keywords), 0.3)  # Cap at 0.3 additional
                        domain_scores[domain] = domain_scores.get(domain, 0) + additional_score
            
            rag_logger.info(f"Domain analysis results: {domain_scores}")
            return domain_scores
            
        except Exception as e:
            rag_logger.error(f"Error in domain analysis: {e}")
            return {}
    
    def process_and_add_to_multiple_domains(
        self, 
        documents: List[Document], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Process documents and add them to multiple relevant domains."""
        try:
            if not documents:
                rag_logger.warning("No documents to process")
                return {}
            
            # Combine all document content for domain analysis
            combined_content = " ".join([doc.page_content for doc in documents])
            
            # Analyze which domains are relevant
            domain_scores = self.analyze_document_domains(combined_content)
            
            # Filter domains based on threshold
            relevant_domains = {
                domain: score for domain, score in domain_scores.items() 
                if score >= self.domain_threshold
            }
            
            if not relevant_domains:
                rag_logger.warning("No relevant domains found, using general_health as fallback")
                relevant_domains = {"general_health": 0.5}
            
            # Sort domains by relevance score
            sorted_domains = sorted(relevant_domains.items(), key=lambda x: x[1], reverse=True)
            
            results = {}
            
            # Process documents for each relevant domain
            for domain, score in sorted_domains:
                rag_logger.info(f"Processing documents for domain: {domain} (relevance: {score:.2f})")
                
                # Add domain-specific metadata
                domain_metadata = metadata.copy() if metadata else {}
                domain_metadata.update({
                    "domain_relevance_score": score,
                    "primary_domain": sorted_domains[0][0] if sorted_domains else domain,
                    "all_relevant_domains": ", ".join(list(relevant_domains.keys()))  # Convert list to string
                })
                
                # Process and add to this domain
                success = self.process_and_add_to_domain(documents, domain, domain_metadata)
                results[domain] = success
                
                if success:
                    rag_logger.info(f"Successfully added documents to domain: {domain}")
                else:
                    rag_logger.error(f"Failed to add documents to domain: {domain}")
            
            return results
            
        except Exception as e:
            rag_logger.error(f"Failed to process documents for multiple domains: {e}")
            return {}
    
    def load_and_process_file(
        self, 
        file_path: Path, 
        domain: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Dict[str, bool]:
        """Load a single file and add it to multiple relevant domains intelligently.
        
        Args:
            file_path: Path to the document file
            domain: Optional target domain
            metadata: Optional metadata dictionary
            force: If True, skip duplicate detection and force re-upload
        """
        try:
            file_hash = compute_file_hash(file_path)
            
            # Check for duplicates if domain is specified and force is False
            needs_caption_update = False
            if domain and not force:
                retriever = get_domain_retriever(domain)
                if retriever.has_document_with_hash(file_hash):
                    # Check if captions are missing - if so, allow re-processing
                    has_captions = self._check_existing_document_has_captions(retriever, file_hash)
                    if has_captions:
                        rag_logger.info(f"Duplicate detected: {file_path.name} (hash: {file_hash[:8]}...) with captions - skipping upload")
                        return {domain: False}  # Return False to indicate duplicate
                    else:
                        rag_logger.info(f"Duplicate file detected but captions missing: {file_path.name} - will delete old chunks and re-add with captions")
                        needs_caption_update = True
                        # Get count of chunks to delete (for logging)
                        try:
                            results = retriever.vectorstore._collection.get(where={"file_hash": file_hash})
                            chunks_to_delete = len(results.get('ids', []))
                            if chunks_to_delete > 0:
                                rag_logger.info(f"   📊 Found {chunks_to_delete} existing chunk(s) without captions - will delete before re-adding")
                        except Exception as e:
                            rag_logger.debug(f"Could not count chunks to delete: {e}")
                            chunks_to_delete = 0
                        
                        # Delete old chunks before adding new ones (to avoid duplicates)
                        try:
                            deleted = retriever.delete_documents(filter_dict={"file_hash": file_hash})
                            if deleted:
                                if chunks_to_delete > 0:
                                    rag_logger.info(f"   ✅ Deleted {chunks_to_delete} old chunk(s) (without captions) - now re-adding with captions")
                                else:
                                    rag_logger.info(f"   ✅ Prepared for re-adding with captions")
                            else:
                                rag_logger.warning(f"   ⚠️ Could not delete old chunks - duplicates may be created! Vector DB size may increase unnecessarily.")
                        except Exception as e:
                            rag_logger.warning(f"   ⚠️ Error deleting old chunks: {e} - duplicates may be created! Vector DB size may increase unnecessarily.")
            elif force:
                rag_logger.info(f"Force mode enabled: skipping duplicate check for {file_path.name}")
            
            # Load document (pass domain for page-aware PDF loading and force flag)
            documents = self.load_document(file_path, domain, force)
            
            # Add file hash to metadata for future duplicate checking
            if metadata is None:
                metadata = {}
            else:
                # Convert to dict if it's a string
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {"raw_metadata": metadata}
            
            metadata['file_hash'] = file_hash
            
            # If domain is specified, use single-domain processing (backward compatibility)
            if domain:
                success = self.process_and_add_to_domain(documents, domain, metadata)
                return {domain: success}
            
            # Otherwise, use multi-domain intelligent processing
            rag_logger.info(f"Processing file {file_path.name} for multiple domains")
            return self.process_and_add_to_multiple_domains(documents, metadata)
            
        except Exception as e:
            rag_logger.error(f"Failed to load and process file {file_path}: {e}")
            return {}
    
    def load_and_process_directory(
        self, 
        directory_path: Path, 
        domain: str = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Load all documents from a directory and add them to multiple relevant domains intelligently."""
        try:
            # Load documents
            documents = self.load_documents_from_directory(directory_path)
            
            # If domain is specified, use single-domain processing (backward compatibility)
            if domain:
                success = self.process_and_add_to_domain(documents, domain, metadata)
                return {domain: success}
            
            # Otherwise, use multi-domain intelligent processing
            rag_logger.info(f"Processing directory {directory_path} for multiple domains")
            return self.process_and_add_to_multiple_domains(documents, metadata)
            
        except Exception as e:
            rag_logger.error(f"Failed to load and process directory {directory_path}: {e}")
            return {}


# Global loader instance
vector_loader = VectorLoader()


def load_documents_to_vectorstore(
    file_path: Path,
    domain: str = None,
    metadata: Optional[Dict[str, Any]] = None,
    force: bool = False
) -> Dict[str, bool]:
    """Convenience function to load documents to vector store(s)."""
    return vector_loader.load_and_process_file(file_path, domain, metadata, force)


def load_directory_to_vectorstore(
    directory_path: Path,
    domain: str = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """Convenience function to load a directory to vector store(s)."""
    return vector_loader.load_and_process_directory(directory_path, domain, metadata)


def load_documents_to_multiple_domains(
    file_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    force: bool = False
) -> Dict[str, bool]:
    """Convenience function to load documents to multiple relevant domains intelligently."""
    return vector_loader.load_and_process_file(file_path, None, metadata, force)


def get_supported_file_types() -> List[str]:
    """Get list of supported file types."""
    return list(VectorLoader.LOADER_MAPPING.keys())


def validate_file_type(file_path: Path) -> bool:
    """Validate if a file type is supported."""
    return file_path.suffix.lower() in VectorLoader.LOADER_MAPPING 


def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# Add this function to your vectorstore logic (pseudo-code):
def has_document_with_hash(vectorstore, domain, file_hash):
    # This function should search the vectorstore for documents in the domain with metadata['file_hash'] == file_hash
    # Implementation will depend on your vectorstore backend (Chroma, FAISS, etc.)
    # For Chroma, you might use a metadata filter
    results = vectorstore.get_documents(domain, filter={"file_hash": file_hash})
    return len(results) > 0 