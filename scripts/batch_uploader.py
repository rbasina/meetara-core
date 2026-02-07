#!/usr/bin/env python3
"""
Batch Upload Script for Downloaded Documents
Uploads downloaded documents to appropriate domains using config-driven mapping
"""

import json
from pathlib import Path
import time
import sys
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Ensure we operate from project root so config paths resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# Add the project root to the Python path to import config modules reliably
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.core.config_loader import config_loader
from app.rag.vector_loader import vector_loader
from app.core.logger import rag_logger
from app.core.security import security_validator

def get_config_driven_domain_mapping():
    """Get domain mapping from configuration files - only exact domain names"""
    print("Loading domain mapping from configuration files...")
    
    # Get all available domains from config
    available_domains = config_loader.get_all_domains()
    print(f"Found {len(available_domains)} configured domains")
    
    # Only map exact domain names
    domain_mapping = {domain: domain for domain in available_domains}
    for domain in available_domains:
        print(f"  Direct mapping: {domain} -> {domain}")
    
    print(f"\nTotal domain mappings created: {len(domain_mapping)} (no aliases)")
    return domain_mapping

def analyze_file_content_for_domain(file_path: Path) -> str:
    """Analyze file content to determine the best domain match using keywords from config"""
    try:
        # For PDF files, try to extract text using vector_loader
        if file_path.suffix.lower() == '.pdf':
            try:
                # Check file size for large PDFs
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 50:
                    print(f"    [EXTRACT] Extracting text from large PDF ({file_size_mb:.1f} MB) for analysis...")
                # Extract a small amount of text for analysis
                content = vector_loader.extract_text_from_file(file_path)
                # Limit to first 10KB for analysis
                content = content[:10240] if len(content) > 10240 else content
                if file_size_mb > 50:
                    print(f"    [EXTRACT] Text extraction complete, analyzing keywords...")
            except Exception as e:
                rag_logger.debug(f"Could not extract PDF text for analysis: {e}")
                # Fallback to filename analysis for PDFs
                return analyze_filename_for_domain(file_path)
        else:
            # For text-based files, read directly
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10240)  # Read first 10KB
        
        content_lower = content.lower()
        
        # Score each domain based on keyword matches from domain_keywords.yaml
        domain_scores = {}
        
        for domain in config_loader.get_all_domains():
            keywords = config_loader.get_domain_keywords(domain)
            if keywords:
                score = 0
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    # Count keyword occurrences
                    count = content_lower.count(keyword_lower)
                    score += count
                
                if score > 0:
                    domain_scores[domain] = score
                    print(f"    [SCORE] {domain}: {score} keyword matches")
        
        # Return the domain with highest score, or fallback to filename analysis
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            print(f"  [RESULT] Content analysis: {file_path.name} -> {best_domain} (score: {domain_scores[best_domain]})")
            return best_domain
        
    except Exception as e:
        print(f"  [WARNING] Could not analyze content of {file_path.name}: {e}")
    
    # Fallback to filename analysis
    return analyze_filename_for_domain(file_path)

def analyze_filename_for_domain(file_path: Path) -> str:
    """Analyze filename to determine domain using simple pattern matching"""
    filename = file_path.stem.lower()  # Get filename without extension
    
    # Check for domain names in filename
    for domain in config_loader.get_all_domains():
        if domain in filename:
            return domain
    
    # Check for common patterns
    if any(word in filename for word in ['health', 'medical', 'doctor', 'patient']):
        return 'general_health'
    elif any(word in filename for word in ['mental', 'psychology', 'therapy', 'depression', 'anxiety']):
        return 'mental_health'
    elif any(word in filename for word in ['nutrition', 'diet', 'food', 'vitamins']):
        return 'nutrition'
    elif any(word in filename for word in ['sleep', 'insomnia', 'rest']):
        return 'sleep'
    elif any(word in filename for word in ['women', 'pregnancy', 'fertility', 'gynecology']):
        return 'women_health'
    elif any(word in filename for word in ['programming', 'coding', 'software', 'python', 'java']):
        return 'programming'
    elif any(word in filename for word in ['ai', 'ml', 'machine', 'neural', 'artificial']):
        return 'ai_ml'
    elif any(word in filename for word in ['education', 'learning', 'study', 'academic']):
        return 'academic_tutoring'
    elif any(word in filename for word in ['writing', 'creative', 'content']):
        return 'writing'
    elif any(word in filename for word in ['music', 'musical']):
        return 'music'
    elif any(word in filename for word in ['tech', 'computer', 'technology']):
        return 'tech_support'
    elif any(word in filename for word in ['cyber', 'security']):
        return 'cybersecurity'
    elif any(word in filename for word in ['data', 'analytics', 'statistics']):
        return 'data_analysis'
    elif any(word in filename for word in ['research', 'scientific', 'science']):
        return 'research'
    elif any(word in filename for word in ['legal', 'law']):
        return 'legal_assistance'
    elif any(word in filename for word in ['emergency', 'urgent', 'crisis']):
        return 'emergency_care'
    elif any(word in filename for word in ['insurance', 'coverage']):
        return 'insurance'
    elif any(word in filename for word in ['real_estate', 'property']):
        return 'real_estate'
    elif any(word in filename for word in ['parenting', 'children', 'family']):
        return 'parenting'
    elif any(word in filename for word in ['relationships', 'dating', 'marriage']):
        return 'relationships'
    elif any(word in filename for word in ['communication', 'speaking']):
        return 'communication'
    elif any(word in filename for word in ['home', 'housekeeping', 'cleaning']):
        return 'home_management'
    elif any(word in filename for word in ['shopping', 'consumer']):
        return 'shopping'
    elif any(word in filename for word in ['planning', 'goals']):
        return 'planning'
    elif any(word in filename for word in ['transportation', 'travel', 'commuting']):
        return 'transportation'
    elif any(word in filename for word in ['time', 'productivity']):
        return 'time_management'
    elif any(word in filename for word in ['decision', 'choices']):
        return 'decision_making'
    elif any(word in filename for word in ['conflict', 'mediation']):
        return 'conflict_resolution'
    elif any(word in filename for word in ['work_life', 'balance']):
        return 'work_life_balance'
    elif any(word in filename for word in ['stress', 'relaxation']):
        return 'stress_management'
    elif any(word in filename for word in ['yoga', 'meditation']):
        return 'yoga'
    elif any(word in filename for word in ['coaching', 'personal_development']):
        return 'life_coaching'
    elif any(word in filename for word in ['social', 'community']):
        return 'social_support'
    elif any(word in filename for word in ['sports', 'recreation']):
        return 'sports_recreation'
    elif any(word in filename for word in ['fitness', 'exercise', 'workout']):
        return 'fitness_healthcare'
    elif any(word in filename for word in ['remote', 'telecommuting']):
        return 'remote_work'
    elif any(word in filename for word in ['social_media', 'digital_marketing']):
        return 'social_media_management'
    elif any(word in filename for word in ['digital', 'computer_literacy']):
        return 'digital_literacy'
    elif any(word in filename for word in ['language', 'linguistics']):
        return 'language_learning_professional'
    elif any(word in filename for word in ['aeronautics', 'aviation', 'flight']):
        return 'aeronautics'
    elif any(word in filename for word in ['automobile', 'car', 'vehicle']):
        return 'automobile'
    elif any(word in filename for word in ['space', 'aerospace']):
        return 'space_technology'
    elif any(word in filename for word in ['agriculture', 'farming']):
        return 'agriculture'
    elif any(word in filename for word in ['manufacturing', 'industrial']):
        return 'manufacturing'
    elif any(word in filename for word in ['travel', 'tourism']):
        return 'travel_tourism'
    
    # Default fallback
    return "general_health"

def upload_downloaded_documents():
    """Upload documents from downloads directory using config-driven domain mapping"""
    # Create log file for this batch upload session
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_upload_{timestamp}.log"
    
    # Checkpoint file to save progress
    checkpoint_file = log_dir / f"batch_upload_{timestamp}.checkpoint.json"
    
    def log_and_print(message: str, print_to_console: bool = True):
        """Log message to file and optionally print to console"""
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")
        if print_to_console:
            print(message)
    
    log_and_print(f"========== BATCH UPLOAD SESSION STARTED ==========")
    log_and_print(f"Log file: {log_file}")
    log_and_print(f"Checkpoint file: {checkpoint_file}")
    log_and_print(f"[START] Starting config-driven batch upload...")
    log_and_print(f"[INFO] You can safely interrupt (Ctrl+C) and resume later - progress is saved")
    
    # Get config-driven domain mapping
    domain_mapping = get_config_driven_domain_mapping()
    
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Upload documents to vector store')
    parser.add_argument('--domain', help='Target domain for upload (all files go to this domain, no auto-detection)')
    parser.add_argument('--filename', help='Specific file to upload')
    parser.add_argument('--dir', help='Source directory to scan (default: downloads/ or downloads/{domain}/ if domain specified)')
    parser.add_argument('--all-domains', action='store_true', help='Auto-detect domain for each file (ignores --domain)')
    parser.add_argument('--force', action='store_true', help='Force re-upload even if document already exists (skip duplicate detection)')
    parser.add_argument('--dry-run', action='store_true', help='Simulate upload without actually uploading (shows what would happen)')
    args = parser.parse_args()
    
    target_domain_arg = args.domain
    target_filename_arg = args.filename
    source_dir_arg = args.dir
    auto_detect = args.all_domains
    force_upload = args.force
    dry_run = args.dry_run
    
    if dry_run:
        print(f"[MODE] DRY-RUN mode enabled - will simulate upload without actually uploading")
        log_and_print(f"[MODE] DRY-RUN mode enabled - will simulate upload without actually uploading")
    if force_upload:
        print(f"[MODE] Force mode enabled - will re-upload existing documents")
    if target_domain_arg and not auto_detect:
        print(f"[TARGET] Target domain specified: {target_domain_arg}")
        print(f"[MODE] All files will be uploaded to '{target_domain_arg}' (no domain detection)")
    elif auto_detect:
        print(f"[MODE] Auto-detection enabled - each file will be routed to its detected domain")
    if target_filename_arg:
        print(f"[FILE] Target filename specified: {target_filename_arg}")
    
    # Handle file selection
    document_files = []
    if target_filename_arg:
        # If a specific file is provided, use it directly
        file_path = Path(target_filename_arg)
        if not file_path.is_absolute():
            # Convert relative path to absolute
            file_path = Path.cwd() / file_path
        
        print(f"[CHECK] Checking file: {file_path}")
        if file_path.exists():
            document_files = [file_path]
            print(f"[OK] File found: {file_path}")
        else:
            print(f"[ERROR] File not found: {file_path}")
            return
    else:
        # Determine source directory
        if source_dir_arg:
            # User specified directory
            downloads_dir = Path(source_dir_arg)
            print(f"[DIR] Using specified directory: {downloads_dir}")
        elif target_domain_arg and not auto_detect:
            # Domain specified - check both structures (backward compatible)
            # 1. Check category-based structure: downloads/category/domain/
            # 2. Check flat structure: downloads/domain/
            
            # Get category from config
            category = None
            try:
                project_root = Path(__file__).resolve().parents[1]
                config_path = project_root / "config/domain_config.yaml"
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                for cat, cat_config in config.get('categories', {}).items():
                    if target_domain_arg in cat_config.get('domains', {}):
                        category = cat
                        break
            except Exception:
                pass
            
            # Try category-based structure first (new structure)
            if category:
                category_domain_dir = Path("downloads") / category / target_domain_arg
                if category_domain_dir.exists() and any(category_domain_dir.iterdir()):
                    downloads_dir = category_domain_dir
                    print(f"[DIR] Using category-based directory: {downloads_dir} (category: {category})")
                else:
                    # Fallback to flat structure (legacy)
                    flat_domain_dir = Path("downloads") / target_domain_arg
                    if flat_domain_dir.exists() and any(flat_domain_dir.iterdir()):
                        downloads_dir = flat_domain_dir
                        print(f"[DIR] Using flat directory: {downloads_dir} (legacy structure)")
                    else:
                        downloads_dir = Path("downloads")
                        print(f"[DIR] Domain folder not found, using: {downloads_dir}")
            else:
                # No category found - try flat structure
                flat_domain_dir = Path("downloads") / target_domain_arg
                if flat_domain_dir.exists() and any(flat_domain_dir.iterdir()):
                    downloads_dir = flat_domain_dir
                    print(f"[DIR] Using flat directory: {downloads_dir} (legacy structure)")
                else:
                    downloads_dir = Path("downloads")
                    print(f"[DIR] Domain folder not found, using: {downloads_dir}")
        else:
            # Default: search entire downloads directory (supports both structures)
            downloads_dir = Path("downloads")
            print(f"[DIR] Using downloads directory: {downloads_dir} (supports both flat and category-based structures)")
        
        if not downloads_dir.exists():
            print(f"[ERROR] Directory not found: {downloads_dir}")
            return
        
        # Find all document files recursively
        for ext in ['*.pdf', '*.txt', '*.docx', '*.md', '*.json', '*.csv']:
            document_files.extend(downloads_dir.rglob(ext))  # rglob for recursive search
        
        if not document_files:
            print(f"[ERROR] No document files found in {downloads_dir}!")
            return
    
    log_and_print(f"[FOUND] Found {len(document_files)} document files")
    
    # Show file sizes before processing
    log_and_print(f"\n[INFO] File sizes summary:")
    total_size = 0
    large_files = []
    for file_path in document_files:
        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        total_size += file_size
        if size_mb > 100:  # Files over 100MB
            large_files.append((file_path.name, size_mb))
    
    # Show first 10 files or all if less than 10
    display_count = min(10, len(document_files))
    for file_path in document_files[:display_count]:
        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        log_and_print(f"  - {file_path.name}: {size_mb:.2f} MB")
    
    if len(document_files) > display_count:
        log_and_print(f"  ... and {len(document_files) - display_count} more files")
    
    if large_files:
        log_and_print(f"\n[WARNING] Found {len(large_files)} large files (>100MB):")
        for name, size in large_files:
            log_and_print(f"  - {name}: {size:.2f} MB")
        log_and_print(f"[INFO] PDFs process page-by-page (extract text/images only) - memory efficient regardless of file size")
    
    total_size_mb = total_size / (1024 * 1024)
    log_and_print(f"\n[INFO] Total size to process: {total_size_mb:.2f} MB ({len(document_files)} files)")
    
    # Load checkpoint if it exists (for resume)
    processed_files = set()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                processed_files = set(checkpoint_data.get('processed_files', []))
                log_and_print(f"[RESUME] Found checkpoint - {len(processed_files)} files already processed")
                log_and_print(f"[RESUME] Will skip already processed files and continue from where it left off")
        except Exception as e:
            log_and_print(f"[WARNING] Could not load checkpoint: {e} - starting fresh")
    
    # Upload files
    uploaded_count = 0
    failed_files: List[Dict[str, Any]] = []  # Store file info and reason for failure
    successful_files: List[Dict[str, Any]] = []  # Store successful file info
    
    def save_checkpoint():
        """Save current progress to checkpoint file"""
        if dry_run:
            return  # Don't save checkpoints in dry-run mode
        try:
            checkpoint_data = {
                'processed_files': list(processed_files),
                'successful_files': [f['file'] for f in successful_files],
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            log_and_print(f"[WARNING] Could not save checkpoint: {e}")
    
    # Handle interruption gracefully
    import signal
    interrupted = False
    
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        log_and_print(f"\n[INTERRUPTED] Received interrupt signal - saving progress...")
        save_checkpoint()
        log_and_print(f"[INFO] Progress saved to checkpoint. You can resume by running the same command again.")
        log_and_print(f"[INFO] Processed {uploaded_count} files successfully before interruption")
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGBREAK'):  # Windows
        signal.signal(signal.SIGBREAK, signal_handler)
    
    for idx, file_path in enumerate(document_files, 1):
        # Check if already processed (from checkpoint or duplicate detection)
        file_str = str(file_path)
        if file_str in processed_files:
            log_and_print(f"\n[SKIP] [{idx}/{len(document_files)}] Already processed: {file_path.name}")
            uploaded_count += 1  # Count as success since it was already done
            continue
        
        if interrupted:
            break
        # Skip if specific filename is requested and this isn't it
        if target_filename_arg:
            # Extract just the filename from the target path for comparison
            target_filename = Path(target_filename_arg).name
            if file_path.name != target_filename:
                continue
        
        # Get file size for reporting
        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        log_and_print(f"\n[PROCESS] [{idx}/{len(document_files)}] Processing: {file_path.name} ({size_mb:.2f} MB)")
        
        # For PDFs, file size doesn't matter - we process page-by-page, extracting text/images incrementally
        # Only warn if it's a non-PDF file that's very large
        if file_path.suffix.lower() == '.pdf':
            if size_mb > 100:
                log_and_print(f"  [INFO] Large PDF ({size_mb:.2f} MB) - will process page-by-page (memory efficient)")
        else:
            # For non-PDF files, check size
            if size_mb > 100:
                log_and_print(f"  [INFO] Large file ({size_mb:.2f} MB) - processing")
        
        # Determine domain
        if auto_detect or not target_domain_arg:
            # Auto-detect domain for each file
            if size_mb > 50:  # Show progress for large files
                log_and_print(f"  [ANALYZE] Analyzing content for domain detection (this may take a moment for large files)...")
            domain = analyze_file_content_for_domain(file_path)
            print(f"  [DETECT] Auto-detected domain: {domain}")
        else:
            # Use specified domain (no detection)
            domain = target_domain_arg
            print(f"  [DOMAIN] Using specified domain: {domain} (no detection)")
        
        # Validate domain: allow both leaf domains and category names (e.g. --domain business)
        # Category names create/use vectorstore/<category>/ (e.g. vectorstore/business/)
        valid_domains = set(config_loader.get_all_domains()) | set(config_loader.get_all_category_names())
        if domain not in valid_domains:
            log_and_print(f"  [WARNING] Domain '{domain}' not found in config, using 'general_health'")
            domain = "general_health"
        
        # Upload the file directly using vector_loader (no API server needed)
        try:
            if dry_run:
                log_and_print(f"  [DRY-RUN] Would upload to domain: {domain}")
                log_and_print(f"  [DRY-RUN] Would use direct vector_loader (enhanced with page numbers for academic domains)")
                
                # In dry-run, check if file would be duplicate
                is_duplicate = False
                try:
                    from app.rag.vector_loader import compute_file_hash
                    from app.rag.domain_retrievers import get_domain_retriever
                    file_hash = compute_file_hash(file_path)
                    retriever = get_domain_retriever(domain)
                    if retriever.has_document_with_hash(file_hash):
                        is_duplicate = True
                        log_and_print(f"  [DRY-RUN] File already exists in vector DB (would skip)")
                except Exception:
                    pass
                
                if not is_duplicate:
                    log_and_print(f"  [DRY-RUN] Would upload successfully (file not found in vector DB)")
                    uploaded_count += 1
                    successful_files.append({
                        'file': str(file_path),
                        'filename': file_path.name,
                        'size_mb': round(size_mb, 2),
                        'domain': domain,
                        'time_seconds': 0.0,  # No actual processing time in dry-run
                        'vector_db_size': "N/A (dry-run)",
                        'total_chunks': 0
                    })
                else:
                    uploaded_count += 1  # Count as success since it would skip
                    log_and_print(f"  [DRY-RUN] Would skip duplicate")
                
                # Skip actual upload and continue to next file
                continue
            
            log_and_print(f"  [UPLOAD] Uploading to domain: {domain}")
            log_and_print(f"  [METHOD] Using direct vector_loader (enhanced with page numbers for academic domains)")
            
            # For large files, add progress indication
            if size_mb > 100:
                log_and_print(f"  [INFO] Large file - this may take several minutes...")
                start_time = time.time()
            else:
                start_time = time.time()
            
            # Use vector_loader directly - works offline, no API needed
            # vector_loader handles large files by processing page-by-page (PDFs) or in chunks (text files)
            # For PDFs: extracts text/images page-by-page, so memory usage is minimal regardless of file size
            result = vector_loader.load_and_process_file(
                file_path=file_path,
                domain=domain,
                metadata={
                    'uploaded_via': 'batch_uploader',
                    'source_file': str(file_path),
                    'batch_upload': True,
                    'file_size_mb': round(size_mb, 2),
                    'upload_timestamp': datetime.now().isoformat()
                },
                force=force_upload
            )
            
            elapsed = time.time() - start_time
            
            # Get Vector DB stats after upload
            vector_db_size = "Unknown"
            chunks_added = 0
            total_chunks = 0
            try:
                from app.rag.domain_retrievers import get_domain_retriever
                retriever = get_domain_retriever(domain)
                stats = retriever.get_collection_stats()
                vector_db_size = stats.get('db_size', 'Unknown')
                total_chunks = stats.get('chunk_count', 0)
                # Note: chunks_added would need to be tracked separately or calculated
            except Exception as e:
                rag_logger.debug(f"Could not get Vector DB stats: {e}")
            
            if result and result.get(domain, False):
                log_and_print(f"  [OK] Successfully uploaded to domain '{domain}' ({elapsed:.1f}s)")
                if vector_db_size != "Unknown":
                    log_and_print(f"  [INFO] Vector DB size for '{domain}': {vector_db_size} | Total chunks: {total_chunks:,}")
                uploaded_count += 1
                processed_files.add(file_str)  # Mark as processed
                successful_files.append({
                    'file': str(file_path),
                    'filename': file_path.name,
                    'size_mb': round(size_mb, 2),
                    'domain': domain,
                    'time_seconds': round(elapsed, 1),
                    'vector_db_size': vector_db_size,
                    'total_chunks': total_chunks
                })
                # Save checkpoint after each successful upload
                save_checkpoint()
            else:
                # Check if it was a duplicate (file already exists in vector DB)
                is_duplicate = False
                try:
                    from app.rag.vector_loader import compute_file_hash
                    from app.rag.domain_retrievers import get_domain_retriever
                    file_hash = compute_file_hash(file_path)
                    retriever = get_domain_retriever(domain)
                    if retriever.has_document_with_hash(file_hash):
                        is_duplicate = True
                        log_and_print(f"  [SKIP] Already in vector DB (duplicate detected) - skipping")
                        processed_files.add(file_str)  # Mark as processed
                        uploaded_count += 1
                        save_checkpoint()
                except Exception:
                    pass
                
                if not is_duplicate:
                    reason = f"Upload returned False for domain '{domain}'"
                    log_and_print(f"  [FAIL] {reason}")
                    failed_files.append({
                        'file': str(file_path),
                        'filename': file_path.name,
                        'size_mb': round(size_mb, 2),
                        'domain': domain,
                        'reason': reason
                    })
                    
        except FileNotFoundError as e:
            reason = f"File not found: {file_path}"
            log_and_print(f"  [ERROR] {reason}")
            failed_files.append({
                'file': str(file_path),
                'filename': file_path.name,
                'size_mb': round(size_mb, 2),
                'domain': domain,
                'reason': reason
            })
        except PermissionError as e:
            reason = f"Permission denied: cannot read file"
            log_and_print(f"  [ERROR] {reason}")
            failed_files.append({
                'file': str(file_path),
                'filename': file_path.name,
                'size_mb': round(size_mb, 2),
                'domain': domain,
                'reason': reason
            })
        except MemoryError as e:
            reason = f"Out of memory processing file ({size_mb:.2f} MB)"
            log_and_print(f"  [ERROR] {reason}")
            log_and_print(f"  [TIP] Try processing this file individually or increase system memory")
            rag_logger.error(f"Memory error for file {file_path} ({size_mb:.2f} MB)")
            failed_files.append({
                'file': str(file_path),
                'filename': file_path.name,
                'size_mb': round(size_mb, 2),
                'domain': domain,
                'reason': reason
            })
        except Exception as e:
            error_msg = str(e)
            reason = f"Error: {error_msg}"
            log_and_print(f"  [ERROR] Error uploading {file_path.name}: {error_msg}")
            
            # Provide helpful error messages for common issues
            if "too large" in error_msg.lower() or "file size" in error_msg.lower():
                log_and_print(f"  [TIP] File size issue - PDFs are processed page-by-page so this shouldn't happen")
            elif "memory" in error_msg.lower() or "MemoryError" in error_msg:
                log_and_print(f"  [TIP] Insufficient memory - PDFs process page-by-page, so this is unusual")
            elif "corrupted" in error_msg.lower() or "invalid" in error_msg.lower():
                log_and_print(f"  [TIP] File may be corrupted - verify the PDF file is valid")
            elif "decode" in error_msg.lower() or "encoding" in error_msg.lower():
                log_and_print(f"  [TIP] Encoding issue - file may have encoding problems")
            
            rag_logger.error(f"Batch upload error for {file_path}: {e}")
            failed_files.append({
                'file': str(file_path),
                'filename': file_path.name,
                'size_mb': round(size_mb, 2),
                'domain': domain,
                'reason': reason
            })
    
    # Comprehensive Summary
    log_and_print(f"\n{'='*60}")
    if dry_run:
        log_and_print(f"[SUMMARY] DRY-RUN SIMULATION COMPLETE")
    else:
        log_and_print(f"[SUMMARY] BATCH UPLOAD COMPLETE")
    log_and_print(f"{'='*60}")
    if dry_run:
        log_and_print(f"  [DRY-RUN] Would upload successfully: {uploaded_count}/{len(document_files)}")
        log_and_print(f"  [DRY-RUN] Would fail uploads: {len(failed_files)}/{len(document_files)}")
    else:
        log_and_print(f"  [OK] Successfully uploaded: {uploaded_count}/{len(document_files)}")
        log_and_print(f"  [FAIL] Failed uploads: {len(failed_files)}/{len(document_files)}")
    
    # Detailed successful files summary
    if successful_files:
        log_and_print(f"\n[SUCCESS] Successfully Loaded Files ({len(successful_files)}):")
        log_and_print(f"{'-'*60}")
        total_success_size = sum(f['size_mb'] for f in successful_files)
        total_success_time = sum(f['time_seconds'] for f in successful_files)
        
        # Get final Vector DB stats for the domain (if all files went to same domain)
        final_vector_db_size = "Unknown"
        final_total_chunks = 0
        if successful_files:
            unique_domains = set(f['domain'] for f in successful_files)
            if len(unique_domains) == 1:
                # All files went to same domain - get final stats
                try:
                    from app.rag.domain_retrievers import get_domain_retriever
                    domain = successful_files[0]['domain']
                    retriever = get_domain_retriever(domain)
                    stats = retriever.get_collection_stats()
                    final_vector_db_size = stats.get('db_size', 'Unknown')
                    final_total_chunks = stats.get('chunk_count', 0)
                except Exception:
                    pass
        
        for file_info in successful_files:
            log_and_print(f"  ✓ {file_info['filename']}")
            log_and_print(f"    Domain: {file_info['domain']} | Size: {file_info['size_mb']:.2f} MB | Time: {file_info['time_seconds']:.1f}s")
            if file_info.get('vector_db_size') and file_info['vector_db_size'] != "Unknown":
                log_and_print(f"    Vector DB: {file_info['vector_db_size']} | Chunks: {file_info.get('total_chunks', 0):,}")
        
        log_and_print(f"\n  Total file size processed: {total_success_size:.2f} MB")
        if final_vector_db_size != "Unknown":
            log_and_print(f"  Final Vector DB size: {final_vector_db_size} | Total chunks: {final_total_chunks:,}")
        log_and_print(f"  Total processing time: {total_success_time:.1f} seconds ({total_success_time/60:.1f} minutes)")
    
    # Detailed failed files summary with reasons
    if failed_files:
        log_and_print(f"\n[FAILED] Failed Files ({len(failed_files)}) with Reasons:")
        log_and_print(f"{'-'*60}")
        total_failed_size = sum(f['size_mb'] for f in failed_files)
        
        # Group failures by reason for better analysis
        failures_by_reason: Dict[str, List[Dict]] = {}
        for file_info in failed_files:
            reason = file_info['reason']
            if reason not in failures_by_reason:
                failures_by_reason[reason] = []
            failures_by_reason[reason].append(file_info)
        
        for reason, files in failures_by_reason.items():
            log_and_print(f"\n  Reason: {reason} ({len(files)} files)")
            for file_info in files:
                log_and_print(f"    ✗ {file_info['filename']} ({file_info['size_mb']:.2f} MB) - Domain: {file_info['domain']}")
        
        log_and_print(f"\n  Total failed size: {total_failed_size:.2f} MB")
        
        # Summary by reason
        log_and_print(f"\n  Failure Breakdown by Reason:")
        for reason, files in failures_by_reason.items():
            log_and_print(f"    - {reason}: {len(files)} file(s)")
    else:
        log_and_print(f"\n[SUCCESS] All files loaded successfully!")
    
    # Clean up checkpoint file on successful completion
    if not dry_run and not interrupted and len(failed_files) == 0:
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                log_and_print(f"[INFO] Checkpoint file removed (upload complete)")
        except Exception as e:
            log_and_print(f"[WARNING] Could not remove checkpoint file: {e}")
    
    log_and_print(f"\n{'='*60}")
    log_and_print(f"Log file saved to: {log_file}")
    if interrupted:
        log_and_print(f"Checkpoint file: {checkpoint_file} (resume from here)")
    log_and_print(f"Session completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(f"{'='*60}")
    
    # Also print summary to console
    if dry_run:
        print(f"\n[SUMMARY] Dry-Run Simulation Summary:")
        print(f"  [DRY-RUN] Would upload successfully: {uploaded_count}/{len(document_files)}")
        print(f"  [DRY-RUN] Would fail uploads: {len(failed_files)}/{len(document_files)}")
    else:
        print(f"\n[SUMMARY] Upload Summary:")
        print(f"  [OK] Successfully uploaded: {uploaded_count}/{len(document_files)}")
        print(f"  [FAIL] Failed uploads: {len(failed_files)}/{len(document_files)}")
    print(f"\nDetailed summary and logs saved to: {log_file}")

if __name__ == "__main__":
    upload_downloaded_documents() 