#!/usr/bin/env python3
"""
Quick verification script to test upload process for a single file.
Verifies: text extraction, image extraction, table extraction, metadata storage.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.vector_loader import vector_loader, compute_file_hash
from app.rag.domain_retrievers import get_domain_retriever
from app.core.logger import rag_logger


def verify_upload(file_path: Path, domain: str):
    """Verify that a file upload extracts all content correctly."""
    
    print(f"\n{'='*80}")
    print(f"🔍 VERIFYING UPLOAD PROCESS")
    print(f"{'='*80}")
    print(f"File: {file_path.name}")
    print(f"Domain: {domain}")
    print(f"File size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"\n{'='*80}\n")
    
    # Step 1: Load document
    print("📄 Step 1: Loading document...")
    try:
        documents = vector_loader.load_document(file_path, domain)
        print(f"   ✅ Loaded {len(documents)} document(s)")
        
        if not documents:
            print("   ❌ ERROR: No documents loaded!")
            return False
        
        # Check first few documents
        sample_docs = documents[:min(3, len(documents))]
        
        for i, doc in enumerate(sample_docs):
            print(f"\n   📄 Document {i+1}:")
            print(f"      - Content length: {len(doc.page_content)} chars")
            print(f"      - Page number: {doc.metadata.get('page', 'N/A')}")
            print(f"      - File name: {doc.metadata.get('file_name', 'N/A')}")
            
            # Check content markers
            content_lower = doc.page_content.lower()
            if '[image ocr text]' in content_lower:
                print(f"      ✅ Has [Image OCR Text] marker")
            if '[table data]' in content_lower:
                print(f"      ✅ Has [Table Data] marker")
            
            # Check metadata
            print(f"      - Metadata keys: {list(doc.metadata.keys())}")
            
            # Check associated images
            if 'associated_images' in doc.metadata:
                try:
                    images_data = json.loads(doc.metadata['associated_images'])
                    print(f"      ✅ Has {len(images_data)} associated image(s)")
                    
                    if images_data:
                        first_img = images_data[0]
                        print(f"      📷 First image metadata:")
                        print(f"         - Page: {first_img.get('page', 'N/A')}")
                        print(f"         - Image path: {first_img.get('image_path', 'N/A')}")
                        print(f"         - OCR text: {len(first_img.get('ocr_text', ''))} chars")
                        print(f"         - Figure number: {first_img.get('figure_number', 'N/A')}")
                        print(f"         - Caption: {first_img.get('caption', 'N/A')[:60]}..." if first_img.get('caption') else "         - Caption: N/A")
                        print(f"         - Captions array: {len(first_img.get('captions', []))} item(s)")
                        print(f"         - Caption details: {len(first_img.get('caption_details', []))} item(s)")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"      ⚠️  Could not parse associated_images: {e}")
            else:
                print(f"      ⚠️  No associated_images in metadata")
        
    except Exception as e:
        print(f"   ❌ ERROR loading document: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Check file hash
    print(f"\n🔐 Step 2: Computing file hash...")
    try:
        file_hash = compute_file_hash(file_path)
        print(f"   ✅ File hash: {file_hash[:16]}...")
    except Exception as e:
        print(f"   ❌ ERROR computing hash: {e}")
        return False
    
    # Step 3: Check duplicate detection
    print(f"\n🔍 Step 3: Checking duplicate detection...")
    try:
        retriever = get_domain_retriever(domain)
        is_duplicate = retriever.has_document_with_hash(file_hash)
        print(f"   {'✅' if is_duplicate else 'ℹ️ '} File {'already exists' if is_duplicate else 'not found'} in vector store")
        
        if is_duplicate:
            # Check if it has captions
            has_captions = vector_loader._check_existing_document_has_captions(retriever, file_hash)
            print(f"   {'✅' if has_captions else '⚠️ '} Captions: {'Present' if has_captions else 'Missing'}")
    except Exception as e:
        print(f"   ⚠️  Could not check duplicates: {e}")
    
    # Step 4: Verify page number consistency
    print(f"\n📄 Step 4: Verifying page number consistency...")
    page_numbers = [doc.metadata.get('page') for doc in documents if doc.metadata.get('page')]
    if page_numbers:
        min_page = min(page_numbers)
        max_page = max(page_numbers)
        print(f"   ✅ Page numbers: {min_page} to {max_page} (1-based: {'✅' if min_page >= 1 else '❌'})")
        
        if min_page < 1:
            print(f"   ❌ ERROR: Page numbers should be 1-based, but found {min_page}")
            return False
    else:
        print(f"   ⚠️  No page numbers found in documents")
    
    # Step 5: Check image files on disk
    print(f"\n🖼️  Step 5: Checking extracted images on disk...")
    try:
        images_dir = Path("images") / domain / file_path.stem
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            print(f"   ✅ Found {len(image_files)} image file(s) in {images_dir}")
            if image_files:
                print(f"   📷 Sample: {image_files[0].name}")
        else:
            print(f"   ⚠️  Images directory not found: {images_dir}")
    except Exception as e:
        print(f"   ⚠️  Could not check images directory: {e}")
    
    # Step 6: Summary
    print(f"\n{'='*80}")
    print(f"✅ VERIFICATION COMPLETE")
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  - Documents loaded: {len(documents)}")
    print(f"  - Documents with images: {len([d for d in documents if 'associated_images' in d.metadata])}")
    print(f"  - Documents with OCR markers: {len([d for d in documents if '[image ocr text]' in d.page_content.lower()])}")
    print(f"  - Documents with table markers: {len([d for d in documents if '[table data]' in d.page_content.lower()])}")
    print(f"  - Page numbers: {'✅ Consistent (1-based)' if page_numbers and min(page_numbers) >= 1 else '⚠️ Check needed'}")
    print(f"\n{'='*80}\n")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/verify_upload.py <file_path> <domain>")
        print("\nExample:")
        print("  python scripts/verify_upload.py downloads/education/academic_tutoring/test.pdf academic_tutoring")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    domain = sys.argv[2]
    
    if not file_path.exists():
        print(f"❌ ERROR: File not found: {file_path}")
        sys.exit(1)
    
    success = verify_upload(file_path, domain)
    sys.exit(0 if success else 1)

