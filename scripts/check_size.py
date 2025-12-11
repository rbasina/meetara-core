#!/usr/bin/env python3
"""
Check vectorstore size and statistics.
Combines size checking and domain statistics in one tool.
"""
from pathlib import Path
import sys

# Add parent directory to path for domain stats
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_domain_stats(domain: str):
    """Get domain statistics if available."""
    try:
        from app.rag.domain_retrievers import get_domain_retriever
        retriever = get_domain_retriever(domain)
        stats = retriever.get_collection_stats()
        return stats
    except Exception:
        return None

if len(sys.argv) > 1 and sys.argv[1] == "all":
    # Show all domains
    p = Path("vectorstore")
    domains = [d for d in p.iterdir() if d.is_dir()]
    
    print(f"\n📊 Total Vectorstore Size (All Domains)")
    print("=" * 60)
    
    total_all = 0
    domain_sizes = []
    
    for domain in sorted(domains):
        size = sum(f.stat().st_size for f in domain.rglob('*') if f.is_file())
        total_all += size
        domain_sizes.append((domain.name, size))
        
        # Try to get stats
        stats = get_domain_stats(domain.name)
        chunks_info = f" ({stats.get('chunk_count', 0):,} chunks)" if stats else ""
        
        print(f"  {domain.name:30s}: {size/(1024**3):.2f} GB{chunks_info}")
    
    print(f"\n{'Total:':30s}: {total_all/(1024**3):.2f} GB")
    print(f"{'Number of domains:':30s}: {len(domains)}")
else:
    # Show single domain with stats
    domain = sys.argv[1] if len(sys.argv) > 1 else "general_health"
    vectorstore_path = Path(f"vectorstore/{domain}")

    if not vectorstore_path.exists():
        print(f"❌ Vectorstore directory not found: {vectorstore_path}")
        sys.exit(1)

    # Calculate total size
    total_size = sum(f.stat().st_size for f in vectorstore_path.rglob('*') if f.is_file())
    file_count = len([f for f in vectorstore_path.rglob('*') if f.is_file()])

    size_mb = total_size / (1024 ** 2)
    size_gb = total_size / (1024 ** 3)

    print(f"\n📊 Vectorstore Size & Stats: {domain}")
    print("=" * 60)
    print(f"📁 Total Size: {size_mb:.2f} MB ({size_gb:.2f} GB)")
    print(f"📄 Files: {file_count}")

    # Get domain statistics
    stats = get_domain_stats(domain)
    if stats:
        print(f"\n📊 Statistics:")
        print(f"  Chunks: {stats.get('chunk_count', 0):,}")
        print(f"  Documents: {stats.get('document_count', 0):,}")
        print(f"  Embedding Model: {stats.get('embedding_model', 'Unknown')}")
        print(f"  Chunk Size: {stats.get('chunk_size', 0)} chars")
        print(f"  Chunk Overlap: {stats.get('chunk_overlap', 0)} chars")
    
    # Show largest files
    print(f"\n📋 Largest files:")
    files = [(f, f.stat().st_size) for f in vectorstore_path.rglob('*') if f.is_file()]
    files.sort(key=lambda x: x[1], reverse=True)

    for file_path, file_size in files[:5]:
        size_mb_file = file_size / (1024 ** 2)
        print(f"   {file_path.name}: {size_mb_file:.2f} MB")

