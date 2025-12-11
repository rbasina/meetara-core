#!/usr/bin/env python3
"""VACUUM a domain's SQLite database to reclaim space."""
import sqlite3
from pathlib import Path
import sys

domain = sys.argv[1] if len(sys.argv) > 1 else "general_health"
db_path = Path(f"vectorstore/{domain}/chroma.sqlite3")

if not db_path.exists():
    print(f"❌ Database not found: {db_path}")
    sys.exit(1)

# Get size before
size_before = db_path.stat().st_size
print(f"\n📊 VACUUM Database: {domain}")
print(f"   Size before: {size_before/(1024**3):.2f} GB ({size_before:,} bytes)")

try:
    # Connect and VACUUM
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check row count
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    count = cursor.fetchone()[0]
    print(f"   Embeddings: {count:,}")
    
    print(f"\n   Running VACUUM (this may take a while)...")
    conn.execute("VACUUM")
    conn.close()
    
    # Get size after
    size_after = db_path.stat().st_size
    saved = size_before - size_after
    
    print(f"\n✅ VACUUM Complete!")
    print(f"   Size after:  {size_after/(1024**3):.2f} GB ({size_after:,} bytes)")
    print(f"   Space saved: {saved/(1024**3):.2f} GB ({saved:,} bytes)")
    print(f"   Reduction:   {(saved/size_before*100):.1f}%")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"   Make sure no processes are using the database")
    sys.exit(1)

