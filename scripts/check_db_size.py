#!/usr/bin/env python3
"""Check SQLite database size and integrity for a domain."""

import sqlite3
import sys
from pathlib import Path

def check_database(domain: str):
    """Check SQLite database for a domain."""
    db_path = Path("vectorstore") / domain / "chroma.sqlite3"
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"Database file size: {size_mb:.2f} MB")
    print(f"Database path: {db_path}")
    print("-" * 60)
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables found: {tables}")
        print("-" * 60)
        
        # Check each table
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table}: {count:,} rows")
                
                # Get table size info
                cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table}'")
                if count > 0:
                    # Try to get page count (SQLite internal)
                    cursor.execute("SELECT page_count FROM pragma_page_count();")
                    try:
                        page_count = cursor.fetchone()[0]
                        print(f"  (SQLite pages: {page_count:,})")
                    except:
                        pass
            except Exception as e:
                print(f"{table}: Error - {e}")
        
        print("-" * 60)
        
        # Check for WAL file (Write-Ahead Logging)
        wal_path = db_path.with_suffix('.sqlite3-wal')
        if wal_path.exists():
            wal_size_mb = wal_path.stat().st_size / (1024 * 1024)
            print(f"WAL file exists: {wal_size_mb:.2f} MB")
            print("WARNING: WAL file should be merged into main database!")
        
        # Check database integrity
        print("-" * 60)
        cursor.execute("PRAGMA integrity_check;")
        integrity = cursor.fetchone()[0]
        if integrity == "ok":
            print("Database integrity: OK")
        else:
            print(f"Database integrity: {integrity}")
        
        # Check page size and other info
        cursor.execute("PRAGMA page_size;")
        page_size = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_count;")
        page_count = cursor.fetchone()[0]
        total_size = page_size * page_count / (1024 * 1024)
        
        print(f"Page size: {page_size} bytes")
        print(f"Page count: {page_count:,}")
        print(f"Expected size: {total_size:.2f} MB")
        
        # Vacuum info
        print("-" * 60)
        print("RECOMMENDATION: Run VACUUM to reclaim space")
        print("This will rebuild the database and remove fragmentation")
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "legal_business"
    check_database(domain)

