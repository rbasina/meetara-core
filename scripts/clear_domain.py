#!/usr/bin/env python3
"""
Clear Domain Script for Meetara Core
Clears all documents from a specific domain in the vectorstore
Useful before re-uploading documents with enhanced features (e.g., page numbers)
"""

import os
import sys
from pathlib import Path

# Determine project root and ensure imports work regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

# Ensure project root is on sys.path for absolute imports
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.domain_retrievers import get_domain_retriever, delete_domain, list_available_domains
from app.core.config import settings
from app.core.logger import rag_logger


def clear_domain(domain: str, confirm: bool = False, delete_database: bool = False):
    """Clear all documents from a domain's vectorstore.
    
    Args:
        domain: Domain name to clear
        confirm: If True, skip confirmation prompt
        delete_database: If True, also delete the SQLite database file (complete cleanup)
    """
    if not confirm:
        print(f"WARNING: This will delete ALL documents from domain '{domain}'")
        if delete_database:
            print(f"   WARNING: This will also DELETE the database file (chroma.sqlite3)")
        response = input(f"   Are you sure? Type 'YES' to confirm: ")
        if response != 'YES':
            print("Operation cancelled")
            return False
    
    try:
        retriever = get_domain_retriever(domain)
        
        # Get stats before deletion
        stats = retriever.get_collection_stats()
        count_before = stats.get('count', 0)
        
        print(f"Domain '{domain}' currently has {count_before} document chunks")
        
        # Delete all documents from the domain
        success = retriever.delete_documents()
        
        if success:
            print(f"Successfully cleared domain '{domain}'")
            print(f"   Removed {count_before} document chunks")
            
            # Verify deletion
            stats_after = retriever.get_collection_stats()
            count_after = stats_after.get('count', 0)
            
            if count_after == 0:
                print(f"   Verified: Domain is now empty ({count_after} chunks)")
            else:
                print(f"   Warning: Domain still has {count_after} chunks (may need manual cleanup)")
            
            # If requested, also delete the SQLite database file and directory
            if delete_database:
                # First, close the vectorstore connection and clear cache
                try:
                    from app.rag.domain_retrievers import _retriever_cache, _cache_lock
                    with _cache_lock:
                        if domain in _retriever_cache:
                            retriever_obj = _retriever_cache[domain]
                            # Try to close/cleanup the vectorstore
                            try:
                                if hasattr(retriever_obj, 'vectorstore') and retriever_obj.vectorstore:
                                    # Close the ChromaDB client if possible
                                    if hasattr(retriever_obj.vectorstore, '_client'):
                                        try:
                                            retriever_obj.vectorstore._client = None
                                        except:
                                            pass
                                    retriever_obj.vectorstore = None
                            except:
                                pass
                            # Remove from cache
                            del _retriever_cache[domain]
                except Exception as e:
                    rag_logger.debug(f"Could not clear cache for {domain}: {e}")
                
                # Force garbage collection to release file handles
                import gc
                gc.collect()
                
                # Small delay to allow file handles to be released
                import time
                time.sleep(0.5)
                
                # Now try to delete the directory
                import shutil
                domain_path = settings.vectorstore_path / domain
                if domain_path.exists():
                    try:
                        # Remove the entire domain directory (including chroma.sqlite3 and UUID folders)
                        shutil.rmtree(domain_path)
                        print(f"   Deleted database file and directory: {domain_path}")
                    except PermissionError as e:
                        print(f"   Warning: Files still locked. Try closing any ChromaDB connections or restart Python.")
                        print(f"   Tip: You may need to restart your Python session before deleting database files.")
                        rag_logger.warning(f"Could not delete database directory for {domain}: {e}")
                    except Exception as e:
                        print(f"   Warning: Could not delete database directory: {e}")
                        rag_logger.warning(f"Could not delete database directory for {domain}: {e}")
            
            return True
        else:
            print(f"Failed to clear domain '{domain}'")
            return False
            
    except Exception as e:
        print(f"Error clearing domain '{domain}': {e}")
        rag_logger.error(f"Clear domain failed: {e}")
        return False


def clear_all_domains(confirm: bool = False, delete_database: bool = False):
    """Clear all domains in the vectorstore.
    
    Args:
        confirm: If True, skip confirmation prompt
        delete_database: If True, also delete the SQLite database files (complete cleanup)
    
    Returns:
        Dictionary mapping domain names to success status
    """
    domains = list_available_domains()
    
    if not domains:
        print("No domains found to clear")
        return {}
    
    if not confirm:
        print(f"WARNING: This will delete ALL documents from {len(domains)} domain(s)!")
        if delete_database:
            print(f"   WARNING: This will also DELETE all database files (chroma.sqlite3)")
        print(f"   Domains to clear: {', '.join(domains)}")
        response = input("\n   Are you sure? Type 'YES' to confirm: ")
        if response != 'YES':
            print("Operation cancelled")
            return {}
    
    results = {}
    print(f"\nClearing {len(domains)} domain(s)...")
    
    for domain in domains:
        try:
            retriever = get_domain_retriever(domain)
            stats_before = retriever.get_collection_stats()
            count_before = stats_before.get('count', 0)
            
            success = retriever.delete_documents()
            
            if success:
                print(f"  Cleared '{domain}' ({count_before} chunks removed)")
                
                # If requested, also delete the SQLite database file
                if delete_database:
                    # First, close the vectorstore connection and clear cache
                    try:
                        from app.rag.domain_retrievers import _retriever_cache, _cache_lock
                        with _cache_lock:
                            if domain in _retriever_cache:
                                retriever = _retriever_cache[domain]
                                # Try to close/cleanup the vectorstore
                                try:
                                    if hasattr(retriever, 'vectorstore') and retriever.vectorstore:
                                        # Close the ChromaDB client if possible
                                        if hasattr(retriever.vectorstore, '_client'):
                                            try:
                                                retriever.vectorstore._client = None
                                            except:
                                                pass
                                        retriever.vectorstore = None
                                except:
                                    pass
                                # Remove from cache
                                del _retriever_cache[domain]
                    except Exception as e:
                        rag_logger.debug(f"Could not clear cache for {domain}: {e}")
                    
                    # Force garbage collection to release file handles
                    import gc
                    gc.collect()
                    
                    # Small delay to allow file handles to be released
                    import time
                    time.sleep(0.5)
                    
                    # Now try to delete the directory
                    import shutil
                    domain_path = settings.vectorstore_path / domain
                    if domain_path.exists():
                        try:
                            shutil.rmtree(domain_path)
                            print(f"      Deleted database directory")
                        except PermissionError as e:
                            print(f"      Warning: Files still locked. Try closing any ChromaDB connections or restart Python.")
                            print(f"      Tip: You may need to restart your Python session before deleting database files.")
                        except Exception as e:
                            print(f"      Warning: Could not delete database: {e}")
                
                results[domain] = True
            else:
                print(f"  Failed to clear '{domain}'")
                results[domain] = False
                
        except Exception as e:
            print(f"  Error clearing '{domain}': {e}")
            rag_logger.error(f"Clear domain failed for {domain}: {e}")
            results[domain] = False
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Clear All Summary:")
    print(f"{'='*60}")
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    print(f"  Successfully cleared: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}\n")
    
    return results


def list_domains_with_stats():
    """List all domains and their document counts with database sizes."""
    vectorstore_path = settings.vectorstore_path
    
    if not vectorstore_path.exists():
        print("Vectorstore directory not found")
        return
    
    domains = []
    total_size_bytes = 0
    for item in vectorstore_path.iterdir():
        if item.is_dir() and item.name not in ['__pycache__', '.git']:
            try:
                retriever = get_domain_retriever(item.name)
                stats = retriever.get_collection_stats()
                count = stats.get('count', 0)
                db_size = stats.get('db_size', '0 B')
                
                # Extract size in bytes for total calculation
                # Format is like "12.67 MB (5 files)" or "0 B" or "Unknown"
                size_bytes = 0
                if db_size and str(db_size) != 'Unknown' and str(db_size) != 'error':
                    size_str = str(db_size).split('(')[0].strip()  # Remove "(X files)" part
                    try:
                        if 'MB' in size_str:
                            size_bytes = float(size_str.replace('MB', '').strip()) * 1024 * 1024
                        elif 'KB' in size_str:
                            size_bytes = float(size_str.replace('KB', '').strip()) * 1024
                        elif 'GB' in size_str:
                            size_bytes = float(size_str.replace('GB', '').strip()) * 1024 * 1024 * 1024
                        elif 'TB' in size_str:
                            size_bytes = float(size_str.replace('TB', '').strip()) * 1024 * 1024 * 1024 * 1024
                        elif 'B' in size_str and 'MB' not in size_str and 'KB' not in size_str and 'GB' not in size_str:
                            size_bytes = float(size_str.replace('B', '').strip())
                        else:
                            # Try to parse as number
                            size_bytes = float(size_str)
                    except (ValueError, AttributeError):
                        size_bytes = 0
                
                total_size_bytes += size_bytes
                
                domains.append({
                    'name': item.name,
                    'count': count,
                    'size': db_size
                })
            except Exception as e:
                domains.append({
                    'name': item.name,
                    'count': 'error',
                    'size': 'error'
                })
    
    domains.sort(key=lambda x: x['name'])
    
    # Format total size
    if total_size_bytes < 1024:
        total_size_str = f"{total_size_bytes:.2f} B"
    elif total_size_bytes < 1024 * 1024:
        total_size_str = f"{total_size_bytes / 1024:.2f} KB"
    else:
        total_size_str = f"{total_size_bytes / (1024 * 1024):.2f} MB"
    
    print(f"\nAvailable Domains ({len(domains)}):")
    print(f"{'='*80}")
    print(f"{'Domain':<35} {'Chunks':>10} {'Database Size':>20}")
    print(f"{'-'*80}")
    for domain in domains:
        print(f"  {domain['name']:<33} {str(domain['count']):>10} {str(domain['size']):>20}")
    print(f"{'-'*80}")
    print(f"{'TOTAL':<33} {'':>10} {total_size_str:>20}")
    
    return domains


def main():
    """Main script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clear documents from a domain in vectorstore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all domains
  python scripts/clear_domain.py list
  
  # Clear specific domain (with confirmation)
  python scripts/clear_domain.py clear academic_tutoring
  
  # Clear all domains (with confirmation)
  python scripts/clear_domain.py clear all
  
  # Clear without confirmation prompt
  python scripts/clear_domain.py clear all --yes
  
  # Clear and delete database files (complete cleanup)
  python scripts/clear_domain.py clear all --yes --delete-db
  
  # Clear specific domain and delete its database
  python scripts/clear_domain.py clear academic_tutoring --delete-db
        """
    )
    
    parser.add_argument(
        "action",
        choices=["clear", "list"],
        help="Action to perform: 'clear' a domain or 'list' all domains"
    )
    parser.add_argument(
        "domain",
        nargs="?",
        help="Domain name to clear (required for 'clear' action)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--delete-db",
        action="store_true",
        help="Also delete the SQLite database files (complete cleanup - removes chroma.sqlite3)"
    )
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_domains_with_stats()
    elif args.action == "clear":
        if not args.domain:
            print("Please specify a domain name or 'all'")
            print("\nAvailable domains:")
            list_domains_with_stats()
        elif args.domain.lower() == "all":
            clear_all_domains(confirm=args.yes, delete_database=args.delete_db)
        else:
            clear_domain(args.domain, confirm=args.yes, delete_database=args.delete_db)


if __name__ == "__main__":
    main()

