"""
Domain-specific vector retrievers for Meetara Core RAG system.
"""
import os
import shutil
import sqlite3
import threading
import hashlib
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings, get_domain_path
from app.core.logger import rag_logger

# Try to import huggingface_hub for cache checking
try:
    from huggingface_hub import try_to_load_from_cache, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


# Global shared embeddings model (loaded once, shared across all domains)
_shared_embeddings = None
_embeddings_lock = threading.Lock()

# Global embedding cache for frequently accessed documents
# LRU cache: (query_hash, domain) -> List[Document]
_embedding_cache: OrderedDict = OrderedDict()
_cache_max_size = 100  # Cache up to 100 query results
_cache_lock = threading.Lock()

# Embedding model metadata
EMBEDDING_MODEL_SIZE_MB = 80  # ~80 MB for all-MiniLM-L6-v2


def _check_model_cache(model_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if the embedding model is cached locally.
    
    Returns:
        (is_cached, cache_path): Tuple indicating if model is cached and cache path if available
    """
    if not HF_HUB_AVAILABLE:
        return False, None
    
    try:
        # Check for model files in cache
        # The model typically has these files: config.json, pytorch_model.bin, tokenizer files
        cache_path = try_to_load_from_cache(
            repo_id=model_name,
            filename="pytorch_model.bin",
            cache_dir=None  # Use default cache
        )
        
        if cache_path and os.path.exists(cache_path):
            # Model is cached
            cache_dir = os.path.dirname(os.path.dirname(cache_path))  # Go up to model directory
            return True, cache_dir
        
        # Also check for safetensors format (alternative format)
        cache_path = try_to_load_from_cache(
            repo_id=model_name,
            filename="model.safetensors",
            cache_dir=None
        )
        
        if cache_path and os.path.exists(cache_path):
            cache_dir = os.path.dirname(os.path.dirname(cache_path))
            return True, cache_dir
            
    except Exception as e:
        rag_logger.debug(f"Could not check cache for {model_name}: {e}")
    
    return False, None


def _get_shared_embeddings():
    """Get or create shared HuggingFace embeddings model (thread-safe, shared across all domains).
    
    Checks cache first and only downloads if model is not cached.
    Model size: ~80 MB (all-MiniLM-L6-v2)
    """
    global _shared_embeddings
    if _shared_embeddings is None:
        with _embeddings_lock:
            if _shared_embeddings is None:
                model_name = settings.embedding_model
                
                # Check if model is cached
                is_cached, cache_path = _check_model_cache(model_name)
                
                if is_cached:
                    rag_logger.info(
                        f"✅ Using cached embeddings model: {model_name} "
                        f"(~{EMBEDDING_MODEL_SIZE_MB} MB) from cache"
                    )
                else:
                    rag_logger.info(
                        f"📥 Downloading embeddings model: {model_name} "
                        f"(~{EMBEDDING_MODEL_SIZE_MB} MB) from Hugging Face Hub..."
                    )
                    rag_logger.info(
                        f"   Note: This is a one-time download. Model will be cached for future use."
                    )
                
                # Initialize embeddings (will use cache if available, download if not)
                _shared_embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'}
                )
                
                if is_cached:
                    rag_logger.info(f"✅ Loaded cached embeddings model: {model_name}")
                else:
                    rag_logger.info(f"✅ Downloaded and initialized embeddings model: {model_name}")
                    rag_logger.info(f"   Model cached for future use (no download needed on next startup)")
    return _shared_embeddings


class DomainRetriever:
    """Domain-specific vector retriever with ChromaDB backend."""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.domain_path = get_domain_path(domain)
        # Use shared embeddings model to avoid concurrent loading conflicts
        self.embeddings = _get_shared_embeddings()
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load the vector store for this domain with optimized settings."""
        # Optimized ChromaDB settings for better performance
        optimized_metadata = {
            "hnsw:space": "cosine",  # Cosine similarity (best for embeddings)
            "hnsw:M": "16",  # Number of connections (higher = better accuracy, slower)
            "hnsw:ef_construction": "200",  # EF construction (higher = better quality)
            "hnsw:ef_search": "50",  # EF search (higher = better recall, slower)
        }

        def _create_new_store():
            self.domain_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma(
                persist_directory=str(self.domain_path),
                embedding_function=self.embeddings,
                collection_metadata=optimized_metadata
            )
            rag_logger.info(f"Created new optimized vector store for domain: {self.domain}")

        try:
            if self.domain_path.exists() and any(self.domain_path.iterdir()):
                # Load existing vector store (don't pass metadata to avoid conflicts)
                try:
                    self.vectorstore = Chroma(
                        persist_directory=str(self.domain_path),
                        embedding_function=self.embeddings
                    )
                    rag_logger.info(f"Loaded existing vector store for domain: {self.domain}")
                except Exception as load_err:
                    err_msg = str(load_err).lower()
                    # Incompatible HNSW segment metadata (e.g. after Chroma upgrade or corrupted index)
                    if "hnsw" in err_msg and ("segment" in err_msg or "parse" in err_msg or "metadata" in err_msg):
                        rag_logger.warning(
                            f"Vector store for domain '{self.domain}' has incompatible HNSW metadata "
                            f"(e.g. ChromaDB version change). Recreating empty store; re-upload documents to repopulate."
                        )
                        shutil.rmtree(self.domain_path, ignore_errors=True)
                        _create_new_store()
                    else:
                        raise
            else:
                _create_new_store()

            # Apply SQLite optimizations after ChromaDB initialization
            self._optimize_sqlite()

        except Exception as e:
            rag_logger.error(f"Failed to initialize vector store for domain {self.domain}: {e}")
            raise
    
    def _optimize_sqlite(self):
        """Apply SQLite optimizations for better performance."""
        try:
            db_path = self.domain_path / "chroma.sqlite3"
            if not db_path.exists():
                return
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Apply optimizations
            cursor.execute("PRAGMA journal_mode = WAL;")  # Write-Ahead Logging for better concurrency
            cursor.execute("PRAGMA synchronous = NORMAL;")  # Faster writes (safe, WAL makes it safe)
            cursor.execute("PRAGMA cache_size = -64000;")  # 64MB cache (negative = KB)
            cursor.execute("PRAGMA temp_store = MEMORY;")  # Temp tables in memory
            cursor.execute("PRAGMA mmap_size = 268435456;")  # 256MB memory-mapped I/O
            cursor.execute("PRAGMA optimize;")  # Optimize for queries
            
            conn.commit()
            conn.close()
            
            rag_logger.debug(f"Applied SQLite optimizations for domain: {self.domain}")
            
        except Exception as e:
            # Don't fail if SQLite optimization fails - it's just an optimization
            rag_logger.debug(f"Could not apply SQLite optimizations for {self.domain}: {e}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the domain's vector store with optimized memory and batching for large document sets."""
        try:
            if not documents:
                rag_logger.warning(f"No documents to add for domain: {self.domain}")
                return
            
            total_docs = len(documents)
            rag_logger.info(f"Processing {total_docs} documents for domain: {self.domain}")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=len,
            )
            
            # Memory-optimized batch processing
            # Smaller batches for large document sets to reduce memory usage
            DOC_BATCH_SIZE = 50  # Process 50 documents at a time (reduced from 100)
            CHUNK_BATCH_SIZE = 500  # Add chunks in batches of 500 to ChromaDB
            
            total_chunks_added = 0
            total_batches = (total_docs + DOC_BATCH_SIZE - 1) // DOC_BATCH_SIZE
            
            for batch_start in range(0, total_docs, DOC_BATCH_SIZE):
                batch_end = min(batch_start + DOC_BATCH_SIZE, total_docs)
                batch_docs = documents[batch_start:batch_end]
                batch_num = batch_start // DOC_BATCH_SIZE + 1
                
                rag_logger.info(f"Processing batch {batch_num}/{total_batches}: documents {batch_start+1}-{batch_end} of {total_docs}")
                
                # Split this batch into chunks (memory-efficient: process and discard)
                split_docs = text_splitter.split_documents(batch_docs)
                
                # Clear original batch from memory
                del batch_docs
                
                # Filter out empty chunks (chunks with no content or only whitespace)
                filtered_docs = []
                for doc in split_docs:
                    content = doc.page_content.strip() if hasattr(doc, 'page_content') else str(doc).strip()
                    if len(content) >= 10:  # Minimum 10 characters to avoid empty/whitespace-only chunks
                        filtered_docs.append(doc)
                
                # Clear split_docs after filtering
                del split_docs
                
                if not filtered_docs:
                    rag_logger.debug(f"Batch {batch_num}: All chunks were empty, skipping")
                    # Force garbage collection for empty batches
                    gc.collect()
                    continue
                
                # Add chunks to ChromaDB in smaller sub-batches for memory optimization
                # This prevents memory spikes when adding large numbers of embeddings
                for chunk_batch_start in range(0, len(filtered_docs), CHUNK_BATCH_SIZE):
                    chunk_batch_end = min(chunk_batch_start + CHUNK_BATCH_SIZE, len(filtered_docs))
                    chunk_batch = filtered_docs[chunk_batch_start:chunk_batch_end]
                    
                    # Add this chunk batch to vector store
                    # ChromaDB will handle embeddings internally (memory-efficient)
                    self.vectorstore.add_documents(chunk_batch)
                    
                    # Clear chunk batch from memory
                    del chunk_batch
                
                batch_chunks = len(filtered_docs)
                total_chunks_added += batch_chunks
                
                # Clear filtered_docs from memory before next batch
                del filtered_docs
                
                # Periodic garbage collection every 5 batches to free memory
                if batch_num % 5 == 0:
                    gc.collect()
                    rag_logger.debug(f"Batch {batch_num}: Garbage collection performed")
                
                rag_logger.info(f"Batch {batch_num} complete: Added {batch_chunks} chunks (total: {total_chunks_added})")
            
            # Final garbage collection
            gc.collect()
            
            # Note: Newer versions of langchain-chroma don't have persist() method
            # The data is automatically persisted to disk
            
            # Get vector DB size and log statistics
            db_size = self._get_vector_db_size()
            stats = self.get_collection_stats()
            
            rag_logger.info(f"✅ Completed: Added {total_chunks_added} total document chunks to domain: {self.domain}")
            rag_logger.info(f"📊 Vector DB Statistics for domain '{self.domain}':")
            rag_logger.info(f"   - Total chunks in database: {stats.get('chunk_count', 0):,}")
            rag_logger.info(f"   - Total documents: {stats.get('document_count', 0):,}")
            rag_logger.info(f"   - Database size: {db_size}")
            rag_logger.info(f"   - Average chunks per document: {total_chunks_added // max(1, len(documents)):.1f}")
            
        except Exception as e:
            rag_logger.error(f"Failed to add documents to domain {self.domain}: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[Document]:
        """Perform similarity search in the domain's vector store with improved relevance and caching."""
        try:
            if not self.vectorstore:
                rag_logger.warning(f"Vector store not initialized for domain: {self.domain}")
                return []
            
            # Check cache first (if enabled)
            if use_cache:
                cache_key = self._get_cache_key(query, k, filter_dict)
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    rag_logger.debug(f"Cache hit for query in domain: {self.domain}")
                    return cached_result
            
            # Enhance query for medical topics
            query_lower = query.lower()
            if "symptoms" in query_lower and "cancer" in query_lower:
                enhanced_query = f"{query} common signs symptoms indicators cancer tumor malignancy"
            elif "treatment" in query_lower and "cancer" in query_lower:
                enhanced_query = f"{query} cancer treatment therapy medication management"
            else:
                enhanced_query = query
            
            # Get more results initially for filtering
            initial_k = min(k * 2, 10)  # Get more results but cap at 10
            
            # Use similarity_search_with_score for better filtering
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query=enhanced_query,
                k=initial_k,
                filter=filter_dict
            )
            
            # Filter by similarity score threshold (lower = more similar for cosine)
            # Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
            score_threshold = 1.5  # Reject documents with cosine distance > 1.5
            results = [doc for doc, score in results_with_scores if score < score_threshold]
            
            # Enhanced relevance filtering
            filtered_results = []
            query_lower = query.lower()
            query_terms = [term.strip() for term in query_lower.split() if len(term.strip()) > 2]
            
            for doc in results:
                content = doc.page_content.lower()
                
                # Calculate relevance score based on multiple factors
                relevance_score = 0
                
                # 1. Exact term matches (highest weight)
                for term in query_terms:
                    if term in content:
                        relevance_score += 3
                
                # 2. Phrase matches (medium weight)
                if any(phrase in content for phrase in query_lower.split()):
                    relevance_score += 2
                
                # 3. Semantic similarity (lower weight)
                if any(word in content for word in query_terms):
                    relevance_score += 1
                
                # 4. Medical context bonus
                medical_terms = ["symptom", "treatment", "diagnosis", "cause", "prevention", "risk"]
                if any(term in query_lower for term in medical_terms) and any(term in content for term in medical_terms):
                    relevance_score += 2
                
                # Only include documents with meaningful relevance
                if relevance_score >= 2:
                    filtered_results.append((doc, relevance_score))
                
                if len(filtered_results) >= k * 2:  # Get more candidates for better filtering
                    break
            
            # Sort by relevance score and take top k
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            final_results = [doc for doc, score in filtered_results[:k]]
            
            # Cache the result
            if use_cache:
                self._add_to_cache(cache_key, final_results)
            
            rag_logger.info(f"Retrieved {len(final_results)} relevant documents for query in domain: {self.domain}")
            return final_results
            
        except Exception as e:
            rag_logger.error(f"Failed to perform similarity search in domain {self.domain}: {e}")
            return []
    
    def batch_similarity_search(
        self,
        queries: List[str],
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Document]]:
        """Batch multiple similarity searches for better performance.
        
        Args:
            queries: List of query strings
            k: Number of results per query
            filter_dict: Optional metadata filter (applied to all queries)
            
        Returns:
            Dictionary mapping each query to its results
        """
        try:
            if not self.vectorstore:
                rag_logger.warning(f"Vector store not initialized for domain: {self.domain}")
                return {query: [] for query in queries}
            
            results = {}
            
            # Process queries in batches for better performance
            BATCH_SIZE = 10  # Process 10 queries at a time
            
            for i in range(0, len(queries), BATCH_SIZE):
                batch_queries = queries[i:i + BATCH_SIZE]
                
                # Process batch (ChromaDB handles batching internally, but we optimize here)
                for query in batch_queries:
                    # Use individual search but with caching enabled
                    results[query] = self.similarity_search(
                        query=query,
                        k=k,
                        filter_dict=filter_dict,
                        use_cache=True
                    )
            
            rag_logger.info(f"Batch processed {len(queries)} queries for domain: {self.domain}")
            return results
            
        except Exception as e:
            rag_logger.error(f"Failed to perform batch similarity search in domain {self.domain}: {e}")
            return {query: [] for query in queries}
    
    def _get_cache_key(self, query: str, k: int, filter_dict: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query."""
        filter_str = str(sorted(filter_dict.items())) if filter_dict else ""
        key_str = f"{self.domain}:{query}:{k}:{filter_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[Document]]:
        """Get result from cache if available."""
        with _cache_lock:
            if cache_key in _embedding_cache:
                # Move to end (most recently used)
                result = _embedding_cache.pop(cache_key)
                _embedding_cache[cache_key] = result
                return result
        return None
    
    def _add_to_cache(self, cache_key: str, results: List[Document]):
        """Add result to cache (LRU eviction if needed)."""
        with _cache_lock:
            # Remove if exists (to move to end)
            if cache_key in _embedding_cache:
                _embedding_cache.pop(cache_key)
            
            # Add to end
            _embedding_cache[cache_key] = results
            
            # Evict oldest if cache is full
            if len(_embedding_cache) > _cache_max_size:
                _embedding_cache.popitem(last=False)  # Remove oldest (first item)
    
    def _get_vector_db_size(self) -> str:
        """Calculate and return the size of the vector database directory in human-readable format."""
        try:
            if not self.domain_path.exists():
                return "0 B"
            
            total_size = 0
            file_count = 0
            
            # Walk through all files in the domain directory
            for dirpath, dirnames, filenames in os.walk(self.domain_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                        file_count += 1
                    except (OSError, FileNotFoundError):
                        # Skip files that can't be accessed
                        continue
            
            # Convert to human-readable format
            if total_size == 0:
                return "0 B"
            
            # Format with appropriate unit
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if total_size < 1024.0:
                    return f"{total_size:.2f} {unit} ({file_count} files)"
                total_size /= 1024.0
            
            return f"{total_size:.2f} PB ({file_count} files)"
            
        except Exception as e:
            rag_logger.warning(f"Failed to calculate vector DB size for {self.domain}: {e}")
            return "Unknown"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the domain's vector store."""
        try:
            if not self.vectorstore:
                return {
                    "domain": self.domain,
                    "chunk_count": 0,
                    "document_count": 0,
                    "count": 0,  # Legacy field for backward compatibility
                    "db_size": "0 B"
                }
            
            collection = self.vectorstore._collection
            chunk_count = collection.count()
            
            # Count unique documents by file_name in metadata
            # Get all metadata to find unique documents
            document_count = 0
            unique_files = set()
            
            try:
                # Efficiently count unique documents by file_name in metadata
                # For large collections, we'll get metadata in batches or use a sample
                # First, try to get all metadata (ChromaDB should handle this efficiently)
                
                # Use a reasonable limit first, then expand if needed
                max_sample_size = 100000  # Large enough for most collections
                
                all_results = collection.get(limit=max_sample_size)
                
                if all_results and "metadatas" in all_results:
                    metadatas = all_results["metadatas"]
                    
                    for metadata in metadatas:
                        if metadata:
                            # Try different metadata field names that might be used
                            file_name = (
                                metadata.get("file_name") or 
                                metadata.get("source") or
                                metadata.get("filename")
                            )
                            if file_name:
                                # Extract just the filename if it's a full path
                                file_name_str = str(file_name)
                                if "/" in file_name_str or "\\" in file_name_str:
                                    file_name_str = file_name_str.split("/")[-1].split("\\")[-1]
                                unique_files.add(file_name_str)
                    
                    document_count = len(unique_files)
                    
                    # If we hit the limit and didn't process all chunks, document count might be incomplete
                    if len(metadatas) >= max_sample_size and chunk_count > max_sample_size:
                        rag_logger.warning(
                            f"Sample limit reached for {self.domain}. "
                            f"Document count ({document_count}) may be incomplete. "
                            f"Total chunks: {chunk_count}"
                        )
                
                # If we couldn't get metadata or found no files, use chunk count as fallback
                # (but mark it as estimated)
                if document_count == 0 and chunk_count > 0:
                    # Rough estimate: assume average 50 chunks per document
                    document_count = max(1, chunk_count // 50)
                    rag_logger.warning(
                        f"Could not determine exact document count for {self.domain}, "
                        f"estimated {document_count} from {chunk_count} chunks"
                    )
            
            except Exception as e:
                rag_logger.warning(f"Error counting unique documents for {self.domain}: {e}")
                # Fallback: estimate from chunk count
                if chunk_count > 0:
                    document_count = max(1, chunk_count // 50)  # Rough estimate
                else:
                    document_count = 0
            
            # Get database size
            db_size = self._get_vector_db_size()
            
            return {
                "domain": self.domain,
                "chunk_count": chunk_count,
                "document_count": document_count,
                "count": document_count,  # Legacy field - now returns document count, not chunk count
                "db_size": db_size,
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            }
            
        except Exception as e:
            rag_logger.error(f"Failed to get stats for domain {self.domain}: {e}")
            return {
                "domain": self.domain,
                "chunk_count": 0,
                "document_count": 0,
                "count": 0,
                "db_size": "0 B",
                "error": str(e)
            }
    
    def has_document_with_hash(self, file_hash: str) -> bool:
        """Check if a document with this hash already exists in the vector store.
        
        Optimized with indexed metadata filtering for faster lookups.
        """
        try:
            if not self.vectorstore:
                return False
            
            # Use indexed metadata filtering (file_hash is commonly filtered)
            # ChromaDB automatically indexes metadata fields used in filters
            results = self.vectorstore.get(
                where={"file_hash": file_hash},  # Indexed field for fast lookup
                limit=1
            )
            
            return len(results.get("ids", [])) > 0
            
        except Exception as e:
            rag_logger.error(f"Failed to check for duplicate document in domain {self.domain}: {e}")
            return False
    
    def delete_documents(self, filter_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Delete documents from the vector store.
        
        Args:
            filter_dict: Dictionary with metadata filters (e.g., {"file_hash": "abc123"})
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            if not self.vectorstore:
                return False
            
            # Note: ChromaDB delete functionality may vary by version
            # This is a basic implementation
            if filter_dict:
                # Delete specific documents matching filter
                try:
                    # Get IDs first to count what will be deleted
                    results = self.vectorstore._collection.get(where=filter_dict)
                    ids_to_delete = results.get('ids', [])
                    count = len(ids_to_delete)
                    
                    if count > 0:
                        self.vectorstore._collection.delete(where=filter_dict)
                        rag_logger.info(f"Deleted {count} document chunk(s) from domain {self.domain} matching filter: {filter_dict}")
                        return True
                    else:
                        rag_logger.debug(f"No documents found matching filter: {filter_dict}")
                        return True  # Success (nothing to delete)
                except Exception as e:
                    rag_logger.warning(f"Error deleting documents with filter {filter_dict}: {e}")
                    # Try alternative: get all IDs matching filter, then delete by ID
                    try:
                        results = self.vectorstore._collection.get(where=filter_dict)
                        ids = results.get('ids', [])
                        if ids:
                            self.vectorstore._collection.delete(ids=ids)
                            rag_logger.info(f"Deleted {len(ids)} document chunk(s) from domain {self.domain} (by ID)")
                            return True
                    except Exception as e2:
                        rag_logger.error(f"Alternative delete method also failed: {e2}")
                        return False
            else:
                # Delete all documents: Get all IDs first, then delete them
                # ChromaDB requires at least one of ids, where, or where_document
                try:
                    # Get all document IDs
                    results = self.vectorstore._collection.get()
                    all_ids = results.get('ids', [])
                    
                    if all_ids:
                        # Delete all documents by IDs
                        self.vectorstore._collection.delete(ids=all_ids)
                        rag_logger.info(f"Deleted {len(all_ids)} documents from domain {self.domain}")
                    else:
                        rag_logger.info(f"No documents to delete in domain {self.domain}")
                except Exception as e:
                    rag_logger.warning(f"Could not get IDs for bulk delete: {e}. Trying alternative method...")
                    # Alternative: Use a where clause that matches everything (if supported)
                    # If that doesn't work, the error will be caught below
                    try:
                        # Try using where with empty dict (might work in some versions)
                        self.vectorstore._collection.delete(where={})
                    except:
                        raise e
            
            # Note: Newer versions of langchain-chroma don't have persist() method
            # The data is automatically persisted to disk
            rag_logger.info(f"Successfully deleted documents from domain: {self.domain}")
            return True
            
        except Exception as e:
            rag_logger.error(f"Failed to delete documents from domain {self.domain}: {e}")
            return False


# Global retriever cache
_retriever_cache: Dict[str, DomainRetriever] = {}
_cache_lock = threading.Lock()  # Thread-safe lock for cache access


def get_domain_retriever(domain: str) -> DomainRetriever:
    """Get or create a domain retriever instance (thread-safe)."""
    # Double-checked locking pattern for thread safety
    if domain not in _retriever_cache:
        with _cache_lock:
            # Check again after acquiring lock (another thread might have created it)
            if domain not in _retriever_cache:
                try:
                    _retriever_cache[domain] = DomainRetriever(domain)
                except Exception as e:
                    rag_logger.error(f"Failed to create retriever for domain {domain}: {e}")
                    raise
    return _retriever_cache[domain]


def list_available_domains() -> List[str]:
    """List all available domains with vector stores."""
    try:
        domains = []
        for item in settings.vectorstore_path.iterdir():
            if item.is_dir() and any(item.iterdir()):
                domains.append(item.name)
        return sorted(domains)
    except Exception as e:
        rag_logger.error(f"Failed to list domains: {e}")
        return []


def create_domain(domain: str) -> bool:
    """Create a new domain vector store."""
    try:
        retriever = get_domain_retriever(domain)
        rag_logger.info(f"Created new domain: {domain}")
        return True
    except Exception as e:
        rag_logger.error(f"Failed to create domain {domain}: {e}")
        return False


def delete_domain(domain: str) -> bool:
    """Delete a domain and its vector store."""
    try:
        domain_path = get_domain_path(domain)
        if domain_path.exists():
            import shutil
            shutil.rmtree(domain_path)
        
        # Remove from cache
        if domain in _retriever_cache:
            del _retriever_cache[domain]
        
        rag_logger.info(f"Deleted domain: {domain}")
        return True
        
    except Exception as e:
        rag_logger.error(f"Failed to delete domain {domain}: {e}")
        return False


def prewarm_retrievers(domains: Optional[List[str]] = None, max_domains: int = 5) -> Dict[str, bool]:
    """
    Pre-warm domain retrievers for faster first queries.
    
    This loads the embedding model and initializes vector stores for specified domains
    in advance, reducing cold-start latency for the first user query.
    
    Args:
        domains: List of domain names to pre-warm. If None, uses most common domains.
        max_domains: Maximum number of domains to pre-warm (to limit memory usage).
    
    Returns:
        Dict mapping domain name to success status.
    """
    results = {}
    
    # If no domains specified, use the most common ones
    if domains is None:
        available = list_available_domains()
        # Prioritize commonly used domains
        priority_domains = ['general_health', 'mental_health', 'education', 'general']
        domains = [d for d in priority_domains if d in available]
        # Add remaining domains up to max
        remaining = [d for d in available if d not in domains]
        domains.extend(remaining[:max_domains - len(domains)])
    
    domains = domains[:max_domains]
    
    if not domains:
        rag_logger.info("🔥 No domains to pre-warm")
        return results
    
    rag_logger.info(f"🔥 Pre-warming {len(domains)} domain retrievers: {domains}")
    
    # First, ensure embedding model is loaded (shared across all domains)
    try:
        _get_shared_embeddings()
        rag_logger.info("✅ Embedding model pre-warmed")
    except Exception as e:
        rag_logger.error(f"❌ Failed to pre-warm embedding model: {e}")
        return {d: False for d in domains}
    
    # Pre-warm each domain retriever
    for domain in domains:
        try:
            get_domain_retriever(domain)
            results[domain] = True
            rag_logger.info(f"✅ Pre-warmed retriever for: {domain}")
        except Exception as e:
            results[domain] = False
            rag_logger.warning(f"⚠️ Failed to pre-warm {domain}: {e}")
    
    success_count = sum(1 for v in results.values() if v)
    rag_logger.info(f"🔥 Pre-warming complete: {success_count}/{len(domains)} domains ready")
    
    return results