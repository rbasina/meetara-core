"""
Vectorstore API endpoints for Meetara Core.

This module provides endpoints for managing and publishing vectorstores
to Hugging Face Hub.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from app.core.logger import api_logger
from app.rag.domain_retrievers import list_available_domains, get_domain_retriever
from app.rag.hf_publisher import VectorstorePublisher, load_vectorstore_from_hub

router = APIRouter(prefix="/vectorstore", tags=["vectorstore"])


class PublishRequest(BaseModel):
    """Request model for publishing vectorstore."""
    domain: str = Field(..., description="Domain name to publish")
    repo_id: str = Field(..., description="Repository ID (e.g., 'meetara-lab/vectorstore-general_health')")
    private: bool = Field(False, description="Whether to create a private repository")
    create_pr: bool = Field(False, description="Whether to create a pull request instead of direct commit")
    commit_message: Optional[str] = Field(None, description="Custom commit message")


class PublishResponse(BaseModel):
    """Response model for publish endpoint."""
    success: bool
    message: str
    repo_url: Optional[str] = None
    domain: str
    stats: Optional[Dict[str, Any]] = None


class ExportRequest(BaseModel):
    """Request model for exporting vectorstore."""
    domain: str = Field(..., description="Domain name to export")
    include_embeddings: bool = Field(True, description="Include embedding vectors")
    include_documents: bool = Field(True, description="Include document text")
    include_metadata: bool = Field(True, description="Include metadata")


class LoadFromHubRequest(BaseModel):
    """Request model for loading vectorstore from HF Hub."""
    repo_id: str = Field(..., description="Repository ID to load from")
    domain: str = Field(..., description="Domain name for the loaded vectorstore")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use")


def get_hf_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract Hugging Face token from Authorization header, .env file, environment, or HF cache."""
    if authorization:
        # Support "Bearer <token>" or just "<token>"
        token = authorization.replace("Bearer ", "").strip()
        if token:
            return token
    
    # Try environment variable (includes .env file if loaded by Settings)
    import os
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token
    
    # Also try to load from .env file directly if not already loaded
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=False)  # Don't override existing env vars
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            if token:
                return token
    except ImportError:
        pass  # python-dotenv not available
    except Exception:
        pass  # Error loading .env
    
    # Try to get from HF cache (from huggingface-cli login)
    try:
        from huggingface_hub import HfFolder
        cached_token = HfFolder.get_token()
        if cached_token:
            return cached_token
    except Exception:
        pass
    
    # Return None - will let huggingface_hub use its default token loading
    return None


@router.post("/publish", response_model=PublishResponse)
async def publish_vectorstore(
    request: PublishRequest,
    token: Optional[str] = Depends(get_hf_token)
) -> PublishResponse:
    """Publish a domain's vectorstore to Hugging Face Hub.
    
    Token can be provided via:
    - Authorization header
    - HF_TOKEN environment variable  
    - Hugging Face login cache (from 'huggingface-cli login')
    """
    try:
        # Token can be None - will use cached token from huggingface-cli login
        if not token:
            api_logger.info(
                "No HF token in request. Will use cached token from 'huggingface-cli login' if available."
            )
        
        api_logger.info(f"Publishing domain '{request.domain}' to {request.repo_id}")
        
        # Get domain stats
        retriever = get_domain_retriever(request.domain)
        stats = retriever.get_collection_stats()
        
        if stats.get("chunk_count", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Domain '{request.domain}' has no documents to publish"
            )
        
        # Initialize publisher
        publisher = VectorstorePublisher(token=token)
        
        # Publish
        success, repo_url = publisher.publish_domain(
            domain=request.domain,
            repo_id=request.repo_id,
            private=request.private,
            create_pr=request.create_pr,
            commit_message=request.commit_message,
            push_to_hub=True
        )
        
        if success:
            return PublishResponse(
                success=True,
                message=f"Successfully published domain '{request.domain}' to Hugging Face Hub",
                repo_url=repo_url,
                domain=request.domain,
                stats=stats
            )
        else:
            # Get more details from the logger if available
            api_logger.error(f"Publishing failed for domain '{request.domain}' - check server logs for details")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to publish domain '{request.domain}'. Check server logs for detailed error message."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error publishing vectorstore: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_vectorstore(
    request: ExportRequest,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Export a domain's vectorstore to local dataset format (without publishing).
    
    Returns path to exported files.
    """
    try:
        from pathlib import Path
        import tempfile
        
        api_logger.info(f"Exporting domain '{request.domain}' to local dataset format")
        
        # Get domain stats
        retriever = get_domain_retriever(request.domain)
        stats = retriever.get_collection_stats()
        
        if stats.get("chunk_count", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Domain '{request.domain}' has no documents to export"
            )
        
        # Initialize publisher (no token needed for local export)
        publisher = VectorstorePublisher(token=None)
        
        # Determine output directory
        if output_dir:
            export_path = Path(output_dir)
        else:
            export_path = Path(tempfile.mkdtemp(prefix=f"meetara_export_{request.domain}_"))
        
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export
        dataset = publisher.export_domain_to_dataset(
            domain=request.domain,
            output_dir=export_path,
            include_embeddings=request.include_embeddings,
            include_documents=request.include_documents,
            include_metadata=request.include_metadata
        )
        
        return {
            "success": True,
            "message": f"Successfully exported domain '{request.domain}'",
            "domain": request.domain,
            "output_dir": str(export_path),
            "stats": stats,
            "dataset_available": dataset is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error exporting vectorstore: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-from-hub")
async def load_vectorstore_from_hub_endpoint(
    request: LoadFromHubRequest
) -> Dict[str, Any]:
    """Load a vectorstore dataset from Hugging Face Hub into local ChromaDB.
    
    This downloads and imports a published vectorstore.
    """
    try:
        api_logger.info(f"Loading vectorstore from HF Hub: {request.repo_id}")
        
        # Load from Hub
        vectorstore = load_vectorstore_from_hub(
            repo_id=request.repo_id,
            domain=request.domain,
            embedding_model=request.embedding_model
        )
        
        # Get stats
        retriever = get_domain_retriever(request.domain)
        stats = retriever.get_collection_stats()
        
        return {
            "success": True,
            "message": f"Successfully loaded vectorstore from {request.repo_id}",
            "domain": request.domain,
            "repo_id": request.repo_id,
            "stats": stats
        }
        
    except Exception as e:
        api_logger.error(f"Error loading vectorstore from Hub: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish-all")
async def publish_all_domains(
    repo_prefix: str = "meetara-lab/vectorstore",
    private: bool = False,
    domains: Optional[List[str]] = None,
    token: Optional[str] = Depends(get_hf_token)
) -> Dict[str, Any]:
    """Publish all available domains to Hugging Face Hub.
    
    This is a convenience endpoint to publish multiple domains at once.
    Token will be automatically used from login cache if not provided.
    """
    try:
        # Token can be None - will use cached token from huggingface-cli login
        if not token:
            api_logger.info(
                "No HF token in request. Will use cached token from 'huggingface-cli login' if available."
            )
        
        if domains is None:
            domains = list_available_domains()
        
        api_logger.info(f"Publishing {len(domains)} domains to HF Hub...")
        
        publisher = VectorstorePublisher(token=token)
        results = publisher.publish_all_domains(
            repo_prefix=repo_prefix,
            private=private,
            domains=domains
        )
        
        successful = [d for d, (s, _) in results.items() if s]
        failed = [d for d, (s, _) in results.items() if not s]
        
        return {
            "success": len(failed) == 0,
            "message": f"Published {len(successful)}/{len(domains)} domains",
            "results": {
                domain: {
                    "success": success,
                    "repo_url": repo_url
                }
                for domain, (success, repo_url) in results.items()
            },
            "successful": successful,
            "failed": failed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error publishing all domains: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains")
async def get_vectorstore_domains() -> Dict[str, Any]:
    """Get list of all available vectorstore domains with statistics."""
    try:
        domains = list_available_domains()
        
        domain_stats = []
        for domain in domains:
            retriever = get_domain_retriever(domain)
            stats = retriever.get_collection_stats()
            domain_stats.append({
                "domain": domain,
                "chunk_count": stats.get("chunk_count", 0),
                "document_count": stats.get("document_count", 0),
                "db_size": stats.get("db_size", "0 B"),
                "embedding_model": stats.get("embedding_model", "unknown")
            })
        
        return {
            "domains": domain_stats,
            "count": len(domains),
            "total_chunks": sum(s.get("chunk_count", 0) for s in domain_stats)
        }
        
    except Exception as e:
        api_logger.error(f"Error getting vectorstore domains: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/{domain}")
async def validate_domain_quality(domain: str) -> Dict[str, Any]:
    """Validate a domain's vectorstore quality.
    
    Checks:
    - Data integrity (chunks, documents, embeddings)
    - Search functionality
    - Sample query results
    """
    try:
        retriever = get_domain_retriever(domain)
        stats = retriever.get_collection_stats()
        
        validation_results = {
            "domain": domain,
            "timestamp": None,
            "checks": {},
            "overall_status": "unknown"
        }
        
        from datetime import datetime
        validation_results["timestamp"] = datetime.now().isoformat()
        
        # 1. Basic stats check
        chunk_count = stats.get("chunk_count", 0)
        doc_count = stats.get("document_count", 0)
        
        validation_results["checks"]["basic_stats"] = {
            "status": "pass" if chunk_count > 0 else "fail",
            "chunk_count": chunk_count,
            "document_count": doc_count,
            "db_size": stats.get("db_size", "0 B")
        }
        
        # 2. Data integrity check
        collection = retriever.vectorstore._collection
        sample = collection.get(limit=5)
        
        has_embeddings = sample and "embeddings" in sample and len(sample.get("embeddings", [])) > 0
        has_documents = sample and "documents" in sample and len(sample.get("documents", [])) > 0
        has_metadata = sample and "metadatas" in sample and len(sample.get("metadatas", [])) > 0
        
        embedding_dim = None
        if has_embeddings and sample["embeddings"]:
            embedding_dim = len(sample["embeddings"][0])
        
        validation_results["checks"]["data_integrity"] = {
            "status": "pass" if (has_embeddings and has_documents) else "fail",
            "has_embeddings": has_embeddings,
            "has_documents": has_documents,
            "has_metadata": has_metadata,
            "embedding_dimension": embedding_dim,
            "expected_dimension": 384
        }
        
        # 3. Search functionality test
        test_query = "health symptoms"
        try:
            results = retriever.similarity_search(test_query, k=3)
            search_works = len(results) > 0
            
            # Get scores for quality check
            results_with_scores = retriever.vectorstore.similarity_search_with_score(test_query, k=3)
            scores = [score for _, score in results_with_scores]
            avg_score = sum(scores) / len(scores) if scores else None
            
            validation_results["checks"]["search_functionality"] = {
                "status": "pass" if search_works else "fail",
                "test_query": test_query,
                "results_returned": len(results),
                "average_similarity_score": round(avg_score, 4) if avg_score else None,
                "note": "Lower scores = better similarity (cosine distance)"
            }
            
            # Sample results
            if results:
                validation_results["checks"]["sample_results"] = [
                    {
                        "preview": doc.page_content[:200].replace('\n', ' '),
                        "source": doc.metadata.get("file_name", "Unknown"),
                        "page": doc.metadata.get("page", "N/A"),
                        "score": round(score, 4)
                    }
                    for (doc, score) in results_with_scores[:3]
                ]
            
        except Exception as e:
            validation_results["checks"]["search_functionality"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Overall status
        all_checks_passed = all(
            check.get("status") == "pass" 
            for check in validation_results["checks"].values() 
            if isinstance(check, dict) and "status" in check
        )
        
        validation_results["overall_status"] = "pass" if all_checks_passed else "fail"
        
        return validation_results
        
    except Exception as e:
        api_logger.error(f"Error validating domain {domain}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

