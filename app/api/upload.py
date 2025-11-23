"""
Upload API endpoints for Meetara Core.

This module provides endpoints for uploading documents to vector stores
and managing domain-specific knowledge bases.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from pathlib import Path
import time
import json
from app.core.logger import api_logger
from app.core.security import input_validator, privacy_enforcer, security_validator
from app.core.document_validator import document_validator
from app.rag.vector_loader import vector_loader, get_supported_file_types
from app.rag.domain_retrievers import (
    create_domain, 
    delete_domain, 
    list_available_domains,
    get_domain_retriever
)


router = APIRouter(prefix="/upload", tags=["upload"])


class UploadResponse(BaseModel):
    """Response model for upload endpoints."""
    success: bool
    message: str
    domain: Optional[str] = None
    documents_added: Optional[int] = None
    file_info: Optional[Dict[str, Any]] = None


class DomainInfo(BaseModel):
    """Model for domain information."""
    domain: str
    document_count: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


@router.post("/doc", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    domain: Optional[str] = Form(None, description="Target domain for the document (optional - will auto-detect if not provided)"),
    metadata: Optional[str] = Form(None, description="Additional metadata as JSON string")
):
    """Upload a document to vector store(s) with intelligent domain detection."""
    try:
        # Validate domain name if provided
        sanitized_domain = None
        if domain:
            sanitized_domain = security_validator.sanitize_domain_name(domain)
        
        # Validate file type
        if not security_validator.validate_file_extension(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(get_supported_file_types())}"
            )
        
        # Validate file size
        file_size = len(await file.read())
        await file.seek(0)  # Reset file pointer
        
        if not security_validator.validate_file_size(file_size):
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {security_validator.MAX_FILE_SIZE / (1024*1024)}MB."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Extract text content for validation
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(file_content)
            temp_file_path = Path(temp_file.name)
        
        # Extract text content for validation
        try:
            text_content = vector_loader.extract_text_from_file(temp_file_path)
        except Exception as e:
            api_logger.error(f"Error extracting text from file: {e}")
            text_content = ""
        
        # Validate document quality and relevance
        validation_domain = sanitized_domain if sanitized_domain else "general_health"
        
        validation_result = document_validator.validate_document(
            text_content, 
            file.filename, 
            validation_domain
        )
        
        api_logger.info(f"Document validation for {file.filename}: Quality={validation_result['quality_score']:.2f}, Relevance={validation_result['relevance_score']:.2f}, Valid={validation_result['is_valid']}")
        
        # Accept documents even if validation fails - truly intelligent system
        if not validation_result["is_valid"]:
            api_logger.warning(f"Document {file.filename} validation failed but accepting for processing: {validation_result['recommendations']}")
        
        # If document needs review, flag it but still process
        if validation_result["quality_score"] < 0.7 or validation_result["relevance_score"] < 0.6:
            api_logger.warning(f"Document {file.filename} accepted but needs review: {validation_result['recommendations']}")
        
        try:
            # Process and add to vector store(s)
            api_logger.info(f"Starting vector processing for {file.filename} to domain: {sanitized_domain}")
            
            # Run in thread pool to avoid blocking and add timeout protection
            import asyncio
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            
            # Use executor to run blocking PDF processing
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            
            try:
                # For very large PDFs (>100MB), process with longer timeout
                file_size_mb = file_size / (1024 * 1024)
                timeout_seconds = 600 if file_size_mb > 100 else 300  # 10 min for large, 5 min for others
                
                api_logger.info(f"Processing PDF ({file_size_mb:.1f} MB) with {timeout_seconds}s timeout")
                
                results = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        vector_loader.load_and_process_file,
                        temp_file_path,
                        sanitized_domain,
                        metadata
                    ),
                    timeout=timeout_seconds
                )
                api_logger.info(f"Vector processing results: {results}")
            except asyncio.TimeoutError:
                api_logger.error(f"PDF processing timed out after {timeout_seconds}s for {file.filename}")
                raise HTTPException(
                    status_code=504,
                    detail=f"PDF processing timed out. File is too large ({file_size_mb:.1f} MB). Try splitting the PDF or use batch upload."
                )
            except FutureTimeoutError:
                api_logger.error(f"PDF processing timed out for {file.filename}")
                raise HTTPException(
                    status_code=504,
                    detail=f"PDF processing timed out. File is too large ({file_size_mb:.1f} MB). Try splitting the PDF or use batch upload."
                )
            finally:
                executor.shutdown(wait=False)
            
            if results:
                # Count successful uploads
                successful_domains = [domain for domain, success in results.items() if success]
                total_domains = len(results)
                successful_count = len(successful_domains)
                
                if successful_count > 0:
                    # Get stats from primary domain (first successful domain)
                    primary_domain = successful_domains[0]
                    try:
                        retriever = get_domain_retriever(primary_domain)
                        stats = retriever.get_collection_stats()
                    except Exception as e:
                        api_logger.warning(f"Could not get stats for domain {primary_domain}: {e}")
                        stats = {}
                    
                    api_logger.info(f"Document uploaded successfully to {successful_count}/{total_domains} domains: {successful_domains}")
                    
                    return UploadResponse(
                        success=True,
                        message=f"Document uploaded successfully to {successful_count} domain(s): {', '.join(successful_domains)}",
                        domain=primary_domain,
                        documents_added=successful_count,
                        file_info={
                            "filename": file.filename,
                            "size": file_size,
                            "content_type": file.content_type,
                            "uploaded_domains": successful_domains,
                            "all_results": results,
                            "validation": validation_result,
                            "domain_stats": stats
                        }
                    )
                else:
                    # All domains failed
                    api_logger.error(f"Document upload failed for all domains: {results}")
                    return UploadResponse(
                        success=False,
                        message=f"Document upload failed for all domains: {results}",
                        domain=sanitized_domain,
                        file_info={
                            "filename": file.filename,
                            "validation": validation_result,
                            "vector_processing_results": results
                        }
                    )
            else:
                # No results returned
                api_logger.error("Vector processing returned no results")
                return UploadResponse(
                    success=False,
                    message="Vector processing failed - no results returned",
                    domain=sanitized_domain,
                    file_info={
                        "filename": file.filename,
                        "validation": validation_result
                    }
                )
                
        finally:
            # Clean up temporary file
            temp_file_path.unlink(missing_ok=True)
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload document. Please try again."
        )


@router.post("/domain", response_model=UploadResponse)
async def create_domain_endpoint(
    domain: str = Form(..., description="Domain name to create")
):
    """Create a new domain for document storage."""
    try:
        # Validate domain name
        sanitized_domain = security_validator.sanitize_domain_name(domain)
        
        # Create domain
        success = create_domain(sanitized_domain)
        
        if success:
            api_logger.info(f"Domain created successfully: {sanitized_domain}")
            
            return UploadResponse(
                success=True,
                message=f"Domain '{sanitized_domain}' created successfully",
                domain=sanitized_domain
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to create domain"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error creating domain: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create domain. Please try again."
        )


@router.delete("/domain/{domain}")
async def delete_domain_endpoint(domain: str):
    """Delete a domain and its vector store."""
    try:
        # Validate domain name
        sanitized_domain = security_validator.sanitize_domain_name(domain)
        
        # Delete domain
        success = delete_domain(sanitized_domain)
        
        if success:
            api_logger.info(f"Domain deleted successfully: {sanitized_domain}")
            
            return {
                "success": True,
                "message": f"Domain '{sanitized_domain}' deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Domain '{sanitized_domain}' not found or could not be deleted"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error deleting domain: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete domain. Please try again."
        )


@router.get("/domains", response_model=List[DomainInfo])
async def list_domains():
    """List all available domains with their statistics."""
    try:
        domains = list_available_domains()
        domain_info = []
        
        for domain in domains:
            try:
                retriever = get_domain_retriever(domain)
                stats = retriever.get_collection_stats()
                
                domain_info.append(DomainInfo(
                    domain=domain,
                    document_count=stats.get("count", 0),
                    embedding_model=stats.get("embedding_model", "unknown"),
                    chunk_size=stats.get("chunk_size", 1000),
                    chunk_overlap=stats.get("chunk_overlap", 200)
                ))
                
            except Exception as e:
                api_logger.warning(f"Error getting stats for domain {domain}: {e}")
                # Add domain with default stats
                domain_info.append(DomainInfo(
                    domain=domain,
                    document_count=0,
                    embedding_model="unknown",
                    chunk_size=1000,
                    chunk_overlap=200
                ))
        
        api_logger.info(f"Retrieved {len(domain_info)} domains")
        
        return domain_info
        
    except Exception as e:
        api_logger.error(f"Error listing domains: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve domain list"
        )


@router.get("/domain/{domain}/stats")
async def get_domain_stats(domain: str):
    """Get detailed statistics for a specific domain."""
    try:
        # Validate domain name
        sanitized_domain = security_validator.sanitize_domain_name(domain)
        
        # Get domain retriever
        retriever = get_domain_retriever(sanitized_domain)
        stats = retriever.get_collection_stats()
        
        return {
            "domain": sanitized_domain,
            "stats": stats,
            "status": "active" if stats.get("count", 0) > 0 else "empty"
        }
        
    except Exception as e:
        api_logger.error(f"Error getting domain stats for {domain}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve domain statistics"
        )


@router.get("/supported-file-types")
async def get_supported_file_types_endpoint():
    """Get list of supported file types for upload."""
    try:
        supported_types = get_supported_file_types()
        
        return {
            "supported_types": supported_types,
            "max_file_size_mb": security_validator.MAX_FILE_SIZE / (1024 * 1024),
            "allowed_extensions": list(security_validator.ALLOWED_EXTENSIONS)
        }
        
    except Exception as e:
        api_logger.error(f"Error getting supported file types: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve supported file types"
        )


class BatchUploadResponse(BaseModel):
    """Response model for batch upload endpoints."""
    success: bool
    message: str
    domain: str
    total_files: int
    uploaded_files: int
    failed_files: int
    errors: List[str] = []
    file_details: Optional[Dict[str, Any]] = None


@router.post("/batch", response_model=BatchUploadResponse)
async def batch_upload_documents(
    domain: str = Form(..., description="Target domain for the documents"),
    directory_path: str = Form(..., description="Directory path containing documents"),
    metadata: Optional[str] = Form(None, description="Additional metadata as JSON string")
):
    """Batch upload all supported documents from a directory to a specific domain."""
    try:
        # Validate domain name
        sanitized_domain = security_validator.sanitize_domain_name(domain)
        
        # Validate directory path
        directory = Path(directory_path)
        if not directory.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Directory not found: {directory_path}"
            )
        
        if not directory.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a directory: {directory_path}"
            )
        
        # Parse metadata
        batch_metadata = {}
        if metadata:
            try:
                import json
                batch_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON format in metadata"
                )
        
        # Find all supported files
        supported_extensions = get_supported_file_types()
        supported_files = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                supported_files.append(file_path)
        
        if not supported_files:
            raise HTTPException(
                status_code=400,
                detail=f"No supported files found in directory: {directory_path}"
            )
        
        # Process batch upload
        uploaded_count = 0
        failed_count = 0
        errors = []
        file_details = []
        
        for file_path in supported_files:
            try:
                # Validate file size
                file_size = file_path.stat().st_size
                if not security_validator.validate_file_size(file_size):
                    errors.append(f"File too large: {file_path.name}")
                    failed_count += 1
                    continue
                
                # Create file-specific metadata
                file_metadata = batch_metadata.copy()
                file_metadata.update({
                    "file_name": file_path.name,
                    "file_size": file_size,
                    "batch_upload": True,
                    "upload_date": time.strftime("%Y-%m-%d"),
                    "category": _extract_category_from_path(file_path),
                    "subdomain": _extract_subdomain_from_path(file_path)
                })
                
                # Process and add to vector store
                success = vector_loader.load_and_process_file(
                    file_path,
                    sanitized_domain,
                    metadata=json.dumps(file_metadata)
                )
                
                if success:
                    uploaded_count += 1
                    file_details.append({
                        "filename": file_path.name,
                        "size": file_size,
                        "status": "uploaded"
                    })
                else:
                    failed_count += 1
                    errors.append(f"Failed to process: {file_path.name}")
                    
            except Exception as e:
                failed_count += 1
                errors.append(f"Error processing {file_path.name}: {str(e)}")
        
        # Get domain stats after batch upload
        try:
            retriever = get_domain_retriever(sanitized_domain)
            stats = retriever.get_collection_stats()
        except Exception:
            stats = {"count": 0}
        
        api_logger.info(f"Batch upload completed for domain {sanitized_domain}: {uploaded_count} uploaded, {failed_count} failed")
        
        return BatchUploadResponse(
            success=uploaded_count > 0,
            message=f"Batch upload completed. {uploaded_count} files uploaded, {failed_count} failed",
            domain=sanitized_domain,
            total_files=len(supported_files),
            uploaded_files=uploaded_count,
            failed_files=failed_count,
            errors=errors,
            file_details={
                "uploaded": file_details,
                "domain_stats": stats
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error in batch upload: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process batch upload. Please try again."
        )


def _extract_category_from_path(file_path: Path) -> str:
    """Extract category from file path."""
    # Example: /medical/diabetes/guide.pdf → "diabetes"
    parts = file_path.parts
    if len(parts) >= 2:
        return parts[-2]  # Second to last part
    return "general"


def _extract_subdomain_from_path(file_path: Path) -> str:
    """Extract subdomain from file path."""
    # Example: /medical/diabetes/treatment.pdf → "treatment"
    return file_path.stem 