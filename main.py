"""
Meetara Core - Main FastAPI Application

A production-ready, modular LangChain-based RAG assistant backend
with knowledge-focused responses and tool-based agent orchestration.
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.core.logger import setup_logging, api_logger
from app.api import chat_router, upload_router, image_generation_router, vectorstore_router
from app.api.models import router as models_router


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    components: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    api_logger.info("Starting Meetara Core backend...")
    
    # Initialize logging
    setup_logging(
        log_level=settings.log_level,
        log_file=settings.log_file
    )
    
    # Create necessary directories
    settings.vectorstore_path.mkdir(parents=True, exist_ok=True)
    
    # Pre-warm domain retrievers for faster first queries (runs in background)
    try:
        from app.rag.domain_retrievers import prewarm_retrievers
        api_logger.info("🔥 Pre-warming domain retrievers...")
        prewarm_retrievers(max_domains=3)  # Pre-warm top 3 domains
    except Exception as e:
        api_logger.warning(f"⚠️ Pre-warming failed (non-critical): {e}")
    
    api_logger.info("Meetara Core backend started successfully")
    
    yield
    
    # Shutdown
    api_logger.info("Shutting down Meetara Core backend...")


# Create FastAPI app
app = FastAPI(
    title="Meetara Core",
    description="Modular LangChain-based RAG Assistant Backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(image_generation_router, prefix="/api")
app.include_router(vectorstore_router, prefix="/api")
app.include_router(models_router)  # Model management API

# Mount images directory for serving images
images_dir = Path("images")
if images_dir.exists():
    app.mount("/api/images", StaticFiles(directory=str(images_dir.resolve())), name="images")
    api_logger.info(f"✅ Images directory mounted at /api/images from {images_dir.resolve()}")
else:
    api_logger.warning(f"⚠️ Images directory not found at {images_dir.absolute()} - image serving disabled")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Meetara Core",
        "version": "1.0.0",
        "description": "Modular LangChain-based RAG Assistant Backend",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check core components
        components = {
            "api": "healthy",
            "logging": "healthy",
            "config": "healthy"
        }
        
        # Check vector store
        try:
            from app.rag.domain_retrievers import list_available_domains
            domains = list_available_domains()
            components["vectorstore"] = "healthy"
            components["domains_count"] = len(domains)
        except Exception as e:
            components["vectorstore"] = f"error: {str(e)}"
        
        # Check agent system
        try:
            from app.agent.planner import get_meetara_agent
            agent = get_meetara_agent()
            components["agent"] = "healthy" if agent else "not_initialized"
        except Exception as e:
            components["agent"] = f"error: {str(e)}"
        
        # Determine overall status
        overall_status = "healthy"
        for component, status in components.items():
            if isinstance(status, str) and status.startswith("error"):
                overall_status = "degraded"
                break
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            components=components
        )
        
    except Exception as e:
        api_logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            components={"error": str(e)}
        )


@app.get("/api/vectorstore/all")
async def get_all_vectorstore_info():
    """Get information about ALL domain vector stores in one request (optimized with parallel loading)."""
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        from app.rag.domain_retrievers import list_available_domains, get_domain_retriever
        
        domains = list_available_domains()
        domain_stats = {}
        
        def get_domain_stats(domain: str):
            """Get stats for a single domain (runs in thread pool)."""
            try:
                retriever = get_domain_retriever(domain)
                stats = retriever.get_collection_stats()
                return {
                    "domain": domain,
                    "stats": stats,
                    "status": "active" if stats.get("count", 0) > 0 else "empty"
                }
            except Exception as e:
                api_logger.warning(f"Failed to get stats for domain {domain}: {e}")
                return {
                    "domain": domain,
                    "stats": {"count": 0},
                    "status": "error"
                }
        
        # Load all domains in parallel, but limit workers to avoid model loading conflicts
        # PyTorch models don't like concurrent initialization, so use fewer workers
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(get_domain_stats, domains))
        
        # Convert results to dictionary
        for result in results:
            domain_stats[result["domain"]] = result
        
        return {
            "domains": domain_stats,
            "total_domains": len(domains),
            "total_documents": sum(s.get("stats", {}).get("count", 0) for s in domain_stats.values())
        }
        
    except Exception as e:
        api_logger.error(f"Error getting all vectorstore info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve vector store information"
        )

@app.get("/api/vectorstore/{domain}")
async def get_vectorstore_info(domain: str):
    """Get information about a specific domain's vector store."""
    try:
        from app.rag.domain_retrievers import get_domain_retriever
        
        retriever = get_domain_retriever(domain)
        stats = retriever.get_collection_stats()
        
        return {
            "domain": domain,
            "stats": stats,
            "status": "active" if stats.get("count", 0) > 0 else "empty"
        }
        
    except Exception as e:
        api_logger.error(f"Error getting vectorstore info for {domain}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve vector store information"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    api_logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later."
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 