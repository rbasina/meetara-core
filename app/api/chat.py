"""
Chat API endpoints for Meetara Core.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from app.core.logger import api_logger, agent_logger
from app.core.security import input_validator, privacy_enforcer
from app.agent.planner import get_meetara_agent
from app.agent.mcp_router import get_mcp_router
# ✅ Image generation now handled in planner.py with RAG context


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., description="User query", min_length=1, max_length=10000)
    topic: Optional[str] = Field(None, description="Topic/domain for the query")
    lang: Optional[str] = Field("en", description="Language code")
    emotion: Optional[str] = Field(None, description="Detected emotion")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    model: Optional[str] = Field(None, description="Model ID to use (e.g., 'meetara-1.7b', 'meetara-4b-thinking')")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    domain: Optional[str] = None
    emotion_context: Optional[Dict[str, Any]] = None
    tool_calls: Optional[list] = None
    confidence: Optional[float] = None
    images: Optional[List[Dict[str, Any]]] = None  # ✅ Images from documents
    request_timestamp: Optional[str] = None
    response_timestamp: Optional[str] = None
    rag_status: Optional[str] = "llm"  # ✅ 'rag', 'llm', or 'mixed' - for UI display
    documents_used: Optional[int] = 0  # ✅ Number of RAG documents used


@router.get("/test-image")
async def test_image_endpoint():
    """Test endpoint to verify image serving."""
    from pathlib import Path
    images_dir = Path("images")
    if not images_dir.exists():
        return {"error": "Images directory not found", "path": str(images_dir.absolute())}
    
    # Find a sample image
    academic_dir = images_dir / "academic_tutoring"
    if academic_dir.exists():
        sample_images = list(academic_dir.rglob("*.png"))[:3]
        return {
            "status": "ok",
            "images_dir": str(images_dir.absolute()),
            "sample_images": [str(img.relative_to(images_dir)) for img in sample_images],
            "sample_urls": [f"/api/images/{img.relative_to(images_dir).as_posix()}" for img in sample_images]
        }
    return {"error": "No academic_tutoring images found"}


@router.post("/")
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """Chat endpoint for intelligent responses."""
    try:
        # 🔒 SECURITY: Sanitize user query before processing
        from app.core.security import security_validator
        sanitized_query = security_validator.sanitize_text(request.query)
        
        # Sanitize topic if provided
        sanitized_topic = None
        if request.topic:
            sanitized_topic = security_validator.sanitize_text(request.topic)
        
        # Log the request (with PII redaction for privacy)
        privacy_enforcer.safe_log("Chat request received", {
            "query_length": len(sanitized_query),
            "topic": sanitized_topic,
            "lang": request.lang,
            "emotion": request.emotion
        })
        
        # ✅ Special handling: Detect queries about me²TARA itself
        query_lower = sanitized_query.lower().strip()
        query_words = query_lower.split()
        
        # Check if query contains "meetara", "me²tara", or "me2tara" (case-insensitive)
        meetara_variants = ["meetara", "me²tara", "me2tara"]
        has_meetara_name = any(variant in query_lower for variant in meetara_variants)
        
        if has_meetara_name:
            # Check if query is asking ABOUT meetara (not asking meetara TO do something)
            question_indicators = [
                "what", "who", "tell", "explain", "describe", 
                "about", "does", "can you", "how", "what's", "who's"
            ]
            is_question_about = any(indicator in query_lower for indicator in question_indicators)
            
            # Very short queries with just "meetara" are likely asking about it
            is_short_query = len(query_words) <= 5
            
            # Exclude clear action requests (e.g., "meetara help me", "meetara find")
            action_requests = [
                "meetara help", "meetara find", "meetara search", "meetara get",
                "meetara show", "meetara give", "meetara create", "meetara make",
                "meetara do this", "meetara do that", "meetara calculate"
            ]
            is_action_request = any(req in query_lower for req in action_requests)
            
            # If it's a question about meetara OR a short query, and not an action request
            is_about_meetara = (is_question_about or is_short_query) and not is_action_request
            
            # Log for debugging
            if is_about_meetara:
                api_logger.info(f"🔍 Detected query about me²TARA: '{sanitized_query[:100]}'")
        else:
            is_about_meetara = False
        
        if is_about_meetara:
            # Return accurate information about me²TARA based on MEETARA_CORE.md
            correct_response = """**Quick Answer:** me²TARA is an offline-first, privacy-focused, and empathetic AI assistant — built from the ground up to run entirely on your own hardware, understand your emotional context, and provide expert-level assistance across over 100 knowledge domains.

## What Does "me²TARA" Mean?

The name **me²TARA** carries deep significance:

- **me²** (me-squared) — This AI is an extension of *you*. It amplifies your capabilities, learns your context, and serves your needs — not a corporation's interests. The "squared" represents exponential empowerment.

- **TARA** — In Sanskrit, "Tara" (तारा) means "star" or "one who guides across." Like a guiding star, me²TARA illuminates your path through complex information, helping you navigate healthcare decisions, legal questions, educational challenges, and life's countless domains of knowledge.

Together, **me²TARA** represents *your personal guiding star* — an AI that belongs to you, works for you, and stays with you.

## Core Philosophy

### 🔒 Privacy by Design
Every computation happens on your machine. Your documents stay in your `vectorstore/` folder. Your conversations remain in your local session. There is no cloud. There is no "trust us" — there is only verifiable, auditable, local processing.

### 🌐 Offline-First
me²TARA is designed to function completely offline with local language models, local vector database, and local embeddings. Your AI assistant doesn't abandon you when WiFi does.

### 💚 Empathetic Intelligence
me²TARA integrates emotion detection to adapt its communication style based on your emotional state, making responses more helpful and contextually appropriate.

## Domain Expertise

me²TARA is a **domain-aware knowledge system** with specialized understanding across critical areas:
- **Safety-Critical Domains**: Healthcare, Legal & Financial, Emergency
- **Expert Domains**: Business, Education, Technology
- **Quality Domains**: Personal Life, Creative, Wellness

## The RAG Advantage

me²TARA uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from YOUR uploaded documents, ensuring responses are grounded in your actual knowledge base rather than hallucinated from training data.

**Welcome to me²TARA — your guiding star.**"""
            
            api_logger.info("Detected query about me²TARA - returning accurate system information")
            return ChatResponse(
                response=correct_response,
                domain="general",
                confidence=1.0,
                request_timestamp=datetime.now().isoformat(),
                response_timestamp=datetime.now().isoformat()
            )
        
        # Get the agent
        agent = get_meetara_agent()
        if not agent:
            raise HTTPException(status_code=500, detail="Agent not available")
        
        # ✅ Image generation now happens in planner.py AFTER RAG retrieval
        # Uses RAG context + LLM analysis (like text generation), NOT keyword matching
        # This ensures context-aware, intelligent image generation
        
        # Process the sanitized query - let the system automatically detect domain
        result = await agent.process_query(
            query=sanitized_query,  # ✅ Using sanitized query
            session_id=request.session_id,  # ✅ Pass session ID for conversation memory
            context={
                "lang": request.lang,
                "emotion": request.emotion,
                "topic": sanitized_topic,  # ✅ Using sanitized topic
                "model": request.model,  # ✅ Model selection
                # Don't force topic - let system auto-detect
                "auto_detect_domain": True,
                **request.context  # ✅ Merge any additional context
            }
        )
        
        # ✅ Generated images are now included in result from planner.py
        # No need to append here - planner handles RAG-based image generation
        
        # Log the response
        api_logger.info("Chat response generated", {
            "query": request.query[:50] + "..." if len(request.query) > 50 else request.query,
            "response_length": len(result.get("response", "")),
            "domain": result.get("domain", ""),
            "confidence": result.get("confidence", 0.0),
            "images_count": len(result.get("images", []))  # ✅ Log image count
        })
        
        # ✅ Debug: Log image URLs if present
        if result.get("images"):
            api_logger.info(f"📷 Returning {len(result['images'])} images in API response")
            for i, img in enumerate(result["images"]):
                api_logger.info(f"   Image {i+1}: url={img.get('image_url', 'NO URL')}, path={img.get('image_path', 'NO PATH')}")
        else:
            api_logger.debug(f"   No images in result (result keys: {list(result.keys())})")
        
        # ✅ CRITICAL: Verify images are in result before creating response
        images_in_result = result.get("images")
        if images_in_result and len(images_in_result) > 0:
            api_logger.info(f"🔍 DEBUG: result dict contains 'images' key with {len(images_in_result)} items")
            api_logger.info(f"🔍 DEBUG: First image in result: {images_in_result[0] if images_in_result else 'N/A'}")
        else:
            images_status = "missing" if "images" not in result else ("empty list" if images_in_result == [] else f"None/null")
            api_logger.warning(f"🔍 DEBUG: result dict 'images' is {images_status}. Available keys: {list(result.keys())}")
        
        # Create response
        chat_response = ChatResponse(**result)
        
        # ✅ Final verification: Log what's being returned
        if chat_response.images:
            api_logger.info(f"✅ ChatResponse created with {len(chat_response.images)} images")
            api_logger.info(f"🔍 DEBUG: First image URL in ChatResponse: {chat_response.images[0].get('image_url', 'NO URL') if chat_response.images else 'N/A'}")
        else:
            api_logger.warning(f"⚠️ ChatResponse created with 0 images (even though result may have had images)")
        
        # ✅ Serialize to dict to verify what's actually being sent
        response_dict = chat_response.model_dump()
        if response_dict.get("images"):
            api_logger.info(f"✅ Response dict has {len(response_dict['images'])} images")
        else:
            api_logger.warning(f"⚠️ Response dict has NO images field!")
        
        return chat_response
        
    except Exception as e:
        agent_logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains")
async def get_available_domains():
    """Get list of available domains for chat."""
    try:
        from app.rag.domain_retrievers import list_available_domains
        
        domains = list_available_domains()
        
        api_logger.info(f"Retrieved {len(domains)} available domains")
        
        return {
            "domains": domains,
            "count": len(domains)
        }
        
    except Exception as e:
        api_logger.error(f"Error retrieving domains: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve available domains"
        )


@router.get("/domains/categorized")
async def get_categorized_domains():
    """Get domains organized by category (for Perplexity-style sidebar)."""
    try:
        from app.rag.domain_retrievers import list_available_domains, get_domain_retriever
        from app.core.config_loader import config_loader
        
        # Get all available domains with their stats
        available_domains = set(list_available_domains())
        
        # Get category structure from config
        categories_data = {}
        domain_config = config_loader.domain_config
        
        if not domain_config:
            raise HTTPException(status_code=500, detail="Domain configuration not loaded")
        
        # Iterate through categories in config
        for category_name, category_config in domain_config.get('categories', {}).items():
            category_domains = category_config.get('domains', {})
            
            # Build domain list with stats for this category
            domains_in_category = []
            total_chunks = 0
            total_size_mb = 0.0
            
            for domain_name in category_domains.keys():
                if domain_name in available_domains:
                    try:
                        retriever = get_domain_retriever(domain_name)
                        stats = retriever.get_collection_stats()
                        chunk_count = stats.get('chunk_count', 0)
                        document_count = stats.get('document_count', 0)  # Get document count
                        db_size = stats.get('db_size', '0 B')
                        
                        # Parse size for total calculation
                        size_str = str(db_size).split('(')[0].strip()
                        size_mb = 0.0
                        try:
                            if 'MB' in size_str:
                                size_mb = float(size_str.replace('MB', '').strip())
                            elif 'GB' in size_str:
                                size_mb = float(size_str.replace('GB', '').strip()) * 1024
                            elif 'KB' in size_str:
                                size_mb = float(size_str.replace('KB', '').strip()) / 1024
                        except:
                            pass
                        
                        total_chunks += chunk_count
                        total_size_mb += size_mb
                        
                        domains_in_category.append({
                            "name": domain_name,
                            "display_name": domain_name.replace('_', ' ').title(),
                            "chunks": chunk_count,
                            "documents": document_count,  # Add document count
                            "database_size": db_size,
                            "has_content": chunk_count > 0
                        })
                    except Exception as e:
                        # Domain might not be initialized yet
                        domains_in_category.append({
                            "name": domain_name,
                            "display_name": domain_name.replace('_', ' ').title(),
                            "chunks": 0,
                            "database_size": "0 B",
                            "has_content": False
                        })
            
            # Only include categories that have at least one domain
            if domains_in_category:
                # Format category display name
                category_display = category_name.replace('_', ' ').title()
                
                categories_data[category_name] = {
                    "name": category_name,
                    "display_name": category_display,
                    "tier": category_config.get('tier', 'quality'),
                    "domains": sorted(domains_in_category, key=lambda x: x['chunks'], reverse=True),
                    "total_domains": len(domains_in_category),
                    "domains_with_content": len([d for d in domains_in_category if d['has_content']]),
                    "total_chunks": total_chunks,
                    "total_size_mb": round(total_size_mb, 2)
                }
        
        # Sort categories by priority (safety_critical first, then expert, then quality)
        tier_order = {'safety_critical': 1, 'expert': 2, 'quality': 3}
        sorted_categories = sorted(
            categories_data.items(),
            key=lambda x: (tier_order.get(x[1]['tier'], 4), -x[1]['total_chunks'])
        )
        
        result = {
            "categories": {name: data for name, data in sorted_categories},
            "total_categories": len(categories_data),
            "total_domains": sum(len(cat['domains']) for cat in categories_data.values())
        }
        
        api_logger.info(f"Retrieved {len(categories_data)} categories with domains")
        return result
        
    except Exception as e:
        api_logger.error(f"Error getting categorized domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains/keywords")
async def get_domain_keywords():
    """Get domain keywords for client-side domain detection.
    
    Returns keywords from domain_keywords.yaml config file.
    This allows the frontend to detect domains without hardcoding.
    """
    try:
        from app.core.config_loader import config_loader
        
        # Get domain keywords from config
        domain_keywords = config_loader.domain_keywords
        
        if not domain_keywords:
            return {
                "keywords": {},
                "stop_words": [],
                "generic_terms": [],
                "message": "Domain keywords not loaded"
            }
        
        # Extract keywords for each domain
        keywords_by_domain = {}
        domains_config = domain_keywords.get('domains', {})
        
        for domain_name, domain_data in domains_config.items():
            if isinstance(domain_data, dict) and 'keywords' in domain_data:
                # Get keywords and limit to most important ones for frontend
                all_keywords = domain_data['keywords']
                # Take first 30 keywords (most important ones)
                keywords_by_domain[domain_name] = all_keywords[:30] if len(all_keywords) > 30 else all_keywords
        
        return {
            "keywords": keywords_by_domain,
            "stop_words": domain_keywords.get('stop_words', []),
            "generic_terms": domain_keywords.get('generic_terms', []),
            "total_domains": len(keywords_by_domain)
        }
        
    except Exception as e:
        api_logger.error(f"Error getting domain keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/info")
async def get_agent_info():
    """Get information about the agent system."""
    try:
        agent = get_meetara_agent()
        if not agent:
            raise HTTPException(
                status_code=503,
                detail="Agent system not initialized"
            )
        
        tool_info = agent.get_tool_info()
        memory_info = {
            "message_count": len(agent.get_memory()),
            "has_memory": len(agent.get_memory()) > 0
        }
        
        return {
            "agent_status": "active",
            "tools": tool_info,
            "memory": memory_info,
            "capabilities": [
                "multi_domain_assistance",
                "emotion_aware_responses",
                "speech_processing",
                "translation",
                "document_retrieval"
            ]
        }
        
    except Exception as e:
        api_logger.error(f"Error getting agent info: {e}")
        return {
            "agent_status": "error",
            "tools": {"total_tools": 0, "tool_names": [], "tool_descriptions": []},
            "memory": {"message_count": 0, "has_memory": False},
            "capabilities": [],
            "error": str(e)
        }


@router.post("/agent/memory/clear")
async def clear_agent_memory():
    """Clear the agent's conversation memory."""
    try:
        agent = get_meetara_agent()
        if not agent:
            raise HTTPException(
                status_code=503,
                detail="Agent system not initialized"
            )
        
        agent.clear_memory()
        
        api_logger.info("Agent memory cleared")
        
        return {
            "message": "Agent memory cleared successfully",
            "status": "success"
        }
        
    except Exception as e:
        api_logger.error(f"Error clearing agent memory: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear agent memory"
        ) 