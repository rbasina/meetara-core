"""
MCP (Multi-Component Planner) Router for Meetara Core.

This module orchestrates the flow between RAG retrieval and response generation
to provide contextually appropriate responses.

Note: Emotion detection has been decommissioned. This module now focuses on
domain-specific context retrieval only.
"""
from typing import Dict, List, Any, Optional
from langchain_core.documents import Document
from app.core.logger import agent_logger
from app.rag.domain_retrievers import get_domain_retriever


class MCPRouter:
    """Multi-Component Planner Router for orchestrating AI responses."""
    
    def __init__(self):
        agent_logger.info("MCPRouter initialized (knowledge-focused mode)")
    
    async def process_with_domain_context(
        self,
        query: str,
        domain: str
    ) -> Dict[str, Any]:
        """Process query with domain-specific context retrieval."""
        try:
            # Get domain-specific context
            domain_context = await self._get_domain_context(query, domain)
            
            # Combine contexts for response generation
            combined_context = {
                "query": query,
                "domain": domain,
                "domain_context": domain_context,
                "response_style": "informative"
            }
            
            agent_logger.info(f"Processed query for domain: {domain}")
            return combined_context
            
        except Exception as e:
            agent_logger.error(f"Error in MCP routing: {e}")
            return {
                "query": query,
                "domain": domain,
                "error": str(e),
                "response_style": "neutral"
            }
    
    async def _get_domain_context(self, query: str, domain: str) -> List[Document]:
        """Retrieve relevant context from the domain's vector store."""
        try:
            retriever = get_domain_retriever(domain)
            context_docs = retriever.similarity_search(query, k=3)
            
            agent_logger.info(f"Retrieved {len(context_docs)} context documents for domain: {domain}")
            return context_docs
            
        except Exception as e:
            agent_logger.warning(f"Failed to get domain context for {domain}: {e}")
            return []
    
    def get_routing_info(self) -> Dict[str, Any]:
        """Get information about the MCP routing capabilities."""
        return {
            "capabilities": [
                "domain_specific_retrieval",
                "contextual_response_adaptation"
            ],
            "response_styles": [
                "informative",
                "neutral"
            ]
        }


def create_mcp_router() -> MCPRouter:
    """Factory function to create an MCPRouter instance."""
    return MCPRouter()


# Global MCP router instance
_mcp_router: Optional[MCPRouter] = None


def get_mcp_router() -> MCPRouter:
    """Get the global MCPRouter instance."""
    global _mcp_router
    if _mcp_router is None:
        _mcp_router = create_mcp_router()
    return _mcp_router
