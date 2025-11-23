"""
Configuration-Driven Adapter Selector Tool for Meetara Core.

This tool selects the most appropriate domain adapter based on the query content
and available domains using configuration-driven approach with hybrid matching.
"""
from typing import Dict, Any, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from app.core.logger import agent_logger
from app.core.config_loader import config_loader
from app.rag.domain_retrievers import list_available_domains


class AdapterSelectorInput(BaseModel):
    """Input schema for adapter selector tool."""
    query: str = Field(description="The user query to analyze for domain selection")
    context: str = Field(default="", description="Additional context for domain selection")


class AdapterSelectorTool(BaseTool):
    """Tool for selecting the most appropriate domain adapter using configuration."""
    
    name: str = "adapter_selector"
    description: str = "Select the best domain adapter for a given query based on content analysis"
    args_schema: type = AdapterSelectorInput
    
    def _run(self, query: str, context: str = "") -> Dict[str, Any]:
        """Run the adapter selector tool."""
        try:
            # Get available domains
            available_domains = list_available_domains()
            
            if not available_domains:
                agent_logger.warning("No domains available for adapter selection")
                return {
                    "selected_domain": "general",
                    "confidence": 0.0,
                    "available_domains": [],
                    "reasoning": "No specific domains available, using general adapter"
                }
            
            # Detect urgency in query
            urgency_info = config_loader.detect_urgency(query)
            
            # Analyze query for domain relevance using hybrid matching
            domain_scores = self._analyze_query_domains(query, available_domains, urgency_info)
            
            # Select best domain
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            
            # Get domain configuration
            domain_config = config_loader.get_domain_config(best_domain[0])
            
            agent_logger.info(f"Selected domain '{best_domain[0]}' for query with urgency: {urgency_info['urgency_type']}")
            
            return {
                "selected_domain": best_domain[0],
                "confidence": best_domain[1],
                "available_domains": available_domains,
                "domain_scores": domain_scores,
                "urgency_info": urgency_info,
                "domain_config": domain_config,
                "reasoning": f"Query matches {best_domain[0]} domain with {best_domain[1]:.2f} confidence"
            }
            
        except Exception as e:
            agent_logger.error(f"Error in adapter selection: {e}")
            return {
                "selected_domain": "general",
                "confidence": 0.0,
                "error": str(e),
                "reasoning": "Error occurred during domain selection"
            }
    
    async def _arun(self, query: str, context: str = "") -> Dict[str, Any]:
        """Async run of the adapter selector tool."""
        return self._run(query, context)
    
    def _analyze_query_domains(self, query: str, available_domains: List[str], urgency_info: Dict[str, Any]) -> Dict[str, float]:
        """Analyze query using hybrid matching approach."""
        query_lower = query.lower()
        domain_scores = {}
        
        # Determine query type for weight selection
        query_type = self._determine_query_type(query, urgency_info)
        weights = config_loader.get_matching_weights(query_type)
        
        for domain in available_domains:
            # Get domain keywords from configuration
            domain_keywords = config_loader.get_domain_keywords(domain)
            
            # 1. Keyword matching (30-40% weight)
            keyword_score = self._calculate_keyword_score(query_lower, domain_keywords)
            
            # 2. Semantic similarity (40-60% weight) - placeholder for embedding-based matching
            semantic_score = self._calculate_semantic_score(query_lower, domain, domain_keywords)
            
            # 3. Context/emotion matching (15-20% weight)
            context_score = self._calculate_context_score(query_lower, domain, urgency_info)
            
            # Combine scores with weights
            final_score = (
                keyword_score * weights['keywords'] +
                semantic_score * weights['semantic'] +
                context_score * weights['context']
            )
            
            domain_scores[domain] = final_score
        
        # Ensure at least one domain has a score
        if not any(domain_scores.values()):
            domain_scores["general"] = 0.5
        
        return domain_scores
    
    def _determine_query_type(self, query: str, urgency_info: Dict[str, Any]) -> str:
        """Determine query type for weight selection."""
        query_lower = query.lower()
        
        if urgency_info['is_urgent']:
            return 'emergency'
        
        # Check for technical terms
        technical_terms = ['code', 'programming', 'algorithm', 'technical', 'engineering', 'data']
        if any(term in query_lower for term in technical_terms):
            return 'technical'
        
        # Check for creative terms
        creative_terms = ['creative', 'art', 'design', 'story', 'writing', 'music']
        if any(term in query_lower for term in creative_terms):
            return 'creative'
        
        return 'default'
    
    def _calculate_keyword_score(self, query_lower: str, domain_keywords: List[str]) -> float:
        """Calculate keyword matching score."""
        if not domain_keywords:
            return 0.0
        
        matches = 0
        for keyword in domain_keywords:
            if keyword.lower() in query_lower:
                matches += 1
        
        return min(matches / len(domain_keywords), 1.0)
    
    def _calculate_semantic_score(self, query_lower: str, domain: str, domain_keywords: List[str]) -> float:
        """Calculate semantic similarity score (placeholder for embedding-based matching)."""
        # This would be implemented with actual embedding similarity
        # For now, use enhanced keyword matching
        
        if not domain_keywords:
            return 0.0
        
        # Enhanced keyword matching with partial matches
        semantic_matches = 0
        query_words = set(query_lower.split())
        
        for keyword in domain_keywords:
            keyword_words = set(keyword.lower().split())
            # Check for word overlap
            overlap = query_words.intersection(keyword_words)
            if overlap:
                semantic_matches += len(overlap) / len(keyword_words)
        
        return min(semantic_matches / len(domain_keywords), 1.0)
    
    def _calculate_context_score(self, query_lower: str, domain: str, urgency_info: Dict[str, Any]) -> float:
        """Calculate context/emotion matching score."""
        context_score = 0.0
        
        # Urgency context
        if urgency_info['is_urgent']:
            # Prioritize emergency domains for urgent queries
            emergency_domains = ['emergency_care', 'crisis_management', 'safety_security']
            if domain in emergency_domains:
                context_score += 0.5
        
        # Domain-specific context cues
        domain_context_cues = {
            'healthcare': ['health', 'medical', 'doctor', 'symptoms'],
            'legal': ['legal', 'law', 'rights', 'court'],
            'financial': ['money', 'finance', 'investment', 'budget'],
            'education': ['study', 'learn', 'education', 'academic']
        }
        
        if domain in domain_context_cues:
            cues = domain_context_cues[domain]
            cue_matches = sum(1 for cue in cues if cue in query_lower)
            context_score += min(cue_matches / len(cues), 0.3)
        
        return min(context_score, 1.0) 