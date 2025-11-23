"""
Semantic-First Domain Detector for Meetara Core

This module implements a scalable semantic-first approach to domain detection:
1. Query → Embed query
2. Search across ALL domains (or candidate domains) using semantic similarity
3. Get top results with similarity scores from each domain
4. Determine domain from which domain has the BEST semantic matches
5. Return domain + best documents

This is more scalable than keyword matching and handles new concepts automatically.
"""

from typing import Dict, List, Tuple, Optional, Any
from langchain_core.documents import Document
from app.core.logger import agent_logger
from app.rag.domain_retrievers import list_available_domains, get_domain_retriever
from app.core.config import Settings


settings = Settings()


class SemanticDomainDetector:
    """
    Semantic-first domain detector that uses vector similarity search
    instead of keyword matching for better scalability.
    """
    
    def __init__(self):
        """Initialize the semantic domain detector."""
        self.settings = settings
        agent_logger.info("SemanticDomainDetector initialized with semantic-first approach")
    
    def detect_domain_and_retrieve(
        self,
        query: str,
        top_k_per_domain: int = 3,
        max_domains_to_check: Optional[int] = None,
        min_similarity_score: float = 0.2,
        domains_to_check: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect domain using semantic search across all domains and retrieve relevant documents.
        
        Process:
        1. Get all available domains
        2. Search each domain with the query (semantic similarity)
        3. Collect results with scores from each domain
        4. Determine best domain based on average similarity scores
        5. Return domain + best documents
        
        Args:
            query: User query to process
            top_k_per_domain: Number of top results to retrieve per domain (default: 3)
            max_domains_to_check: Limit number of domains to check (None = check all)
            min_similarity_score: Minimum similarity score threshold (0.0-1.0, higher = stricter)
            
        Returns:
            Dictionary with:
            - detected_domain: Best matching domain
            - confidence: Average similarity score of top results
            - documents: List of relevant documents from best domain
            - domain_scores: Dict of domain -> avg_score for all checked domains
            - all_results: Dict of domain -> (documents, scores) for all domains
        """
        try:
            # Get available domains
            available_domains = list_available_domains()
            if domains_to_check:
                normalized = {domain for domain in domains_to_check if domain in available_domains}
                domains = [domain for domain in domains_to_check if domain in normalized]
                if not domains:
                    agent_logger.warning("Keyword hinted domains missing from available list; falling back to all domains")
                    domains = available_domains
            else:
                domains = available_domains

            if not domains:
                agent_logger.warning("No domains available for semantic domain detection")
                return {
                    "detected_domain": "general_health",
                    "confidence": 0.0,
                    "documents": [],
                    "domain_scores": {},
                    "all_results": {},
                    "error": "No domains available"
                }
            
            # Limit domains to check if specified
            if max_domains_to_check and len(domains) > max_domains_to_check:
                domains = domains[:max_domains_to_check]
                agent_logger.info(f"Checking top {len(domains)} domains for semantic search")
            
            # Search across all domains
            domain_results: Dict[str, List[Tuple[Document, float]]] = {}
            domain_scores: Dict[str, float] = {}
            
            agent_logger.info(f"🔍 Semantic search across {len(domains)} domains for query: {query[:100]}...")
            
            for domain in domains:
                try:
                    retriever = get_domain_retriever(domain)
                    
                    # Check if vectorstore exists
                    if not retriever.vectorstore:
                        continue
                    
                    # Use similarity_search_with_score to get scores
                    # Note: ChromaDB returns distance (lower = more similar)
                    # ChromaDB uses L2 distance typically, not cosine
                    results_with_scores = retriever.vectorstore.similarity_search_with_score(
                        query=query,
                        k=top_k_per_domain
                    )
                    
                    if results_with_scores:
                        agent_logger.info(f"   {domain}: Got {len(results_with_scores)} results from vectorstore")
                        
                        # Convert distance to similarity score
                        # ChromaDB can use different distance metrics:
                        # - Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
                        # - L2/Euclidean: 0 = identical, larger = less similar
                        # We'll detect based on typical ranges and convert appropriately
                        similarity_results = []
                        first_distance = results_with_scores[0][1] if results_with_scores else None
                        
                        # Determine distance metric based on first result
                        if first_distance is not None:
                            # If distance is typically in [0, 2] range, it's likely cosine
                            # If distance is typically larger (>2), it's likely L2
                            is_cosine_like = first_distance <= 2.0
                            
                            for i, (doc, distance) in enumerate(results_with_scores):
                                if is_cosine_like:
                                    # Cosine distance: convert to similarity
                                    # similarity = 1 - (distance / 2) for [0, 2] range
                                    similarity = max(0.0, 1.0 - (distance / 2.0))
                                else:
                                    # L2/Euclidean distance: convert using inverse
                                    similarity = 1.0 / (1.0 + distance) if distance >= 0 else 0.0
                                
                                # Log first result for debugging
                                if i == 0:
                                    agent_logger.info(
                                        f"      {domain}: distance={distance:.4f} "
                                        f"({'cosine-like' if is_cosine_like else 'L2-like'}), "
                                        f"similarity={similarity:.4f}"
                                    )
                                
                                # Accept results even with moderate similarity
                                if similarity >= min_similarity_score:
                                    similarity_results.append((doc, similarity))
                        else:
                            # No results to process
                            similarity_results = []
                        
                        if similarity_results:
                            domain_results[domain] = similarity_results
                            
                            # Calculate average similarity score for this domain
                            avg_score = sum(score for _, score in similarity_results) / len(similarity_results)
                            domain_scores[domain] = avg_score
                            
                            agent_logger.info(
                                f"   {domain}: {len(similarity_results)}/{len(results_with_scores)} results passed threshold, "
                                f"avg similarity: {avg_score:.3f}"
                            )
                        else:
                            agent_logger.warning(
                                f"   {domain}: All {len(results_with_scores)} results filtered by threshold "
                                f"(min={min_similarity_score:.3f}). Max similarity: {max(1.0/(1.0+d) for _, d in results_with_scores):.3f}"
                            )
                    else:
                        agent_logger.warning(f"   {domain}: No results from vectorstore (might be empty)")
                    
                except Exception as e:
                    agent_logger.warning(f"Failed to search domain {domain}: {e}")
                    continue
            
            if not domain_results:
                agent_logger.warning("No semantic matches found in any domain")
                return {
                    "detected_domain": "general_health",
                    "confidence": 0.0,
                    "documents": [],
                    "domain_scores": domain_scores,
                    "all_results": {},
                    "warning": "No semantic matches found"
                }
            
            # Find domain with highest average similarity
            best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            best_confidence = domain_scores[best_domain]
            best_documents = [doc for doc, _ in domain_results[best_domain]]
            
            # Sort all domain scores for reporting
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            top_3_domains = sorted_domains[:3]
            
            agent_logger.info(
                f"✅ Semantic detection: {best_domain} (confidence: {best_confidence:.3f})"
            )
            agent_logger.info(
                f"   Top 3 domains: {[(d, f'{s:.3f}') for d, s in top_3_domains]}"
            )
            
            return {
                "detected_domain": best_domain,
                "confidence": best_confidence,
                "documents": best_documents,
                "domain_scores": domain_scores,
                "all_results": {
                    domain: {
                        "documents": [doc for doc, _ in results],
                        "scores": [score for _, score in results]
                    }
                    for domain, results in domain_results.items()
                },
                "top_k_per_domain": top_k_per_domain
            }
            
        except Exception as e:
            agent_logger.error(f"Error in semantic domain detection: {e}")
            return {
                "detected_domain": "general_health",
                "confidence": 0.0,
                "documents": [],
                "domain_scores": {},
                "all_results": {},
                "error": str(e)
            }
    
    def hybrid_detect(
        self,
        query: str,
        keyword_hint_domains: Optional[List[str]] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> Dict[str, Any]:
        """
        Hybrid approach: Use keyword hints to narrow domains, then semantic search.
        
        This combines:
        - Fast keyword pre-filtering (to reduce search space)
        - Accurate semantic search (to determine best domain)
        
        Args:
            query: User query
            keyword_hint_domains: Optional list of domains to check first (from keyword matching)
            semantic_weight: Weight for semantic scores (default: 0.7)
            keyword_weight: Weight for keyword scores (default: 0.3)
            
        Returns:
            Same structure as detect_domain_and_retrieve
        """
        # If keyword hints available, use them to narrow search (FAST)
        if keyword_hint_domains:
            agent_logger.info(
                f"🔍 Hybrid detection: Semantic search on {len(keyword_hint_domains)} keyword-hinted domains: {keyword_hint_domains}"
            )
            # Search ONLY the hinted domains (much faster than all domains)
            result = self.detect_domain_and_retrieve(
                query=query,
                top_k_per_domain=3,
                max_domains_to_check=len(keyword_hint_domains),
                domains_to_check=keyword_hint_domains,
            )
            
            # Use semantic result if it has documents (even if confidence is low)
            # Semantic similarity is more accurate than keyword matching
            if result.get("documents") and len(result["documents"]) > 0:
                agent_logger.info(f"✅ Semantic search found {len(result['documents'])} docs in domain: {result['detected_domain']}")
                return result
            elif result.get("confidence", 0) >= 0.3:  # Lower threshold - semantic is more accurate
                agent_logger.info(f"✅ Semantic search confidence {result['confidence']:.3f} for domain: {result['detected_domain']}")
                return result
        
        # If no keyword hints OR semantic search found nothing, do broader search
        # But limit to top domains with data (safety)
        agent_logger.info("🔍 Semantic search: No keyword hints or low confidence, trying broader search")
        return self.detect_domain_and_retrieve(
            query=query,
            top_k_per_domain=3,
            max_domains_to_check=10  # Limit to top 10 domains to avoid performance issues
        )


# Singleton instance
_semantic_detector: Optional[SemanticDomainDetector] = None


def get_semantic_domain_detector() -> SemanticDomainDetector:
    """Get or create the semantic domain detector singleton."""
    global _semantic_detector
    if _semantic_detector is None:
        _semantic_detector = SemanticDomainDetector()
    return _semantic_detector

